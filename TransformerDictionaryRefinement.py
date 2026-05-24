import copy
import difflib
import gc
import json
import os
import random
import re
import time
from pathlib import Path
import Levenshtein
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
from App.CRNNCNTRHCR import MATRIX_DIR
from captum.attr import IntegratedGradients
from datasets import load_dataset
from evaluate import load
from spellchecker import SpellChecker
from torch.nn.utils import prune
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback, T5ForConditionalGeneration, ByT5Tokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from celery import Celery
from App.CRNNCNTROCR import (
    CRNNInferencePipeline,
    DubiousRegionSelector,
    CascadeRefinementNetwork, PersistentAuthorMemory
)
from Models.DeepCapsNetCharRecognition import CapsNet
from Preprocessing.OpticalLayoutRecognition import PageToLineSegmentor, LineToWordSegmentor
from Preprocessing.Preprocessing import Preprocessing
BASE_DIR = "/app/data"
VISUAL_DEBUG_DIR = MATRIX_DIR / "visual_debug_output"
VISUAL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
IAM_ROOT = os.path.join(BASE_DIR, "iam_words")
XML_DIR = os.path.join(IAM_ROOT, "xml")
FORMS_DIR = os.path.join(IAM_ROOT, "forms")
SENTENCES_TXT = os.path.join(IAM_ROOT, "sentences.txt")
CHECKPOINT_BASE = os.path.join(BASE_DIR, "HandwrittenTextRecognition", "output_data", "checkpoints")
CRNN_CHECKPOINT = os.path.join(CHECKPOINT_BASE, "hwr", "best_cer_model.pth")
CAPS_CHECKPOINT = os.path.join(CHECKPOINT_BASE, "hcr", "CharLevelCapsNet.pth")
YOLO_WEIGHTS = "best_word_det.pt"
TARGET_LANGUAGE = "en"

CACHE_DIR = os.path.join(BASE_DIR, "HandwrittenTextRecognition", "output_data", "refiner_cache")
CACHE_PAIRS = os.path.join(CACHE_DIR, "real_context_pairs.json")
CACHE_MATRIX = os.path.join(CACHE_DIR, "confusion_matrix.npy")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "google/byt5-base"
OUTPUT_DIR = "./byt5_htr_refiner_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LC_TOKEN = "~"
UNC_START = "<unc>"
UNC_END = "</unc>"
SEED = 42

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
CELERY_APP = Celery('htr_tasks', broker=f'redis://{REDIS_HOST}:6379/0')

def now():
    return time.strftime('%H:%M:%S')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_iam_forms_to_lines(sentences_txt):
    """ Zwraca słownik dla zachowania kontekstu. """
    forms_lines = {}
    with open(sentences_txt, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(' ')
            f_id = "-".join(parts[0].split('-')[:2])
            text = " ".join(parts[9:]).replace("|", " ")
            if f_id not in forms_lines: forms_lines[f_id] = []
            forms_lines[f_id].append(text)
    return forms_lines


def generate_contextual_noise_data(pipeline, forms_dict, char_list):
    """ Generuje pary 'poprzednia z 30% sznasy na błędy', 'obecna, jeszcze do poprawy' symulując sliding window. """
    char_to_idx = {c: i for i, c in enumerate(char_list)}
    matrix = np.zeros((len(char_list), len(char_list)), dtype=np.int32)
    real_contextual_pairs = []
    preprocessor = Preprocessing()

    # Inicjalizacja Twoich sprawdzonych segmentatorów
    line_segmentor = PageToLineSegmentor(min_line_height=15)
    word_segmentor = LineToWordSegmentor()

    print(f"Rozpoczynam analizę wizyjną z oknem kontekstowym.")
    for f_id, lines in tqdm(forms_dict.items(), desc="Przetwarzanie dokumentów"):
        img_path = os.path.join(FORMS_DIR, f"{f_id}.png")
        if not os.path.exists(img_path): continue
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None: continue

        # Wykorzystanie pełnego potoku preprocessingu z poprzedniego etapu
        clean_inv = preprocessor.full_pipeline(img)
        clean_norm = cv.bitwise_not(clean_inv)

        # Wyciąganie linii
        lines_crops_norm = line_segmentor.extract_lines(clean_norm)
        if not lines_crops_norm: continue

        prev_line_clean = ""

        # Grupujemy wyniki wizyjne w linie
        for l_idx, line_img_norm in enumerate(lines_crops_norm):
            tensors_crnn = []
            tensors_caps = []

            # Wyciąganie słów z linii
            word_crops_data = word_segmentor.extract_atomic_crops(line_img_norm)

            for crop_norm, (x, y, w, h) in word_crops_data:
                if h < 10 or w < 5: continue

                new_w = max(16, int(w * (64.0 / h)))
                crop_resized = cv.resize(crop_norm, (new_w, 64), interpolation=cv.INTER_CUBIC)
                _, crop_bin_norm = cv.threshold(crop_resized, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                t_c = (torch.from_numpy(crop_bin_norm).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

                crop_bin_inv = cv.bitwise_not(crop_bin_norm)
                caps_ready_img = Preprocessing.standardize_ink_thickness(crop_bin_inv, target_thickness=3)
                t_k = (torch.from_numpy(caps_ready_img).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

                tensors_crnn.append(t_c)
                tensors_caps.append(t_k)

            if not tensors_crnn: continue

            try:
                # Wywołanie z obydwoma listami tensorów
                res = pipeline.predict_batch(tensors_crnn, tensors_caps)
                curr_noisy_line = " ".join([r['final_result'] for r in res])

                # Szukamy GT dla tej linii w forms_dict
                if l_idx < len(lines):
                    target_clean = lines[l_idx]

                    matcher = difflib.SequenceMatcher(None, target_clean, curr_noisy_line)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'replace':
                            t_chunk = target_clean[i1:i2]  # Co miało być
                            p_chunk = curr_noisy_line[j1:j2]  # Co wyszło z CRNN i poprawki CapsNet

                            # Zliczamy tylko zamiany 1:1, żeby macierz miała sens
                            if len(t_chunk) == len(p_chunk):
                                for tc, pc in zip(t_chunk, p_chunk):
                                    if tc in char_to_idx and pc in char_to_idx:
                                        matrix[char_to_idx[tc], char_to_idx[pc]] += 1

                    real_contextual_pairs.append({
                        "context": prev_line_clean,  # Poprzednia linia
                        "input": curr_noisy_line,
                        "target": target_clean
                    })
                    prev_line_clean = target_clean
            except Exception as e:
                print(f"Błąd podczas predykcji: {e}")
                continue

    return real_contextual_pairs, matrix


class TransformerRefiner:
    """ Klasa implementująca model transformer do poprawy słów wykrytych przez modele. Obsługuje język polski i angielski.
        Polski nie jest używany przez CRNN, więc jego poprawa jest dość problematyczna. """
    def __init__(self, model_path, device, author_id="default_user", language='pl'):
        self.device = device
        self.language = language
        print(f"[{time.strftime('%H:%M:%S')}] Ładowanie modelu językowego ByT5 ({self.language.upper()}).")

        # Ładowanie tokenizatora z tokenami '~', '<unc>'
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_path)

        # Ładowanie wytrenowanego modelu
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"[{time.strftime('%H:%M:%S')}] Ładowanie adaptera LoRA dla ByT5.")
            # Ładujemy czystą bazę
            base_model = T5ForConditionalGeneration.from_pretrained(
                "google/byt5-base", torch_dtype=torch.float16
            ).to(self.device)
            # Nakładamy mały adapter użytkownika
            self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        else:
            # Fallback do pełnego modelu
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16
            ).to(self.device)
        self.model.eval()

        # Lista profili użytkowników
        self.loaded_adapters = []

        # Pamięć trwała autora (Słownikowa)
        self.author_memory = PersistentAuthorMemory(author_id)

        # Cache do optymalizacji powtarzających się błędów
        self._correction_cache = {}
        self.cache_size = 512

        # Słownik dla szybkiej weryfikacji zależny od wybranego języka
        try:
            self.spell = SpellChecker(language=self.language)
        except ValueError:
            print(f"[{time.strftime('%H:%M:%S')}] Brak słownika dla języka '{self.language}'. SpellChecker wyłączony.")
            self.spell = None

    def set_user_expert(self, user_id):
        """Dynamicznie przełącza wagi LoRA oraz bazę wiedzy autora w locie."""
        adapter_name = f"adapter_{user_id}"
        adapter_dir = f"/app/user_data/{user_id}/lora_weights"

        # Przełączamy pamięć słownikową
        self.author_memory = PersistentAuthorMemory(user_id)

        if not os.path.exists(adapter_dir):
            self.model.disable_adapter_layers()  # Powrót do czystego modelu
            return

        # Jeśli profil nie jest w RAM, ładujemy go
        if adapter_name not in self.loaded_adapters:
            self.model.load_adapter(adapter_dir, adapter_name=adapter_name)
            self.loaded_adapters.append(adapter_name)

        # Aktywacja wag konkretnego użytkownika
        self.model.enable_adapter_layers()
        self.model.set_adapter(adapter_name)

    def refine(self, noisy_text, prefix_context=""):
        """ Główna metoda korekty semantycznej z mechanizmem Cache. """
        context_text = prefix_context if prefix_context is not None else ""

        if not noisy_text.strip():
            return noisy_text

        # Sprawdzenie w cache dla szybkich poprawek krótkich fragmentów bez kontekstu
        cache_key = (noisy_text, context_text)
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]

        # Przygotowanie wejścia zgodnie z formatem treningowym i wybranym językiem
        if self.language == 'pl':
            full_prompt = f"kontekst: {context_text} popraw ocr: {noisy_text}"
        else:
            full_prompt = f"context: {context_text} fix ocr: {noisy_text}"

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                length_penalty=1.2,  # Lekkie promowanie dłuższych, sensownych zdań
                repetition_penalty=1.5,  # Blokuje powtarzanie tych samych słów
                early_stopping=True
            )

        refined_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Czyszczenie ewentualnych pozostałości po tagach, jeśli model ich nie usunął
        refined_text = re.sub(r'<.*?>|~', '', refined_text).strip()

        # Zarządzanie rozmiarem Cache
        if len(self._correction_cache) > self.cache_size:
            self._correction_cache.clear()
        self._correction_cache[cache_key] = refined_text

        return refined_text

    def get_word_embeddings(self, texts):
        """ Zamienia listę słów na ich reprezentacje wektorowe w celu ułatwienia współpracy między modelami. """
        if self.model is None or self.tokenizer is None:
            # Fallback, jeśli model nie jest załadowany
            return torch.zeros((len(texts), 768)).to(self.device)

        # Tokenizacja
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            if hasattr(self.model, 'encoder'):
                outputs = self.model.encoder(**encoded)
            else:
                outputs = self.model(**encoded)

            # Pobieramy ostatni stan ukryty
            last_hidden_state = outputs.last_hidden_state

            # Uśredniamy wektory liter, ignorując padding
            attention_mask = encoded['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            # Sumujemy wektory i dzielimy przez liczbę faktycznych tokenów
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Wynikowy wektor (embedding słowa)
            word_embeddings = sum_embeddings / sum_mask

        return word_embeddings

    def score_text(self, text, context_text=""):
        """ Oblicza prawdopodobieństwo tekstu w danym kontekście. """
        if self.language == 'pl':
            full_prompt = f"kontekst: {context_text} popraw ocr: {text}"
        else:
            full_prompt = f"context: {context_text} fix ocr: {text}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        labels = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, labels=labels)

            # Zwracamy log-likelihood (im wyższy, tym lepiej)
            return -outputs.loss.item()

    def predict_page_batched(self, lines_list):
        """ Przetwarza całą stronę (listę linii) sekwencyjnie. Poprzednia poprawiona linia staje się kontekstem dla następnej. """
        refined_page = []
        prev_context = ""

        print(f"[{time.strftime('%H:%M:%S')}] Start korekty semantycznej strony ({len(lines_list)} linii).")

        for line in lines_list:
            # Korekta obecnej linii przy użyciu kontekstu poprzedniej
            refined_line = self.refine(line, prefix_context=prev_context)
            refined_page.append(refined_line)

            # Aktualizacja kontekstu dla następnego kroku
            prev_context = refined_line

        return refined_page


class DubiousRegionSelector:
    """ Moduł selekcji regionów niepewnych. Identyfikuje fragmenty słowa, które wymagają weryfikacji przez CapsNet.
        Ignoruje interpunkcję (zostawiając ją CRNN i Transformerowi) oraz identyfikuje ryzykowne znaki pod kątem
        polskich diakrytyków."""
    def __init__(self, char_list: list[str], split_risky_chars: list[str]):
        self.char_list = char_list
        self.split_risky_chars = split_risky_chars

        # Znaki, które CRNN często podaje zamiast polskich ogonków
        self.polish_substitution_risky = {"a", "c", "e", "l", "n", "o", "s", "z", "x"}

        # Interpunkcja ignorowana przez CapsNet
        self.punctuation_marks = {".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")"}

        # Cache dla macierzy pomyłek
        self._confusing_indices = None
        self._last_matrix_path = None

    def _get_confusing_indices(self, path_str: str) -> set[int]:
        """ Ładuje indeksy pomyłek z walidacją struktury i obsługą błędów wejścia/wyjścia. """
        if not path_str:
            return set()

        # Używamy pathlib dla czystszej manipulacji ścieżkami
        path = Path(path_str)

        # Cache hit: jeśli ścieżka i dane są zgodne, nie dotykaj dysku
        if path == self._last_matrix_path and self._confusing_indices is not None:
            return self._confusing_indices

        try:
            # Sprawdzenie istnienia i czy to na pewno plik
            if not path.is_file():
                return set()

            # Próba załadowania danych
            cm_data = np.load(path)

            # Walidacja: czy to macierz 2D? (CapsNet wymaga kwadratowej macierzy klas)
            if not isinstance(cm_data, np.ndarray) or cm_data.ndim != 2:
                tqdm.write(
                    f"[{now()}] Warning: Macierz w {path.name} ma zły format (ndim={getattr(cm_data, 'ndim', 'N/A')}).")
                return set()

            cm = cm_data.astype(np.float64)

            # Obliczamy błędy (E): Suma wiersza (wszystkie próby) minus przekątna (poprawne)
            # E_i = \sum_{j=1}^{N} CM_{i,j} - CM_{i,i}
            error_scores = cm.sum(axis=1) - np.diag(cm)

            # Wybieramy top 15 najbardziej problematycznych indeksów
            result = set(np.argsort(error_scores)[-15:].tolist())

            # Aktualizacja cache'u tylko przy sukcesie
            self._confusing_indices = result
            self._last_matrix_path = path

            return result

        except (OSError, ValueError, TypeError) as e:
            # OSError: brak dostępu, błędy dysku, ValueError: uszkodzony plik .npy (np. niepełny zapis), TypeError: błędy typów przy operacjach NumPy
            tqdm.write(f"[{now()}] Błąd krytyczny macierzy pomyłek ({path.name}): {e}")
            return set()

    def select_dubious_groups(self, crnn_probs: torch.Tensor, width: int, confusion_matrix_path: str = "", intensive: bool = False) -> list[dict]:
        """  Inteligentny selektor regionów niepewnych. Decyduje, co wysłać do CapsNetu, a co zostawić dla Transformera. """
        steps = crnn_probs.shape[0]
        if steps == 0: return []

        # Softmax, aby operować na prawdopodobieństwach (0.0 - 1.0)
        probs = torch.softmax(crnn_probs, dim=-1)
        top_p, top_idx = torch.topk(probs, 2, dim=-1)
        top1_p, top1_idx = top_p[:, 0], top_idx[:, 0]
        margin = top1_p - top_p[:, 1]

        stride = width / steps
        # Szerokość okna zależy od trybu (intensive = głębsza analiza)
        window_factor = 3.5 if intensive else 2.8
        half_window = int((stride * window_factor) / 2)

        candidates = []

        # Progi decyzyjne
        base_threshold = 0.60 if intensive else 0.50
        substitution_threshold = 0.75  # Dla par często mylonych (z macierzy pomyłek)
        polish_risk_threshold = 0.85  # Wysoki próg dla 'a', 'c', 'e' itp.
        margin_min = 0.18  # Jeśli różnica między Top1 a Top2 jest mała, wysyłaj do CapsNet

        # Definicja znaków wspieranych przez CapsNet
        capsnet_targets = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

        for t in range(steps):
            char_idx = int(top1_idx[t].item())
            char_str = self.char_list[char_idx] if char_idx < len(self.char_list) else ""

            # Jeśli to interpunkcja, nie obciążaj CapsNetu
            if char_str in self.punctuation_marks or char_str not in capsnet_targets:
                continue

            # Zwiększamy czujność dla polskich znaków
            current_limit = base_threshold
            if char_str.lower() in self.polish_substitution_risky:
                current_limit = polish_risk_threshold

            # Czy CRNN jest wystarczająco pewny
            if top1_p[t] < current_limit or margin[t] < margin_min:
                center_x = int((t + 0.5) * stride)
                candidates.append({
                    't': t,
                    'x1': max(0, center_x - half_window),
                    'x2': min(width, center_x + half_window),
                    'prob': top1_p[t].item(),
                    'char': char_str
                })

        if not candidates: return []

        # Agregacja bliskich kroków czasowych w jedno "słowo/znak" dla CapsNetu
        final_groups = []
        current_group = [candidates[0]]

        for i in range(1, len(candidates)):
            # Jeśli kroki są blisko siebie (do 3 klatek), traktujemy to jako jeden obiekt/błąd
            if candidates[i]['t'] - current_group[-1]['t'] <= 3:
                current_group.append(candidates[i])
            else:
                final_groups.append(self._merge_group(current_group))
                current_group = [candidates[i]]

        final_groups.append(self._merge_group(current_group))
        return final_groups

    @staticmethod
    def _merge_group(group: list[dict]) -> dict:
        """ Łączy kilka kroków czasowych w jeden spójny wycinek dla CapsNetu. """
        return {
            'timestep': group[len(group) // 2]['t'],
            'x1': min(c['x1'] for c in group),
            'x2': max(c['x2'] for c in group)
        }


class VisualExplainer:
    """ Klasa tłumacząca decyzje modelu CapsNet na czytelne mapy istotności pikseli (XAI). """
    def __init__(self, model):
        self.model = model
        self.model.eval()

        # Inicjalizacja algorytmu Integrated Gradients
        self.ig = IntegratedGradients(self.forward_wrapper)

    def forward_wrapper(self, inputs, word_context=None):
        """ Wrapper dla Captum, ponieważ CapsNet zwraca słownik/krotkę, a potrzebny jest Tensor norm. """
        outputs = self.model(inputs, word_context=word_context)

        # Pobieramy normy kapsułek (prawdopodobieństwa klas)
        return outputs["norms"] if isinstance(outputs, dict) else outputs[0]

    def explain(self, img_tensor, context_vector, target_class_idx):
        """ Generuje mapę atrybucji (istotności) dla wybranego znaku. """
        img_tensor.requires_grad = True
        
        # Obliczanie atrybucji
        attributions = self.ig.attribute(
            img_tensor,
            additional_forward_args=(context_vector,),
            target=target_class_idx,
            n_steps=24  # Liczba kroków całkowania (im więcej, tym dokładniej)
        )
        return attributions.squeeze().cpu().detach().numpy()

    @staticmethod
    def plot_explanation(img_tensor, attributions, char_label, save_path):
        """ Tworzy porównanie oryginału z mapą ciepła XAI i zapisuje do pliku. """
        img = img_tensor.squeeze().cpu().detach().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Oryginał: {char_label}")
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Analiza istotności (XAI)")
        # Nakładamy mapę ciepła (hot) na piksele
        plt.imshow(attributions, cmap='hot')
        plt.colorbar()

        plt.savefig(save_path)
        plt.close()


class FinalInferenceEngine:
    """ Pełny proces kaskadowy: CRNN -> CapsNet -> Transformer. """
    def __init__(self, crnn_pipe, transformer_refiner):
        self.vision = crnn_pipe  # CRNNInferencePipeline połączony z CascadeRefinementNetwork
        self.semantic = transformer_refiner  # TransformerRefiner

    def get_decoded_text(self, probs: torch.Tensor) -> list[str]:
        """ Dynamiczne pobranie metody omija statyczną analizę lintera. """
        decoder = getattr(self.vision, "_decode_ctc")
        return decoder(probs)

    def process_word(self, img_crnn, img_caps, prev_context=""):
        self.vision.model.eval()

        # Wynik surowy CRNN
        with torch.no_grad():
            raw_logits, context_maps = self.vision.model(img_crnn.to(DEVICE), return_context=True)
            crnn_text = self.vision.encoder.decode_greedy(raw_logits)[0]
            log_probs = raw_logits.log_softmax(2)

        # Poprawka CapsNet
        probs_refined, unc_mask, caps_conf, debug_data = self.vision.refiner.refine_logits(
            log_probs.squeeze(1),
            img_caps.to(DEVICE),
            context_map=context_maps[0]
        )
        caps_text = self.get_decoded_text(probs_refined)[0]

        if caps_conf < 0.7 or any(c in caps_text for c in "ąćęłńóśźż"):
            try:
                # Inicjalizacja wyjaśniacza dla sieci kapsułowej
                explainer = VisualExplainer(self.vision.refiner.caps_net)

                # debug_data zawiera listę wycinków i ich pozycji
                for i, meta in enumerate(debug_data['metadata']):
                    t = meta['timestep']

                    # Jeśli ten krok czasowy jest oznaczony jako niepewny
                    if unc_mask[t] > 0:
                        char = caps_text[t] if t < len(caps_text) else "?"
                        target_idx = self.vision.encoder.char_to_idx.get(char, 0)

                        # Pobieramy konkretny wycinek i wektor kontekstu
                        crop_tensor = debug_data['tensors'][i]
                        context_vec = debug_data['contexts'][i]

                        # Generujemy mapę istotności pikseli
                        attr_map = explainer.explain(crop_tensor, context_vec, target_idx)

                        # Zapis do folderu
                        debug_path = os.path.join(VISUAL_DEBUG_DIR, f"xai_{char}_{int(time.time())}.png")
                        explainer.plot_explanation(crop_tensor, attr_map, char, debug_path)

            except Exception as e:
                print(f"Błąd XAI: {e}")

        # Przygotowanie wejścia dla Transformera
        tagged_input = ""
        for char, m in zip(caps_text, unc_mask):
            if m > 0:
                tagged_input += f"<unc>{char}</unc>"
            else:
                tagged_input += char

        # Korekta Semantyczna ByT5
        semantic_proposal = self.semantic.refine(tagged_input, context_text=prev_context)
        semantic_score = self.semantic.score_text(semantic_proposal, context_text=prev_context)
        semantic_conf = min(max(1.0 + (semantic_score / 20.0), 0.1), 1.0)

        # Dynamiczne wagowanie
        alpha = 0.6

        # Po długości słowa (dłuższe słowa = silniejszy kontekst językowy)
        if len(caps_text) > 7:
            alpha -= 0.15
        elif len(caps_text) <= 3:
            alpha += 0.15

        # Relatywna pewność (jeśli CapsNet jest bardzo pewny, podbijamy jego wagę)
        if caps_conf > 0.92:
            alpha += 0.1
        elif caps_conf < 0.6:
            alpha -= 0.2

        # Zabezpieczenie zakresu alfy (
        alpha = max(min(alpha, 0.9), 0.2)

        # Finalna fuzja
        hybrid_conf = (alpha * caps_conf) + ((1.0 - alpha) * semantic_conf)

        # Veto
        dist = Levenshtein.distance(caps_text, semantic_proposal)
        change_ratio = dist / max(len(caps_text), 1)
        len_ratio = abs(len(caps_text) - len(semantic_proposal)) / max(len(caps_text), 1)

        is_rejected = False
        final_text = semantic_proposal
        final_text = re.sub(r'<.*?>|~', '', final_text).strip()

        # Blokada na Transformera (zbyt chętny do dodawania znaków)
        if change_ratio > 0.4 and caps_conf > 0.80:
            final_text = caps_text
            is_rejected = True

        # Ekstremalne zaufanie do wizji (np. nazwiska, których nie ma w słowniku)
        elif caps_conf > 0.96 and semantic_conf < 0.5:
            final_text = caps_text
            is_rejected = True

        # Zabezpieczenie przed halucynacjami (drastyczna zmiana długości)
        elif len_ratio > 0.3 and caps_conf > 0.7:
            final_text = caps_text
            is_rejected = True

        # Całościowy brak zaufania do wyniku hybrydowego
        elif hybrid_conf < 0.35:
            final_text = caps_text
            is_rejected = True

        return {
            "crnn_result": crnn_text,
            "caps_result": caps_text,
            "final_result": final_text,
            "is_semantic_rejected": is_rejected,
            "uncertain_zones": unc_mask,
            "caps_confidence": round(caps_conf, 3),
            "hybrid_confidence": round(float(hybrid_conf), 3),
            "dynamic_alpha": round(alpha, 2)  # Dodajemy dla celów debugowania
        }

class CascadeRefinementNetwork:
    """ Moduł kaskadowej weryfikacji predykcji HTR łączący analizę sekwencyjną z lokalną.
        Klasa implementuje mechanizm 'drugiego spojrzenia'. Działa jako warstwa pośrednia między modelem
        CRNN a Transformerem. Na podstawie map prawdopodobieństwa identyfikuje fragmenty obrazu, które są niejednoznaczne
        dla modelu sekwencyjnego, a następnie wykorzystuje sieć kapsułową do ich precyzyjnej re-klasyfikacji.
        Główne zadania:
        1. Selekcja regionów o niskiej ufności.
        2. Ekstrakcja wielowariantowych wycinków w celu stabilizacji rozpoznawania.
        3. Fuzja cech wizualnych wycinka z 1024-wymiarowym wektorem kontekstu rekurencyjnego.
        4. Decyzyjne nadpisywanie rozkładów prawdopodobieństwa przed etapem korekty semantycznej. """
    def __init__(self, capsnet_predictor, char_list, encoder_obj, matrix_dir=None, matrix_path=None):
        self.matrix_path = matrix_path
        if self.matrix_path is None and matrix_dir is not None:
            self.matrix_path = os.path.join(matrix_dir, "confusion_matrix.npy")

        # Selektor odpowiada za precyzyjne wskazywanie 'podejrzanych' fragmentów obrazu
        self.region_selector = DubiousRegionSelector(char_list, matrix_dir)

        self.caps_net = capsnet_predictor
        self.encoder = encoder_obj
        self.char_list = char_list
        self.device = next(capsnet_predictor.parameters()).device

        # Obsługa tokena sygnalizującego niepewność (~) dla Transformera
        self.lc_token = "~"
        self.lc_idx = self.encoder.char_to_idx.get(self.lc_token, len(char_list))

        # Mapowanie wyjść CapsNet na alfabet EMNIST-62
        self.caps_mapping = self._get_emnist_62_mapping()
        self.caps_net.eval()

    def refine_logits(self, crnn_probs: torch.Tensor, image_tensor: torch.Tensor, context_map: torch.Tensor = None, force_intensive: bool = False):
        """ Kaskadowa optymalizacja: CapsNet weryfikuje 'dubious regions' z CRNN. Wykorzystuje mechanizm Deep Fusion (Context 1024d). """
        if image_tensor.dim() == 5:
            image_tensor = image_tensor.squeeze(0)

        # Szerokość obrazu do mapowania kroków czasowych na piksele
        width = image_tensor.shape[-1]
        device = crnn_probs.device

        # Selekcja grup na podstawie entropii i macierzy pomyłek
        groups = self.select_dubious_groups(
            crnn_probs,
            width,
            self.encoder.char_to_idx,
            self.matrix_path,
            intensive=force_intensive
        )

        uncertainty_mask = torch.zeros(crnn_probs.size(0), dtype=torch.int32, device=device)

        if not groups:
            return crnn_probs, uncertainty_mask.cpu().tolist(), 1.0

        batch_tensors, batch_contexts, batch_metadata = [], [], []

        # Pre-generacja zerowego kontekstu (optymalizacja)
        zero_context = torch.zeros(1024, device=device)

        for g in groups:
            mid_cand = g[len(g) // 2]
            t_idx = mid_cand['timestep']

            # Wycięcie fragmentu dla CapsNet
            crop = image_tensor[..., mid_cand['x1']:mid_cand['x2']]
            if crop.shape[-1] < 8:
                continue

            # Normalizacja i zmiana rozmiaru pod CapsNet
            if crop.dim() == 3: crop = crop.unsqueeze(0)
            crop_norm = (func.interpolate(crop, size=(64, 64), mode='bilinear', align_corners=False) - 0.5) / 0.5
            batch_tensors.append(crop_norm)

            # Pobranie kontekstu rekurencyjnego
            if context_map is not None:
                ctx = context_map[min(t_idx, context_map.size(0) - 1)].view(-1)
            else:
                ctx = zero_context
            batch_contexts.append(ctx)
            batch_metadata.append({'timestep': t_idx})

        if not batch_tensors:
            return crnn_probs, uncertainty_mask.cpu().tolist(), 1.0

        # Batchowa inferencja CapsNet (multimodalna)
        with torch.no_grad():
            input_batch = torch.cat(batch_tensors)
            context_batch = torch.stack(batch_contexts)

            caps_output = self.caps_net(input_batch, word_context=context_batch)
            caps_probs = caps_output["norms"]

        top_probs, top_indices = torch.topk(caps_probs, 2, dim=1)
        confidences = []

        # Proces nadpisywania logitów
        refined_probs = crnn_probs.clone()  # Unikamy in-place na oryginale

        for i, meta in enumerate(batch_metadata):
            t = meta['timestep']
            margin = (top_probs[i, 0] - top_probs[i, 1]).item()
            confidences.append(top_probs[i, 0].item())

            # Jeśli CapsNet się waha, oznaczamy to dla Transformera
            if margin < 0.2:
                uncertainty_mask[t] = 1

            # Próg interwencji: CapsNet musi być pewny swego
            if top_probs[i, 0] > 0.85:
                char_code = top_indices[i, 0].item()
                char = self.caps_mapping.get(char_code, '?')
                idx = self.encoder.char_to_idx.get(char)

                if idx is not None:
                    refined_probs[t, :] *= 0.1
                    refined_probs[t, idx] = 15.0

        avg_caps_conf = sum(confidences) / len(confidences) if confidences else 1.0
        return refined_probs, uncertainty_mask.cpu().tolist(), avg_caps_conf

    @staticmethod
    def _get_emnist_62_mapping():
        m = {}
        for i in range(10): m[i] = chr(48 + i)
        for i in range(26): m[10 + i] = chr(65 + i)
        for i in range(26): m[36 + i] = chr(97 + i)
        return m

    @staticmethod
    def _preprocess_crop_gpu(crop_tensor):
        crop_resized = func.interpolate(crop_tensor, size=(64, 64), mode='bilinear', align_corners=False)
        crop_norm = (crop_resized * 0.5) + 0.5
        mean, std = 0.1307, 0.3081
        return (crop_norm - mean) / std

    @staticmethod
    def select_dubious_groups(crnn_probs, width, char_to_idx, confusion_matrix_path, intensive=False):
        """ Identyfikuje regiony obrazu, przy których model CRNN wykazuje wysoką niepewność predykcji.
            Logika działania:
            1. Oblicza entropię Shannona dla każdego kroku czasowego sekwencji.
               Wysoka entropia oznacza, że rozkład prawdopodobieństwa jest płaski (model nie jest pewny).
            2. Mapuje kroki czasowe na współrzędne pikselowe obrazu wejściowego, uwzględniając
               skalowanie wynikające z architektury sieci.
            3. Grupuje sąsiadujące kroki o niskiej ufności w spójne regiony. """
        steps = crnn_probs.shape[0]
        if steps == 0: return []

        top1_p, top1_idx = torch.max(crnn_probs, dim=-1)

        # Obliczamy szerokość jednego kroku
        stride = width / steps

        # Dynamiczny margines okna
        window_factor = 3.2 if intensive else 2.6
        half_window = int((stride * window_factor) / 2)

        candidates = []

        # Logika Smart Adaptive Thresholding
        try:
            if os.path.exists(confusion_matrix_path):
                cm = np.load(confusion_matrix_path).astype(np.float64)
                row_sums = cm.sum(axis=1)
                errors_per_char = row_sums - np.diag(cm)
                confusing_indices = set(np.argsort(errors_per_char)[-15:])
            else:
                confusing_indices = set()
        except (OSError, ValueError, TypeError, AttributeError) as e:
                # OSError: błędy dysku, ValueError: błąd unpickling, TypeError/AttributeError: błędy NumPy
                tqdm.write(f"[{now()}] Błąd Smart Thresholding: {e}. Używam domyślnych progów.")
                confusing_indices = set()

        # Klasy ryzykowne (rozbicia sekwencji)
        split_components = {"r", "n", "i", "l", "c", "v", "u"}
        split_indices = {char_to_idx.get(c) for c in split_components if char_to_idx.get(c) is not None}

        # Progi decyzyjne
        base_limit = 0.65 if intensive else 0.55
        substitution_limit = 0.78  # Dla znaków z macierzy
        split_limit = 0.88  # Dla ryzyka rozbicia

        for t in range(steps):
            char_idx = top1_idx[t].item()

            # Wybór progu
            limit = split_limit if char_idx in split_indices else (
                substitution_limit if char_idx in confusing_indices else base_limit)

            if top1_p[t] < limit:
                center_x = int((t + 0.5) * stride)
                candidates.append({
                    'timestep': t,
                    'x1': max(0, center_x - half_window),
                    'x2': min(width, center_x + half_window)
                })

        if not candidates: return []

        # Grupowanie bliskich niepewności
        groups, current = [], [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i]['timestep'] - current[-1]['timestep'] <= 3:
                current.append(candidates[i])
            else:
                groups.append(current)
                current = [candidates[i]]
        groups.append(current)
        return groups

    @staticmethod
    def save_transformer_training_set(data, filename="user_transformer_hard_cases.json"):
        """ Zapisuje dane wygenerowane przez wizję do formatu akceptowalnego przez Transformera. """
        formatted_data = []
        for item in data:
            formatted_data.append({
                "input": item.get("vision_output", item.get("input", "")),
                "target": item.get("ground_truth", item.get("target", "")),
                "confidence": float(item.get("confidence", 0.0)),
                "needs_fix": bool(item.get("needs_fix", True))
            })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        print(f"[{time.strftime('%H:%M:%S')}] Dane semantyczne zapisane: {filename}")


class HybridVisualErrorGenerator:
    """ Zaawansowany generator szumu symulujący błędy fizyczne i statystyczne HTR. """
    def __init__(self, matrix, char_list):
        self.char_list = char_list
        self.char_to_idx = {c: i for i, c in enumerate(char_list)}

        # Słownik złączeń i rozbić
        self.common_merges = {
            "rn": "m", "nn": "m", "ni": "m", "cl": "d",
            "ol": "d", "vv": "w", "ri": "n", "li": "u"
        }
        self.common_splits = {
            "m": "rn", "w": "vv", "d": "cl", "n": "ri", "u": "li"
        }

        # Przygotowanie prawdopodobieństw z macierzy pomyłek
        m = matrix.astype('float32')
        np.fill_diagonal(m, 0)
        sums = m.sum(axis=1, keepdims=True)
        self.probs = np.divide(m, sums, out=np.zeros_like(m), where=sums > 1e-6)


    def apply_errors(self, text, rate=0.15, use_markers=False):
        """ Generuje błędy z uwzględnieniem dryfu OCR i fizycznych zniekształceń. """
        res = []
        i = 0
        while i < len(text):
            char = text[i]
            two_chars = text[i:i + 2]
            rand = random.random()

            # Symulacja dryfu (OCR drift) - sąsiedni znak wpływa na obecny
            if 0 < i < len(text) - 1 and random.random() < 0.02:
                # Znak zlewa się z sąsiadem
                neighbor = text[i+1] if random.random() > 0.5 else text[i-1]
                char = random.choice([char, neighbor, char + neighbor])
                if len(char) > 1:
                    res.append(f"{UNC_START}{char}{UNC_END}" if use_markers else char)
                    i += 1
                    continue

            # Symulacja złączeń znaków
            if two_chars in self.common_merges and rand < 0.30:
                res.append(self.common_merges[two_chars])
                i += 2
                continue

            # Symulacja rozbić znaku
            if char in self.common_splits and rand < 0.12:
                split_val = self.common_splits[char]
                if use_markers:
                    res.append("".join([f"{UNC_START}{c}{UNC_END}" for c in split_val]))
                else:
                    res.append(split_val)
                i += 1
                continue

            # Statystyczne błędy z Macierzy Pomyłek
            if rand < rate:
                src_idx = self.char_to_idx.get(char)
                if src_idx is not None and self.probs[src_idx].sum() > 0.5:
                    p_row = self.probs[src_idx]
                    p_row = p_row / p_row.sum()
                    new_c = np.random.choice(self.char_list, p=p_row)
                else:
                    new_c = random.choice(self.char_list) if random.random() < 0.15 else char

                if use_markers and random.random() < 0.7:
                    res.append(f"{UNC_START}{new_c}{UNC_END}")
                else:
                    res.append(new_c)
            else:
                # Prawidłowy znak, ale z małą szansą na fałszywy alarm (False Positive)
                if use_markers and random.random() < 0.03:
                    res.append(f"{UNC_START}{char}{UNC_END}")
                else:
                    res.append(char)

            i += 1
        return "".join(res)


class ContextualCurriculumDataset(torch.utils.data.Dataset):
    """ Zbiór danych implementujący mechanizm przesuwnego okna: [Kontekst] + [Zaszumione wejście] -> [Oczyszczony cel] """
    def __init__(self, wiki_texts, real_pairs, generator, tokenizer):
        self.wiki_texts = wiki_texts  # Lista par (prev, curr) z Wiki
        self.real_pairs = real_pairs  # Lista słowników {context, input, target} z IAM/CVL
        self.generator = generator
        self.tokenizer = tokenizer
        self.rate = 0.05
        self.markers = False

    def __len__(self):
        return len(self.wiki_texts) + len(self.real_pairs)

    def __getitem__(self, idx):
        if idx < len(self.real_pairs):
            item = self.real_pairs[idx]
            clean_prev = item['context']
            clean_curr = item['target']
            source_text = item['input'] # Jawne przypisanie tekstu źródłowego
            is_real = True
        else:
            idx_wiki = idx - len(self.real_pairs)
            clean_prev, clean_curr = self.wiki_texts[idx_wiki]
            source_text = clean_curr
            is_real = False

        # Obsługa kontekstu z poprzednią linijką jako referencja
        if not clean_prev or random.random() < 0.2:
            context_for_model = ""
        else:
            clean_prev = clean_prev[-150:]
            # 30% Szans na wygenerowany błąd
            if random.random() < 0.3:
                context_for_model = self.generator.apply_errors(clean_prev, rate=0.03)
            else:
                context_for_model = clean_prev

        # Zaszumianie aktualnego wejścia (Data Augmentation dla Transformera)
        if is_real:
            # Sprawdzamy, czy zmienna w ogóle dotarła do tego miejsca
            actual_source = locals().get('real_noisy_curr', clean_curr)

            if random.random() < 0.2:
                # 20% szansy: bierzemy "brudny" wynik z OCR i psujemy go jeszcze bardziej (Hard Mining)
                noisy_curr = self.generator.apply_errors(source_text, self.rate, self.markers)
            else:
                # 80% szansy: bierzemy autentyczny błąd z modelu CRNN
                noisy_curr = actual_source
        else:
            # Scenariusz syntetyczny. Mamy idealny tekst, więc zawsze musimy nałożyć błędy sztucznie.
            noisy_curr = self.generator.apply_errors(clean_curr, self.rate, self.markers)

        # Składanie wejścia dla modelu
        src = f"context: {context_for_model} fix ocr: {noisy_curr}"
        tgt = clean_curr

        m_in = self.tokenizer(src, max_length=512, truncation=True, return_tensors="pt")
        m_tg = self.tokenizer(text_target=tgt, max_length=512, truncation=True, return_tensors="pt")

        # Przygotowanie etykiet
        labels = m_tg["input_ids"].squeeze(0).clone()

        return {
            "input_ids": m_in["input_ids"].squeeze(0),
            "attention_mask": m_in["attention_mask"].squeeze(0),
            "labels": labels
        }


class CurriculumCallback(TrainerCallback):
    """ Zarządca progresywnego uczenia. Dynamicznie zwiększa poziom trudności zadania w każdej epoce poprzez
        podnoszenie współczynnika błędów aktywację markerów niepewności, co zapobiega overfittingowi modelu ByT5."""
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch)
        """ 1-2 epoka: lekki szum (nauka literówek)
            3-4 epoka: silny szum + markery (nauka na błędach CapsNetu)
            5-6 epoka: ekstremalny szum (wyciskanie ostatnich procentów) """
        rate = 0.05 if ep < 2 else (0.15 if ep < 4 else 0.2)
        markers = ep >= 2

        self.dataset.rate = rate
        self.dataset.markers = markers
        print(f"\n[Curriculum] Epoka {ep + 1}: Szum {rate * 100:.1f}%, Markery: {markers}")


def get_contextual_data_cached(pipeline, layout_engine, forms_dict, char_list, cache_path="context_cache.json"):
    """ Ładuje dane z cache lub generuje je, jeśli nie istnieją. """
    matrix_path = "confusion_matrix.npy"
    if os.path.exists(cache_path) and os.path.exists(matrix_path):
        print(f"[{time.strftime('%H:%M:%S')}] Ładowanie danych kontekstowych z cache.")
        with open(cache_path, 'r', encoding='utf-8') as f:
            real_pairs = json.load(f)
        matrix = np.load(matrix_path)
        return real_pairs, matrix

    # Jeśli nie ma cache, uruchom oryginalną funkcję
    real_pairs, matrix = generate_contextual_noise_data(pipeline, forms_dict, char_list)

    # Zapisz do cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(real_pairs, f)
    np.save(matrix_path, matrix)
    return real_pairs, matrix

# Ładujemy metrykę raz, na poziomie modułu
cer_metric = load("cer")
def compute_metrics(eval_preds, tokenizer):
    """ Oblicza CER dla Transformera (ByT5/T5). """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Zastępujemy -100 (maska straty) ID padu, aby tokenizer mógł to zdekodować
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Dekodowanie bajtów/znaków na tekst
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Czyszczenie tagów XML/specjalnych (kluczowe dla PHSF i geoms)
    decoded_preds = [re.sub(r'<.*?>|~', '', p).strip() for p in decoded_preds]
    decoded_labels = [re.sub(r'<.*?>|~', '', l).strip() for l in decoded_labels]

    # Obliczenie finalnego wyniku CER
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"cer": cer}

def set_user_expert(self, user_id):
    """ Dynamicznie przełącza wagi LoRA dla użytkownika. """
    adapter_name = f"adapter_{user_id}"
    adapter_dir = f"/app/user_data/{user_id}/lora_weights"

    # Jeśli brak profilu na dysku, to czysty Transformer
    if not os.path.exists(adapter_dir):
        self.model.disable_adapter_layers()
        return

    # Jeśli profil jest, ale nie ma go w RAM, to ładujemy model personalizowany
    if adapter_name not in self.loaded_adapters:
        self.model.load_adapter(adapter_dir, adapter_name=adapter_name)
        self.loaded_adapters.append(adapter_name)

    # Aktywacja konkretnej wiedzy eksperckiej
    self.model.enable_adapter_layers()
    self.model.set_adapter(adapter_name)

def run_transformer_adaptation(json_data_path, model_path, output_path, matrix_path, encoder):
    """ Funkcja do personalizacji modelu ByT5 na błędach specyficznych dla użytkownika.
        Wczytuje wstępnie wytrenowany model i dotrenowuje go na dostarczonych przykładach. """
    print(f"[{time.strftime('%H:%M:%S')}] Rozpoczynam adaptację Transformera dla użytkownika.")

    if not os.path.exists(json_data_path):
        print(f"[{time.strftime('%H:%M:%S')}] Błąd: Nie znaleziono pliku z danymi do adaptacji: {json_data_path}")
        return

    # Wczytanie danych błędów semantycznych użytkownika
    with open(json_data_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)

    # Przygotowanie danych w formacie dla Dataset
    user_pairs = []
    for item in user_data:
        # Formatowanie: wejście (to co widzi wizja) -> cel (poprawny tekst)
        vision_output = item.get("vision_output", "")
        ground_truth = item.get("ground_truth", "")
        # Symulacja pustego kontekstu, skupiamy się na naprawie słowa/linii
        user_pairs.append({"context": "", "input": vision_output, "target": ground_truth})

    print(f"[{time.strftime('%H:%M:%S')}] Wczytano {len(user_pairs)} próbek do personalizacji.")

    # Inicjalizacja tokenizatora z tokenami specjalnymi
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)

    # Wczytanie macierzy pomyłek użytkownika (lub bazy) w celu zachowania ciągłości generatora szumu
    if os.path.exists(matrix_path):
        confusion_matrix = np.load(matrix_path)
    else:
        # Fallback do pustej macierzy, jeśli nie podano
        confusion_matrix = np.zeros((len(encoder.char_list) if hasattr(encoder, 'char_list') else 100,
                                     len(encoder.char_list) if hasattr(encoder, 'char_list') else 100))

    # Tworzenie uproszczonego generatora szumu na potrzeby zbioru walidacyjnego (tutaj 100% realnych błędów)
    error_gen = HybridVisualErrorGenerator(
        confusion_matrix,
        encoder.char_list if hasattr(encoder, 'char_list') else list(range(confusion_matrix.shape[0]))
    )

    # Wykorzystujemy ContextualCurriculumDataset, podając do niego puste listy
    train_ds = ContextualCurriculumDataset(wiki_texts=[], real_pairs=user_pairs, generator=error_gen, tokenizer=tokenizer)

    # Niski poziom dodatkowego szumu
    train_ds.rate = 0.05
    train_ds.markers = False

    # Ładowanie modelu bazowego
    base_model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Konfiguracja LoRA (Low-Rank Adaptation) - Trenujemy tylko warstwy atencji (q, v), reszta sieci jest zamrożona.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q", "k", "v", "o",
            """ Warstwy Atencji (Multi-head Attention): Odpowiadają za dynamiczne kierowanie uwagi 
                modelu na istotne glify w zaszumionym tekście OCR oraz rozumienie instrukcji 
                zawartych w prefixach bilingwalnych <pl> i <en>. """,

            "wi_0", "wi_1", "wo",
            """ Bloki Feed-Forward (FFN): Pełnią rolę 'pamięci masowej' modelu, przechowując 
                wiedzę o słownictwie i strukturach gramatycznych obu języków. Ich adaptacja 
                jest kluczowa dla skutecznej korekty semantycznej i unikania halucynacji. """
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # Zastosowanie LoRA na model bazowy
    model = get_peft_model(base_model, lora_config)

    # Pokaże ile % parametrów będzie trenowanych
    model.print_trainable_parameters()
    model.to(DEVICE)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Parametry fine-tuningu (znacznie krótszy i łagodniejszy trening niż główny)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,  # Krótki fine-tuning, aby uniknąć overfittingu do małej próbki
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,  # Mniejszy learning rate (dociąganie)
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True, # Zmniejsza zużycie vRAM
        optim="adamw_torch_fused", # Fused jest szybsza i lżejsza (może się zmieści na 4gb vRAM)
        save_strategy="no",  # Nie zapisujemy checkpointów pośrednich
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,  # Pozwala na załadowanie najlepszych wag po treningu
        metric_for_best_model="cer",  # Wskazuje na metrykę CER, jako wskaźnik jakości
        greater_is_better=False,  # Informuje, że mniejszy CER = lepszy model
        predict_with_generate=True  # Niezbędne do poprawnego obliczania metryk tekstowych

    )

    if train_ds is not None:
        # Obliczamy rozmiary (90% na trening, reszta na walidację)
        train_size = int(0.9 * len(train_ds))
        val_size = len(train_ds) - train_size

        # Zapewnia to, że przy każdym uruchomieniu te same próbki trafią do walidacji.
        train_ds, val_ds = random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        tqdm.write(f"[{now()}] Podział zakończony: Train={len(train_ds)}, Val={len(val_ds)}")
    else:
        val_ds = None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)  # Stop po 3 epokach bez poprawy
        ]
    )

    print(f"[{time.strftime('%H:%M:%S')}] Rozpoczynam dociąganie wag ByT5.")
    trainer.train()

    # Zapis spersonalizowanego modelu
    os.makedirs(output_path, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[{time.strftime('%H:%M:%S')}] Personalizacja zakończona. Zapisano do: {output_path}")


def apply_stochastic_lth_step(model, initial_state_dict, base_amount=0.05, mode="magnitude_jitter"):
    """ Wykonuje krok Pruningu z elementami stochastycznymi. Usuwa najsłabsze wagi, ale z losowym odchyleniem progu.
        Całkowicie losowe usuwanie wag, jako test hipotezy "Random Ticket". """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):

            # Usuwamy stare maski pruningu z poprzedniej epoki, żeby nie nakładać w nieskończoność
            if prune.is_pruned(module):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass

            if mode == "magnitude_jitter":
                # Dodajemy losową wariancję do progu cięcia (+/- 20% bazowej wartości)
                jitter = random.uniform(0.8, 1.2)
                actual_amount = min(0.99, base_amount * jitter)
                prune.l1_unstructured(module, name='weight', amount=actual_amount)

            elif mode == "random":
                # Usuwa losowe wagi niezależnie od ich wielkości
                prune.random_unstructured(module, name='weight', amount=base_amount)


class StochasticPruningCallback(TrainerCallback):
    def __init__(self, initial_state, base_amount=0.1):
        self.initial_state = initial_state
        self.base_amount = base_amount

    def on_epoch_end(self, args, state, control, **kwargs):
        # Wyciągamy model ze słownika kwargs
        model = kwargs.get("model")

        if model is None:
            return

        # Pomijamy pruning w ostatniej epoce
        if state.epoch < args.num_train_epochs:
            print(f"\n[LTH Pruning] Stochastyczna kompresja wag po epoce {int(state.epoch)}.")

            # Używamy magnitude_jitter, co jest najbezpieczniejsze dla Transformerów
            apply_stochastic_lth_step(
                model,
                self.initial_state,
                base_amount=self.base_amount,
                mode="magnitude_jitter"
            )


def refine_ocr_text(noisy_text, context_text, model, tokenizer, device, language="pl"):
    """ Pojedyncza funkcja realizująca pełną logikę korekty semantycznej.
        Przekształca zaszumiony tekst z systemów wizyjnych (CRNN/CapsNet) w poprawne zdanie. """
    if not noisy_text.strip():
        return noisy_text

    # Budowa promptu sterującego w zależności od wybranego języka
    if language == "pl":
        prompt = f"kontekst: {context_text} popraw HCR: {noisy_text}"
    else:
        prompt = f"context: {context_text} fix HCR: {noisy_text}"

    # Tokenizacja (zamiana tekstu na wektory zrozumiałe dla modelu)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)

    # Generowanie poprawionego tekstu przez model
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=2,  # Analizuje 4 ścieżki naraz, żeby wybrać najbardziej sensowne słowo
            length_penalty=1,  # Lekko promuje dłuższe, naturalniejsze zdania
            repetition_penalty=1.2,  # Mniejsza kara, na wypadek podwójnych zbitek tych samych liter
            early_stopping=True
        )

    # Dekodowanie z powrotem na czytelny tekst
    refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Oczyszczanie z pozostałości znaczników niepewności
    refined_text = re.sub(r'<.*?>|~', '', refined_text).strip()

    return refined_text


def run_full_training():
    """ Uruchamia bilingwalny proces treningu ByT5. Model uczy się jednocześnie poprawiać błędy polskie i angielskie. """
    seed_everything(SEED)
    print(f"[{now()}] START: Trening bilingwalny (PL + EN). Urządzenie: {DEVICE}")

    # Inicjalizacja bazowa, aby uniknąć błędów 'referenced before assignment'
    real_pairs_pl = []
    real_pairs_en = []

    num_en = len(real_pairs_en)
    num_pl = len(real_pairs_pl)

    # Jeśli polskich jest mniej, powielamy je
    if num_pl < num_en:
        multiplier = num_en // num_pl
        real_pairs_pl = real_pairs_pl * multiplier
        print(f"[{now()}] Wykonano oversampling danych PL (x{multiplier}).")

    ckpt = torch.load(CRNN_CHECKPOINT, map_location=DEVICE)
    chars = ckpt.get("char_list", [])

    # Ładowanie danych z macierzy pomyłek (dla generatora szumu)
    if os.path.exists(CACHE_MATRIX):
        confusion_matrix = np.load(CACHE_MATRIX)
        print(f"[{now()}] Załadowano macierz pomyłek z cache.")
    else:
        confusion_matrix = np.zeros((len(chars), len(chars)))
        print(f"[{now()}] Warning: Brak macierzy pomyłek. Używam zerowej.")

    # Błędy polskie
    PL_JSON_PATH = os.path.join(BASE_DIR, "transformer_train_pl_pages.json")
    if os.path.exists(PL_JSON_PATH):
        with open(PL_JSON_PATH, 'r', encoding='utf-8') as f:
            real_pairs_pl = json.load(f)
        print(f"[{now()}] Wczytano {len(real_pairs_pl)} polskich par.")

    # Angielskie
    if os.path.exists(CACHE_PAIRS):
        with open(CACHE_PAIRS, 'r', encoding='utf-8') as f:
            real_pairs_en = json.load(f)
        print(f"[{now()}] Wczytano {len(real_pairs_en)} angielskich par z cache.")
    else:
        # Generowanie EN, jeśli brak cache
        from Models.ResNetCRNNWordRecognition import ResNetCRNN
        print(f"[{now()}] Generowanie danych EN z IAM.")
        model_crnn = ResNetCRNN(len(chars) + 1).to(DEVICE)
        model_crnn.load_state_dict(ckpt["model_state"])
        model_caps = CapsNet(len(chars), context_dim=1024).to(DEVICE)
        model_caps.load_state_dict(torch.load(CAPS_CHECKPOINT, map_location=DEVICE)["model_state"])
        inference_pipe = CRNNInferencePipeline(model_crnn, chars, DEVICE)
        inference_pipe.refiner = CascadeRefinementNetwork(model_caps, chars, inference_pipe)

        forms_lines = parse_iam_forms_to_lines(SENTENCES_TXT)
        real_pairs_en, _ = generate_contextual_noise_data(inference_pipe, forms_lines, chars)

        with open(CACHE_PAIRS, 'w', encoding='utf-8') as f:
            json.dump(real_pairs_en, f)

        del model_crnn, model_caps, inference_pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Łączenie zbiorów <pl> i <en>
    real_context_pairs = []
    for p in real_pairs_pl:
        real_context_pairs.append({
            "context": f"<pl> {p.get('context', '')}",
            "input": p['input'],
            "target": p['target']
        })
    for p in real_pairs_en:
        real_context_pairs.append({
            "context": f"<en> {p.get('context', '')}",
            "input": p['input'],
            "target": p['target']
        })
    random.shuffle(real_context_pairs)

    # Inicjalizacja
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_NAME)
    LANGUAGE_TOKENS = ['<pl>', '<en>']
    SPECIAL_TOKENS = [LC_TOKEN, UNC_START, UNC_END] + LANGUAGE_TOKENS
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    model_byt5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model_byt5.resize_token_embeddings(len(tokenizer))
    model_byt5.to(DEVICE)

    # Ładowanie wikipedii
    print(f"[{now()}] Pobieranie bilingwalnej Wikipedii.")
    wiki_pl = load_dataset("wikimedia/wikipedia", "20231101.pl", split="train", streaming=True)
    wiki_en = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    wiki_combined_pairs = []
    pl_iter, en_iter = iter(wiki_pl), iter(wiki_en)

    # Tworzymy pary syntetyczne w proporcji 1:1
    for _ in range(len(real_context_pairs) * 5):
        try:
            # Polskie
            p1, p2 = next(pl_iter)['text'][:200], next(pl_iter)['text'][:200]
            wiki_combined_pairs.append((f"<pl> {p1}", f"<pl> {p2}"))
            # Angielskie
            e1, e2 = next(en_iter)['text'][:200], next(en_iter)['text'][:200]
            wiki_combined_pairs.append((f"<en> {e1}", f"<en> {e2}"))
        except StopIteration:
            break

    # Tworzenie datasetów
    train_wiki, val_wiki = random_split(wiki_combined_pairs, [0.95, 0.05])
    train_real, val_real = random_split(real_context_pairs, [0.95, 0.05])

    error_gen = HybridVisualErrorGenerator(confusion_matrix, chars)

    train_ds = ContextualCurriculumDataset(list(train_wiki), list(train_real), error_gen, tokenizer)
    val_ds = ContextualCurriculumDataset(list(val_wiki), list(val_real), error_gen, tokenizer)

    # Trening
    initial_state_dict = copy.deepcopy(model_byt5.state_dict())

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_byt5, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=1,  # Mały vRAM, ale nadrabiamy akumulacją
        gradient_accumulation_steps=32,  # Efektywny batch size = 32
        learning_rate=1e-2,  # Wyższy LR dla SGD
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=True, # dla lepszej stabilności Transformera
        # Wybrano SGD, aby wymusić lepszą generalizację reguł języka polskiego. 
        # W przeciwieństwie do Adama, SGD rzadziej 'wykuwa słownik na blachę' (overfitting), 
        # co jest kluczowe, gdy model ma poprawiać nieznane wcześniej rękopisy. 
        # Jest również matematycznie najlżejszy dla zasobów VRAM.
        optim="sgd",
        # Optymalizacja dynamiki uczenia przez Momentum i Nesterova.
        # momentum=0.9: Nadaje 'pęd' aktualizacji wag, co pozwala uniknąć lokalnych dołków błędu.
        # nesterov=True: Algorytm 'patrzy w przód', co przyspiesza zbieżność i stabilizuje naukę.
        optim_args="momentum=0.9,nesterov=True",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",  # Wyłączone, by nie zaśmiecać logów
        load_best_model_at_end=True,
        metric_for_best_model="cer",  # Transformer ma bić rekordy w CER - działa, jak słownik
        greater_is_better=False  # Mniejszy CER = lepszy model
    )

    trainer = Seq2SeqTrainer(
        model=model_byt5,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            # Stochastyczny pruning: usuwa najsłabsze wagi, zapobiegając przeuczeniu i kompresując model
            CurriculumCallback(train_ds), StochasticPruningCallback(initial_state_dict, base_amount=0.1),
            EarlyStoppingCallback(early_stopping_patience=2)  # Stop po 2 epokach bez poprawy
        ]
    )

    print(f"[{time.strftime('%H:%M:%S')}] Uruchamianie treningu.")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[{time.strftime('%H:%M:%S')}] Trening zakończony. Model zapisany w {OUTPUT_DIR}")


def run_inference_test():
    """ Uruchamia test inferencji na wytrenowanym modelu Refinera. """
    print(f"[{time.strftime('%H:%M:%S')}] Tryb TESTU INFERENCJI (Walidacja końcowa).")

    # Ładujemy model z folderu wyjściowego
    trained_model_path = OUTPUT_DIR

    if not os.path.exists(trained_model_path):
        print(f"[{now()}] Warning: Nie znaleziono wytrenowanego modelu w {OUTPUT_DIR}. Używam bazy.")
        trained_model_path = MODEL_NAME

    # Ładowanie tokenizatora (już powinien mieć zapisane tokeny w OUTPUT_DIR)
    tokenizer = ByT5Tokenizer.from_pretrained(trained_model_path)

    # Jeśli jednak ładujemy bazowy, musimy dodać tokeny i zmienić rozmiar modelu
    if trained_model_path == MODEL_NAME:
        tokenizer.add_special_tokens({'additional_special_tokens': [LC_TOKEN, UNC_START, UNC_END]})

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(DEVICE)

    # Zawsze synchronizuj rozmiar słownika z modelem
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Sekcja Testowa
    tests = [
        {
            "lang": "pl",
            "ctx": "Zgubiłem wczoraj klucze do mieszkania.",
            "noisy": "Musz<unc>e</unc> t<unc>e</unc>raz wymi<unc>c</unc>nić zamek w drzw1ach."
        },
        {
            "lang": "en",
            "ctx": "The handwritten manuscript was found in the old library.",
            "noisy": "It conta1n<unc>c</unc>d s<unc>c</unc>veral unk<unc>rn</unc>own syrnbols."
        }
    ]

    for t in tests:
        print(f"\nTest dla języka {t['lang'].upper()}")

        # Wywołanie funkcji głównej
        poprawka = refine_ocr_text(t['noisy'], t['ctx'], model, tokenizer, DEVICE, t['lang'])

        print(f"Kontekst: {t['ctx']}")
        print(f"Wejście (Vision + CapsNet): {t['noisy']}")
        print(f"Wynik końcowy (Transformer): {poprawka}")

if __name__ == "__main__":
    run_full_training()