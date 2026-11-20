import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:///app/output_data/mlruns"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Transformer_Semantic_Refiner"

import copy
import difflib
import gc
import json
import random
import re
import time
from pathlib import Path

import Levenshtein
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as func
from torch.nn.utils import prune
from torch.utils.data import random_split

from captum.attr import IntegratedGradients
from datasets import load_dataset
from evaluate import load
from spellchecker import SpellChecker
from tqdm import tqdm
from functools import partial


from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback, 
    T5ForConditionalGeneration, 
    ByT5Tokenizer, 
    DataCollatorForSeq2Seq, 
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# Zależności wewnątrzprojektowe
BASE_DIR = "/app/data"
CHECKPOINT_BASE = "/app/output_data/checkpoints"
MATRIX_DIR = Path(CHECKPOINT_BASE) / "hcr"

try:
    from App.CRNNCNTRHCR import CRNNInferencePipeline, PersistentAuthorMemory
except ImportError:
    pass  # Zabezpieczenie, jeśli te klasy zostały przeniesione gdzie indziej

from Models.DeepCapsNetCharRecognition import CapsNet
from Preprocessing.OpticalLayoutRecognition import PageToLineSegmentor, LineToWordSegmentor
from Preprocessing.Preprocessing import Preprocessing

# Inicjalizacja głównych ścieżek
VISUAL_DEBUG_DIR = MATRIX_DIR / "visual_debug_output"
VISUAL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

IAM_ROOT = os.path.join(BASE_DIR, "iam_words")
XML_DIR = os.path.join(IAM_ROOT, "xml")
FORMS_DIR = os.path.join(IAM_ROOT, "forms")
SENTENCES_TXT = os.path.join(IAM_ROOT, "sentences.txt")

CRNN_CHECKPOINT = os.path.join(CHECKPOINT_BASE, "hwr", "best_cer_model.pth")
CAPS_CHECKPOINT = os.path.join(CHECKPOINT_BASE, "hcr", "CharLevelCapsNet.pth")
YOLO_WEIGHTS = "best_word_det.pt"
TARGET_LANGUAGE = "en"

CACHE_DIR = os.path.join(CHECKPOINT_BASE, "refiner_cache")
CACHE_PAIRS = os.path.join(CACHE_DIR, "real_context_pairs.json")
CACHE_MATRIX = os.path.join(CACHE_DIR, "confusion_matrix.npy")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "google/byt5-base"
OUTPUT_DIR = os.path.join(CHECKPOINT_BASE, "transformer", "byt5_htr_refiner_final")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LC_TOKEN = "~"
UNC_START = "<unc>"
UNC_END = "</unc>"
SEED = 42

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

def now():
    return time.strftime('%H:%M:%S')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


try:
    from run_all import epoch_progress_bar
except ImportError:
    def epoch_progress_bar(iterable, total_epochs: int, stage_name: str, initial: int = 0):
        return tqdm(
            iterable,
            total=total_epochs,
            initial=initial,
            desc=f"{stage_name} | Epoki",
            unit="ep",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            dynamic_ncols=True,
        )


class EpochProgressCallback(TrainerCallback):
    """Aktualizuje pasek postępu epok dla Seq2SeqTrainer z zachowaniem logów w konsoli."""
    def __init__(self, stage_name: str = "ByT5 Refiner"):
        self.stage_name = stage_name
        self.bar = None
        self.total_epochs = 1
        self.last_logged_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_epochs = max(1, int(round(args.num_train_epochs)))
        initial_ep = min(self.total_epochs, max(0, int(round(state.epoch or 0.0))))
        self.last_logged_epoch = initial_ep
        self.bar = epoch_progress_bar(
            range(initial_ep, self.total_epochs),
            total_epochs=self.total_epochs,
            stage_name=self.stage_name,
            initial=initial_ep,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.bar is None:
            return
        current_ep = min(self.total_epochs, int(state.epoch or 0) + 1)
        self.bar.set_description_str(f"{self.stage_name} | Epoka {current_ep}/{self.total_epochs}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.bar is None or not logs:
            return
        postfix = {}
        if "loss" in logs:
            postfix["loss"] = f"{logs['loss']:.4f}"
        if "eval_cer" in logs:
            postfix["cer"] = f"{logs['eval_cer']:.4f}"
        elif "eval_loss" in logs:
            postfix["val"] = f"{logs['eval_loss']:.4f}"
        if postfix:
            self.bar.set_postfix(**postfix)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.bar is None:
            return
        completed_ep = min(self.total_epochs, max(self.last_logged_epoch + 1, int(round(state.epoch or 0.0))))
        steps_to_advance = max(0, completed_ep - self.last_logged_epoch)
        if steps_to_advance > 0:
            self.bar.update(steps_to_advance)
            self.last_logged_epoch = completed_ep

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            remaining = self.total_epochs - self.last_logged_epoch
            if remaining > 0:
                self.bar.update(remaining)
            self.bar.close()
            self.bar = None


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
    """ Generuje pary 'poprzednia z 30% szansy na błędy', 'obecna, jeszcze do poprawy' symulując sliding window. """
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

                # Normalizacja dla CRNN
                t_c = (torch.from_numpy(crop_bin_norm).float().unsqueeze(0) / 255.0 - 0.8491) / 0.2259

                crop_bin_inv = cv.bitwise_not(crop_bin_norm)
                caps_ready_img = Preprocessing.standardize_ink_thickness(crop_bin_inv, target_thickness=3)
                
                # Normalizacja eMNIST dla CapsNetu
                t_k = (torch.from_numpy(caps_ready_img).float().unsqueeze(0) / 255.0 - 0.1307) / 0.3081

                tensors_crnn.append(t_c)
                tensors_caps.append(t_k)

            if not tensors_crnn: continue

            try:
                # Wywołanie z obydwoma listami tensorów
                res = pipeline.predict_batch(tensors_crnn, tensors_caps)
                curr_noisy_line = " ".join(
                    str(r["final_result"]) for r in res if isinstance(r, dict) and "final_result" in r
                )

                # Szukamy GT dla tej linii w forms_dict
                if l_idx < len(lines):
                    target_clean = lines[l_idx]

                    matcher = difflib.SequenceMatcher(None, target_clean, curr_noisy_line)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == "replace":
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

            except RuntimeError as e:
                print(f"[{now()}] Błąd inferencji modeli (PyTorch/CUDA) w linii {l_idx} ({f_id}): {e}")
                continue
            except (KeyError, IndexError) as e:
                print(f"[{now()}] Błąd struktury wyników lub indeksowania w linii {l_idx} ({f_id}): {e}")
                continue
            except (ValueError, TypeError) as e:
                print(f"[{now()}] Nieprawidłowy typ lub wartość danych w linii {l_idx} ({f_id}): {e}")
                continue
            except Exception as e:  # noqa: BLE001 - awaria pojedynczej linii nie przerywa przetwarzania formularza
                print(f"[{now()}] Nieoczekiwany błąd predykcji w linii {l_idx} ({f_id}): {e}")
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

        # Bezpieczna precyzja dla ByT5 (float16 powoduje przepełnienie i NaN w generacji)
        model_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

        # Ładowanie wytrenowanego modelu
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"[{time.strftime('%H:%M:%S')}] Ładowanie adaptera LoRA dla ByT5.")
            # Ładujemy czystą bazę z wyłączonym wiązaniem wag embeddingów
            base_model = T5ForConditionalGeneration.from_pretrained(
                "google/byt5-base",
                torch_dtype=model_dtype,
                tie_word_embeddings=False,
            )
            # Synchronizacja rozmiaru embeddingów przed nałożeniem adaptera (uwzględnia dodane tagi <pl>, <en>, <unc>)
            base_model.resize_token_embeddings(len(self.tokenizer))

            # Nakładamy mały adapter użytkownika
            self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        else:
            # Fallback do pełnego modelu
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                tie_word_embeddings=False,
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

    def refine(self, noisy_text, prefix_context="", context_text=None):
        """ Główna metoda korekty semantycznej z mechanizmem Cache. """
        raw_ctx = context_text if context_text is not None else prefix_context
        resolved_context = raw_ctx if raw_ctx is not None else ""

        if not noisy_text.strip():
            return noisy_text

        cache_key = (noisy_text, resolved_context)
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]

        # Wymuszenie dodania tagu językowego, jeśli go nie ma
        lang_tag = f"<{self.language}> "
        if not resolved_context.startswith(lang_tag.strip()):
            full_context = lang_tag + resolved_context if resolved_context else lang_tag.strip()
        else:
            full_context = resolved_context

        # Przygotowanie wejścia (teraz idealnie zgodne z treningiem)
        if self.language == 'pl':
            full_prompt = f"kontekst: {full_context} popraw ocr: {noisy_text}"
        else:
            full_prompt = f"context: {full_context} fix ocr: {noisy_text}"

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
                length_penalty=1.2,
                repetition_penalty=1.5,
                early_stopping=True
            )

        refined_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_text = re.sub(r'<.*?>|~', '', refined_text).strip()

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
        safe_context = context_text or ""
        lang_tag = f"<{self.language}> "
        if not safe_context.startswith(lang_tag.strip()):
            full_context = lang_tag + safe_context if safe_context else lang_tag.strip()
        else:
            full_context = safe_context

        if self.language == 'pl':
            full_prompt = f"kontekst: {full_context} popraw ocr: {text}"
        else:
            full_prompt = f"context: {full_context} fix ocr: {text}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        labels = self.tokenizer(text_target=text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs, labels=labels)
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

        # Fallback używany wyłącznie wtedy, gdy plik macierzy pomyłek jeszcze nie istnieje
        self.fallback_risky_chars = {"a", "c", "e", "l", "n", "o", "s", "z", "x"}
        self.punctuation_marks = {".", ",", "!", "?", ":", ";", "-", "'", '"', "(", ")"}

        # Cache dla macierzy pomyłek: (high_risk_fp_indices, confusing_fn_indices)
        self._cached_risk_sets = None
        self._last_matrix_path = None
        self._last_matrix_mtime = None

    def _get_confusion_risk_sets(self, path_str: str) -> tuple[set[int], set[int]]:
        """Wyznacza z macierzy pomyłek dwa zbiory indeksów (w przestrzeni char_list):
        1. high_risk (błędy kolumnowe / FP): znaki, które CRNN często podstawia zamiast innych (np. 'a' za 'ą').
        2. medium_risk (błędy wierszowe / FN): znaki, które CRNN najczęściej gubi lub myli.
        """
        if not path_str:
            return set(), set()

        path = Path(path_str)

        try:
            if not path.is_file():
                return set(), set()

            current_mtime = path.stat().st_mtime
            if (
                path == self._last_matrix_path
                and current_mtime == self._last_matrix_mtime
                and self._cached_risk_sets is not None
            ):
                return self._cached_risk_sets

            cm_data = np.load(path, allow_pickle=False)

            if hasattr(cm_data, "close"):
                cm_data.close()
                tqdm.write(f"[{now()}] Warning: Plik {path.name} jest archiwum .npz zamiast macierzy .npy.")
                return set(), set()

            if (
                not isinstance(cm_data, np.ndarray)
                or cm_data.ndim != 2
                or cm_data.size == 0
                or cm_data.shape[0] != cm_data.shape[1]
            ):
                tqdm.write(
                    f"[{now()}] Warning: Macierz w {path.name} ma nieprawidłowy kształt "
                    f"(shape={getattr(cm_data, 'shape', 'N/A')})."
                )
                return set(), set()

            cm = cm_data.astype(np.float64, copy=False)

            # Jeśli macierz CRNN zawiera klasę [blank] na indeksie 0, odcinamy ją, aby wyrównać indeksy do char_list
            if cm.shape[0] == len(self.char_list) + 1:
                cm = cm[1:, 1:]

            diag = np.diag(cm)
            # Błędy kolumnowe (FP): ile razy model przewidział dany znak błędnie zamiast innego
            fp_scores = np.nan_to_num(cm.sum(axis=0) - diag, nan=0.0, posinf=0.0, neginf=0.0)

            # Błędy wierszowe (FN): ile razy prawdziwy znak został pomylony z innym
            fn_scores = np.nan_to_num(cm.sum(axis=1) - diag, nan=0.0, posinf=0.0, neginf=0.0)

            # Wyznaczamy indeksy znaków, które są w najwyższym ryzyku błędów FP i FN
            fp_nonzero = np.flatnonzero(fp_scores > 0)
            high_risk = (
                set(fp_nonzero[np.argsort(fp_scores[fp_nonzero])[-10:]].tolist())
                if fp_nonzero.size > 0
                else set()
            )

            # Poza tym, wyznaczamy indeksy znaków, które są w najwyższym ryzyku błędów FN (gubienie lub mylenie)
            fn_nonzero = np.flatnonzero(fn_scores > 0)
            medium_risk = (
                # 15 najczęściej gubionych znaków, znowu liczba wyznaczona eksperymentalnie, bo nie było danych do optymalizacji
                set(fn_nonzero[np.argsort(fn_scores[fn_nonzero])[-15:]].tolist())
                if fn_nonzero.size > 0
                else set()
            )

            # W ten sposób mamy kompletne zbiory ryzykownych znaków do dalszej analizy w selektorze regionów niepewnych
            self._cached_risk_sets = (high_risk, medium_risk)
            self._last_matrix_path = path
            self._last_matrix_mtime = current_mtime

            return self._cached_risk_sets

        # Obsługa błędów przy wczytywaniu macierzy pomyłek, po return, bo nie chcemy przerywać działania programu, tylko logować problem
        except (OSError, ValueError, TypeError, EOFError) as e:
            tqdm.write(f"[{now()}] Błąd krytyczny macierzy pomyłek ({path.name}): {e}")
            return set(), set()
        
        
    def select_dubious_groups(
        self,
        crnn_probs: torch.Tensor,
        width: int,
        confusion_matrix_path: str = "",
        intensive: bool = False,
        merge: bool = True,
    ) -> list:
        """ Wybiera grupy kroków czasowych, które są niepewne i wymagają weryfikacji przez CapsNet.
            Zwraca listę słowników z informacjami o wycinkach (x1, x2) i prawdopodobieństwie. """
        steps = crnn_probs.shape[0]
        if steps == 0:
            return []

        # Sprawdzamy, czy tensor prawdopodobieństw jest już w formacie softmax, żeby nie robić ponownego obliczania softmaxu
        is_already_probs = (
            crnn_probs.min() >= 0.0
            and crnn_probs.max() <= 1.0
            and torch.allclose(
                crnn_probs.sum(dim=-1),
                torch.ones(steps, device=crnn_probs.device, dtype=crnn_probs.dtype),
                atol=1e-2,
            )
        )

        # Obliczamy prawdopodobieństwa i marginesy dla każdego kroku czasowego
        probs = crnn_probs if is_already_probs else torch.softmax(crnn_probs, dim=-1)
        top_p, top_idx = torch.topk(probs, min(2, probs.shape[-1]), dim=-1)
        top1_p, top1_idx = top_p[:, 0], top_idx[:, 0]
        margin = (top1_p - top_p[:, 1]) if top_p.shape[-1] > 1 else top1_p

        # Wyznaczamy okno czasowe dla CapsNetu, jako proporcję szerokości obrazu do liczby kroków czasowych, z dodatkowym marginesem
        stride = width / steps
        window_factor = 3.5 if intensive else 2.8
        half_window = int((stride * window_factor) / 2)

        # Pobieramy indeksy znaków, które są często mylone z innymi
        high_risk_indices, medium_risk_indices = self._get_confusion_risk_sets(confusion_matrix_path)
        risky_chars = (
            {c.lower() for c in self.split_risky_chars}
            if self.split_risky_chars
            else self.fallback_risky_chars
        )
        has_blank_offset = probs.shape[-1] == len(self.char_list) + 1

        # Tworzymy listę kandydatów do weryfikacji przez CapsNet
        candidates = []
        base_threshold = 0.60 if intensive else 0.50
        substitution_threshold = 0.75
        polish_risk_threshold = 0.85
        margin_min = 0.18

        capsnet_targets = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

        # Dla każdego kroku czasowego sprawdzamy, czy spełnia kryteria niepewności, i ewentualnie dodajemy go do listy kandydatów
        for t in range(steps):
            char_idx = int(top1_idx[t].item())
            
            # Pomijamy pustą klatkę CTC [blank]
            if has_blank_offset and char_idx == 0:
                continue

            list_idx = char_idx - 1 if has_blank_offset else char_idx
            char_str = self.char_list[list_idx] if 0 <= list_idx < len(self.char_list) else ""

            # Pomijamy znaki interpunkcyjne i znaki, które nie są obsługiwane przez CapsNet
            if char_str in self.punctuation_marks or char_str not in capsnet_targets:
                continue

            current_limit = base_threshold
            if char_str.lower() in risky_chars or list_idx in high_risk_indices:
                current_limit = polish_risk_threshold
            elif list_idx in medium_risk_indices:
                current_limit = substitution_threshold

            if top1_p[t] < current_limit or margin[t] < margin_min:
                center_x = int((t + 0.5) * stride)
                candidates.append({
                    'timestep': t,
                    'x1': max(0, center_x - half_window),
                    'x2': min(width, center_x + half_window),
                    'prob': top1_p[t].item(),
                    'char': char_str
                })

        if not candidates:
            return []

        # Grupujemy sąsiadujące kroki czasowe w spójne wycinki dla CapsNetu
        final_groups = []
        current_group = [candidates[0]]

        """ Wyznacza niepewne fragmenty predykcji CRNN i grupuje je w przedziały do weryfikacji przez CapsNet.
            Metoda analizuje rozkład prawdopodobieństw w kolejnych krokach czasowych CTC:
            1. Pomija puste klatki, znaki interpunkcyjne oraz znaki spoza docelowego alfabetu.
            2. Kwalifikuje klatki jako niepewne, gdy pewność top-1 spada poniżej progu adaptacyjnego
                (podwyższanego dla polskich znaków wysokiego ryzyka oraz klas z największą liczbą błędów
                w macierzy pomyłek) lub gdy różnica między top-1 a top-2 jest mniejsza niż `0.18`, wyznaczone
                na sztywno, bo nie ma możliwości użycia wartości dynamicznych, ze względu na zbyt mały rozmiar danych.
            3. Scala sąsiednie niepewne klatki odległe o maksymalnie 3 kroki czasowe w pojedyncze grupy,
                aby pojedyncza, szersza litera była analizowana przez CapsNet tylko raz. Zakładamy tutaj,
                że jeśli kilka kroków czasowych jest niepewnych, to prawdopodobnie dotyczą tego samego znaku."""
        for i in range(1, len(candidates)):
            if candidates[i]['timestep'] - current_group[-1]['timestep'] <= 3:
                current_group.append(candidates[i])
            else:
                final_groups.append(self._merge_group(current_group) if merge else current_group)
                current_group = [candidates[i]]

        final_groups.append(self._merge_group(current_group) if merge else current_group)
        return final_groups

    @staticmethod
    def _merge_group(group: list[dict]) -> dict:
        """ Łączy kilka kroków czasowych w jeden spójny wycinek dla CapsNetu. """
        return {
            'timestep': group[len(group) // 2]['timestep'],
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
    """ Pełny proces kaskadowy: CRNN -> CapsNet -> Transformer z Trybem Paniki i Early Drop. """
    def __init__(self, crnn_pipe, transformer_refiner):
        self.vision = crnn_pipe  # CRNNInferencePipeline połączony z CascadeRefinementNetwork
        self.semantic = transformer_refiner  # TransformerRefiner
        
        # Statystyki z normalizacji IAM do odwracania transformacji przy filtrach
        self.iam_mean = 0.8491
        self.iam_std = 0.2259

    def get_decoded_text(self, probs: torch.Tensor) -> list[str]:
        """ Dynamiczne pobranie metody omija statyczną analizę lintera. """
        decoder = getattr(self.vision, "_decode_ctc")
        return decoder(probs)

    def _calculate_crnn_confidence(self, log_probs):
        """ Oblicza pewność słowa opartą o Power Mean (silniej karze klatki niepewne). """
        probs = torch.softmax(log_probs, dim=-1).squeeze(1)
        max_probs, top_idx = torch.max(probs, dim=-1)
        valid_probs = max_probs[top_idx != 0] # Ignorujemy blanki
        
        if len(valid_probs) == 0: return 0.0
        return float(torch.sum(valid_probs ** 2) / torch.sum(valid_probs))

    def _run_crnn_pass(self, img_tensor):
        """ Uruchamia pojedynczy przebieg CRNN i zwraca komplet danych. """
        with torch.no_grad():
            raw_logits, context_maps = self.vision.model(img_tensor.to(DEVICE), return_context=True)
            crnn_text = self.vision.encoder.decode_greedy(raw_logits)[0][0]
            log_probs = raw_logits.log_softmax(2)
            conf = self._calculate_crnn_confidence(log_probs)
        return crnn_text, log_probs, context_maps, conf

    def _apply_clahe(self, img_tensor):
        """ Adaptacyjne wyrównywanie histogramu - ratuje kontrast. """
        img_np = img_tensor.squeeze().cpu().numpy()
        img_unnorm = np.clip((img_np * self.iam_std) + self.iam_mean, 0, 1)
        img_uint8 = (img_unnorm * 255).astype(np.uint8)
        
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        img_clahe = clahe.apply(img_uint8)
        
        img_norm = ((img_clahe.astype(np.float32) / 255.0) - self.iam_mean) / self.iam_std
        return torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _apply_morphology(self, img_tensor):
        """ Pogrubienie atramentu (dylatacja białych pikseli na czarnym tle). """
        img_np = img_tensor.squeeze().cpu().numpy()
        img_unnorm = np.clip((img_np * self.iam_std) + self.iam_mean, 0, 1)
        img_uint8 = (img_unnorm * 255).astype(np.uint8)
        
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
        img_morph = cv.dilate(img_uint8, kernel, iterations=1)
        
        img_norm = ((img_morph.astype(np.float32) / 255.0) - self.iam_mean) / self.iam_std
        return torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def _run_tta_pass(self, img_tensor):
        """ Test-Time Augmentation - uśrednianie kilku predykcji. """
        self.vision.model.eval()
        predictions = []
        
        with torch.no_grad():
            _, base_context = self.vision.model(img_tensor.to(DEVICE), return_context=True)
            
            for _ in range(3):
                aug_image = img_tensor.clone().to(DEVICE)
                brightness_shift = torch.randn(1).item() * 0.1 * (1.0 / self.iam_std)
                aug_image = aug_image + brightness_shift

                logits, _ = self.vision.model(aug_image, return_context=True)
                predictions.append(torch.softmax(logits.float(), dim=-1))

        avg_probs = torch.stack(predictions).mean(dim=0)
        avg_log_probs = torch.log(avg_probs + 1e-8)
        
        crnn_text = self.vision.encoder.decode_greedy(avg_log_probs)[0][0]
        conf = self._calculate_crnn_confidence(avg_log_probs)
        
        return crnn_text, avg_log_probs, base_context, conf

    def _decode_with_char_alignment(self, log_probs_2d: torch.Tensor, step_unc_mask: list[int]):
        """ Dekoduje CTC greedy i rzutuje maskę niepewności klatek czasowych [T] na konkretne znaki słowa [L]. """
        preds = torch.argmax(log_probs_2d, dim=-1).cpu().tolist()
        idx_to_char = getattr(
            self.vision.encoder, "num_to_char",
            getattr(self.vision.encoder, "idx_to_char", {})
        )

        chars = []
        char_unc_mask = []
        prev_idx = 0

        for t, idx in enumerate(preds):
            if idx != 0 and idx != prev_idx:
                ch = idx_to_char.get(idx, "")
                if ch:
                    chars.append(ch)
                    is_unc = 1 if (t < len(step_unc_mask) and step_unc_mask[t] > 0) else 0
                    char_unc_mask.append(is_unc)
            elif idx == prev_idx and idx != 0 and char_unc_mask:
                if t < len(step_unc_mask) and step_unc_mask[t] > 0:
                    char_unc_mask[-1] = 1
            prev_idx = idx

        return "".join(chars), char_unc_mask

    def _measure_ink_ratio(self, img_tensor):
        """ Zwraca udział pikseli atramentu [0.0 - 1.0] niezależnie od skali tensora. """
        arr = img_tensor.detach().cpu().float().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            arr = arr.mean(axis=0)

        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max - arr_min < 1e-6:
            return 0.0

        gray = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Zakładamy, że tło dominuje na brzegach obrazu
        border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        bg_is_white = np.median(border) > 127
        ink_mask = (binary == 0) if bg_is_white else (binary == 255)
        return float(np.mean(ink_mask))


    def _apply_ink_aware_morphology(self, img_tensor, ink_ratio):
        """ Dobiera kierunek morfologii do gęstości atramentu: rozkleja nadmiar lub pogrubia braki. """
        device = img_tensor.device
        dtype = img_tensor.dtype
        arr = img_tensor.detach().cpu().float().numpy()
        squeeze_batch = False
        if arr.ndim == 4:
            arr = arr[0]
            squeeze_batch = True

        c, h, w = arr.shape
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max - arr_min < 1e-6:
            return img_tensor

        gray = ((arr[0] - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_is_white = np.median(border) > 127

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

        if ink_ratio > 0.34:
            # Za dużo atramentu (rozlany tusz / sklejone znaki) -> ścieńczenie atramentu
            processed = cv.dilate(gray, kernel, iterations=1) if bg_is_white else cv.erode(gray, kernel, iterations=1)
        elif ink_ratio < 0.10:
            # Za mało atramentu (wyblakły tusz) -> pogrubienie atramentu
            processed = cv.erode(gray, kernel, iterations=1) if bg_is_white else cv.dilate(gray, kernel, iterations=1)
        else:
            # Normalna gęstość -> delikatne otwarcie usuwające drobny szum bez sklejania
            processed = cv.morphologyEx(gray, cv.MORPH_CLOSE if bg_is_white else cv.MORPH_OPEN, kernel)

        norm = processed.astype(np.float32) / 255.0 * (arr_max - arr_min) + arr_min
        out = np.repeat(norm[None, :, :], c, axis=0)
        if squeeze_batch:
            out = out[None, :, :, :]
        return torch.from_numpy(out).to(device=device, dtype=dtype)


    def _compute_dynamic_thresholds(self, log_probs, ink_ratio):
        """ Wyznacza adaptacyjny próg paniki oraz najniższą pewność pojedynczego znaku. """
        base_panic = getattr(self, "base_panic_threshold", 0.50)
        early_drop = getattr(self, "early_drop_threshold", 0.15)

        probs = torch.softmax(log_probs.squeeze(1), dim=-1)  # [T, C]
        max_probs, pred_ids = probs.max(dim=-1)

        # Pomijamy klatki blank (indeks 0), aby ocenić pewność samych znaków
        non_blank_mask = pred_ids != 0
        if non_blank_mask.any():
            char_probs = max_probs[non_blank_mask]
            min_char_conf = float(char_probs.min().item())
            est_len = int(non_blank_mask.sum().item())
        else:
            min_char_conf = 0.0
            est_len = 0

        # Dłuższe wyrazy mają nieco niższy próg średni, ale ekstremalny ink_ratio podnosi czujność
        len_offset = -0.07 * min(est_len / 12.0, 1.0)
        ink_offset = 0.06 if (ink_ratio > 0.36 or ink_ratio < 0.08) else 0.0

        dynamic_panic = float(np.clip(base_panic + len_offset + ink_offset, 0.38, 0.65))
        return dynamic_panic, early_drop, min_char_conf


    def _apply_semantic_veto(self, caps_text, semantic_proposal, char_unc_mask, caps_conf, semantic_conf, hybrid_conf):
        """ Rozstrzyga konflikt między CapsNet a ByT5 na podstawie budżetu niepewnych znaków, żeby uniknąć halucynacji. """
        proposal = (semantic_proposal or "").strip()

        # Pusta odpowiedź lub brak zmian ze strony Transformera, to akceptacja CapsNet
        if not proposal:
            return caps_text, True
        if proposal == caps_text:
            return caps_text, False

        dist = Levenshtein.distance(caps_text, proposal)
        len_diff = abs(len(caps_text) - len(proposal))
        unc_count = sum(1 for m in char_unc_mask if m > 0)

        # CapsNet jest pewny i nie zgłosił żadnych niepewnych znaków
        if caps_conf > 0.95 and unc_count == 0:
            return caps_text, True

        # Ochrona przed halucynacją długości: jeśli różnica długości jest większa niż liczba niepewnych znaków + 1, odrzucamy propozycję
        max_allowed_len_diff = max(2, unc_count + 1)
        if len_diff > max_allowed_len_diff:
            return caps_text, True

        # ByT5 może zmienić tyle znaków, ile było w <unc> (+1 marginesu tylko przy słabej wizji)
        extra_tolerance = 1 if caps_conf < 0.65 else 0
        allowed_edits = max(1, unc_count) + extra_tolerance
        if dist > allowed_edits and caps_conf >= 0.60:
            return caps_text, True

        # Wybieramy źródło o wyższej pewności, jeśli oba niewystarczająco pewne
        if semantic_conf + 0.25 < caps_conf and caps_conf > 0.70:
            return caps_text, True
        if hybrid_conf < 0.35 and caps_conf >= semantic_conf:
            return caps_text, True

        return proposal, False


    def process_word(self, img_crnn, img_caps, prev_context=""):
        self.vision.model.eval()

        # Standardowa predykcja CRNN + analiza gęstości atramentu i progów adaptacyjnych
        crnn_text, log_probs, context_maps, crnn_conf = self._run_crnn_pass(img_crnn)
        ink_ratio = self._measure_ink_ratio(img_crnn)
        panic_threshold, early_drop_threshold, min_char_conf = self._compute_dynamic_thresholds(log_probs, ink_ratio)

        needs_rescue = (crnn_conf < panic_threshold) or (min_char_conf < 0.32 and len(crnn_text) > 1)

        if needs_rescue:
            img_clahe = self._apply_clahe(img_crnn)
            c_text, c_log_probs, c_ctx, c_conf = self._run_crnn_pass(img_clahe)
            if c_conf > crnn_conf:
                crnn_text, log_probs, context_maps, crnn_conf = c_text, c_log_probs, c_ctx, c_conf
                img_crnn = img_clahe
                ink_ratio = self._measure_ink_ratio(img_crnn)

            if crnn_conf < panic_threshold:
                img_morph = self._apply_ink_aware_morphology(img_crnn, ink_ratio)
                m_text, m_log_probs, m_ctx, m_conf = self._run_crnn_pass(img_morph)
                if m_conf > crnn_conf:
                    crnn_text, log_probs, context_maps, crnn_conf = m_text, m_log_probs, m_ctx, m_conf
                    img_crnn = img_morph

            if crnn_conf < (panic_threshold - 0.05):
                t_text, t_log_probs, t_ctx, t_conf = self._run_tta_pass(img_crnn)
                if t_conf > crnn_conf:
                    crnn_text, log_probs, context_maps, crnn_conf = t_text, t_log_probs, t_ctx, t_conf

        if crnn_conf < early_drop_threshold:
            rejected_label = "[Nierozpoznano]"
            return {
                "crnn_result": crnn_text,
                "caps_result": "",
                "final_result": rejected_label,
                "is_semantic_rejected": True,
                "uncertain_zones": [1] * len(rejected_label),
                "caps_confidence": 0.0,
                "hybrid_confidence": 0.0,
                "dynamic_alpha": 0.0
            }

        # Poprawka CapsNet
        probs_refined, step_unc_mask, caps_conf, debug_data = self.vision.refiner.refine_logits(
            log_probs.squeeze(1),
            img_caps.to(DEVICE),
            context_map=context_maps[0]
        )

        caps_text, char_unc_mask = self._decode_with_char_alignment(probs_refined, step_unc_mask)
        if not caps_text:
            # Obejmujemy całą sekwencję jako niepewną, jeśli CapsNet nie zwrócił żadnego znaku
            caps_text = self.vision.encoder.decode_greedy(probs_refined.unsqueeze(1))[0][0]
            char_unc_mask = [0] * len(caps_text)

        meta_list = debug_data.get('batch_metadata') or (
            debug_data.get('metadata') if isinstance(debug_data.get('metadata'), list) else []
        )

        if (caps_conf < 0.7 or any(c in caps_text for c in "ąćęłńóśźż")) and meta_list:
            try:
                explainer = VisualExplainer(self.vision.refiner.caps_net)

                for i, meta in enumerate(meta_list):
                    t = meta['timestep']
                    char = meta.get('char', '')

                    if char and t < len(step_unc_mask) and step_unc_mask[t] > 0:
                        char_to_idx = getattr(
                            self.vision.encoder, "char_to_num",
                            getattr(self.vision.encoder, "char_to_idx", {})
                        )
                        target_idx = char_to_idx.get(char, 0)

                        crop_tensor = debug_data['tensors'][i]
                        if crop_tensor.dim() == 3:
                            crop_tensor = crop_tensor.unsqueeze(0)
                        crop_tensor = crop_tensor.to(DEVICE)

                        context_vec = debug_data['contexts'][i]
                        if context_vec.dim() == 1:
                            context_vec = context_vec.unsqueeze(0)
                        context_vec = context_vec.to(DEVICE)

                        attr_map = explainer.explain(crop_tensor, context_vec, target_idx)

                        safe_char = char if char.isalnum() else f"ord{ord(char)}"
                        debug_path = os.path.join(VISUAL_DEBUG_DIR, f"xai_{safe_char}_{int(time.time())}_{i}.png")
                        explainer.plot_explanation(crop_tensor, attr_map, char, debug_path)
            except RuntimeError as e:
                print(f"Błąd obliczeń tensorowych lub pamięci CUDA w XAI: {e}")
            except (KeyError, IndexError) as e:
                print(f"Błąd indeksowania lub brakujący klucz metadanych w XAI: {e}")
            except (ValueError, TypeError) as e:
                print(f"Błąd wartości lub nieprawidłowy typ danych w XAI: {e}")
            except OSError as e:
                print(f"Błąd zapisu wykresu XAI na dysku: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"Nieoczekiwany błąd modułu XAI (pomijam wizualizację): {e}")

        # Przygotowanie wejścia dla Transformera (na wyrównanej masce znaków)
        tagged_input = ""
        for char, m in zip(caps_text, char_unc_mask):
            if m > 0:
                tagged_input += f"<unc>{char}</unc>"
            else:
                tagged_input += char

        # Korekta Semantyczna ByT5
        semantic_proposal = self.semantic.refine(tagged_input, prefix_context=prev_context)
        semantic_score = self.semantic.score_text(semantic_proposal, context_text=prev_context)
        semantic_conf = min(max(1.0 + (semantic_score / 20.0), 0.1), 1.0)

        # Dynamiczne wagowanie
        alpha = 0.6
        if len(caps_text) > 7:
            alpha -= 0.15
        elif len(caps_text) <= 3:
            alpha += 0.15

        if caps_conf > 0.92:
            alpha += 0.1
        elif caps_conf < 0.6:
            alpha -= 0.2

        alpha = max(min(alpha, 0.9), 0.2)

        # Finalna fuzja i spójne Veto
        hybrid_conf = (alpha * caps_conf) + ((1.0 - alpha) * semantic_conf)
        final_text, is_rejected = self._apply_semantic_veto(
            caps_text=caps_text,
            semantic_proposal=semantic_proposal,
            char_unc_mask=char_unc_mask,
            caps_conf=caps_conf,
            semantic_conf=semantic_conf,
            hybrid_conf=hybrid_conf
        )

        return {
            "crnn_result": crnn_text,
            "caps_result": tagged_input,
            "final_result": final_text,
            "is_semantic_rejected": is_rejected,
            "uncertain_zones": char_unc_mask,
            "caps_confidence": round(caps_conf, 3),
            "hybrid_confidence": round(float(hybrid_conf), 3),
            "dynamic_alpha": round(alpha, 2)
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

        self.caps_net = capsnet_predictor
        self.encoder = encoder_obj
        self.char_list = char_list
        self.device = next(capsnet_predictor.parameters()).device

        self.char_to_idx = getattr(self.encoder, "char_to_num", getattr(self.encoder, "char_to_idx", {}))
        self.idx_to_char = getattr(self.encoder, "num_to_char", getattr(self.encoder, "idx_to_char", None))

        self.lc_token = "~"
        self.lc_idx = self.char_to_idx.get(self.lc_token, len(char_list))

        if self.idx_to_char is not None:
            self.caps_mapping = self.idx_to_char
        else:
            self.caps_mapping = {0: '[blank]', **{i + 1: c for i, c in enumerate(char_list)}}

        split_risky_chars = self._load_split_risky_chars(matrix_dir)
        self.region_selector = DubiousRegionSelector(char_list, split_risky_chars)

        self.caps_net.eval()

    def _load_split_risky_chars(self, matrix_dir: str | None, error_threshold: float = 0.15) -> list[str]:
        """ Wczytuje ryzykowne znaki lub wyznacza je z confusion_matrix.npy (obsługuje 93x93 i 94x94). """
        if matrix_dir is not None:
            # Domyślnie wczytujemy z pliku JSON, jeśli istnieje, aby uniknąć kosztownej analizy macierzy pomyłek
            for fname in ("split_risky_chars.json", "risky_chars.json"):
                json_path = os.path.join(matrix_dir, fname)
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return list(data["risky_chars"] if isinstance(data, dict) else data)

        # Jeśli nie ma pliku JSON, analizujemy macierz pomyłek, aby znaleźć znaki o wysokim wskaźniku błędu
        if self.matrix_path is not None and os.path.exists(self.matrix_path):
            cm = np.load(self.matrix_path)
            has_blank_row = cm.shape[0] == len(self.char_list) + 1
            risky = []
            # Przechodzimy przez wszystkie znaki i obliczamy wskaźnik błędu dla każdego wiersza macierzy pomyłek
            for i, ch in enumerate(self.char_list):
                cm_idx = i + 1 if has_blank_row else i
                if ch in ("[blank]", self.lc_token) or cm_idx >= cm.shape[0]:
                    continue
                row_sum = float(cm[cm_idx].sum())
                if row_sum > 0:
                    # Obliczamy wskaźnik błędu jako 1 - (prawidłowe przewidywania / suma wiersza)
                    error_rate = 1.0 - (float(cm[cm_idx, cm_idx]) / row_sum)
                    if error_rate >= error_threshold:
                        risky.append(ch)
            return risky

        return []

    def _get_char_threshold(self, char: str, base_threshold: float = 0.85, min_thresh: float = 0.72, max_thresh: float = 0.92) -> float:
        """ Wyznacza i cachuje indywidualny próg interwencji CapsNetu na podstawie rozkładu błędów w confusion_matrix.npy. """
        if not hasattr(self, "_char_thresholds") or self._char_thresholds is None:
            self._char_thresholds = {}
            if getattr(self, "matrix_path", None) and os.path.exists(self.matrix_path):
                try:
                    # Wczytujemy macierz pomyłek i obliczamy wskaźniki błędów dla każdego znaku, aby ustalić dynamiczne progi
                    cm = np.load(self.matrix_path).astype(np.float64)
                    has_blank_row = cm.shape[0] == len(self.char_list) + 1
                    raw_errors = {}

                    for i, ch in enumerate(self.char_list):
                        cm_idx = i + 1 if has_blank_row else i
                        if ch in ("[blank]", self.lc_token) or cm_idx >= cm.shape[0]:
                            continue
                        row_sum = float(cm[cm_idx].sum())
                        if row_sum > 0:
                            # Obliczamy wskaźnik błędu jako 1 - (prawidłowe przewidywania / suma wiersza)
                            raw_errors[ch] = 1.0 - (float(cm[cm_idx, cm_idx]) / row_sum)

                    if raw_errors:
                        """ Punkt nasycenia wyznaczany dynamicznie z rozkładu błędów danego checkpointu CRNN, dajemy tylko 10%
                            najwyższych wartości, aby uniknąć ekstremalnych outlierów. """
                        saturation_err = max(float(np.percentile(list(raw_errors.values()), 90)), 1e-6)
                        for ch, err in raw_errors.items():
                            norm_err = min(max(err / saturation_err, 0.0), 1.0)
                            self._char_thresholds[ch] = max_thresh - norm_err * (max_thresh - min_thresh)
                except (OSError, ValueError):
                    self._char_thresholds = {}

        return self._char_thresholds.get(char, base_threshold)

    def refine_logits(self, crnn_probs: torch.Tensor, image_tensor: torch.Tensor, context_map: torch.Tensor = None, force_intensive: bool = False):
        """Kaskadowa optymalizacja z dynamicznym progiem CapsNetu skalibrowanym pod macierz pomyłek CRNN."""
        # Usuwamy nadmiarowy wymiar wsadu (1, B, C, H, W) -> (B, C, H, W), jeśli występuje
        if image_tensor.dim() == 5:
            image_tensor = image_tensor.squeeze(0)

        width = image_tensor.shape[-1]
        device = crnn_probs.device

        # Wybór podejrzanych kroków czasowych CRNN na podstawie entropii, marginu oraz macierzy pomyłek
        groups = self.region_selector.select_dubious_groups(
            crnn_probs,
            width,
            confusion_matrix_path=self.matrix_path or "",
            intensive=force_intensive,
            merge=False,
        )

        # Maska niepewności dla generatora tagów <unc> (ByT5) oraz pusta struktura diagnostyczna dla UI
        uncertainty_mask = torch.zeros(crnn_probs.size(0), dtype=torch.int32, device=device)
        empty_debug = {
            "batch_metadata": [],
            "metadata": {"uncertain_boxes": [], "interventions": []},
            "tensors": None,
            "contexts": None,
        }

        # Pomocnicza funkcja gwarantująca spójny zwrot log-prawdopodobieństw także przy wczesnym wyjściu
        def _ensure_log_probs(probs_tensor: torch.Tensor) -> torch.Tensor:
            if probs_tensor.min() >= 0.0:
                return torch.log(probs_tensor.clamp(min=1e-8))
            return probs_tensor

        if not groups:
            return _ensure_log_probs(crnn_probs), uncertainty_mask.cpu().tolist(), 1.0, empty_debug

        batch_tensors, batch_contexts, batch_metadata = [], [], []
        zero_context = torch.zeros(1024, device=device)

        for g in groups:
            # Wyznaczamy reprezentatywny krok czasowy (wyznaczający środek) oraz granice poziome wycinka
            mid_cand = g[len(g) // 2]
            t_idx = mid_cand['timestep']
            x1 = max(0, int(min(c['x1'] for c in g)))
            x2 = min(width, int(max(c['x2'] for c in g)))

            # Odrzucamy zbyt wąskie wycinki (szum lub artefakty na krawędziach)
            crop = image_tensor[..., x1:x2]
            if crop.shape[-1] < 8:
                continue

            # Skalowanie wycinka do stałego rozmiaru 64x64 wymaganego przez CapsNet
            if crop.dim() == 3:
                crop = crop.unsqueeze(0)
            crop_resized = func.interpolate(crop, size=(64, 64), mode='bilinear', align_corners=False)

            # Ujednolicenie polaryzacji (jasny atrament na ciemnym tle) oraz normalizacja Z-score
            if crop_resized.min() >= 0.0:
                if crop_resized.max() > 1.0:
                    crop_resized = crop_resized / 255.0
                crop_inv = 1.0 - crop_resized if crop_resized.mean() > 0.5 else crop_resized
                crop_norm = (crop_inv - 0.1307) / 0.3081
            else:
                # Tensor jest już po standaryzacji Z-score (zawiera wartości ujemne)
                crop_norm = crop_resized

            batch_tensors.append(crop_norm)

            # Dopasowanie wymiaru wektora kontekstowego BiLSTM do oczekiwanego przez CapsNet wejścia 1024D
            if context_map is not None:
                raw_ctx = context_map[min(t_idx, context_map.size(0) - 1)].view(-1)
                if raw_ctx.numel() < 1024:
                    ctx = torch.zeros(1024, device=device, dtype=raw_ctx.dtype)
                    ctx[:raw_ctx.numel()] = raw_ctx
                else:
                    ctx = raw_ctx[:1024]
            else:
                ctx = zero_context

            batch_contexts.append(ctx)
            batch_metadata.append({
                'timestep': t_idx,
                'x1': x1,
                'x2': x2,
            })

        if not batch_tensors:
            return _ensure_log_probs(crnn_probs), uncertainty_mask.cpu().tolist(), 1.0, empty_debug

        # Wsadowa inferencja CapsNet dla wszystkich wyciętych kandydatów jednocześnie
        with torch.no_grad():
            input_batch = torch.cat(batch_tensors)
            context_batch = torch.stack(batch_contexts)

            caps_output = self.caps_net(input_batch, word_context=context_batch)
            caps_probs = caps_output["norms"]

        # Pobieramy dwie najbardziej prawdopodobne klasy z CapsNetu do obliczenia pewności i marginu
        top_probs, top_indices = torch.topk(caps_probs, min(2, caps_probs.size(1)), dim=1)
        confidences = []
        interventions = []
        uncertain_boxes = []

        # Konwersja wejścia CRNN do liniowych prawdopodobieństw przed miękką fuzją
        base_probs = torch.softmax(crnn_probs, dim=-1) if crnn_probs.min() < 0 else crnn_probs.clone()
        refined_probs = base_probs.clone()

        # Mapowanie znaków CapsNetu na indeksy CRNN, aby umożliwić fuzję rozkładów prawdopodobieństwa
        char_to_idx = getattr(self, "char_to_idx", getattr(self.encoder, "char_to_num", {}))

        for i, meta in enumerate(batch_metadata):
            t = meta['timestep']
            top_conf = top_probs[i, 0].item()
            margin = (top_probs[i, 0] - top_probs[i, 1]).item() if top_probs.size(1) > 1 else top_conf
            confidences.append(top_conf)

            char_code = top_indices[i, 0].item()
            char = self.caps_mapping.get(char_code, '?')
            meta['char'] = char

            dynamic_thresh = self._get_char_threshold(char)

            if top_conf > dynamic_thresh and margin >= 0.15:
                # CapsNet przekroczył próg pewności, więc dodajemy jego predykcję do rozkładu prawdopodobieństwa CRNN
                idx = char_to_idx.get(char)
                if idx is not None and idx < refined_probs.size(-1):
                    # Waga alpha zachowuje alternatywne hipotezy CRNN dla dekodera CTC i modelu ByT5
                    alpha = min(0.88, max(0.65, top_conf))
                    refined_probs[t, :] *= (1.0 - alpha)
                    refined_probs[t, idx] += alpha
                    refined_probs[t, :] = refined_probs[t, :] / refined_probs[t, :].sum()

                    interventions.append({
                        "timestep": t,
                        "x1": meta["x1"],
                        "x2": meta["x2"],
                        "caps_char": char,
                        "caps_conf": top_conf,
                        "threshold_used": round(dynamic_thresh, 3),
                    })
            else:
                # Region niepewny w CRNN, którego CapsNet nie rozstrzygnął z wymaganą pewnością
                uncertainty_mask[t] = 1
                uncertain_boxes.append({
                    "timestep": t,
                    "x1": meta["x1"],
                    "x2": meta["x2"],
                    "caps_char": char,
                    "caps_conf": top_conf,
                    "margin": margin,
                })

        avg_caps_conf = sum(confidences) / len(confidences) if confidences else 1.0

        debug_data = {
            "batch_metadata": batch_metadata,
            "metadata": {
                "uncertain_boxes": uncertain_boxes,
                "interventions": interventions,
            },
            "tensors": input_batch.detach().cpu(),
            "contexts": context_batch.detach().cpu(),
        }

        # Powrót do skali logarytmicznej wymaganej przez dekoder CTC
        refined_log_probs = torch.log(refined_probs.clamp(min=1e-8))
        return refined_log_probs, uncertainty_mask.cpu().tolist(), avg_caps_conf, debug_data

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
            1. Przelicza logity/log_probs na rozkład prawdopodobieństwa [0, 1] i wybiera klatki poniżej progu.
            2. Mapuje kroki czasowe na współrzędne pikselowe obrazu wejściowego, uwzględniając
               skalowanie wynikające z architektury sieci.
            3. Grupuje sąsiadujące kroki o niskiej ufności w spójne regiony. """
        steps = crnn_probs.shape[0]
        if steps == 0:
            return []

        is_probs = (
            crnn_probs.min() >= 0.0
            and crnn_probs.max() <= 1.0
            and torch.allclose(
                crnn_probs.sum(dim=-1),
                torch.ones(steps, device=crnn_probs.device, dtype=crnn_probs.dtype),
                atol=1e-2,
            )
        )
        probs = crnn_probs if is_probs else torch.softmax(crnn_probs, dim=-1)
        top1_p, top1_idx = torch.max(probs, dim=-1)

        # Obliczamy szerokość jednego kroku
        stride = width / steps

        # Dynamiczny margines okna
        window_factor = 3.2 if intensive else 2.6
        half_window = int((stride * window_factor) / 2)

        candidates = []

        # Logika Smart Adaptive Thresholding
        try:
            if confusion_matrix_path and os.path.exists(confusion_matrix_path):
                cm = np.load(confusion_matrix_path).astype(np.float64)
                row_sums = cm.sum(axis=1)
                errors_per_char = row_sums - np.diag(cm)
                mapping_has_blank = (
                    0 in char_to_idx.values() or "[blank]" in char_to_idx
                )
                if mapping_has_blank:
                    offset = 1 if cm.shape[0] == len(char_to_idx) - 1 else 0
                else:
                    offset = 1 if cm.shape[0] == len(char_to_idx) + 1 else 0
                confusing_indices = {int(idx) + offset for idx in np.argsort(errors_per_char)[-15:]}
            else:
                confusing_indices = set()
        except (OSError, ValueError, TypeError, AttributeError) as e:
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

        if not candidates:
            return []

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
        """ Generuje błędy z uwzględnieniem dryfu OCR, zniekształceń fizycznych i błędów segmentacji. """
        res = []
        i = 0
        while i < len(text):
            char = text[i]
            two_chars = text[i:i + 2]
            rand = random.random()

            # Jeśli to spacja, istnieje szansa, że Seam Carving jej nie zauważył (sklejenie słów)
            if char == " " and rand < (rate * 1.5): 
                i += 1
                continue

            # Symulacja dryfu - sąsiedni znak wpływa na obecny
            if 0 < i < len(text) - 1 and random.random() < 0.02:
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
                # Prawidłowy znak
                if use_markers and random.random() < 0.03:
                    res.append(f"{UNC_START}{char}{UNC_END}")
                else:
                    res.append(char)

            # Sztuczne cięcie wewnątrz słowa (wstawienie fałszywej spacji po znaku) - nadsegmentacja
            if char != " " and random.random() < (rate * 0.5):
                res.append(" ")

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

        # POPRAWKA 1: Wyciąganie tagu językowego, żeby go nie zgubić przy usuwaniu kontekstu
        lang_tag = "<pl>" if "<pl>" in clean_prev else "<en>"
        clean_prev = clean_prev.replace("<pl>", "").replace("<en>", "").strip()

        # Obsługa kontekstu z poprzednią linijką jako referencja
        if not clean_prev or random.random() < 0.2:
            context_for_model = lang_tag # Zostawiamy sam tag dla rozpoznania języka
        else:
            clean_prev = clean_prev[-150:]
            if random.random() < 0.3:
                context_for_model = f"{lang_tag} {self.generator.apply_errors(clean_prev, rate=0.03)}"
            else:
                context_for_model = f"{lang_tag} {clean_prev}"

        # Zaszumianie aktualnego wejścia (Data Augmentation dla Transformera)
        if is_real:
            # POPRAWKA 2: Używamy source_text, a nie niezdefiniowanej zmiennej z locals()
            actual_source = source_text

            if random.random() < 0.2:
                # 20% szansy: bierzemy "brudny" wynik z OCR i psujemy go jeszcze bardziej
                noisy_curr = self.generator.apply_errors(source_text, self.rate, self.markers)
            else:
                # 80% szansy: bierzemy autentyczny błąd z modelu CRNN
                noisy_curr = actual_source
        else:
            # Scenariusz syntetyczny
            noisy_curr = self.generator.apply_errors(clean_curr, self.rate, self.markers)

        # POPRAWKA 3: Generowanie promptu zależnie od języka (żeby model nauczył się obu formatów!)
        if lang_tag == "<pl>":
            src = f"kontekst: {context_for_model} popraw ocr: {noisy_curr}"
        else:
            src = f"context: {context_for_model} fix ocr: {noisy_curr}"

        tgt = clean_curr

        m_in = self.tokenizer(src, max_length=512, truncation=True, return_tensors="pt")
        m_tg = self.tokenizer(text_target=tgt, max_length=512, truncation=True, return_tensors="pt")

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
        print(f"[Curriculum] Epoka {ep + 1}: Szum {rate * 100:.1f}%, Markery: {markers}")


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
def compute_metrics(eval_preds, tokenizer=None):
    """ Oblicza CER dla Transformera (ByT5/T5). """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Zastępujemy -100 (maska straty) ID padu, aby tokenizer mógł to zdekodować (zarówno w preds, jak i w labels)
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0
    preds = np.where(preds != -100, preds, pad_id)
    labels = np.where(labels != -100, labels, pad_id)

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
    """Dynamicznie przełącza wagi LoRA oraz bazę wiedzy autora w locie."""
    adapter_name = f"adapter_{user_id}"
    adapter_dir = f"/app/user_data/{user_id}/lora_weights"

    # Przełączamy pamięć słownikową
    self.author_memory = PersistentAuthorMemory(user_id)

    if not os.path.exists(adapter_dir):
        if isinstance(self.model, PeftModel):
            self.model.disable_adapter_layers()  # Powrót do czystego modelu
        return

    # Jeśli model bazowy nie jest jeszcze opakowany w PeftModel, inicjalizujemy go pierwszym adapterem
    if not isinstance(self.model, PeftModel):
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_dir,
            adapter_name=adapter_name,
        ).to(self.device)
        self.model.eval()
        self.loaded_adapters.append(adapter_name)
    elif adapter_name not in self.loaded_adapters:
        self.model.load_adapter(adapter_dir, adapter_name=adapter_name)
        self.loaded_adapters.append(adapter_name)

    # Aktywacja wag konkretnego użytkownika
    self.model.enable_adapter_layers()
    self.model.set_adapter(adapter_name)


def run_transformer_adaptation(json_data_path, model_path, output_path, matrix_path, encoder, language="pl"):
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
    lang_tag = f"<{language}>"
    user_pairs = []
    for item in user_data:
        # Formatowanie: wejście (to co widzi wizja) -> cel (poprawny tekst)
        vision_output = item.get("vision_output", item.get("input", ""))
        ground_truth = item.get("ground_truth", item.get("target", ""))

        # Symulacja pustego kontekstu, skupiamy się na naprawie słowa/linii
        raw_ctx = item.get("context", "")
        user_pairs.append({
            "context": f"{lang_tag} {raw_ctx}".strip(),
            "input": vision_output,
            "target": ground_truth,
        })

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

    """ Konfiguracja LoRA (Low-Rank Adaptation) — zamraża wagi bazowe ByT5 i trenuje
        wyłącznie niskorangowe adaptery (r=16, alpha=32) w dwóch grupach warstw:
        - Projekcje atencji ("q", "k", "v", "o"): kierują uwagę modelu na zaszumione
          fragmenty tekstu OCR oraz obsługują prefixy bilingwalne (<pl>, <en>).
        - Bloki Feed-Forward ("wi_0", "wi_1", "wo"): dopasowują słownictwo i reguły
          gramatyczne do specyfiki pisma użytkownika bez ryzyka halucynacji. """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
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
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adafactor",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=256
    )

    if len(train_ds) > 1:
        # Obliczamy rozmiary (90% na trening, reszta na walidację)
        train_size = max(1, int(0.9 * len(train_ds)))
        val_size = len(train_ds) - train_size

        # Zapewnia to, że przy każdym uruchomieniu te same próbki trafią do walidacji
        train_ds, val_ds = random_split(
            train_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        tqdm.write(f"[{now()}] Podział zakończony: Train={len(train_ds)}, Val={len(val_ds)}")
    else:
        val_ds = train_ds

    adapt_epoch_cb = EpochProgressCallback(stage_name="ByT5 LoRA Adapt")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            adapt_epoch_cb,
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
                # Utrwalamy wyzerowane wagi w tensorze weight, aby nie psuć kluczy state_dict przy load_best_model_at_end
                prune.remove(module, 'weight')

            elif mode == "random":
                # Usuwa losowe wagi niezależnie od ich wielkości
                prune.random_unstructured(module, name='weight', amount=base_amount)
                prune.remove(module, 'weight')


class StochasticPruningCallback(TrainerCallback):
    def __init__(self, initial_state, base_amount=0.1):
        self.initial_state = initial_state
        self.base_amount = base_amount

    def on_epoch_end(self, args, state, control, **kwargs):
        # Wyciągamy model ze słownika kwargs
        model = kwargs.get("model")

        if model is None:
            return
        
        if state.best_metric is not None and state.best_metric > 0.4:
            print("[LTH Pruning] CER za wysoki - pomijam przycinanie wag.")
            return

        # Pomijamy pruning w ostatniej epoce
        if state.epoch < args.num_train_epochs:
            print(f"[LTH Pruning] Stochastyczna kompresja wag po epoce {int(state.epoch)}.")

            # Używamy magnitude_jitter, co jest najbezpieczniejsze dla Transformerów
            apply_stochastic_lth_step(
                model,
                self.initial_state,
                base_amount=self.base_amount,
                mode="magnitude_jitter"
            )


def refine_ocr_text(noisy_text, context_text, model, tokenizer, device, language="pl"):
    """ Pojedyncza funkcja realizująca pełną logikę korekty semantycznej. """
    if not noisy_text.strip():
        return noisy_text

    # Wstrzykiwanie odpowiedniego tagu, aby model wiedział jakiego języka użyć
    lang_tag = f"<{language}> "
    full_context = lang_tag + context_text if context_text else lang_tag.strip()

    # Zmiana 'popraw HCR:' na 'popraw ocr:', aby zgadzało się z formatem z treningu!
    if language == "pl":
        prompt = f"kontekst: {full_context} popraw ocr: {noisy_text}"
    else:
        prompt = f"context: {full_context} fix ocr: {noisy_text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=2,
            length_penalty=1,
            repetition_penalty=1.2,
            early_stopping=True
        )

    refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    refined_text = re.sub(r'<.*?>|~', '', refined_text).strip()

    return refined_text


def run_full_training():
    """ Uruchamia bilingwalny proces treningu ByT5. Model uczy się jednocześnie poprawiać błędy polskie i angielskie. """
    seed_everything(SEED)
    print(f"[{now()}] START: Trening bilingwalny (PL + EN). Urządzenie: {DEVICE}")

    results_dir = globals().get("RESULTS_DIR", os.path.join(os.path.dirname(CHECKPOINT_BASE), "results"))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    real_pairs_pl = []
    real_pairs_en = []

    # Definiowanie alfabetu (92 znaki + [blank] na indeksie 0)
    chars_raw = [
        '!', '"', "'", '(', ')', ',', '-', '.', '/', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        ':', ';', '?',                                    
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z',                     
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
        'u', 'v', 'w', 'x', 'y', 'z',                     
        'Ó', 'ó', 'Ą', 'ą', 'Ć', 'ć', 'Ę', 'ę', 'Ł', 'ł', 
        'Ń', 'ń', 'Ś', 'ś', 'Ź', 'ź', 'Ż', 'ż'            
    ]
    num_classes_total = len(chars_raw) + 1

    # Ładowanie macierzy pomyłek (dla generatora szumu)
    gen_matrix_path = os.path.join(results_dir, "smart_confusion_matrix.npy")
    confusion_matrix = np.zeros((len(chars_raw), len(chars_raw)), dtype=np.float32)

    if os.path.exists(gen_matrix_path):
        loaded_cm = np.load(gen_matrix_path)
        if loaded_cm.shape == confusion_matrix.shape and np.sum(loaded_cm) > 0:
            confusion_matrix = loaded_cm
            np.save(CACHE_MATRIX, confusion_matrix)
            print(f"[{now()}] Załadowano macierz pomyłek z {gen_matrix_path}.")
    elif os.path.exists(CACHE_MATRIX):
        loaded_cm = np.load(CACHE_MATRIX)
        if loaded_cm.shape == confusion_matrix.shape:
            confusion_matrix = loaded_cm
            print(f"[{now()}] Załadowano macierz pomyłek z cache.")

    if np.sum(confusion_matrix) == 0:
        print(f"[{now()}] Warning: Brak niezerowej macierzy pomyłek. Używam zerowej.")

    # Wczytywanie polskich par z RESULTS_DIR (z fallbackiem do BASE_DIR)
    pl_json_path = os.path.join(results_dir, "transformer_train_pl_pages.json")
    if not os.path.exists(pl_json_path):
        pl_json_path = os.path.join(BASE_DIR, "transformer_train_pl_pages.json")

    if os.path.exists(pl_json_path):
        with open(pl_json_path, 'r', encoding='utf-8') as f:
            real_pairs_pl = json.load(f)
        print(f"[{now()}] Wczytano {len(real_pairs_pl)} polskich par z {pl_json_path}.")
    else:
        print(f"[{now()}] Warning: Nie znaleziono pliku z polskimi parami ({pl_json_path}).")

    # Wczytywanie lub generowanie par angielskich (IAM)
    if os.path.exists(CACHE_PAIRS):
        with open(CACHE_PAIRS, 'r', encoding='utf-8') as f:
            real_pairs_en = json.load(f)
        print(f"[{now()}] Wczytano {len(real_pairs_en)} angielskich par z cache.")
    elif os.path.exists(SENTENCES_TXT) and os.path.exists(FORMS_DIR):
        from Models.ResNetCRNNWordRecognition import ResNetCRNN
        print(f"[{now()}] Generowanie danych EN z IAM.")

        ckpt = torch.load(CRNN_CHECKPOINT, map_location=DEVICE)
        sd_crnn = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        weight_key = 'output.weight' if 'output.weight' in sd_crnn else ('fc.weight' if 'fc.weight' in sd_crnn else None)
        n_classes_crnn = sd_crnn[weight_key].shape[0] if weight_key else num_classes_total

        if isinstance(ckpt, dict) and "char_list" in ckpt:
            crnn_chars = ckpt["char_list"]
        else:
            crnn_chars = chars_raw[:n_classes_crnn - 1]

        model_crnn = ResNetCRNN(n_classes_crnn).to(DEVICE)
        model_crnn.load_state_dict(sd_crnn)

        ckpt_caps = torch.load(CAPS_CHECKPOINT, map_location=DEVICE)
        sd_caps = ckpt_caps["model_state"] if isinstance(ckpt_caps, dict) and "model_state" in ckpt_caps else ckpt_caps
        n_classes_caps = ckpt_caps.get("num_classes", num_classes_total) if isinstance(ckpt_caps, dict) else num_classes_total

        model_caps = CapsNet(num_classes=n_classes_caps, context_dim=1024).to(DEVICE)
        model_caps.load_state_dict(sd_caps)

        inference_pipe = CRNNInferencePipeline(model_crnn, crnn_chars, DEVICE)
        encoder_obj = getattr(inference_pipe, "encoder", inference_pipe)
        inference_pipe.refiner = CascadeRefinementNetwork(model_caps, chars_raw, encoder_obj, matrix_path=CACHE_MATRIX)

        forms_lines = parse_iam_forms_to_lines(SENTENCES_TXT)
        real_pairs_en, generated_matrix = generate_contextual_noise_data(inference_pipe, forms_lines, chars_raw)

        if np.sum(confusion_matrix) == 0 and np.sum(generated_matrix) > 0:
            confusion_matrix = generated_matrix
            np.save(CACHE_MATRIX, confusion_matrix)

        with open(CACHE_PAIRS, 'w', encoding='utf-8') as f:
            json.dump(real_pairs_en, f, ensure_ascii=False)

        del model_crnn, model_caps, inference_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[{now()}] Pomijam generowanie par EN z IAM (brak {FORMS_DIR} lub {SENTENCES_TXT}).")

    # Podział na train/val przed, żeby uniknąć wycieku duplikatów do walidacji
    tagged_pl = [
        {"context": f"<pl> {p.get('context') or ''}".strip(), "input": p['input'], "target": p['target']}
        for p in real_pairs_pl
    ]
    tagged_en = [
        {"context": f"<en> {p.get('context') or ''}".strip(), "input": p['input'], "target": p['target']}
        for p in real_pairs_en
    ]

    if len(tagged_pl) > 1:
        train_pl, val_pl = random_split(tagged_pl, [0.95, 0.05], generator=torch.Generator().manual_seed(SEED))
        train_pl, val_pl = list(train_pl), list(val_pl)
    else:
        train_pl, val_pl = tagged_pl, []

    if len(tagged_en) > 1:
        train_en, val_en = random_split(tagged_en, [0.95, 0.05], generator=torch.Generator().manual_seed(SEED))
        train_en, val_en = list(train_en), list(val_en)
    else:
        train_en, val_en = tagged_en, []

    num_en = len(train_en)
    num_pl = len(train_pl)

    if num_pl > 0:
        if num_en > 0 and num_pl < num_en:
            multiplier = max(1, num_en // num_pl)
            train_pl = train_pl * multiplier
            print(f"[{now()}] Wykonano oversampling danych PL (x{multiplier}).")
        elif num_en == 0 and num_pl < 250:
            multiplier = max(1, 250 // num_pl)
            train_pl = train_pl * multiplier
            print(f"[{now()}] Wykonano oversampling danych PL (x{multiplier}) dla równowagi z korpusem Wiki.")

    train_real = train_pl + train_en
    val_real = val_pl + val_en
    random.shuffle(train_real)

    # Inicjalizacja Transformera ByT5 z tokenami specjalnymi
    tokenizer = ByT5Tokenizer.from_pretrained(MODEL_NAME)
    LANGUAGE_TOKENS = ['<pl>', '<en>']
    SPECIAL_TOKENS = [LC_TOKEN, UNC_START, UNC_END] + LANGUAGE_TOKENS
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    model_byt5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model_byt5.resize_token_embeddings(len(tokenizer))
    model_byt5.to(DEVICE)

    # Ładowanie bilingwalnej Wikipedii, dzielimy jeden artykuł na dwie sąsiednie linie kontekstu
    print(f"[{now()}] Pobieranie bilingwalnej Wikipedii.")
    wiki_combined_pairs = []
    try:
        wiki_pl = load_dataset("wikimedia/wikipedia", "20231101.pl", split="train", streaming=True)
        wiki_en = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        pl_iter, en_iter = iter(wiki_pl), iter(wiki_en)

        def _next_paired_chunks(iterator):
            for sample in iterator:
                txt = re.sub(r'\s+', ' ', sample.get('text', '')).strip()
                if len(txt) >= 80 and not txt.startswith('='):
                    split_pt = txt.rfind(' ', 35, min(160, len(txt) - 25))
                    if split_pt == -1:
                        split_pt = len(txt) // 2
                    prev_chunk = txt[:split_pt].strip()
                    curr_chunk = txt[split_pt:split_pt + 180].strip()
                    if len(prev_chunk) >= 15 and len(curr_chunk) >= 15:
                        return prev_chunk, curr_chunk
            return None, None

        target_wiki_pairs = max((len(train_real) + len(val_real)) * 3, 1000)
        for _ in range(target_wiki_pairs):
            p1, p2 = _next_paired_chunks(pl_iter)
            if not p1 or not p2:
                break
            wiki_combined_pairs.append((f"<pl> {p1}", p2))

            e1, e2 = _next_paired_chunks(en_iter)
            if not e1 or not e2:
                break
            wiki_combined_pairs.append((f"<en> {e1}", e2))
    except KeyboardInterrupt:
        print(f"[{now()}] Przerwano pobieranie Wikipedii przez użytkownika.")
    except FileNotFoundError as fnf_error:
        print(f"[{now()}] Nie znaleziono pliku z Wikipedią: {fnf_error}")
    except (ConnectionError, TimeoutError, RuntimeError, ValueError, OSError) as e:
        print(f"[{now()}] Problem podczas pobierania strumienia Wikipedii: {e}")
    except Exception as e:
        print(f"[{now()}] Nieznany problem podczas pobierania strumienia Wikipedii: {e}")

    print(f"[{now()}] Przygotowano {len(wiki_combined_pairs)} par z Wikipedii oraz {len(train_real)} train / {len(val_real)} val par rzeczywistych.")

    # Podział na train/val wyłącznie dla Wikipedii + limit rozmiaru walidacji
    if len(wiki_combined_pairs) > 1:
        val_wiki_len = min(300, max(1, int(0.05 * len(wiki_combined_pairs))))
        train_wiki_len = len(wiki_combined_pairs) - val_wiki_len
        train_wiki, val_wiki = random_split(
            wiki_combined_pairs, [train_wiki_len, val_wiki_len], generator=torch.Generator().manual_seed(SEED)
        )
    else:
        train_wiki, val_wiki = wiki_combined_pairs, []

    error_gen = HybridVisualErrorGenerator(confusion_matrix, chars_raw)

    train_ds = ContextualCurriculumDataset(list(train_wiki), train_real, error_gen, tokenizer)
    val_ds = ContextualCurriculumDataset(list(val_wiki), val_real, error_gen, tokenizer)
    val_ds.rate = 0.10
    val_ds.markers = True

    if len(val_ds) == 0 and len(train_ds) > 1:
        train_ds, val_ds = random_split(
            train_ds,
            [len(train_ds) - 1, 1],
            generator=torch.Generator().manual_seed(SEED)
        )

    # Wstępne wygenerowanie próbek walidacyjnych raz, aby nie zmieniały się w trakcie treningu
    random.seed(SEED)
    np.random.seed(SEED)
    fixed_val_ds = [val_ds[i] for i in range(len(val_ds))]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model_byt5,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=32,
        learning_rate=3e-4,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=256,
        fp16=False,
        bf16=use_bf16,
        optim="adafactor",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        report_to="mlflow",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False
    )

    pruning_cb = StochasticPruningCallback(None, base_amount=0.1)
    orig_on_epoch_end = pruning_cb.on_epoch_end

    def _safe_pruning_on_epoch_end(args, state, control, **kwargs):
        # Pomijamy pruning po 1. epoce, bo model dopiero stabilizuje nowe tokeny specjalne, więc wczesne przycinanie może zniszczyć uczenie
        if state.epoch is None or state.epoch < 2:
            return
        return orig_on_epoch_end(args, state, control, **kwargs)

    pruning_cb.on_epoch_end = _safe_pruning_on_epoch_end

    epoch_cb = EpochProgressCallback(stage_name="ByT5 Main")
    trainer = Seq2SeqTrainer(
        model=model_byt5,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=fixed_val_ds,
        processing_class=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        data_collator=data_collator,
        callbacks=[
            CurriculumCallback(train_ds),
            pruning_cb,
            EarlyStoppingCallback(early_stopping_patience=2),
            epoch_cb,
        ]
    )

    print(f"[{now()}] Uruchamianie treningu.")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[{now()}] Trening zakończony. Model zapisany w {OUTPUT_DIR}")


def run_inference_test():
    """ Uruchamia test inferencji na wytrenowanym modelu Refinera. """
    print(f"[{time.strftime('%H:%M:%S')}] Tryb TESTU INFERENCJI (Walidacja końcowa).")

    trained_model_path = OUTPUT_DIR

    if not os.path.exists(trained_model_path):
        print(f"[{now()}] Nie znaleziono wytrenowanego modelu w {OUTPUT_DIR}. Używam bazy.")
        trained_model_path = MODEL_NAME

    tokenizer = ByT5Tokenizer.from_pretrained(trained_model_path)

    if trained_model_path == MODEL_NAME:
        tokenizer.add_special_tokens({'additional_special_tokens': [LC_TOKEN, UNC_START, UNC_END, '<pl>', '<en>']})

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    tests = [
        {
            "lang": "pl",
            "ctx": "Zgubiłem wczoraj klucze do mieszkania",
            "noisy": "Musz<unc>e</unc> t<unc>e</unc>raz wymi<unc>c</unc>nić zamek w drzw1ach."
        },
        {
            "lang": "en",
            "ctx": "The handwritten manuscript was found in the old library.",
            "noisy": "It conta1n<unc>c</unc>d s<unc>c</unc>veral unk<unc>rn</unc>own syrnbols."
        }
    ]

    for t in tests:
        print(f"Test dla języka {t['lang'].upper()}")
        fix = refine_ocr_text(t['noisy'], t['ctx'], model, tokenizer, DEVICE, t['lang'])
        print(f"Kontekst: {t['ctx']}")
        print(f"Wejście (Vision + CapsNet): {t['noisy']}")
        print(f"Wynik końcowy (Transformer): {fix}")

if __name__ == "__main__":
    run_full_training()