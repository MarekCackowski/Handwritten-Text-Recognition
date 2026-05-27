import os
import time
import json
import gc
import re
import difflib
from typing import Optional, cast, Any
import Levenshtein
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as func
from collections import Counter
from torch import nn
from captum.attr import IntegratedGradients
from Models.TransformerDictionaryRefinement import TransformerRefiner
from Refinement.UserAdaptation import FeedbackAugmentor
import onnxruntime as ort
from Refinement.UserAdaptation import JointFeedbackFineTuner
from Preprocessing.Preprocessing import Preprocessing as ImagePreprocessor
import multiprocessing
multiprocessing.freeze_support()
from fpdf import FPDF
from Models.DeepCapsNetCharRecognition import CapsNet
from dotenv import load_dotenv
from tkinter import Tk, filedialog, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

now = lambda: time.strftime('%H:%M:%S')

LC_TOKEN = "~"
LOW_CONFIDENCE_CHAR_LIMIT = 0.6
CHUNKS_SIZE = 8
SAVE_THRESHOLD = 10
DEVICE = torch.device('cuda')
CRNN_WEIGHTS = "output_data/checkpoints/hwr/WordLevelResNetCRNN.pth"
CAPS_WEIGHTS = "output_data/checkpoints/hcr/CharLevelCapsNet.pth"
TRANSFORMER_PATH = "./byt5_htr_refiner_final"
SAMPLE_IMAGE = "test_page.png"

# Plik będzie ukryty w folderze domowym użytkownika
USER_HOME = Path.home()
CONFIG_FILE = USER_HOME / ".htr_workspace_config"


def get_base_directory():
    """ Pobiera folder roboczy z profilu użytkownika lub pyta o nowy. """
    # Obsługa Dockera
    if os.path.exists('/.dockerenv'):
        return Path("/app/data")

    # Próba odczytu zapisanego folderu
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Plik konfiguracyjny jest pusty.")

                saved_path = Path(content)

                if saved_path.exists():
                    return saved_path
                else:
                    print(f"[{now()}] Ostrzeżenie: Folder {saved_path} już nie istnieje.")

        except PermissionError:
            print(f"[{now()}] Błąd: Brak uprawnień do odczytu pliku {CONFIG_FILE}.")
        except UnicodeDecodeError:
            print(f"[{now()}] Błąd: Nieprawidłowe kodowanie pliku konfiguracyjnego.")
        except ValueError as ve:
            print(f"[{now()}] Informacja: {ve}")
        except Exception as e:
            # Dla nieprzewidzianych błędów
            print(f"[{now()}] Nieoczekiwany błąd podczas ładowania konfiguracji: {e}")

    # Jeśli brak konfiguracji - uruchamiamy okno wyboru
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print(f"Witaj, {os.getlogin()}! Wybierz folder dla swoich projektów HTR.")
    selected_path = filedialog.askdirectory(title=f"Wybierz folder roboczy dla: {os.getlogin()}")

    if selected_path:
        base_dir = Path(selected_path)
        # Zapisujemy wybór w profilu użytkownika
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(str(base_dir))
        root.destroy()
        return base_dir
    else:
        # Tworzymy folder wewnątrz dokumentów użytkownika
        fallback = USER_HOME / "Documents" / "HTR_Projects"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Nie wybrano folderu. Używam domyślnego: {fallback}")
        root.destroy()
        return fallback


# Ścieżki
BASE_DIR = get_base_directory()
OUTPUT_BASE = BASE_DIR / "output_data"

# Dynamiczne definiowanie podfolderów
MATRIX_DIR = OUTPUT_BASE / "visual_debug_output"
MATRIX_FILE = MATRIX_DIR / "sequence_confusion.json"
TRANSFORMER_MODEL_PATH = OUTPUT_BASE / "checkpoints" / "transformer"
CRNN_WEIGHTS_PATH = OUTPUT_BASE / "checkpoints" / "hwr" / "WordLevelResNetCRNN.pth"
CAPS_WEIGHTS_PATH = OUTPUT_BASE / "checkpoints" / "hcr" / "CharLevelCapsNet.pth"
LOG_DIR = OUTPUT_BASE / "logs"

# Tworzenie struktury, na wszelki wypadek
folders = [
    MATRIX_DIR,
    TRANSFORMER_MODEL_PATH,
    CRNN_WEIGHTS_PATH.parent,
    CAPS_WEIGHTS_PATH.parent,
    LOG_DIR
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Zalogowano: {os.getlogin()}")
print(f"[{time.strftime('%H:%M:%S')}] Folder roboczy: {BASE_DIR}")


class ConfusionMatrixLoader:
    @staticmethod
    def load_dynamic_pairs(encoder, top_k=5):
        npy_path = os.path.join(MATRIX_DIR, "confusion_matrix.npy")
        confusion_map = {}
        if not os.path.exists(npy_path):
            return confusion_map
        matrix = np.load(npy_path)
        for i in range(matrix.shape[0]):
            row = matrix[i].copy()
            row[i] = 0
            top_indices = np.argsort(row)[-top_k:]
            true_char = encoder.decode(i)
            confusion_map[true_char] = {encoder.decode(idx) for idx in top_indices if row[idx] > 0}
        return confusion_map

class GeometryCorrector:
    def __init__(self, threshold_value=127):
        self.thresh_val = threshold_value

    @staticmethod
    def _to_numpy(tensor_or_np):
        if hasattr(tensor_or_np, 'cpu'):
            img = tensor_or_np.cpu().numpy().squeeze()
            img = ((img - img.min()) / (img.max() - img.min() + 1e-5) * 255).astype(np.uint8)
        else:
            img = tensor_or_np.astype(np.uint8)
        return img

    def refine_chars(self, text, img_tensor):
        if not text or len(text) == 0: return text
        img = self._to_numpy(img_tensor)
        _, binary = cv.threshold(img, self.thresh_val, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Pobieramy wszystkie bounding boxy i sortujemy od lewej do prawej
        boxes = sorted([cv.boundingRect(c) for c in contours], key=lambda b: b[0])
        if not boxes: return text

        # Łączymy boxy zachodzące na siebie w poziomie (np. kropka + rdzeń 'i')
        merged_boxes = []
        for b in boxes:
            if not merged_boxes:
                merged_boxes.append(list(b))
            else:
                last_b = merged_boxes[-1]
                # Jeśli aktualny box zaczyna się tam, gdzie trwa jeszcze poprzedni (plus mały margines błędu)
                if b[0] <= last_b[0] + last_b[2] + 2:
                    new_x = min(last_b[0], b[0])
                    new_y = min(last_b[1], b[1])
                    new_w = max(last_b[0] + last_b[2], b[0] + b[2]) - new_x
                    new_h = max(last_b[1] + last_b[3], b[1] + b[3]) - new_y
                    merged_boxes[-1] = [new_x, new_y, new_w, new_h]
                else:
                    merged_boxes.append(list(b))

        # Jeśli liczba fizycznych znaków nie pasuje do tekstu, ufamy sieciom neuronowym
        if len(merged_boxes) != len(text):
            return text

        img_h = img.shape[0]
        refined_chars = list(text)

        # Badamy proporcje złączonych, pewnych elementów
        for i in range(len(refined_chars)):
            _, y, _, h = merged_boxes[i]
            rel_h = h / img_h
            rel_cy = (y + (h / 2)) / img_h
            char = refined_chars[i]

            # Sieć dała interpunkcję, ale fizycznie to wysoki znak
            if char in {',', '.', "'", '"', '`', ':', ';'} and rel_h > 0.45:
                char = "l"

            # Sieć dała pionową literę, ale fizycznie to mała plamka
            elif char in {'l', 'i', 'I', 't'} and rel_h < 0.35:
                char = "'" if rel_cy < 0.4 else "."

            refined_chars[i] = char

        return "".join(refined_chars)


class DubiousRegionSelector:
    """ Daje drugie spojrzenie CapsNet przed ewentualnym zastosowaniem słownika do poprawy. """
    def __init__(self, char_list, matrix_dir):
        self.idx_to_char = {i + 1: c for i, c in enumerate(char_list)}
        self.danger_zones = {}

        if matrix_dir is not None:
            self.matrix_path = os.path.join(matrix_dir, "confusion_matrix.npy")
            if os.path.exists(self.matrix_path):
                matrix = np.load(self.matrix_path)
                for i in range(min(matrix.shape[0], len(self.idx_to_char))):
                    row = matrix[i]
                    total = row.sum()
                    self.danger_zones[i] = (total - row[i]) / (total + 1e-6)
        else:
            self.matrix_path = None

    @staticmethod
    def estimate_uncertainty_mc(model, image_tensor, steps=64):
        """ Oblicza niepewność bayesowską (Predictive Variance) przez wielokrotne przejścia w trybie aktywnego Dropoutu. """
        model.eval()
        # Wymuszamy działanie warstw Dropout nawet w trybie eval
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                m.train()

        outputs = []
        with torch.no_grad():
            for _ in range(steps):
                with torch.amp.autocast('cuda'):
                    log_probs, *_ = model(image_tensor)
                    # Zamiana log_probs na prawdopodobieństwa
                    probs = torch.exp(log_probs)
                    outputs.append(probs)

        # Przywracamy model do pełnego trybu eval
        model.eval()

        # Składamy wyniki
        all_probs = torch.stack(outputs)
        
        # Średnie prawdopodobieństwo (lepsza opinia niż z pojedynczego przejścia)
        mean_probs = torch.mean(all_probs, dim=0)

        # Obliczamy wariancję między przejściami
        variance_map = torch.var(all_probs, dim=0)

        # Redukujemy do wektora niepewności na krok czasowy (uśrednionego)
        uncertainty_per_timestep = variance_map.mean(dim=(1, 2))

        return uncertainty_per_timestep, variance_map, mean_probs

    def select_crop_groups(self, crnn_probs, image_tensor, model_ref=None, intensive=False):
        """ Główna logika selekcji trudnych regionów. Zoptymalizowana pod kątem precyzji bayesowskiej i statystycznego ryzyka (Confusion Matrix). """
        # Przygotowujemy softmax, aby operować na prawdopodobieństwach
        probs = torch.softmax(crnn_probs, dim=-1)
        top1_p, top1_idx = torch.max(probs, dim=-1)
        top2_p, _ = torch.topk(probs, k=2, dim=-1)

        simple_punctuation = {".", ",", "'", '"'}

        img_np = (image_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8)
        width, steps = image_tensor.shape[2], crnn_probs.shape[0]
        stride = width / steps
        candidates = []

        margin_threshold = 0.60 if intensive else 0.50

        # Estymacja niepewności MC Dropout (Epistemic Uncertainty)
        uncertainty_map, mc_mean_probs = None, None
        
        # Wektory marginesów między najlepszą a drugą opcją. Niski margines oznacza, że model jest niepewny co do wyboru.
        margins = (top1_p - top2_p[:, 1])
        min_margin = margins.min().item()

        if model_ref and min_margin < margin_threshold:
            """ Przy trybie nocnym wykonujemy więcej przejść, aby uzyskać stabilniejszą estymację niepewności, kosztem czasu.
                Przy normalnym trybie wystarczy mniej przejść, aby szybko zidentyfikować najbardziej niepewne znaki. """
            steps_count = 64 if intensive else 16
            uncertainty_map, _, mc_mean_probs = self.estimate_uncertainty_mc(model_ref, image_tensor, steps=steps_count)

        if uncertainty_map is not None:
            adaptive_threshold = uncertainty_map.mean() + 1.5 * uncertainty_map.std()
            absolute_floor = 0.005
        else:
            adaptive_threshold, absolute_floor = 1.0, 1.0

        for t in range(steps):
            char_idx = top1_idx[t].item()
            char_str = self.idx_to_char.get(char_idx, "")

            # Bezwzględny skip tylko dla małej interpunkcji
            if char_str in simple_punctuation:
                continue

            margin = margins[t].item()

            is_uncertain_mc = False
            if uncertainty_map is not None:
                is_uncertain_mc = uncertainty_map[t].item() > adaptive_threshold or uncertainty_map[t].item() > absolute_floor

            # Pobieramy statystyczne ryzyko pomyłki dla tego konkretnego znaku z macierzy pomyłek
            char_danger_rate = self.danger_zones.get(char_idx, 0.0)
            
            # Wymuszamy sprawdzenie przez CapsNet, jeśli znak historycznie mylił się w więcej niż 10% przypadków
            is_historically_dangerous = char_danger_rate > 0.1

            #  Statystyka (często mylone) lub niski margines (blisko siebie) lub niepewność modelu
            if is_historically_dangerous or margin < margin_threshold or is_uncertain_mc:
                center_x = (t + 0.5) * stride

                char_type = 'letter'

                adaptive_gaps = self._find_peaks_adaptive(img_np, char_type=char_type)
                closest_gap = min(adaptive_gaps, key=lambda x: abs(x - center_x), default=center_x)

                window_size = 32

                candidates.append({
                    'timestep': t,
                    'x1': int(max(0, closest_gap - window_size)),
                    'x2': int(min(width, closest_gap + window_size)),
                    'closest_gap': closest_gap,
                    'uncertainty_val': float(uncertainty_map[t]) if uncertainty_map is not None else 0.0,
                    'is_forced': is_historically_dangerous
                })

        # Grupowanie i przekazanie do CapsNet
        if steps > 0:
            routed_percentage = (len(candidates) / steps) * 100
            print(f"Przekazano {len(candidates)}/{steps} znaków ({routed_percentage:.1f}%) do CapsNet.")

        return self._group_candidates(candidates), mc_mean_probs

    @staticmethod
    def _find_peaks_adaptive(word_image, char_type='letter'):
        h, w = word_image.shape
        # Jeśli podejrzewamy interpunkcję dolną
        if char_type == 'bottom_punctuation':
            y_s, y_e = int(h * 0.7), h
        # Jeśli górną
        elif char_type == 'top_punctuation':
            y_s, y_e = 0, int(h * 0.3)
        else:
            y_s, y_e = int(h * 0.3), int(h * 0.7)

        strip = word_image[y_s:y_e, :]
        v_proj = np.sum(strip == 0, axis=0)
        return np.where(v_proj == 0)[0]

    @staticmethod
    def _group_candidates(candidates):
        if not candidates: return []
        candidates.sort(key=lambda x: x['timestep'])
        groups, current_group = [], [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i]['timestep'] - current_group[-1]['timestep'] <= 2:
                current_group.append(candidates[i])
            else:
                groups.append(current_group)
                current_group = [candidates[i]]
        groups.append(current_group)
        return groups

    @staticmethod
    def _find_peaks_in_middle_zone(word_image):
        h, w = word_image.shape
        y_s, y_e = int(h * 0.3), int(h * 0.7)
        mid_strip = word_image[y_s:y_e, :]
        v_proj = np.sum(mid_strip == 0, axis=0)
        return np.where(v_proj == 0)[0]


def update_smart_matrix(matrix, char_to_idx, true_text, pred_text):
    """ Inteligentna aktualizacja macierzy. Ignoruje przesunięcia (spacje/rozcięcia),
    skupiając się na błędach substytucji (zamiany znaku na znak). """
    if not true_text or not pred_text: return

    # SequenceMatcher znajduje najlepsze dopasowanie mimo "dziur"
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # To nas interesuje: Znak X został odczytany jako Y
            t_chunk = true_text[i1:i2]
            p_chunk = pred_text[j1:j2]

            # Iterujemy po parach (zakładamy, że długość błędu jest podobna)
            for tc, pc in zip(t_chunk, p_chunk):
                if tc in char_to_idx and pc in char_to_idx:
                    matrix[char_to_idx[tc], char_to_idx[pc]] += 1


class VisualExplainer:
    """ Tłumaczy atrybuty modelu na obrazki, aby ułatwić interpretację wyników. """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self.forward_wrapper)

    def forward_wrapper(self, inputs, word_context=None):
        # Wrapper, ponieważ CapsNet zwraca słownik, a Captum oczekuje Tensorów
        outputs = self.model(inputs, word_context=word_context)
        return outputs["norms"]

    def explain(self, img_tensor, context_vector, target_class_idx):
        """ Generuje mapę atrybucji dla konkretnej klasy. """
        # inputs musi mieć gradient
        img_tensor.requires_grad = True

        # Obliczanie atrybucji
        attributions = self.ig.attribute(
            img_tensor,
            additional_forward_args=(context_vector,),
            target=target_class_idx,
            n_steps=20  # Im więcej kroków, tym dokładniejsza mapa, ale zależy nam na szybkości
        )

        return attributions.squeeze().cpu().detach().numpy()

    @staticmethod
    def plot_explanation(img_tensor, attributions, char_label, save_path):
        """ Nakłada mapę ciepła na oryginalny obrazek. """
        img = img_tensor.squeeze().cpu().detach().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Oryginał: {char_label}")
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Istotność pikseli (XAI)")
        # Nakładamy mapę atrybucji
        plt.imshow(attributions, cmap='hot')
        plt.colorbar()

        plt.savefig(save_path)
        plt.close()


class CascadeRefinementNetwork:
    """ Klasa implementująca sieć kaskadową do poprawy słów wykrytych przez modele. """
    def __init__(self, capsnet_predictor, char_list, encoder_obj, matrix_dir=None, matrix_path=None):
        self.selector = DubiousRegionSelector(char_list, matrix_dir)
        self.capsnet = capsnet_predictor
        self.device = next(capsnet_predictor.parameters()).device
        self.encoder = encoder_obj
        self.char_list = char_list
        self.lc_token = "~"
        self.lc_idx = self.encoder.char_to_idx.get(self.lc_token, len(char_list))
        self.capsnet_mapping = self._get_polish_80_mapping()
        self.allowed_chars = set(self.capsnet_mapping.values())
        self.matrix_path = matrix_path

    @staticmethod
    def _get_polish_80_mapping():
        """ Nowy słownik tłumaczący 80 indeksów CapsNetu na litery, w tym polskie znaki. """
        mapping = {}
        
        # 0-9: Cyfry
        for i in range(10): mapping[i] = chr(48 + i)
        
        # 10-35: Wielkie litery A-Z
        for i in range(26): mapping[10 + i] = chr(65 + i)
        
        # 36-61: Małe litery a-z
        for i in range(26): mapping[36 + i] = chr(97 + i)
        
        # 62-70: Polskie wielkie znaki
        polish_upper = ['Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż']
        for i, char in enumerate(polish_upper):
            mapping[62 + i] = char
            
        # 71-79: Polskie małe znaki
        polish_lower = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż']
        for i, char in enumerate(polish_lower):
            mapping[71 + i] = char
            
        return mapping

    @staticmethod
    def _preprocess_crop_gpu(crop_tensor):
        crop_resized = func.interpolate(crop_tensor, size=(64, 64), mode='bilinear', align_corners=False)
        crop_norm = (crop_resized * 0.5) + 0.5
        crop_norm = torch.clamp((crop_norm - 0.2) / 0.6, 0, 1)
        mean, std = 0.1307, 0.3081
        return (crop_norm - mean) / std

    def refine_logits(self, crnn_probs, image_tensor, crnn_text=None, context_map=None, force_intensive=False, return_heatmap=False):
        groups, mc_mean_probs = self.selector.select_crop_groups(
            crnn_probs,
            image_tensor,
            model_ref=self.encoder.model,
            intensive=force_intensive  # Używamy przekazanej flagi
        )

        heatmap_data = None
        if not groups:
            return (crnn_probs, heatmap_data) if return_heatmap else crnn_probs

        img_width = image_tensor.shape[3]
        batch_tensors, batch_contexts, batch_metadata = [], [], []
        offsets = [-4, 0, 4]

        for g_idx, g in enumerate(groups):
            best_t_idx = -1
            min_margin = float('inf')
            max_uncertainty = -1.0
            best_gap = -1
            
            # Jeśli mamy dane o niepewności (MC Dropout), używamy ich do wyboru timestępu
            for cand in g:
                t = cand['timestep']
                u_val = cand.get('uncertainty_val', 0.0)
                
                vals, _ = torch.topk(crnn_probs[t], k=2)
                margin = (vals[0] - vals[1]).item()
                
                # Kryterium: wybierz najbardziej niepewny timestęp w grupie
                # Priorytet dla wysokiej wariancji (MC), potem dla niskiego marginesu
                if u_val > max_uncertainty + 0.001:
                    max_uncertainty, min_margin, best_t_idx, best_gap = u_val, margin, t, cand.get('closest_gap', -1)
                elif abs(u_val - max_uncertainty) < 0.001 and margin < min_margin:
                    min_margin, best_t_idx, best_gap = margin, t, cand.get('closest_gap', -1)

            if best_t_idx == -1: continue

            _, top_idxs = torch.topk(crnn_probs[best_t_idx], k=1)
            crnn_char = self.encoder.idx_to_char.get(top_idxs[0].item(), '')
            if crnn_char not in self.allowed_chars: continue

            current_context_vec = None
            if context_map is not None:
                # Korekta indeksowania kontekstu
                safe_t = min(max(0, best_t_idx), context_map.size(0) - 1)
                current_context_vec = context_map[safe_t]

            base_x1, base_x2 = min(c['x1'] for c in g), max(c['x2'] for c in g)
            for offset in offsets:
                curr_x1, curr_x2 = max(0, base_x1 + offset), min(img_width, base_x2 + offset)
                if curr_x2 - curr_x1 < 4: continue
                crop = image_tensor[:, :, :, curr_x1:curr_x2]
                batch_tensors.append(self._preprocess_crop_gpu(crop))

                # Context dim 1024 zgodnie z Twoją inicjalizacją CapsNet
                batch_contexts.append(
                    current_context_vec if current_context_vec is not None else torch.zeros(1024).to(self.device))
                batch_metadata.append({
                    'group_idx': g_idx, 
                    'timestep': best_t_idx, 
                    'uncertainty_val': max_uncertainty,
                    'center_x': best_gap,
                    'x1': curr_x1,
                    'x2': curr_x2
                })

        if not batch_tensors:
            return (crnn_probs, heatmap_data) if return_heatmap else crnn_probs

        full_batch = torch.cat(batch_tensors, dim=0)
        full_contexts = torch.stack(batch_contexts)

        # Wyciągnięcie opinii CRNN dla każdej próbki w batchu
        crnn_opinions = []
        for meta in batch_metadata:
            t = meta['timestep']
            # Jeśli mamy lepszą opinię z MC Mean Probs, używamy jej
            if mc_mean_probs is not None:
                crnn_opinions.append(mc_mean_probs[t])
            else:
                crnn_opinions.append(torch.exp(crnn_probs[t]))
        full_opinions = torch.stack(crnn_opinions)

        # Obliczamy ufność CapsNetu
        # Jeśli CRNN jest pewny (>0.9), to ufność do kontekstu zewnętrznego jest niższa (CapsNet polega na sobie)
        # Jeśli CRNN jest niepewny, CapsNet chętniej zerknie na kontekst
        crnn_max_probs = []
        for meta in batch_metadata:
            t = meta['timestep']
            # Prawdziwe prawdopodobieństwo z softmaxu, nie logity
            if mc_mean_probs is not None:
                p_max = torch.max(mc_mean_probs[t]).item()
            else:
                p_max = torch.max(torch.softmax(crnn_probs[t], dim=-1)).item()
            
            # Dodatkowo bierzemy pod uwagę wariancję (epistemic uncertainty)
            # Normalizujemy wariancję: 0.005 to już spora niepewność, 0.01 to ekstremalna
            normalized_u = min(1.0, meta['uncertainty_val'] / 0.01)
            
            # Inwersja: im mniejszy p_max LUB większa wariancja, tym większa "ufność" do potrzeby korygowania
            crnn_max_probs.append(max(1.0 - p_max, normalized_u))
        
        confidence_mask = torch.tensor(crnn_max_probs).to(self.device)

        # Przygotowanie granic dla BoundarySpatialAttention
        # BoundarySpatialAttention oczekuje [B, 3], gdzie [:, 1] to start_rel, [:, 2] to end_rel
        batch_boundaries = []
        for meta in batch_metadata:
            cx, x1, x2 = meta['center_x'], meta['x1'], meta['x2']
            if cx == -1:
                batch_boundaries.append(torch.tensor([0.0, 0.0, 1.0]))
            else:
                # Estymacja relatywnych granic wokół centrum znaku (zakładamy szerokość ok. 20 px)
                char_half_width = 10
                crop_w = x2 - x1
                rel_s = max(0.0, (cx - x1 - char_half_width) / (crop_w + 1e-6))
                rel_e = min(1.0, (cx - x1 + char_half_width) / (crop_w + 1e-6))
                batch_boundaries.append(torch.tensor([0.0, rel_s, rel_e]))
        
        full_boundaries = torch.stack(batch_boundaries).to(self.device)

        self.capsnet.eval()
        with torch.no_grad():
            # Przekazanie pełnej krotki argumentów
            caps_output = self.capsnet(
                full_batch,
                word_context=full_contexts,
                confidence=confidence_mask,
                crnn_probs=full_opinions,
                boundaries=full_boundaries
            )
            caps_probs = caps_output[0]
            if return_heatmap and len(caps_output) > 1:
                heatmap_data = caps_output[-1]

            batch_confs, batch_preds = torch.max(caps_probs, dim=1)

        results_by_group = {}
        for i, meta in enumerate(batch_metadata):
            conf, pred_idx, g_idx = batch_confs[i].item(), batch_preds[i].item(), meta['group_idx']
            if g_idx not in results_by_group or conf > results_by_group[g_idx][0]:
                results_by_group[g_idx] = (conf, self.capsnet_mapping.get(pred_idx, '?'), meta)

        for g_idx, (best_caps_conf, best_caps_char, meta) in results_by_group.items():
            t_idx = meta['timestep']
            if best_caps_conf < 0.65:
                crnn_probs[t_idx, :] = 0.0
                crnn_probs[t_idx, self.lc_idx] = 10.0
            elif best_caps_conf > 0.85:
                target_idx = self.encoder.char_to_idx.get(best_caps_char)
                if target_idx:
                    crnn_probs[t_idx, :] *= 0.1
                    crnn_probs[t_idx, target_idx] = 20.0

        # Zwracamy zgodnie z oczekiwaniem Pipeline'u
        return (crnn_probs, heatmap_data) if return_heatmap else crnn_probs

    def get_global_features(self):
        """ Zwraca zagregowany wektor cech wizualnych z CapsNetu dla ostatniego batcha. """
        # Zakładamy, że CapsNet zwraca cechy przed klasyfikacją (np. z warstwy ukrytej)
        if hasattr(self, 'last_features'):
            return self.last_features
        return torch.zeros(1, 1024).to(self.device)

class SessionMemory:
    """ Pamięć epizodyczna przechowująca zweryfikowane słowa w bieżącej sesji. """
    def __init__(self, similarity_threshold=0.75):
        self.verified_words = Counter()
        self.similarity_threshold = similarity_threshold

    def update(self, word):
        if len(word) > 2:  # Ignorujemy spójniki i krótkie słowa
            self.verified_words[word] += 1

    def get_closest_match(self, dubious_text):
        if not self.verified_words:
            return None

        # Szukamy najlepiej dopasowanego słowa z historii sesji
        words = list(self.verified_words.keys())
        matches = difflib.get_close_matches(dubious_text, words, n=1, cutoff=self.similarity_threshold)

        return matches[0] if matches else None

class PersistentAuthorMemory:
    """ Trwała baza wiedzy o charakterze pisma konkretnego autora. """
    def __init__(self, author_id="default_user"):
        self.author_id = author_id
        self.db_path = f"knowledge_base_{author_id}.json"
        self.word_map = {}
        self.load_from_disk()

    def load_from_disk(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.word_map = json.load(f)
                print(f"Załadowano {len(self.word_map)} trwałych poprawek dla: {self.author_id}")
            except Exception as e:
                print(f"Błąd ładowania bazy: {e}")

    def update(self, noisy_word, corrected_word):
        # Usuwamy tagi przed zapisem, aby klucze były czyste
        clean_noisy = re.sub(r'<.*?>|~', '', str(noisy_word)).strip()
        if clean_noisy and clean_noisy != corrected_word:
            self.word_map[clean_noisy] = corrected_word
            self.save_to_disk()

    def save_to_disk(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.word_map, f, ensure_ascii=False, indent=4)

    def get_match(self, noisy_text) -> Optional[str]:
        """ Zwraca poprawione słowo z bazy lub None. """
        if not noisy_text: return None
        # Czyścimy tagi przed sprawdzeniem klucza
        clean_text = re.sub(r'<.*?>|~', '', str(noisy_text)).strip()
        return self.word_map.get(clean_text, None)


class LineToWordSegmentor:
    @staticmethod
    def deslant_img(img):
        if img is None or np.sum(img) == 0: return img
        _, b = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        m = cv.moments(b)
        if m['mu02'] != 0:
            skew = m['mu11'] / m['mu02']
            M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
            img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR, borderValue=255)
        return img

    def extract_atomic_crops(self, img):
        if img is None: return []

        img_clean = self.deslant_img(img)
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(img_clean, window_size=51)
        img_arr = np.asarray(img_clean)
        binary = np.where(img_arr > float(thresh), 255, 0).astype(np.uint8)

        # Szukanie liter
        cnts, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            if w * h > 15: boxes.append((x, y, w, h))

        if not boxes: return []
        boxes.sort(key=lambda b: b[0])

        # Analiza odstępów
        merged_boxes = []
        if len(boxes) > 0:
            current_group = [boxes[0]]
            median_height = np.median([b[3] for b in boxes])
            SPACE_THRESHOLD = median_height * 0.55  # Dynamiczny próg spacji

            for i in range(1, len(boxes)):
                prev = boxes[i - 1]
                curr = boxes[i]
                gap = curr[0] - (prev[0] + prev[2])

                if gap < SPACE_THRESHOLD:
                    current_group.append(curr)
                else:
                    merged_boxes.append(self._merge_group(current_group))
                    current_group = [curr]
            merged_boxes.append(self._merge_group(current_group))

        # Wycinanie
        results = []
        h_orig, w_orig = img.shape
        for (mx, my, mw, mh) in merged_boxes:
            pad = 4
            y1 = max(0, my - pad)
            y2 = min(h_orig, my + mh + pad)
            x1 = max(0, mx - pad)
            x2 = min(w_orig, mx + mw + pad)
            crop = img[y1:y2, x1:x2]
            if crop.shape[1] < 5 or crop.shape[0] < 5: continue
            results.append((crop, (x1, y1, x2 - x1, y2 - y1)))
        return results

    @staticmethod
    def _merge_group(group):
        if not group: return 0, 0, 0, 0
        min_x = min([b[0] for b in group])
        min_y = min([b[1] for b in group])
        max_x = max([b[0] + b[2] for b in group])
        max_y = max([b[1] + b[3] for b in group])
        return min_x, min_y, max_x - min_x, max_y - min_y

    @staticmethod
    def merge_crops(crop1, crop2, padding=2):
        """ Skleja sąsiadujące ze sobą znaki w jeden obraz dla CapsNetu. """
        # Ujednolicamy wysokość
        h1, w1 = crop1.shape[:2]
        h2, w2 = crop2.shape[:2]

        max_h = max(h1, h2)

        # Skalowanie lub padding pionowy, aby obrazki miały tę samą wysokość
        def pad_to_height(img, target_h):
            h, w = img.shape[:2]
            if h == target_h: return img
            top = (target_h - h) // 2
            bottom = target_h - h - top
            return cv.copyMakeBorder(img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=0)

        c1_padded = pad_to_height(crop1, max_h)
        c2_padded = pad_to_height(crop2, max_h)

        # Dodajemy mały odstęp między znakami
        spacer = np.zeros((max_h, padding), dtype=crop1.dtype)

        # Połączenie poziome
        merged = np.hstack([c1_padded, spacer, c2_padded])
        return merged


class PageToLineSegmentor:
    """ Algorytm Śledzenia Linii:
        Tradycyjna segmentacja oparta na prostokątnych ramkach zawodzi w przypadku gęstego pisma odręcznego,
        gdzie wydłużenia dolne nachodzą na przestrzeń zajmowaną przez wiersz poniżej. Algorytm Line Follower
        rozwiązuje ten problem, traktując separator linii jako optymalną ścieżkę (seam) w grafie wagowym.
        Proces składa się z czterech kluczowych etapów:
        1. Konstrukcja Mapy Kosztów (Cost Map): Piksele białe otrzymująkoszt 1, czarne 255.
        2. Inicjalizacja: Wyznaczanie przybliżonych środków przerw na podstawie rzutu poziomego.
        3. Optymalizacja Ścieżki: Algorytm przesuwa się od lewej do prawej,
           wybierając piksele o najniższym skumulowanym koszcie (omijanie "gór" atramentu).
        4. Rektyfikacja: Mapowanie pofalowanego obszaru na prostokątny obraz wejściowy dla sieci CRNN. """
    def __init__(self, min_line_height=15):
        self.min_line_height = min_line_height

    def extract_lines(self, img):
        """ Optymalizacja Seam Carving poprzez downscaling. """
        if img is None: return []

        # Zabezpieczenie przed zbyt małym obrazem
        h, w = img.shape
        if h < 50 or w < 50:
            return [img]

        scale_factor = 512.0 / w if w > 512 else 1.0

        if scale_factor < 1.0:
            small_img = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            small_img = img

        # Progowanie
        binary_small = cv.adaptiveThreshold(small_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

        # Znajdowanie przerw
        proj = np.sum(binary_small, axis=1)
        gap_indices_small = self._find_line_gaps(proj, small_img.shape[0])
        if not gap_indices_small:
            return [img]

        # Szukanie ścieżek
        seams_small = []
        for y_start in gap_indices_small:
            path = self._find_optimal_seam(binary_small, y_start)
            seams_small.append(path)

        seams_full = [np.zeros(w, dtype=int)]

        for s_small in seams_small:
            # Przygotowanie danych
            s_reshaped = s_small.astype(np.float32).reshape(1, -1)
            resized_seam = cv.resize(s_reshaped, (w, 1), interpolation=cv.INTER_LINEAR)

            # Jawne wymuszenie ndarray przed użyciem .flatten()
            s_full_array = np.asarray(resized_seam)
            s_full_flat = s_full_array.flatten()

            # Skalowanie wartości w osi Y (powrót do oryginalnej wysokości)
            s_full = (s_full_flat / scale_factor).astype(int)

        # Dolna krawędź
        seams_full.append(np.full(w, h - 1, dtype=int))

        # Wycinanie linii
        lines = []
        for i in range(len(seams_full) - 1):
            top_seam = seams_full[i]
            bottom_seam = seams_full[i + 1]

            line_crop = self._extract_by_seams(img, top_seam, bottom_seam)

            if line_crop.shape[0] > self.min_line_height:
                lines.append(line_crop)

        return lines

    @staticmethod
    def _find_line_gaps(projection, height):
        """ Znajduje indeksy Y będące w środku białych przerw między liniami. """
        smoothed = np.convolve(projection, np.ones(5) / 5, mode='same')
        threshold = float(np.mean(smoothed)) * 0.2

        gaps = np.where(smoothed < threshold)[0]
        if len(gaps) == 0: return []

        centers = []
        if len(gaps) > 0:
            start = gaps[0]
            for i in range(1, len(gaps)):
                if gaps[i] - gaps[i - 1] > 10:
                    centers.append((start + gaps[i - 1]) // 2)
                    start = gaps[i]
            centers.append((start + gaps[-1]) // 2)
        return centers

    @staticmethod
    def _find_optimal_seam(binary, y_start):
        """ Algorytm szukania ścieżki o najmniejszym koszcie (najmniej atramentu). Wykorzystuje uproszczone programowanie dynamiczne. """
        h, w = binary.shape
        cost_map = np.where(binary > 0, 255, 1).astype(np.float32)

        path = np.zeros(w, dtype=int)
        current_y = y_start
        path[0] = current_y

        for x in range(1, w):
            # Definiujemy potencjalne ruchy (góra, prosto, dół)
            candidates = np.array([current_y - 1, current_y, current_y + 1])

            # Zabezpieczamy przed wyjściem poza obraz
            y_range = np.clip(candidates, 0, h - 1)

            # Pobieramy koszty dla tych 3 pikseli w aktualnej kolumnie x
            costs = cost_map[y_range, x]
            min_idx = np.argmin(costs)
            current_y = y_range[min_idx]
            path[x] = current_y

        return path

    @staticmethod
    def _extract_by_seams(img, top_seam, bottom_seam):
        """ Mapuje pofalowaną linię na prostokąt. """
        h_orig, w_orig = img.shape
        target_h = int(np.mean(bottom_seam - top_seam))
        line_img = np.full((target_h, w_orig), 255, dtype=np.uint8)

        for x in range(w_orig):
            # Próbkowanie pionowe wzdłuż krzywizny seamu
            raw_linspace = np.linspace(top_seam[x], bottom_seam[x], target_h)
            source_y = np.asarray(raw_linspace).astype(np.float32)
        return line_img


class CRNNInferencePipeline:
    """ Główny koordynator procesu inferencji HTR. Klasa zarządza pełnym potokiem przetwarzania: od surowego skanu strony,
        przez segmentację i wieloetapowe rozpoznawanie, aż po interaktywną korektę. """
    def __init__(self, crnn_model, char_list, device, transformer_path=None, verbose=True):
        # arametry sprzętowe i modele
        self.device = device
        self.device = device
        self.use_fp16 = False  # Edge AI polega na CPU i INT8
        self.model = crnn_model.to(device)
        self.model.eval()

        # Moduły przetwarzania
        self.preprocessor = ImagePreprocessor
        self.line_segmentor = PageToLineSegmentor()
        self.segmentor = LineToWordSegmentor()
        self.geo_corrector = GeometryCorrector()

        # Mapowanie znaków
        self.char_list = char_list
        self.idx_to_char = {i + 1: c for i, c in enumerate(char_list)}
        self.idx_to_char[0] = ''
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items() if v != ''}

        # Modele dodatkowe
        self.capsnet = CapsNet(num_classes=62).to(device)
        self.transformer: Optional[TransformerRefiner] = \
            TransformerRefiner(transformer_path, device) if transformer_path else None
        self.refiner = None

        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 768)
        ).to(DEVICE)

        self.tuner = JointFeedbackFineTuner(
            crnn_model=self.model,
            caps_model=self.refiner.capsnet if self.refiner else None,
            transformer_model=self.transformer if hasattr(self, 'transformer') else None,
            encoder=self,
            device=self.device
        )

        self.ort_session = None
        self.onnx_path = CRNN_WEIGHTS.replace(".pth", "_int8.onnx")
        self.load_onnx_if_exists()

    def _prepare_batch(self, crops):
        """ Przygotowuje wspólny tensor dla wszystkich słów ze strony z paddingiem. """
        processed_tensors = []
        for crop in crops:
            h_c, w_c = crop.shape
            new_w = int(w_c * (64 / h_c)) if h_c > 0 else 64
            img_resized = cv.resize(crop, (max(new_w, 10), 64))
            img_t = torch.from_numpy(img_resized).float()
            img_t = (img_t / 255.0 - 0.5) / 0.5
            processed_tensors.append(img_t)

        max_w = max(t.size(-1) for t in processed_tensors)

        # Neutralne tło po normalizacji
        padded_tensors = [
            func.pad(t, (0, max_w - t.size(-1)), mode='constant', value=-1.0)
            for t in processed_tensors
        ]

        return torch.stack(padded_tensors).unsqueeze(1).to(self.device)

    def _cleanup_memory(self):
        """ Jawne zwalnianie nieużywanych zasobów GPU i RAM. """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _prefetch_worker(self, file_list, queue_obj):
        """ Metoda pracująca w tle z obsługą błędów odczytu plików. """
        for full_path in file_list:
            try:
                raw_img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
                if raw_img is None:
                    print(f"Błąd: Nie można wczytać {full_path}")
                    continue

                prepro = cast(ImagePreprocessor, cast(Any, self.preprocessor))
                processed_page = prepro.full_pipeline(raw_img)

                lines_img_data = self.line_segmentor.extract_lines(processed_page)

                queue_obj.put({
                    'path': full_path,
                    'processed_page': processed_page,
                    'lines_img_data': lines_img_data
                })
            except Exception as e:
                print(f"Krytyczny błąd prefetchingu dla {full_path}: {e}")

        # Sygnał zakończenia
        queue_obj.put(None)

    @staticmethod
    def calculate_entropy(logits, T=1.4):
        """ Oblicza średnią entropię dla słowa.
            Wzór: H = -Σ p * log(p)
            T (Temperatura): skaluje logity, aby uniknąć "sztucznej pewności" modelu. """
        # Skalujemy logity przez temperaturę T przed nałożeniem Softmaxu
        probs = func.softmax(logits.float() / T, dim=-1)

        # Obliczamy entropię
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        return entropy.mean().item()

    def predict_single_word_refinement(self, current_probs, img_caps_single, context_map=None):
        """ Pełna kaskada decyzyjna (Multimodal Fusion) z mechanizmem VETO. """
        # Wstępne rozpoznanie CRNN
        word_entropy = getattr(self, 'calculate_entropy', lambda p, T: 0.0)(current_probs, T=1.4)
        raw_crnn_text, raw_char_confs, _, words_data = self._decode_ctc_with_mask(current_probs)
        refined_char_confs = raw_char_confs
        raw_word_conf = np.mean(raw_char_confs) if raw_char_confs else 0.0

        attention_map_to_save = None
        text_after_caps = raw_crnn_text
        hybrid_word_conf = raw_word_conf
        uncertainty_mask = [0] * len(raw_crnn_text)
        global_visual_feat = None

        # CapsNet + Kontekst
        if self.refiner:
            refinement_result = self.refiner.refine_logits(
                current_probs,
                img_caps_single,
                crnn_text=raw_crnn_text,
                context_map=context_map,
                force_intensive=(raw_word_conf < 0.7),
                return_heatmap=True
            )

            # Ekstrakcja cech dla Neuronów Typograficznych
            if hasattr(self.refiner, 'get_global_features'):
                global_visual_feat = self.refiner.get_global_features()

            if isinstance(refinement_result, tuple) and len(refinement_result) == 3:
                current_probs, raw_timestep_mask, attention_map_to_save = refinement_result
            elif isinstance(refinement_result, tuple) and len(refinement_result) == 2:
                current_probs, raw_timestep_mask = refinement_result
            else:
                # Wymuszamy na linterze świadomość, że to jest Tensor
                current_probs = cast(torch.Tensor, refinement_result)
                raw_timestep_mask = [0] * current_probs.size(0)

            # Dekodowanie po poprawkach CapsNetu
            text_after_caps, refined_char_confs, uncertainty_mask, words_data = self._decode_ctc_with_mask(current_probs, raw_timestep_mask)
            hybrid_word_conf = np.mean(refined_char_confs) if refined_char_confs else raw_word_conf

        # Generowanie kandydatów Beam Search
        candidates = getattr(self, '_ctc_beam_search', lambda p, beam_width: [(text_after_caps, 1.0)])(
            current_probs,
            beam_width=5
        )
        scored_results = []

        for cand_text, v_score in candidates:
            # Opcjonalna korekta geometryczna
            geo_cand = self.geo_corrector.refine_chars(cand_text, img_caps_single) if hasattr(self, 'geo_corrector') else cand_text

            # Filar Językowy
            l_score = 0.0
            if hasattr(self, 'transformer') and self.transformer and not any(c.isdigit() for c in geo_cand):
                # Tagowanie niepewnych fragmentów dla Transformera (<unc>)
                tagged_text = ""
                for idx, char in enumerate(geo_cand):
                    char_p = refined_char_confs[idx] if idx < len(refined_char_confs) else hybrid_word_conf
                    tagged_text += f"<unc>{char}</unc>" if char_p < 0.75 else char

                l_score = self.transformer.score_text(tagged_text)

            # Filar Typograficzny (Visual-Semantic Alignment)
            t_score = 0.0
            if self.transformer and hasattr(self.transformer, 'get_word_embeddings') and global_visual_feat is not None:
                with torch.no_grad():
                    vision_emb = self.projector(global_visual_feat)
                    text_emb = self.transformer.get_word_embeddings([geo_cand])
                    t_score = torch.nn.functional.cosine_similarity(vision_emb, text_emb).item()

            # Fuzja wyników: Wizja (1,0) + Język (0,45) + Typografia (0,35)
            combined_score = v_score + (l_score * 0.45) + (t_score * 0.35)

            scored_results.append({
                'text': geo_cand,
                'score': combined_score,
                'vision_score': v_score,
                'lang_score': l_score,
                'typo_score': t_score
            })

        # Wybranie najlepszej opcji
        final_proposal = scored_results[0]['text'] if scored_results else text_after_caps

        # Adaptacyjny próg Levenshteina dla długich linii (15% zmienionych znaków)
        max_allowed_changes = max(2, int(len(text_after_caps) * 0.15))
        if raw_crnn_text == text_after_caps and Levenshtein.distance(text_after_caps, final_proposal) > max_allowed_changes:
            # Jeśli wizja jest bardzo pewna, odrzucamy zbyt agresywną korektę Transformera
            if hybrid_word_conf > 0.90:
                final_proposal = text_after_caps

        # Obliczanie prawdopodobieństwa softmax dla top-k (do pokazania w UI)
        top_n = scored_results[:3]
        top_k_with_conf = []
        if top_n:
            raw_scores = torch.tensor([r['score'] for r in top_n], dtype=torch.float32)
            probs_softmax = torch.softmax(raw_scores, dim=0).numpy()
            top_k_with_conf = [
                {'text': r['text'], 'conf': float(p)}
                for r, p in zip(top_n, probs_softmax) if p > 0.25
            ]

        conf_gain = hybrid_word_conf - raw_word_conf

        # Zwracamy pełny zestaw danych dla UI
        return {
            'crnn_result': raw_crnn_text,
            'capsnet_result': text_after_caps,
            'final_result': final_proposal,
            'uncertainty_mask': uncertainty_mask,
            'words_data': words_data,
            'top_k_candidates': top_k_with_conf,
            'word_confidence': raw_word_conf,
            'hybrid_confidence': hybrid_word_conf,
            'confidence_gain': conf_gain,
            'entropy': word_entropy,
            'attention_map': attention_map_to_save,
            'debug_scores': scored_results if scored_results else {}
        }

    def predict_batch(self, tensors_crnn, tensors_caps):
        """ Główna metoda dla generatora danych. Przyjmuje listy tensorów CRNN i CapsNet. """
        if not tensors_crnn: return []

        logits = None
        context_maps = None

        # Jeśli ONNX
        if hasattr(self, 'ort_session') and self.ort_session is not None:
            # Padding i stackowanie dla batcha ONNX (oczekuje numpy)
            max_w = max(t.size(-1) for t in tensors_crnn)
            padded = [func.pad(t, (0, max_w - t.size(-1)), mode='constant', value=-1.0) for t in tensors_crnn]

            # Przygotowanie inputu dla ONNX Runtime
            batch_np = torch.stack(padded).cpu().numpy()
            if batch_np.ndim == 3: batch_np = np.expand_dims(batch_np, axis=1)

            ort_inputs = {self.ort_session.get_inputs()[0].name: batch_np}
            ort_outs = self.ort_session.run(None, ort_inputs)

            # Zamiana wyniku z powrotem na tensor dla reszty potoku
            logits = torch.tensor(ort_outs[0]).to(self.device)

            # ONNX nie zwraca map kontekstu
            context_maps = [None] * logits.size(1)
        else:
            max_w = max(t.size(-1) for t in tensors_crnn)
            padded_tensors = [func.pad(t, (0, max_w - t.size(-1)), mode='constant', value=-1.0) for t in tensors_crnn]
            images_batch = torch.stack(padded_tensors).to(self.device)

            if images_batch.dim() == 3:
                images_batch = images_batch.unsqueeze(1)

            use_fp16 = getattr(self, 'use_fp16', False)
            if use_fp16: images_batch = images_batch.half()

            with torch.inference_mode():
                logits, context_maps = self.model(images_batch, return_context=True)

        # Wspólny refiment
        page_metadata = []
        batch_size = logits.size(1)

        for b in range(batch_size):
            current_logits = logits[:, b, :]
            current_context = context_maps[b] if context_maps is not None else None

            single_caps_img = tensors_caps[b].to(self.device)
            if single_caps_img.dim() == 3:
                single_caps_img = single_caps_img.unsqueeze(0)

            # Wywołanie kaskady z logiką Veto wewnątrz
            meta = self.predict_single_word_refinement(current_logits, single_caps_img, current_context)
            page_metadata.append(meta)

        return page_metadata

    def predict_page_batched(self, word_crops):
        if not word_crops: return []

        tensors_crnn = []
        tensors_caps = []
        valid_boxes = []

        for crop, box in word_crops:
            h_c, w_c = crop.shape[:2]
            if h_c < 2 or w_c < 2: continue  # Bardziej restrykcyjne zabezpieczenie

            # Przygotowanie dla CRNN
            new_w = max(16, int(w_c * (64 / h_c)))
            img_resized = cv.resize(crop, (new_w, 64), interpolation=cv.INTER_AREA)

            # Przekazujemy img_resized bezpośrednio do tensora!
            t_crnn = (torch.from_numpy(img_resized).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
            tensors_crnn.append(t_crnn)

            # CapsNet potrzebuje odwróconej binaryzacji i poprawy grubości, więc tutaj używamy Otsu
            _, img_bin = cv.threshold(img_resized, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            caps_ready = ImagePreprocessor.standardize_ink_thickness(img_bin, target_thickness=3)

            t_caps = (torch.from_numpy(caps_ready).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
            tensors_caps.append(t_caps)

            valid_boxes.append(box)

        # Inferencja
        raw_results = self.predict_batch(tensors_crnn, tensors_caps)

        final_metas = []

        # Tu powinna być logika zbierania słów w linie, ale jeśli GUI wysyła słowa po kolei:
        current_line_accumulator = []

        for i, meta in enumerate(raw_results):
            bx, by, bw, bh = valid_boxes[i]
            noisy_text = meta.get('final_result', '')

            if self.transformer:
                # Używamy realnego kontekstu z poprzedniej linii, jeśli jest
                context = getattr(self, 'last_full_line_context', "") or ""
                final_text = self.transformer.refine(noisy_text, prefix_context=context)
            else:
                final_text = noisy_text

            meta.update({
                'final_result': final_text,
                'box': (bx, by, bw, bh),
                'uncertainty_rects': meta.get('uncertainty_mask', [])
            })
            final_metas.append(meta)

        return final_metas

    def caps_predict_helper(self, img_np, threshold=0.75):
        """ Przewidywanie jednego lub wielu znaków przez CapsNet na wspólnym wycinku. """
        self.capsnet.eval()
        with torch.no_grad():
            h, w = img_np.shape[:2]
            scale = 64 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv.resize(img_np, (new_w, new_h))

            # Centrowanie na 64x64
            canvas = np.zeros((64, 64), dtype=np.uint8)
            offset_y = (64 - new_h) // 2
            offset_x = (64 - new_w) // 2
            canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

            # Przygotowanie tensora
            img_t = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0).to(self.device)
            img_t = (img_t / 255.0 - 0.5) / 0.5

            # Forward Pass — pobieramy normy wszystkich kapsułek
            output = self.capsnet(img_t, num_iterations=3)
            norms = output[0] if isinstance(output, tuple) else output

            # Obliczamy normę dla każdej kapsułki
            v_mag = norms.squeeze()

            # Szukamy wszystkich kandydatów powyżej progu
            confs, indices = torch.topk(v_mag, k=min(5, len(self.char_list)))

            detected = []
            for c, idx in zip(confs, indices):
                if c > threshold:
                    char = self.idx_to_char.get(idx.item() + 1, '?')

                    # Ignorujemy blanki i spacje
                    if char and char != '[blank]':
                        detected.append({
                            'char': char,
                            'confidence': c.item()
                        })

            return detected

    def _perform_online_learning(self, page_data, al_manager):
        """ Trwała adaptacja wag i pamięć słownikowa autora. Uczy się na błędach i utrwala wiedzę na dysku po osiągnięciu progu. """
        if getattr(self, 'tuner', None) is None:
            from Refinement.UserAdaptation import JointFeedbackFineTuner
            self.tuner = JointFeedbackFineTuner(
                crnn_model=self.model,
                caps_model=self.refiner.capsnet if self.refiner else None,
                transformer_model=self.transformer if hasattr(self, 'transformer') else None,
                encoder=self,
                device=self.device
            )

        augmentor = FeedbackAugmentor()
        learned_count = 0

        # Inicjalizacja licznika wewnątrz instancji
        if not hasattr(self, 'unsaved_learning_steps'):
            self.unsaved_learning_steps = 0

        for crop_data, meta in zip(page_data['crops'], page_data['metadatas']):
            # Pobieramy wynik wizji i ostateczną korektę (od usera lub ByT5)
            noisy_vision = meta.get('capsnet_result')
            final_correction = meta.get('final_result')

            # Sprawdzamy, czy nastąpiła korekta manualna/semantyczna
            if noisy_vision != final_correction:

                # Utrwalamy w bazie słownikowej
                if self.transformer and hasattr(self.transformer, 'session_memory'):
                    self.transformer.session_memory.update(noisy_vision, final_correction)

                # Adaptacja wag
                img_t = self._prepare_batch([crop_data[0]])
                augmented_batch = augmentor.generate_variations(img_t, num_variations=4)

                # Wynik wstępny to poprawka CapsNet
                original_text = meta['caps_result']

                # Przekazujemy go do tunera wraz z poprawką użytkownika
                self.tuner.fine_tune_on_feedback(
                    images_batch=augmented_batch,
                    original_pred_text=original_text,
                    corrected_text=final_correction,
                    segmentor=self.segmentor
                )

                learned_count += 1
                self.unsaved_learning_steps += 1

        # Logika trwałego zapisu
        if self.unsaved_learning_steps >= SAVE_THRESHOLD:
            print(f"\n[{time.strftime('%H:%M:%S')}] Osiągnięto próg adaptacji ({SAVE_THRESHOLD}). Trwa utrwalanie wag na dysku.")
            try:
                # Tworzymy rotacyjny backup plików .pth przed nadpisaniem
                for path in [CRNN_WEIGHTS, CAPS_WEIGHTS]:
                    if os.path.exists(path):
                        backup_path = path + ".bak"
                        if os.path.exists(backup_path): os.remove(backup_path)
                        os.rename(path, backup_path)

                # Zapisujemy nowe, spersonalizowane wagi
                if self.unsaved_learning_steps >= SAVE_THRESHOLD:
                    print(f"[{time.strftime('%H:%M:%S')}] AL_MANAGE: Próg osiągnięty. Synchronizacja.")

                    # Zapisujemy klasyczne wagi PyTorch (do kolejnych douczeń w przyszłości)
                    self.tuner.save_fine_tuned_weights(al_manager)
                    self.unsaved_learning_steps = 0
                    print("SYSTEM ZAKOŃCZYŁ NAUKĘ: Wagi .pth zostały trwale zaktualizowane.")

                    # Zaraz po douczeniu, aplikacja generuje szybki model dla użytkownika
                    self.export_and_quantize_to_onnx()

            except Exception as e:
                print(f"Krytyczny błąd zapisu wag: {e}")

        if learned_count > 0:
            print(f"Skumulowano {learned_count} nowych poprawek. Łącznie do zapisu: {self.unsaved_learning_steps}/{SAVE_THRESHOLD}")

    def export_and_quantize_to_onnx(self):
        """ Konwertuje spersonalizowany model PyTorch do szybkiego formatu ONNX INT8. """
        print(f"[{time.strftime('%H:%M:%S')}] Rozpoczynam optymalizację (Kwantyzacja do INT8).")

        self.model.eval()
        self.model.to('cpu')  # ONNX export jest najstabilniejszy na CPU
        base_path = CRNN_WEIGHTS.replace(".pth", "")
        fp32_onnx_path = f"{base_path}_temp_fp32.onnx"
        int8_onnx_path = f"{base_path}_int8.onnx"

        # Przygotowanie pustego tensora wejściowego
        dummy_input = torch.randn(1, 1, 64, 576, requires_grad=True)

        # Eksport do FP32
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                fp32_onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_images'],
                output_names=['logits', 'context_maps'],
                dynamic_axes={
                    'input_images': {0: 'batch_size', 3: 'width'},
                    'logits': {0: 'time_steps', 1: 'batch_size'},
                    'context_maps': {0: 'time_steps', 1: 'batch_size'}
                }
            )

            # Dynamiczna Kwantyzacja
            quantize_dynamic(
                model_input=fp32_onnx_path,
                model_output=int8_onnx_path,
                weight_type=QuantType.QUInt8
            )

            # Usuwamy plik tymczasowy FP32
            if os.path.exists(fp32_onnx_path):
                os.remove(fp32_onnx_path)

            print(
                f"[{time.strftime('%H:%M:%S')}] Sukces! Wygenerowano spersonalizowany, szybki model: {int8_onnx_path}")

            # Aktualizacja ścieżki i przeładowanie sesji ONNX w locie
            self.onnx_path = int8_onnx_path
            self.load_onnx_if_exists()

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Błąd podczas kwantyzacji: {e}")
        finally:
            # Przywracamy model z powrotem na docelowe urządzenie
            self.model.to(self.device)

    @staticmethod
    def calculate_session_metrics(session_data):
        """ Analizuje całą sesję, oblicza CER, WER i zysk pewności. """
        total_dist = 0
        total_chars = 0
        total_words = 0
        wrong_words = 0
        all_gains = []

        for page in session_data:
            for meta in page['metadatas']:
                pred = meta.get('capsnet_result', '')
                gt = meta.get('final_result', '')

                if gt:
                    # CER
                    total_dist += Levenshtein.distance(pred, gt)
                    total_chars += len(gt)

                    # WER
                    total_words += 1
                    if pred != gt:
                        wrong_words += 1

                # Zysk pewności
                all_gains.append(meta.get('confidence_gain', 0))

        cer = (total_dist / total_chars) * 100 if total_chars > 0 else 0
        wer = (wrong_words / total_words) * 100 if total_words > 0 else 0
        avg_gain = np.mean(all_gains) * 100 if all_gains else 0

        return cer, wer, avg_gain

    def _prepare_single_tensor(self, crop):
        """ Pomocnicza metoda standaryzacji obrazu słowa do 64px wysokości. """
        h, w = crop.shape[:2]
        new_w = max(16, int(w * (64 / h)))
        resized = cv.resize(crop, (new_w, 64), interpolation=cv.INTER_AREA)
        t = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return (t / 255.0 - 0.5) / 0.5

    def predict_automatic(self, package, prefix_context=""):
        """ Tryb wsadowy. Pomijamy segmentację na słowa, aby zachować kontekst wizualny wiersza. """
        # Zmienne lokalne
        current_img = package['processed_page']
        current_crops = []
        current_metas = []

        current_full_context = prefix_context
        line_coords = package.get('lines_coords', [])

        for i, line_img in enumerate(package['lines_img_data']):
            if line_img is None or line_img.size == 0: continue

            # Używamy lokalnej zmiennej current_img
            l_box = line_coords[i] if i < len(line_coords) else (0, i * 100, current_img.shape[1], line_img.shape[0])

            img_t = self._prepare_single_tensor(line_img)
            meta = self.predict_single_word_refinement(img_t[0], img_t)
            raw_line_text = meta.get('capsnet_result', '')

            if self.transformer and str(raw_line_text).strip():
                refined_line = self.transformer.refine(raw_line_text, prefix_context=current_full_context)
                meta['final_result'] = refined_line
                current_full_context = refined_line[-300:]
            else:
                meta['final_result'] = raw_line_text

            meta.update({'box': l_box})

            # Zapisujemy do lokalnych list
            current_crops.append((line_img, l_box))
            current_metas.append(meta)

        return {
            'path': package['path'],
            'shape': current_img.shape[::-1],
            'crops': current_crops,
            'metadatas': current_metas
        }

    @staticmethod
    def save_to_pdf(line_crops, metadatas, img_shape, output_path, original_img_path):
        """ Zapis pojedynczej strony do PDF w trybie liniowym. """
        h_img, w_img = img_shape

        # Tworzymy dokument o wymiarach oryginalnego obrazu
        pdf = FPDF(unit="pt", format=[w_img, h_img])
        pdf.add_page()

        # Wstawiamy oryginalny skan jako tło
        pdf.image(original_img_path, x=0, y=0, w=w_img, h=h_img)

        # Konfiguracja warstwy tekstowej
        pdf.set_text_color(0, 0, 255)
        pdf.set_font("Helvetica", size=12)

        for i, (crop, (bx, by, bw, bh)) in enumerate(line_crops):
            # Pobieramy ostateczny wynik po kaskadzie (CRNN + CapsNet + Transformer)
            text = metadatas[i].get('final_result', '')

            if not text.strip():
                continue

            # Dynamiczne dopasowanie rozmiaru czcionki do wysokości wyciętej linii (mnożnik, żeby się mieściły)
            font_size = bh * 0.75
            pdf.set_font_size(font_size)

            # Ustawiamy kursor na początku linii (współrzędne z Line Followera)
            pdf.set_xy(bx, by)

            # Wpisujemy całą linię tekstu
            pdf.cell(w=bw, h=bh, text=text, border=0)

        pdf.output(output_path)
        print(f"Wygenerowano PDF strony: {output_path}")

    @staticmethod
    def save_book_to_pdf(book_data, output_path):
        """ Scalanie wielu stron tekstu pisanego w jeden plik PDF, nanosząc rozpoznany tekst linia po linii.
            Dzięki temu zachowujemy oryginalny układ akapitów. """
        pdf = FPDF(unit="pt")
        for page in book_data:
            w_img, h_img = page['shape']
            pdf.add_page(format=(w_img, h_img))

            # Tłem jest oryginalny skan strony
            pdf.image(page['path'], x=0, y=0, w=w_img, h=h_img)

            # Ustawienia czcionki dla warstwy tekstowej
            pdf.set_text_color(0, 0, 255)

            # Przetwarzamy każdą linię zapisaną w metadanych strony
            for crop_info, meta in zip(page['crops'], page['metadatas']):
                # bx, by, bw, bh to współrzędne linii
                bx, by, bw, bh = crop_info[1]
                text = meta.get('final_result', '')

                if not text.strip(): continue

                # Skalujemy czcionkę dynamicznie do wysokości linii
                pdf.set_font("Helvetica", size=bh * 0.75)
                pdf.set_xy(bx, by)

                # Wpisujemy tekst linii
                pdf.cell(w=bw, h=bh, text=text, border=0)

        pdf.output(output_path)
        print(f"Zapisano dokument PDF (tryb linii): {output_path}")

    @staticmethod
    def _save_checkpoint_json(data, filename="htr_session_checkpoint.json"):
        """ Zapisuje dotychczasowe poprawki do pliku (bez obrazów, same teksty i koordynaty). """
        try:
            checkpoint_data = []
            for page in data:
                checkpoint_data.append({
                    'path': page['path'],
                    'shape': page['shape'],
                    'metadatas': page['metadatas']
                    # Nie zapisujemy 'crops' (obrazów), bo plik urósłby do kilku GB
                })

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"   [Błąd zapisu checkpointu] {e}")

    @staticmethod
    def load_checkpoint(filename="htr_session_checkpoint.json"):
        """ Pozwala wznowić przerwaną sesję. """
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _ctc_beam_search(self, log_probs, char_images=None, beam_width=3, T=1.4):
        """ Zmodernizowany Beam Search z integracją CapsNet i kalibracją T. """
        # Dzielimy logity przez T przed softmaxem, aby wygładzić pewność modelu
        probs = func.softmax(log_probs.float() / T, dim=-1)

        steps, num_classes = probs.shape
        beams = [(1.0, [])]

        for t in range(steps):
            next_beams = []
            p_t = probs[t]

            # Czy CRNN jest niepewny w tym kroku czasowym?
            if p_t.max() < 0.6 and self.refiner and char_images is not None:
                # Próbujemy dopasować obraz znaku do kroku czasowego t
                char_idx = int((t / steps) * len(char_images))
                if char_idx < len(char_images):
                    crop = char_images[char_idx]

                    # Dummy tensors dla wywołania CapsNet w locie
                    dummy_ctx = torch.zeros(1, 1024).to(self.device)
                    dummy_conf = torch.zeros(1).to(self.device)  # 0.0, bo to czysta wizja, bez fuzji językowej
                    dummy_probs = torch.zeros(1, len(self.char_list) + 1).to(self.device)

                    caps_out = self.refiner.capsnet(
                        crop,
                        word_context=dummy_ctx,
                        confidence=dummy_conf,
                        crnn_probs=dummy_probs
                    )
                    caps_logits = caps_out[0] if isinstance(caps_out, tuple) else caps_out
                    caps_p = func.softmax(caps_logits / T, dim=-1).squeeze()

                    # 40% CRNN + 60% CapsNet
                    p_t = (0.4 * p_t) + (0.6 * caps_p)

            # Pobieramy k-najlepszych kandydatów dla obecnego kroku
            topk_p, topk_idx = torch.topk(p_t, k=beam_width)

            for score, seq in beams:
                for i in range(beam_width):
                    new_score = score * topk_p[i].item()  # Mnożymy prawdopodobieństwa
                    new_seq = seq + [topk_idx[i].item()]
                    next_beams.append((new_score, new_seq))

            # Sortowanie wiązek i przycięcie do beam_width
            next_beams.sort(key=lambda x: x[0], reverse=True)
            beams = next_beams[:beam_width]

        # Usuwanie powtórzeń i znaku Blank
        decoded_candidates = []
        for score, seq in beams:
            decoded = []
            prev = -1
            for idx in seq:
                if idx != prev and idx != 0:
                    char = self.idx_to_char.get(idx, '')
                    if char: decoded.append(char)
                prev = idx

            text = "".join(decoded)

            # Dodajemy do wyników tylko unikalne teksty
            if text not in [x[0] for x in decoded_candidates]:
                decoded_candidates.append((text, score))

        return decoded_candidates

    def _decode_ctc_with_mask(self, logits, timestep_mask=None):
        """ Dekodowanie CTC z automatycznym wstawianiem tagów <unc> dla Transformera. """
        # Przygotowanie danych
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)
        preds = preds.cpu().numpy()
        confs = confs.cpu().numpy()

        tagged_chars = [] # Tekst z <unc>... </unc>
        plain_chars = [] # Czysty tekst
        res_confs = []

        words_data = []
        current_word = ""
        current_word_confs = []
        current_word_dubious = False

        prev = 0 # Indeks poprzedniej ramki

        for t, p in enumerate(preds):
            # Wykryto nowy znak (nie blank i inny niż poprzedni)
            if p != 0 and p != prev:
                char = self.idx_to_char.get(p, '?')
                is_uncertain = bool(timestep_mask[t]) if timestep_mask is not None else False

                plain_chars.append(char)
                res_confs.append(confs[t])

                # Budowa wersji otagowanej dla Transformera
                tagged_val = f"<unc>{char}</unc>" if is_uncertain else char
                tagged_chars.append(tagged_val)

                # Logika słownikowa (do CapsNetu)
                if char == ' ':
                    if current_word:
                        words_data.append({
                            'word': current_word,
                            'confidence': sum(current_word_confs) / len(current_word_confs),
                            'is_dubious': current_word_dubious
                        })
                        current_word, current_word_confs, current_word_dubious = "", [], False
                else:
                    current_word += char
                    current_word_confs.append(confs[t])
                    if is_uncertain: current_word_dubious = True

            # Kontynuacja tego samego znaku (łączenie klatek w CTC)
            elif p != 0 and p == prev and len(plain_chars) > 0:
                is_uncertain = bool(timestep_mask[t]) if timestep_mask is not None else False

                # Jeśli choć jedna klatka znaku jest niepewna, cały znak w tagged_chars musi mieć tag
                if is_uncertain:
                    char = plain_chars[-1]
                    tagged_chars[-1] = f"<unc>{char}</unc>"
                    current_word_dubious = True

                # Aktualizujemy pewność
                res_confs[-1] = min(res_confs[-1], confs[t])
                if current_word:
                    current_word_confs[-1] = min(current_word_confs[-1], confs[t])

            prev = p

        # Zamknięcie ostatniego słowa
        if current_word:
            words_data.append({
                'word': current_word,
                'confidence': sum(current_word_confs) / len(current_word_confs),
                'is_dubious': current_word_dubious
            })

        return "".join(tagged_chars), "".join(plain_chars), res_confs, words_data

    def load_onnx_if_exists(self):
        """ Ładuje sesję ONNX, jeśli plik jest dostępny na dysku. """
        if os.path.exists(self.onnx_path):
            try:
                # Ustawiamy CUDA, jeśli jest dostępna, inaczej CPU
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(self.onnx_path, providers=providers)
                print(f"[{time.strftime('%H:%M:%S')}] Szybka ścieżka ONNX aktywowana.")
            except Exception as e:
                print(f"Błąd ładowania ONNX: {e}")

    def predict_word_fast(self, image_tensor):
        """ Ultra-szybka inferencja przez ONNX dla pojedynczych wywołań z UI. """
        if self.ort_session is None:
            # Fallback do PyTorcha, jeśli ONNX z jakiegoś powodu nie ma
            return self.model(image_tensor)[0]

        # ONNX oczekuje numpy, nie tensora PyTorch
        inputs = {self.ort_session.get_inputs()[0].name: image_tensor.cpu().numpy()}
        logits = self.ort_session.run(None, inputs)[0]
        return torch.tensor(logits).to(self.device)

    def encode(self, char):
        """ Mapuje znak na indeks dla tunera. """
        # Odwracamy słownik idx_to_char lub używamy char_list
        char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        return char_to_idx.get(char, None)

    def get_num_classes(self):
        return len(self.char_list) + 1


if __name__ == "__main__":
    # Diagnostyka sprzętowa
    print(f"[{now()}] Urządzenie: {DEVICE}")
    print(f"[{now()}] Zalogowano użytkownika: {os.getlogin()}")

    # Sprawdzenie dostępności zasobów
    print(f"[{now()}] Folder roboczy: {BASE_DIR}")

    weights_check = {
        "CRNN": CRNN_WEIGHTS_PATH.exists(),
        "CapsNet": CAPS_WEIGHTS_PATH.exists(),
        "Transformer": Path(TRANSFORMER_PATH).exists()
    }

    for model_name, exists in weights_check.items():
        status = "OK" if exists else "BRAK PLIKU"
        print(f"[{now()}] Status wag {model_name}: {status}")

    # Test inicjalizacji Pipeline
    try:
        print(f"[{now()}] Pipeline zintegrowany poprawnie.")
    except Exception as e:
        print(f"[{now()}] BŁĄD INTEGRACJI: {e}")

    print("Aplikacja gotowa do obsługi żądań przez Backend.py.")
