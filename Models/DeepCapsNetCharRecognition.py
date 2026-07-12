import gc
import os
import random
import warnings
import time
import glob
from datetime import datetime
import numpy as np
import seaborn as sns
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision.transforms import v2
from torchvision.ops import DeformConv2d
from tqdm import tqdm
import h5py
import cv2 as cv
cv.setNumThreads(0)
from PIL import Image, UnidentifiedImageError
from torch.utils.data import WeightedRandomSampler
import json
import shapiq
import xml.etree.ElementTree as elTree
import pandas as pd
from torchvision.transforms.v2 import functional as f_v2
from Models.ResNetCRNNWordRecognition import ResNetCRNN, get_safe_char_name
import difflib
from Preprocessing.Preprocessing import Preprocessing
warnings.filterwarnings("ignore")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 16
ACCUMULATION_STEPS = 12
WEIGHT_DECAY = 0.0
SMOOTHING = 0.1
STEPS_PER_EPOCH = 7500
VAL_STEPS_LIMIT = 500

""" MAIN (z dodatkiem PURE)
    Zaczynamy na silnie zaugmentowanych danych łączących eMNIST oraz realne wycinki PURE z CRNN.
    Celem jest budowa szerokiej bazy cech wizualnych i wstępna adaptacja do domeny docelowej od pierwszej epoki.
    Dzięki hybrydzie model uczy się tekstury połączonego pisma, mając jednocześnie geometryczny punkt odniesienia. """
MAIN_EPOCHS = 8
INITIAL_LR = 1e-4
MAIN_WORKERS = 4

""" FINE TUNE (wyostrzenie modelu na czystym eMNIST)
    Powrót do czystego zbioru eMNIST (tylko idealne wzorce). Celem jest Stabilizacja wag i wyprostowanie filtrów.
    Po kontakcie z szumem w fazie MAIN, model przypomina sobie idealną geometrię znaków, co zapobiega overfittingowi
    do artefaktów preprocessingu. """
FINE_TUNE_EPOCHS = 6
FINE_TUNE_LR = 5e-5
FINE_WORKERS = 2

""" HARD MINING (włączanie decyzyjności modelu)
    Trening na wycinkach PURE oraz zidentyfikowanych błędach CRNN przy zamrożonym backbone.
    Celem jest przełamywanie minimów lokalnych dla krytycznych par znaków (np. g/q, rn/m, S/5). 
    Zamrożenie bazy wizualnej pozwala na wyższy impuls, który re-kalibruje wyłącznie mechanizm 
    routingu i głowice pod specyficzne pomyłki kaskady, chroniąc model przed katastroficznym zapominaniem. """
HARD_MINING_EPOCHS = 8
HARD_MINING_LR = 1e-4
HARD_WORKERS = 0

LR_USER_CAPS = 1e-6  # Bardzo mały LR, by dociągnąć model, nie psując go
ADAPT_EPOCHS = 4

DEVICE = torch.device('cuda')

# BASE_DIR mapuje na C:\OCR przez wolumen /app/data
BASE_DIR = "/app/data"

# OUTPUT_ROOT mapuje na ./output_data przez wolumen /app/output_data
OUTPUT_ROOT = "/app/output_data"

# Ścieżki do modeli i debugowania
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints", "hcr")
DEBUG_DIR = os.path.join(OUTPUT_ROOT, "debug_chars")
MODEL_NAME = "CharLevelCapsNet.pth"

# Referencja do wag CRNN (potrzebna do etapu kaskady/hard miningu)
CRNN_CHECKPOINT = os.path.join(OUTPUT_ROOT, "checkpoints", "hwr", "CRNN_Aligned_for_CapsNet.pth")

# Dane wejściowe (zrelatywizowane do punktu montowania BASE_DIR)
DATA_ROOT_EMNIST = BASE_DIR
DATA_ROOT_PUNCTUATION = os.path.join(BASE_DIR, "punctuation")
CRNN_CUSTOM_SAMPLES = "/app/output_data/crnn_crops"
PHSF_DATA_PATH = os.path.join(BASE_DIR, "dataset.npz")

# Ścieżka do macierzy pomyłek (Confusion Matrix)
MATRIX_PATH = os.path.join(OUTPUT_ROOT, "confusion_matrix_final")

# Normalizacja - standardowa dla eMNIST, używana przez CapsNet
EMNIST_NORM_MEAN = [0.1307]
EMNIST_NORM_STD = [0.3081]
FINAL_NORM = v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)

# Inicjalizacja folderów (Docker sam je utworzy na Twoim dysku Windows)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

def val_process_fn(image: np.ndarray, **kwargs) -> np.ndarray:
    """ Wrapper wywołujący logikę preprocessingu (deskewing/centering). """
    return Preprocessing.process_for_crnn(np.asarray(image, dtype=np.uint8), target_h=64)

VAL_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Lambda(val_process_fn), # Geometryczna normalizacja
    v2.Resize(IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR, antialias=True), # Chroni przed wygładzaniem małych znaków
    v2.ToDtype(torch.float32, scale=True), # Skalowanie [0, 1]
    v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD) # Globalne statystyki eMNIST
])

def now():
    """ Zwraca aktualną godzinę w formacie dla logów. """
    return time.strftime('%H:%M:%S')


def safe_collate_fn(batch):
    """ Łączy próbki w paczki, filtrując uszkodzone dane (None).
        Obsługuje niejednorodność danych (np. stałe wymiary obrazów eMNIST vs
        zmienne długości etykiet tekstowych), zapewniając stabilność ładowania
        dla modeli CRNN i CapsNet. """
    transposed = list(zip(*batch))

    result = []
    for field in transposed:
        first_item = field[0]

        # Jeśli to Tensor - sprawdźmy, czy wszystkie mają ten sam kształt
        if isinstance(first_item, torch.Tensor):
            shapes = [f.shape for f in field]
            if all(s == shapes[0] for s in shapes):
                result.append(torch.stack(field))  # Składamy w jeden tensor
            else:
                result.append(list(field))  # Różne rozmiary (np. zmienna szerokość) -> lista

        # Jeśli to liczby (np. indeksy, długości), to zamieniamy na prosty tensor
        elif isinstance(first_item, (int, float)):
            result.append(torch.tensor(field))

        # Napisy, listy (geoms), słowniki - zostają listami
        else:
            result.append(list(field))

    return result


def calculate_stats(loader):
    """ Oblicza średnią i odchylenie standardowe pikseli dla całego zbioru danych.
        Zabezpiecza również przed dzieleniem przez zero w przypadku całkowicie pustego zbioru. """
    sum_ = torch.tensor(0.0)
    sum_sq = torch.tensor(0.0)
    total_pixels = 0

    for batch in tqdm(loader, desc="Obliczanie statystyk", leave=False, disable=False):
        # Batch z HardCharsDataset to krotka, gdzie images to batch[0]
        if batch is None or len(batch) == 0:
            continue

        images, *metadata = batch
        total_pixels += images.numel()

        sum_ += torch.sum(images).cpu()
        sum_sq += torch.sum(images ** 2).cpu()

    # Zabezpieczenie przed dzieleniem przez zero, gdyby wszystkie paczki były puste
    if total_pixels == 0:
        return 0.0, 1.0

    mean = sum_ / total_pixels
    var = (sum_sq / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean.item(), std.item()

# Służy do surowego odczytu danych przed wyliczeniem normalizacji
base_transform = v2.Compose([

    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Resize(IMAGE_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])


def prepare_emnist_sample(img, degrees=-90):
    """ Korekta orientacji eMNIST. Łączy rotację, odbicie lustrzane i transpozycję w jedną operację. """
    if torch.is_tensor(img):
        return img.transpose(-1, -2)
    img = f_v2.rotate(img, degrees)
    return f_v2.hflip(img)


def get_base_transforms(grayscale=True):
    """ Zwraca listę podstawowych transformacji wizyjnych. W skali szarości, z antyaliasingiem. """
    # Jawne otypowanie listy jako list[nn.Module]
    layers: list[nn.Module] = [v2.ToImage()]

    if grayscale:
        layers.append(v2.Grayscale(num_output_channels=1))

    layers.append(v2.Resize(IMAGE_SIZE, antialias=True))

    return layers

# EMNIST Trening
EMNIST_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True), #Uint8 dla JPEG
    v2.Lambda(prepare_emnist_sample),
    v2.Resize(IMAGE_SIZE, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.RandomChoice([v2.Identity(), v2.JPEG(quality=(30, 70))]),
        v2.ToDtype(torch.float32, scale=True), # Przechodzimy na float
    v2.RandomPerspective(distortion_scale=0.2, p=0.4),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    v2.RandomApply([v2.ElasticTransform(alpha=25.0, sigma=4.0)], p=0.3),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
    v2.GaussianNoise(mean=0.0, sigma=0.03),
    FINAL_NORM
])

# EMNIST Test / Walidacja
emnist_test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Lambda(prepare_emnist_sample),
    v2.Resize(IMAGE_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    FINAL_NORM
])

# Wycinki CRNN do fazy Hard Mining
crnn_crop_transform = v2.Compose([
    *get_base_transforms(grayscale=True),
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    v2.ElasticTransform(alpha=50.0, sigma=5.0),
    v2.RandomRotation(degrees=10, fill=0),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=0),
    FINAL_NORM
])

def pil_to_numpy(img_pil):
    """" Konwertuje obraz w formacie PIL (Python Imaging Library) na tablicę NumPy. """
    return np.array(img_pil, dtype=np.uint8)


def get_zone_mapping(encoder):
    """ Tworzy mapowanie stref geometrycznych znaków na podstawie dekodera klas.
        0 - dla wielkich liter i znaków specjalnych (strefa górna/pełna),
        2 - dla liter opadających pod linię bazową (np. g, j, p, q, y, przecinek),
        1 - dla pozostałych, domyślnie małych liter (strefa środkowa). """
    num_classes = encoder.get_num_classes()
    mapping = {i: 1 for i in range(num_classes)}

    # Dodajemy interpunkcję, angielskie (łacińskie ?) i polskie litery do odpowiednich stref
    upper_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789bdfhklt!\"'?:()@#$%^&*ŁŚŹŻĆŃÓ"
    lower_chars = "gjpqy,.;ąę"

    for i in range(num_classes):
        char = encoder.decode(i)
        if char in upper_chars:
            mapping[i] = 0
        elif char in lower_chars:
            mapping[i] = 2
    return mapping


def load_phsf_mapping(json_path):
    """ Zwraca rozszerzone mapowanie znaków zbioru PHSF. """
    with open(json_path, 'r', encoding='utf-8') as f:
        phcd_dict = json.load(f)

    mapping = {}
    for char, idx in phcd_dict.items():
        mapping[int(idx)] = char

    return mapping


class PHSFDataset(Dataset):
    def __init__(self, npz_path, encoder, transform=None):
        self.encoder = encoder
        self.transform = transform
        self.samples = []

        # Tabela mapowania (przeniesiona do self, aby była dostępna w metodach)
        self.phsf_mapping = {
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            '10': 'A', '11': 'a', '12': 'B', '13': 'b', '14': 'C', '15': 'c', '16': 'D', '17': 'd', '18': 'E', '19': 'e',
            '20': 'F', '21': 'f', '22': 'G', '23': 'g', '24': 'H', '25': 'h', '26': 'I', '27': 'i', '28': 'J', '29': 'j',
            '30': 'K', '31': 'k', '32': 'L', '33': 'l', '34': 'M', '35': 'm', '36': 'N', '37': 'n', '38': 'O', '39': 'o',
            '40': 'P', '41': 'p', '42': 'Q', '43': 'q', '44': 'R', '45': 'r', '46': 'S', '47': 's', '48': 'T', '49': 't',
            '50': 'U', '51': 'u', '52': 'V', '53': 'v', '54': 'W', '55': 'w', '56': 'X', '57': 'x', '58': 'Y', '59': 'y',
            '60': 'Z', '61': 'z', '62': 'Ą', '63': 'ą', '64': 'Ć', '65': 'ć', '66': 'Ę', '67': 'ę', '68': 'Ł', '69': 'ł',
            '70': 'Ń', '71': 'ń', '72': 'Ó', '73': 'ó', '74': 'Ś', '75': 'ś', '76': 'Ź', '77': 'ź', '78': 'Ż', '79': 'ż',
            '80': '!', '81': ',', '82': '.', '83': ':', '84': ';', '85': '?', '86': '-', '87': '(', '88': ')'
        }

        if not os.path.exists(npz_path):
            print(f"[OSTRZEŻENIE] Brak pliku PHSF: {npz_path}.")
            return

        print(f"[{now()}] Ładowanie danych PHSF z {npz_path}.")
        try:
            data = np.load(npz_path, allow_pickle=True)
            raw_images = data.get('signs', data.get('images', data.get('arr_0')))
            raw_labels = data.get('labels', data.get('arr_1'))
        except Exception as e:
            print(f"[BŁĄD] Problem z odczytem archiwum PHSF: {e}")
            return

        valid_samples = 0
        skipped = {}

        for img, raw_label in zip(raw_images, raw_labels):
            label_str = str(raw_label).strip()

            # LOGIKA ROZSTRZYGANIA:
            # 1. Jeśli jest w mapowaniu (stare dane), pobierz znak.
            # 2. Jeśli nie ma w mapowaniu, załóż, że to już jest znak (np. 'P').
            actual_char = self.phsf_mapping.get(label_str, label_str)

            # Pobranie indeksu z encodera
            label_idx = self.encoder.char_to_idx.get(actual_char)
            
            if label_idx is None:
                skipped[actual_char] = skipped.get(actual_char, 0) + 1
                continue

            self.samples.append((img, label_idx))
            valid_samples += 1
            
        print(f"[{now()}] PHSF: Załadowano {valid_samples} próbek.")
        if skipped:
            print(f"Najczęstsze pominięte: {dict(list(skipped.items())[:5])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        # Wymuszenie polaryzacji
        img = img.astype(np.float32)
        if np.mean(img) > 127:
            img = 255.0 - img

        # Konwersja na PIL
        img_pil = Image.fromarray(img.astype(np.uint8), mode='L')

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = (torch.tensor(np.array(img_pil), dtype=torch.float32) / 255.0).unsqueeze(0)

        num_classes = self.encoder.get_num_classes()

        return (
            img_tensor,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),            
            torch.zeros(7, dtype=torch.float32),          
            torch.zeros(1024, dtype=torch.float32),       
            torch.tensor(1.0, dtype=torch.float32),       
            torch.zeros(num_classes, dtype=torch.float32),
            torch.tensor(0, dtype=torch.long)             
        )
      

class CharLabelEncoder:
    """ Klasa odpowiedzialna za konwersję między znakami tekstowymi a ich indeksami rozszerzona o język polski. """
    def __init__(self):
        polish_chars = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ" # Dodajemy polskie znaki diakrytyczne (18 znaków)
        combined_list = " !\"'(),-./:;?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + polish_chars
        raw_chars = sorted(list(set([c for c in combined_list if c != ''])))

        self.char_list = ['[blank]'] + sorted(list(raw_chars))
        self.idx_to_char = {i: c for i, c in enumerate(self.char_list)}
        self.char_to_idx = {c: i for i, c in enumerate(self.char_list)}

    def encode(self, char):
        """ Zwraca indeks dla pojedynczego znaku (używane przy PHCF). """
        return self.char_to_idx.get(char)

    def encode_sequence(self, text):
        """ Konwertuje napis na listę indeksów (używane przy IAM Words). """
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, idx_seq, ignore_blanks=True):
        """ Dekoduje indeksy na tekst. Obsługuje zarówno pojedyncze liczby, jak i sekwencje. """
        # Jeśli dostaliśmy pojedynczy int, po prostu zwróć odpowiadający mu znak
        if isinstance(idx_seq, int):
            return self.idx_to_char.get(idx_seq, '?')

        # Jeśli to Tensor, zamień na listę
        if isinstance(idx_seq, torch.Tensor):
            # Jeśli Tensor jest skalarem (0-wymiarowy)
            if idx_seq.dim() == 0:
                return self.idx_to_char.get(idx_seq.item(), '?')
            idx_seq = idx_seq.tolist()

        # Logika dla sekwencji (np. wyjście z CRNN)
        res = []
        for i, idx in enumerate(idx_seq):
            char = self.idx_to_char.get(idx, '')
            if char == '[blank]' and ignore_blanks:
                continue
            # Logika CTC: pomijaj powtórzenia znaków obok siebie
            if i > 0 and idx == idx_seq[i - 1]:
                continue
            res.append(char)

        return "".join(res)

    def get_num_classes(self):
        return len(self.char_list)


def save_debug_snapshots(model, loader, device, encoder, epoch):
    """ Pobiera jedną paczkę danych walidacyjnych i generuje siatkę wizualizacji predykcji. Tytuły nad kafelkami
        porównują predykcję z wartością rzeczywistą, oznaczając poprawne rozpoznania na zielono, a błędy na czerwono. """
    model.eval()
    try:
        images, labels, zones, geoms, *rest = next(iter(loader))
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = outputs["probs"] if isinstance(outputs, dict) else outputs[0]
            _, preds = torch.max(probs, 1)

        num_imgs = min(len(images), 16)
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        mean, std = EMNIST_NORM_MEAN, EMNIST_NORM_STD

        for i, ax in enumerate(axes.flatten()):
            if i >= num_imgs: break
            img = images[i].cpu().squeeze().numpy()
            img = (img * std[0]) + mean[0]
            img = np.clip(img, 0, 1)
            p = encoder.decode(preds[i])
            t = encoder.decode(labels[i])
            color = 'green' if p == t else 'red'
            ax.imshow(img, cmap='gray')
            ax.set_title(f"P:{p} T:{t}", color=color, fontweight='bold', fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(DEBUG_DIR, f"epoch_{epoch}_snapshot.png"))
        plt.close()
    except Exception as e:
        print(f"Snapshot error: {e}")


def visual_test(model, loader, device, encoder, num_images=15):
    """ Wizualizacja predykcji CapsNet.
        Wyświetla obraz, przewidziany znak, strefę oraz informację o poprawności. """
    model.eval()
    zone_names = ["Upper", "Middle", "Lower"]

    try:
        batch = next(iter(loader))
        inputs, targets, zones, boundaries, *rest = [
            b.to(device) if torch.is_tensor(b) else b for b in batch
        ]
    except Exception as e:
        print(f"Błąd przy ładowaniu oceny wizualnej: {e}")
        return

    with torch.no_grad():
        # Czyste wywołanie wizualne
        outputs = model(inputs, boundaries=boundaries)
        
        classes_norms = outputs["norms"]
        z_logits = outputs["z_logits"]

        # Predykcja głównego znaku
        _, preds = torch.max(classes_norms, 1)

        # Predykcja strefy
        if z_logits is not None:
            _, zone_preds = torch.max(z_logits, 1)
        else:
            zone_preds = torch.zeros(inputs.size(0), dtype=torch.long, device=device)

    # Konfiguracja wyświetlania
    num_to_show = min(num_images, len(inputs))
    rows = (num_to_show - 1) // 5 + 1
    plt.figure(figsize=(18, 4 * rows))

    # Parametry denormalizacji eMNIST
    mean, std = EMNIST_NORM_MEAN[0], EMNIST_NORM_STD[0]

    for i in range(num_to_show):
        plt.subplot(rows, 5, i + 1)

        # Denormalizacja do poprawnego wyświetlania grayscale
        img = inputs[i].cpu().squeeze().numpy()
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        pred_char = encoder.decode(preds[i])
        true_char = encoder.decode(targets[i])
        pred_zone = zone_names[zone_preds[i]]

        # Kolor zielony dla trafień, czerwony dla pomyłek
        color = 'green' if pred_char == true_char else 'red'

        plt.imshow(img, cmap='gray')
        plt.title(f"P: {pred_char} ({pred_zone})\nT: {true_char}",
                  color=color, fontweight='bold', fontsize=11)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


class HardCharsDataset(Dataset):
    """ Dataset zoptymalizowany pod kątem oszczędności RAM i szybkości inicjalizacji (Fast I/O). """
    def __init__(self, root_dir, encoder, case_type="hard", transform=None):
        self.root_dir = root_dir
        self.encoder = encoder
        self.transform = transform
        self.samples = []

        if not os.path.exists(root_dir):
            print(f"[{now()}] Ścieżka {root_dir} nie istnieje.")
            return

        # Używamy os.scandir, które jest znacznie szybsze w Dockerze niż os.listdir
        with os.scandir(root_dir) as it_root:
            for entry in it_root:
                if not entry.is_dir(): continue
                char_folder = entry.name

                # Próba bezpośredniego dopasowania
                label_idx = self.encoder.char_to_idx.get(char_folder)
                
                # Jeśli nie ma, spróbujmy obsłużyć wersję z "_cap"
                if label_idx is None:
                    base_name = char_folder.replace("_cap", "")
                    label_idx = self.encoder.char_to_idx.get(base_name)
                
                if label_idx is None: 
                    continue

                target_path = os.path.join(entry.path, case_type)
                if not os.path.isdir(target_path): continue

                # Błyskawiczne skanowanie docelowych plików
                with os.scandir(target_path) as it_target:
                    for f_entry in it_target:
                        if f_entry.name.lower().endswith(('.png', '.jpg')):
                            img_path = f_entry.path
                            npy_path = os.path.splitext(img_path)[0] + ".npz"

                            # Weryfikacja przeniesiona do __getitem__
                            self.samples.append((img_path, npy_path, label_idx))
        
        print(f"[{now()}] Załadowano {len(self.samples)} próbek ze zbioru {case_type.upper()}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, npy_path, label = self.samples[idx]
        num_classes = len(self.encoder.char_to_idx)

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            return (
                torch.zeros((1, 64, 64), dtype=torch.float32),
                torch.tensor(-1, dtype=torch.long),
                torch.tensor(1, dtype=torch.long),
                torch.zeros(7, dtype=torch.float32),
                torch.zeros(1024, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
                torch.zeros(num_classes, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long)
            )
        
        img = img.astype(np.float32) / 255.0
        if np.mean(img) > 0.5:
            img = 1.0 - img

        # Aplikacja transformacji razem z normalizacją
        if self.transform:
            img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Blok bezpiecznego ładowania metadanych w locie
        try:
            data = np.load(npy_path, allow_pickle=True).item()
        except:
            data = {}
            
        context_vec = torch.tensor(data.get('context_vector', np.zeros(1024)), dtype=torch.float32)
        
        crnn_probs_raw = data.get('crnn_probs', np.zeros(num_classes))
        if len(crnn_probs_raw) != num_classes:
            padded_probs = np.zeros(num_classes, dtype=np.float32)
            min_len = min(len(crnn_probs_raw), num_classes)
            padded_probs[:min_len] = crnn_probs_raw[:min_len]
            crnn_probs = torch.tensor(padded_probs, dtype=torch.float32)
        else:
            crnn_probs = torch.tensor(crnn_probs_raw, dtype=torch.float32)

        lang_id = 0 if ("PHSF" in img_path or "pl_line" in img_path) else 1

        return (
            img_tensor,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(1, dtype=torch.long), 
            torch.zeros(7, dtype=torch.float32), 
            context_vec,
            torch.tensor(1.0, dtype=torch.float32), 
            crnn_probs,
            torch.tensor(lang_id, dtype=torch.long)
        )


def squash(x, dim=-1):
        """ Nieliniowość 'Squashing' specyficzna dla sieci kapsułkowych.
            Działa jak funkcja normalizująca, która:
            1. Skaluje długość krótkich wektorów do bliskiej 0 (brak pewności).
            2. Skaluje długość długich wektorów do bliskiej 1 (wysoka pewność).
            3. Zachowuje kierunek wektora (to kluczowe, bo kierunek koduje cechy znaku, np. rotację). """
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        norm = torch.sqrt(squared_norm + 1e-8)
        scale = squared_norm / (1 + squared_norm)
        return scale * (x / norm)


class PrimaryCaps(nn.Module):
    """ Warstwa Primary Capsules. Konwertuje cechy splotowe na wektory, wzbogacając je o mapy współrzędnych X i Y
        w celu zachowania informacji o położeniu przestrzennym. """
    def __init__(self, in_channels=256, out_channels=32, dim_caps=8, kernel_size=5, stride=2):
        super(PrimaryCaps, self).__init__()
        # Dodajemy + 2 do in_channels, bo w forward robimy torch.cat ze współrzędnymi
        self.conv2d = nn.Conv2d(
            in_channels=in_channels + 2,
            out_channels=out_channels * dim_caps,
            kernel_size=kernel_size,
            stride=stride
        )
        self.dim_caps = dim_caps

    def forward(self, x):
        batch_size, _, h, w = x.size()

        # Tworzenie mapy współrzędnych
        grid_x = torch.linspace(-1, 1, w, device=x.device)
        grid_y = torch.linspace(-1, 1, h, device=x.device)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        # Rozszerzanie do wymiarów batcha
        coords = torch.stack([mesh_x, mesh_y], dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        x = torch.cat([x, coords], dim=1)
        outputs = self.conv2d(x)
        outputs = outputs.view(batch_size, -1, self.dim_caps)

        return squash(outputs)


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block — selekcja istotnych kanałów. """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(channels // reduction),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """ Blok rezydualny zintegrowany z modułem SE. Łączy ekstrakcję cech przez sploty z uwagą kanałową,
        wykorzystując połączenie skrótowe dla lepszej propagacji gradientu. """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.prelu2 = nn.PReLU(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        # Używamy zwykłego dodawania, które tworzy nowy tensor
        out = out + identity
        return self.prelu2(out)


class SpatialAttention(nn.Module):
    """ Pomaga sieci CapsNet skupić się na istotnych morfologicznie częściach znaku, ignorując szum tła. """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        mask = self.sigmoid(res)
        # Zwracamy wynik i maskę atencji
        return x * mask, mask


class AttentionRouting(nn.Module):
    """ Attention-based Capsule Routing (AR-Caps).
        Zastępuje powolną pętlę iteracyjną Saboura szybkim mechanizmem atencji,
        który od razu oblicza wagi dopasowania na podstawie iloczynu skalarnego. """
    def __init__(self, dim_caps=32):
        super().__init__()
        self.scale = (dim_caps ** 0.5)

    def forward(self, u_hat, num_iterations=None):
        # Skalarne mnożenie wektorów u_hat z ich uśrednionym kontekstem
        context = u_hat.mean(dim=1, keepdim=True)
        scores = (u_hat * context).sum(dim=-1, keepdim=True) / self.scale

        # Softmax po wymiarze kapsuł wyjściowych
        c_ij = torch.softmax(scores, dim=2)

        # Tylko ważona suma, bez squasha wewnątrz routingu (zapobiega ucinaniu gradientu)
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=False)

        return s_j
    

class DeformableBlock(nn.Module):
    """ Blok z konwolucją deformowalną. Uczy się przesunięć, aby dopasować siatkę do krzywizny znaku. """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeformableBlock, self).__init__()
        # Warstwa przewidująca przesunięcia pikseli
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1, stride=stride)

        # Właściwa konwolucja deformowalna
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        offsets = self.offset_conv(x)
        out = self.dcn(x, offsets)
        out = self.bn(out)
        return self.prelu(out)


class DualAttentionBlock(nn.Module):
    """ Moduł podwójnej uwagi (Channel oraz Spatial) do selekcji cech przed wektoryzacją. """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Uwaga kanałowa z zabezpieczeniem przed spadkiem liczby kanałów poniżej 1 w warstwach wąskich
        mid_channels = max(1, channels // reduction)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Uwaga przestrzenna
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Aplikacja uwagi kanałowej
        x_ca = x * self.channel_attention(x)

        # Wyciągnięcie zagęszczonych cech do uwagi przestrzennej
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        spatial_in = torch.cat([max_pool, avg_pool], dim=1)

        # Aplikacja uwagi przestrzennej
        x_out = x_ca * self.spatial_attention(spatial_in)
        return x_out

class GVFBalloonFilter(nn.Module):
    """ Moduł rekonstrukcji uszkodzonych znaków wykorzystujący fizykę aktywnych konturów. Dopełnia niedociągnięte
        pociągnięcia, co jest ważne dla CapsNet, analizującego cechy geometryczne, przypisując je do konkretnych kapsułek
        reprezentujących parametry instancji. Zapewnia to spójność topologiczną znaku, nawet w przypadku silnej
        degradacji lub przerwania ciągłości pociągnięć na obrazie wejściowym. """
    def __init__(self, iterations=5, mu=0.1, balloon_force=0.05):
        super().__init__()
        self.iterations = iterations
        self.mu = mu
        self.kappa = balloon_force

        # Filtry Sobela do wyliczania gradientów krawędzi
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Operator Laplace'a do wygładzania pola sił (dyfuzja)
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

    def forward(self, x):
        with torch.no_grad():

            # Obliczenie mapy krawędzi i ich siły
            fx = func.conv2d(x, self.sobel_x, padding=1)
            fy = func.conv2d(x, self.sobel_y, padding=1)
            f_mag_sq = fx**2 + fy**2

            # Inicjalizacja wektorów pola sił GVF
            u = torch.zeros_like(x)
            v_vec = torch.zeros_like(x)

            # Iteracyjna propagacja wektorów gradientu
            for _ in range(self.iterations):
                u_lap = func.conv2d(u, self.laplacian, padding=1)
                v_lap = func.conv2d(v_vec, self.laplacian, padding=1)

                u = u + self.mu * u_lap - f_mag_sq * (u - fx)
                v_vec = v_vec + self.mu * v_lap - f_mag_sq * (v_vec - fy)

            # Obliczenie dywergencji pola GVF
            div = func.conv2d(u, self.sobel_x, padding=1) + func.conv2d(v_vec, self.sobel_y, padding=1)

            # Aplikacja Siły Balonu
            restored = x + self.kappa * torch.sign(div) * (1.0 - f_mag_sq)

        # Ograniczenie wartości tensorów powraca do grafu głównego
        return torch.clamp(restored, min=x.min(), max=x.max())


class FastMorphologyFilter(nn.Module):
    """ Błyskawiczny zamiennik GVF. Domyka przerwane pociągnięcia atramentu używając MaxPool2d. """
    def __init__(self, kernel_size=3):
        super().__init__()
        # Dylatacja (pogrubienie jasnych pikseli, czyli w naszym odwróconym obrazie - atramentu)
        self.dilate = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        #  Dylatacja, żeby połączyć przerwane linie
        dilated = self.dilate(x)

        # Lekkie wygładzenie krawędzi
        smoothed = func.avg_pool2d(dilated, kernel_size=3, stride=1, padding=1)

        # 70% oryginału + 30% domkniętych luk
        return torch.clamp(x * 0.7 + smoothed * 0.3, 0, 1)


class AffineSTN(nn.Module):
    """ Lekki moduł afiniczny dedykowany dla pojedynczych wyciętych znaków. """
    def __init__(self, in_channels=1, input_size=64):
        super(AffineSTN, self).__init__()

        # Prosta sieć lokalizująca
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            # Uodpornienie: 32 kanały * (64//4) * (64//4) = 32 * 16 * 16 = 8192
            nn.Linear(8192, 64),
            nn.ReLU(False)
        )

        # Regresor parametrów afinicznych
        self.fc_loc = nn.Linear(64, 6)

        # Inicjalizacja tożsamościowa — STN na starcie nie zmienia obrazu
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        # Opcjonalne zabezpieczenie przed ekstremalnym skalowaniem pojedynczej litery
        theta[:, 0, 0] = torch.clamp(theta[:, 0, 0], min=0.7, max=1.3)
        theta[:, 1, 1] = torch.clamp(theta[:, 1, 1], min=0.7, max=1.3)

        grid = func.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = func.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        return x_transformed


class CapsNet(nn.Module):
    """ Implementacja Capsule Network dedykowana do rozpoznawania trudnych/niejednoznacznych znaków.
        1. GVF Restoration: Wstępna, fizyczna naprawa przerwanych pociągnięć znaków.
        2. Feature Extractor: Zmodyfikowany ResNet z blokami Squeeze-and-Excitation.
        3. Primary Capsules: Przekształca skalarne mapy cech z CNN na wektorowe kapsułki.
        4. Attention Routing: Mechanizm atencji zamiast Dynamic Routing.
        5. Multi-task Learning Heads: Głowice pomocnicze wspierające weryfikację kształtu. """
    def  __init__(self, num_classes=89, context_dim=1024):
        super().__init__()
        self.num_classes = num_classes

        self.stn = AffineSTN(in_channels=1, input_size=64)
        self.gvf_restoration = FastMorphologyFilter()

        # Ekstraktor cech
        self.backbone = nn.Sequential(
            ResidualBlock(1, 64, stride=1),
            DualAttentionBlock(64),
            ResidualBlock(64, 128, stride=1),
            DeformableBlock(128, 256, stride=2),
            DualAttentionBlock(256)
        )

        # PrimaryCaps — tworzy 1024 kapsuły
        self.primary = PrimaryCaps(in_channels=256, out_channels=16, dim_caps=8, kernel_size=9, stride=3)
        self.num_primary = 1024

        # Dzielimy wejście na 16 niezależnych podprzestrzeni
        self.num_groups = 16
        self.caps_per_group = self.num_primary // self.num_groups

        # Macierz wag (Złota strefa inicjalizacji: 0.1)
        self.W = nn.Parameter(torch.randn(1, self.num_groups, self.caps_per_group, num_classes, 32, 8) * 0.1)

        # Routing z użyciem mechanizmu atencji
        self.routing = AttentionRouting(dim_caps=32)

        # Głowice pomocnicze
        self.dropout = nn.Dropout(p=0.2)
        self.zone_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(64, 3)
        )
        self.geom_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(False), nn.Linear(64, 7))

        # Dekoder rekonstrukcyjny
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(False),
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(False),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(False),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(False),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y=None, word_context=None, confidence=None, crnn_probs=None, boundaries=None,
                num_iterations=3, force_dropout=False, lang_id=None) -> dict:
        """ Pełny przepływ w przód czystego wizualnie modelu CapsNet. """
        batch_size = x.size(0)

        if force_dropout:
            self.train()
        else:
            self.eval()

        x_restored = self.gvf_restoration(x)
        x_aligned = self.stn(x_restored)
        features = self.backbone(x_aligned)
        
        # Cechy z ResNetu trafiają bezpośrednio do kapsułek
        u = self.primary(features)
        batch_size = u.size(0)

        u_grouped = u.view(batch_size, self.num_groups, self.caps_per_group, 1, -1, 1)
        u_hat = torch.matmul(self.W, u_grouped).squeeze(-1)
        u_hat_reshaped = u_hat.view(batch_size * self.num_groups, self.caps_per_group, self.num_classes, 32)

        v_j_grouped = self.routing(u_hat_reshaped, num_iterations=num_iterations)

        v_j_unflattened = v_j_grouped.view(batch_size, self.num_groups, self.num_classes, 32)
        s_final = v_j_unflattened.sum(dim=1)

        norm_squared = (s_final ** 2).sum(dim=-1, keepdim=True)
        scale = norm_squared / (1 + norm_squared)
        v_j = scale * s_final / torch.sqrt(norm_squared + 1e-9)

        classes_norms = torch.sqrt((v_j ** 2).sum(dim=-1) + 1e-9)

        probs = classes_norms.clamp(0.0, 1.0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        if y is None:
            y_indices = classes_norms.argmax(dim=-1)
        else:
            # Jeśli y to indeksy (0-88), to po prostu użyj y
            y_indices = y

        masked_v = v_j[torch.arange(batch_size), y_indices]
        
        # Aplikujemy dropout dla regularyzacji dekodera i głowic
        stochastic_v = self.dropout(masked_v)

        # Głowice pomocnicze i dekoder korzystają z wektora z regularyzacją (stochastic_v)
        reconstruction = self.decoder(stochastic_v)
        z_logits = self.zone_head(stochastic_v)
        g_stats = self.geom_head(stochastic_v)

        return {
            "norms": classes_norms,
            "probs": probs,
            "entropy": entropy,
            "reconstruction": reconstruction,
            "z_logits": z_logits,
            "g_stats": g_stats,
            "capsules": v_j
        }

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "backbone.0" in name or "backbone.1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Backbone częściowo odmrożony (Deformable + Attention).")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print(f"[{now()}] Cały model odmrożony.")

    def generate_shapiq_explanation(self, image_tensor: torch.Tensor, target_class: int, grid_size: tuple = (4, 8), budget: int = 512):
        self.eval()
        device = image_tensor.device

        B, C, H, W = image_tensor.shape
        n_players = grid_size * grid_size[1]
        patch_h = H // grid_size
        patch_w = W // grid_size[1]

        def model_predict_wrapper(masks: np.ndarray) -> np.ndarray:
            num_coalitions = masks.shape
            scores = list()

            with torch.no_grad():
                batch_size = 32
                for i in range(0, num_coalitions, batch_size):
                    mask_batch = masks[i:i + batch_size]
                    current_bs = mask_batch.shape

                    masked_imgs = torch.zeros(current_bs, C, H, W, device=device)

                    for b in range(current_bs):
                        for row in range(grid_size):
                            for col in range(grid_size[1]):
                                player_idx = row * grid_size[1] + col
                                if mask_batch[b, player_idx] == 1:
                                    masked_imgs[b, :, row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w] = image_tensor[0, :, row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w]

                    outputs = self.forward(masked_imgs)
                    norms = outputs["norms"] if isinstance(outputs, dict) else outputs
                    batch_scores = norms[:, target_class].cpu().numpy()
                    scores.extend(batch_scores)

            return np.array(scores)

        approximator = shapiq.ProxySPEX(n=n_players, index="k-SII", max_order=2)
        interaction_values = approximator.approximate(budget=budget, game=model_predict_wrapper)

        return interaction_values

    def predict_char(self, crop_tensor, encoder, word_context=None, confidence=None, boundaries=None, crnn_probs=None, mc_samples=10):
        self.eval()
        all_probs = []
        all_capsules = []

        with torch.no_grad():
            for _ in range(mc_samples):
                features = self.backbone(self.stn(self.gvf_restoration(crop_tensor)))

                u = self.primary(features)
                batch_size = u.size(0)

                u_grouped = u.view(batch_size, self.num_groups, self.caps_per_group, 1, -1, 1)
                u_hat = torch.matmul(self.W, u_grouped).squeeze(-1)
                u_hat_reshaped = u_hat.view(batch_size * self.num_groups, self.caps_per_group, self.num_classes, 32)

                v_j_grouped = self.routing(u_hat_reshaped, num_iterations=3)

                v_j_unflattened = v_j_grouped.view(batch_size, self.num_groups, self.num_classes, 32)
                s_final = v_j_unflattened.sum(dim=1)

                norm_squared = (s_final ** 2).sum(dim=-1, keepdim=True)
                scale = norm_squared / (1 + norm_squared) / torch.sqrt(norm_squared + 1e-8)
                v_j = scale * s_final

                # Usunięto całkowicie moduł fuzji

                norms = (v_j ** 2).sum(dim=-1) ** 0.5
                probs = norms.clamp(0.0, 1.0)

                all_probs.append(probs)
                all_capsules.append(v_j)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        avg_capsules = torch.stack(all_capsules).mean(dim=0)

        epistemic_variance = torch.stack(all_probs).var(dim=0).mean()

        top_probs, top_indices = torch.topk(avg_probs, 2, dim=-1)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        if top_indices.dim() == 1:
            winning_idx = top_indices[0].item()
            conf = top_probs[0].item()
            margin = (top_probs[0] - top_probs[1]).item() if top_probs.numel() > 1 else 0.0
        else:
            winning_idx = top_indices[0, 0].item()
            conf = top_probs[0, 0].item()
            margin = (top_probs[0, 0] - top_probs[0, 1]).item() if top_probs.size(1) > 1 else 0.0

        winning_capsule_vector = avg_capsules[0, winning_idx, :] if avg_capsules.dim() == 3 else avg_capsules[winning_idx, :]

        return {
            'char': encoder.decode(winning_idx),
            'confidence': conf,
            'margin': margin,
            'entropy': entropy.item(),
            'epistemic_unc': epistemic_variance.item(),
            'capsule_embedding': winning_capsule_vector
        }

class CapsuleFusionDataset(torch.utils.data.Dataset):
    """ Zestaw danych dla modułu Capsule Network w architekturze fuzji. Wczytuje wycięte obrazy znaków powiązane
        z metadanymi wyeksportowanymi przez model CRNN, przygotowując je do ostatecznej weryfikacji geometrycznej. """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Pobieramy, ignorując 'dummy'
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != 'dummy'])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for target_class in self.classes:
            class_path = os.path.join(root_dir, target_class)
            # Szukamy tylko plików PNG
            for f in os.listdir(class_path):
                if f.endswith('.png'):
                    img_path = os.path.join(class_path, f)
                    # Zakładamy, że .npz ma tę samą nazwę co .png
                    npy_path = img_path.replace('.png', '.npz')
                    if os.path.exists(npy_path):
                        self.samples.append((img_path, npy_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, npy_path, label = self.samples[idx]

        # Wczytujemy obraz
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image=image)['image'] if hasattr(self.transform, 'call') else self.transform(image)

        # Wczytujemy kontekst (Deep Fusion)
        metadata = np.load(npy_path, allow_pickle=True).item()
        context_vec = torch.tensor(metadata['context'], dtype=torch.float32)

        return image, context_vec, label


def get_crnn_error_weights(char_true, char_pred, encoder, top_n=20):
    """ Oblicza wagi błędów CRNN dla CapsNet w celu skupienia się na trudnych literach. """
    errors = [t for t, p in zip(char_true, char_pred) if t != p and t != '[pusty]']

    if not errors:
        return torch.ones(encoder.get_num_classes())

    # Liczymy pomyłki dla każdego znaku
    error_counts = pd.Series(errors).value_counts()

    # Wybieramy top_n najtrudniejszych znaków
    top_errors = error_counts.head(top_n).index.tolist()

    # Tworzymy tensor wag
    weights = torch.ones(encoder.get_num_classes())
    for char in top_errors:
        idx = encoder.encode(char)
        if idx is not None:
            # Zwiększamy wagę dla trudnych znaków o 50%
            weights[idx] = 2.0

    return weights


class SSIMLoss(nn.Module):
    """ Implementacja Structural Similarity do rekonstrukcji znaków. """
    def __init__(self, window_size=11, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    @staticmethod
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return nn.Parameter(_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)

    def forward(self, img1, img2):
        self.window = self.window.to(img1.device)
        mu1 = func.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = func.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = func.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = func.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = func.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class CapsNetDecoder(nn.Module):
    """ Dekoder rekonstrukcyjny dla sieci CapsNet. Odgrywa rolę regularyzatora wymuszającego na kapsułkach
        kodowanie wizualnych cech znaków, niezbędnych do odtworzenia oryginalnego obrazu. Do sieci w pełni
        połączonej przepuszczany jest wyłącznie wektor aktywnej kapsułki. """
    def __init__(self, num_classes=89, capsule_dim=16, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.Linear(capsule_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, img_size * img_size),
            nn.Sigmoid()  # Piksele w zakresie [0, 1]
        )

    def forward(self, x, target=None):
        batch_size = x.size(0)
        if target is not None:
            # Maskowanie: wybieramy tylko wektor kapsułki odpowiadającej etykiecie
            mask = func.one_hot(target, num_classes=self.num_classes).unsqueeze(-1)
            x = (x * mask).sum(dim=1)
        else:
            # Podczas inferencji: wybieramy najsilniejszą kapsułkę (norma wektora)
            v_mag = torch.norm(x, dim=-1)
            _, max_idx = v_mag.max(dim=1)
            mask = func.one_hot(max_idx, num_classes=self.num_classes).unsqueeze(-1)
            x = (x * mask).sum(dim=1)

        reconstructed = self.decoder(x)
        return reconstructed.view(batch_size, 1, self.img_size, self.img_size)


def calculate_capsule_loss(margin_loss, images, reconstructions, alpha=0.0005):
    """ Łączy Margin Loss z Reconstruction Loss. """
    recon_loss = func.mse_loss(reconstructions, images, reduction='sum') / images.size(0)
    return margin_loss + alpha * recon_loss


class UncertaintyAwareMarginLoss(nn.Module):
    """ Margin Loss ważona niepewnością CRNN ze zintegrowaną relaksacją marginesów.
        Zastępuje destrukcyjny Label Smoothing poprzez dynamiczne zacieśnianie 
        celów optymalizacji, chroniąc gradienty wektorów przed konfliktem. """
    def __init__(self, num_classes, device, gamma=2.0, smoothing=0.0): 
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.smoothing = smoothing

    def forward(self, norms, labels, crnn_confidence=None):
        # Czysty One-hot bez rozmazywania ułamkami (zapobiega sprzecznym gradientom)
        labels_one_hot = torch.eye(self.num_classes, device=self.device)[labels]

        # Relaksacja marginesów
        m_pos_target = 0.9 - (self.smoothing if self.smoothing > 0 else 0.0)
        m_neg_target = 0.1 + (self.smoothing * 0.5 if self.smoothing > 0 else 0.0)

        # Składowe funkcji straty z nowymi celami
        m_pos = torch.pow(torch.relu(m_pos_target - norms), 2)
        m_neg = torch.pow(torch.relu(norms - m_neg_target), 2)

        # Aplikacja maski, zapewnia czysty kierunek gradientu
        lambda_val = 0.5 * (10.0 / self.num_classes) 
        margin_loss = (labels_one_hot * m_pos + lambda_val * (1.0 - labels_one_hot) * m_neg)
        margin_loss = margin_loss.sum(dim=1)

        # Odwrotność pewności modelu CRNN
        if crnn_confidence is not None:
            uncertainty_weight = 2.0 - crnn_confidence.clamp(0, 1)
            margin_loss = margin_loss * uncertainty_weight

        return margin_loss.mean()


class ContrastiveCapsuleLoss(nn.Module):
    """ Contrastive loss dla lepszej separacji kapsułek.
        Redukuje confusion na podobnych znakach: l/I, 0/O, 5/S, rn/m
        Kluczowe dla routing quality do Transformera. """
    def __init__(self, margin=1.0, temperature=0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, capsules, labels):
        """
        capsules: [N, num_classes, caps_dim] - All class capsules
        labels: [N] - True labels
        """
        N = capsules.size(0)
        num_classes = capsules.size(1)

        # Extract winning capsules (predicted class)
        predicted = capsules.norm(dim=-1).argmax(dim=1)  # [N]

        # Get capsule vectors for predicted classes
        batch_idx = torch.arange(N, device=capsules.device)
        pred_capsules = capsules[batch_idx, predicted]  # [N, caps_dim]

        # Normalize for cosine similarity
        pred_norm = func.normalize(pred_capsules, dim=-1)

        # Compute pairwise similarities
        sim_matrix = torch.matmul(pred_norm, pred_norm.t())  # [N, N]

        # Create positive/negative masks
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
        positive_mask = label_matrix.float()
        negative_mask = 1 - positive_mask

        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(N, device=capsules.device)

        # Avoid division by zero
        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        neg_count = negative_mask.sum(dim=1).clamp(min=1)

        # Average similarities
        pos_sim = (sim_matrix * positive_mask).sum(dim=1) / pos_count
        neg_sim = (sim_matrix * negative_mask).sum(dim=1) / neg_count

        # Margin-based contrastive loss
        loss = torch.relu(self.margin - pos_sim + neg_sim).mean()

        return loss


class CapsNetLoss(nn.Module):
    """ Hybrydowa funkcja straty dla sieci CapsNet, łącząca klasyfikację z weryfikacją morfologiczną.
        Wspiera uncertainty-aware training dla CRNN → CapsNet.
        Składa się z:
        1. Uncertainty-Aware Focal Margin Loss: Ważona niepewnością CRNN.
        2. Reconstruction Loss (BCE + SSIM): Weryfikacja kształtu znaku.
        3. Contrastive Capsule Loss: Separacja podobnych znaków. """
    def __init__(self, num_classes, device, reconstruction_weight=5.0, crnn_weights=None, smoothing=0.1, gamma=2.0):
        super(CapsNetLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.reconstruction_weight = reconstruction_weight
        self.smoothing = smoothing
        self.gamma = gamma

        # Zapewniamy, że wagi mają odpowiedni kształt
        if crnn_weights is not None:
            self.weights = crnn_weights.to(device).view(1, -1)
        else:
            self.weights = torch.ones(1, num_classes).to(device)

        self.ssim = SSIMLoss(window_size=11).to(device)

        # Uncertainty-aware margin loss
        self.uncertainty_margin = UncertaintyAwareMarginLoss(num_classes, device, gamma, smoothing)

        # Contrastive capsule loss
        self.contrastive_caps = ContrastiveCapsuleLoss(margin=1.0)

    def forward(self, images, labels, probs, reconstructions, g_stats, boundaries,
                crnn_confidence=None, class_capsules=None, lambda_contrastive=0.0):
        batch_size = images.size(0)
        labels = labels.long()

        # Uncertantity-aware margin loss
        weighted_margin = self.uncertainty_margin(probs, labels, crnn_confidence=crnn_confidence)

        # Reconstruction Loss (BCE + SSIM) z zabezpieczeniem przed NaN
        target = torch.nn.functional.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)

        std_val = float(EMNIST_NORM_STD[0]) if isinstance(EMNIST_NORM_STD, (list, torch.Tensor)) else float(EMNIST_NORM_STD)
        mean_val = float(EMNIST_NORM_MEAN[0]) if isinstance(EMNIST_NORM_MEAN, (list, torch.Tensor)) else float(EMNIST_NORM_MEAN)

        t_norm = (target * std_val) + mean_val
        t_norm = t_norm.clamp(0, 1)

        # Binary Cross Entropy wywala NaN przy 0 lub 1 logarytmu, dlatego dodajemy margines
        epsilon = 1e-7
        r_norm = reconstructions.clamp(epsilon, 1.0 - epsilon)
        
        # Liczymy błędy na wspólnej skali
        bce_l = torch.nn.functional.binary_cross_entropy(r_norm, t_norm)
        ssim_l = self.ssim(r_norm, t_norm)

        recon_total = 0.7 * bce_l + 0.3 * ssim_l

        # Dodatkowe zabezpieczenie: jeśli total_loss jest NaN (np. z MarginLoss), przytnij go
        total_loss = weighted_margin + self.reconstruction_weight * recon_total
        
        if torch.isnan(total_loss).any():
            # Zamiast pozwolić na propagację NaN, zastępujemy je zere
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Contrastive capsule loss
        if lambda_contrastive > 0 and class_capsules is not None:
            contrastive_loss = self.contrastive_caps(class_capsules, labels)
            total_loss = total_loss + lambda_contrastive * contrastive_loss

        return total_loss


def calculate_cer(preds, targets, encoder) -> float:
    """ Oblicza CER na poziomie pojedynczych znaków (Character Error Rate).
        Idealna do walidacji CapsNet na zbiorze eMNIST lub wyciętych znakach PURE."""
    # Konwersja na CPU i numpy (jeśli to Tensory)
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Jawne wymuszenie typu ndarray i spłaszczenie
    p_arr: np.ndarray = np.asarray(preds).ravel()
    t_arr: np.ndarray = np.asarray(targets).ravel()

    if len(t_arr) == 0:
        return 0.0

    total_dist = 0
    # Używamy min, aby zip nie uciął danych bez ostrzeżenia, choć przy klasyfikacji znaków długości powinny być równe.
    for p, t in zip(p_arr, t_arr):
        try:
            # Rzutowanie na int, aby uniknąć problemów z typami np.int64
            p_char = encoder.decode(int(p))
            t_char = encoder.decode(int(t))
            if p_char != t_char:
                total_dist += 1
        except (TypeError, ValueError):
            # Obsługa błędnych indeksów w encoderze
            total_dist += 1

    # Matematyczna definicja CER dla klasyfikacji: (S+D+I)/N
    return float(total_dist / len(t_arr))


def calculate_cer_on_hard_cases(model, loader, device, encoder, desc="Test Hard Cases"):
    """ Oblicza CER do wyników na poziomie pojedynczych znaków na zbiorze HARD z treningu CRNN. """
    if loader is None or len(loader) == 0:
        print("Brak danych ze zbioru Hard w dataloaderze.")
        return 0.0, 0.0

    model.eval()
    total_cer = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch_cuda = [b.to(device) if torch.is_tensor(b) else b for b in batch]

            # Bezpieczne rozpakowanie
            inputs = batch_cuda[0]
            targets = batch_cuda[1]
            zones = batch_cuda[2] if len(batch_cuda) > 2 else None
            boundaries = batch_cuda[3] if len(batch_cuda) > 3 else None
            context_vecs = batch_cuda[4] if len(batch_cuda) > 4 else None
            confs = batch_cuda[5] if len(batch_cuda) > 5 else None
            crnn_p = batch_cuda[6] if len(batch_cuda) > 6 else None

            # Forward pass
            outputs = model(inputs, word_context=context_vecs, confidence=confs, crnn_probs=crnn_p,
                            boundaries=boundaries)
            
            # Wyciągamy predykcję ze słownika
            probs = outputs["probs"] if isinstance(outputs, dict) else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
            preds = torch.argmax(probs, dim=1)

            # Spłaszczanie
            preds = preds.view(-1)
            targets = targets.view(-1)

            # Statystyki
            total_acc += (preds == targets).sum().item()
            total_cer += calculate_cer(preds, targets, encoder) * targets.size(0)
            total_samples += targets.size(0)

    if total_samples == 0:
        print("Brak danych ze zbioru Hard (same puste batche).")
        return 0.0, 0.0

    return total_cer / total_samples, (total_acc / total_samples) * 100


def extract_crnn_hard_cases(model, loader, device, encoder, output_dir, confidence_threshold=0.85):
    """ Przechodzi przez zbiór danych, analizuje predykcje i zapisuje wycinki znaków,
        które model błędnie zaklasyfikował, lub co, do których miał niską pewność (Hard-Negative Mining). """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    saved_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Ekstrakcja Hard Cases", leave=False):
            batch_cuda = [b.to(device) if torch.is_tensor(b) else b for b in batch]

            inputs = batch_cuda[0]
            targets = batch_cuda[1]
            boundaries = batch_cuda[3] if len(batch_cuda) > 3 else None
            context_vecs = batch_cuda[4] if len(batch_cuda) > 4 else None
            confs = batch_cuda[5] if len(batch_cuda) > 5 else None
            crnn_p = batch_cuda[6] if len(batch_cuda) > 6 else None

            # Wyciągamy predykcje ze słownika
            outputs = model(inputs, word_context=context_vecs, confidence=confs, crnn_probs=crnn_p, boundaries=boundaries)
            probs = outputs["probs"] if isinstance(outputs, dict) else (outputs[0] if isinstance(outputs, (list, tuple)) else outputs)

            # Obliczanie pewności (Softmax)
            softmax_probs = torch.softmax(probs, dim=1)
            max_probs, preds = torch.max(softmax_probs, dim=1)

            preds = preds.view(-1)
            targets = targets.view(-1)
            max_probs = max_probs.view(-1)

            # Iteracja po elementach batcha i filtracja
            for i in range(inputs.size(0)):
                is_incorrect = preds[i] != targets[i]
                is_uncertain = max_probs[i].item() < confidence_threshold

                if is_incorrect or is_uncertain:
                    # Wyciągamy obraz do formatu numpy (odwracamy ewentualną normalizację, jeśli była w zakresie 0-1)
                    img_tensor = inputs[i].cpu().squeeze()
                    img_numpy = img_tensor.numpy()

                    if img_numpy.max() <= 1.0:
                        img_numpy = (img_numpy * 255).astype(np.uint8)

                    true_char = encoder.decode(int(targets[i].item()))
                    pred_char = encoder.decode(int(preds[i].item()))

                    # Zabezpieczenie nazw plików przed znakami niedozwolonymi w Windows
                    safe_true = "".join([c if c.isalnum() else f"_{ord(c)}_" for c in true_char])
                    safe_pred = "".join([c if c.isalnum() else f"_{ord(c)}_" for c in pred_char])

                    filename = f"hard_{saved_count:05d}_T_{safe_true}_P_{safe_pred}_conf_{max_probs[i].item():.2f}.png"
                    filepath = os.path.join(output_dir, filename)

                    cv.imwrite(filepath, img_numpy)
                    saved_count += 1

    print(f"Wyodrębniono i zapisano {saved_count} trudnych przypadków do folderu: {output_dir}")
    return saved_count


def load_samples_from_folders(root_path, encoder):
    """ Wczytuje obrazy z podfolderów, ignorując tylko uszkodzone pliki. """
    samples = []
    if not os.path.exists(root_path):
        print(f"[{now()}] Ścieżka nie istnieje: {root_path}")
        return samples

    for label in os.listdir(root_path):
        folder_path = os.path.join(root_path, label)
        if os.path.isdir(folder_path):
            char_idx = encoder.encode(label)
            if char_idx is None:
                continue

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    try:
                        # Otwieramy i wymuszamy wczytanie
                        with Image.open(img_path) as img_raw:
                            img = img_raw.convert('L')
                            # Kopiujemy do pamięci, by zamknąć plik
                            samples.append((img.copy(), char_idx))

                    except (UnidentifiedImageError, IOError) as e:
                        # Logujemy tylko realne problemy z obrazem
                        print(f"[{now()}] Pominęto uszkodzony plik {img_name}: {e}")
                        continue
    return samples


class FocalLoss(nn.Module):
    """ Implementacja funkcji straty Focal Loss.
        Rozwiązuje problem niezbalansowanych klas i wymusza na sieci skupienie się na najtrudniejszych przypadkach.
        Dynamicznie redukuje wagę błędu dla próbek, które model rozpoznaje już z dużą pewnością,
        zapobiegając ich dominacji podczas aktualizacji gradientów. """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = func.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def setup_optimizer_and_scheduler(model, train_loader, epochs, max_lr=1e-3):
    """ Konfiguruje optymalizator AdamW oraz harmonogram stopy uczenia OneCycleLR.
        Metoda One Cycle pozwala na osiągnięcie zjawiska super-konwergencji. Algorytm najpierw płynnie
        zwiększa stopę uczenia, co pozwala modelowi na szybkie wyrwanie się z suboptymalnych, lokalnych minimów.
        Następnie przez resztę treningu stopa uczenia jest wygaszana, co pozwala dokładnie osiąść w stabilnym,
        płaskim minimum funkcji straty. """
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr / 10, weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,  # 30% czasu na rozgrzewkę
        anneal_strategy='cos',  # Płynne wygaszanie cosinusoidalne
        div_factor=10.0,
        final_div_factor=100.0
    )
    return optimizer, scheduler


def train_and_save(model, train_loader, val_loader, num_classes, device, epochs_to_run, save_path, zone_map, crnn_weights, num_iterations, lr, phase_name="main", checkpoint_dict=None):
    """ Realizuje wieloetapowy proces uczenia i adaptacji modelu CapsNet w kaskadowym potoku HTR.
        Architektura faz:
            1. MAIN (Inicjalizacja Hybrydowa): Łączy syntetyczną bazę eMNIST z rzeczywistymi wycinkami PURE.
                Konstruuje filtry konwolucyjne odporne na teksturę papieru, szum tła oraz zmienność grubości atramentu.
            2. FINE-TUNE (Adaptacja Domenowa): Etap z dominacją rzeczywistych danych docelowych (PURE/PHSF).
                Stabilizuje wektorowe reprezentacje kapsuł, płynnie transferując wiedzę o ogólnej topologii
                znaków na specyficzny styl odręcznego i historycznego pisma.
            3. HARD MINING (Selektywna Korekta): Precyzyjna faza operująca na wycinkach, które sprawiają trudność
                modelowi CRNN (Hard-Negative Mining) oraz błyskawicznie wydobywanych w locie błędach z linii CVL.
                Uczy model rozstrzygania optycznych niejednoznaczności przy zamrożonym ekstraktorze cech.
        Mechanizmy stabilizacji, ewaluacji i zapobiegania przeuczeniu:
            - Hybrydowa Rekonstrukcja (BCE + SSIM): Wskaźnik podobieństwa strukturalnego wymusza
                naukę morfologii znaku (np. domknięcia pętli, relacje przestrzenne), blokując ślepe zapamiętywanie etykiet.
            - Kotwica Wiedzy (eMNIST): Dzięki zastosowaniu ważonego próbkowania, model
                stale utrzymuje kontakt z kanonicznymi kształtami, co zapobiega zjawisku katastroficznego zapominania
                podczas agresywnego dostrajania do domeny docelowej.
            - Contrastive Capsule Loss: Aktywnie rozdziela w przestrzeni wielowymiarowej reprezentacje wektorowe 
                podobnych do siebie znaków (np. 'g'/'q', 'l'/'I', 'rn'/'m').
            - Loss-driven Checkpointing: System zapisu wag promuje stabilność sieci - priorytetem jest globalny
                spadek funkcji straty, a metryka CER pełni rolę wskaźnika rozstrzygającego przy remisach. """
    if not hasattr(train_and_save, 'loader_pure_val'): train_and_save.loader_pure_val = None
    if not hasattr(train_and_save, 'loader_hard_val'): train_and_save.loader_hard_val = None

    start_epoch = 0
    patience_counter = 0
    max_patience = 3
    best_val_cer = 1.0
    best_loss = float('inf')

    encoder = CharLabelEncoder()

    # Inicjalizacja strat
    if phase_name == "MAIN":
        criterion_char = UncertaintyAwareMarginLoss(num_classes=num_classes, device=device, smoothing=SMOOTHING).to(device)
    else:
        criterion_char = CapsNetLoss(num_classes=num_classes, device=device, crnn_weights=crnn_weights, smoothing=SMOOTHING).to(device)

    criterion_zone = nn.CrossEntropyLoss().to(device)

    # Zarządzanie zamrażaniem Backbone w fazie HARD_MINING
    if phase_name == "HARD_MINING":
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    # Optymalizator zbiera tylko te parametry, które mają requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad] + list(criterion_char.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)

    # Synchronizacja Schedulera z aktualizacjami wag
    steps_per_epoch = min(len(train_loader), STEPS_PER_EPOCH)
    updates_per_epoch = steps_per_epoch // ACCUMULATION_STEPS
    total_updates = epochs_to_run * updates_per_epoch

    warmup_steps = max(10, int(0.1 * total_updates))
    decay_steps = max(1, total_updates - warmup_steps)

    warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_steps])

    # Wczytywanie stanu
    start_epoch = 0
    if checkpoint_dict and checkpoint_dict.get('phase') == phase_name:
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer_state'])
            start_epoch = checkpoint_dict.get('epoch', -1) + 1

            # Przesuwamy scheduler do odpowiedniego miejsca
            for _ in range(start_epoch * updates_per_epoch):
                scheduler.step()
        except (RuntimeError, KeyError, ValueError) as e:
            # Przechwytujemy tylko błędy związane z niedopasowaniem wag lub brakiem kluczy
            tqdm.write(f"[{now()}] Reset stanu optymalizatora/schedulera. Błąd: {e}")
            start_epoch = 0

    # Główny trening
    for epoch in range(start_epoch, epochs_to_run):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"[{phase_name.upper()}] Epoka {epoch + 1}", total=steps_per_epoch, ncols=120, leave=False, disable=False)

        optimizer.zero_grad()  # Czyścimy na początku epoki

        for i, batch in enumerate(loop):
            if i >= steps_per_epoch: break

            batch_cuda = [b.to(device, non_blocking=True) if torch.is_tensor(b) else b for b in batch]
            inputs, labels, zones, boundaries, context_vecs, confs, *rest = batch_cuda
            crnn_p = rest[0] if rest else None

            # Używamy checkpointingu
            def forward_pass(in_p, lbl_p, ctx_p, conf_p, crnn_p, bnd_p):
                # CapsNet forward zwraca słownik
                return model(in_p, lbl_p, word_context=ctx_p, confidence=conf_p, 
                             crnn_probs=crnn_p, boundaries=bnd_p)

            in_cpy = inputs.detach().clone().requires_grad_(True) 

            # outputs to słownik: {"norms": ..., "reconstruction": ..., "z_logits": ..., "g_stats": ...}
            outputs = torch.utils.checkpoint.checkpoint(
                forward_pass, 
                in_cpy, labels.detach().clone(), context_vecs.detach().clone(), 
                confs.detach().clone(), crnn_p.detach().clone() if crnn_p is not None else None, 
                boundaries.detach().clone(),
                use_reentrant=False
            )
            
            # Pobranie probs
            probs = outputs["norms"] 
            z_logits = outputs["z_logits"]

            # Definicja straty
            if phase_name == "MAIN":
                char_loss = criterion_char(probs, labels) 
                contrastive_loss = 0.0 # W fazie MAIN skupiamy się na klasyfikacji
            else:
                recs = outputs["reconstruction"]
                g_stats = outputs["g_stats"]
                # Zakładamy, że criterion_char zwraca krotkę lub obiekt z contrastive_loss
                char_loss, contrastive_loss = criterion_char(
                    inputs, labels, probs.clone(), recs.clone(), 
                    g_stats.clone(), boundaries.clone()
                )
            
            """ Balansowanie składników straty
                1.0 * char_loss (kluczowe)
                0.1 * zone_loss (pomocnicze)
                0.05 * contrastive_loss (wymusza różnice między 't' a '7') """
            zone_loss = criterion_zone(z_logits.clone(), zones.long())
            
            total_loss = (char_loss + 0.1 * zone_loss + 0.05 * contrastive_loss) 
            loss = total_loss / ACCUMULATION_STEPS
            
            loss.backward()

            # Akumulacja gradientu
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS
            loop.set_postfix(
                loss=f"{(loss.item() * ACCUMULATION_STEPS):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        train_loss = running_loss / steps_per_epoch

        save_debug_snapshots(model, val_loader, device, encoder, epoch)

        def get_cer_score(loader, debug_name=None):
            if loader is None: return 1.0, [], []
            
            total_errors = 0.0
            total_samples = 0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for j, b_v in enumerate(loader):
                    if j >= VAL_STEPS_LIMIT: break
                    b_v_c = [b.to(device) if torch.is_tensor(b) else b for b in b_v]
                    v_in, v_lbl, _, v_bnd, *rest = b_v_c
                    
                    v_out = model(v_in, word_context=None, confidence=None,
                                  crnn_probs=None, boundaries=v_bnd, num_iterations=num_iterations)
                    
                    # Wektory kapsułkowe na prawdopodobieństwa
                    v_probs = torch.nn.functional.softmax(v_out["norms"], dim=1)
                    top_probs, preds = torch.max(v_probs, 1)
                    
                    # Zbieranie logów do raportu
                    for p_idx, p_val, t_idx in zip(preds, top_probs, v_lbl):
                        char_pred = encoder.decode(p_idx.item())
                        char_target = encoder.decode(t_idx.item())
                        chance = p_val.item() * 100.0
                        
                        all_preds.append(f"'{char_pred}' ({chance:.1f}%)")
                        all_targets.append(char_target)

                    # Obliczanie ilości błędów w konkretnym batchu i akumulacja
                    batch_cer = calculate_cer(preds, v_lbl, encoder)
                    total_errors += batch_cer * v_lbl.size(0)
                    total_samples += v_lbl.size(0)
            
            # Ostateczny wynik CER
            final_cer = total_errors / total_samples if total_samples > 0 else 1.0
            return final_cer, all_preds, all_targets

        model.eval()
        metrics = {}

        # Ewaluacja na PURE
        if train_and_save.loader_pure_val:
            metrics['Pure'], _, _ = get_cer_score(train_and_save.loader_pure_val, debug_name="PURE")

        # Ewaluacja eMNIST
        if val_loader:
            metrics['eMNIST'], _, _ = get_cer_score(val_loader, debug_name="EMNIST")

        # Walidator HARD (Błędy CRNN)
        if phase_name == "HARD_MINING":
            if train_and_save.loader_hard_val is None:
                if os.path.exists(CRNN_CUSTOM_SAMPLES):
                    ds_h = HardCharsDataset(CRNN_CUSTOM_SAMPLES, encoder=encoder, case_type="hard_case", transform=VAL_TRANSFORMS)
                    if len(ds_h) > 0:
                        train_and_save.loader_hard_val = DataLoader(ds_h, batch_size=BATCH_SIZE, num_workers=0)
                    else:
                        print(f" Katalog {CRNN_CUSTOM_SAMPLES} istnieje, ale nie znaleziono w nim obrazów.")
                else:
                    print(f" Brak docelowego katalogu: {CRNN_CUSTOM_SAMPLES}. Pomięto ładowanie zbioru.")

            if train_and_save.loader_hard_val:
                metrics['Hard'], _, _ = get_cer_score(train_and_save.loader_hard_val)

        # Zapewniamy, że val_cer jest zawsze floatem
        raw_val = metrics.get('Pure')
        
        # Pobieramy wyniki w bezpieczny sposób
        pure_val = metrics.get('Pure')
        hard_val = metrics.get('Hard')
        emnist_val = metrics.get('eMNIST', 1.0) 

        # Jawne logowanie wszystkich metryk, by uniknąć złudzeń!
        log_msg = f"[{now()}] WYNIKI EPOKI -> "
        if metrics.get('eMNIST') is not None: log_msg += f"eMNIST: {emnist_val*100:.2f}% | "
        if pure_val is not None: log_msg += f"PURE: {pure_val*100:.2f}% | "
        if hard_val is not None: log_msg += f"HARD: {hard_val*100:.2f}%"
        tqdm.write(log_msg)

        # Wybieramy val_cer informacyjnie, do logów i statystyk checkpointu
        if phase_name == "HARD_MINING" and hard_val is not None:
            val_cer = float(hard_val)
        elif pure_val is not None:
            val_cer = float(pure_val)
        else:
            val_cer = float(emnist_val)

        tqdm.write(f"[{now()}] val_cer: {val_cer*100:.2f}%")

        # Logika
        is_best = False
        current_train_loss = float(train_loss) if train_loss is not None else float('inf')

        # Zapisujemy model, gdy funkcja straty spada. 
        if current_train_loss < best_loss:
            is_best = True
        # W przypadku remisu, sprawdzamy, czy osiągnęliśmy mniejszy CER
        elif abs(current_train_loss - best_loss) < 1e-6 and val_cer < best_val_cer:
            is_best = True

        if is_best:
            best_loss = current_train_loss
            best_val_cer = val_cer  # Aktualizujemy rekord CER jako statystykę towarzyszącą

            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_val_cer': best_val_cer,
                'phase': phase_name,
                'epoch': epoch
            }, save_path)
            tqdm.write(f"[{now()}] Nowy rekord ({phase_name})! Loss: {best_loss:.4f} | CER: {best_val_cer * 100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            tqdm.write(f"[{phase_name.upper()}] Brak poprawy Loss/CER ({patience_counter}/{max_patience}).")

        if patience_counter >= max_patience:
            tqdm.write(f"[{now()}] Brak progresu przez {max_patience} epok. Przejście do następnej fazy.")
            break
            
        tqdm.write(f"[{now()}] Loss: {train_loss:.4f} | CER: {val_cer * 100:.2f}%")

        torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'phase': phase_name}, save_path.replace(".pth", "_latest.pth"))


def clear_memory(variable):
    del variable
    gc.collect() # Czyści RAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class EMNISTWithContext(Dataset):
    def __init__(self, raw_dataset, encoder, zone_map):
        self.raw_dataset = raw_dataset
        self.encoder = encoder
        self.zone_map = zone_map # Dodano mapowanie stref do ładowania w locie

        # Standardowy maping eMNIST
        self.emnist_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        image, raw_label = self.raw_dataset[index]

        # Mapujemy label eMNIST na znak, a potem znak na indeks
        actual_char = self.emnist_mapping[raw_label]
        target_label = self.encoder.char_to_idx.get(actual_char, 0)

        # Dynamicznie pobieramy strefę geometryczną
        zone = torch.tensor(self.zone_map.get(target_label, 1), dtype=torch.long)

        # Stałe metadane dla eMNIST (brak kontekstu słownego)
        context_vec = torch.zeros(1024, dtype=torch.float32)
        boundaries = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return image, torch.tensor(target_label, dtype=torch.long), zone, boundaries, context_vec, torch.tensor(0.0)


def generate_confusion_matrix(model, loader, device, encoder, save_path, top_n=20):
    """ Generuje macierz pomyłek dla TOP_N najczęściej mylonych par znaków i zapisuje pod wskazaną ścieżką. """
    # Automatyczne tworzenie folderu, w którym ma znaleźć się plik
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds, all_labels = [], []

    tqdm.write(f"[{now()}] Rozpoczynam zbieranie danych do macierzy pomyłek.")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Ewaluacja Macierzy", leave=False):
            # Rozpakowanie batcha (obsługa obrazów, etykiet i opcjonalnie innych danych)
            images, labels = batch[0], batch[1]
            images = images.to(device)

            # Forward pass (pobieramy tylko prawdopodobieństwa)
            outputs = model(images)
            probs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            preds = torch.max(probs, 1)[1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Obliczenie pełnej macierzy
    full_cm = confusion_matrix(all_labels, all_preds)

    # Kopia do szukania błędów (zerujemy przekątną, by nie widzieć poprawnych klas)
    cm_errors = full_cm.copy()
    np.fill_diagonal(cm_errors, 0)

    # Szukanie TOP_N par z największą liczbą błędów
    pairs = []
    for i in range(cm_errors.shape[0]):
        for j in range(cm_errors.shape[1]):
            if cm_errors[i, j] > 0:
                pairs.append(((i, j), cm_errors[i, j]))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:top_n]

    if not top_pairs:
        tqdm.write(f"[{now()}] Brak błędów do wyświetlenia w macierzy!")
        return

    # Budowanie pod-macierzy do wizualizacji
    unique_indices = sorted(list(set([p[0][0] for p in top_pairs] + [p[0][1] for p in top_pairs])))
    subset_labels = [encoder.decode(idx) for idx in unique_indices]
    idx_map = {orig: new for new, orig in enumerate(unique_indices)}

    subset_cm = np.zeros((len(unique_indices), len(unique_indices)), dtype=int)
    for r_orig in unique_indices:
        for c_orig in unique_indices:
            subset_cm[idx_map[r_orig], idx_map[c_orig]] = full_cm[r_orig, c_orig]

    # Rysowanie i zapis
    plt.figure(figsize=(12, 10))
    sns.heatmap(subset_cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=subset_labels, yticklabels=subset_labels)
    plt.ylabel('Prawdziwa Etykieta (True)')
    plt.xlabel('Przewidziana Etykieta (Pred)')
    plt.title(f'Macierz pomyłek - TOP {top_n} najczęstszych błędów')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    tqdm.write(f"[{now()}] Macierz pomyłek zapisana: {save_path}")


def seed_everything(seed: int = 3407):
    """ Ustawia ziarna losowości dla wszystkich bibliotek, gwarantując determinizm eksperymentu. """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Poniższe dwie linie sprawiają, że CuDNN używa tylko deterministycznych algorytmów.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"[{now()}] Ziarno losowości ustawione na: {seed}")


def clear_system_memory():
    """ Czyści RAM i pamięć VRAM GPU. """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tqdm.write(f"[{now()}] Pamięć systemowa została wyczyszczona.")


class RAMCachedEMNIST(Dataset):
    """ Przenoszenie całego zbioru eMNIST do RAM, dla szybkości treningu. """
    def __init__(self, base_dataset, zone_map, desc="Ładowanie eMNIST do RAM"):
        self.images = []
        self.labels = []
        self.zones = []
        self.contexts = []

        temp_loader = DataLoader(
            base_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        for batch in tqdm(temp_loader, desc=desc, disable=False, dynamic_ncols=True, ascii=True, mininterval=0.5, leave=False):
            # Ignorujemy domyślne strefy z batcha
            imgs, lbls, _, _, ctxs, _ = batch

            # Rzutowanie na float16, kluczowe dla oszczędności RAM-u
            self.images.append(imgs.half())
            self.labels.append(lbls)
            
            # Wektory kontekstowe (1024 wymiary) też zajmują dużo miejsca
            self.contexts.append(ctxs.half())

            # Wyliczamy prawidłowe strefy na podstawie prawdziwego znaku
            z = torch.tensor([zone_map.get(int(l), 1) for l in lbls], dtype=torch.long)
            self.zones.append(z)

        # Agregacja do dużych tensorów — zapewnia stały czas dostępu
        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.zones = torch.cat(self.zones, dim=0)
        self.contexts = torch.cat(self.contexts, dim=0)

        # Obliczenie realnego zużycia pamięci przez główne tensory
        total_mem = (self.images.element_size() * self.images.nelement() + self.contexts.element_size() * self.contexts.nelement()) / 1e9
        tqdm.write(f"[{now()}] Zużycie RAM dla obrazów i kontekstów: {total_mem:.2f} GB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Konwersja z powrotem do float32 tylko dla aktualnie pobieranego elementu
        return (
            self.images[idx],
            self.labels[idx],
            self.zones[idx],
            torch.zeros(7, dtype=torch.float32),
            self.contexts[idx],
            torch.tensor(0.0, dtype=torch.float32)
        )


class CustomDatasetWithContext(torch.utils.data.Dataset):
    """ Wrapper dopasowujący ImageFolder do formatu (img, label, zone, geom, context, conf). """
    def __init__(self, image_folder_dataset):
        self.ds = image_folder_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        zone = torch.tensor(1, dtype=torch.long)
        geom = torch.zeros(7, dtype=torch.float32)
        context = torch.zeros(1024)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img, label_tensor, zone, geom, context, torch.tensor(1.0, dtype=torch.float32)


def build_train_loader(dataset, subset="emnist", batch_size=32):
    """ Loader eMNIST. """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=MAIN_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=2
    )


def build_fine_tune_loader(data_path, encoder, transform, batch_size, num_workers=0):
    """ Buduje DataLoader dla fazy Fine-tune z użyciem wycinków PURE. """
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print(f"[{now()}] Folder {data_path} jest pusty lub nie istnieje. Pomijam ten etap.")
        return None

    custom_dataset = HardCharsDataset(data_path, encoder=encoder, transform=transform)

    # Wymuszenie przynajmniej 2 workerów przy czytaniu ogromnej liczby małych plików
    workers = max(2, num_workers)

    return DataLoader(
        dataset=custom_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=2
    )


def build_hard_mining_loader(ds_emnist, ds_hard, ds_pure, ds_phsf, batch_size):
    """ Loader dla Fazy hard mining. Dynamicznie łączy zbiory, nadając priorytet błędom CRNN oraz trudnym danym PHSF. """
    datasets_to_combine = []
    weights = []

    # Baza: EMNIST (Waga 1.0 - fundament rozpoznawania kształtów)
    if ds_emnist is not None:
        datasets_to_combine.append(ds_emnist)
        weights.extend([1.0] * len(ds_emnist))

    # Błędy: Hard Mining (Waga 15.0 - bezpośrednie pomyłki modelu)
    if ds_hard is not None:
        datasets_to_combine.append(ds_hard)
        weights.extend([15.0] * len(ds_hard))

    # Realne pismo: (Waga 5.0 - współczesne próbki)
    if ds_pure is not None:
        datasets_to_combine.append(ds_pure)
        weights.extend([5.0] * len(ds_pure))

    # PHSF: Historyczne Formularze (Waga 10.0 - wysoki stopień trudności)
    if ds_phsf is not None:
        datasets_to_combine.append(ds_phsf)
        weights.extend([10.0] * len(ds_phsf))

    if not datasets_to_combine:
        print(f"[{now()}] Brak danych do budowy Hard Mining Loader!")
        return None

    combined_ds = ConcatDataset(datasets_to_combine)

    # Sampler wymusza częstsze pojawianie się trudnych próbek w batchu
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined_ds),
        replacement=True
    )

    return DataLoader(
        combined_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )


class UniversalContextWrapper(Dataset):
    """ Adapter ujednolicający strukturę danych do formatu 8-elementowego.
        Gwarantuje zgodność z ConcatDataset i modelem CapsNet w trybie bilingwalnym. 
        Zawiera aktywny mechanizm wymuszający zgodność wymiarów (Sledgehammer). """
    def __init__(self, dataset, context_size=1024, num_classes=None):
        self.dataset = dataset
        self.context_size = context_size
        
        # Pobieramy prawidłowy wymiar
        if num_classes is not None:
            self.num_classes = num_classes
        elif hasattr(dataset, 'encoder') and hasattr(dataset.encoder, 'get_num_classes'):
            self.num_classes = dataset.encoder.get_num_classes()
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'encoder') and hasattr(dataset.dataset.encoder, 'get_num_classes'):
            self.num_classes = dataset.dataset.encoder.get_num_classes()
        else:
            self.num_classes = 95  # Fallback zgodny z architekturą CRNN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Konwersja krotki na listę, by móc w locie nadpisywać jej elementy
        data = list(self.dataset[idx])

        # Uzupełnianie brakujących elementów (starsze formaty)
        if len(data) == 6:
            img, lbl, zone, bnd, ctx, conf = data
            data = [img, lbl, zone, bnd, ctx, conf, torch.zeros(self.num_classes, dtype=torch.float32), torch.tensor(1, dtype=torch.long)]
        elif len(data) == 7:
            img, lbl, zone, bnd, ctx, conf, probs = data
            data = [img, lbl, zone, bnd, ctx, conf, probs, torch.tensor(1, dtype=torch.long)]

        # wymuszamy zgodność wektora na indeksie crnn_probs z modelem
        if len(data) == 8:
            probs = data[6]
            if probs.shape[0] != self.num_classes:
                padded_probs = torch.zeros(self.num_classes, dtype=torch.float32)
                min_len = min(probs.shape[0], self.num_classes)
                padded_probs[:min_len] = probs[:min_len]
                data[6] = padded_probs

        # Zwracamy z powrotem jako krotkę
        return tuple(data)


class CVLDatasetHDF5(Dataset):
    def __init__(self, h5_path, split='words', transform=None):
        self.h5_path = h5_path
        self.split = split
        self.transform = transform
        # Sprawdzenie długości bez ładowania wszystkiego do RAM
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[f'{self.split}/images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img_encoded = f[f'{self.split}/images'][idx]
            label = f[f'{self.split}/labels'][idx].decode('utf-8')
            
            # Dekodowanie obrazu
            img = cv.imdecode(img_encoded, cv.IMREAD_GRAYSCALE)
            
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
                
            return img, label
          

def run_phase(phase_id, model, device, encoder, current_zone_map, test_loader, num_classes, flag_path):
    final_save_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    checkpoint = {}

    # Konfiguracja transformacji
    emnist_transform_aggressive = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Lambda(prepare_emnist_sample),
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.RandomApply([v2.JPEG(quality=(30, 65))], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=12, fill=0),
        v2.RandomPerspective(distortion_scale=0.25, p=0.4),
        v2.RandomApply([v2.ElasticTransform(alpha=10.0, sigma=4.0)], p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        v2.GaussianNoise(mean=0.0, sigma=0.02),
        v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
    ])

    pure_transform_standard = v2.Compose([
        v2.ToImage(),
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomPerspective(distortion_scale=0.15, p=0.3),
        v2.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0),
        v2.RandomApply([v2.ElasticTransform(alpha=20.0, sigma=4.0)], p=0.3),
        v2.RandomChoice([v2.Identity(), v2.ColorJitter(brightness=0.2, contrast=0.2)]),
        v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
    ])

    crnn_error_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomAffine(degrees=3, translate=(0.15, 0.15), scale=(0.95, 1.05), fill=0),
        v2.RandomApply([v2.ElasticTransform(alpha=18.0, sigma=3.0)], p=0.3),
        v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
    ])

    if os.path.exists(final_save_path):
        checkpoint = torch.load(final_save_path, map_location=device)
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            
            # Wymuszamy usunięcie wag warstwy fusion.prob_project, na wypadek zmiany liczby klas
            if 'fusion.prob_project.weight' in state_dict:
                del state_dict['fusion.prob_project.weight']
            if 'fusion.prob_project.bias' in state_dict:
                del state_dict['fusion.prob_project.bias']
            
            # Teraz load_state_dict zignoruje te warstwy
            model.load_state_dict(state_dict, strict=False)
            tqdm.write(f"[{now()}] Wagi załadowane.")

    if phase_id == 1:
        tqdm.write(f"[{now()}] Faza MAIN: Połączenie eMNIST + PURE + PHSF")

        ds_e_raw = datasets.EMNIST(root=DATA_ROOT_EMNIST, split='byclass', train=True, transform=emnist_transform_aggressive)
        ds_e = UniversalContextWrapper(EMNISTWithContext(ds_e_raw, encoder, current_zone_map))

        # Walidacja na zbiorze PURE
        pure_test_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
        ])

        # Ładujemy dane dwa razy - z i bez augmentacji
        full_ds_p_train = HardCharsDataset(root_dir=CRNN_CUSTOM_SAMPLES, encoder=encoder, case_type="pure", transform=pure_transform_standard)
        full_ds_p_val = HardCharsDataset(root_dir=CRNN_CUSTOM_SAMPLES, encoder=encoder, case_type="pure", transform=pure_test_transform)
        
        train_p_len = int(0.9 * len(full_ds_p_train))
        val_p_len = len(full_ds_p_train) - train_p_len
        
        # Generujemy indeksy podziału
        generator = torch.Generator().manual_seed(3407)
        indices = list(range(len(full_ds_p_train)))
        train_indices, val_indices = torch.utils.data.random_split(indices, [train_p_len, val_p_len], generator=generator)
        
        # Tworzymy podzbiory z właściwymi transformacjami
        ds_p_train = torch.utils.data.Subset(full_ds_p_train, train_indices.indices)
        ds_p_val = torch.utils.data.Subset(full_ds_p_val, val_indices.indices)
        
        ds_p_train_wrapped = UniversalContextWrapper(ds_p_train)
        train_and_save.loader_pure_val = DataLoader(UniversalContextWrapper(ds_p_val), batch_size=BATCH_SIZE, shuffle=False)

        phsf_dataset = PHSFDataset(npz_path=PHSF_DATA_PATH, encoder=encoder, transform=pure_transform_standard)

        # Wagowanie
        datasets_to_combine = [ds_e, ds_p_train_wrapped]
        weights = []
        has_phsf = phsf_dataset is not None and len(phsf_dataset) > 0

        w_e = 0.6 if has_phsf else 0.8  
        w_p = 0.1 if has_phsf else 0.2

        weights.extend([w_e / len(ds_e)] * len(ds_e))
        weights.extend([w_p / len(ds_p_train_wrapped)] * len(ds_p_train_wrapped))

        if has_phsf:
            datasets_to_combine.append(phsf_dataset)
            weights.extend([0.3 / len(phsf_dataset)] * len(phsf_dataset))

        combined = ConcatDataset(datasets_to_combine)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
        loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=MAIN_WORKERS, pin_memory=True)

        train_and_save(model, loader, test_loader, num_classes, device, MAIN_EPOCHS,
                       final_save_path, current_zone_map, None, 3, INITIAL_LR, "MAIN", checkpoint)

    elif phase_id == 2:
        tqdm.write(f"[{now()}] Faza FINE_TUNE: Domena realna (PURE + PHSF)")

        # Walidacja PURE
        pure_test_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
        ])

        # Ładujemy dane dwa razy: z i bez augmentacji
        full_ds_p_train = HardCharsDataset(root_dir=CRNN_CUSTOM_SAMPLES, encoder=encoder, case_type="pure", transform=pure_transform_standard)
        full_ds_p_val = HardCharsDataset(root_dir=CRNN_CUSTOM_SAMPLES, encoder=encoder, case_type="pure", transform=pure_test_transform)
        
        train_p_len = int(0.9 * len(full_ds_p_train))
        val_p_len = len(full_ds_p_train) - train_p_len
        
        # Generujemy indeksy podziału
        generator = torch.Generator().manual_seed(3407)
        indices = list(range(len(full_ds_p_train)))
        train_indices, val_indices = torch.utils.data.random_split(indices, [train_p_len, val_p_len], generator=generator)
        
        # Tworzymy podzbiory z właściwymi transformacjami
        ds_p_train = torch.utils.data.Subset(full_ds_p_train, train_indices.indices)
        ds_p_val = torch.utils.data.Subset(full_ds_p_val, val_indices.indices)
        
        ds_p_train_wrapped = UniversalContextWrapper(ds_p_train)
        
        # Zabezpieczenie ciągłości loadera walidacyjnego na wypadek wznowienia od Fazy 2
        train_and_save.loader_pure_val = DataLoader(UniversalContextWrapper(ds_p_val), batch_size=BATCH_SIZE, shuffle=False)
    
        ds_e = UniversalContextWrapper(EMNISTWithContext(
            datasets.EMNIST(root=DATA_ROOT_EMNIST, split='byclass', train=True, transform=emnist_transform_aggressive),
            encoder, current_zone_map))

        phsf_raw = PHSFDataset(npz_path=PHSF_DATA_PATH, encoder=encoder, transform=pure_transform_standard)
        
        datasets_to_combine = [ds_e, ds_p_train_wrapped]
        weights = []
        has_phsf = phsf_raw is not None and len(phsf_raw) > 0

        w_e = 0.1 if has_phsf else 0.2  # eMNIST tylko jako kotwica
        w_p = 0.6 if has_phsf else 0.8

        weights.extend([w_e / len(ds_e)] * len(ds_e))
        weights.extend([w_p / len(ds_p_train_wrapped)] * len(ds_p_train_wrapped))

        if has_phsf:
            ds_phsf = UniversalContextWrapper(phsf_raw)
            datasets_to_combine.append(ds_phsf)
            weights.extend([0.4 / len(ds_phsf)] * len(ds_phsf))

        combined = ConcatDataset(datasets_to_combine)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
        loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=FINE_WORKERS, pin_memory=True)

        train_and_save(model, loader, test_loader, num_classes, device, FINE_TUNE_EPOCHS,
                       final_save_path, current_zone_map, None, 3, FINE_TUNE_LR, "FINE_TUNE", checkpoint)
        

    elif phase_id == 3:
        tqdm.write(f"[{now()}] Faza HARD_MINING: Hard-Negative Mining (Błędy CRNN) + Czysta domena realna")

        # Polegamy na wycinkach z błędy wyeksportowanych przez CRNN
        if not os.path.exists(CRNN_CUSTOM_SAMPLES) or len(os.listdir(CRNN_CUSTOM_SAMPLES)) == 0:
            tqdm.write(f"Brak wycinków CRNN w {CRNN_CUSTOM_SAMPLES}. Upewnij się, że CRNN zakończył eksport.")

        # Ładowanie zbiorów
        ds_e = UniversalContextWrapper(cached_emnist_train) 
        ds_h = UniversalContextWrapper(HardCharsDataset(CRNN_CUSTOM_SAMPLES, encoder, "hard_case", crnn_error_transform))
        ds_p = UniversalContextWrapper(HardCharsDataset(CRNN_CUSTOM_SAMPLES, encoder, "pure", pure_transform_standard))

        # Wsparcie PHSF (jeśli istnieje)
        phsf_raw = PHSFDataset(npz_path=PHSF_DATA_PATH, encoder=encoder, transform=pure_transform_standard)
        ds_phsf = UniversalContextWrapper(phsf_raw) if len(phsf_raw.samples) > 0 else None

        # Loader zbalansuje wagi, faworyzując ds_h (błędy) i ds_phsf
        loader_hard = build_hard_mining_loader(
            ds_emnist=ds_e, 
            ds_hard=ds_h, 
            ds_pure=ds_p, 
            ds_phsf=ds_phsf, 
            batch_size=BATCH_SIZE
        )

        crnn_weights = torch.ones(num_classes).to(device)
        polish_indices = [encoder.char_to_idx[c] for c in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ" if c in encoder.char_to_idx]
        for idx in polish_indices:
            crnn_weights[idx] = 2.5 

        model.freeze_backbone()
        
        if loader_hard is not None:
            train_and_save(model, loader_hard, test_loader, num_classes, device, HARD_MINING_EPOCHS,
                           final_save_path, current_zone_map, crnn_weights, 3, HARD_MINING_LR, "HARD_MINING", checkpoint)
        else:
            tqdm.write(f"[{now()}] Pomięto trening HARD_MINING z powodu braku danych w loaderze.")

    with open(flag_path, 'w') as f:
        f.write('done')

    gc.collect()

    clear_system_memory()


if __name__ == "__main__":
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tqdm.write(f"[{now()}] Start systemu CapsNet. Urządzenie obliczeniowe: {str(device).upper()}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    encoder = CharLabelEncoder()
    num_classes = encoder.get_num_classes()
    current_zone_map = get_zone_mapping(encoder)

    model = CapsNet(num_classes).to(device)

    tqdm.write(f"[{now()}] Inicjalizacja połączenia ze zbiorem eMNIST (Lazy Loading).")
    base_emnist = datasets.EMNIST(root=DATA_ROOT_EMNIST, split='byclass', train=True, download=True, transform=EMNIST_TRANSFORM)
    cached_emnist_train = EMNISTWithContext(base_emnist, encoder, current_zone_map)

    # Logika wyszukiwania klas w nowej strukturze
    if not os.path.exists(CRNN_CUSTOM_SAMPLES):
        print(f"[{now()}] Folder bazowy {CRNN_CUSTOM_SAMPLES} nie istnieje. Używam domyślnej normalizacji eMNIST.")
        my_mean, my_std = [0.1307], [0.3081]
    else:
        # Sprawdzamy, czy folder faktycznie zawiera podfolder 'pure', na którym liczymy statystyki
        actual_classes = [
            d for d in os.listdir(CRNN_CUSTOM_SAMPLES)
            if os.path.isdir(os.path.join(CRNN_CUSTOM_SAMPLES, d)) and d != 'dummy'
        ]

        if not actual_classes:
            print(f"[{now()}] Brak folderów znaków w {CRNN_CUSTOM_SAMPLES}. Fallback: parametry eMNIST.")
            my_mean, my_std = [0.1307], [0.3081]
        else:
            temp_ds = HardCharsDataset(
                root_dir=CRNN_CUSTOM_SAMPLES,
                encoder=CharLabelEncoder(),
                case_type="pure",
                transform=base_transform
            )

            if len(temp_ds) > 0:
                # Tworzymy loader
                temp_loader = DataLoader(
                    temp_ds,
                    batch_size=64,
                    shuffle=False,
                    collate_fn=safe_collate_fn,
                    num_workers=0
                )
                my_mean, my_std = calculate_stats(temp_loader)
                print(f"[{now()}] Obliczone statystyki dla {len(temp_ds)} wycinków PURE: Mean={my_mean:.4f}, Std={my_std:.4f}")
            else:
                print(f"[{now()}] Podfoldery 'pure' są puste. Fallback na eMNIST.")
                my_mean, my_std = [EMNIST_NORM_MEAN], [EMNIST_NORM_STD]

    # Stały zbiór testowy
    emnist_test_transform = v2.Compose([
        v2.ToImage(),
        v2.Lambda(prepare_emnist_sample),
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=EMNIST_NORM_MEAN, std=EMNIST_NORM_STD)
    ])

    hard_test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[my_mean], std=[my_std]) 
    ])

    ds_test_raw = datasets.EMNIST(root=DATA_ROOT_EMNIST, split='byclass', train=False,
                                  download=True, transform=emnist_test_transform)

    # Zbiór testowy eMNIST z wymuszoną zgodnością wymiarów
    translated_emnist = UniversalContextWrapper(EMNISTWithContext(ds_test_raw, encoder, current_zone_map))
    test_loader = DataLoader(
        dataset=translated_emnist,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Mapowanie flag
    flags = {
        1: os.path.join(CHECKPOINT_DIR, "completed_Sam_eMNIST.flag"),
        2: os.path.join(CHECKPOINT_DIR, "completed_Fine-tune+PURE.flag"),
        3: os.path.join(CHECKPOINT_DIR, "completed_Hard-mining_na_bledach_CRNN.flag")
    }

    best_cer = float('inf')
    final_save_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

    # Sprawdzamy od tyłu, która faza
    start_phase = 1
    if os.path.exists(flags[3]):
        start_phase = 4
    elif os.path.exists(flags[2]):
        start_phase = 3
    elif os.path.exists(flags[1]):
        start_phase = 2

    # Uruchomienie pętli faz
    if start_phase <= 3:
        for phase_id in range(start_phase, 4):
            run_phase(phase_id, model, device, encoder, current_zone_map, test_loader, num_classes, flags[phase_id])
    else:
        tqdm.write(f"[{now()}] Wszystkie fazy treningu zostały już ukończone.")

    #  Wczytanie najlepszego modelu zapisanego w CHECKPOINT_DIR
    if os.path.exists(final_save_path):
        try:
            tqdm.write(f"[{now()}] Wykryto istniejący model. Wczytywanie stanu.")
            checkpoint = torch.load(final_save_path, map_location=device)

            if checkpoint:
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'], strict=False)
                    best_cer = checkpoint.get('best_cer', float('inf'))
                    tqdm.write(f"[{now()}] Wagi załadowane. Rekord CER do pobicia: {best_cer:.2f}%")

                tqdm.write(f"[{now()}] Planowany INITIAL_LR dla tej fazy: {INITIAL_LR}")
        except Exception as e:
            tqdm.write(f"[{now()}] Nie udało się wczytać pliku: {e}")
            checkpoint = {}

    model.eval()

    # Ewaluacja na ogólnym zbiorze testowym eMNIST
    tqdm.write(f"[{now()}] Testowanie na eMNIST.")
    emnist_cer, emnist_acc = calculate_cer_on_hard_cases(model, test_loader, device, encoder, "EMNIST Test")

    hard_data_path = os.path.join(CRNN_CUSTOM_SAMPLES, "hard_case")

    # Ładowanie zbioru HardCharsDataset
    if os.path.exists(hard_data_path) and len(os.listdir(hard_data_path)) > 0:
        ds_hard_final = HardCharsDataset(
            root_dir=CRNN_CUSTOM_SAMPLES,
            encoder=encoder,
            case_type="hard_case",
            transform=hard_test_transform
        )

        if len(ds_hard_final) > 0:
            final_hard_loader = DataLoader(
                ds_hard_final,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            tqdm.write(f"[{now()}] Testowanie na pełnym zbiorze wątpliwych dla CRNN.")
            hard_cer, hard_acc = calculate_cer_on_hard_cases(model, final_hard_loader, device, encoder, "Poprawa wątpliwości CRNN")
        else:
            hard_cer, hard_acc = 0.0, 0.0
            print("Zbiór hard_case został zainicjowany, ale jest pusty.")
    else:
        hard_cer, hard_acc = 0.0, 0.0
        print("Brak folderu hard_case lub brak w nim danych.")

    # Wyświetlenie tabeli wyników
    print(f"RAPORT SKUTECZNOŚCI")
    print(f" Zbiór eMNIST               - CER: {emnist_cer * 100:.2f}% | Celność: {emnist_acc:.2f}%")
    print(f" Zbiór wątpliwych dla CRNN  - CER: {hard_cer * 100:.2f}% | Celność: {hard_acc:.2f}%")

    with open(os.path.join(OUTPUT_ROOT, "final_results_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Data raportu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" eMNIST         - CER: {emnist_cer * 100:.2f}%, ACC: {emnist_acc:.2f}%\n")
        f.write(f" Problematyczne - CER: {hard_cer * 100:.2f}%, ACC: {hard_acc:.2f}%\n")

    # Rezerwacja nazwy na początku sekcji wynikowej
    final_hard_loader = locals().get('final_hard_loader', None)

    # Wizualizacje
    print(f"[{now()}] Generowanie wizualizacji testowych.")
    visual_test(model, test_loader, device, encoder)

    # Macierz pomyłek dla eMNIST
    os.makedirs(MATRIX_PATH, exist_ok=True)
    emnist_plot_path = os.path.join(MATRIX_PATH, "confusion_matrix_emnist.png")

    print(f"[{now()}] Generowanie macierzy pomyłek eMNIST.")
    generate_confusion_matrix(model, test_loader, device, encoder, emnist_plot_path)

    # Dodatkowa macierz pomyłek dla trudnych przypadków
    if hard_cer > 0:
        if isinstance(final_hard_loader, DataLoader):
            # Sprawdzamy, czy dataset nie jest pusty (bezpieczny dostęp)
            dataset = final_hard_loader.dataset
            if dataset is not None and len(dataset) > 0:  # type: ignore
                hard_plot_path = os.path.join(MATRIX_PATH, "confusion_matrix_hard_cases.png")

                print(f"[{now()}] Generowanie macierzy pomyłek dla zbioru Hard Mining.")
                generate_confusion_matrix(
                    model=model,
                    loader=final_hard_loader,
                    device=device,
                    encoder=encoder,
                    save_path=hard_plot_path
                )
        else:
            print(f"[{now()}] Pominięto macierz Hard Mining: loader jest pusty lub nie istnieje.")

""" systemd-inhibit docker run -it --rm --name trening_capsnet \
    --gpus all \
    --ipc=host \
    --network=host \
    --shm-size=16gb \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/app \
    -v /home/marek/.cache/torch:/root/.cache/torch \
    -v /home/marek/OCR/HandwrittenTextRecognition/Data:/app/data:rw \
    -v /home/marek/OCR/HandwrittenTextRecognition/output_data:/app/output_data \
    -v /home/marek/OCR/HandwrittenTextRecognition/Models:/app/Models \
    hcr-resnet-crnn \
    bash -c "python3 -m pip install --no-cache-dir langdetect shapiq && python3 -B Models/DeepCapsNetCharRecognition.py" """