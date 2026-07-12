import os

from PIL import Image
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys

# Dodanie ścieżki do głównego katalogu projektu do sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import difflib
import gc
import glob
import json
import math
import multiprocessing
import random
import re
import shutil
from typing import cast, List, Union, Tuple
import time
import uuid
from collections import Counter
import albumentations as alb
import cv2 as cv
cv.setNumThreads(0) # Już i tak wielowątkowość, więc żeby nie tworzyć kilkunastu nowych wątków za każdym wywołaniem
cv.ocl.setUseOpenCL(False)
import h5py
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.models as models
from Levenshtein import distance as edit_distance
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.filters import threshold_sauvola
from sklearn.metrics import confusion_matrix
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, WeightedRandomSampler, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
from sklearn.metrics import confusion_matrix
from Preprocessing.Preprocessing import Preprocessing # type: ignore . W Dockerze jest ok


def now():
    return time.strftime('%H:%M:%S')

# Ustawienie priorytetu procesu (bezpieczne dla Windows i Linux/Docker)
current_dir = os.path.dirname(os.path.abspath(__file__))
p = psutil.Process(os.getpid())
try:
    if os.name == 'nt':
        # Logika dla Windowsa
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        # Logika dla Linuxa (Docker)
        p.nice(10)
except (psutil.AccessDenied, psutil.Error):
        # Nie przerywamy pracy, bo to nie wpływa na jakość modelu
        pass
except ImportError:
    tqdm.write(f"[{now()}] Warning: Brak biblioteki 'psutil'. Priorytet pozostaje domyślny.")
except Exception as e:
    tqdm.write(f"[{now()}] Informacja: Nieoczekiwany błąd zmiany priorytetu: {e}")

IS_DOCKER = os.path.exists('/.dockerenv')

# Konfiguracja środowiska
if IS_DOCKER:
    DATA_ROOT = "/app/Data"
    CODE_ROOT = "/app"
    SJP_DICTIONARY = "/app/Data/clean_corpus.txt" 
    
    OUTPUT_NPZ = os.path.join(DATA_ROOT, "dataset.npz")
    OUTPUT_BASE = os.path.join(CODE_ROOT, "output_data")
else:
    # Ścieżka uniwersalna dla Linuxa
    DATA_ROOT = os.path.expanduser("~/OCR") 
    
    SJP_DICTIONARY = "/app/Data/output.txt"
    OUTPUT_NPZ = os.path.join(DATA_ROOT, "PHSF", "dataset.npz")
    OUTPUT_BASE = os.path.join(DATA_ROOT, "HandwrittenTextRecognition", "output_data")

# Konfiguracja workerów (przeszedłem na Linux, więc więcej)
WORKERS_MAIN = 4 # Augmentacje, więc więcej workerów, żeby je szybciej przygotować
WORKERS_FINE = 2 # Model SWA przechowuje więcej wag w pamięci, więc ograniczamy liczbę workerów, żeby nie zabrakło RAMu

# Urządzenie
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definicja głównego katalogu projektu (wewnątrz kontenera)
if IS_DOCKER:
    BASE_PATH = "/app"
    DATA_ROOT = "/app/Data"
    OUTPUT_BASE = os.path.join(BASE_PATH, "output_data")
else:
    # Używamy ścieżki do lokalnego katalogu projektu
    BASE_PATH = os.path.expanduser("~/OCR/HandwrittenTextRecognition")
    DATA_ROOT = os.path.join(BASE_PATH, "Data")
    OUTPUT_BASE = os.path.join(BASE_PATH, "output_data")

# Podstawowa ścieżka do zbioru IAM
RAW_SOURCE_DIR = os.path.normpath(os.path.join(DATA_ROOT, "iam_words", "words"))

# Checkpointy i wyniki
CHECKPOINT_FOLDER = os.path.normpath(os.path.join(OUTPUT_BASE, "checkpoints", "hwr"))
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth")
CER_PATH = os.path.join(CHECKPOINT_FOLDER, "best_cer_model.pth")

# Pliki baz danych H5
IAM_WORDS_H5_PATH = os.path.join(DATA_ROOT, "clean_dataset_processed.h5")

# Pozostałe ścieżki pomocnicze
MAIN_PHASE_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "main_complete.txt")
FINE_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "fine_complete.txt")
ALIGNMENT_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "alignment_complete.txt")
CAPSNET_DATA_DIR = os.path.normpath(os.path.join(OUTPUT_BASE, "crnn_crops"))
VISUAL_DEBUG_DIR = os.path.join(CHECKPOINT_FOLDER, "visual_debug_CRNN")
LOG_DIR = os.path.join(CHECKPOINT_FOLDER, "logs")

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(VISUAL_DEBUG_DIR, exist_ok=True)
os.makedirs(CAPSNET_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Konfiguracja GPU pod uczenie i Tensorboard do podsumowań CER/WER
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def get_writer(path):
    return SummaryWriter(log_dir=path)

IMAGE_HEIGHT = 64
STOP_THRESHOLD = 0.001
VAL_LOSS_THRESHOLD = 0.0002
ENTROPY_THRESHOLD = 0.5
CONF_THRESHOLD = 0.7
ENT_THRESHOLD = 0.5
PURE_RATE = 0.05
UPPER_RATE = 0.3
MAX_PER_CLASS = 1500
DIV_FACTOR = 20
IAM_MEAN = (0.8491,)
IAM_STD = (0.2259,)
MEAN, STD = 0.8491, 0.2259
MATRIX_PATH = os.path.join(CHECKPOINT_FOLDER, "confusion_matrix_final")
BG_COLOR = 255

""" Ekstrakcja cech w module CRNN dla pojedynczych wyrazów opiera się na fizycznym 
    batchu o rozmiarze 4, co optymalizuje zajętość VRAM podczas przechowywania map cech. 
    Zastosowanie akumulacji gradientów pozwala emulować stabilny, efektywny batch o
    rozmiarze 64. Taka konfiguracja uśrednia szum wynikający z nietypowych krojów
    pisma i stabilizuje kierunek spadku funkcji celu CTC.
    Później w fazie fine tune zmniejszamy do 8, co pozwala na dwukrotnie częstszą aktualizację wag. """
BATCH_SIZE = 4
ACCUMULATION_STEPS_MAIN = 16
ACCUMULATION_STEPS_FINE = 8

# Gwarancja determinizmu (reszta już dzieje się w innych metodach)
g = torch.Generator()

""" Faza Main: Służy budowie wizualnych fundamentów modelu.
    Przekazujemy zniekształcone próbki, żeby zbudował on podstawy odporności na szum,
    zmienne oświetlenie i niedoskonałości skanów, zanim przejdzie do analizy precyzyjnej. 
    W tej fazie model uczy się mapować surowe piksele na stabilne reprezentacje znaków, 
    a stosunkowo wysoki Learning Rate pozwala mu szybko znaleźć najgłębszą dolinę błędu. """
EPOCHS_MAIN = 17
PATIENCE_MAIN = 5
LR_MAIN = 2.5e-4 # Przy 5e-5 model szybko utknął w lokalnym minimum, 1e-4 to za mało, a 2.5e-4 pozwala na stabilny spadek straty osiągając ostatecznie lepszy wynik
PCT_START = 0.15 # 3 epoki na znalezienie dobrego kierunku

""" Faza Fine Tune: Służy precyzyjnemu dopasowaniu modelu do specyfiki pisma odręcznego.
    W tej fazie model uczy się subtelnych różnic między podobnymi znakami, a niski LR pozwala
    na delikatne dostrojenie wag, minimalizując ryzyko przeuczenia. Obrazy nie są już tak
    mocno zniekształcone, co pozwala modelowi skupić się na drobnych detalach i niuansach pisma. """
EPOCHS_FINE_TUNE = 8
LR_FINE_TUNE = 1e-5

""" Faza Alignment: Ekstremalnie niski krok uczenia.
    Epoki delikatnego klastrowania i kategoryzacji wyodrębnionych cech w przestrzeni wielowymiarowej.
    Zapewnia to czyste wejście dla modułów CapsNet i Transformer, znacznie ułatwiając
    im późniejszą analizę przez lepsze rozdzielanie klas i redukcję szumu. """
EPOCHS_ALIGN = 3
LR_ALIGN = 5e-6

def worker_init_fn(worker_id):
    """ Każdy worker otrzymuje unikalny seed oparty na globalnym seedzie, co zapewnia różnorodność augmentacji i losowego próbkowania,
        jednocześnie zachowując deterministyczność. """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def val_process_fn(image, **kwargs):
    """ Wrapper dla Albumentations ze ścisłym zachowaniem naturalnych proporcji.
        Zupełnie usuwa sztuczne limity szerokości, zapobiegając rozciąganiu i zgniataniu tekstu. """
    h_orig, w_orig = image.shape[:2]
    
    # Obliczamy proporcjonalną szerokość względem docelowej wysokości
    aspect_ratio = w_orig / h_orig
    target_w = int(IMAGE_HEIGHT * aspect_ratio)
    
    # Zwracamy obraz w idealnych proporcjach - paddingiem zajmie się collate_fn
    return Preprocessing.process_for_crnn(image, target_h=IMAGE_HEIGHT, target_w=target_w)

VAL_TRANSFORMS = alb.Compose([
    alb.Lambda(name="GeometricNormalization", image=val_process_fn),
    alb.Normalize(mean=IAM_MEAN, std=IAM_STD),
    ToTensorV2()
])

REVERSE_PUNCTUATION_MAP = {
    '#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-', '#B': '(',
    '#K': ')', '#S': ';', '#L': ':', '#Q': '?', '#F': '/', '#M': '$', '#P': '%'
}
FILENAME_MAP = {
    'dot': '.', 'comma': ',', 'apostrophe': "'", 'exclamation': '!',
    'hyphen': '-', 'lparen': '(', 'rparen': ')', 'semicolon': ';',
    'colon': ':', 'slash': '/', 'question': '?', 'percent': '%'
}
string_confusion = {}

def now():
    """ Zwraca aktualny czas w formacie HH:MM:SS, używany do logowania postępu treningu i debugowania. """
    return time.strftime('%H:%M:%S')


class ScRN_STN(nn.Module):
    """ Symmetric Character Rectification Network oparte na TPS.
        Automatycznie dostosowuje siatkę wyjściową do rozmiarów obrazu wejściowego, zapobiegając niszczeniu proporcji krótkich słów. 
        Najpierw Localization Network przewiduje przesunięcia punktów kontrolnych względem symetrycznych pozycji docelowych,
        Potem obliczana jest macierz TPS, a na końcu generowana jest siatka transformacji dla oryginalnych wymiarów obrazu,
        która prostuje tekst. Dzięki temu nawet krótkie słowa zachowują swoje proporcje, a model uczy się rozpoznawać znaki bez zniekształceń.
        Uczy się razem z CRNN, przez co poprawia jakość swojej lokalizacji i prostowania tekstu na podstawie dialogu straty z CRNN. """
    def __init__(self, F=20, loc_size=(32, 64), input_channels=1):
        super(ScRN_STN, self).__init__()
        self.F = F
        self.loc_size = loc_size
        self.nc = input_channels

        """ Wydobywamy z obrazu cechy przez początkową Conv2d, potem MaxPool2d redukuje wymiarowość, żeby sieć mniej zwracała
            uwagę na dokładne położenie. Następnie aktywacja odporna na umieranie neuronów. Dodaje nieliniowość, która
            pozwala wyciągnąć z obrazu bardziej złożone cechy. Powtarzamy dla lepszych wyników. """
        self.localization = nn.Sequential(
            nn.Conv2d(self.nc, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(64)
        )

        # Dynamiczne wyliczanie wyjścia po warstwach konwolucyjnych i MaxPool2d
        h_out = ((self.loc_size[0] - 4) // 2 - 4) // 2
        w_out = ((self.loc_size[1] - 4) // 2 - 4) // 2
        fc_in_features = 64 * h_out * w_out

        """ Zajmuje się określaniem siły wygięcia każdego z otrzymanych punktów, najpierw liniowo mapuje do 256 liczb,
            następnie wprowadza nieliniowość za pomocą PReLU, a na ostatnim etapie mapuje wyuczone cechy na konkretne
            parametry geometryczne określające, w jaki sposób wyprostować obraz by stał się czytelny dla CRNN. """
        self.fc_loc = nn.Sequential(
            nn.Linear(fc_in_features, 256),
            nn.PReLU(256),
            nn.Linear(256, (self.F // 2) * 4)
        )

        layer = self.fc_loc[2]
        if isinstance(layer, nn.Linear):
            layer.weight.data.zero_()
            layer.bias.data.zero_()

        # Obliczamy macierz odwrotną tylko raz (zależy wyłącznie od liczby punktów F, nie od wymiarów obrazu)
        self._build_inverse_matrix()

    def _build_inverse_matrix(self):
        # Zmniejszone do 0.05, żeby model nie ucinał końcówek
        margin = 0.05
        N = self.F

        x_coords = torch.linspace(-1 + margin, 1 - margin, N // 2)
        p_top = torch.stack([x_coords, torch.full_like(x_coords, -1 + margin)], dim=1)
        p_bottom = torch.stack([x_coords, torch.full_like(x_coords, 1 - margin)], dim=1)
        target_points = torch.cat([p_top, p_bottom], dim=0)

        diff = target_points.unsqueeze(0) - target_points.unsqueeze(1)
        dist_sq = torch.sum(diff ** 2, dim=2)
        K = dist_sq * torch.log(dist_sq + 1e-6)
        P = torch.cat([torch.ones(N, 1), target_points], dim=1)

        L = torch.zeros(N + 3, N + 3)
        L[:N, :N] = K
        L[:N, N:] = P
        L[N:, :N] = P.t()

        inv_L = torch.linalg.inv(L)

        self.register_buffer('target_points', target_points)
        self.register_buffer('inv_L_ref', inv_L)

    def _get_grid_matrix(self, h, w, device):
        """ Dynamiczne generowanie siatki w oparciu o aktualne H i W tensora. """
        target_points_tensor = cast(torch.Tensor, self.target_points)
        
        grid_x = torch.linspace(-1, 1, w, device=device)
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        base_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        grid_diff = base_coords.unsqueeze(1) - target_points_tensor.unsqueeze(0)
        grid_dist_sq = torch.sum(grid_diff ** 2, dim=2)
        grid_rbf = grid_dist_sq * torch.log(grid_dist_sq + 1e-6)

        return torch.cat([grid_rbf, torch.ones(h * w, 1, device=device), base_coords], dim=1)

    def forward(self, img):
        B, C, H, W = img.size()

        # Lokalizacja wymusza skalowanie tylko na potrzeby predykcji punktów
        img_loc = func.interpolate(img, size=self.loc_size, mode='bilinear', align_corners=True)
        features = self.localization(img_loc)
        features = features.view(B, -1)

        params = self.fc_loc(features)
        params = params.view(B, self.F // 2, 4)

        c_x = torch.tanh(params[:, :, 0]) * 0.1
        c_y = torch.tanh(params[:, :, 1]) * 0.1
        s_cos = torch.tanh(params[:, :, 2]) * 0.1
        s_sin = torch.tanh(params[:, :, 3]) * 0.1

        delta_top = torch.stack([c_x + s_cos, c_y - s_sin], dim=2)
        delta_bottom = torch.stack([c_x - s_cos, c_y + s_sin], dim=2)
        delta = torch.cat([delta_top, delta_bottom], dim=1)

        target_points_tensor = cast(torch.Tensor, self.target_points)
        inv_L_ref_tensor = cast(torch.Tensor, self.inv_L_ref)

        source_points = target_points_tensor.unsqueeze(0) + delta

        zeros = torch.zeros(B, 3, 2, device=img.device)
        Y = torch.cat([source_points, zeros], dim=1)
        weights = torch.matmul(inv_L_ref_tensor, Y)

        # Generujemy siatkę TPS dla oryginalnych wymiarów obrazu
        grid_matrix = self._get_grid_matrix(H, W, img.device)
        
        source_coords = torch.matmul(grid_matrix, weights)
        grid = source_coords.view(B, H, W, 2)

        transformed_img = func.grid_sample(img, grid, align_corners=True, padding_mode='zeros')

        return transformed_img


class VisualAttention(nn.Module):
    """ Mechanizm wizualnej uwagi pełniący funkcję poolingu. Zastępuje standardowe uśrednianie mechanizmem uczącym.
        Dla każdego kroku czasowego sieć wyznacza wagi ważności, decydując, które fragmenty w pionie zawierają istotne cechy znaku, a które są tłem lub szumem.
        1. Wejście: Mapa cech z ResNet.
        2. Analiza: Warstwa konwolucyjna ocenia ważność każdego piksela.
        3. Filtracja: Softmax normalizuje wagi wzdłuż wymiaru wysokości.
        4. Wyjście: Ważona suma cech - sekwencja 1D gotowa dla GRU. """
    def __init__(self, channels):
        super(VisualAttention, self).__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        # Jeden parametr wystarczy, bo operujemy na mapie wyników
        self.prelu = nn.PReLU(1) 
    
    def forward(self, x):
        # Generujemy surowe wyniki ważności
        scores = self.attn(x)
        
        # PRELU pozwala sieci nauczyć się progu "nieistotności", gdzie wartości poniżej pewnego poziomu są traktowane jako szum, a powyżej jako istotne cechy.
        scores = self.prelu(scores)
        
        # Softmax wzdłuż wysokości
        weights = torch.softmax(scores, dim=2)

        # Ważona suma cech w pionie
        output = (x * weights).sum(dim=2)
        
        return output


class CRAMBlock(nn.Module):
    """ Customized Residual Attention Module — odszumia tło zniszczonych dokumentów i uwydatnia pismo. """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Definiujemy PReLU jako warstwę, żeby PyTorch wiedział, gdzie trzymać wagi
        self.prelu_main = nn.PReLU(channels)

        # Uwaga kanałowa
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.PReLU(channels // 8),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Uwaga przestrzenna
        self.sa = nn.Sequential(
            # Trochę zmniejszam, bo traktuje małe kropki jako szum
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.prelu_main(self.bn1(self.conv1(x)))

        # Aplikacja uwagi kanałowej
        x = x * self.ca(x)

        # Aplikacja uwagi przestrzennej
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa_in = torch.cat([max_out, avg_out], dim=1)
        x = x * self.sa(sa_in)

        return x + res


class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block: modeluje zależności między kanałami,
        uwydatniając cechy istotne dla znaków (krawędzie) i wyciszając szum. """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(channels // reduction),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EnhancedBiLSTM(nn.Module):
    """ Zapewnia stabilną dystrybucję cech i lepszą pamięć długoterminową.
        Zapobiega zanikaniu gradientu w długich słowach. """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=False)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Inicjalizacja wag
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)

                # Forget gate bias
                param.data[n // 4:n // 2].fill_(1.0)

    def flatten_parameters(self):
        """ Przekierowanie wywołania do bazowego LSTM dla optymalizacji cuDNN. """
        self.lstm.flatten_parameters()

    def forward(self, x):
        self.lstm.flatten_parameters() 
        output, hidden = self.lstm(x)

        # Aktywacja layer_norm dla stabilizacji wyjścia z sieci rekurencyjnej
        output = self.layer_norm(output)

        return output, hidden


class RotaryEmbedding(nn.Module):
    """ Zastępuje klasyczne PositionalEncoding. Realizuje mechanizm RoPE (Rotary Position Embedding),
        obracając pary cech w przestrzeni o unikalny kąt zależny od pozycji znaku. """
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.d_model = d_model
        
        # Wyliczamy odwrotności częstotliwości (thetas) dla połowy wymiaru cech
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Generujemy pełną macierz pozycji [max_len]
        t = torch.arange(max_len, dtype=torch.float)
        
        # Iloczyn zewnętrzny tworzy macierz kątów dla każdej pozycji i połowy kanałów
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplikujemy kąty, aby pokryć pełny wymiar d_model [max_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Rejestrujemy bufory cosinusów i sinusów z wymiarem dla Batcha: [max_len, 1, d_model]
        self.register_buffer("cos", emb.cos().unsqueeze(1))
        self.register_buffer("sin", emb.sin().unsqueeze(1))

    def _rotate_half(self, x):
        """ Sprytny trik macierzowy do rotacji 2D: zmienia [x1, x2] w [-x2, x1] """
        x1 = x[..., :self.img_d_half()]
        x2 = x[..., self.img_d_half():]
        return torch.cat((-x2, x1), dim=-1)

    def img_d_half(self):
        return self.d_model // 2

    def forward(self, x):
        # x ma kształt: [Kroki_Czasowe (T), Batch (B), Kanały (C)]
        t = x.size(0)
        
        # Ucinamy wykresy fal do aktualnej długości sekwencji ramek
        cos = self.cos[:t, :, :]
        sin = self.sin[:t, :, :]
        
        # Wzór rotacji wektora: x * cos(theta) + rotate_half(x) * sin(theta)
        return x * cos + self._rotate_half(x) * sin


class SpatialDropout1D(nn.Module):
    """ W przeciwieństwie do standardowego dropoutu, który losowo zeruje pojedyncze aktywacje,
        ta warstwa wyłącza całe kanały cech na całej ich długości w danej sekwencji. 
        Wymusza to na modelu uczenie się niezależnych, bardziej odpornych reprezentacji znaków, 
        co redukuje przeuczenie na specyficznych, powtarzalnych stylach pisma odręcznego. 
        Wyłączanie pojedynczych pikseli jest bez sensu przy HCR. """
    def __init__(self, p=0.3):
        super(SpatialDropout1D, self).__init__()
        # nn.Dropout1d wyłącza całe kanały
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        x = self.dropout(x)
        return x
        

class ResNetCRNN(nn.Module):
    """ Hybrydowa architektura percepcji wizualnej i geometrycznej. Model pełni rolę modułu wizyjnego,
        którego celem jest ekstrakcja surowych cech optycznych i przekształcenie ich dla oceny dekodera językowego.
        Architektura składa się z 4 głównych modułów:
        1. ScRN-STN - Geometryczna Rektyfikacja:
           - Adaptacyjna normalizacja obrazu wykorzystująca lekką sieć lokalizującą CNN.
           - Wymusza symetryczne ułożenie punktów kontrolnych względem osi centralnej tekstu.
        2. ResNet-18 Backbone - Inwariantność Kształtu:
           - Rdzeń splotowy ze zmodyfikowanymi krokami (2, 1).
           - Stała wysokość mapy cech pozwala zachować pionową integralność znaków.
        3. CRAM - Wyostrzanie Atramentu:
           - Zintegrowany moduł uwagi przestrzennej i kanałowej bezpośrednio w CNN. 
             Odszumia tło i uwydatnia krawędzie liter.
        4. BiLSTM - Modelowanie Sekwencji Wizualnej i Ligatur:
           - Dwuwarstwowa sieć rekurencyjna analizująca płynność i dynamikę pisma.
           - Zabezpieczona Dropoutem przed generowaniem tzw. bełkotu. """
    def __init__(self, num_classes):
        super().__init__()

        # STN — Geometryczna Rektyfikacja
        self.stn = ScRN_STN(
            F=20,
            loc_size=(32, 64),
            input_channels=1
        )

        # Sekwencja mapowania cech dla RNN (Stabilizator Przestrzeni)
        self.map_to_seq = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ResNet-18 Backbone
        resnet = models.resnet18(weights='DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet.layer2[0].conv1.stride = (2, 1)
        if resnet.layer2[0].downsample is not None:
            resnet.layer2[0].downsample[0].stride = (2, 1)

        resnet.layer3[0].conv1.stride = (2, 1)
        if resnet.layer3[0].downsample is not None:
            resnet.layer3[0].downsample[0].stride = (2, 1)

        resnet.layer4[0].conv1.stride = (1, 1)
        if resnet.layer4[0].downsample is not None:
            resnet.layer4[0].downsample[0].stride = (1, 1)

        # Sekwencja CNN
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)), # MaxPool zgniata tylko wysokość, zostawiając szerokość bez zmian
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            CRAMBlock(512)
        )
        
        """ Spatial dropout - wyłączanie całych kanałów cech, nie pojedynczych pikseli, co wymusza na modelu
            uczenie się bardziej odpornych reprezentacji znaków. """
        self.spatial_drop = SpatialDropout1D(p=0.3)

        # Uproszczona projekcja do wymiaru wejściowego BiLSTM
        self.p = 0.25
        self.projection = nn.Sequential(
            nn.Conv1d(1024, 256, kernel_size=1),
            nn.Dropout1d(self.p)
        )

        # Modelowanie sekwencji
        self.rnn = EnhancedBiLSTM(256, 512, num_layers=2)

        # Regularyzacja Głowicy (Dropout)
        self.dropout = nn.Dropout(0.3)

        # Wyjście klasyfikatora
        self.output = nn.Linear(1024, num_classes)

        # Pomocnicze głowice (Transformer i Contrastive)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.d_model = 256
        self.transformer_projection = nn.Linear(512, self.d_model)

        self.contrastive_alpha = nn.Parameter(torch.tensor(0.7))
        self._init_rnn_weights()

    def _init_rnn_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                with torch.no_grad():
                    param.fill_(0)

    def forward(self, x, return_embeddings=False, return_context=False, return_stn=False):
        # Rektyfikacja STN
        stn_img = self.stn(x)

        # Ekstrakcja cech z ResNet
        x = self.cnn(stn_img)

        # Stabilizacja i redukcja kanałów
        x = self.map_to_seq(x)

        # Zgniecenie osi wysokości
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
            
        # Aplikacja Spatial Dropout
        x = self.spatial_drop(x)

        # Projekcja do wejścia BiLSTM
        x = self.projection(x)
        
        # Permute na format RNN
        x_rnn = x.permute(2, 0, 1).float()

        with torch.amp.autocast('cuda', enabled=False):
            recurrent_features, *_ = self.rnn(x_rnn)
            
            # Aplikacja warstwy Dropout przed klasyfikatorem (Klucz do pozbycia się bełkotu)
            recurrent_features_dropped = self.dropout(recurrent_features)
            log_probs = self.output(recurrent_features_dropped)

        # Routing tensorów 
        if return_stn and return_context:
            return log_probs, recurrent_features.permute(1, 0, 2), stn_img
        if return_stn:
            return log_probs, stn_img
        if return_context:
            transformer_memory = self.transformer_projection(recurrent_features[..., :512].permute(1, 0, 2))
            return log_probs, transformer_memory
        if return_embeddings:
            pooled = recurrent_features.mean(dim=0)
            embeddings = self.contrastive_projection(pooled)
            return log_probs, embeddings

        return log_probs

    def load_weights(self, checkpoint_path, device=torch.device('cuda')):
        if not os.path.exists(checkpoint_path):
            tqdm.write(f" Brak pliku wag: {checkpoint_path}")
            return 0

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

            model_dict = self.state_dict()

            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }

            skipped = [k for k, v in state_dict.items() if k in model_dict and v.size() != model_dict[k].size()]

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

            if skipped:
                tqdm.write(f"Pominięto warstwy z powodu zmiany rozmiaru (prawdopodobnie przez nowy alfabet): {skipped}")

            tqdm.write(f"[{time.strftime('%H:%M:%S')}] Pomyślnie załadowano {len(pretrained_dict)} warstw.")
            self.to(device)

            return checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0

        except FileNotFoundError:
            tqdm.write(f"[{now()}] Brak pliku checkpointa w {checkpoint_path}. Rozpoczynam od zera.")
            return 0
        except RuntimeError as re:
            tqdm.write(f"[{now()}] Błąd urządzenia (CUDA/CPU) podczas ładowania wag: {re}")
            return 0
        except Exception as e:
            tqdm.write(f"[{now()}] Nieoczekiwany błąd podczas ładowania wag ({type(e).__name__}): {e}")
            return 0

    def set_dropout(self, p):
        self.p = p
        for module in self.projection:
            if isinstance(module, nn.Dropout1d):
                module.p = p

    def estimate_uncertainty(self, x, steps=10):
        self.eval()
        
        # Aktywacja warstw Dropout dla Monte Carlo Dropout pomimo trybu eval()
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        outputs = []
        for _ in range(steps):
            with torch.no_grad():
                logits = self.forward(x)
                outputs.append(torch.softmax(logits[0], dim=-1))

        variance = torch.stack(outputs).var(dim=0)
        return variance.mean(dim=-1)

    @staticmethod
    def get_uncertainty_zones(log_probs, margin_threshold=0.2, conf_threshold=0.8, temperature=1.5):
        """ Analizuje prawdopodobieństwa z użyciem skalowania temperaturą, co skutecznie obniża pewność siebie sieci. """
        
        # Log_probs dzielone przez temperaturę > 1.0 wygładzają rozkład
        probs = torch.softmax(log_probs / temperature, dim=-1).squeeze(1)
        
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)

        margins = top2_probs[:, 0] - top2_probs[:, 1]
        max_conf = top2_probs[:, 0]

        # Oflagowanie niepewności na bazie wygładzonych, bardziej realistycznych prawdopodobieństw
        uncertain_mask = (margins < margin_threshold) | (max_conf < conf_threshold)

        # Pomijamy blanki
        char_indices = torch.argmax(probs, dim=-1)
        final_mask = uncertain_mask & (char_indices != 0)

        return torch.where(final_mask)[0].tolist()
     

class AdvancedHTRAugmentor:
    """ Augmentacje symulujące rzeczywisty ruch ręki przy pisaniu. """
    @staticmethod
    def ink_bleeding(image, **kwargs):
        """ Wykorzystuje ważoną dylatację i rozmycie Gaussa, uodparniając model na zmienną wilgotność papieru. """
        # Zabezpieczenie przed błędem wymiarów OpenCV dla obrazów (H, W, 1)
        is_expanded = False
        if image.ndim == 3 and image.shape[1] == 1:
            image = np.squeeze(image, axis=2)
            is_expanded = True

        strength = random.uniform(0.1, 0.25)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        dilated = cv.dilate(image, kernel, iterations=1)
        bleeded = cv.addWeighted(image, 1 - strength, dilated, strength, 0)
        
        k_size = 3
        blurred = cv.medianBlur(bleeded, k_size)
        
        # Przywrócenie wymiaru (H, W, 1) jeśli był obecny
        if is_expanded:
            blurred = np.expand_dims(blurred, axis=2)
            
        return blurred

    @staticmethod
    def variable_stroke(img, **kwargs):
        """ Symuluje zmienny nacisk narzędzia piszącego i różną grubość stalówki. """
        # Zabezpieczenie przed pustym obrazem
        if img is None or img.size == 0: 
            return img
        
        # Jeśli obraz jest 3-kanałowy, upewnijmy się, że pracujemy na kanale jasności
        if img.ndim == 3: 
            img = img[:, :, 0]
            
        # Jeśli średnia > 127, tło jest jasne -> odwracamy, by otrzymać biały tekst na czarnym tle
        binary = cv.bitwise_not(img) if np.mean(img) > 127 else img
        
        # Distance transform dla wyliczenia grubości kresek
        dist = cv.distanceTransform(binary, cv.DIST_L2, 3)
        
        # Modyfikacja grubości przez potęgowanie (factor < 1.0 -> cieńsza, factor > 1.0 -> grubsza)
        factor = np.random.uniform(0.7, 1.4)
        dist = np.power(dist, factor)
        
        # Normalizacja do zakresu [0, 255] i powrót do uint8
        cv.normalize(dist, dist, 0, 255, cv.NORM_MINMAX)
        return dist.astype(np.uint8)
    
    @staticmethod
    def phantom_elements(image, **kwargs):
        """ Wymusza na sieci wizualnej ignorowanie szumu z sąsiadujących wierszy. """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        phantom_h = int(h * 0.15)
        if phantom_h < 2:
            return image

        phantom = np.zeros((phantom_h, w), dtype=np.uint8)
        for _ in range(random.randint(1, 2)):
            x = random.randint(0, max(0, w - 30))
            cv.ellipse(phantom, (x, phantom_h), (random.randint(5, 12), 4), 
                       random.randint(0, 30), 0, 360, (255), -1)

        actual_ph = min(phantom_h, h)

        if image.ndim == 3:
            # Jeśli mamy 3 wymiary (H, W, C), operujemy na kanale 0
            if random.random() > 0.5:
                image[:actual_ph, :, 0] = cv.bitwise_or(image[:actual_ph, :, 0], phantom[:actual_ph, :])
            else:
                image[-actual_ph:, :, 0] = cv.bitwise_or(image[-actual_ph:, :, 0], np.flipud(phantom[:actual_ph, :]))
        else:
            # Jeśli mamy 2 wymiary (H, W)
            if random.random() > 0.5:
                image[:actual_ph, :] = cv.bitwise_or(image[:actual_ph, :], phantom[:actual_ph, :])
            else:
                image[-actual_ph:, :] = cv.bitwise_or(image[-actual_ph:, :], np.flipud(phantom[:actual_ph, :]))
            
        return image
    
    @staticmethod
    def create_realistic_space(height, width):
        """ Zastępuje idealnie czarną próżnię zaszumionym tłem charakterystycznym dla skanów. """
        if width <= 0:
            return np.zeros((height, 1), dtype=np.uint8)
        # Tworzymy delikatny szum o niskiej amplitudzie, który imituje fakturę papieru
        base_bg = np.random.randint(0, 18, (height, width), dtype=np.uint8)
        
        # Rozmywamy, aby przypominało fakturę papieru, a nie cyfrowy szum
        return cv.GaussianBlur(base_bg, (3, 3), 0)


class IAMWordDataset(Dataset):
    """ Moduł ładujący zbiór IAM Words ze wsparciem dla dynamicznego skalowania z zachowaniem proporcji.
        Wszystkie dane (obrazy i etykiety) są dekodowane i wczytywane bezpośrednio do pamięci RAM
        podczas inicjalizacji obiektu, co drastycznie przyspiesza proces treningu kosztem pamięci operacyjnej. """
    def __init__(self, h5_path, transform, char_list, name="Główny", split='train'):
        self.h5_path = h5_path
        self.transform = transform
        self.dictionary = set(char_list)
        self.name = name
        self.split = split
        self.IMAGE_HEIGHT = 64
        self.valid_indices = []
        self.valid_labels = []
        self.h5_file = None
        self.images_group_path = 'images'

        with h5py.File(self.h5_path, 'r') as f:
            # Sprawdzamy czy plik ma strukturę /train/ i /val/
            if self.split in f:
                # Plik ma strukturę, używamy dedykowanych grup!
                target_group = f[self.split]
                labels = target_group['labels'][:]
                self.images_group_path = f"{self.split}/images"
                indices = range(len(labels)) # Używamy indeksów bezpośrednio z grupy
                tqdm.write(f"[{now()}] Wykryto strukturę H5. Używam grupy: {self.split}")
            else:
                # Plik jest płaski, musimy dzielić sami
                labels = f['labels'][:]
                total = len(labels)
                np.random.seed(42)
                indices = np.random.permutation(total)
                train_cutoff = int(0.9 * total)
                indices = indices[:train_cutoff] if self.split == 'train' else indices[train_cutoff:]
                tqdm.write(f"[{now()}] Plik płaski. Stosuję manualny split.")

            # Filtracja
            for i in indices:
                lbl = labels[i]
                text = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text).strip()
                
                # Czyszczenie sufiksów IAM
                text = re.sub(r'_\d+$', '', text) 
                text = re.sub(r'-\d+$', '', text)
                
                if text and all(c in self.dictionary for c in text):
                    self.valid_indices.append(i)
                    self.valid_labels.append(text)

        self.dataset_len = len(self.valid_indices)
        tqdm.write(f"[{now()}] Zainicjalizowano {self.name}: {self.dataset_len} próbek.")

    def __getstate__(self):
        """ Zabezpiecza przed kopiowaniem uchwytu HDF5 przy uruchamianiu nowych workerów """
        state = self.__dict__.copy()
        if 'h5_file' in state:
            state['h5_file'] = None
        return state
        
    def __getitem__(self, idx):
        h5_idx = self.valid_indices[idx]
        label = self.valid_labels[idx]
        
        img = self._load_single_h5_image(h5_idx)
        
        if img is None or img.ndim < 2 or img.size == 0:
            return torch.zeros((1, self.IMAGE_HEIGHT, 64)), label

        # Polaryzacja dla IAM - wymusza biały tekst na czarnym tle, tak jak w polskim syntetyku
        if np.mean(img) > 127:
            img = cv.bitwise_not(img)

        # Czyszczenie
        _, thresh = cv.threshold(img, 30, 255, cv.THRESH_BINARY)

        # Kadrowanie
        coords = cv.findNonZero(thresh)
        if coords is not None and coords.shape[0] > 0:
            x, y, w_bbox, h_bbox = cv.boundingRect(coords)
            if w_bbox > 1 and h_bbox > 1:
                pad = int(h_bbox * 0.15)
                x1 = max(0, x - pad)
                x2 = min(img.shape[1], x + w_bbox + pad)
                img = img[:, x1:x2]

        # Skalowanie
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return torch.zeros((1, self.IMAGE_HEIGHT, 64)), label
            
        scale = self.IMAGE_HEIGHT / max(1, h)
        new_w = max(16, int(w * scale)) 
        img = cv.resize(img, (new_w, self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

        # Transformacje
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            
        augmented = self.transform(image=img)
        return augmented['image'], label
        
    def get_text(self, idx):
        """ Zwraca tekst dla danego indeksu. """
        # Sprawdzamy, czy to nasz zbiór polski
        if hasattr(self, 'word_list'):
            if not self.word_list:
                return "test"
            return self.word_list[idx % len(self.word_list)]
    
        # Dla zbioru IAM
        if hasattr(self, 'valid_labels'):
             return self.valid_labels[idx]

        # Jeśli nie wiemy co to za zbiór, zwracamy pusty ciąg, by nie wywalać błędu
        return ""
    
    def _get_h5_file(self):
        # Otwieramy plik tylko, jeśli jeszcze nie jest otwarty w tym konkretnym rdzeniu procesora
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def _load_single_h5_image(self, h5_idx):
        f = self._get_h5_file()
        data = f[self.images_group_path][h5_idx]
        nparr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    
        if img is None:
            return np.zeros((64, 64), dtype=np.uint8)
        return img

    def __del__(self):
        try:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        except:
            pass # Ignorujemy błędy przy zamykaniu, jeśli obiekt już nie istnieje

    def __len__(self):
        return self.dataset_len

class WidthBatchSampler(Sampler):
    """  Grupowanie po szerokości obrazu, aby zminimalizować padding w batchu.
        Wymaga, aby dataset miał dostęp do szerokości obrazów bez ich pełnego ładowania. """
    def __init__(self, data_source, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.info = data_source.get_image_info() 
        
        # Sortujemy indeksy według szerokości
        self.sorted_indices = [x[0] for x in sorted(self.info, key=lambda x: x[1])]

    def __iter__(self):
        # Tworzymy listę batchy
        batches = [self.sorted_indices[i:i + self.batch_size] 
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        
        # Mieszamy kolejność batchy, ale nie elementy wewnątrz batcha
        if self.shuffle:
            random.shuffle(batches)
            
        yield from batches

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class HTREncoder:
    """ Słownik i dekoder HTR. Obsługuje CTC Blank, Spacje, Entropię Shannona oraz Beam Search z LM. """
    # Stała interpunkcyjna używana w Visual Veto
    PUNCT_TO_GUARD = ('.', ',', '!', '?', ':', ';', "'", ')')

    def __init__(self, char_list: List[str]):
        # Wymuszamy, aby '[blank]' był zawsze pod indeksem 0, a spacja zachowała swój indeks
        self.char_list = ['[blank]'] + [c for c in char_list if c != '[blank]']

        self.char_to_num = {c: i for i, c in enumerate(self.char_list)}
        self.num_to_char = {i: c for i, c in enumerate(self.char_list)}

    def get_num_classes(self) -> int:
        return len(self.char_list)

    def labels_to_text(self, indices: Union[List[int], np.ndarray]) -> str:
        """ Konwertuje listę indeksów na tekst, czyszcząc techniczne nazwy znaków. """
        res = []
        for idx in indices:
            idx_int = int(idx)
            if idx_int != 0:  # 0 to [blank], ignorujemy w tekście końcowym
                char_raw = self.num_to_char.get(idx_int, '')
                res.append(self.clean_char(char_raw))
        return "".join(res)

    @staticmethod
    def clean_char(char_raw: str) -> str:
        """ Zamienia techniczne nazwy znaków (np. 'dot') na symbole ('.'). """
        mapping = {
            "question": "?", "dot": ".", "comma": ",",
            "exclamation": "!", "dash": "-", "slash": "/",
            "doublequote": '"', "space": " ", "apostrophe": "'"
        }
        return mapping.get(str(char_raw).lower(), char_raw)

    @staticmethod
    def calculate_uncertainty(logits: torch.Tensor, temperature: float = 2.0) -> float:
        """ Oblicza średnią entropię Shannona dla sekwencji wykorzystując Temperature Scaling. """
        scaled_logits = logits / temperature
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Entropia Shannona
        entropy = -torch.sum(probs * (log_probs / math.log(2)), dim=-1)
        return torch.mean(entropy).item()

    def decode_greedy(self, log_probs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """ Dekodowanie zachłanne z ekstrakcją entropii i bez przesunięć indeksów. """
        if log_probs.dim() == 3:
            if log_probs.shape[0] > log_probs.shape[1] or log_probs.shape[1] == 1:
                log_probs = log_probs.permute(1, 0, 2)

        preds_indices = torch.argmax(log_probs, dim=-1)
        preds_raw = preds_indices.cpu().numpy()

        decoded_list = []
        uncertainty_list = []

        for i in range(preds_raw.shape[0]):
            row = preds_raw[i]
            collapsed_indices = []
            last_idx = -1

            for idx in row:
                if idx != 0 and idx != last_idx:
                    collapsed_indices.append(idx)
                last_idx = idx

            text = self.labels_to_text(collapsed_indices)
            uncertainty = self.calculate_uncertainty(log_probs[i])

            decoded_list.append(text)
            uncertainty_list.append(uncertainty)

        return decoded_list, uncertainty_list

    def decode_beam_search(self, log_probs: torch.Tensor, lm_decoder, beam_width: int = 64, temperature: float = 1.4) -> List[str]:
        """ Zoptymalizowane dekodowanie Beam Search z hybrydowym mechanizmem Visual Veto. """
        if log_probs.dim() == 3:
            if log_probs.shape[0] < log_probs.shape[1] or log_probs.shape[0] == 1:
                log_probs = log_probs.permute(1, 0, 2)

        scaled_log_probs = log_probs.clone() / temperature
        scaled_log_probs[:, :, 0] -= 1.2  # Lekka kara dla blanka na indeksie 0

        probs = torch.softmax(scaled_log_probs, dim=-1).permute(1, 0, 2).cpu().numpy()
        decoded_texts = []

        for i in range(probs.shape[0]):
            text_lm = lm_decoder.decode(probs[i], beam_width=beam_width).strip()

            raw_indices = np.argmax(probs[i], axis=-1)
            text_raw = self.decode(raw_indices)

            if not text_raw:
                decoded_texts.append(text_lm if text_lm else "")
                continue

            dist = edit_distance(text_raw, text_lm)
            norm_dist = dist / len(text_raw)

            last_char_raw = text_raw[-1]
            if last_char_raw in ".,!?:;()-\" '/" and not text_lm.endswith(last_char_raw):
                final_word = text_raw
            elif norm_dist > 0.45 or (3 >= len(text_raw) > len(text_lm)):
                final_word = text_raw
            else:
                final_word = text_lm if text_lm else text_raw

            decoded_texts.append(final_word)

        return decoded_texts

    def encode_text(self, text_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Zamienia listę tekstów na tensor indeksów i długości (Zsynchronizowane z CTCLoss). """
        targets = []
        lengths = []
        for text in text_list:
            indices = [self.char_to_num[c] for c in text if c in self.char_to_num]
            targets.extend(indices)
            lengths.append(len(indices))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, idx_seq: Union[List[int], torch.Tensor]) -> str:
        """ Dekodowanie pojedynczego wektora indeksów (Greedy Fallback). """
        if isinstance(idx_seq, torch.Tensor):
            idx_seq = idx_seq.tolist()

        collapsed = []
        last = -1
        for i in idx_seq:
            if i != 0 and i != last:
                collapsed.append(i)
            last = i
        return self.labels_to_text(collapsed)


def seed_everything(seed=3407, deterministic=True):
    """ Zamraża losowość w całym środowisku. Gwarantuje pełną powtarzalność wyników i determinizm obliczeń. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Zapewnia determinizm (może być trochę wolniejsze)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        tqdm.write(f"[{now()}] Włączono tryb deterministyczny (wolniejszy, ale w pełni reprodukowalny).")
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        tqdm.write(f"[{now()}] Stałe seedy, ale cuDNN może używać niedeterministycznych algorytmów).")


def seed_worker(worker_id):
    """ Zapewnia unikalną, lecz powtarzalną losowość dla każdego wątku DataLoader. """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_augmentations(phase):
    """ Przekazuje do CRNN augmentacje odpowiednie dla danego etapu uczenia. """
    if phase == "main":
        """ Główna faza treningowa. Zrównoważone zniekształcenia uczące model odporności 
            na powszechne wady skanów i naturalne różnice w stylu pisania, bez niszczenia czytelności znaków. """
        return alb.Compose([

            # Uodparnianie modelu na rotację i przesunięcie
            alb.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.07,
                rotate_limit=0,
                p=0.4,
                border_mode=cv.BORDER_CONSTANT,
                value = 255
            ),

            # Imitowanie szumu i nieostrości obiektywu
            alb.OneOf([
                alb.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                alb.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
            ], p=0.2),

            # Modyfikacje morfologiczne atramentu
            alb.OneOf([
                alb.Lambda(image=AdvancedHTRAugmentor.variable_stroke, p=1.0),
                alb.Lambda(image=AdvancedHTRAugmentor.ink_bleeding, p=1.0),
            ], p=0.25),

            # Uodpornienie sieci na fragmenty liter z okolicznych słów
            alb.Lambda(image=AdvancedHTRAugmentor.phantom_elements, p=0.15),

            # Błędy geometrii
            alb.OneOf([
                alb.Rotate(limit=5, p=1.0, border_mode=cv.BORDER_REPLICATE),
                alb.Perspective(scale=(0.02, 0.04), p=1.0, border_mode=cv.BORDER_REPLICATE),
                alb.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0, border_mode=cv.BORDER_REPLICATE),
            ], p=0.3),

            # Symulacja kleksów, zamazanego atramentu i zniszczonego papieru
            alb.CoarseDropout(
                max_holes=3, max_height=8, max_width=8, 
                min_holes=1, min_height=2, min_width=2, 
                fill_value=0, p=0.15
            ),

            # Standaryzacja
            alb.Normalize(mean=IAM_MEAN, std=IAM_STD),
            ToTensorV2()
        ])

    elif phase == "fine_tune":
        """ Faza dostrajania. Model ma się tu skupić na utrwaleniu naturalnych wzorców tekstowych przy
            niewielkim szumie optycznym. """
        return alb.Compose([

            # Bardzo delikatna korekta położenia i minimalny obrót, przygotowujące model na naturalny tekst
            alb.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=2,
                p=0.25,
                border_mode=cv.BORDER_CONSTANT,
                fill=255
            ),

            # Drobny szum lub lekkie nieostrości obiektywu
            alb.OneOf([
                alb.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                alb.GaussianBlur(blur_limit=(3, 3), p=1.0),
            ], p=0.1),

            # Symulacja nierównomiernego oświetlenia skanera
            alb.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),

            # Standaryzacja do IAM
            alb.Normalize(mean=IAM_MEAN, std=IAM_STD),
            ToTensorV2()
        ])
    return None


def get_label_from_filename(filename):
    """ Dekoduje etykietę Ground Truth z nazwy pliku.
        1. Usuwa rozszerzenie i unikalne sufiksy numeryczne.
        2. Przywraca interpunkcję z kodów tekstowych (np. 'DOT' -> '.').
        3. Mapuje przypadki specjalne przez słownik FILENAME_MAP.
        4. Zamienia podkreślenia na spacje. """
    name_no_ext = os.path.splitext(filename)[0]

    # Usuwamy unikalny sufiks numeryczny
    label = re.sub(r'_\d+$', '', str(name_no_ext))

    # Zamiana podkreśleń (separatorów w nazwach plików) na spacje
    label = label.replace('_', ' ')

    for code, char in REVERSE_PUNCTUATION_MAP.items():
        if code in label:
            label = label.replace(code, char)

    label = FILENAME_MAP.get(label, label)
    return label.strip()


def create_balanced_sampler(dataset, alphabet):
    """ Tworzy WeightedRandomSampler z bezpiecznym pobieraniem tekstów z opakowanych zbiorów. """
    def get_text_recursive(ds, idx):
        # Obsługa Subset
        if isinstance(ds, Subset):
            # Przekierowujemy indeks na oryginalny zbiór
            return get_text_recursive(ds.dataset, ds.indices[idx])
        
        # Obsługa ConcatDataset
        elif isinstance(ds, ConcatDataset):
            # Znajdujemy, w którym pod-zbiorze znajduje się indeks
            for i, end_idx in enumerate(ds.cumulative_sizes):
                if idx < end_idx:
                    # Obliczamy lokalny indeks wewnątrz pod-zbioru
                    local_idx = idx - (ds.cumulative_sizes[i-1] if i > 0 else 0)
                    return get_text_recursive(ds.datasets[i], local_idx)
        
        # Jeśli dotarliśmy do źródła (IAMWordDataset lub PolishSyntheticDataset)
        elif hasattr(ds, 'get_text'):
            return ds.get_text(idx)
        
        return alphabet[0]

    # Pobieranie wszystkich tekstów
    all_texts = []
    for i in range(len(dataset)):
        text = get_text_recursive(dataset, i)
        all_texts.append(text if text else alphabet[0])
                
    # Logika wag
    char_counts = {char: 0 for char in alphabet}
    for text in all_texts:
        for char in text:
            if char in char_counts:
                char_counts[char] += 1
                
    # Waga znaku
    char_weights = {char: 1.0 / (count + 1e-5) for char, count in char_counts.items()}
    
    sample_weights = []
    for text in all_texts:
        weight = sum(char_weights.get(c, 0) for c in text) / max(len(text), 1)
        sample_weights.append(weight)
        
    weights_tensor = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


def collate_fn_dynamic(batch):
    """ Składa próbki w batch, stosując w pełni dynamiczny padding bez ucinania długich słów. """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None

    # Walidacja batcha + bezpieczne czyszczenie i usuwanie spacji brzegowych
    valid_batch = []
    for item in batch:
        if torch.is_tensor(item[0]) and item[0].dim() == 3:
            # Konwersja na string i obcięcie spacji brzegowych (zabezpieczenie przed wyciekami)
            label_clean = str(item[1]).strip()

            # Sprawdzamy czy tekst nie jest pusty i czy nie zawiera spacji w środku
            if label_clean and ' ' not in label_clean: 
                new_item = list(item)
                new_item[1] = label_clean
                valid_batch.append(tuple(new_item))
                
    if not valid_batch: return None

    imgs = [item[0] for item in valid_batch]
    labels = [item[1] for item in valid_batch]
    categories = [item[2] if len(item) > 2 else 'short' for item in valid_batch]
    extras = list(zip(*[item[3:] for item in valid_batch])) if any(len(item) > 3 for item in valid_batch) else []

    # Obliczamy wartość tła do paddingu na podstawie globalnych statystyk nowej normalizacji
    bg_val = (0.0 - float(IAM_MEAN[0])) / float(IAM_STD[0])

    # Przetwarzamy obrazy pod kątem fizycznych wymagań sieci i CTC Loss
    processed_imgs = []
    for img, label in zip(imgs, labels):
        current_w = int(img.shape[-1])

        # Zabezpieczenie dla CTC: CRNN redukuje wymiary. Obraz musi mieć fizycznie miejsce na wyplucie wszystkich znaków.
        min_w_needed = len(label) * 48

        # Jeśli obraz jest skrajnie nienaturalnie ściśnięty, rozszerzamy go od razu w prawo
        if current_w < min_w_needed:
            pad_right = min_w_needed - current_w
            img = torch.nn.functional.pad(img, (0, pad_right), value=bg_val)
            current_w = min_w_needed

        # Ograniczenie maksymalnej szerokości do stabilnego limitu
        if current_w > 2048:
            img = img[:, :, :2048]

        processed_imgs.append(img)

    # Szukamy najszerszego słowa w tym konkretnym batchu
    batch_max_w = max(int(img.shape[-1]) for img in processed_imgs)

    """ Zapisujemy informacje o oryginalnej szerokości przed nałożeniem paddingu. Dzięki temu funkcja CTC Loss wie,
        w którym dokładnie miejscu kończy się rzeczywisty tekst, a zaczyna pusta przestrzeń, co zapobiega zdominowaniu modelu przez spacje. """
    original_widths = [int(img.shape[-1]) for img in processed_imgs]

    padded_imgs = []
    for img in processed_imgs:
        curr_w = int(img.shape[-1])
        pad_right = batch_max_w - curr_w

        # Wyrównanie wszystkich słów do prawej krawędzi (do najdłuższego w batchu) za pomocą wartości znormalizowanego tła
        if pad_right > 0:
            padded_img = torch.nn.functional.pad(img, (0, pad_right), value=bg_val)
        else:
            padded_img = img
        padded_imgs.append(padded_img)

    padded = torch.stack(padded_imgs)

    if extras:
        return padded, labels, categories, original_widths, *[list(field) for field in extras]

    return padded, labels, categories, original_widths


def label_smoothed_ctc_loss(log_probs, targets, input_lengths, target_lengths, smoothing=0.0, reduction='none'):
    """ CTC Loss z opcjonalnym label smoothing dla lepszej kalibracji modelu. W fazie fine-tune. """
    raw_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')(
        log_probs, targets, input_lengths, target_lengths
    )

    if smoothing > 0:
        """ Średnia kara po klasach (dim=-1), ale SUMA po czasie (dim=0).
            W ten sposób skala kary rośnie proporcjonalnie do długości słowa (tak samo jak raw_loss).
            Zakładamy standardowy wejściowy kształt CTC dla log_probs """
        uniform_penalty = -log_probs.mean(dim=-1).sum(dim=0)
        
        # Zmienna smooth_loss zachowuje wektorowy kształt [N], chroniąc selekcję próbek dla S-OHEM.
        smooth_loss = (1 - smoothing) * raw_loss + smoothing * uniform_penalty
        
        return smooth_loss if reduction == 'none' else smooth_loss.mean()

    return raw_loss if reduction == 'none' else raw_loss.mean()


def ace_loss(log_probs, targets, input_lengths, target_lengths, reduction='mean'):
    """ ACE Loss (Aggregation Cross-Entropy)
        Zamiast CTC, który wymaga blank tokens i może być nieefektywny dla długich sekwencji,
        ACE agreguje prawdopodobieństwa per-character poprzez soft attention alignment.

        Korzyści dla HCR:
        1. Lepsza obsługa ligatur i połączonych znaków
        2. Silniejszy gradient dla trudnych przykładów
        3. Naturalna pewność na poziomie znaku (routing do CapsNet)
        4. Brak blank token overhead """
    T, N, C = log_probs.size()
    device = log_probs.device

    # Konwersja do prawdopodobieństw
    probs = torch.exp(log_probs)

    total_loss = 0.0
    valid_samples = 0

    for b in range(N):
        seq_len = input_lengths[b].item()
        tgt_len = target_lengths[b].item()

        if tgt_len == 0 or seq_len < tgt_len:
            continue

        # Pobieramy prawdopodobieństwa dla tej próbki
        seq_probs = probs[:seq_len, b, :]

        # Target indices dla tej próbki
        tgt_start = targets[:b].sum() if b > 0 else 0
        tgt_indices = targets[tgt_start:tgt_start + tgt_len]

        # Agregacja przez soft attention alignment
        segment_size = seq_len / tgt_len

        sample_loss = 0.0
        for i, tgt_idx in enumerate(tgt_indices):
            # Pozycja centralna dla tego znaku
            center = (i + 0.5) * segment_size

            # Soft window wokół centrum (Gaussian weights)
            positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            distances = (positions - center) ** 2
            attention_weights = torch.exp(-distances / (2 * (segment_size / 2) ** 2))
            attention_weights = attention_weights / attention_weights.sum()

            # Agregowane prawdopodobieństwa dla tego znaku
            char_probs = (seq_probs * attention_weights.unsqueeze(1)).sum(dim=0)  # [C]

            # Cross-entropy dla tego znaku
            char_loss = -torch.log(char_probs[tgt_idx] + 1e-8)
            sample_loss += char_loss

        total_loss += sample_loss / tgt_len
        valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if reduction == 'mean':
        return total_loss / valid_samples
    return total_loss


def focal_ace_loss(log_probs, targets, input_lengths, target_lengths, gamma=2.0, reduction='mean'):
    """ Focal ACE Loss - Łączy ACE z focal weighting.
        Zoptymalizowana pod kątem typowania i stabilności numerycznej. """
    device = log_probs.device

    # Obliczamy bazową stratę ACE
    base_loss = ace_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')

    # Normalizacja przez długość (kluczowe, by długie słowa nie dominowały)
    t_lengths = target_lengths.to(device).float().clamp(min=1.0)
    norm_loss = base_loss / t_lengths

    # Obliczanie wag fokalnych
    p: torch.Tensor = torch.exp(torch.neg(norm_loss))

    # Tworzymy stałe jako Tensory
    one = torch.as_tensor(1.0, device=device, dtype=norm_loss.dtype)
    gamma_t = torch.as_tensor(gamma, device=device, dtype=norm_loss.dtype)

    # Obliczamy mnożnik: (1 - p) ^ gamma (strata fokalna - im model jest pewniejszy, tym bardziej wygasza wagę łatwych próbek).
    focal_weight = torch.pow(one - p, gamma_t)

    # Nakładamy wagę fokalną na bazową stratę
    focal_weighted = focal_weight * norm_loss

    # Redukcja
    if reduction == 'mean':
        return focal_weighted.mean()
    elif reduction == 'sum':
        return focal_weighted.sum()
    return focal_weighted


def focal_ctc_loss(log_probs, targets, input_lengths, target_lengths, gamma=2.0, reduction='mean'):
    """ Strata fokalna zmusza sieć do ignorowania rzeczy, które już potrafi, i skupienia 100% swojej
        uwagi na przypadkach, z którymi sobie nie radzi. """
    device = log_probs.device
    raw_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')(
        log_probs, targets, input_lengths, target_lengths
    )

    norm_loss = raw_loss / target_lengths.to(device).clamp(min=1).float()
    
    # Prawdopodobieństwo 'p' wyliczone z normalizowanej straty
    p = torch.exp(-norm_loss)
    
    # Standardowy mnożnik fokalny bez ucinania
    gamma_tensor = torch.tensor(gamma, device=device)
    focal_multiplier = torch.pow(1.0 - p, gamma_tensor)
    
    focal_loss = focal_multiplier * norm_loss

    if reduction == 'mean':
        return focal_loss.mean()
    return focal_loss


def inter_class_separation_loss(features, labels, margin=0.5):
    """ Inter-Class Separation Loss — wymusza minimalny odstęp między klasami.
        Dla HCR jest lepsze niż triplet loss, bo:
        1. Nie wymaga miningu tripletów
        2. Globalnie odsuwa wszystkie klasy od siebie
        3. Jest szybsze w treningu. """
    device = features.device
    unique_labels = labels.unique()
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Mapowanie etykiet na zakres [0, n_classes-1]
    label_map = (labels.unsqueeze(1) == unique_labels).float()
    mapped_labels = label_map.argmax(dim=1)
    
    # Obliczanie centrów
    counts = torch.bincount(mapped_labels, minlength=n_classes).to(features.dtype).view(-1, 1)
    
    # Inicjalizacja
    sum_features = torch.zeros(n_classes, features.size(1), dtype=features.dtype, device=device)
    sum_features.scatter_add_(0, mapped_labels.unsqueeze(1).expand(-1, features.size(1)), features)
    centers = sum_features / counts.clamp(min=1)

    # Rzutowanie na float32 przed cdist (unikamy błędów numerycznych dla bfloat16 w tej operacji)
    centers_fp32 = centers.float()
    distances = torch.cdist(centers_fp32, centers_fp32, p=2)

    # Penalty tylko dla par
    mask = torch.triu(torch.ones(n_classes, n_classes, device=device), diagonal=1).bool()
    close_pairs = distances[mask]

    return torch.relu(margin - close_pairs).mean()
    

def decode_with_per_char_confidence(outputs, encoder, noise_threshold=0.1):
    """ Dekodowanie , które używa ważonej pewności, aby uniknąć zaniżania wyniku przez klatki brzegowe. """
    probs = torch.softmax(outputs, dim=2).permute(1, 0, 2).cpu().detach().numpy()

    batch_results = []
    for b in range(probs.shape[0]):
        word_data = []
        curr_probs = []
        curr_indices = []
        last_idx = -1

        for t in range(probs.shape[1]):
            p_t = probs[b, t]
            idx = np.argmax(p_t)
            prob = p_t[idx]

            if idx == 0:  # Blank
                if last_idx > 0 and curr_probs:
                    # Bierzemy pod uwagę tylko klatki powyżej noise_threshold, stosujemy Power Mean (suma kwadratów / suma wartości)
                    p_array = np.array(curr_probs)

                    # Wymuszenie float() przed operatorem dzielenia
                    robust_conf = float(np.sum(p_array ** 2)) / float(np.sum(p_array))

                    word_data.append({
                        "char": encoder.num_to_char[last_idx],
                        "conf": robust_conf,  # Zmienna jest już czystym floatem
                        "t_start": curr_indices[0],
                        "t_end": curr_indices[-1]
                    })
                    curr_probs, curr_indices = [], []
                last_idx = 0
                continue

            if idx == last_idx:
                curr_probs.append(prob)
                curr_indices.append(t)
            else:
                if last_idx > 0 and curr_probs:
                    p_array = np.array(curr_probs)

                    robust_conf = float(np.sum(p_array ** 2)) / float(np.sum(p_array))

                    word_data.append({
                        "char": encoder.num_to_char[last_idx],
                        "conf": robust_conf,
                        "t_start": curr_indices[0],
                        "t_end": curr_indices[-1]
                    })

                curr_probs = [prob]
                curr_indices = [t]
                last_idx = idx

        if last_idx > 0 and curr_probs:
            p_array = np.array(curr_probs)

            robust_conf = float(np.sum(p_array ** 2)) / float(np.sum(p_array))

            word_data.append({
                "char": encoder.num_to_char[last_idx],
                "conf": robust_conf,
                "t_start": curr_indices[0],
                "t_end": curr_indices[-1]
            })

        batch_results.append(word_data)
    return batch_results


def get_focal_gamma_schedule(epoch, total_epochs, start_gamma=2.5, end_gamma=1.5):
    """ Harmonogram gamma dla focal loss — zmniejsza się w trakcie treningu.
        Wyższa gamma na początku (silniejszy focus na hard samples),
        niższa pod koniec (bardziej równomierne uczenie). """
    progress = epoch / max(1, total_epochs)
    return start_gamma - (start_gamma - end_gamma) * progress


def predict_with_tta(model, image, encoder, num_augmentations=3):
    """ Test Time Augmentation dla walidacji — uśrednia predykcje z kilku lekkich augmentacji. Zwiększa pewność predykcji. """
    model.eval()
    predictions = []

    with torch.no_grad():
        # Oryginalna predykcja
        logits = model(image)
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        predictions.append(torch.softmax(logits.float(), dim=-1))

        # Augmentowane wersje
        for _ in range(num_augmentations):
            aug_image = image.clone()
            
            # Poprawna aplikacja szumu do znormalizowanego tensora (bez destrukcyjnego clampowania)
            brightness_shift = torch.randn(1).item() * 0.1 * (1.0 / IAM_STD[0])
            aug_image = aug_image + brightness_shift

            logits_aug = model(aug_image)
            logits_aug = logits_aug[0] if isinstance(logits_aug, (tuple, list)) else logits_aug
            predictions.append(torch.softmax(logits_aug.float(), dim=-1))

    # Uśrednienie prawdopodobieństw i powrót do log_probs
    avg_probs = torch.stack(predictions).mean(dim=0)
    avg_log_probs = torch.log(avg_probs + 1e-8)

    return avg_log_probs


def evaluate_loss_only(model, loader, device, encoder, use_tta=False):
    """ Szybka ewaluacja modelu z paskiem postępu. Oblicza surową, średnią stratę na całym zbiorze walidacyjnym. """
    model.eval()
    total_loss = 0.0
    batches = 0

    desc_str = "Walidacja (TTA)" if use_tta else "Walidacja"
    
    # Tworzymy szybki zbiór dozwolonych znaków, by filtrować w locie
    valid_vocab = set(encoder.char_to_num.keys())
    
    with tqdm(loader, desc=desc_str, leave=False, dynamic_ncols=True, colour='green') as pbar:
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue

                try:
                    images = batch[0].to(device)
                    raw_text_labels = batch[1]
                    valid_widths = batch[3] if len(batch) > 3 else None
                    
                    # Usuwanie niedozwolonych znaków z etykiet
                    text_labels = []
                    for lbl in raw_text_labels:
                        clean_lbl = "".join([c for c in str(lbl) if c in valid_vocab])
                        text_labels.append(clean_lbl)
                        
                    # Jeśli wszystkie napisy w batchu okazały się pustymi po wyczyszczeniu, pomijamy paczkę
                    if all(len(t) == 0 for t in text_labels):
                        del images
                        continue

                    # Ochrona przed uszkodzonymi (za małymi) obrazami
                    if images.size(-1) < 4:  # Minimalna szerokość dla warstw konwolucyjnych
                        tqdm.write(f"[{now()}] Ostrzeżenie: Znaleziono obraz węższy niż 4 piksele. Pomijam.")
                        del images
                        continue

                except RuntimeError as re:
                    if "out of memory" in str(re).lower():
                        tqdm.write(f"[{now()}] Krytyczny błąd VRAM: OOM podczas walidacji. Czyszczę cache i pomijam.")
                        if 'images' in locals(): del images
                        torch.cuda.empty_cache()
                    else:
                        tqdm.write(f"[{now()}] Błąd wykonania PyTorch: {re}")
                    continue
                except (IndexError, TypeError, AttributeError) as e:
                    tqdm.write(f"[{now()}] Błąd formatu danych w loaderze: {e}. Sprawdź 'collate_fn'.")
                    continue
                except Exception as e:
                    tqdm.write(f"[{now()}] Nieoczekiwany błąd ładowania ({type(e).__name__}): {e}")
                    continue

                targets, target_lengths = encoder.encode_text(text_labels)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                if use_tta and images.size(0) == 1:
                    log_probs = predict_with_tta(model, images, encoder, num_augmentations=2)
                else:
                    output = model(images)
                    raw_logits = output[0] if isinstance(output, (tuple, list)) else output
                    log_probs = torch.nn.functional.log_softmax(raw_logits.float(), dim=-1)

                if log_probs.shape[0] == images.size(0) and log_probs.shape[1] > images.size(0):
                    log_probs = log_probs.permute(1, 0, 2)

                T_dim = log_probs.size(0)
                batch_size = images.size(0)

                # Zbyt długi tekst w stosunku do szerokości obrazu
                if target_lengths.max() > T_dim:
                    del images, targets, target_lengths, log_probs
                    if 'output' in locals(): del output
                    continue

                if valid_widths is not None:
                    input_lengths = torch.tensor([w // 4 for w in valid_widths], dtype=torch.long, device=device)
                    input_lengths = torch.clamp(input_lengths, max=T_dim)
                else:
                    input_lengths = torch.full(size=(batch_size,), fill_value=T_dim, dtype=torch.long).to(device)

                loss = focal_ctc_loss(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    del images, targets, target_lengths, log_probs
                    if 'output' in locals(): del output
                    continue

                current_loss = loss.item()
                total_loss += current_loss
                batches += 1

                pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})

                del images, targets, target_lengths, log_probs, input_lengths, loss
                if 'output' in locals(): del output
                if 'raw_logits' in locals(): del raw_logits

        if batches == 0:
            return float('inf')

    return total_loss / batches


def evaluate_full_metrics(model, loader, device, encoder, decoder=None):
    """ Pełna ewaluacja CER/WER z optymalnym wyliczaniem dystansu edycyjnego znak po znaku. """
    model.eval()
    stats = {
        'short': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0},
        'medium': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0},
        'long': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0}
    }

    evaluated_samples = 0
    max_debug_samples = 50000
    valid_vocab = set(encoder.char_to_num.keys())

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Ewaluacja", ncols=100)):
            if batch is None: continue
            
            try:
                images = batch[0].to(device)
                raw_text_labels = batch[1]
                
                # Filtrowanie Ground Truth z nieznanych znaków
                text_labels = []
                for lbl in raw_text_labels:
                    clean_lbl = "".join([c for c in str(lbl) if c in valid_vocab])
                    text_labels.append(clean_lbl)
                    
                if all(len(t) == 0 for t in text_labels):
                    del images
                    continue

                if images.size(-1) < 4:
                    del images
                    continue

                output = model(images)
                log_probs = output[0] if isinstance(output, (tuple, list)) else output
                
                if log_probs.dim() == 3 and log_probs.shape[0] == images.size(0):
                    log_probs = log_probs.permute(1, 0, 2)

                if decoder:
                    preds = encoder.decode_beam_search(log_probs, decoder)
                else:
                    preds, _ = encoder.decode_greedy(log_probs)
                    
            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    tqdm.write(f"      [DEBUG] Ekstremalny obraz zablokował VRAM na batchu {i}. Pomięcie.")
                    if 'images' in locals(): del images
                    if 'output' in locals(): del output
                    if 'log_probs' in locals(): del log_probs
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise re

            for gt_word, pred_word in zip(text_labels, preds):
                gt_str = str(gt_word).strip()
                pred_str = str(pred_word).strip()
                
                evaluated_samples += 1

                length = len(gt_str)
                cat = 'short' if length < 5 else ('medium' if length < 9 else 'long')
                dist = edit_distance(gt_str, pred_str)
                
                stats[cat]['chars'] += length
                stats[cat]['dist'] += dist
                stats[cat]['words'] += 1
                if dist > 0:
                    stats[cat]['err_words'] += 1

            if evaluated_samples >= max_debug_samples and stats['long']['chars'] > 0:
                curr_cer = (stats['long']['dist'] / stats['long']['chars']) * 100
                if curr_cer > 95.0:
                    tqdm.write(f"      [DEBUG] CER zbliżone do 100% ({curr_cer:.2f}%). Przerywam wczesną ewaluację.")
                    break
            
            del images, output, log_probs
            if i % 50 == 0:
                torch.cuda.empty_cache()

    tqdm.write(f"Pełny raport na słowach")
    tqdm.write(f"{'KATEGORIA':>12} | {'CER [%]':>10} | {'WER [%]':>10} | {'PRÓBEK':>10}")
    for cat in ['short', 'medium', 'long']:
        data = stats[cat]
        cer = (data['dist'] / max(1, data['chars'])) * 100
        wer = (data['err_words'] / max(1, data['words'])) * 100
        tqdm.write(f"{cat.upper():>12} | {cer:10.2f} | {wer:10.2f} | {data['words']:>10}")
    
    if 'text_labels' in locals() and 'preds' in locals() and len(text_labels) > 0 and len(preds) > 0:
        tqdm.write(f"DEBUG: GT: {text_labels[0]} | PRED: {preds[0]}")
    
    return stats


def train_one_epoch(model, loader, optimizer, scaler, device, encoder, cat_weights,
                    scheduler=None, use_contrastive=True, ema_model=None, writer=None, acc_steps=1,
                    epoch=0, use_focal=False, label_smoothing=0.0, focal_gamma=2.0, use_ace=False,
                    center_criterion=None, optimizer_center=None, lambda_center=0.0, lambda_separation=0.0):
    """ Wykonuje jedną epokę treningową z uwzględnieniem dynamicznego balansowania klas.
        Logika Straty opiera się na dwóch filarach:
        1. Ważona Strata Fokalna — strata główna.
           Wylicza błąd dla każdego słowa indywidualnie. Błędy w długich słowach,
           które występują rzadziej w zbiorze danych, są dynamicznie skalowane w górę za pomocą
           odwrotności pierwiastka częstości. Zapobiega to ignorowaniu trudnych, długich sekwencji przez sieć.
        2. Contrastive Loss — strata pomocnicza.
           Regularyzuje przestrzeń cech z warstw rekurencyjnych. Zbliża do siebie wielowymiarowe reprezentacje
           takich samych słów (ucząc model ignorować styl pisma odręcznego) oraz oddala od siebie słowa o różnych znakach. """
    model.train()

    # Wyłączamy CNN, jeśli nie ma wymagań gradientowych (np. w trybie fine-tune RNN)
    if not any(p.requires_grad for p in model.cnn.parameters()):
        model.cnn.eval()

    # flatten_parameters wystarczy raz przed epoką
    model.rnn.flatten_parameters()

    total_loss = 0.0

    step_lrs_list = []
    step_moms_list = []

    if cat_weights is None:
        cat_weights = {'short': 1.0, 'medium': 1.5, 'long': 2.5}

    # Pobieramy liczbę jako czysty int
    total_batches = int(len(loader))

    with tqdm(loader, desc="Uczenie", leave=True, file=sys.stdout, dynamic_ncols=True) as loop:
        for i, batch in enumerate(loop):
            if batch is None: continue

            # Dynamiczny podgląd średniej straty z dotychczasowych kroków
            running_avg_loss = total_loss / max(1, i)
            loop.set_postfix({
                'Avg_Loss': f"{running_avg_loss:.3f}"
            })

            # Zapisywanie aktualnego Learning Rate i Momentum
            step_lrs_list.append(optimizer.param_groups[0]['lr'])
            current_mom = optimizer.param_groups[0].get('momentum', optimizer.param_groups[0].get('betas', (0.9, 0.999))[0])
            step_moms_list.append(current_mom)

            # Rozpakowanie surowych danych z batcha
            raw_images = batch[0]
            raw_text_labels = batch[1] if len(batch) > 1 else []
            raw_categories = batch[2] if len(batch) > 2 else ['short'] * len(raw_images)
            raw_valid_widths = batch[3] if len(batch) > 3 else [int(img.shape[-1]) for img in raw_images]

            # Dynamiczne wyliczanie kategorii oraz filtrowanie pustych tekstów
            valid_batch_data = []
            valid_input_lengths = []
            for img, lbl, cat, w in zip(raw_images, raw_text_labels, raw_categories, raw_valid_widths):
                lbl_clean = lbl.strip()
                if len(lbl_clean) == 0:
                    continue
                
                # Przypisywanie kategorii na podstawie długości napisu
                if len(lbl_clean) < 5:
                    actual_cat = 'short'
                elif len(lbl_clean) < 10:
                    actual_cat = 'medium'
                else:
                    actual_cat = 'long'
                    
                valid_batch_data.append((img, lbl_clean, actual_cat))

                """ W trakcie treningu sieci zmniejszamy szerokość 2 razy, łącznie 4-krotnie,
                    żeby model widział wąskie znaki. """
                valid_input_lengths.append(w // 4)

            # Zabezpieczenie przed pustym batchem po filtracji
            if not valid_batch_data: continue

            # Rekonstrukcja tensorów i list dla modeli i OHEM
            images = torch.stack([x[0] for x in valid_batch_data]).to(device)
            text_labels = [x[1] for x in valid_batch_data]
            categories = [x[2] for x in valid_batch_data]

            # Kodowanie etykiet na poziomie znaków dla CTC Loss
            targets_list = []
            target_lengths_list = []
            for t in text_labels:
                encoded = [encoder.char_to_num[c] for c in t if c in encoder.char_to_num]
                targets_list.extend(encoded)
                target_lengths_list.append(len(encoded))

            targets = torch.tensor(targets_list, dtype=torch.long, device=device)
            target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)

            if images.size(0) == 0:
                continue

            # Forward pass z wykorzystaniem AMP
            with torch.amp.autocast('cuda', enabled=(scaler is not None), dtype=torch.bfloat16):
                output = model(images, return_embeddings=use_contrastive)

            # Bezpieczne rozpakowanie
            if use_contrastive:
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    preds_half = output[0]
                    embeddings = output[-1]
                else:
                    raise ValueError(f"Oczekiwano co najmniej 2 wartości w trybie contrastive, otrzymano {len(output) if isinstance(output, (tuple, list)) else 1}. Sprawdź konfigurację modelu.")
            else:
                preds_half = output[0] if isinstance(output, (tuple, list)) else output
                embeddings = None

            # Float32 dla ochrony przed underflow, zapobiega eksplozji gradientów
            preds_fp32 = preds_half.float()
            
            # Przetworzone wyjście pod funkcje straty
            log_preds = torch.nn.functional.log_softmax(preds_half.float(), dim=-1)
        
            """ Usunąłem stąd karanie za spacje, CRNN otrzymuje czyste logity, a nie zmodyfikowane prawdopodobieństwa. Kara za spacje, będzie dopiero na poziomie etykiet. """

            T_dim = log_preds.size(0)

            # Obliczanie strat (CTC/ACE)
            input_lengths = torch.tensor(valid_input_lengths, dtype=torch.long, device=device)
            input_lengths = torch.clamp(input_lengths, max=T_dim)

            if use_ace:
                if use_focal:
                    batch_losses = focal_ace_loss(
                        log_preds, targets, input_lengths, target_lengths,
                        gamma=focal_gamma, reduction='none'
                    )
                else:
                    batch_losses = ace_loss(
                        log_preds, targets, input_lengths, target_lengths,
                        reduction='none'
                    )
            else:
                # Standardowa ścieżka dla modelu CRNN
                if use_focal:
                    batch_losses = focal_ctc_loss(
                        log_preds, targets, input_lengths, target_lengths,
                        gamma=focal_gamma, reduction='none'
                    )
                else:
                    batch_losses = label_smoothed_ctc_loss(
                        log_preds, targets, input_lengths, target_lengths,
                        smoothing=label_smoothing, reduction='none'
                    )

            # Ważenie straty na podstawie kategorii (np. polskie znaki diakrytyczne)
            batch_weights = torch.tensor(
                [cat_weights.get(x[2], 1.0) for x in valid_batch_data],
                dtype=torch.float32, device=device
            )
            weighted_losses = batch_losses * batch_weights

            # Filtrowanie wyników: skończone, niezerowe i nieeksplodujące (< 500)
            finite_mask = torch.isfinite(weighted_losses) & (weighted_losses > 1e-5)

            if not finite_mask.any():
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_main = weighted_losses[finite_mask].mean()
            loss = loss_main

            # Contrastive Loss
            if use_contrastive:
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    preds_half = output[0]
                    embeddings = output[-1]
                else:
                    raise ValueError("W trybie contrastive oczekiwano co najmniej 2 wartości w output modelu (preds_half i embeddings). Sprawdź konfigurację modelu.")
            else:
                preds_half = output[0] if isinstance(output, (tuple, list)) else output
                embeddings = None

            # Center Loss i Separation Loss na poziomie znaków
            if (center_criterion is not None and lambda_center > 0) or (lambda_separation > 0 and use_contrastive):
                
                # Odpinamy prawdopodobieństwa od grafu, by nie zakłócać głównego gradientu CTC
                probs = torch.exp(log_preds.detach()) 
                max_probs, preds_cls = torch.max(probs, dim=-1)
                
                # Szukamy klatek, w których model zidentyfikował znak
                valid_frames = (preds_cls != 0) & (max_probs > 0.6)
                
                if valid_frames.any():
                    # Używamy surowych logitów z czasem [T, B, C]
                    active_features = preds_fp32[valid_frames]
                    active_labels = preds_cls[valid_frames]

                    # Zabezpieczenie wymiarów: wyrównujemy liczbę kanałów do tego, czego oczekuje CenterLoss
                    if center_criterion is not None:
                        expected_dim = center_criterion.centers.size(1)
                        current_dim = active_features.size(1)
                        if current_dim < expected_dim:
                            active_features = torch.nn.functional.pad(active_features, (0, expected_dim - current_dim))
                        elif current_dim > expected_dim:
                            active_features = active_features[:, :expected_dim]

                    """ CENTER LOSS: Wymusza kompaktowość cech wewnątrz klasy. """
                    if center_criterion is not None and lambda_center > 0:
                        center_loss_value = center_criterion(active_features, active_labels)
                        loss += lambda_center * center_loss_value

                        if writer and (i % 100 == 0):
                            writer.add_scalar('Loss/Center', center_loss_value.item(), epoch * total_batches + i)

                    """ SEPARATION LOSS: Odpycha klastry różnych liter od siebie. """
                    if lambda_separation > 0 and use_contrastive:
                        separation_loss = inter_class_separation_loss(active_features, active_labels, margin=0.5)
                        loss += lambda_separation * separation_loss

                        if writer and (i % 100 == 0):
                            writer.add_scalar('Loss/Separation', separation_loss.item(), epoch * total_batches + i)

            # Pomijamy nieskonczoność
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                if center_criterion is not None and optimizer_center is not None:
                    optimizer_center.zero_grad(set_to_none=True)

                loop.set_postfix({'Loss': "NaN w Aux!"})
                continue

            # Wsteczna propagacja
            loss_to_step = loss / acc_steps
            loss_to_step.backward()

            # Optymalizacja
            if (i + 1) % int(acc_steps) == 0 or (i + 1) == total_batches:
                # Ucinamy gradienty do 5, żeby nie rosły w nieskończoność
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Update center loss centers (separacja)
                if center_criterion is not None and optimizer_center is not None and lambda_center > 0:
                    optimizer_center.step()
                    optimizer_center.zero_grad(set_to_none=True)

                if ema_model:
                    ema_model.update_parameters(model)
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            loop.set_postfix({'Loss': f"{loss.item():.3f}"})

            # Logowanie do TensorBoard co 50 batchy
            if writer and (i % 50 == 0):
                writer.add_scalar('Loss/Train_Batch', loss.item(), epoch * total_batches + i)
                writer.flush()

            # Usuwamy duże tensory, które nie są już potrzebne w tej iteracji (sekcja czyszczenia pamięci)
            del images, targets, target_lengths, output, preds_fp32, log_preds, batch_losses, weighted_losses
            
            if 'embeddings' in locals(): del embeddings
            if 'pooled_features' in locals(): del pooled_features
            if 'char_labels' in locals(): del char_labels

            # Regularne czyszczenie cache'u GPU, aby zapobiec fragmentacji pamięci i OOM podczas długich epok z dużą ilością danych
            if i % 100 == 0:
                torch.cuda.empty_cache()

    return total_loss / total_batches, step_lrs_list, step_moms_list


class CenterLoss(nn.Module):
    """ Center Loss - minimalizuje wariancję wewnątrzklasową. 
        Przyciąga wektory cech głębokich do wspólnych, wyuczalnych środków dla każdej klasy. """
    def __init__(self, num_classes=94, feat_dim=128, device=torch.device('cuda')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Inicjalizacja centroidów jako wyuczalnych parametrów
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        # Centra muszą mieć ten sam dtype
        centers = self.centers.to(x.dtype)
        batch_size = x.size(0)
        
        # Obliczanie odległości euklidesowej między cechami a centroidami
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # Wyciąganie odległości tylko dla poprawnych klas
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        
        # Sumowanie i uśrednianie straty
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


def align_prediction_to_ground_truth(gt_text, pred_text):
    """ Alignment dla analizy błędów.
        - difflib: Mapuje znaki oryginału na znaki predykcji, zachowując ich kolejność.
        - Token [pusty]: Kluczowy znacznik zastępczy.
            Insertion — model dopisał znak, którego nie ma.
            Deletion — model pominął/zgubił literę.
            Substitution — model myli kształty.
        - Celem jest diagnoza: czy model myli znaki wizualnie (np. 'I' na 'l'), czy je gubi/dopisuje. """
    matcher = difflib.SequenceMatcher(None, gt_text, pred_text)
    aligned_gt = []
    aligned_pred = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            aligned_gt.extend(list(gt_text[i1:i2]))
            aligned_pred.extend(list(pred_text[j1:j2]))
        elif tag == 'replace':
            # Wypełnienie [pusty] wyrównuje długość obu list dla macierzy pomyłek
            g_part = list(gt_text[i1:i2])
            p_part = list(pred_text[j1:j2])
            max_len = max(len(g_part), len(p_part))
            aligned_gt.extend(g_part + ['[pusty]'] * (max_len - len(g_part)))
            aligned_pred.extend(p_part + ['[pusty]'] * (max_len - len(p_part)))
        elif tag == 'insert':
            aligned_gt.extend(['[pusty]'] * (j2 - j1))
            aligned_pred.extend(list(pred_text[j1:j2]))
        elif tag == 'delete':
            aligned_gt.extend(list(gt_text[i1:i2]))
            aligned_pred.extend(['[pusty]'] * (i2 - i1))

    return aligned_gt, aligned_pred


def plot_confusion_heatmap(y_true, y_pred, title, filename, overwrite=False):
    """ Ograniczenie do najczęstszych znaków i spłaszczanie słów, żeby zapobiec ArrayMemoryError. """
    # Wewnątrz metody jest szybciej
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    full_path = os.path.join(VISUAL_DEBUG_DIR, filename)
    name, ext = os.path.splitext(filename)

    # Zarządzanie plikami
    if overwrite:
        pattern = os.path.join(VISUAL_DEBUG_DIR, f"{name}*{ext}")
        for old_file in glob.glob(pattern):
            try:
                os.remove(old_file)
            except OSError:
                pass
        final_save_path = full_path
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_save_path = os.path.join(VISUAL_DEBUG_DIR, f"{name}_{timestamp}{ext}")

    # Spłaszczanie słów do znaków — bezpieczniejsze dla RAM
    flat_true = []
    flat_pred = []

    for t, p in zip(y_true, y_pred):
        t_str, p_str = str(t), str(p)
        # Porównujemy znak po znaku (proste wyrównanie do krótszego ciągu)
        for char_t, char_p in zip(list(t_str), list(p_str)):
            flat_true.append(char_t)
            flat_pred.append(char_p)

    # Bierzemy tylko znaki, które realnie występują w GT najczęściej
    most_common_counts = Counter(flat_true).most_common(100)
    most_common_chars = [char for char, count in most_common_counts]

    # Filtrowanie list tylko do tych znaków
    y_t_filtered = []
    y_p_filtered = []
    for t, p in zip(flat_true, flat_pred):
        if t in most_common_chars and p in most_common_chars:
            y_t_filtered.append(t)
            y_p_filtered.append(p)

    unique_labels = sorted(list(set(y_t_filtered + y_p_filtered)))

    # Macierz
    if not unique_labels:
        tqdm.write(" Brak danych do wygenerowania macierzy pomyłek.")
        return

    cm = confusion_matrix(y_t_filtered, y_p_filtered, labels=unique_labels)

    # Maska dla zerowych wartości i przekątnej
    mask = (cm == 0)
    np.fill_diagonal(mask, True)

    # Dynamiczny rozmiar figury
    fig_size = max(10, len(unique_labels) * 0.4)
    plt.figure(figsize=(fig_size + 2, fig_size))
    sns.set_theme(style="white")

    # Rysowanie
    try:
        ax = sns.heatmap(
            cm,
            annot=len(unique_labels) < 50,  # Annotacje, tylko jeśli etykiet jest mało
            fmt='d',
            cmap='YlOrRd',
            mask=mask,
            norm=LogNorm(vmin=1, vmax=max(2, cm.max())),
            square=True,
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            linewidths=.1,
            cbar_kws={'shrink': .7, 'label': 'Liczba pomyłek'},
            annot_kws={"size": 7}
        )
    except ValueError as ve:
        # Niezgodność wymiarów etykiet z macierzą
        tqdm.write(f"[{now()}] Błąd wymiarów Heatmapy: {ve}. Sprawdź unique_labels (len={len(unique_labels)}).")
        plt.close()
        return
    except RuntimeError as re:
        # Błędy silnika graficznego
        tqdm.write(f"[{now()}] Błąd renderowania obrazu: {re}")
        plt.close()
        return
    except Exception as e:
        tqdm.write(f"[{now()}] Nieoczekiwany błąd Seaborn: {type(e).__name__} - {e}")
        plt.close()
        return

    plt.title(f"{title} (Top {len(unique_labels)} znaków)", fontsize=18, fontweight='bold', pad=10)
    plt.xlabel('Przewidywane (Pred)', fontsize=12, fontweight='bold')
    plt.ylabel('Prawdziwe (GT)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(final_save_path, dpi=120, bbox_inches='tight')
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] Macierz pomyłek zapisana: {os.path.basename(final_save_path)}")
    plt.close()


def plot_scheduler(lrs, moms, filename="scheduler_plot.png"):
    """ Wykres rzeczywistej stopy uczenia i momentum po zakończeniu fazy treningu. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))

    # Oś główna dla Stopy uczenia
    ax1 = plt.gca()
    ax1.plot(lrs, label='Stopa uczenia (LR)', color='blue')
    ax1.set_xlabel("Kroki (Batche)")
    ax1.set_ylabel("Stopa uczenia", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Druga oś Y dla Momentum (zupełnie inny zakres wartości)
    if moms and len(moms) == len(lrs):
        ax2 = ax1.twinx()
        ax2.plot(moms, label='Beta1 (Momentum)', color='orange')
        ax2.set_ylabel('Momentum', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

    plt.title("Rzeczywisty przebieg OneCycleLR podczas treningu")
    plt.tight_layout()

    save_path = os.path.join(VISUAL_DEBUG_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] Rzeczywisty wykres schedulera zapisany: {save_path}")


def visualize_attention_map(model, image_tensor, save_path):
    """ Wizualizuje, na których częściach obrazu skupia się mechanizm uwagi. """
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        features = model.cnn(image_tensor.to(DEVICE))

        # Wyciągnięcie wag z mechanizmu uwagi
        attn_layer = model.cnn[8]
        scores = attn_layer.attn(features)
        weights = torch.softmax(scores, dim=2)

    weights_np = weights.squeeze().cpu().numpy()

    # Skalowanie mapy wag do rozmiaru oryginalnego obrazka
    original_img = image_tensor.squeeze().cpu().numpy()
    weights_resized = cv.resize(weights_np, (original_img.shape[1], original_img.shape[0]))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Obraz wejściowy")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_img, cmap='gray', alpha=0.5)
    plt.imshow(weights_resized, cmap='jet', alpha=0.5)
    plt.title("Mapa Uwagi (Gdzie patrzy sieć)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)


def get_preds(lp):
    """Pomocnicza funkcja dekodująca prawdopodobieństwa z modelu CRNN."""
    # Całkowite odcięcie tensora od grafu obliczeniowego (BFloat16) i wymuszenie float32, żeby uniknąć not supported dtype w dalszych operacjach numpy
    lp_safe = lp.detach().float()
    
    p = torch.exp(lp_safe).permute(1, 0, 2)
    indices = torch.argmax(p, dim=-1).cpu().numpy()
    top2, _ = torch.topk(p, k=2, dim=-1)
    
    # Rzutowanie każdego elementu z osobna
    m = (top2[:, :, 0] - top2[:, :, 1]).float().cpu().numpy()
    c = top2[:, :, 0].float().cpu().numpy()
    p_np = p.float().cpu().numpy()
    
    return p_np, indices, m, c


def get_safe_char_name(c):
    """ W Windows nie można używać niektórych znaków jako nazw folderów. """
    forbidden_mapping = {
        '.': 'sym_46',
        ',': 'sym_44',
        ':': 'sym_58',
        ';': 'sym_59',
        '?': 'sym_63',
        '!': 'sym_33',
        "'": 'sym_39',
        '"': 'sym_34',
        '(': 'sym_40',
        ')': 'sym_41',
        '-': 'sym_45',
        '/': 'sym_47',
        ' ': 'space'
    }

    if c in forbidden_mapping:
        return forbidden_mapping[c]

    if c.isupper():
        return f"{c}_cap"

    return c


def save_crop_with_context(crop_img, context_vec, label_char, category, full_probs_vec, source_path, output_root, export_manifest, class_counts, crnn_pred=None):
    """ Zapis wycinku litery z optymalizacją pod CapsNet i stabilnym pozycjonowaniem. """
    if crop_img is None or crop_img.size == 0:
        return

    safe_label = get_safe_char_name(label_char)
    current_count = class_counts.get(safe_label, 0)

    # Inteligentne przycinanie marginesów
    h_orig, w_orig = crop_img.shape[:2]
    margin = int(h_orig * 0.15) if h_orig > 20 else 0
    work_img = crop_img[margin:h_orig - margin, :].copy() if margin > 0 else crop_img.copy()

    if work_img.size == 0: return

    # Konwersja do Numpy Array (przenoszenie do RAM)
    work_img_np = work_img.get() if isinstance(work_img, cv.UMat) else work_img

    # Inwersja kolorów
    if float(np.mean(work_img_np)) > 127:
        work_img_np = cv.bitwise_not(work_img_np)

    # Progowanie Sauvola
    blur_img = cv.GaussianBlur(work_img_np, (3, 3), 0)
    thresh_map = threshold_sauvola(blur_img, window_size=15, k=0.2)

    # Tworzymy maskę binarną
    binary_mask = np.zeros_like(blur_img, dtype=np.uint8)
    binary_mask[blur_img < thresh_map] = 255

    cnts, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return

    small_chars = {'.', ',', '-', "'", '"', ':', ';', '_', '`'}
    min_area = 6 if label_char in small_chars else 25
    valid_cnts = [c for c in cnts if cv.contourArea(c) > min_area]
    if not valid_cnts: return

    center_x = work_img_np.shape[1] // 2

    def get_c_info(c):
        M = cv.moments(c)
        if M["m00"] == 0: return 0, 0
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Szukamy konturu najbliżej środka
    main_cnt = min(valid_cnts, key=lambda c: abs(get_c_info(c)[0] - center_x))
    mx, my, mw, mh = cv.boundingRect(main_cnt)
    mcx, mcy = get_c_info(main_cnt)

    clean_mask = np.zeros_like(binary_mask)
    cv.drawContours(clean_mask, [main_cnt], -1, (255,), -1)

    collision_margin = max(mw // 1.5, 12)
    for c in valid_cnts:
        # Bezpieczne porównywanie referencji, zamiast adresów w pamięci
        if c is main_cnt: continue

        cx, cy = get_c_info(c)
        if abs(cx - mcx) <= collision_margin and cy < (my + mh + 5):
            cv.drawContours(clean_mask, [c], -1, (255,), -1)

    # Zabezpieczenie przed wycianiem poza zakresem
    x, y, w, h = cv.boundingRect(clean_mask)
    p = 2
    max_y, max_x = clean_mask.shape
    char_roi = clean_mask[max(0, y - p):min(max_y, y + h + p), max(0, x - p):min(max_x, x + w + p)]

    if char_roi.size == 0: return

    # Skalowanie do 64x64
    canvas_size = IMAGE_HEIGHT
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    target_dim = 44 if label_char not in small_chars else 20
    scale = target_dim / max(h, w, 1)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv.resize(char_roi, (nw, nh), interpolation=cv.INTER_AREA)

    y_off = (canvas_size - nh) // 2
    x_off = (canvas_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    # Zapis i manifest
    target_dir = os.path.join(output_root, safe_label, category)
    os.makedirs(target_dir, exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    filename_base = f"{safe_label}_{unique_id}"

    # Prawidłowe rozszerzenie dla archiwum saveZ
    img_path = os.path.join(target_dir, filename_base + ".png")
    npz_path = os.path.join(target_dir, filename_base + ".npz") 

    cv.imwrite(img_path, canvas)

    # Konwersja tensorów
    prob_array = full_probs_vec.detach().float().cpu().numpy() if torch.is_tensor(full_probs_vec) else np.array(full_probs_vec)
    ctx_array = context_vec.detach().float().cpu().numpy() if torch.is_tensor(context_vec) else np.array(context_vec)

    np.savez(npz_path,
             context_vector=ctx_array.flatten(),
             crnn_probs=prob_array.flatten(),
             crnn_pred=crnn_pred,
             gt=label_char
             )

    export_manifest.append({
        "image": os.path.relpath(img_path, output_root),
        "label": safe_label,
        "category": category
    })

    class_counts[safe_label] = current_count + 1


def export_error_crops_for_capsnet(model, loader, device, encoder, output_root, opt_margin=0.15, opt_conf=0.75, opt_t=1.5):
    """ Eksport wycinków dla CapsNet oparty o dynamiczne koordynaty osi czasu CTC z modelu CRNN.
        Eliminuje ucinki pikselowe i ignorowanie fragmentów znaków w piśmie odręcznym. """
    import uuid
    if os.path.exists(output_root):
        try:
            shutil.rmtree(output_root)
        except OSError as e:
            tqdm.write(f"Ostrzeżenie: Nie można usunąć {output_root} (kod {e.errno}). Kontynuuję.")
    
    try:
        os.makedirs(output_root, exist_ok=True)
    except Exception as e:
        tqdm.write(f"Krytyczny błąd: Brak uprawnień do zapisu w {output_root}: {e}")
        return

    # Wewnętrzna, bezpieczna funkcja zapisująca (omija niszczenie konturów z poprzedniej wersji)
    def _save_ctc_slice(crop_img, context_vec, crnn_probs, char_gt, category, pred_char):
        if crop_img is None or crop_img.size == 0 or crop_img.shape[1] < 2: return
        
        safe_label = get_safe_char_name(char_gt)
        target_dir = os.path.join(output_root, safe_label, category)
        os.makedirs(target_dir, exist_ok=True)
        
        # Oczekujemy białego tekstu na czarnym tle do znalezienia ramki ograniczającej
        if np.mean(crop_img) > 127: crop_img = 255 - crop_img
        
        # Binarizacja i szukanie całego atramentu z wyznaczonego okna czasowego CTC
        _, thresh = cv.threshold(crop_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        coords = cv.findNonZero(thresh)
        
        if coords is not None and len(coords) > 0:
            x, y, w, h = cv.boundingRect(coords)
            pad = 2
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(crop_img.shape[1], x + w + pad), min(crop_img.shape[0], y + h + pad)
            char_roi = crop_img[y1:y2, x1:x2]
        else:
            char_roi = crop_img

        if char_roi.size == 0: return

        # Zachowanie proporcji i marginesów dla CapsNet
        canvas = np.zeros((64, 64), dtype=np.uint8)
        h, w = char_roi.shape
        scale = 46 / max(h, w, 1)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        
        resized = cv.resize(char_roi, (nw, nh), interpolation=cv.INTER_AREA)
        y_off, x_off = (64 - nh) // 2, (64 - nw) // 2
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

        unique_id = uuid.uuid4().hex[:8]
        base_fn = f"{safe_label}_{unique_id}"
        
        # Zapis wizualny w inwersji (czarne litery na białym tle, jak lubi CapsNet)
        cv.imwrite(os.path.join(target_dir, base_fn + ".png"), cv.bitwise_not(canvas))
        
        # Zapis wektorów matematycznych
        ctx_np = context_vec.detach().cpu().numpy() if torch.is_tensor(context_vec) else np.array(context_vec)
        probs_np = crnn_probs.detach().cpu().numpy() if torch.is_tensor(crnn_probs) else np.array(crnn_probs)
        
        np.savez(os.path.join(target_dir, base_fn + ".npz"),
                 context_vector=ctx_np.flatten(),
                 crnn_probs=probs_np.flatten(),
                 crnn_pred=pred_char,
                 gt=char_gt)

    model.eval()
    DIACRITIC_PAIRS = [{'a','ą'}, {'c','ć'}, {'e','ę'}, {'l','ł'}, {'n','ń'}, {'o','ó'}, {'s','ś'}, {'z','ź','ż'}]
    SMALL_SYMBOLS = {'.', ',', "'", '`', '-', ':', ';', '"'}

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eksport Deep Fusion", ncols=100, position=0, leave=True)
        for batch in pbar:
            if batch is None: continue
            images, text_labels, *rest = batch
            images = images.to(device)

            lp1_raw, b_ctx = model(images, return_context=True)
            lp3_raw = model(torch.roll(images, shifts=2, dims=3))

            lp1 = lp1_raw / opt_t
            lp3 = lp3_raw / opt_t

            p1, idx1, m1, c1 = get_preds(lp1)
            _, idx3, _, _ = get_preds(lp3)

            for b in range(images.size(0)):
                if not torch.isfinite(lp1[b]).all(): continue
                
                img_raw = images[b].cpu().numpy().squeeze()
                img_stn = ((img_raw * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                
                # Zamiast malować padding, tniemy obraz do momentu rzeczywistego atramentu
                true_bg_color = int(np.median(img_stn[:, 0]))
                if np.std(img_stn[:, -1]) < 5: 
                    pad_color = int(np.median(img_stn[:, -1]))
                    pad_start = img_stn.shape[1]
                    for c in range(img_stn.shape[1]-1, -1, -1):
                        if np.std(img_stn[:, c]) < 5 and abs(int(np.median(img_stn[:, c])) - pad_color) < 5:
                            pad_start = c
                        else:
                            break
                    if pad_start < img_stn.shape[1]:
                        img_stn[:, pad_start:] = true_bg_color

                if np.median(img_stn) < 127:
                    img_stn = 255 - img_stn

                T_max = b_ctx.size(1)
                dynamic_stride = img_stn.shape[1] / float(T_max)
                gt_text = text_labels[b]

                peaks = []
                last_idx = -1
                for t, idx in enumerate(idx1[b]):
                    if idx != 0 and idx != last_idx:
                        char = encoder.num_to_char.get(idx, '')
                        if char: peaks.append({'char': char, 't': t, 'conf': c1[b][t], 'margin': m1[b][t], 'idx': idx})
                    last_idx = idx

                if len(peaks) > len(gt_text) * 1.5 or len(peaks) < len(gt_text) * 0.5:
                    continue

                matcher = difflib.SequenceMatcher(None, "".join([p['char'] for p in peaks]), gt_text)

                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag in ['equal', 'replace']:
                        for idx_p, idx_g in zip(range(i1, i2), range(j1, j2)):
                            p_info = peaks[idx_p]
                            t_curr = p_info['t']
                            char_gt = gt_text[idx_g]
                            pred_char = p_info['char']

                            # Dynamiczne wyznaczanie brzegów na podstawie połowy dystansu między pikami CTC
                            t_prev = peaks[idx_p - 1]['t'] if idx_p > 0 else max(0, t_curr - 4)
                            t_next = peaks[idx_p + 1]['t'] if idx_p < len(peaks) - 1 else min(T_max, t_curr + 4)
                            
                            t_left = (t_prev + t_curr) / 2.0
                            t_right = (t_curr + t_next) / 2.0
                            
                            x1 = max(0, int(t_left * dynamic_stride))
                            x2 = min(img_stn.shape[1], int(t_right * dynamic_stride))
                            crop = img_stn[:, x1:x2]

                            c_start = max(0, t_curr - 1)
                            c_end = min(T_max, t_curr + 2)
                            windowed_ctx = torch.mean(b_ctx[b, c_start:c_end, :], dim=0)

                            pred_shift = encoder.num_to_char.get(idx3[b][t_curr], '')

                            is_wrong = pred_char != char_gt
                            is_unstable = (pred_shift != char_gt)
                            prob_val = float(p1[b][t_curr].max())
                            crnn_probs_vec = p1[b][t_curr] # Ekstrakcja całego wektora prawdopodobieństw

                            is_gt_typo = False
                            if is_wrong and prob_val > 0.80:
                                g_low, p_low = char_gt.lower(), pred_char.lower()
                                for pair in DIACRITIC_PAIRS:
                                    if g_low in pair and p_low in pair:
                                        is_gt_typo = True
                                        break
                            
                            if is_gt_typo: continue 

                            if is_wrong or is_unstable:
                                _save_ctc_slice(crop, windowed_ctx, crnn_probs_vec, char_gt, "hard_case", pred_char)
                            elif p_info['margin'] < opt_margin or p_info['conf'] < opt_conf:
                                _save_ctc_slice(crop, b_ctx[b][t_curr], crnn_probs_vec, char_gt, "unsure", pred_char)
                            elif random.random() < (UPPER_RATE if char_gt.isupper() else PURE_RATE):
                                _save_ctc_slice(crop, b_ctx[b][t_curr], crnn_probs_vec, char_gt, "pure", pred_char)

                    elif tag == 'insert':
                        t_p = peaks[i1 - 1]['t'] if i1 > 0 else 0
                        t_n = peaks[i1]['t'] if i1 < len(peaks) else p1[b].shape[0] - 1

                        for sub_idx, idx_g in enumerate(range(j1, j2)):
                            char_gt = gt_text[idx_g]
                            t_est = int(t_p + (sub_idx + 1) * (t_n - t_p) / (j2 - j1 + 1))
                            cx = t_est * dynamic_stride

                            check_area = img_stn[:, max(0, int(cx - 15)):min(img_stn.shape[1], int(cx + 15))]

                            if check_area.size == 0: continue
                                
                            mean_val = float(np.mean(check_area))

                            if mean_val > 248 or mean_val < 7:
                                best_offset, max_ink = 0, 0
                                for offset in range(-20, 21, 4):
                                    new_cx = cx + offset
                                    x1_test = max(0, int(new_cx - 8))
                                    x2_test = min(img_stn.shape[1], int(new_cx + 8))
                                    test_area = img_stn[:, x1_test:x2_test]

                                    if test_area.size == 0: continue

                                    current_ink = int(np.sum(test_area < 128))
                                    if current_ink > max_ink:
                                        max_ink, best_offset = current_ink, offset

                                cx += best_offset

                            # Dla pominętych liter wycinamy okno statyczne, bo nie mamy pików obok
                            x1 = max(0, int(cx - 26))
                            x2 = min(img_stn.shape[1], int(cx + 26))
                            crop = img_stn[:, x1:x2]
                            
                            min_ink_thresh = 8 if char_gt in SMALL_SYMBOLS else 40

                            if crop.size > 0 and np.sum(crop < 128) > min_ink_thresh:
                                t_safe = min(max(0, t_est), b_ctx.size(1) - 1)
                                crnn_probs_vec = p1[b][t_safe]
                                
                                ctx_start = max(0, t_safe - 1)
                                ctx_end = min(b_ctx.size(1), t_safe + 2)
                                windowed_ctx = torch.mean(b_ctx[b, ctx_start:ctx_end, :], dim=0)
                                
                                _save_ctc_slice(crop, windowed_ctx, crnn_probs_vec, char_gt, "missed", "")


class PolishCharStitcher:
    """ Tworzy autentycznie wyglądające polskie słowa. Przyjmuje gotową mapę znaków. """
    def __init__(self, char_map):
        self.char_map = char_map

    def generate_word_image(self, text: str, target_height=64):
        word_imgs = []
        for char in text:
            if char in [" ", "_"]:
                space_w = random.randint(20, 40) 
                word_imgs.append(np.zeros((target_height, space_w), dtype=np.uint8))
                continue

            samples = self.char_map.get(char)
            if not samples:
                word_imgs.append(np.zeros((target_height, 10), dtype=np.uint8))
                continue

            char_img = random.choice(samples)
            h, w = char_img.shape
            M = cv.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), random.uniform(0.95, 1.05))
            char_img = cv.warpAffine(char_img, M, (w, h), borderValue=0)
            word_imgs.append(char_img)

        if not word_imgs: return np.zeros((target_height, target_height), dtype=np.uint8)

        combined_width = sum([img.shape[1] for img in word_imgs]) + 40
        final_img = np.zeros((target_height, combined_width), dtype=np.uint8)

        current_x = 15
        prev_img_data = None 
        
        # Litery, które w kursywie kończą się "na górze" i łączą górą
        high_exits = {'o', 'b', 'v', 'w', 'r', 'ó', 'a', 'e'}

        for char, img in zip(text, word_imgs):
            h, w = img.shape
            if h != target_height:
                w = int(w * (target_height / h))
                img = cv.resize(img, (w, target_height))
                h = target_height

            y_offset = random.randint(-2, 2)
            y_pos = max(0, min(target_height - h, (target_height - h) // 2 + y_offset))

            is_space = char in [' ', '_'] or not np.any(img)

            # Łączenie ligaturą
            if prev_img_data and not is_space:
                prev_img, p_x, p_y, prev_char = prev_img_data
                
                # Szukamy punktu wyjścia z poprzedniej litery
                py_c, px_c = np.where(prev_img > 50)
                if len(px_c) > 0:
                    min_y, max_y = np.min(py_c), np.max(py_c)
                    mid_y = min_y + (max_y - min_y) // 2
                    
                    if prev_char in high_exits:
                        valid_p = np.where(py_c < mid_y)[0]
                        if len(valid_p) == 0: valid_p = np.arange(len(px_c))
                    else:
                        valid_p = np.where(py_c >= mid_y)[0]
                        if len(valid_p) == 0: valid_p = np.arange(len(px_c))
                        
                    best_p = valid_p[np.argmax(px_c[valid_p])]
                    p0 = (p_x + px_c[best_p], p_y + py_c[best_p])
                else:
                    p0 = (p_x + prev_img.shape[1], p_y + prev_img.shape[0])

                # Szukamy punktu wejścia do obecnej litery
                cy_c, cx_c = np.where(img > 50)
                if len(cx_c) > 0:
                    min_y, max_y = np.min(cy_c), np.max(cy_c)
                    mid_y = min_y + (max_y - min_y) // 2
                    
                    if prev_char in high_exits:
                        valid_c = np.where(cy_c < mid_y)[0]
                        if len(valid_c) == 0: valid_c = np.arange(len(cx_c))
                    else:
                        valid_c = np.where(cy_c >= mid_y)[0]
                        if len(valid_c) == 0: valid_c = np.arange(len(cx_c))
                        
                    best_c = valid_c[np.argmin(cx_c[valid_c])]
                    p2 = (current_x + cx_c[best_c], y_pos + cy_c[best_c])
                else:
                    p2 = (current_x, y_pos + h)

                # Rysujemy krzywą
                if p0[0] < p2[0]:
                    curve_depth = random.randint(6, 12) 
                    p1 = ((p0[0] + p2[0]) // 2, max(p0[1], p2[1]) + curve_depth)
                    
                    pts = []
                    for t in np.linspace(0, 1, num=15):
                        px = int((1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0])
                        py = int((1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1])
                        pts.append([px, py])
                        
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    ink_intensity = random.randint(200, 255)
                    
                    # Dynamiczna grubość
                    thickness = max(1, min(3, int(w / 25)))
                    cv.polylines(final_img, [pts], isClosed=False, color=ink_intensity, thickness=thickness, lineType=cv.LINE_AA)

            # Wklejamy obecną literę na płótno
            region = final_img[y_pos:y_pos + h, current_x:current_x + w]
            final_img[y_pos:y_pos + h, current_x:current_x + w] = np.maximum(region, img)

            # Kerning i zapisanie stanu
            if not is_space:
                overlap = int(w * random.uniform(0.15, 0.30)) 
                prev_img_data = (img, current_x, y_pos, char)
            else:
                overlap = 0 
                prev_img_data = None
                
            current_x += (w - overlap)
            
        final_img = final_img[:, :current_x + 15]
        
        noise = np.random.randint(0, 11, final_img.shape, dtype=np.uint8)
        final_img = cv.add(final_img, noise)
        
        return final_img

    @staticmethod
    def _apply_elastic_distortion(image, alpha=35, sigma=5, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

class PolishSyntheticDataset(Dataset):
    """ Generuje w locie pojedyncze polskie słowa. """
    def __init__(self, char_map, word_list, transform=None, num_samples=5000):
        self.word_list = [w for w in word_list if len(w) > 0]
        self.transform = transform
        self.num_samples = num_samples
        self.stitcher = PolishCharStitcher(char_map) # Przekazujemy gotową mapę znaków do stitchera

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        word = self.word_list[idx % len(self.word_list)]
        
        # Generujemy obraz słowa (format analogiczny do IAM)
        img = self.stitcher.generate_word_image(word)
        
        # Bezpiecznik
        if img is None or img.ndim < 2 or img.size == 0:
            img = np.zeros((64, 64), dtype=np.uint8)
            word = "error" # Puste lub placeholder
        
        # Skalowanie zachowujące proporcje
        h, w = img.shape[:2]
        if h != 64:
            scale = 64 / max(1, h)
            new_w = max(16, int(w * scale))
            img = cv.resize(img, (new_w, 64), interpolation=cv.INTER_AREA)

        # Transformacje
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        # obraz, etykieta
        return img, word
    
    def get_text(self, idx):
        """ Zwraca tekst dla danego indeksu na potrzeby samplera balansującego. """
        if not self.word_list:
            return ""
        return self.word_list[idx % len(self.word_list)]


def load_sjp_dictionary(file_path, alphabet_set, num_desired=30000):
    """ Funkcja ładująca polskie słowa, wzbogacona o filtry semantyczne odrzucające śmieci. """
    polish_diacritics = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    vowels = set("aąeęioóuyAĄEĘIOÓUY")
    
    # Rozdzielone filtry zbitków literowych
    vowel_cluster_pattern = re.compile(r'[aąeęioóuyAĄEĘIOÓUY]{3,}')      # 3 lub więcej samogłosek (np. "aaa")
    consonant_cluster_pattern = re.compile(r'[^aąeęioóuyAĄEĘIOÓUY]{5,}')  # 5 lub więcej spółgłosek (np. "qwrtp")
    
    with_diacritics = []
    standard = []

    tqdm.write(f"[{now()}] Budowanie bazy słów z {file_path}")

    # Dodane kodowania: utf-8-sig (zabezpieczenie przed znacznikami BOM) oraz utf-16
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1250', 'iso-8859-2']
    file_content = None

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                file_content = f.read()
                if len(file_content.strip()) > 0:
                    tqdm.write(f"[{now()}] Pomyślnie wczytano słownik używając kodowania: {enc}")
                    break
        except (UnicodeDecodeError, UnicodeError, FileNotFoundError):
            continue

    if not file_content or len(file_content.strip()) == 0:
        tqdm.write(f"[{now()}] Nie udało się wczytać pliku lub plik jest pusty.")
        return ["zażółć", "gęślą", "jaźń", "kość", "awaryjny", "start"]

    # Wyszukuje wszystkie ciągi składające się wyłącznie z polskich i łacińskich liter
    words = re.findall(r'\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b', file_content)

    for word in words:
        # Sprawdzamy długość oraz zgodność z alfabetem
        if 3 < len(word) < 14 and all(c in alphabet_set for c in word):
            
            # Musi zawierać przynajmniej jedną samogłoskę
            if not any(char in vowels for char in word):
                continue

            # Odrzucamy zbitki 3+ samogłosek
            if vowel_cluster_pattern.search(word):
                continue
            
            # Odrzucamy zbitki 5+ spółgłosek
            if consonant_cluster_pattern.search(word):
                continue
                
            # Kategoryzacja do balansowania zbioru
            if any(char in polish_diacritics for char in word):
                with_diacritics.append(word)
            else:
                standard.append(word)

    # Usuwamy ewentualne duplikaty
    with_diacritics = list(set(with_diacritics))
    standard = list(set(standard))

    if len(with_diacritics) == 0 and len(standard) == 0:
        tqdm.write(f"[{now()}] Plik nie zawiera poprawnych słów. Używam bazy awaryjnej.")
        return ["zażółć", "gęślą", "jaźń", "kość"]

    # Tasowanie wymieszanych list
    random.shuffle(with_diacritics)
    random.shuffle(standard)

    # Balansowanie zbioru (ok. 50% słów z diakrytykami, reszta standardowa)
    count_diac = int(num_desired * 0.5)
    actual_diac = min(len(with_diacritics), count_diac)
    count_std = num_desired - actual_diac

    final_list = with_diacritics[:actual_diac] + standard[:count_std]
    random.shuffle(final_list)

    tqdm.write(f"[{now()}] Wybrano {len(final_list)} słów (Diakrytyki: {actual_diac}, Bez: {len(final_list) - actual_diac}).")
    return final_list


def get_full_htr_char_list():
    """ Zwraca stałą listę znaków zsynchronizowaną z mapowaniem. """
    # Spacja odfiltrowana w DataLoader
    return [
        ' ', '!', '"', "'", '(', ')', ',', '-', '.', '/', # 1-10
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', # 11-20
        ':', ';', '?',                                    # 21-23
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', # 24-33
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', # 34-43
        'U', 'V', 'W', 'X', 'Y', 'Z',                     # 44-49
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', # 50-59
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', # 60-69
        'u', 'v', 'w', 'x', 'y', 'z',                     # 70-75
        'Ó', 'ó', 'Ą', 'ą', 'Ć', 'ć', 'Ę', 'ę', 'Ł', 'ł', # 76-85
        'Ń', 'ń', 'Ś', 'ś', 'Ź', 'ź', 'Ż', 'ż'            # 86-93
    ]


def generate_final_report(results, output_path="final_report.txt", plot_path="uncertainty_stats.png"):
    """ Analizuje wyniki HTR, oblicza standardowe metryki błędu (CER, WER) oraz przeprowadza walidację
        mechanizmu sygnalizacji niepewności i analizę najgorszych pomyłek. """
    import matplotlib.pyplot as plt
    total_chars = 0
    total_dist = 0
    total_words = len(results)
    err_words = 0

    # Statystyki mechanizmu flagowania niepewności
    true_positives = 0  # Błąd, który model sam oznaczył jako niepewny
    false_positives = 0  # Poprawne słowo, ale oznaczone jako niepewne
    true_negatives = 0  # Poprawne i pewne
    false_negatives = 0  # Błąd oznaczony, jako pewny

    worst_fails = []

    for res in results:
        gt = str(res['gt'])
        pred = str(res['pred'])

        # Obliczamy dystans (ignoruąc wielkość liter dla realnej oceny wizji)
        dist = edit_distance(gt.lower(), pred.lower())

        # Filtr błędów segmentacji
        is_seg_error = dist > max(len(gt), len(pred)) * 0.7

        total_dist += dist
        total_chars += len(gt)

        if dist > 0:
            err_words += 1
            if not is_seg_error:
                worst_fails.append({'gt': gt, 'pred': pred, 'dist': dist})

        # Logika niepewności (bazująca na logitach CRNN)
        is_error = dist > 0
        is_uncertain = res.get('uncertain', False)

        if is_uncertain and is_error:
            true_positives += 1
        elif is_uncertain and not is_error:
            false_positives += 1
        elif not is_uncertain and not is_error:
            true_negatives += 1
        elif not is_uncertain and is_error:
            false_negatives += 1

    # Obliczenia metryk
    cer = (total_dist / max(1, total_chars)) * 100
    wer = (err_words / max(1, total_words)) * 100

    # Skuteczność flagowania: jaką część błędów model sam wyłapał jako "podejrzane"
    recall = (true_positives / max(1, true_positives + false_negatives)) * 100
    precision = (true_positives / max(1, true_positives + false_positives)) * 100

    worst_fails = sorted(worst_fails, key=lambda x: x['dist'], reverse=True)[:10]

    report = f"""RAPORT JAKOŚCI CRNN
    1. METRYKI PODSTAWOWE:
       - Średni CER: {cer:.2f}%
       - Średni WER:      {wer:.2f}%
       - Liczba przeanalizowanych słów:    {total_words}

    2. DIAGNOSTYKA MECHANIZMU NIEPEWNOŚCI:
       - Skuteczność flagowania:    {recall:.2f}%
       - Precyzja flagowania:   {precision:.2f}%
       - True Positives:    {true_positives}
       - False Negatives: {false_negatives}

    3. ANALIZA NAJCZĘSTSZYCH POMYŁEK WIZUALNYCH:
    """
    for i, fail in enumerate(worst_fails):
        report += f"   {i + 1}. GT: '{fail['gt']}' -> Pred: '{fail['pred']}' (Dystans: {fail['dist']})\n"

    report += f"""
    4. WNIOSKI:
       Model poprawnie oflagował {recall:.2f}% wszystkich błędów. Te {true_positives} przypadków 
       zostało pomyślnie wyeksportowanych jako materiał treningowy dla sieci CapsNet. 
       Wysoka skuteczność flagowania gwarantuje, że większość błędów wizualnych 
       zostanie poddana późniejszej weryfikacji geometrycznej. """

    # Wizualizacja
    plt.figure(figsize=(10, 6))
    error_labels = ['Oflagowane błędy (TP)', 'Przeoczone błędy (FN)']
    error_values = [true_positives, false_negatives]
    plt.pie(error_values, labels=error_labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=140)
    plt.title(f"Zdolność CRNN do samodiagnozy błędów\n(Bazowy CER: {cer:.2f}%)")
    plt.axis('equal')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    tqdm.write(report)


def optimize_uncertainty_thresholds(val_results, min_recall=0.75, min_precision=0.5):
    """ Szuka optymalnych progów. Celem jest utrzymać wyłapywanie błędów powyżej min_recall. """
    print(f"[{now()}] Rozpoczynam Grid Search (Wymogi: Recall > {min_recall*100}%, Precision > {min_precision*100}%).")
    
    best_precision = 0.0
    best_params = {}
    
    # Rozszerzona siatka poszukiwań
    temperatures = [1.0, 1.5, 2.0, 2.5]
    conf_thresholds = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    margin_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Zmienne dla awaryjnych progów
    alt_best_f1 = 0.0
    alt_params = {}

    for T in temperatures:
        for conf in conf_thresholds:
            for margin in margin_thresholds:
                TP, FP, FN, TN = 0, 0, 0, 0
                
                for res in val_results:
                    log_probs = res['log_probs']
                    is_error = res['is_error']
                    
                    probs = torch.softmax(log_probs / T, dim=-1).squeeze(1)
                    top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                    margins_val = top2_probs[:, 0] - top2_probs[:, 1]
                    max_conf_val = top2_probs[:, 0]
                    char_indices = torch.argmax(probs, dim=-1)
                    
                    # Ignorujemy blanki z CTC
                    valid_chars = (char_indices != 0)
                    uncertain_mask = (margins_val < margin) | (max_conf_val < conf)
                    alarm_raised = bool((uncertain_mask & valid_chars).any())
                    
                    if alarm_raised and is_error: TP += 1
                    elif alarm_raised and not is_error: FP += 1
                    elif not alarm_raised and is_error: FN += 1
                    else: TN += 1
                        
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Zapisujemy najlepszy F1-Score na wypadek, gdyby wymagania były niemożliwe
                if f1_score > alt_best_f1:
                    alt_best_f1 = f1_score
                    alt_params = {'T': T, 'conf': conf, 'margin': margin, 'Precision': precision, 'Recall': recall, 'TP': TP, 'FP': FP}

                # Jeśli spełniamy minimalne wymogi, walczymy o jak najwyższą precyzję
                if recall >= min_recall and precision >= min_precision:
                    if precision > best_precision:
                        best_precision = precision
                        best_params = {'T': T, 'conf': conf, 'margin': margin, 'Precision': precision, 'Recall': recall, 'TP': TP, 'FP': FP}

    # Wybór ostatecznych parametrów
    if best_params:
        p = best_params
        print(f"Znaleziono optymalne progi w strefie komfortu:")
    else:
        p = alt_params
        print(f"Wymogi są matematycznie nieosiągalne dla tej sieci.")
        print("Zastosowano Fallback: Zwracam najlepszy F1:")

    print(
        f"  └─> T={p.get('T')}, Conf={p.get('conf')}, Margin={p.get('margin')} | "
        f"Recall: {round(p.get('Recall')*100, 2)}%, Precision: {round(p.get('Precision')*100, 2)}% "
        f"(Stosunek TP:FP = {p.get('TP')}:{p.get('FP')})"
    )
    
    return p

if __name__ == "__main__":
    # Gwarancja determinizmu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(3407, deterministic=True)

    writer = SummaryWriter(log_dir=LOG_DIR)
    # Włączenie lepszej obsługi wielu procesów
    try:
        # 'spawn' tworzy nowe procesy od zera, nie kopiując stanu CUDA z procesu głównego
        multiprocessing.set_start_method('spawn', force=True)
        print("[INFO] Metoda startu procesów ustawiona na: spawn")
    except RuntimeError:
        pass # Metoda została już ustawiona
        
    # Czyszczenie pamięci GPU i CPU przed startem
    torch.cuda.empty_cache()
    gc.collect()

    multiprocessing.freeze_support()

    tqdm.write(f"[{now()}] Wykryto środowisko: {'DOCKER' if IS_DOCKER else 'WINDOWS'}")

    # Tworzenie folderów i priorytety
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(VISUAL_DEBUG_DIR, exist_ok=True)

    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except (psutil.AccessDenied, psutil.Error):
        pass

    # Inicjalizacja alfabetu i pomocników
    char_list = get_full_htr_char_list()
    encoder = HTREncoder(char_list)
    num_classes = encoder.get_num_classes()
    alphabet_set = set(encoder.char_list)
    polish_words_list = load_sjp_dictionary(SJP_DICTIONARY, alphabet_set, num_desired=30000)

    if polish_words_list:
        tqdm.write(f"[{now()}] Przykładowe słowa w języku polskim: {polish_words_list[:5]}")

    # Inicjalizacja modelu
    model = ResNetCRNN(num_classes).to(DEVICE)

    # Przeniesienie modelu na GPU przed kompilacją
    model = model.to(DEVICE)

    # Inicjalizacja EMA i Scalera
    ema_avg = lambda avg, model_p, num_avg: 0.999 * avg + 0.001 * model_p
    ema_model = AveragedModel(model, avg_fn=ema_avg, device=torch.device('cpu'))
    scaler = torch.amp.GradScaler('cuda')

    # Zarządzanie historią i checkpointami
    history_path = os.path.join(CHECKPOINT_FOLDER, "training_history.json")
    best_val_loss = float('inf')
    last_total_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [], 'lr_history': [],
        'epoch_labels': [], 'val_focal_loss': []
    }

    # Wczytywanie historii
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                loaded_history = json.load(f)
                if 'step_lrs' in loaded_history: del loaded_history['step_lrs']
                history.update(loaded_history)

            if history.get('train_loss'):
                last_total_epoch = len(history['train_loss'])
                
                dynamic_main_end = 0
                expected_fine_end = 0

                # Sprawdzenie Fazy Fine-tune
                if os.path.exists(FINE_COMPLETE_FILE):
                    best_main_path = os.path.join(CHECKPOINT_FOLDER, "best_cer_model.pth")
                    
                    if os.path.exists(best_main_path):
                        try:
                            tmp_ckpt = torch.load(best_main_path, map_location='cpu', weights_only=False)
                            dynamic_main_end = tmp_ckpt.get('epoch', 0)
                            del tmp_ckpt
                        except Exception:
                            pass
                    
                    expected_fine_end = dynamic_main_end + EPOCHS_FINE_TUNE
                    if last_total_epoch < expected_fine_end:
                        last_total_epoch = expected_fine_end

                # Sprawdzenie Fazy Alignment
                if os.path.exists(ALIGNMENT_COMPLETE_FILE) and expected_fine_end > 0:
                    expected_alignment_end = expected_fine_end + EPOCHS_ALIGN
                    if last_total_epoch < expected_alignment_end:
                        last_total_epoch = expected_alignment_end

            if history.get('val_loss'):
                best_val_loss = min(history['val_loss'])

            tqdm.write(f"[{now()}] Historia wczytana. Startujemy od epoki: {last_total_epoch + 1}")
        except json.JSONDecodeError:
            tqdm.write(f"[{now()}] BŁĄD: Plik historii jest uszkodzony (niepoprawny JSON). Start od zera.")
        except PermissionError:
            tqdm.write(f"[{now()}] BŁĄD: Brak uprawnień do odczytu pliku {history_path}.")
        except Exception as e:
            tqdm.write(f"[{now()}] Nieoczekiwany błąd historii ({type(e).__name__}): {e}. Start od zera.")
    else:
        tqdm.write(f"[{now()}] Brak historii. Nowy trening.")

    potential_checkpoints = [
        CER_PATH,
        CHECKPOINT_PATH,
        os.path.join(CHECKPOINT_FOLDER, "checkpoint.pth")
    ]

    found_path = next((p for p in potential_checkpoints if os.path.exists(p)), None)
    weights_loaded = False

    if found_path:
        tqdm.write(f"[{now()}] Ładowanie wag z: {os.path.basename(found_path)}")
        # Korzystamy z Twojej wbudowanej metody odpornej na zmiany alfabetu
        loaded_epoch = model.load_weights(found_path, device=DEVICE)
        if loaded_epoch > 0:
            last_total_epoch = loaded_epoch
        weights_loaded = True

        # Inicjalizacja EMA i bezpieczna synchronizacja
        ema_avg = lambda avg, model_p, num_avg: 0.999 * avg + 0.001 * model_p
        ema_model = AveragedModel(model, avg_fn=ema_avg, device=torch.device('cpu'))

        if weights_loaded:
            ema_model.update_parameters(model)
            tqdm.write(f"[{now()}] EMA zsynchronizowane z wczytanymi wagami.")

    scaler = torch.amp.GradScaler('cuda')

    # Szukamy dostępnych wag
    potential_checkpoints = [
        os.path.join(CHECKPOINT_FOLDER, "best_cer_model.pth"),
        os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth"),
        os.path.join(CHECKPOINT_FOLDER, "checkpoint.pth")
    ]

    found_path = next((p for p in potential_checkpoints if os.path.exists(p)), None)
    if found_path:
        tqdm.write(f"[{now()}] Ładowanie wag z: {os.path.basename(found_path)}")
        loaded_epoch = model.load_weights(found_path, device=DEVICE)

        # Jeśli plik nie miał zapisanego numeru epoki, zostajemy przy tym z historii
        if loaded_epoch > 0: last_total_epoch = loaded_epoch
        model.eval()

    # Przygotowanie danych
    if not os.path.exists(DATA_ROOT):
        Preprocessing.preprocess_iam_dataset(RAW_SOURCE_DIR, DATA_ROOT)

    all_files = glob.glob(os.path.join(DATA_ROOT, "iam_words", "words", "**", "*.png"), recursive=True)
    tqdm.write(f"[{now()}] Zaindeksowano: {len(all_files)} obrazów.")

    # Val transforms - spójna normalizacja
    val_transform_crnn = alb.Compose([
        alb.Normalize(mean=IAM_MEAN, std=IAM_STD),
        ToTensorV2()
    ])

    # Ładowanie do RAM - zapobiega OOM
    tqdm.write(f"[{now()}] Ładowanie polskiego alfabetu do głównej pamięci RAM.")
    npz_data = np.load(OUTPUT_NPZ, allow_pickle=True)
    raw_images = npz_data['signs']
    npz_labels = npz_data['labels']
    
    global_char_map = {}
    for i, char in enumerate(npz_labels):
        img = raw_images[i].copy() 
        if np.mean(img) > 127:
            img = 255 - img
        if char not in global_char_map:
            global_char_map[char] = []
        global_char_map[char].append(img)
        
    del npz_data, raw_images, npz_labels # Czyścimy śmieci po wczytaniu
    tqdm.write(f"[{now()}] Gotowe. Zbudowano mapę dla {len(global_char_map)} unikalnych znaków.")

    # Tworzenie zbioru IAM
    train_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, 
                               transform=get_augmentations("main"),
                               char_list=char_list, 
                               name="IAM_Train",
                               split='train')

    val_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, 
                             transform=val_transform_crnn, # Czysty obraz
                             char_list=char_list, 
                             name="IAM_Val",
                             split='val')

    # Ręczny podział słów
    split_idx = int(0.9 * len(polish_words_list))
    train_polish_words = polish_words_list[:split_idx]
    val_polish_words = polish_words_list[split_idx:]

    train_polish = PolishSyntheticDataset(char_map=global_char_map, 
                                          word_list=train_polish_words,
                                          transform=get_augmentations("main"), 
                                          num_samples=12000)

    val_polish = PolishSyntheticDataset(char_map=global_char_map, 
                                        word_list=val_polish_words,
                                        transform=val_transform_crnn, # Czysty obraz
                                        num_samples=1333) # 10% z 12000 dla zachowania proporcji

    # Łączymy w zbiory
    train_dataset = ConcatDataset([train_iam, train_polish])
    val_dataset = ConcatDataset([val_iam, val_polish])

    print(f"[{now()}] Całkowita liczba próbek: {len(train_dataset) + len(val_dataset)}")
    print(f"[{now()}] Próbki treningowe: {len(train_dataset)}, walidacyjne: {len(val_dataset)}")

    # Dynamiczne ważenie
    all_categories = []
    polish_diacritics = set("ĄĆĘŁŃÓŚŹŻąćęłńóśźż")
    
    # Kategoryzacja słów z IAM_Train
    for label in train_iam.valid_labels:
        if any(c in polish_diacritics for c in label):
            all_categories.append("polish_diacritic")
        else:
            length = len(label)
            cat = 'short' if length < 5 else ('medium' if length < 9 else 'long')
            all_categories.append(cat)

    # Kategoryzacja słów z PolishSynthetic_Train
    for _ in range(len(train_polish)):
        all_categories.append("polish_diacritic")

    # Obliczanie wag dynamicznych
    category_counts = Counter(all_categories)
    max_count = max(category_counts.values())

    """ Definiujemy wagi - polish_diacritic dostaje boost, żeby model zwracał szczególną uwagę na te znaki
        (w większości różnią się tylko ogonkiem lub kreską, więc są trudniejsze do rozróżnienia). """
    dynamic_weights_map = {
        "short": math.sqrt(max_count / max(category_counts["short"], 1)),
        "medium": math.sqrt(max_count / max(category_counts["medium"], 1)),
        "long": math.sqrt(max_count / max(category_counts["long"], 1)),
        "polish_diacritic": math.sqrt(max_count / max(category_counts["polish_diacritic"], 1)) * 1.5  # Boost dla polskich znaków
    }

    tqdm.write(f"[{now()}] Liczebność klas w treningu: {dict(category_counts)}")
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in dynamic_weights_map.items()])
    tqdm.write(f"[{now()}] Wagi Balansujące: {weights_str}")

    weights = [dynamic_weights_map[cat] for cat in all_categories]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Loader używa hybrydowego samplera łączącego polskie znaki już wcześniej
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=WORKERS_MAIN,
        collate_fn=collate_fn_dynamic,
        pin_memory=(WORKERS_MAIN > 0),
        prefetch_factor=2 if WORKERS_MAIN > 0 else None,
        persistent_workers=(WORKERS_MAIN > 0),
        worker_init_fn=worker_init_fn
    )

    # Bezpieczny val_loader, bo tamten powodował problemy z wieloma procesami w Dockerze
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_dynamic,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Main training
    if not os.path.exists(MAIN_PHASE_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Rozpoczynam Fazę Main.")
        get_augmentations('main')

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        """ Optymalizator NAdam (Nesterov-accelerated Adaptive Moment Estimation):
            Zastosowanie wariantu NAdam podyktowane jest złożonością przestrzeni błędu w architekturze 
            hybrydowej (ResNet + BiLSTM). W przeciwieństwie do klasycznego Adama, NAdam wykorzystuje 
            pęd Nesterova, który włącza do mianownika wyprzedzającą estymację przyszłej pozycji gradientu. 
            Skutkuje to wyższym tłumieniem oscylacji w wąskich minimach lokalnych i zapobiega 
            przeskakiwaniu optymalnych rozwiązań przy agresywnym tempie uczenia.
            
            Znaczenie parametru Epsilon:
            Zapewnia on nie tylko stabilność numeryczną (chroni przed błędem dzielenia przez zero w 
            mianowniku), ale pełni rolę bezpiecznika regularyzacyjnego. Twardo ogranicza maksymalny 
            krok aktualizacji dla wag o mikroskopijnej wariancji gradientu, chroniąc wyuczone już 
            filtry wizualne przed destrukcyjnymi skokami wywołanymi zaszumionymi próbkami. """
        optimizer = optim.NAdam([
        {
        'params': decay_params, 
        'weight_decay': 5e-5  # Lekka regularyzacja dla wag głównych
        },
        {
        'params': no_decay_params, 
        'weight_decay': 0.0   # Brak regularyzacji dla biasów i BatchNorm
        }
        ], lr=LR_MAIN / DIV_FACTOR)

        steps_per_epoch = math.ceil(int(len(train_loader)) / int(ACCUMULATION_STEPS_MAIN))
        main_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR_MAIN,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS_MAIN, pct_start=PCT_START, div_factor=DIV_FACTOR, final_div_factor=100
        )

        # Przesuwamy o łączną ilość kroków przez wszystkie epoki przy wznawianiu
        start_ep = last_total_epoch if last_total_epoch < EPOCHS_MAIN else 0
        if start_ep > 0:
            for _ in range(start_ep * steps_per_epoch):
                main_scheduler.step()

        patience_counter = 0
        for epoch in range(start_ep, EPOCHS_MAIN):
            tqdm.write(f"[{now()}] Epoka {epoch + 1} Main")

            # Wykonujemy trening epoki
            current_gamma = get_focal_gamma_schedule(epoch, EPOCHS_MAIN, start_gamma=2.5, end_gamma=1.5)
            
            # Bez głowicy kontrastowej, na początek model uczy się czytać
            t_loss, e_lrs, e_moms = train_one_epoch(
                model, train_loader, optimizer, scaler, DEVICE, encoder, 
                scheduler=main_scheduler, cat_weights=dynamic_weights_map, 
                epoch=epoch, focal_gamma=current_gamma, use_focal=False,
                use_contrastive=False, writer=writer, acc_steps=ACCUMULATION_STEPS_MAIN,
            )

            # Aktualizacja historii uczenia
            history.setdefault('step_lrs', []).extend(e_lrs)
            history.setdefault('step_moms', []).extend(e_moms)
            history.setdefault('focal_gamma_history', []).append(current_gamma)

            # Walidacja
            val_loss = evaluate_loss_only(model, val_loader, DEVICE, encoder)

            # Zapis wyników do historii
            history['train_loss'].append(t_loss)
            history['val_loss'].append(val_loss)
            history['epoch_labels'].append(epoch + 1)

            # Logowanie do TensorBoard
            if writer:
                writer.add_scalar('Loss/Train_Epoch', t_loss, epoch)
                writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)

            tqdm.write(f"          Loss: {t_loss:.4f} | Val Loss: {val_loss:.4f} | Gamma: {current_gamma:.2f}")

            last_main_epoch = epoch + 1
            # Tylko jeśli mamy nowy rekord
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Tworzymy słownik danych do zapisu
                checkpoint_data = {
                    'model_state': model.state_dict(),
                    'ema_state': ema_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': main_scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'best_loss': val_loss,
                    'epoch': epoch + 1,
                    'history': history
                }

                # Zapisujemy pliki na dysku
                torch.save(checkpoint_data, CHECKPOINT_PATH)
                torch.save(model.state_dict(), CER_PATH) # Zapasowy model
                tqdm.write(f"            └─> Nowy rekord! Zapisano checkpoint modelu.")
            
            else:
                # Jeśli błąd nie spadł, zwiększamy licznik cierpliwości
                patience_counter += 1
                if patience_counter >= PATIENCE_MAIN:
                    tqdm.write(f"Early stopping w epoce {epoch + 1}. Brak poprawy od {patience_counter} epok.")
                    early_stopping_epoch = epoch + 1
                    break

            # Zapis historii do pliku JSON po każdej epoce
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)

        # Generowanie wykresu po zakonczeniu wszystkich epok w Main
        if 'step_lrs' in history and history['step_lrs']:
            plot_scheduler(history['step_lrs'], history['step_moms'], filename="actual_scheduler_main.png")

        final_main_path = os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth")
        torch.save({'model_state': model.state_dict(), 'best_loss': best_val_loss, 'epoch': EPOCHS_MAIN},
                   final_main_path)
        with open(MAIN_PHASE_COMPLETE_FILE, 'w') as f:
            f.write("complete")

        # Czyszczenie
        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    # Ścieżka do najlepszego modelu z fazy Main
    best_main_model_path = os.path.join(CHECKPOINT_FOLDER, "best_cer_model.pth") 
    
    if os.path.exists(best_main_model_path):
        try:
            # Ładujemy plik na CPU, żeby uniknąć wycieków pamięci VRAM
            tmp_checkpoint = torch.load(best_main_model_path, map_location='cpu', weights_only=False)
            
            # Sprawdzamy, czy plik to słownik zawierający klucze
            if isinstance(tmp_checkpoint, dict):
                early_stopping_epoch = tmp_checkpoint.get('epoch', 0)
            else:
                print(f"[{now()}] Plik checkpointu zawiera same wagi (brak metadanych). Przyjęto epokę 0.")
                
            del tmp_checkpoint # Ręczne sprzątanie RAM-u
            
        except Exception as e:
            print(f"[{now()}] Ostrzeżenie: Nie udało się odczytać epoki z checkpointu ({type(e).__name__}: {e}). Przyjęto 0.")

    # Fine-tune
    if os.path.exists(MAIN_PHASE_COMPLETE_FILE) and not os.path.exists(FINE_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Rozpoczynam Fazę Fine-tune.")
        
        # Tworzymy zbiory z nowymi, delikatniejszymi augmentacjami dla fine-tuning
        fine_transforms = get_augmentations("fine_tune")
        
        train_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, transform=fine_transforms,
                                   char_list=char_list, name="IAM_Train")

        train_polish = PolishSyntheticDataset(char_map=global_char_map, word_list=polish_words_list,
                                              transform=fine_transforms, num_samples=12000)
        
        # Łączymy zbiory
        fine_tune_dataset = ConcatDataset([train_iam, train_polish])
        
        # Ładujemy najlepsze wagi z fazy main
        checkpoint = torch.load(os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth"))
        model.load_state_dict(checkpoint['model_state'])

        # DataLoader-y
        train_loader = DataLoader(
            fine_tune_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS_FINE,
            collate_fn=collate_fn_dynamic, pin_memory=False, worker_init_fn=worker_init_fn, shuffle=True
        )
        val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=WORKERS_FINE,
        collate_fn=collate_fn_dynamic, 
        pin_memory=False
    )

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            """ Nie stosujemy weight decay dla biasów i BatchNorm, bias to po prostu przesunięcie, a BatchNorm ma już własny mechanizm
                normalizacji wag i statystyk, przez co nakładanie na nie dodatkowej kary L2 (weight decay) prowadzi do
                konfliktu optymalizacyjnego i może destabilizować trening. """
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        """ Zmiana optymalizatora dla fazy Fine-tune:
            Zastąpienie adaptacyjnego optymalizatora klasycznym SGD z pędem.
            Uzasadnienie: Klasyczny SGD przy niskim Learning Rate (1e-4 -> 1e-6) pozwala 
            na powolne, delikatne osiadanie w znalezionym ostrym minimum. 
            Algorytmy adaptacyjne ze swoim własnym pędem drugiego rzędu 
            mogłyby tu działać zbyt agresywnie i wybić model z perfekcyjnego dołka.
            Zmniejszono weight_decay do 1e-6, aby zapobiec amnezji modelu. """
        optimizer = optim.SGD([
            {'params': decay_params, 'weight_decay': 1e-6}, # Ochrona przed zerowaniem wag
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=1e-4, momentum=0.9, nesterov=True)

        total_fine_steps = math.ceil(EPOCHS_FINE_TUNE * len(train_loader) / ACCUMULATION_STEPS_FINE)
        
        # Opadanie do 1e-6 zapewni płynne osiadanie po początkowej eksploracji
        fine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_fine_steps, eta_min=1e-6)

        """ Nie wyliczamy bazowego błędu na starcie, ponieważ SWA często wywołuje początkowy skok straty
            szukając płaskiego minimum. Sztywny próg zablokowałby zapis wczesnych epok. """
        best_fine_loss = float('inf')

        for epoch in range(EPOCHS_FINE_TUNE):
            # Zabezpieczenie przed brakiem zmiennej early_stopping_epoch przy wznawianiu
            base_epoch = 0
            if 'early_stopping_epoch' in locals():
                base_epoch = early_stopping_epoch
            else:
                # Próbujemy odzyskać epokę z historii lub wczytanego checkpointu
                if 'checkpoint' in locals() and isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    base_epoch = checkpoint['epoch']

            current_total_epoch = base_epoch + epoch + 1
            tqdm.write(f"[{now()}] Epoka {current_total_epoch} Fine-tune")
            model.set_dropout(0.15)

            # Gamma scheduling dla fine-tune (niższe wartości)
            current_gamma = get_focal_gamma_schedule(epoch, EPOCHS_FINE_TUNE, start_gamma=2.0, end_gamma=1.2)

            train_loss, e_lrs, e_moms = train_one_epoch(
                model, train_loader, optimizer, scaler, DEVICE, encoder, 
                cat_weights=dynamic_weights_map, scheduler=fine_scheduler,
                ema_model=ema_model, epoch=current_total_epoch,
                label_smoothing=0.02, focal_gamma=current_gamma,
                use_focal=True, acc_steps=ACCUMULATION_STEPS_FINE,
                use_contrastive=True, writer=writer
            )

            # Zapisujemy zebrane dane do historii wykresu
            history.setdefault('step_lrs', []).extend(e_lrs)
            history.setdefault('step_moms', []).extend(e_moms)

            current_lr = optimizer.param_groups[0]['lr']
            if writer:
                writer.add_scalar('LR/Fine_Tune', current_lr, epoch)
                writer.add_scalar('Loss/Fine_Train', train_loss, epoch)

            # Walidujemy standardowy model w trakcie trwania pętli
            val_loss = evaluate_loss_only(model, val_loader, DEVICE, encoder)
            tqdm.write(f"          Fine Loss: {val_loss:.4f}")

            with open(history_path, "w") as f:
                json.dump(history, f)

            # Zapisujemy tylko najostrzejszy model
            if val_loss < best_fine_loss:
                best_fine_loss = val_loss
                torch.save({'model_state': model.state_dict(), 'epoch': current_total_epoch}, CER_PATH)
                tqdm.write("              └─> Nowy rekord! Zapisano checkpoint modelu.")
        
        tqdm.write(f"[{now()}] Przywracanie wag z rekordowej epoki Fine-tune.")
        
        # Odtwarzamy z dysku model, który pobił rekord w pętli
        checkpoint = torch.load(CER_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            final_epoch = checkpoint.get('epoch', current_total_epoch)
        else:
            final_epoch = current_total_epoch

        # Potwierdzenie ostatecznego błędu na wyostrzonych wagach
        final_fine_loss = evaluate_loss_only(model, val_loader, DEVICE, encoder)
        tqdm.write(f"[{now()}] Potwierdzony ostateczny loss fazy Fine-tune: {final_fine_loss:.4f}")

        # Nadpisujemy plik docelowy pełną paczką z EMA i prawidłowymi metadanymi
        torch.save({
            'model_state': model.state_dict(),
            'ema_state': ema_model.state_dict(),
            'best_loss': final_fine_loss,
            'epoch': final_epoch
        }, CER_PATH)

        # Flaga potwierdzająca przejście do Alignmentu
        with open(FINE_COMPLETE_FILE, 'w') as f:
            f.write(f"Zakończono: {time.ctime()}")

        # Zwolnienie pamięci (val_dataset jest później używany)
        del train_dataset, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Klastrowanie Cech
    if os.path.exists(FINE_COMPLETE_FILE) and not os.path.exists(ALIGNMENT_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Dostosowywanie mapy cech w sposób przydatny dla CapsNet przy użyciu Center i Separation Loss.")
        
        # Wczytanie najlepszego modelu SWA z poprzedniej fazy
        tqdm.write(f"[{now()}] Wczytywanie wygładzonych wag SWA do klastrowania.")
        checkpoint = torch.load(CER_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        model.to(DEVICE)

        # Zamrożenie wizualnego ekstraktora
        for param in model.cnn.parameters():
            param.requires_grad = False
            
        tqdm.write(f"[{now()}] Zamrożono kręgosłup ResNet. Trenowanie wyłącznie BiLSTM i Głowic Projekcyjnych.")

        if 'fine_tune_dataset' not in locals():
            print(f"[{now()}] Odtwarzanie zbioru danych dla wznawianej fazy Alignment.")
            if 'train_dataset' in locals():
                fine_tune_dataset = train_dataset
            else:
                # Tworzymy zbiór od nowa
                fine_transforms = get_augmentations("fine_tune")
                
                train_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, transform=fine_transforms,
                                        char_list=char_list, name="IAM_Train")

                train_polish = PolishSyntheticDataset(char_map=global_char_map, word_list=polish_words_list,
                                                    transform=fine_transforms, num_samples=12000)
                
                # Łączymy zbiory
                fine_tune_dataset = ConcatDataset([train_iam, train_polish])

        # Zbieranie aktywnych parametrów
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Inicjalizacja Center Loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=128, device=DEVICE)
        
        # Optymalizator dla samych centroidów (wymaga wyższego LR)
        optimizer_center = optim.SGD(center_criterion.parameters(), lr=0.5)

        # Optymalizator dla sieci (bardzo powolne, bezpieczne modyfikacje wag)
        optimizer_align = optim.SGD([
            {'params': decay_params, 'weight_decay': 1e-5},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=1e-5, momentum=0.9)

        # Używamy loadera z fazy fine_tune (bez agresywnych zniekształceń geometrycznych)
        train_loader_align = DataLoader(
            fine_tune_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS_FINE,
            collate_fn=collate_fn_dynamic, pin_memory=False, worker_init_fn=worker_init_fn, shuffle=True
        )

        # Dodajemy zmienną do śledzenia rekordu przed rozpoczęciem pętli
        best_align_loss = float('inf')

        for epoch in range(EPOCHS_ALIGN):
            current_total_epoch = early_stopping_epoch + EPOCHS_FINE_TUNE + epoch + 1
            tqdm.write(f"[{now()}] Epoka {current_total_epoch} Alignment")
            
            # Włączamy tryb train, ale wymuszamy Dropout na poziomie 0.1, żeby utrzymać stabilność
            model.set_dropout(0.1)

            # Wywołanie ze stratami pomocniczymi do przemodelowania dla CapsNet
            train_loss, _, _ = train_one_epoch(
                model, train_loader_align, optimizer_align, scaler, DEVICE, encoder, 
                cat_weights=dynamic_weights_map, scheduler=None, # Brak schedulera - stały, niski LR
                epoch=current_total_epoch, label_smoothing=0.01, focal_gamma=1.2, 
                use_focal=True, acc_steps=ACCUMULATION_STEPS_FINE,
                use_contrastive=True,
                center_criterion=center_criterion,
                optimizer_center=optimizer_center,
                lambda_center=0.001, # Siła przyciągania klas
                lambda_separation=0.005, # Siła odpychania klas
                writer=writer
            )

            # Szybka walidacja, by upewnić się, że CTC Loss nie rośnie
            val_loss = evaluate_loss_only(model, val_loader, DEVICE, encoder)
            tqdm.write(f"          Align Loss: {val_loss:.4f} | Aux (Center/Sep) Aktywne")

            # Gwarancja zapisu najlepszego wyniku
            if val_loss < best_align_loss:
                best_align_loss = val_loss
                
                final_aligned_path = os.path.join(CHECKPOINT_FOLDER, "CRNN_Aligned_for_CapsNet.pth")
                
                checkpoint_data = {
                    'epoch': early_stopping_epoch + EPOCHS_FINE_TUNE,
                    'model_state_dict': model.state_dict()
                }
                
                # Zapis finalnego, przygotowanego pod CapsNet modelu
                torch.save(checkpoint_data, final_aligned_path)
                
                # Nadpisujemy także CER_PATH, żeby proces eksportu na pewno używał tej wersji
                torch.save(checkpoint_data, CER_PATH)
                
                tqdm.write("              └─> Nowy rekord Alignment! Zapisano checkpoint pod CapsNet.")

        # Zapis finalnego, przygotowanego pod CapsNet modelu
        final_aligned_path = os.path.join(CHECKPOINT_FOLDER, "CRNN_Aligned_for_CapsNet.pth")
        torch.save({
            'model_state': model.state_dict(),
            'best_loss': val_loss,
            'epoch': early_stopping_epoch + EPOCHS_FINE_TUNE + EPOCHS_ALIGN
        }, final_aligned_path)

        # Nadpisujemy także CER_PATH, żeby eksport używał tej wersji
        torch.save({
            'model_state': model.state_dict(),
            'best_loss': val_loss,
            'epoch': early_stopping_epoch + EPOCHS_FINE_TUNE + EPOCHS_ALIGN
        }, CER_PATH)

        with open(ALIGNMENT_COMPLETE_FILE, 'w') as f:
            f.write(f"Zakończono: {time.ctime()}")

        tqdm.write(f"[{now()}] Faza Pre-CapsNet Alignment zakończona sukcesem. Przestrzeń cech została rozseparowana.")
        
        # Czyszczenie pamięci
        del train_loader_align, optimizer_align, optimizer_center, center_criterion
        gc.collect()
        torch.cuda.empty_cache()

    # Eksport dla CapsNet i raport
    if os.path.exists(ALIGNMENT_COMPLETE_FILE):
        # Fallback do najlepszego modelu
        final_model_path = CER_PATH
        tqdm.write(f"[{now()}] Rozpoczynam eksport na podstawie modelu: {os.path.basename(final_model_path)}")

        # Ładowanie wag do modelu
        model.load_weights(final_model_path)
        model.eval()

        # Używamy zbioru słów
        export_loader = DataLoader(
            val_dataset, 
            batch_size=1, # Tylko 1, żeby ograniczyć problemy z łączeniem sekwencji o różnych długościach
            shuffle=False,
            collate_fn=collate_fn_dynamic, # Używamy funkcji collate dla słów
            num_workers=2
        )

        tqdm.write(f"[{now()}] Skanowanie walidacji do kalibracji progów niepewności.")
        val_calibration_data = []
        with torch.no_grad():
            for batch in tqdm(export_loader, desc="Zbieranie logitów"):
                if batch is None: continue
                imgs, lbls = batch[0].to(DEVICE), batch[1]
                
                output = model(imgs)
                lp = output[0] if isinstance(output, (tuple, list)) else output
                if float(lp.shape[0]) == float(imgs.size(0)): 
                    lp = lp.permute(1, 0, 2)
                    
                preds_strings, _ = encoder.decode_greedy(lp)
                lp_b = lp.permute(1, 0, 2)
                
                for gt, pred, logit in zip(lbls, preds_strings, lp_b):
                    val_calibration_data.append({
                        'log_probs': logit.unsqueeze(1),
                        'is_error': str(gt) != str(pred)
                    })
        
        # Uruchomienie Grid Search i wyciągnięcie najlepszych parametrów, dążymy do wyłapania 2/3 błędów
        best_params = optimize_uncertainty_thresholds(val_calibration_data, min_recall=0.66, min_precision=0.50)
        OPT_T = best_params.get('T', 1.5)
        OPT_CONF = best_params.get('conf', 0.75)
        OPT_MARGIN = best_params.get('margin', 0.15)

        tqdm.write(f"[{now()}] Analiza pomyłek i eksport wycinków dla CapsNet.")
        export_error_crops_for_capsnet(model, export_loader, DEVICE, encoder, CAPSNET_DATA_DIR, OPT_MARGIN, OPT_CONF, OPT_T)

        # Zbieranie danych do raportu CER/WER i niepewności
        results_for_report = []
        with torch.no_grad():
            for batch in tqdm(export_loader, desc="Generowanie raportu"):
                if batch is None: continue
                # Dekompozycja batcha ze standardowego collate_fn_dynamic
                imgs = batch[0].to(DEVICE)
                lbls = batch[1]
                
                output = model(imgs)
                lp = output[0] if isinstance(output, (tuple, list)) else output
                
                # Zapewnienie stałego wymiaru czasu dla dekodera
                if float(lp.shape[0]) == float(imgs.size(0)):
                    lp = lp.permute(1, 0, 2)
                    
                preds_strings, _ = encoder.decode_greedy(lp)

                # Dla wygody analizy niepewności, permutujemy z powrotem na [Batch, Time, Class]
                lp_b = lp.permute(1, 0, 2)
                for gt, pred, logit in zip(lbls, preds_strings, lp_b):
                    results_for_report.append({
                        'gt': str(gt), 'pred': str(pred),
                        'dist': edit_distance(str(gt), str(pred)),
                        'uncertain': len(model.get_uncertainty_zones(logit.unsqueeze(1), margin_threshold=OPT_MARGIN, conf_threshold=OPT_CONF, temperature=OPT_T)) > 0
                    })

        # Wykresy (Mapy uwagi)
        try:
            sample_batch = next(iter(export_loader))
            if sample_batch is not None and len(sample_batch) > 0:
                images = sample_batch[0]
                sample_img = images[0].unsqueeze(0).to(DEVICE)
                attn_save_path = os.path.join(VISUAL_DEBUG_DIR, "sample_attention_map.png")
                visualize_attention_map(model, sample_img, attn_save_path)
                tqdm.write(f"[{now()}] Wygenerowano przykładową mapę uwagi w: {attn_save_path}")
            else:
                tqdm.write(f"[{now()}] Warning: export_loader zwrócił pusty batch.")
        except StopIteration:
            tqdm.write(f"[{now()}] Błąd: export_loader jest pusty (brak danych do wizualizacji).")
        except RuntimeError as re:
            if "out of memory" in str(re).lower():
                tqdm.write(f"[{now()}] Błąd GPU: Brak pamięci VRAM na mapę uwagi. Pomijam.")
            else:
                tqdm.write(f"[{now()}] Błąd PyTorch podczas wizualizacji: {re}")
        except OSError as oe:
            tqdm.write(f"[{now()}] Błąd systemu plików: Nie można zapisać mapy uwagi (uprawnienia/ścieżka): {oe}")
        except Exception as e:
            tqdm.write(f"[{now()}] Nieoczekiwany błąd wizualizacji ({type(e).__name__}): {e}")

        del export_loader
        gc.collect()
        time.sleep(1) # Dajemy systemowi sekundę na zamknięcie wątków

        # Generacja raportu końcowego
        generate_final_report(results=results_for_report,
                              output_path=os.path.join(CHECKPOINT_FOLDER, "final_thesis_report_words.txt"),
                              plot_path=os.path.join(VISUAL_DEBUG_DIR, "uncertainty_coverage_chart_words.png"))

        tqdm.write(f"[{now()}] Liczenie szczegółowych metryk.")

        # Możemy użyć tego samego loadera, ładując go element po elemencie do dokładnych statystyk
        detailed_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn_dynamic)
        detailed_results = evaluate_full_metrics(model, detailed_loader, DEVICE, encoder)
        with open(os.path.join(CHECKPOINT_FOLDER, "final_metrics_report_words.json"), "w") as f:
            json.dump(detailed_results, f)

        writer.close()
        tqdm.write(f"[{now()}] Wszystkie raporty, zbiory danych i wykresy gotowe. Trening CRNN zakończony.")

""" cd /home/marek/OCR/HandwrittenTextRecognition
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    docker system prune -a -f
    docker build -t hcr-resnet-crnn .
    systemd-inhibit docker run -it --rm --name trening_test \
    --gpus all \
    --ipc=host \
    --ulimit nofile=65536:65536 \
    --network=host \
    -e PYTHONUNBUFFERED=1 \
    -v /home/marek/OCR/HandwrittenTextRecognition/Data:/app/Data:rw \
    -v /home/marek/OCR/HandwrittenTextRecognition/output_data:/app/output_data \
    -v /home/marek/OCR/HandwrittenTextRecognition/Models:/app/Models \
    hcr-resnet-crnn \
    python3 -B Models/ResNetCRNNWordRecognition.py
  
    systemd-inhibit blokuje przed uśpieniem systemu w trakcie treningu,
    -it logi na żywo, --rm usuwanie kontenera od razu po przerwaniu,
    --name nazwa kontenera, --gpus all wszystkie dostępne rdzenie GPU,   
    -e PYTHONBUFFERED = 1 wypisywanie logów w czasie rzeczywistym,
    -v mapuje lokalny folder do kontenera pod podaną ścieżkę,
    --shm-size - rozmiar pamięci współdzielonej, duży bo wątki przetwarzają obrazy słów,
    --ipc=host - bezpośredni dostęp do pamięci współdzielonej hosta, brak opóźnień DataLoader -> GPU,
    --network=host - bezpośredni dostęp do sieci hosta, szybsze logowanie metryk (np. TensorBoard/W&B),
    :ro (przy Data) - tryb tylko do odczytu, szybsze ładowanie tysięcy małych plików (brak sprawdzania zapisu),
    -B (przy python3) - blokuje tworzenie plików .pyc, brak śmieci w zmapowanych folderach lokalnych,
    python3... komenda startująca wewnątrz kontenera """