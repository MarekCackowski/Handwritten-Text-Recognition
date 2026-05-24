import os
from unittest import loader
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

import concurrent.futures
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import psutil
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
WORKERS_MAIN = 4 # Augmentacje
WORKERS_FINE = 8
WORKERS_HARD_MINING = 4 # H5PY na Linuxie działa stabilnie przy wielu wątkach, jeśli plik jest otwierany w __getitem__

# Urządzenie
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Podstawowe ścieżki
RAW_SOURCE_DIR = os.path.normpath(os.path.join(DATA_ROOT, "iam_words", "words"))
PROCESSED_DATA_DIR = os.path.normpath(os.path.join(DATA_ROOT, "iam_words", "words_processed"))

# Checkpointy i wyniki
CHECKPOINT_FOLDER = os.path.normpath(os.path.join(OUTPUT_BASE, "checkpoints", "hwr"))
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth")
CER_PATH = os.path.join(CHECKPOINT_FOLDER, "best_cer_model.pth")

# Pliki baz danych H5
IAM_LINES_H5_PATH = "/app/Data/ocr_dataset_cleaned.h5"
IAM_WORDS_H5_PATH = "/app/Data/iam_words.h5"
CVL_H5_PATH = "/app/Data/cvl_lines_en.h5"

# Pozostałe ścieżki pomocnicze
MAIN_PHASE_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "main_complete.txt")
FINE_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "fine_complete.txt")
HARD_MINING_COMPLETE_FILE = os.path.join(CHECKPOINT_FOLDER, "hard_mining_complete.txt")
CAPSNET_DATA_DIR = os.path.normpath(os.path.join(DATA_ROOT, "archive", "iam_words"))
VISUAL_DEBUG_DIR = os.path.join(CHECKPOINT_FOLDER, "visual_debug_CRNN")
LOG_DIR = os.path.join(CHECKPOINT_FOLDER, "logs")

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(VISUAL_DEBUG_DIR, exist_ok=True)
os.makedirs(CAPSNET_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Konfiguracja GPU pod uczenie i Tensorboard do podsumowań CER/WER
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
writer = SummaryWriter(log_dir=LOG_DIR)

IMAGE_HEIGHT = 64
WORD_WIDTH = 576
LINE_WIDTH = 3000
TOTAL_HARD_LINES = 64
SAFE_BATCH_SIZE = 4
STOP_THRESHOLD = 0.001
VAL_LOSS_THRESHOLD = 0.0002
ENTROPY_THRESHOLD = 0.5
CONF_THRESHOLD = 0.7
ENT_THRESHOLD = 0.5
PURE_RATE = 0.05
UPPER_RATE = 0.3
MAX_PER_CLASS = 1500
DIV_FACTOR = 20
PCT_START = 0.3
EMNIST_MEAN = (0.1736,)
EMNIST_STD = (0.3317,)
MEAN, STD = 0.1307, 0.3081
MATRIX_PATH = os.path.join(CHECKPOINT_FOLDER, "confusion_matrix_final")
FINAL_MODEL_LINES_PATH = os.path.join(CHECKPOINT_FOLDER, "Final_HTR_Model_Lines.pth")

""" Parametry batchowania i akumulacji gradientów na poziomie słów:
    Głównym wąskim gardłem w module CRNN jest ekstrakcja cech oraz kosztowna 
    pamięciowo obsługa sekwencji podczas fazy uczenia. Zastosowano fizyczny batch 4, 
    aby pomieścić w pamięci VRAM wszystkie mapy cech niezbędne do obliczenia gradientów. 
    Dzięki akumulacji uzyskujemy efektywny batch 64, co pozwala modelowi na przegląd
    szerokiego spektrum stylów pisma przed aktualizacją wag. Stabilizuje to gradienty
    i uśrednia szum wynikający z nietypowych liter. W fazie inferencji proces jest
    błyskawiczny, ponieważ nie wymaga zapamiętywania stanów dla backpropagation. """
BATCH_SIZE_WORDS = 4
ACCUMULATION_STEPS = 16

""" Parametry batchowania i akumulacji gradientów na poziomie linii:
    Ustawienie BATCH_SIZE_LINES = 1 wynika z dużej szerokości obrazów linii oraz kosztów 
    algorytmu Backpropagation Through Time. Przy długich sekwencjach warstwy splotowe 
    generują ogromne mapy cech, a warstwy rekurencyjne muszą przechowywać historię stanów 
    ukrytych dla każdej klatki czasowej w pamięci VRAM, aby wyliczyć gradienty. 
    Poprzez ACCUMULATION_STEPS_LINES = 8 model aktualizuje wagi dopiero po zebraniu 
    informacji z ośmiu pełnych linii. Gwarantuje to odpowiednią różnorodność stylów 
    pisma niezbędną do stabilizacji wag i nauki długofalowych zależności sekwencyjnych. """
BATCH_SIZE_LINES = 1
ACCUMULATION_STEPS_LINES = 8

# Gwarancja determinizmu
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)

EPOCHS_MAIN = 17
PATIENCE_MAIN = 5
LR_MAIN = 1e-4 # Przy 5e-5 model szybko utknął w lokalnym minimum, a 1e-4 pozwala na stabilny spadek straty osiągając ostatecznie lepszy wynik

EPOCHS_FINE_TUNE = 8
LR_FINE_TUNE = 1e-5

""" W hard mining biblioteka h5py posiada własny, rygorystyczny mechanizm blokad plików i wąskie gardła I/O,
    co spowalnia odczyt lub crashuje skrypt przy większej liczbie wątków.
    Uzasadnienie Differential Learning Rates w fazie Hard Mining:
    1. CNN (1e-5): Chroni bazowe filtry wizualne (krawędzie, pociągnięcia) przed szumem 
       trudnych próbek, zapobiega katastroficznemu zapominaniu ogólnych kształtów pisma.
       Model już powinien się dobrze nauczyć filtrów wizualnych, nie ma sensu ich dalej trenować na ciężkich próbkach.
    2. RNN (5e-5): Pozwala na elastyczną adaptację do nietypowych ligatur i specyficznej 
       dynamiki trudnych autorów, bez destabilizacji fundamentów optycznych.
    3. Output (1e-4): Umożliwia najszybszą korektę granic decyzyjnych dla znaków 
       i interpunkcji, najwyższy LR amortyzuje uderzenie wysokiego Loss na froncie modelu. """
EPOCHS_HARD_MINING = 5
CNN_LR = 1e-5    # Pozwalamy CNN na lekką korektę filtrów dla nowego atramentu
RNN_LR = 5e-5    # RNN musi zaktualizować dynamikę spacji między wyrazami
OUTPUT_LR = 1e-4 # Głowica musi szybko oduczyć się rzucania [blank]


def worker_init_fn(worker_id):
    """ Każdy worker otrzymuje unikalny seed oparty na globalnym seedzie, co zapewnia różnorodność augmentacji i losowego próbkowania,
        jednocześnie zachowując deterministyczność. """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def val_process_fn(image, **kwargs):
    """ Wrapper dla Albumentations z dynamicznym zachowaniem proporcji.
        Zapobiega zniekształcaniu tekstu przy niestandardowych długościach. """
    h_orig, w_orig = image.shape[:2]
    
    # Obliczamy proporcjonalną szerokość względem docelowej wysokości
    aspect_ratio = w_orig / h_orig
    calculated_w = int(IMAGE_HEIGHT * aspect_ratio)
    
    # Nie więcej niż LINE_WIDTH, ale też nie mniej niż WORD_WIDTH
    target_w = min(max(calculated_w, WORD_WIDTH), LINE_WIDTH)
    
    return Preprocessing.process_for_crnn(image, target_h=IMAGE_HEIGHT, target_w=target_w)

VAL_TRANSFORMS = alb.Compose([
    alb.Lambda(name="GeometricNormalization", image=val_process_fn),
    alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
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
        Automatycznie dostosowuje siatkę wyjściową do rozmiarów obrazu wejściowego,
        zapobiegając niszczeniu proporcji krótkich słów. 
        Najpierw Localization Network przewiduje przesunięcia punktów kontrolnych względem symetrycznych pozycji docelowych,
        Potem obliczana jest macierz TPS, a na końcu generowana jest siatka transformacji dla oryginalnych wymiarów obrazu,
        która prostuje tekst. Dzięki temu nawet krótkie słowa zachowują swoje proporcje, a model uczy się rozpoznawać znaki bez zniekształceń. """
    def __init__(self, F=20, loc_size=(32, 64), input_channels=1):
        super(ScRN_STN, self).__init__()
        self.F = F
        self.loc_size = loc_size
        self.nc = input_channels

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

        c_x, c_y = params[:, :, 0], params[:, :, 1]
        s_cos, s_sin = params[:, :, 2], params[:, :, 3]

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

        transformed_img = func.grid_sample(img, grid, align_corners=True, padding_mode='border')

        return transformed_img


class VisualAttention(nn.Module):
    """ Mechanizm wizualnej uwagi pełniący funkcję poolingu. Zastępuje standardowe uśrednianie mechanizmem uczącym.
        Dla każdego kroku czasowego sieć wyznacza wagi ważności, decydując, które fragmenty w pionie zawierają istotne cechy znaku, a które są tłem lub szumem.
        1. Wejście: Mapa cech z ResNet.
        2. Analiza: Warstwa konwolucyjna ocenia ważność każdego piksela.
        3. Filtracja: Softmax normalizuje wagi wzdłuż wymiaru wysokości.
        4. Wyjście: Ważona suma cech – sekwencja 1D gotowa dla GRU. """
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
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
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

        return output, hidden

class WindowedAttention(nn.Module):
    """ Okienkowy mechanizm uwagi wykorzystujący maskę diagonalną. 
        Zamiast sztywno ciąć sekwencję, tworzy płynne, nakładające się okno (Sliding Window), 
        gdzie każdy znak widzi tylko swoje najbliższe sąsiedztwo. """
    def __init__(self, d_model, num_heads=8, window_size=32, adaptive_threshold=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.window_size = window_size

    def forward(self, x):
        t, b, c = x.size()

        # Jeśli sekwencja jest krótka, używamy globalnej uwagi
        if t <= self.window_size:
            attn_out, _ = self.mha(x, x, x)
            return attn_out

        # Tworzymy maskę: True oznacza zablokowaną atencję
        idx = torch.arange(t, device=x.device).unsqueeze(1)
        dist = torch.abs(idx - idx.t())
        attn_mask = dist > self.window_size 

        # PyTorch MHA automatycznie ignoruje relacje tam, gdzie attn_mask jest True
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_mask)

        return attn_out


class SpatialDropout1D(nn.Module):
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
        Architektura składa się z 5 modułów:
        1. ScRN-STN - Geometryczna Rektyfikacja:
           - Adaptacyjna normalizacja obrazu wykorzystująca lekką sieć lokalizującą CNN.
           - Wymusza symetryczne ułożenie punktów kontrolnych względem osi centralnej tekstu,
             co zapobiega deformacji znaków przy prostowaniu falującego pisma.
        2. ResNet-18 Backbone - Inwariantność Kształtu:
           - Rdzeń splotowy ze zmodyfikowanymi krokami (2, 1).
           - Stała wysokość mapy cech pozwala zachować pionową integralność znaków (i, l, f).
        3. CRAM - Wyostrzanie Atramentu:
           - Zintegrowany moduł uwagi przestrzennej i kanałowej. Odszumia tło dokumentu
             i uwydatnia krawędzie liter, zastępując nadmiarowe bloki SE.
        4. Hybrid Adaptive Attention - Kontekstualizacja Wizualna:
           - Visual Attention: Ważenie istotności pionowych kolumn cech.
           - Conditional Windowed Attention: Aktywowany wyłącznie dla długich sekwencji,
             optymalizuje relacje lokalno-globalne bez nadmiarowego obciążenia przy słowach.
        5. BiLSTM - Modelowanie Sekwencji Wizualnej i Ligatur:
           - Dwuwarstwowa sieć rekurencyjna analizująca płynność i dynamikę pisma.
           - Mapuje cechy wizualne na prawdopodobieństwa znaków. """
    def __init__(self, num_classes):
        super().__init__()

        # STN — Geometryczna Rektyfikacja
        self.stn_word = ScRN_STN(
            F=20,
            loc_size=(32, 64),
            input_channels=1
        )

        # STN dla długich linii (CVL/Sztuczne)
        self.stn_line = ScRN_STN(
            F=20,
            loc_size=(32, 256),
            input_channels=1
        )

        # ResNet-18 Backbone
        resnet = models.resnet18(weights='DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modyfikacja Stride (2, 1) - kluczowa dla zachowania T > L przy długich liniach
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            CRAMBlock(512)
        )
        
        # Spatial dropout - wyłączanie całych kanałów cech zamiast pojedynczych wartości
        self.spatial_drop = SpatialDropout1D(p=0.3)

        # Hierarchiczny System Atencji
        self.attention = VisualAttention(512)
        self.self_attn = WindowedAttention(d_model=512, num_heads=8, window_size=64)

        # Uproszczona projekcja (Krótszy backprop do CNN)
        self.p = 0.25
        self.projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Dropout(self.p)
        )
        
        # Po warstwie projection mamy 256 kanałów, mapujemy je bezpośrednio na klasy znaków - CTC Shortcut omijający warstwy RNN
        self.ctc_shortcut = nn.Conv1d(256, num_classes, kernel_size=1)

        # Modelowanie sekwencji
        self.rnn = EnhancedBiLSTM(256, 512, num_layers=2)

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

        # Learnable weight dla contrastive loss
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
        if x.shape[-1] > self.stn_bypass_threshold:
            stn_img = x
        else:
            stn_img = self.stn(x)

        # Ekstrakcja cech (ResNet + CRAM)
        x = self.cnn(stn_img)

        # Visual Attention Pooling - bezpieczniejsza redukcja wymiaru 2
        x = self.attention(x)
        if x.dim() == 4:
            x = torch.mean(x, dim=2)  # Zamiast squeeze(2), uśredniamy wysokość do 1

        # Adaptacyjna atencja okienkowa (dla linii)
        x = x.permute(2, 0, 1)
        if x.size(0) > 128:
            x = self.self_attn(x)

        x = x.permute(1, 2, 0).unsqueeze(2)

        # Upewniamy się, że tensor jest kontynuowalny dla operacji na GPU (traci ciągłość przy zmianie wymiarów)
        x = x.contiguous()

        # Projekcja do BiLSTM
        x = self.projection(x)

        # Przygotowujemy tensor o wymiarach [Batch, Channels, Width]
        if x.dim() == 4:
            x = x.mean(dim=2)

        # Permute(2, 0, 1) zamienia go na [Width, Batch, Channels] - dla RNN
        x = x.permute(2, 0, 1).float()

        with torch.amp.autocast('cuda', enabled=False):
            # Rozpakowujemy rnn_output i zwracamy surowe logity
            recurrent_features, *_ = self.rnn(x)
            log_probs = self.output(recurrent_features)

        # Routing tensorów
        if return_stn and return_context:
            return log_probs, recurrent_features.permute(1, 0, 2), stn_img
        if return_stn:
            return log_probs, stn_img
        if return_context:
            transformer_memory = self.transformer_projection(recurrent_features.permute(1, 0, 2))
            return log_probs, transformer_memory
        if return_embeddings:
            pooled = recurrent_features.mean(dim=0) # Pooling po czasie dla embeddingu całego słowa
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
            if isinstance(module, nn.Dropout2d):
                module.p = p

    def estimate_uncertainty(self, x, steps=10):
        self.eval()
        outputs = []
        for _ in range(steps):
            with torch.no_grad():
                logits = self.forward(x, force_dropout=True)
                outputs.append(torch.softmax(logits[0], dim=-1))

        variance = torch.stack(outputs).var(dim=0)
        return variance.mean(dim=-1)

    @staticmethod
    def get_uncertainty_zones(log_probs, margin_threshold=0.1, conf_threshold=0.5):
        """ Analizuje prawdopodobieństwa i zwraca listę miejsc, które wymagają weryfikacji przez CapsNet. """
        probs = torch.exp(log_probs).squeeze(1)
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)

        margins = top2_probs[:, 0] - top2_probs[:, 1]
        max_conf = top2_probs[:, 0]

        # Niepewność występuje, gdy różnica między 1. a 2. wyborem jest mała lub gdy nawet najlepszy wybór jest słaby
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
    def variable_stroke(image, **kwargs):
        """ Symuluje zmienny nacisk narzędzia piszącego i różną grubość stalówki. """
        # Konwersja na format binarny dla Distance Transform
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Biały tekst, czarne tło
        if np.mean(image) > 127:
            image = cv.bitwise_not(image)

        dist = cv.distanceTransform(image, cv.DIST_L2, 5)
        factor = np.random.uniform(0.8, 1.3)

        # Konwertujemy dist na float32 przed mnożeniem, żeby uniknąć overflow
        dist_float = dist.astype(np.float32)

        # Bezpieczne skalowanie
        max_val = dist_float.max()
        if max_val > 0:
            # Liczymy mnożnik osobno, żeby zachować precyzję
            multiplier = factor * (255.0 / (max_val + 1e-6))
            new_image = np.clip(dist_float * multiplier, 0, 255)
        else:
            new_image = dist_float

        return new_image.astype(np.uint8)

    @staticmethod
    def phantom_elements(image, **kwargs):
        """ Wymusza na sieci wizualnej ignorowanie szumu z sąsiadujących wierszy i skupienie
            uwagi wyłącznie na głównym rdzeniu analizowanego słowa. """
        h, w = image.shape
        phantom_h = int(h * 0.15)
        phantom = np.zeros((phantom_h, w), dtype=np.uint8)
        for _ in range(random.randint(1, 2)):
            x = random.randint(0, max(0, w-30))
            cv.ellipse(phantom, (x, phantom_h), (random.randint(5, 12), 4), random.randint(0, 30), 0, 360, (255, -1))
        if random.random() > 0.5:
            image[:phantom_h, :] = cv.bitwise_or(image[:phantom_h, :], phantom)
        else:
            image[-phantom_h:, :] = cv.bitwise_or(image[-phantom_h:, :], np.flipud(phantom))
        return image


class IAMWordDataset(Dataset):
    """ Moduł ładujący zbiór IAM Words ze wsparciem dla dynamicznego skalowania z zachowaniem proporcji.
        Wszystkie dane (obrazy i etykiety) są dekodowane i wczytywane bezpośrednio do pamięci RAM
        podczas inicjalizacji obiektu, co drastycznie przyspiesza proces treningu kosztem pamięci operacyjnej. """
    def __init__(self, h5_path, transform, char_list, name="Główny", split='train', existing_cache=None, existing_labels=None):
        self.h5_path = h5_path
        self.transform = transform
        self.char_list = char_list
        self.dictionary = char_list
        self.name = name
        self.split = split
        self.IMAGE_HEIGHT = 64

        # Zdefiniowanie pustych list na próbki
        self.valid_indices = []
        self.valid_labels = []
        self.h5_file = None

        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            raw_labels = f[self.split]['labels'][:]

            for i, lbl in enumerate(raw_labels):
                text = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                text = text.strip()
                if text and all(c in self.dictionary for c in text):
                    self.valid_indices.append(i)
                    self.valid_labels.append(text) # <-- POPRAWKA: Zapisujemy do poprawnej listy

        self.dataset_len = len(self.valid_indices)
        tqdm.write(f" Zainicjalizowano {self.name}: {self.dataset_len} próbek.")

    def _decode_and_preprocess(self, idx):
        """ Funkcja ładuje, dekoduje i przetwarza pojedynczą próbkę. """
        try:
            # Pobieranie danych z pliku HDF5
            with h5py.File(self.h5_path, 'r', swmr=True) as f:
                raw_img = f[self.split]['images'][idx]
                raw_label = self.labels[idx]

            # Dekodowanie i wstępna weryfikacja etykiety
            label_text = raw_label.decode('utf-8') if isinstance(raw_label, bytes) else str(raw_label)
            label_text = label_text.strip()

            # Odrzucamy puste etykiety lub te ze znakami spoza zdefiniowanego słownika
            if not label_text or not all(c in self.dictionary for c in label_text):
                return None

            # Dekodowanie obrazu (obsługa formatu skompresowanego lub surowych macierzy)
            if isinstance(raw_img, np.ndarray) and raw_img.ndim == 1:
                img = cv.imdecode(raw_img, cv.IMREAD_GRAYSCALE)
            else:
                img = raw_img

            if img is None or img.size == 0:
                return None

            # Inwersja kolorów
            if np.mean(img) > 127:
                img = cv.bitwise_not(img)

            # Skalowanie z zachowaniem proporcji
            h, w = img.shape[:2]
            scale = self.IMAGE_HEIGHT / max(1, h)
            
            # Obliczamy nową szerokość:
            # $$new\_w = \max(16, \text{int}(w \cdot scale))$$
            new_w = max(16, int(w * scale))

            # Szerokość obrazu musi być proporcjonalna do długości tekstu
            min_w_needed = len(label_text) * 16
            new_w = max(new_w, min_w_needed)

            img = cv.resize(img, (new_w, self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

            # Zwracamy spójną krotkę gotową do użycia w treningu
            return np.ascontiguousarray(img, dtype=np.uint8), label_text

        except (cv.error, UnicodeDecodeError, TypeError, ValueError, KeyError) as e:
            # W przypadku jakiegokolwiek błędu zwracamy None, co pozwoli loaderowi pominąć tę próbkę
            return None
        
    def __getitem__(self, idx):
        h5_idx = self.valid_indices[idx]
        label = self.valid_labels[idx]
        
        # Leniwe otwieranie pliku - tylko raz dla danego procesu
        if getattr(self, 'h5_file', None) is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        # Odczyt bezpośrednio z trzymanego w pamięci otwartego uchwytu
        img_data = self.h5_file[self.split]['images'][h5_idx]

        if isinstance(img_data, np.ndarray) and img_data.ndim == 1:
            img = cv.imdecode(img_data, cv.IMREAD_GRAYSCALE)
        else:
            img = img_data

        if img is None or img.size == 0:
            img = np.zeros((self.IMAGE_HEIGHT, 64), dtype=np.uint8)

        if np.mean(img) > 127: 
            img = cv.bitwise_not(img)

        # Skalowanie
        h, w = img.shape[:2]
        scale = self.IMAGE_HEIGHT / max(1, h)
        new_w = max(16, int(w * scale))
        min_w_needed = len(label) * 16  # Bezpieczeństwo pod CTC
        img = cv.resize(img, (max(new_w, min_w_needed), self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

        augmented = self.transform(image=img)
        category = 'short' if len(label) < 5 else 'medium' if len(label) < 10 else 'long'
        return augmented['image'], label, category, "disk"
        
    def get_text(self, idx):
        """ Zwraca tekst dla danego indeksu """
        # Sprawdzamy, czy to nasz zbiór polski (ma atrybut word_list)
        if hasattr(self, 'word_list'):
            if not self.word_list:
                return "test"
            return self.word_list[idx % len(self.word_list)]
    
        # Dla zbioru IAM (zazwyczaj tekst jest w self.labels lub wyciągany z H5)
        if hasattr(self, 'valid_labels'):
             return self.valid_labels[idx]

        # Jeśli nie wiemy co to za zbiór, zwracamy pusty ciąg, by nie wywalać błędu
        return ""

    def __len__(self):
        return self.dataset_len


class CVLLineDataset(Dataset):
    """ Zestaw danych dla linii tekstu z bazy CVL.
        Wersja Lazy Loading: wczytuje obrazy z dysku/H5 tylko podczas treningu,
        co drastycznie oszczędza RAM. """
    def __init__(self, h5_path, transform, char_list, split='train'):
        self.h5_path = h5_path
        self.transform = transform
        self.char_list = char_list
        self.dictionary = set(char_list)
        self.split = split
        self.IMAGE_HEIGHT = 64
        self.MAX_WIDTH = 2048

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Nie znaleziono pliku H5: {self.h5_path}")

        self.valid_indices = []
        self.valid_labels = []
        self.h5_file = None

        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            if self.split not in f:
                raise KeyError(f"Sekcja '{self.split}' nie istnieje w pliku H5.")

            raw_labels = f[self.split]['labels'][:]

            tqdm.write(f"[{time.strftime('%H:%M:%S')}] Filtrowanie linii CVL {self.split}.")

            for i, lbl in enumerate(raw_labels):
                if isinstance(lbl, bytes):
                    lbl = lbl.decode('utf-8')

                # Filtr języka
                if "-de-" in lbl.lower():
                    continue

                # Czyszczenie i filtr znaków
                lbl_clean = lbl.replace('_', ' ').strip()
                if lbl_clean and all(c in self.dictionary for c in lbl_clean):
                    self.valid_indices.append(i)
                    self.valid_labels.append(lbl_clean)

        self.dataset_len = len(self.valid_labels)
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] Załadowano {self.dataset_len} linii CVL (Lazy Mode).")

    def __getitem__(self, idx):
        h5_idx = self.valid_indices[idx]
        label = self.get_text(idx)
        
        # Leniwe otwieranie pliku - tylko raz dla danego procesu
        if getattr(self, 'h5_file', None) is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        # Odczyt bezpośrednio z trzymanego w pamięci otwartego uchwytu
        img_data = self.h5_file[self.split]['images'][h5_idx]

        if isinstance(img_data, np.ndarray) and img_data.ndim == 1:
            img = cv.imdecode(img_data, cv.IMREAD_GRAYSCALE)
        else:
            img = img_data

        if img is None or img.size == 0:
            img = np.zeros((self.IMAGE_HEIGHT, 64), dtype=np.uint8)

        # Inwersja (jeśli tło jasne)
        if np.mean(img) > 127:
            img = cv.bitwise_not(img)

        # Kadrowanie pustego tła pionowego przed skalowaniem, żeby wypełniło całe tło
        coords = cv.findNonZero(img)
        if coords is not None:
            x, y, w_bbox, h_bbox = cv.boundingRect(coords)
            # Zostawiamy mały margines (np. 5 pikseli)
            y1 = max(0, y - 5)
            y2 = min(img.shape[0], y + h_bbox + 5)
            img = img[y1:y2, :]

        # Skalowanie z zachowaniem proporcji
        h, w = img.shape[:2]
        scale = self.IMAGE_HEIGHT / max(1, h)
        new_w = int(w * scale)

        # Zabezpieczenie CTC (szerokość musi być min. 24x większa od liczby znaków dla linii)
        min_width_for_ctc = len(label) * 24
        new_w = max(new_w, min_width_for_ctc)
        if new_w > self.MAX_WIDTH:
            new_w = self.MAX_WIDTH

        img = cv.resize(img, (new_w, self.IMAGE_HEIGHT), interpolation=cv.INTER_LANCZOS4)

        # Augmentacja (wykonywana w locie na CPU)
        augmented = self.transform(image=img)

        return augmented['image'], label, 'line', 'disk_source'

    def __len__(self):
        return self.dataset_len
        
    def get_text(self, idx):
        return self.valid_labels[idx]


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


class CharLabelEncoder:
    """ Klasa odpowiedzialna za konwersję między znakami tekstowymi a ich indeksami rozszerzona o język polski. """
    def __init__(self):
        polish_chars = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ" # Dodajemy polskie znaki diakrytyczne (18 znaków)
        raw_chars = " !\'(),-.:;?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + polish_chars

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


class HTREncoder:
    """ Słownik i dekoder HTR. Obsługuje CTC Blank, Spacje, Entropię Shannona oraz Beam Search z LM. """
    # Stała interpunkcyjna używana w Visual Veto
    PUNCT_TO_GUARD = ('.', ',', '!', '?', ':', ';', "'", ')')

    def __init__(self, char_list: List[str]):
        self.char_list = char_list 

        self.char_to_num = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        self.num_to_char[0] = ''
        self.char_to_num[''] = 0

    def get_num_classes(self) -> int:
        """ Zwraca liczbę klas wyjściowych (alfabet + blank). """
        return len(self.char_list) + 1

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
    def calculate_uncertainty(logits: torch.Tensor) -> float:
        """ Oblicza średnią entropię Shannona dla sekwencji. Stabilna wersja wykorzystująca log_softmax. """
        # Używamy log_softmax zamiast softmax + log dla stabilności numerycznej
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Entropia Shannona: H(X) = -sum(p * log2(p) na log naturalny
        entropy = -torch.sum(probs * (log_probs / math.log(2)), dim=-1)
        return torch.mean(entropy).item()

    def labels_to_text(self, indices: Union[List[int], np.ndarray]) -> str:
        """ Konwertuje listę indeksów na tekst. """
        res = []
        for idx in indices:
            if idx != 0:  # Ignorujemy CTC Blank
                char_raw = self.num_to_char.get(int(idx), '')
                res.append(self.clean_char(char_raw))
        return "".join(res)

    def decode_greedy(self, log_probs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """ Dekodowanie zachłanne z ekstrakcją entropii. """
        # Upewniamy się, że wymiary to [Batch, Time, Class]
        if log_probs.dim() == 3:
            # Jeśli Time jest większe od Batch, lub jeśli Batch jest równy 1 (co często zdarza się przy testowaniu pojedynczych próbek)
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
                # Logika CTC: pomiń blanki i powtórzenia bez separatora
                if idx != 0 and idx != last_idx:
                    collapsed_indices.append(idx)
                last_idx = idx

            text = self.labels_to_text(collapsed_indices)
            uncertainty = self.calculate_uncertainty(log_probs[i])

            decoded_list.append(text)
            uncertainty_list.append(uncertainty)

        return decoded_list, uncertainty_list

    def decode_beam_search(self, log_probs: torch.Tensor, lm_decoder, beam_width: int = 64, temperature: float = 1.4) -> List[str]:
        """ Zoptymalizowane dekodowanie Beam Search z hybrydowym mechanizmem Visual Veto. Rozwiązuje problem
            halucynacji LM przy zachowaniu polskich znaków diakrytycznych. """
        # Upewniamy się, że format to [Time, Batch, Class] przed wejściem do algorytmu
        if log_probs.dim() == 3:
            # Jeśli Batch jest pierwszy (T > B lub B == 1)
            if log_probs.shape[0] < log_probs.shape[1] or log_probs.shape[0] == 1:
                log_probs = log_probs.permute(1, 0, 2)

        # Skalowanie i kara za blanki
        scaled_log_probs = log_probs.clone() / temperature
        scaled_log_probs[:, :, 0] -= 1.2  

        # Konwersja do prawdopodobieństw i powrót na (Batch, Time, Class) do iteracji
        probs = torch.softmax(scaled_log_probs, dim=-1).permute(1, 0, 2).cpu().numpy()
        decoded_texts = []

        for i in range(probs.shape[0]):
            # Odczyt z Modelu Językowego - sugeruje najbardziej prawdopodobne słowo
            text_lm = lm_decoder.decode(probs[i], beam_width=beam_width).strip()

            # Surowy odczyt wizualny
            raw_indices = np.argmax(probs[i], axis=-1)
            text_raw = self.decode(raw_indices)

            if not text_raw:
                decoded_texts.append(text_lm if text_lm else "")
                continue

            # Mechanizm Visual Veto (jeśli pewny to wierzymy jemu, nie słownikowi)
            dist = edit_distance(text_raw, text_lm)
            norm_dist = dist / len(text_raw)

            # LM usuwa interpunkcję, którą model optyczny widzi wyraźnie
            last_char_raw = text_raw[-1]
            if last_char_raw in ".,!?:;()-\" '/" and not text_lm.endswith(last_char_raw):
                final_word = text_raw

            # LM halucynuje zupełnie inne słowo albo drastycznie skraca bardzo krótkie słowa
            elif norm_dist > 0.45 or (3 >= len(text_raw) > len(text_lm)):
                final_word = text_raw

            # Standardowa poprawka językowa
            else:
                final_word = text_lm if text_lm else text_raw

            decoded_texts.append(final_word)

        return decoded_texts

    def encode_text(self, text_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Zamienia listę tekstów na tensor indeksów i długości (pod CTCLoss). """
        targets = []
        lengths = []
        for text in text_list:
            indices = [self.char_to_num[c] for c in text if c in self.char_to_num]
            targets.extend(indices)
            lengths.append(len(indices))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, idx_seq: Union[List[int], torch.Tensor]) -> str:
        """ Proste dekodowanie pojedynczej sekwencji indeksów. """
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

def get_emnist_char_list_byclass() -> list:
    """ Mapuje indeks ASCII do eMNIST. """
    return [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]


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
                fill = 255
            ),

            # Imitowanie szumu i nieostrości obiektywu
            alb.OneOf([
                alb.GaussNoise(std_range=(0.01, 0.05), p=1.0),
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
                alb.Rotate(limit=5, p=1.0, border_mode=cv.BORDER_CONSTANT, fill=0),
                alb.Perspective(scale=(0.02, 0.04), p=1.0, border_mode=cv.BORDER_CONSTANT, fill=0),
                alb.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0, border_mode=cv.BORDER_CONSTANT, fill=0),
            ], p=0.3),

            # Standaryzacja do eMNIST
            alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
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
                alb.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                alb.GaussianBlur(blur_limit=(3, 3), p=1.0),
            ], p=0.1),

            # Symulacja nierównomiernego oświetlenia skanera
            alb.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),

            # Standaryzacja do eMNIST
            alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
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
    """ Tworzy WeightedRandomSampler wyliczając wagi na podstawie uśrednionej rzadkości znaków w danej sekwencji tekstowej. """
    # Bezpieczne pobieranie tekstów z uwzględnieniem połączonych zbiorów (ConcatDataset)
    all_texts = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            # Pobieramy teksty bezpiecznie
            for i in range(len(ds)):
                text = ds.get_text(i)
                # Jeśli z jakiegoś powodu tekst jest pusty, dajemy znak z alfabetu
                all_texts.append(text if text else alphabet[0])
    else:
        all_texts = [dataset.get_text(i) for i in range(len(dataset))]
        
    char_counts = {char: 0 for char in alphabet}
    for text in all_texts:
        for char in text:
            if char in char_counts:
                char_counts[char] += 1
                
    # Waga znaku: $W_c = \frac{1}{count + 10^{-5}}$
    char_weights = {char: 1.0 / (count + 1e-5) for char, count in char_counts.items()}
    
    sample_weights = []
    for text in all_texts:
        # Waga słowa to średnia waga jego znaków
        weight = sum(char_weights.get(c, 0) for c in text) / max(len(text), 1)
        sample_weights.append(weight)
        
    weights_tensor = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)


def collate_fn_dynamic(batch):
    """ Składa próbki w batch, stosując inteligentny padding i bezpieczne ucinanie. """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None

    # Walidacja batcha
    valid_batch = [item for item in batch if torch.is_tensor(item[0]) and item[0].dim() == 3]
    if not valid_batch: return None

    imgs = [item[0] for item in valid_batch]
    labels = [item[1] for item in valid_batch]
    categories = [item[2] if len(item) > 2 else 'short' for item in valid_batch]
    extras = list(zip(*[item[3:] for item in valid_batch])) if any(len(item) > 3 for item in valid_batch) else []

    is_line_batch = ('line' in categories)
    current_limit = LINE_WIDTH if is_line_batch else WORD_WIDTH

    # Przetwarzamy obrazy pojedynczo
    processed_imgs = []
    for img, label in zip(imgs, labels):
        current_w = int(img.shape[-1])

        # Ustalenie mnożnika szerokości dla bezpieczeństwa CTC
        multiplier = 48 if is_line_batch else 32
        min_w_needed = len(label) * multiplier

        # Bezpieczny limit szerokości
        safe_limit = max(current_limit, min_w_needed)

        # Jeśli obraz przekracza limit, ucinamy
        if current_w > safe_limit:
            img = img[:, :, :safe_limit]
            current_w = safe_limit

        processed_imgs.append(img)

    # Obliczamy max_w dla tego konkretnego batcha
    batch_max_w = max(int(img.shape[-1]) for img in processed_imgs)

    # Normalizacja tła
    bg_val = (0.0 - float(EMNIST_MEAN[0])) / float(EMNIST_STD[0])

    padded_imgs = []
    for img in processed_imgs:
        curr_w = int(img.shape[-1])
        pad_right = int(batch_max_w - curr_w)

        # Padding do prawej krawędzi
        if pad_right > 0:
            padded_img = torch.nn.functional.pad(img, (0, pad_right), value=bg_val)
        else:
            padded_img = img
        padded_imgs.append(padded_img)

    padded = torch.stack(padded_imgs)

    if extras:
        return padded, labels, categories, *[list(field) for field in extras]

    return padded, labels, categories


def label_smoothed_ctc_loss(log_probs, targets, input_lengths, target_lengths, smoothing=0.0, reduction='none'):
    """ CTC Loss z opcjonalnym label smoothing dla lepszej kalibracji modelu. """
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


class CenterLoss(nn.Module):
    """ Center Loss dla lepszej separacji cech między klasami.
        Kluczowe dla pipeline CRNN → CapsNet:
        - Tworzy dobrze rozdzielone clustry dla każdego znaku w przestrzeni cech.
        - CapsNet dostaje lepsze embeddingi do routingu.
        - Redukuje confusion między podobnymi znakami (l/I, 0/O, etc.) """
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Centra klas — uczą się podczas treningu
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)

        # Odległości od centrów
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)

        # Wybieramy odległość od właściwego centrum dla każdego sample
        classes = torch.arange(self.num_classes, device=self.device).long()
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def focal_ctc_loss(log_probs, targets, input_lengths, target_lengths, gamma=2.0, smoothing=0.0, reduction='mean', truncation_threshold=0.01):
    """ Strata fokalna z mechanizmem obcinania (Truncation). Zapobiega skupianiu się modelu na próbkach
    	niemożliwych do odczytania (ze względu na mnogość augmentacji, ciężko nad tym zapanować). """
    device = log_probs.device
    raw_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')(
        log_probs, targets, input_lengths, target_lengths
    )

    norm_loss = raw_loss / target_lengths.to(device).clamp(min=1).float()

    # Prawdopodobieństwo 'p' wyliczone z normalizowanej straty
    p = torch.exp(-norm_loss)
    
    # Jeśli p < truncation_threshold, ograniczamy wpływ tej próbki - mechanizm obcinania
    gamma_tensor = torch.tensor(gamma, device=device)
    
    # Standardowy mnożnik fokalny
    focal_multiplier = torch.pow(1.0 - p, gamma_tensor)
    
    # Jeśli p jest zbyt małe, traktujemy próbkę jako szum i nakładamy stałą, małą wagę
    weight = torch.where(p < truncation_threshold, torch.tensor(0.1, device=device), torch.tensor(1.0, device=device))
    
    focal_loss = weight * focal_multiplier * norm_loss

    if reduction == 'mean':
        return focal_loss.mean()
    return focal_loss


def focal_ace_loss(log_probs, targets, input_lengths, target_lengths, gamma=2.0, reduction='mean'):
    """ Focal ACE Loss - Łączy ACE z focal weighting. """
    device = log_probs.device

    # Obliczamy bazową stratę ACE
    base_loss = ace_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')

    # Normalizacja przez długość (kluczowe, by długie słowa nie dominowały)
    t_lengths = target_lengths.to(device).float().clamp(min=1.0)
    norm_loss = base_loss / t_lengths

    # Prawdopodobieństwo poprawnej predykcji
    p = torch.exp(-base_loss) 

    # Tworzymy stałe jako Tensory
    one = torch.as_tensor(1.0, device=device, dtype=norm_loss.dtype)
    gamma_t = torch.as_tensor(gamma, device=device, dtype=norm_loss.dtype)

    # Obliczamy mnożnik fokalny
    focal_weight = torch.pow(one - p, gamma_t)

    # Nakładamy wagę fokalną na znormalizowaną bazową stratę
    focal_weighted = focal_weight * norm_loss

    # Redukcja
    if reduction == 'mean':
        return focal_weighted.mean()
    elif reduction == 'sum':
        return focal_weighted.sum()
    return focal_weighted


def inter_class_separation_loss(features, labels, margin=0.5):
    """ Inter-Class Separation Loss — wymusza minimalny odstęp między klasami.
        Dla HCR jest lepsze niż triplet loss, bo:
        1. Nie wymaga hardnego mining tripletów
        2. Globalnie odsuwa wszystkie klasy od siebie
        3. Szybsze w treningu. """
    device = features.device
    unique_labels = labels.unique()
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Mapowanie etykiet na zakres [0, n_classes-1]
    label_map = (labels.unsqueeze(1) == unique_labels).float()
    mapped_labels = label_map.argmax(dim=1)
    
    # Obliczanie centrów: Suma cech podzielona przez zliczenia
    counts = torch.bincount(mapped_labels, minlength=n_classes).float().view(-1, 1)
    sum_features = torch.zeros(n_classes, features.size(1), device=device)
    sum_features.scatter_add_(0, mapped_labels.unsqueeze(1).expand(-1, features.size(1)), features)
    centers = sum_features / counts.clamp(min=1)

    # Odległości
    distances = torch.cdist(centers, centers, p=2)

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


def calculate_cer(pred_text, gt_text):
    """ Liczy Character Error Rate dla Hard Miningu. """
    if len(gt_text) == 0: return 1.0
    return edit_distance(pred_text, gt_text) / len(gt_text)


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
            brightness_shift = torch.randn(1).item() * 0.1 * (1.0 / EMNIST_STD[0])
            aug_image = aug_image + brightness_shift

            logits_aug = model(aug_image)
            logits_aug = logits_aug[0] if isinstance(logits_aug, (tuple, list)) else logits_aug
            predictions.append(torch.softmax(logits_aug.float(), dim=-1))

    # Uśrednienie prawdopodobieństw i powrót do log_probs
    avg_probs = torch.stack(predictions).mean(dim=0)
    avg_log_probs = torch.log(avg_probs + 1e-8)

    return avg_log_probs


def evaluate_loss_only(model, loader, device, encoder, smoothing=0.05, use_tta=False):
    """ Szybka ewaluacja modelu z paskiem postępu. Oblicza średnią stratę na całym zbiorze walidacyjnym. """
    model.eval()
    total_loss = 0.0
    batches = 0

    # Wyświetlanie nazwy paska postępu
    desc_str = "Walidacja (TTA)" if use_tta else "Walidacja"
    
    with tqdm(loader, desc=desc_str, leave=False, dynamic_ncols=True, colour='green') as pbar:
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue

                try:
                    images = batch[0].to(device)
                    text_labels = batch[1]
                except RuntimeError as re:
                    # Device Mismatch lub CUDA Out of Memory
                    if "out of memory" in str(re).lower():
                        tqdm.write(f"[{now()}] Krytyczny błąd VRAM: Brak pamięci podczas walidacji. Spróbuj zmniejszyć 'val_batch_size'.")
                    else:
                        tqdm.write(f"[{now()}] Błąd wykonania PyTorch: {re}")
                    continue

                except (IndexError, TypeError, AttributeError) as e:
                    # Błędy struktury danych (np. gdy batch nie jest krotką/listą)
                    tqdm.write(f"[{now()}] Błąd formatu danych w loaderze: {e}. Sprawdź 'collate_fn'.")
                    continue

                except Exception as e:
                    tqdm.write(f"[{now()}] Nieoczekiwany błąd ładowania ({type(e).__name__}): {e}")
                    continue

                targets, target_lengths = encoder.encode_text(text_labels)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                # TTA lub standard prediction
                if use_tta and images.size(0) == 1:
                    log_probs = predict_with_tta(model, images, encoder, num_augmentations=2)
                else:
                    output = model(images)
                    raw_logits = output[0] if isinstance(output, (tuple, list)) else output

                    # Konwersja surowych logitów na log_probs przed CTC Loss
                    log_probs = torch.nn.functional.log_softmax(raw_logits.float(), dim=-1)

                # Jeśli z jakiegoś powodu z TTA wyszedł (Batch, Time, Class), odwracamy to do standardu CTC
                if log_probs.shape[0] == images.size(0) and log_probs.shape[1] > images.size(0):
                    log_probs = log_probs.permute(1, 0, 2)

                T_dim = log_probs.size(0)
                batch_size = images.size(0)

                if target_lengths.max() > T_dim:
                    tqdm.write(f" Cel (len={target_lengths.max()}) jest dłuższy niż mapa cech (len={T_dim}). Pomięcie.")
                    continue

                input_lengths = torch.full(size=(batch_size,), fill_value=T_dim, dtype=torch.long).to(device)

                loss = focal_ctc_loss(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    tqdm.write(" Wykryto NaN lub Inf w walidacji. Pomięcie paczki.")
                    continue

                current_loss = loss.item()
                total_loss += current_loss
                batches += 1

                pbar.set_postfix({'batch_loss': f"{current_loss:.4f}"})

        # Zabezpieczenie przed błędem 0.0000 gdy wszystkie paczki padną
        if batches == 0:
            return float('inf')

    return total_loss / batches


def evaluate_full_metrics(model, loader, device, encoder, decoder=None):
    """ Zaawansowana ewaluacja z TTA i podziałem na długość słów. """
    model.eval()

    stats = {
        'short': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0},
        'medium': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0},
        'long': {'dist': 0, 'chars': 0, 'words': 0, 'err_words': 0}
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Ewaluacja TTA", ncols=100, leave=False, file=sys.stdout):
            if batch is None: continue
            images, text_labels, _, _ = batch
            images = images.to(device)
            B_orig = images.size(0)

            # Test Time Augmentation (3 widoki)
            v1 = images
            v2 = torch.roll(images, shifts=2, dims=3)
            v3 = torch.roll(images, shifts=-2, dims=3)
            tta_batch = torch.cat([v1, v2, v3], dim=0)

            # Forward pass
            output = model(tta_batch)
            raw_logits = output[0] if isinstance(output, (tuple, list)) else output

            # Konwersja na prawdopodobieństwa PRZED uśrednieniem dla poprawnej matematyki TTA
            probs = torch.softmax(raw_logits.float(), dim=-1)

            # Średnia pewność
            T, B3, C = probs.shape
            avg_probs = probs.view(T, 3, B_orig, C).mean(dim=1)
            log_probs = torch.log(avg_probs + 1e-8)

            # Dekodowanie
            if decoder is not None:
                preds = encoder.decode_beam_search(log_probs, decoder)
            else:
                preds, _ = encoder.decode_greedy(log_probs)

            # Akumulacja statystyk
            for gt, pred in zip(text_labels, preds):
                pred_str = str(pred)
                dist = edit_distance(gt, pred_str)
                length = len(gt)
                category = 'short' if length < 5 else ('medium' if length < 9 else 'long')

                stats[category]['dist'] += dist
                stats[category]['chars'] += length
                stats[category]['words'] += 1
                if dist > 0:
                    stats[category]['err_words'] += 1

    tqdm.write(f"{'KATEGORIA':>10} | {'CER [%]':>10} | {'WER [%]':>10} | {'PRÓBEK':>6}")
    for cat, data in stats.items():
        cer = (data['dist'] / max(1, data['chars'])) * 100
        wer = (data['err_words'] / max(1, data['words'])) * 100
        tqdm.write(f"{cat.upper():>10} | {cer:10.2f} | {wer:10.2f} | {data['words']:>6}")

    return stats


def train_one_epoch(model, loader, optimizer, scaler, device, encoder, cat_weights,
                    scheduler=None, use_contrastive=True, ema_model=None, writer=None,
                    epoch=0, blank_penalty=0.05, use_focal=False, is_hard_mining=False, acc_steps=ACCUMULATION_STEPS,
                    label_smoothing=0.0, focal_gamma=2.0, use_ace=False, center_criterion=None,
                    optimizer_center=None, lambda_center=0.0, lambda_separation=0.0):
    """ Wykonuje jedną epokę treningową z uwzględnieniem dynamicznego balansowania klas.
        Logika Straty opiera się na dwóch filarach:
        1. Ważona Strata Fokalna — strata główna.
           Wylicza błąd dla każdego słowa indywidualnie. Błędy w długich słowach,
           które występują rzadziej w zbiorze danych, są dynamicznie skalowane w górę za pomocą
           odwrotności pierwiastka częstości. Zapobiega to ignorowaniu trudnych, długich sekwencji przez sieć.
        2. Contrastive Loss — strata pomocnicza.
           Regularyzuje przestrzeń cech z warstw rekurencyjnych (BiGRU). Zbliża do siebie
           wielowymiarowe reprezentacje takich samych słów (ucząc model ignorować styl pisma odręcznego)
           oraz oddala od siebie słowa o różnych znakach. """
    model.train()

    # flatten_parameters wystarczy raz przed epoką
    model.rnn.flatten_parameters()

    total_loss = 0.0

    step_lrs_list = []
    step_moms_list = []

    # Zbieramy trudne próbki do przetworzenia po epoce
    hard_samples = []

    if cat_weights is None:
        cat_weights = {'short': 1.0, 'medium': 1.5, 'long': 2.5, 'line': 5.0}

    #Pobieramy liczbę jako czysty int
    total_batches = int(len(loader))

    with tqdm(loader, desc="Uczenie", leave=True, file=sys.stdout, dynamic_ncols=True) as loop:
        for i, batch in enumerate(loop):
            if batch is None: continue

            # Początkowa aktualizacja paska postępu o liczbę zebranych trudnych próbek i średnią stratę
            loop.set_postfix({
                'Hard': len(hard_samples),
                'Loss': total_loss / max(1, i)
            })

            # Zapisywanie aktualnego Learning Rate i Momentum
            step_lrs_list.append(optimizer.param_groups[0]['lr'])
            current_mom = optimizer.param_groups[0].get('momentum', optimizer.param_groups[0].get('betas', (0.9, 0.999))[0])
            step_moms_list.append(current_mom)

            # Rozpakowanie danych
            if not isinstance(batch, (list, tuple)):
                images, text_labels, categories = batch, [], []
            else:
                images = batch[0]
                text_labels = batch[1] if len(batch) > 1 else []
                categories = batch[2] if len(batch) > 2 else ['short'] * len(text_labels)

            valid_batch_data = [(img, lbl, cat) for img, lbl, cat in zip(images, text_labels, categories) if
                                len(lbl.strip()) > 0]
            if not valid_batch_data: continue

            images = torch.stack([x[0] for x in valid_batch_data]).to(device)
            text_labels = [x[1] for x in valid_batch_data]

            # Kodowanie etykiet
            targets_list = []
            target_lengths_list = []
            for t in text_labels:
                encoded = [encoder.char_to_num[c] for c in t if c in encoder.char_to_num]
                targets_list.extend(encoded)
                target_lengths_list.append(len(encoded))

            targets = torch.tensor(targets_list, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths_list, dtype=torch.long).to(device)

            # Zabezpieczenie przed pustymi batchami po wyrzuceniu łatwych przykładów
            if images.size(0) == 0:
                continue

            # Forward pass
            with torch.amp.autocast('cuda', enabled=(scaler is not None), dtype=torch.bfloat16):
                output = model(images, return_embeddings=use_contrastive)
            
            # Bezpieczne rozpakowanie
            if use_contrastive:
                if isinstance(output, (tuple, list)) and len(output) == 3:
                    preds_half, shortcut_preds, embeddings = output
                else:
                    raise ValueError(f"Oczekiwano 3 wartości w trybie contrastive, otrzymano {len(output)}. Sprawdź konfigurację modelu.")
            else:
                preds_half, shortcut_preds = output
                embeddings = None

            # Float32 dla ochrony przed underflow, zapobiega eksplozji gradientów
            preds_fp32 = preds_half.float()

            # Bezpieczne nałożenie kary
            if blank_penalty > 0:
                # Modyfikujemy istniejący kanał 'blank'
                preds_fp32[:, :, 0] -= blank_penalty

            # log_softmax zawsze w float32 dla stabilności numerycznej
            log_preds = torch.nn.functional.log_softmax(preds_fp32, dim=-1)
            T_dim = log_preds.size(0)
            batch_size_current = images.size(0)
        
            # Zbieranie Hard Samples
            if is_hard_mining:
                with torch.no_grad():
                    # Zabezpieczenie przed błędami numerycznymi w predykcjach
                    if not torch.isnan(log_preds).any():
                        # Dekodujemy cały batch na tekst
                        decoded, uncertainties = encoder.decode_greedy(log_preds.detach())
                        
                        # Zamiana nan na 0.0 w wynikach niepewności
                        uncertainties = torch.nan_to_num(torch.tensor(uncertainties), nan=0.0).tolist()
                        
                        # Próbki spełniające kryteria trudności
                        batch_scores = []
                        
                        # Wyliczenie progu trudności na podstawie średniej CER z dotychczasowych batchy, aby zapewnić stabilność i adaptacyjność procesu hard miningu
                        dynamic_threshold = 0.25 
                        
                        for idx in range(batch_size_current):
                            if torch.isnan(log_preds[:, idx, :]).any():
                                continue

                            pred_text = decoded[idx]
                            true_text = text_labels[idx]
                            cer = calculate_cer(pred_text, true_text)
                            
                            # odrzucamy próbki nieczytelne, idealne lub całkowicie błędne
                            if (cer > 0.8 and uncertainties[idx] > 0.9) or cer == 0.0 or cer >= 1.0:
                                continue

                            # Pobieramy kategorię dla danej próbki
                            sample_category = categories[idx] if idx < len(categories) else 'short'
                            
                            # Definiujemy wskaźnik "trudności" (uwzględnia CER i niepewność modelu)
                            hardness_score = cer + (uncertainties[idx] * 0.5)

                            # Zbieramy tylko próbki, które są trudniejsze niż obecny próg modelu
                            if hardness_score > dynamic_threshold:
                                batch_scores.append({
                                    'idx': idx,
                                    'score': hardness_score,
                                    'cer': cer,
                                    'uncertainty': uncertainties[idx],
                                    'true_text': true_text,
                                    'category': sample_category
                                })

                        # Logika S-OHEM: Wybieramy najtrudniejsze próbki z każdej kategorii
                        if batch_size_current > 1:
                            loss_strata = {}
                            for item in batch_scores:
                                c = item['category']
                                if c not in loss_strata:
                                    loss_strata[c] = []
                                loss_strata[c].append(item)

                            top_hard_samples = []
                            ratio = 0.25

                            for c, items in loss_strata.items():
                                items.sort(key=lambda x: x['score'], reverse=True)
                                num_hard_in_stratum = max(int(len(items) * ratio), 1)
                                top_hard_samples.extend(items[:num_hard_in_stratum])
                        else:
                            top_hard_samples = batch_scores

                        # Dodajemy wyselekcjonowane próbki do globalnej pamięci
                        for item in top_hard_samples:
                            hard_samples.append({
                                'image': images[item['idx']].detach().cpu().to(torch.uint8), # Przenosimy do RAM jako uint8, żeby oszczędzić miejsce
                                'label': item['true_text'],
                                'cer': item['cer'],
                                'score': item['score'] 
                            })

                        # Logika S-OHEM: Wybieramy najtrudniejsze próbki z każdej kategorii, aby zapewnić różnorodność trudnych przykładów
                        if batch_size_current > 1:
                            # S-OHEM: Grupowanie próbek po kategoriach (warstwach) - działa dla słów
                            loss = {}
                            for item in batch_scores:
                                c = item['category']
                                if c not in loss:
                                    loss[c] = []
                                loss[c].append(item)

                            top_hard_samples = []
                            ratio = 0.25

                            # Dynamiczny próg: wybieramy 25% najtrudniejszych próbek Z KAŻDEJ warstwy osobno
                            for c, items in loss.items():
                                # Sortujemy w obrębie danej kategorii
                                items.sort(key=lambda x: x['score'], reverse=True)
                                
                                # Obliczamy limit dla tej konkretnej kategorii
                                num_hard_in_stratum = max(int(len(items) * ratio), 1)
                                
                                # Doklejamy najgorsze przykłady z tej kategorii do wspólnej puli
                                top_hard_samples.extend(items[:num_hard_in_stratum])
                        else:
                            # Gdy Batch = 1 (Linie), po prostu przekazujemy próbkę do rygorystycznej oceny progowej
                            top_hard_samples = batch_scores

                        # Dodajemy wyselekcjonowane, zróżnicowane próbki do pamięci do fazy hybrydowej
                        for item in top_hard_samples:
                            # Próg minimalny (żeby nie zbierać literówek, gdy model już się poduczy)
                            if (item['cer'] > 0.10 or item['uncertainty'] > 0.3) and item['cer'] < 1.0:
                                hard_samples.append({
                                    'image': images[item['idx']].detach().cpu().to(torch.uint8), # Przenosimy do RAM jako uint8, żeby oszczędzić miejsce
                                    'label': item['true_text'],
                                    'cer': item['cer'],
                                    'score': item['score'] 
                                })

                        SAFE_RAM_LIMIT = 15000 
                        if len(hard_samples) > SAFE_RAM_LIMIT:
                            # Jeśli zebrano za dużo próbek, układamy je od najtrudniejszych do najprostszych i odrzucamy nadmiar
                            hard_samples.sort(key=lambda x: x.get('score', x['cer']), reverse=True)
                            hard_samples = hard_samples[:SAFE_RAM_LIMIT]

            # Obliczanie strat (CTC/ACE)
            input_lengths = torch.full((batch_size_current,), T_dim, dtype=torch.long, device=device)

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
                        gamma=focal_gamma, smoothing=label_smoothing, reduction='none'
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
                loss_aux = torch.norm(embeddings + 1e-8, p=2, dim=1).mean()
                alpha = torch.sigmoid(model.contrastive_alpha) * 0.4 + 0.5
                loss = (alpha * loss_main) + ((1.0 - alpha) * loss_aux)

            # Wspólne generowanie cech i etykiet
            char_labels = None
            if (center_criterion is not None and lambda_center > 0) or (lambda_separation > 0 and use_contrastive):
                pooled_features = embeddings if use_contrastive else preds_fp32.mean(dim=0)
                
                # Szybkie tworzenie tensorów etykiet (bez pętli for na CPU)
                if target_lengths.size(0) > 0:
                    # Obliczamy offsety dla każdego słowa w batchu
                    offsets = torch.cat([torch.tensor([0], device=device), target_lengths.cumsum(dim=0)[:-1]])
                    
                    # Pobieramy pierwszy znak każdego słowa
                    valid_mask = target_lengths > 0
                    if valid_mask.any():
                        char_labels = targets[offsets[valid_mask]]

            """ CENTER LOSS (Kompaktowość wewnątrzklasowa):
                Mechanizm ten wymusza, aby reprezentacje obrazów należących do tej samej klasy skupiały się jak
                najbliżej wspólnego centroidu. Zmniejsza to wariancję wewnątrzklasową,
                co jest kluczowe przy różnorodnych stylach pisma. """
            if center_criterion is not None and lambda_center > 0 and char_labels is not None:
                # Jeśli rozmiar batcha cech nie pasuje do przefiltrowanych etykiet znaków, dopasowujemy je maską
                if pooled_features.size(0) == char_labels.size(0):
                    center_loss_value = center_criterion(pooled_features, char_labels)
                    loss += lambda_center * center_loss_value

                    if writer and (i % 100 == 0):
                        writer.add_scalar('Loss/Center', center_loss_value.item(), epoch * total_batches + i)

            """ INTER-CLASS SEPARATION LOSS (Rozróżnialność międzyklasowa):
                Podczas gdy Center Loss przyciąga próbki do siebie, Separation Loss działa jak 'odpychanie' 
                różnych klas. Ustawia on minimalny margines bezpieczeństwa między klastrami różnych znaków.
                Zapobiega to sytuacji, w której wizualnie podobne litery (np. 'o' oraz 'a') nakładają się na siebie 
                w przestrzeni cech, co drastycznie ułatwia pracę algorytmom klasyfikującym (jak CapsNet).
                Czyli najpierw skupiamy reprezentacje blisko siebie, a następnie oddzielamy takie skupiska
                od siebie, żeby wybór klasy był łatwiejszym zadaniem. """
            if lambda_separation > 0 and use_contrastive and char_labels is not None:
                if embeddings.size(0) == char_labels.size(0):
                    separation_loss = inter_class_separation_loss(embeddings, char_labels, margin=0.5)
                    loss += lambda_separation * separation_loss

                    if writer and (i % 100 == 0):
                        writer.add_scalar('Loss/Separation', separation_loss.item(), epoch * total_batches + i)

            # Pomijamy nieskonczoność
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                if center_criterion is not None and optimizer_center is not None:
                    optimizer_center.zero_grad(set_to_none=True)

                loop.set_postfix({'Loss': "NaN w Aux!", 'Hard': len(hard_samples)})
                continue

            # Wsteczna propagacja
            loss_to_step = loss / acc_steps
            loss_to_step.backward()

            # Optymalizacja
            if (i + 1) % int(acc_steps) == 0 or (i + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            loop.set_postfix({'Loss': f"{loss.item() * int(acc_steps):.3f}", 'Hard': len(hard_samples)})

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

    if is_hard_mining and len(hard_samples) > 0:
        # Sortujemy całą zebraną poczekalnię od najtrudniejszych do najprostszych
        hard_samples.sort(key=lambda x: x['score'], reverse=True)
        
        # Obliczamy ile to jest dokładnie 40% zebranych błędów
        keep_count = max(1, int(len(hard_samples) * 0.40))
        
        # Brutalnie ucinamy listę do absolutnej czołówki
        hard_samples = hard_samples[:keep_count]
        
        tqdm.write(f"Zakończono epokę. OHEM wyselekcjonował {keep_count} najtrudniejszych linii (Top 40%).")

    return total_loss / total_batches, hard_samples, step_lrs_list, step_moms_list


def execute_hybrid_ohem_phase(model, optimizer, scaler, encoder, device, hard_samples, iam_loader=None, polish_loader=None):
    """ Kompletna faza OHEM:
        1. Trenuje na najtrudniejszych liniach.
        2. Zapobiega zapominaniu poprzez szybki update na słowach (IAM + Polskie znaki). """
    if not hard_samples:
        return

    # Sprawdzenie wag (blokada przed NaN)
    if any(torch.isnan(p).any() for p in model.parameters() if p is not None):
        tqdm.write(f" KRYTYCZNY BŁĄD: Wykryto NaN w wagach modelu przed OHEM!")
        return

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    model.train()
    
    bg_val = (0.0 - float(EMNIST_MEAN[0])) / float(EMNIST_STD[0])

    hard_samples.sort(key=lambda x: x['cer'], reverse=True)
    line_batch_pool = hard_samples

    optimizer.zero_grad(set_to_none=True) 
    num_steps = max(1, math.ceil(len(line_batch_pool) / SAFE_BATCH_SIZE))
    accumulated_grads = False

    # Trening na najtrudniejszych liniach (CVL)
    for i in range(0, len(line_batch_pool), SAFE_BATCH_SIZE):
        chunk = line_batch_pool[i : i + SAFE_BATCH_SIZE]
        raw_imgs = [x['image'] for x in chunk]
        lbls = [x['label'] for x in chunk]

        processed_imgs = []
        for img, label in zip(raw_imgs, lbls):
            min_w = len(label) * 32 
            curr_w = img.shape[2] 
            if curr_w < min_w:
                img = torch.nn.functional.pad(img, (0, min_w - curr_w), value=bg_val)
            processed_imgs.append(img)

        max_w = max(img.shape[2] for img in processed_imgs)
        padded = torch.stack([
            torch.nn.functional.pad(img, (0, max_w - img.shape[2]), value=bg_val)
            for img in processed_imgs
        ]).to(device)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            output = model(padded)
            preds = output[0] if isinstance(output, tuple) else output
            preds = torch.clamp(preds, min=-35, max=35)
            lp = torch.nn.functional.log_softmax(preds.float(), dim=-1)
            
            if torch.isnan(lp).any(): continue

            t, tl = encoder.encode_text(lbls)
            il = torch.full((padded.size(0),), lp.size(0), dtype=torch.long, device=device)
            l_ohem = criterion(lp, t.to(device), il, tl) / num_steps

        if torch.isfinite(l_ohem):
            l_ohem.backward() 
            accumulated_grads = True

    if accumulated_grads:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tqdm.write(f" OHEM: Linie przetworzone stabilnie.")

    # Faza Retencyjna (IAM + Polskie Słowa)
    retention_imgs = []
    retention_lbls = []

    # Pobieramy 4 słowa z IAM
    if iam_loader:
        try:
            iam_batch = next(iter(iam_loader))
            retention_imgs.append(iam_batch[0][:4])
            retention_lbls.extend(iam_batch[1][:4])
        except Exception: pass

    # Pobieramy 4 słowa Polskie (z naciskiem na znaki diakrytyczne)
    if polish_loader:
        try:
            pol_batch = next(iter(polish_loader))
            retention_imgs.append(pol_batch[0][:4])
            retention_lbls.extend(pol_batch[1][:4])
        except Exception: pass

    # Łączymy w jeden batch i trenujemy
    if retention_imgs:
        try:
            # Znajdujemy maksymalną szerokość wśród wszystkich zebranych tensorów retencyjnych
            max_ret_w = max(t.shape[3] for t in retention_imgs) 
            
            padded_retention = []
            for t in retention_imgs:
                # Obliczamy ile brakuje do najszerszego elementu w tym batchu
                diff = max_ret_w - t.shape[3]
                if diff > 0:
		    # Padding tła do prawej krawędzi wartością bg_val
                    t_padded = torch.nn.functional.pad(t, (0, diff), value=bg_val)
                    padded_retention.append(t_padded)
                else:
                    padded_retention.append(t)
            
            # Teraz wszystkie mają ten sam kształt [4, 1, 64, max_ret_w]
            imgs_word = torch.cat(padded_retention, dim=0).to(device)

            lbls_word = retention_lbls

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                output_w = model(imgs_word)
                preds_w = output_w[0] if isinstance(output_w, tuple) else output_w
                preds_w = torch.clamp(preds_w, min=-35, max=35)
                
                lp_w = torch.nn.functional.log_softmax(preds_w.float(), dim=-1)
                
                if torch.isnan(lp_w).any(): raise ValueError("NaN w fazie retencyjnej")

                t_w, tl_w = encoder.encode_text(lbls_word)
                il_w = torch.full((imgs_word.size(0),), lp_w.size(0), dtype=torch.long, device=device)
                l_forget = criterion(lp_w, t_w.to(device), il_w, tl_w)

            if torch.isfinite(l_forget):
                l_forget.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(" OHEM: Faza retencyjna (PL + EN) zakończona.")
        except Exception as e:
            tqdm.write(f" OHEM Retency Warning: {e}")


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
        attn_layer = model.attention
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
    """ Zapis wycinku litery z optymalizacją pod CapsNet i stabilnym pozycjonowaniem. (Funkcja wyciągnięta z zakresu lokalnego). """
    if crop_img is None or crop_img.size == 0:
        return

    safe_label = get_safe_char_name(label_char)
    current_count = class_counts.get(safe_label, 0)

    # Inteligentne przycinanie marginesów
    h_orig, w_orig = crop_img.shape[:2]
    margin = int(h_orig * 0.15) if h_orig > 20 else 0
    work_img = crop_img[margin:h_orig - margin, :].copy() if margin > 0 else crop_img.copy()

    if work_img.size == 0: return

    # Jawne rzutowanie na float
    if float(np.mean(work_img)) > 127:
        work_img = cv.bitwise_not(work_img)

    # .get() zamienia UMat na ndarray
    img_array = work_img.get() if isinstance(work_img, cv.UMat) else work_img

    if img_array.mean() > 127:
        work_img = cv.bitwise_not(work_img)

    # Progowanie Sauvola
    work_img_np = work_img.get() if isinstance(work_img, cv.UMat) else work_img
    work_img_np = cv.GaussianBlur(work_img_np, (3, 3), 0)

    # Sauvola zwraca tablicę progów lokalnych
    thresh_map = threshold_sauvola(work_img_np, window_size=15, k=0.2)

    # Tworzymy maskę binarną przez porównanie tablic
    binary_mask = np.zeros(work_img_np.shape, dtype=np.uint8)
    binary_mask[work_img_np < thresh_map] = 255

    cnts, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return

    small_chars = {'.', ',', '-', "'", '"', ':', ';', '_', '`'}
    min_area = 6 if label_char in small_chars else 25
    valid_cnts = [c for c in cnts if cv.contourArea(c) > min_area]
    if not valid_cnts: return

    if isinstance(work_img, cv.UMat):
        img_np = work_img.get()
    else:
        img_np = work_img

    center_x = img_np.shape[1] // 2

    def get_c_info(c):
        M = cv.moments(c)
        if M["m00"] == 0: return 0, 0
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Szukamy konturu, którego środek ciężkości jest najbliżej środka okna
    main_cnt = min(valid_cnts, key=lambda c: abs(get_c_info(c)[0] - center_x))
    mx, my, mw, mh = cv.boundingRect(main_cnt)
    mcx, mcy = get_c_info(main_cnt)

    clean_mask = np.zeros_like(binary_mask)
    cv.drawContours(
        image=clean_mask,
        contours=[main_cnt],
        contourIdx=-1,
        color=(255,),
        thickness=-1
    )

    # Dodawanie kropek/akcentów z bezpiecznikiem pionowym
    collision_margin = max(mw // 1.5, 12)
    for c in valid_cnts:
        if id(c) == id(main_cnt): continue

        cx, cy = get_c_info(c)
        if abs(cx - mcx) <= collision_margin and cy < (my + mh + 5):
            cv.drawContours(clean_mask, [c], -1, (255,), -1)

    # Kadrowanie z zachowaniem proporcji (Padding)
    x, y, w, h = cv.boundingRect(clean_mask)
    p = 2  # Mniejszy padding
    char_roi = clean_mask[max(0, y - p):y + h + p, max(0, x - p):x + w + p]

    # Skalowanie do Canvas 64x64
    canvas_size = 64
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    target_dim = 44 if label_char not in small_chars else 20
    scale = target_dim / max(h, w, 1)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv.resize(char_roi, (nw, nh), interpolation=cv.INTER_AREA)

    # Centrowanie na środku canvasu
    y_off = (canvas_size - nh) // 2
    x_off = (canvas_size - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    # Zapis i manifest
    target_dir = os.path.join(output_root, safe_label, category)
    os.makedirs(target_dir, exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    filename_base = f"{safe_label}_{unique_id}"

    img_path = os.path.join(target_dir, filename_base + ".png")
    npy_path = os.path.join(target_dir, filename_base + ".npy")

    cv.imwrite(img_path, canvas)

    # Konwersja tensorów
    prob_array = full_probs_vec.detach().float().cpu().numpy() if torch.is_tensor(full_probs_vec) else np.array(full_probs_vec)
    ctx_array = context_vec.detach().float().cpu().numpy() if torch.is_tensor(context_vec) else np.array(context_vec)

    np.savez(npy_path,
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


def export_error_crops_for_capsnet(model, loader, device, encoder, output_root):
    """ Eksport wycinków dla CapsNet z podwójną weryfikacją: normalna predykcja + przesunięcie geometryczne. """
    if os.path.exists(output_root): shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    model.eval()
    export_manifest = []
    class_counts = {}
    STRIDE = 8  # Downsampling ResNet
    WIN_PURE = 32  # Rozmiar okna dla pewnych znaków
    WIN_HARD = 26  # Rozmiar okna dla błędów (nie chcemy szumu)

    SMALL_SYMBOLS = {'.', ',', "'", '`', '-', ':', ';', '"'}

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eksport Deep Fusion", ncols=100, position=0, leave=True)
        for batch in pbar:
            if batch is None: continue
            images, text_labels, _, paths = batch
            images = images.to(device)

            # Model zwraca 3 elementy (log_probs, b_ctx, stn_imgs) przy tych flagach
            lp1, b_ctx, stn_imgs = model(images, return_stn=True, return_context=True)
            
            # Model zwraca 2 elementy (log_probs, stn_imgs) przy przesunięciu
            lp3, _ = model(torch.roll(images, shifts=2, dims=3), return_stn=True)

            p1, idx1, m1, c1 = get_preds(lp1)
            _, idx3, _, _ = get_preds(lp3)

            for b in range(images.size(0)):
                if not torch.isfinite(lp1[b]).all():
                    continue
                img_stn = ((stn_imgs[b].cpu().numpy().squeeze() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
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
                            t_idx = p_info['t']
                            char_gt = gt_text[idx_g]

                            T_max = b_ctx.size(1)
                            c_start = max(0, t_idx - 1)
                            c_end = min(T_max, t_idx + 2)
                            raw_window = b_ctx[b, c_start:c_end, :]
                            windowed_ctx = torch.mean(raw_window, dim=0)

                            pred_shift = encoder.num_to_char.get(idx3[b][t_idx], '')

                            is_wrong = p_info['char'] != char_gt
                            is_unstable = (pred_shift != char_gt)

                            cx = t_idx * STRIDE
                            win = WIN_HARD if (is_wrong or is_unstable) else WIN_PURE

                            x1 = max(0, int(cx - win))
                            x2 = min(img_stn.shape[1], int(cx + win))
                            crop = img_stn[:, x1:x2]

                            prob_val = p1[b][t_idx]
                            
                            # Wywołania z zewnętrznym przekazaniem manifestu (bez safe_ep)
                            if is_wrong or is_unstable:
                                save_crop_with_context(crop, windowed_ctx, char_gt, "hard", prob_val, paths[b], output_root, export_manifest, class_counts, p_info['char'])
                            elif p_info['margin'] < 0.20:
                                save_crop_with_context(crop, b_ctx[b][t_idx], char_gt, "unsure", prob_val, paths[b], output_root, export_manifest, class_counts, p_info['char'])
                            elif random.random() < (UPPER_RATE if char_gt.isupper() else PURE_RATE):
                                save_crop_with_context(crop, b_ctx[b][t_idx], char_gt, "pure", prob_val, paths[b], output_root, export_manifest, class_counts)

                    elif tag == 'insert':
                        t_p = peaks[i1 - 1]['t'] if i1 > 0 else 0
                        t_n = peaks[i1]['t'] if i1 < len(peaks) else p1[b].shape[0] - 1

                        for sub_idx, idx_g in enumerate(range(j1, j2)):
                            char_gt = gt_text[idx_g]
                            t_est = int(t_p + (sub_idx + 1) * (t_n - t_p) / (j2 - j1 + 1))
                            cx = t_est * STRIDE

                            check_area = img_stn[:, max(0, int(cx - 15)):min(img_stn.shape[1], int(cx + 15))]

                            if check_area.size == 0:
                                continue
                                
                            mean_val = float(np.mean(check_area))

                            if mean_val > 248 or mean_val < 7:
                                best_offset, max_ink = 0, 0
                                for offset in range(-20, 21, 4):
                                    new_cx = cx + offset
                                    x1 = max(0, int(new_cx - 8))
                                    x2 = min(img_stn.shape[1], int(new_cx + 8))
                                    test_area = img_stn[:, x1:x2]

                                    if test_area.size == 0:
                                        continue

                                    current_ink = int(np.sum(test_area < 128))
                                    if current_ink > max_ink:
                                        max_ink, best_offset = current_ink, offset

                                cx += best_offset

                            crop = img_stn[:, max(0, int(cx - WIN_HARD)):min(img_stn.shape[1], int(cx + WIN_HARD))]
                            min_ink_thresh = 8 if char_gt in SMALL_SYMBOLS else 40

                            if crop.size > 0 and np.sum(crop < 128) > min_ink_thresh:
                                prob_val_missed = p1[b][t_est] if torch.is_tensor(p1[b][t_est]) else p1[b][t_est]

                                save_crop_with_context(crop, b_ctx[b][t_est], char_gt, "missed", prob_val_missed, paths[b], output_root, export_manifest, class_counts)
                
                if len(export_manifest) % 500 == 0:
                    torch.cuda.empty_cache()

    with open(os.path.join(output_root, "export_manifest.json"), "w") as f:
        json.dump(export_manifest, f, indent=4)

    tqdm.write(f"Eksport zakończony. Wygenerowano {len(export_manifest)} próbek.")


class PolishCharStitcher:
    """ Tworzy autentycznie wyglądające polskie słowa i pełne linie. """
    def __init__(self, npz_path=OUTPUT_NPZ):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data['signs']
        self.labels = data['labels']

        # Grupowanie obrazków po znakach dla szybkiego dostępu
        self.char_map = {}
        for i, char in enumerate(self.labels):
            if char not in self.char_map:
                self.char_map[char] = []
            self.char_map[char].append(self.images[i])

    def generate_word_image(self, text: str, target_height=64):
        word_imgs = []
        for char in text:
            # ROZWIĄZANIE PROBLEMU POMIJANIA: Obsługa spacji
            if char == " ":
                space_w = random.randint(20, 35) # Losowa szerokość naturalnej spacji
                word_imgs.append(np.zeros((target_height, space_w), dtype=np.uint8))
                continue

            samples = self.char_map.get(char)
            if not samples:
                # Jeśli znaku nie ma, dajemy mały odstęp, by nie przerywać linii
                word_imgs.append(np.zeros((target_height, 10), dtype=np.uint8))
                continue

            char_img = random.choice(samples)
            h, w = char_img.shape
            
            # Augmentacja glifu
            M = cv.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), random.uniform(0.95, 1.05))
            char_img = cv.warpAffine(char_img, M, (w, h), borderValue=0)
            word_imgs.append(char_img)

        if not word_imgs: return np.zeros((target_height, target_height), dtype=np.uint8)

        # Obliczanie szerokości z marginesem bezpieczeństwa
        combined_width = sum([img.shape[1] for img in word_imgs]) + 40
        final_img = np.zeros((target_height, combined_width), dtype=np.uint8)

        current_x = 15
        prev_anchor = None

        for img in word_imgs:
            h, w = img.shape
            if h != target_height:
                w = int(w * (target_height / h))
                img = cv.resize(img, (w, target_height))
                h = target_height

            y_offset = random.randint(-2, 2) # Lekkie drganie pionowe pisma
            y_pos = max(0, min(target_height - h, (target_height - h) // 2 + y_offset))

            # Rysowanie ligatury (tylko dla liter, pomijamy spacje o szerokości > 15)
            if prev_anchor and w > 15:
                current_anchor = (current_x + w // 5, y_pos + h // 2)
                cv.line(final_img, prev_anchor, current_anchor, random.randint(80, 160), thickness=1)

            region = final_img[y_pos:y_pos + h, current_x:current_x + w]
            final_img[y_pos:y_pos + h, current_x:current_x + w] = np.maximum(region, img)

            # Kerning: zacieśniamy litery (overlap), ale nie spacje
            overlap = int(w * random.uniform(0.15, 0.25)) if w > 35 else 0
            current_x += (w - overlap)
            prev_anchor = (current_x, y_pos + h // 2)

        final_img = final_img[:, :current_x + 10]

        # Post-processing dla realizmu
        kernel = np.ones((2, 2), np.uint8)
        final_img = cv.dilate(final_img, kernel, iterations=1)

        if random.random() > 0.4:
            final_img = self._apply_elastic_distortion(final_img)

        final_img = cv.GaussianBlur(final_img, (3, 3), 0)
        noise = np.random.randint(0, 10, final_img.shape, dtype=np.uint8)
        return cv.add(final_img, noise)

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


class IAMSyntheticLineDataset(Dataset):
    """ Generuje w locie syntetyczne angielskie linie tekstu, sklejając losowe słowa z IAM.
        Tworzy pomost edukacyjny między pojedynczymi słowami a rzeczywistymi liniami CVL. """
    def __init__(self, h5_path, transform, char_list, split='train', num_samples=15000):
        self.h5_path = h5_path
        self.transform = transform
        self.dictionary = set(char_list)
        self.split = split
        self.num_samples = num_samples
        self.IMAGE_HEIGHT = 64
        self.MAX_WIDTH = 2048

        self.valid_indices = []
        self.valid_labels = []
        self.h5_file = None

        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            raw_labels = f[self.split]['labels'][:]
            for i, lbl in enumerate(raw_labels):
                text = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                text = text.strip()
                if text and all(c in self.dictionary for c in text):
                    self.valid_indices.append(i)
                    self.valid_labels.append(text)

        self.dataset_len = len(self.valid_indices)
        tqdm.write(f" Zainicjalizowano IAM Syntetyczne Linie: Baza {self.dataset_len} słów.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if getattr(self, 'h5_file', None) is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        # Losujemy od 4 do 8 słów, aby stworzyć pełną linię
        num_words = random.randint(4, 8)
        word_imgs = []
        word_texts = []

        for _ in range(num_words):
            rand_idx = random.randint(0, self.dataset_len - 1)
            h5_idx = self.valid_indices[rand_idx]
            label = self.valid_labels[rand_idx]

            img_data = self.h5_file[self.split]['images'][h5_idx]
            if isinstance(img_data, np.ndarray) and img_data.ndim == 1:
                img = cv.imdecode(img_data, cv.IMREAD_GRAYSCALE)
            else:
                img = img_data

            if img is None or img.size == 0: continue
            
            # Wymuszamy czarne tło dla łatwego sklejania i dopełniania
            if np.mean(img) > 127: img = cv.bitwise_not(img)

            # Wyrównanie wysokości poszczególnych słów przed sklejeniem
            h, w = img.shape[:2]
            scale = self.IMAGE_HEIGHT / max(1, h)
            new_w = max(16, int(w * scale))
            img = cv.resize(img, (new_w, self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

            word_imgs.append(img)
            word_texts.append(label)

        if not word_imgs:
            return self.__getitem__((idx + 1) % self.num_samples)

        # Inicjalizacja linii pierwszym słowem
        line_img = word_imgs[0]
        line_text = word_texts[0]

        # Iteracyjne doklejanie kolejnych słów z losowymi odstępami
        for i in range(1, len(word_imgs)):
            space_width = random.randint(15, 45) # Naturalna, zmienna szerokość spacji
            space_img = np.zeros((self.IMAGE_HEIGHT, space_width), dtype=np.uint8)

            # Hstack (horyzontalne łączenie macierzy numpy)
            line_img = np.hstack((line_img, space_img, word_imgs[i]))
            line_text += " " + word_texts[i]

        # Zabezpieczenie szerokości dla zoptymalizowanego batchowania
        if line_img.shape[1] > self.MAX_WIDTH:
            line_img = cv.resize(line_img, (self.MAX_WIDTH, self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

        # Aplikacja augmentacji (już na całej wygenerowanej linii)
        if self.transform:
            augmented = self.transform(image=line_img)
            line_img = augmented['image']

        return line_img, line_text, "line", "synthetic_iam_line"

    def get_text(self, idx):
        return "Syntetyczna linia angielska z IAM"


class PolishSyntheticDataset(Dataset):
    """ Generuje w locie pojedyncze polskie słowa. 
        Zmieniono z LineDataset na słowa, aby odciążyć pamięć VRAM. """
    def __init__(self, npz_path, word_list, transform=None, num_samples=5000):
        self.npz_path = npz_path
        # Filtracja: bierzemy tylko słowa o długości min. 2 znaków
        self.word_list = [w for w in word_list if len(w) > 1]
        self.transform = transform
        self.num_samples = num_samples
        self.stitcher = None 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Leniwa inicjalizacja Stitchera
        if self.stitcher is None:
            self.stitcher = PolishCharStitcher(self.npz_path)

        # Losowanie słowa z listy
        word = random.choice(self.word_list)
        
        # Generowanie obrazu tylko dla jednego słowa
        img = self.stitcher.generate_word_image(word)

        if img is None:
            return self.__getitem__((idx + 1) % self.num_samples)

        # Dostosowanie do szerokości modelu
        h, w = img.shape
        if w > WORD_WIDTH:
            img = cv.resize(img, (WORD_WIDTH, 64), interpolation=cv.INTER_AREA)

        if self.transform:
            img = self.transform(image=img)['image']

        if len(word) < 5:
            category = 'short'
        elif len(word) < 10:
            category = 'medium'
        else:
            category = 'long'
        
        return img, word, category, "synthetic_polish_word"

    def get_text(self, idx):
        return self.word_list[idx % len(self.word_list)]


def load_sjp_dictionary(file_path, alphabet_set, num_desired=30000):
    """ Funkcja ładująca polskie słowa. """
    polish_diacritics = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
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
        word = word.lower() # Sprowadzamy wszystko do małych liter dla spójności
        # Sprawdzamy długość oraz czy WSZYSTKIE znaki są w naszym alfabecie modelu
        if 3 < len(word) < 14 and all(c in alphabet_set for c in word):
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

    # Balansowanie zbioru (Chcemy ok. 50% słów z ogonkami w celu poprawnego trenowania CRNN)
    count_diac = int(num_desired * 0.5)
    actual_diac = min(len(with_diacritics), count_diac)
    count_std = num_desired - actual_diac

    final_list = with_diacritics[:actual_diac] + standard[:count_std]
    random.shuffle(final_list)

    tqdm.write(f"[{now()}] Wybrano {len(final_list)} słów (Ogonki: {actual_diac}, Bez: {len(final_list) - actual_diac}).")
    return final_list


def get_full_htr_char_list():
    """ Zwraca stałą listę znaków zsynchronizowaną z Twoim mapowaniem. """
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


def compute_char_confusion_data(model, dataloader, device, encoder):
    """ 
    Analizuje pomyłki wizualne modelu na całym zbiorze danych.
    Zwraca spłaszczone listy znaków GT i Pred do macierzy pomyłek.
    """
    model.eval()
    char_true = []
    char_pred = []

    with torch.no_grad():
        # dynamic_ncols=True i leave=False zapewniają czysty terminal w Dockerze
        pbar = tqdm(dataloader, desc="Analiza pomyłek", ncols=100, leave=False, file=sys.stdout, dynamic_ncols=True)
        
        for batch in pbar:
            if batch is None:
                continue

            images, text_labels, _, _ = batch
            images = images.to(device)

            # Bezpieczne rozpakowanie
            output = model(images)
            if isinstance(output, (tuple, list)):
                log_probs = output[0]  # Logity są zawsze pierwsze
            else:
                log_probs = output

            try:
                preds_str_list, _ = encoder.decode_greedy(log_probs)
            except Exception as e:
                tqdm.write(f"Błąd dekodowania: {e}")
                continue

            # Mapowanie znaków
            for gt, pred in zip(text_labels, preds_str_list):

                aligned_gt, aligned_pred = align_prediction_to_ground_truth(gt, pred)
                
                # Dodajemy tylko jeśli faktycznie dostaliśmy dane
                if aligned_gt and aligned_pred:
                    char_true.extend([str(c) for c in aligned_gt])
                    char_pred.extend([str(c) for c in aligned_pred])

    tqdm.write(f"[{time.strftime('%H:%M:%S')}] Analiza zakończona. Zebrano {len(char_true)} par znaków.")
    return char_true, char_pred


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
       - Średni CER (Character Error Rate): {cer:.2f}%
       - Średni WER (Word Error Rate):      {wer:.2f}%
       - Liczba przeanalizowanych słów:    {total_words}

    2. DIAGNOSTYKA MECHANIZMU NIEPEWNOŚCI:
       - Skuteczność flagowania (Recall):    {recall:.2f}%
       - Precyzja flagowania (Precision):   {precision:.2f}%
       - True Positives (Wykryte błędy):    {true_positives}
       - False Negatives (Błędy przeoczone): {false_negatives}

    3. ANALIZA NAJCZĘSTSZYCH POMYŁEK WIZUALNYCH:
    """
    for i, fail in enumerate(worst_fails):
        report += f"   {i + 1}. GT: '{fail['gt']}' -> Pred: '{fail['pred']}' (Dystans: {fail['dist']})\n"

    report += f"""
    4. WNIOSKI:
       Model poprawnie oflagował {recall:.2f}% wszystkich błędów. Te {true_positives} przypadków 
       zostało pomyślnie wyeksportowanych jako materiał treningowy dla sieci CapsNet. 
       Wysoka skuteczność flagowania (Recall) gwarantuje, że większość błędów wizualnych 
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


if __name__ == "__main__":
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

    seed_everything(3407, deterministic=True)

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

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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
                # Faza Fine-tune: jeśli plik flagi istnieje, wymuszamy licznik na 18
                if os.path.exists(FINE_COMPLETE_FILE) and last_total_epoch < 18:
                    last_total_epoch = 18

            if history.get('val_loss'):
                best_val_loss = min(history['val_loss'])

            tqdm.write(f"[{now()}] Historia wczytana. Startujemy od epoki: {last_total_epoch + 1}")
        except json.JSONDecodeError:
            tqdm.write(f"[{now()}] BŁĄD: Plik historii jest uszkodzony (niepoprawny JSON). Start od zera.")
        except PermissionError:
            tqdm.write(f"[{now()}] BŁĄD: Brak uprawnień do odczytu pliku {history_path}. Sprawdź uprawnienia Windows.")
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

    random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))

    # Val transforms - spójna normalizacja
    val_transform_crnn = alb.Compose([
        alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
        ToTensorV2()
    ])

    # Inicjalizacja zbiorów
    train_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, transform=get_augmentations("main"),
                               char_list=char_list, name="IAM_Train", split='train')

    train_polish = PolishSyntheticDataset(npz_path=OUTPUT_NPZ, word_list=polish_words_list,
                                      transform=get_augmentations("main"), num_samples=12000)

    # Łączymy w jeden hybrydowy zbiór
    train_dataset = ConcatDataset([train_iam, train_polish])

    val_dataset = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, transform=val_transform_crnn,
                                 char_list=char_list, name="Walidacyjny", split='val')

    # Dynamiczne ważenie klas - definiujemy kategorie na podstawie długości słowa i obecności polskich znaków diakrytycznych
    all_categories = []
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZĄĆĘŁŃÓŚŹŻąćęłńóśźż.,!?:;()-\" '/"
    
    # Zbiór polskich znaków do filtrowania kategorii
    polish_diacritics = set("ĄĆĘŁŃÓŚŹŻąćęłńóśźż")

    for ds in train_dataset.datasets:
        # Sprawdzamy, czy to nasz syntetyczny dataset
        if isinstance(ds, PolishSyntheticDataset):
            # Dodajemy kategorię tyle razy, ile wynosi num_samples
            all_categories.extend(["polish_diacritic"] * ds.num_samples)
        else:
            # Dla IAM Words iterujemy po faktycznych etykietach w RAM
            for lbl in ds.valid_labels:
                if any(c in polish_diacritics for c in lbl):
                    all_categories.append("polish_diacritic")
                else:
                    length = len(lbl)
                    cat = 'short' if length < 5 else ('medium' if length < 9 else 'long')
                    all_categories.append(cat)

    # Obliczanie wag dynamicznych
    category_counts = Counter(all_categories)
    max_count = max(category_counts.values())

    """ Definiujemy wagi - polish_diacritic dostaje boost, żeby model zwracał szczególną uwagę na te znaki
        (w większości różnią się tylko ogonkiem lub kreską, więc są trudniejsze do rozróżnienia) """
    dynamic_weights_map = {
        "short": math.sqrt(max_count / max(category_counts["short"], 1)),
        "medium": math.sqrt(max_count / max(category_counts["medium"], 1)),
        "long": math.sqrt(max_count / max(category_counts["long"], 1)),
        "polish_diacritic": math.sqrt(max_count / max(category_counts["polish_diacritic"], 1)) * 1.5  # Boost dla polskich znaków
    }

    tqdm.write(f"[{now()}] Liczebność klas: {dict(category_counts)}")
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in dynamic_weights_map.items()])
    tqdm.write(f"[{now()}] Wagi Balansujące: {weights_str}")

    weights = [dynamic_weights_map[cat] for cat in all_categories]
    sampler = create_balanced_sampler(train_dataset, alphabet)

    # Loader używa hybrydowego samplera łączącego polskie znaki już wcześniej
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_WORDS,
        sampler=sampler,
        shuffle=False,
        num_workers=WORKERS_MAIN,
        collate_fn=collate_fn_dynamic,
        prefetch_factor=2, # Każdy worker trzyma 2 batche w gotowości
        pin_memory=True, # Przyspiesza trening na GPU
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_WORDS,
        shuffle=False,
        num_workers=WORKERS_MAIN,
        collate_fn=collate_fn_dynamic,
        pin_memory=False,
        persistent_workers=(WORKERS_MAIN > 0),
        worker_init_fn=seed_worker,
        generator=g
    )

    # Spróbuje NAdam, który często dobrze radzi sobie z CRNN-ami, ale z mniejszym LR i wagami L2 dla stabilności
    optimizer = optim.NAdam(model.parameters(), lr=LR_MAIN, weight_decay=1e-4)
    acc_steps = ACCUMULATION_STEPS

    # Scheduler (OneCycle)
    steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_MAIN,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_MAIN,
        pct_start=PCT_START,
        div_factor=DIV_FACTOR,
        final_div_factor=100
    )

    # Przesuwamy scheduler, jeśli wznawiamy trening w fazie MAIN
    if 0 < last_total_epoch < EPOCHS_MAIN:
        tqdm.write(f"[{now()}] Przesuwanie schedulera epok o {last_total_epoch}.")
        for _ in range(last_total_epoch * steps_per_epoch):
            # OneCycleLR wymaga i+1 kroku lub wywołania .step() co partię (batch)
            scheduler.step()

    # Main training
    if not os.path.exists(MAIN_PHASE_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Rozpoczynam Fazę Main (Agresywna augmentacja).")
        get_augmentations('main')

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name or 'bn' in name or 'instance_norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

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

        steps_per_epoch = math.ceil(int(len(train_loader)) / int(acc_steps))
        main_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR_MAIN,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS_MAIN, pct_start=PCT_START, div_factor=DIV_FACTOR, final_div_factor=100
        )

        start_ep = last_total_epoch if last_total_epoch < EPOCHS_MAIN else 0
        if start_ep > 0:
            for _ in range(steps_per_epoch):
                main_scheduler.step()

        patience_counter = 0
        for epoch in range(start_ep, EPOCHS_MAIN):
            tqdm.write(f"[{now()}] Epoka {epoch + 1} Main")

            # Wykonujemy trening epoki
            current_gamma = get_focal_gamma_schedule(epoch, EPOCHS_MAIN, start_gamma=2.5, end_gamma=1.5)
            
            t_loss, _, e_lrs, e_moms = train_one_epoch(
                model, train_loader, optimizer, scaler, DEVICE, encoder, 
                scheduler=main_scheduler, cat_weights=dynamic_weights_map, 
                epoch=epoch, focal_gamma=current_gamma, # Przekazujemy dynamicznie parametr skupienia
                writer=writer
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

    # Fine-tune (Focal CTC Loss, więc na początku znacznie wyższy loss, bo inaczej liczony)
    if os.path.exists(MAIN_PHASE_COMPLETE_FILE) and not os.path.exists(FINE_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Rozpoczynam Fazę Fine-tune (Optymalizacja wag używająca SWA do znalezienia płaskiego, "
                   f"bezpiecznego minimum straty, aby uniknąć overfittingu).")
        get_augmentations("fine_tune")

        swa_model = AveragedModel(model).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_FOLDER, "WordLevelResNetCRNN.pth"))['model_state'])

        synth_iam_lines = IAMSyntheticLineDataset(
            h5_path=IAM_WORDS_H5_PATH, 
            transform=get_augmentations("fine_tune"), 
            char_list=char_list, 
            split='train',
            num_samples=8000
        )

        fine_tune_dataset = ConcatDataset([train_iam, train_polish, synth_iam_lines])

        train_loader = DataLoader(
            fine_tune_dataset, batch_size=BATCH_SIZE_WORDS, num_workers=WORKERS_FINE,
            collate_fn=collate_fn_dynamic, pin_memory=False, worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE_WORDS,
            shuffle=False,
            num_workers= WORKERS_FINE,
            collate_fn=collate_fn_dynamic,
            pin_memory=False
        )

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            # Nie stosujemy weight decay dla biasów i BatchNorm
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        """ Zmiana optymalizatora dla fazy Fine-tune:
            Zastąpienie adaptacyjnego optymalizatora (NAdam) klasycznym SGD z pędem.
            Uzasadnienie: SWA osiąga optymalne rezultaty, gdy optymalizator "krąży" na brzegach szerokiego
            i płaskiego minimum. Adaptacyjne algorytmy mają tendencję do zbyt szybkiego zbiegania w ostre
            minima lokalne. SGD pozwala na stabilną eksplorację Płaskich regionów, co po uśrednieniu wag
            drastycznie poprawia generalizację modelu. Opcja nesterov=True dodaje przewidywanie kierunku gradientu,
            zapobiegając nadmiernym oscylacjom."""
        optimizer = optim.SGD([
            {'params': decay_params, 'weight_decay': 1e-4}, # Niższy decay dla fine-tune
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=LR_FINE_TUNE, momentum=0.9, nesterov=True)
        total_fine_steps = math.ceil(EPOCHS_FINE_TUNE * len(train_loader) / acc_steps)
        fine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_fine_steps, eta_min=1e-8)

        """ Nie wyliczamy bazowego błędu na starcie, ponieważ SWA często wywołuje początkowy skok straty
            szukając płaskiego minimum. Sztywny próg zablokowałby zapis wczesnych epok. """
        best_fine_loss = float('inf')

        for epoch in range(EPOCHS_FINE_TUNE):
            current_total_epoch = EPOCHS_MAIN + epoch + 1
            tqdm.write(f"[{now()}] Epoka {current_total_epoch} Fine-tune")
            model.set_dropout(0.15)

            # Gamma scheduling dla fine-tune (niższe wartości)
            current_gamma = get_focal_gamma_schedule(epoch, EPOCHS_FINE_TUNE, start_gamma=2.0, end_gamma=1.2)

            train_loss, current_hard_pool, e_lrs, e_moms = train_one_epoch(
                model, train_loader, optimizer, scaler, DEVICE, encoder, # type: ignore
                cat_weights=dynamic_weights_map, scheduler=fine_scheduler, is_hard_mining=False,
                ema_model=ema_model, epoch=current_total_epoch, blank_penalty=0.05,
                label_smoothing=0.02, focal_gamma=current_gamma,
                use_focal=True, acc_steps=ACCUMULATION_STEPS,
                writer=writer
            )

            # Zapisujemy zebrane dane do historii wykresu
            history.setdefault('step_lrs', []).extend(e_lrs)
            history.setdefault('step_moms', []).extend(e_moms)

            swa_model.update_parameters(model)

            # Walidujemy standardowy model w trakcie trwania pętli
            val_loss = evaluate_loss_only(model, val_loader, DEVICE, encoder)
            tqdm.write(f"          Fine Loss: {val_loss:.4f}")

            with open(history_path, "w") as f:
                json.dump(history, f)

            if val_loss < best_fine_loss:
                best_fine_loss = val_loss
                torch.save({'model_state': model.state_dict(), 'epoch': current_total_epoch}, CER_PATH)
                tqdm.write("              └─> Nowy rekord! Zapisano checkpoint modelu.")

        # Aktualizujemy statystyki BN dla SWA i to ten model staje się naszym najlepszym
        tqdm.write(f"[{now()}] Kalibracja BatchNorm dla modelu SWA.")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

        swa_final_loss = evaluate_loss_only(swa_model, val_loader, DEVICE, encoder)
        tqdm.write(f"[{now()}] Ostateczny loss SWA: {swa_final_loss:.4f}")

        # Nadpisujemy cer_path ostatecznym, uśrednionym modelem
        torch.save({
            'model_state': swa_model.module.state_dict(),
            'best_loss': swa_final_loss,
            'epoch': EPOCHS_MAIN + EPOCHS_FINE_TUNE
        }, CER_PATH)

        with open(FINE_COMPLETE_FILE, 'w') as f:
            f.write(f"Zakończono: {time.ctime()}")

        # val_dataset jest później używany
        del train_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Hard-Mining
    hard_mining_history = []
    if os.path.exists(FINE_COMPLETE_FILE) and not os.path.exists(HARD_MINING_COMPLETE_FILE):
        tqdm.write(f"[{now()}] Rozpoczynam Fazę Hard Mining (Skupienie na najtrudniejszych próbkach, które model ma problemy z rozpoznaniem).")
        if os.path.exists("hard_mining_history.json"):
            with open("hard_mining_history.json", "r") as f:
                try:
                    hard_mining_history = json.load(f)
                except:
                    hard_mining_history = []

        # Gradient checkpointing - zamiast zapamiętywać wszystkiego, model robi checkpointy, co jest mniej obciążające dla pamięci
        if hasattr(model.cnn, 'gradient_checkpointing_enable'):
            model.cnn.gradient_checkpointing_enable()
            tqdm.write(f"[{now()}] Gradient Checkpointing włączony dla modułu ResNet.")
        elif hasattr(model.cnn, 'model') and hasattr(model.cnn.model, 'set_grad_checkpointing'):
            # Czasami trzeba wejść głębiej
            model.cnn.model.set_grad_checkpointing(True)
            tqdm.write(f"[{now()}] Gradient Checkpointing włączony dla CNN.")

        # Inicjalizacja transformacji przed pętlą
        line_transform = alb.Compose([
            alb.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            alb.GaussianBlur(blur_limit=(3, 3), p=0.3),
            alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
            ToTensorV2()
        ])

        # Loader treningowy
        cvl_train_dataset = CVLLineDataset(CVL_H5_PATH, line_transform, char_list, split='train')
        cvl_train_loader = DataLoader(
            cvl_train_dataset,
            batch_size=BATCH_SIZE_LINES,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=worker_init_fn
        )

        # Loader walidacyjny dla CVL
        cvl_val_dataset = CVLLineDataset(CVL_H5_PATH, line_transform, char_list, split='val')
        cvl_val_loader = DataLoader(
            cvl_val_dataset,
            batch_size=BATCH_SIZE_LINES,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=WORKERS_HARD_MINING
        )

        # Loader walidacyjny dla IAM (Strażnik Generalizacji)
        iam_val_dataset = CVLLineDataset(IAM_LINES_H5_PATH, line_transform, char_list, split='val')
        iam_val_loader = DataLoader(
            iam_val_dataset,
            batch_size=BATCH_SIZE_LINES,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=WORKERS_HARD_MINING
        )

        # Mix walidacyjny: oba datasety dla bezpieczeństwa
        iam_lines_val_dataset = CVLLineDataset(IAM_LINES_H5_PATH, line_transform, char_list, split='val')
        val_dataset_mix = ConcatDataset([cvl_val_dataset, iam_lines_val_dataset])
        val_loader_mix = DataLoader(
            val_dataset_mix,
            batch_size=BATCH_SIZE_LINES,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=WORKERS_HARD_MINING
        )

        mining_params = []
        
        for module_name, module, lr_val in [
            ('cnn', model.cnn, CNN_LR), 
            ('rnn', model.rnn, RNN_LR), 
            ('output', model.output, OUTPUT_LR)
        ]:
            decay_group = []
            no_decay_group = []
            for name, param in module.named_parameters():
                if not param.requires_grad: continue
                if "bias" in name or "bn" in name or "norm" in name:
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)
            
            mining_params.append({'params': decay_group, 'lr': lr_val, 'weight_decay': 0.01})
            mining_params.append({'params': no_decay_group, 'lr': lr_val, 'weight_decay': 0.0})

        """ Zmiana optymalizatora dla fazy Hard Mining:
            Zastąpienie NAdam optymalizatorem AdamW z zachowawczym epsilonem.
            Uzasadnienie: Faza ta eksponuje model na najtrudniejsze i najsilniej zniekształcone próbki, 
            które generują wysokie wartości błędu. NAdam, wykorzystujący agresywny pęd Nesterova, mógłby 
            zbyt gwałtownie zaktualizować wagi i zniszczyć wyuczone wcześniej filtry wizualne (tzw. 
            katastroficzne zapominanie). AdamW rozdziela regularyzację od kroku adaptacyjnego, 
            zapewniając chłodną i precyzyjną korektę wag BiLSTM, szanując przy tym zróżnicowane stopy 
            uczenia poszczególnych warstw. """
        mining_optimizer = optim.AdamW(mining_params, eps=1e-8)

        mining_scheduler = optim.lr_scheduler.CosineAnnealingLR(mining_optimizer, T_max=5)

        iam_word_dataset = IAMWordDataset(
            h5_path=IAM_WORDS_H5_PATH,
            transform=get_augmentations("fine_tune"),
            char_list=char_list,
            name="IAM_HardMining",
            split='train',
            existing_cache=None,
            existing_labels=None
        )

        iam_loader = DataLoader(
            iam_word_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0, 
            collate_fn=collate_fn_dynamic,
            pin_memory=False
        )

        polish_retention_loader = DataLoader(
            train_polish, 
            batch_size=8, 
            shuffle=True, # Losowe polskie znaki z generatora
            num_workers=0,
            collate_fn=collate_fn_dynamic, 
            pin_memory=False
        )

        # Inicjalizacja
        best_combined_loss = float('inf')

        for epoch in range(EPOCHS_HARD_MINING):
            tqdm.write(f"[{now()}] Hard Mining Epoka {epoch + 1}/{EPOCHS_HARD_MINING}")

            # Zamrażamy BN w ResNecie — szlifujemy tylko wagi, nie chcemy psuć statystyk batcha
            model.train()
            for m in model.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            # Trening z OHEM
            train_loss, current_hard_pool, _, _ = train_one_epoch(
                model,
                cvl_train_loader,
                mining_optimizer,
                scaler,
                DEVICE,
                encoder,
                cat_weights=dynamic_weights_map,
                scheduler=mining_scheduler,
                epoch=epoch,
                is_hard_mining=True,
                use_focal=False,
                acc_steps=ACCUMULATION_STEPS_LINES,
                blank_penalty=0.4,
                label_smoothing=0.02,  # Małe wygładzanie dla stabilności
            )

            if current_hard_pool and len(current_hard_pool) > 0:
                execute_hybrid_ohem_phase(
                    model=model,
                    optimizer=mining_optimizer,
                    scaler=scaler,
                    encoder=encoder,
                    device=DEVICE,
                    hard_samples=current_hard_pool,
                    iam_loader=iam_loader,
                    polish_loader=polish_retention_loader
                )

            mining_scheduler.step()

            # Po fazie optymalizacji miningu przywracamy śledzenie statystyk przed ewaluacją
            for m in model.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            # Pełna Ewaluacja
            cvl_loss = evaluate_loss_only(model, cvl_val_loader, DEVICE, encoder)
            iam_loss = evaluate_loss_only(model, iam_val_loader, DEVICE, encoder)

            avg_loss = (cvl_loss + iam_loss) / 2.0
            avg_cer = 0.0  # Placeholder dla statystyk epoki

            # Zapis do historii
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'cvl': {'loss': cvl_loss, 'cer': 0.0, 'wer': 0.0},
                'iam': {'loss': iam_loss, 'cer': 0.0, 'wer': 0.0},
                'avg_loss': avg_loss,
                'avg_cer': avg_cer,
                'hard_samples_found': len(current_hard_pool)
            }
            hard_mining_history.append(epoch_stats)

            # Logowanie do TensorBoard
            if writer:
                writer.add_scalar("HardMining/CombinedLoss", avg_loss, epoch)
                writer.flush()

            # Zapisywanie najlepszego modelu
            if avg_loss < best_combined_loss:
                best_combined_loss = avg_loss

                checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f"crnn_checkpoint_e{epoch + 1}_loss{avg_loss:.4f}.pth")
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch + 1,
                    'history': hard_mining_history,
                    'metrics': {'cer': avg_cer, 'loss': avg_loss}
                }, checkpoint_path)
                save_status = f" Zapisano (Loss: {avg_loss:.4f})"
            else:
                save_status = " Brak zapisu (Wzrost błędu)"

            tqdm.write(f"      CVL - Loss: {cvl_loss:.4f}")
            tqdm.write(f"      IAM - Loss: {iam_loss:.4f}")
            tqdm.write(f"       └─> {save_status}")

            # Monitorowanie przeuczenia
            if 'initial_line_loss' not in globals():
                globals()['initial_line_loss'] = cvl_loss
                globals()['initial_word_loss'] = iam_loss
                globals()['best_efficiency_score'] = 0.0

            init_line = globals()['initial_line_loss']
            init_word = globals()['initial_word_loss']

            # Obliczamy deltę (ile poprawiliśmy/pogorszyliśmy od startu Hard Miningu)
            line_improvement = init_line - cvl_loss
            word_degradation = iam_loss - init_word

            # Obliczamy wskaźnik efektywności
            if word_degradation <= 0:
                efficiency_score = float('inf')
            else:
                efficiency_score = line_improvement / word_degradation

            if epoch != 0:
                tqdm.write(f" Zysk: {line_improvement:.2f} | Strata: {word_degradation:.2f} | Efektywność: {efficiency_score:.2f}")

            # Jeśli strata jakości na IAM, a zysk na CVL jest znikomy przez 2 epoki — zatrzymujemy trening
            if epoch > 2:
                if word_degradation > 0 and efficiency_score < 2.0:
                    tqdm.write(
                        f"[{time.strftime('%H:%M:%S')}] Niska efektywność nauki. Dalszy trening psuje model ogólny.")
                    break

                # Jeśli błąd na IAM wzrośnie o więcej niż 2 odchylenia standardowe to model przeucza się pod CVL
                if word_degradation > (init_word * 0.15):
                    tqdm.write(f"[{time.strftime('%H:%M:%S')}] Krytyczna utrata generalizacji (IAM Loss > 15%).")
                    break

    # Wyłączamy po zakończeniu fazy hard-mining
    if hasattr(model.cnn, 'gradient_checkpointing_disable'):
        model.cnn.gradient_checkpointing_disable()
        tqdm.write(f"[{now()}] Gradient Checkpointing WYŁĄCZONY.")

    # Finalny zapis historii
    save_path = os.path.join(CHECKPOINT_FOLDER, "hard_mining_history.json")
    with open(save_path, "w") as f:
        json.dump(hard_mining_history, f, indent=4)

    # Znacznik zakończenia fazy Hard Mining
    with open(HARD_MINING_COMPLETE_FILE, 'w') as f:
        f.write("complete")

    # Eksport dla CapsNet i raport
    if os.path.exists(FINE_COMPLETE_FILE):
        # Wzorzec wyszukiwania
        hm_pattern = os.path.join(CHECKPOINT_FOLDER, "crnn_checkpoint_e*_loss*.pth")
        hm_files = glob.glob(hm_pattern)
        
        best_hm_model = None
        if hm_files:
            try:
                # Sortujemy rosnąco: im mniejszy loss, tym lepszy model
                hm_files.sort(key=lambda x: float(re.findall(r"loss(\d+\.\d+)", x)[0]))
                best_hm_model = hm_files[0]
                tqdm.write(f"[{now()}] Wybrano najlepszy model Hard Mining: {os.path.basename(best_hm_model)}")
            except Exception:
                # Najnowszy plik, jeśli nazwa nie pasuje do wzorca
                hm_files.sort(key=os.path.getmtime, reverse=True)
                best_hm_model = hm_files[0]

        # Definiujemy ścieżkę do załadowania
        if best_hm_model and os.path.exists(best_hm_model):
            final_model_path = best_hm_model
            tqdm.write(f"[{now()}] SUKCES: Eksport na podstawie najlepszego modelu Hard Mining: {os.path.basename(final_model_path)}")
        elif os.path.exists(FINAL_MODEL_LINES_PATH):
            final_model_path = FINAL_MODEL_LINES_PATH
            tqdm.write(f"[{now()}] Eksport na podstawie modelu HCR na poziomie linii (CVL + IAM).")
        else:
            final_model_path = CER_PATH
            tqdm.write(f"[{now()}] UWAGA: Brak wag z Hard Miningu. Fallback do najlepszego modelu bazowego (SWA/IAM).")

        # Ładowanie wag do modelu
        model.load_weights(final_model_path)
        model.eval()

        # Łączymy obie bazy linii do ostatecznego raportu i ekstrakcji
        val_line_transform = alb.Compose([
            alb.Normalize(mean=EMNIST_MEAN, std=EMNIST_STD),
            ToTensorV2()
        ])

        cvl_val_dataset = CVLLineDataset(CVL_H5_PATH, val_line_transform, char_list, split='val')
        iam_lines_val_dataset = CVLLineDataset(IAM_LINES_H5_PATH, val_line_transform, char_list, split='val')
        full_line_export_dataset = ConcatDataset([cvl_val_dataset, iam_lines_val_dataset])


        export_loader = DataLoader(
            full_line_export_dataset,
            batch_size=BATCH_SIZE_LINES,
            shuffle=False,
            collate_fn=collate_fn_dynamic,
            num_workers=WORKERS_HARD_MINING
        )

        tqdm.write(f"[{now()}] Analiza pomyłek i eksport wycinków dla CapsNet.")
        char_true, char_pred = compute_char_confusion_data(model, export_loader, DEVICE, encoder)
        export_error_crops_for_capsnet(model, export_loader, DEVICE, encoder, CAPSNET_DATA_DIR)

        # Zapis analizy błędów
        with open(os.path.join(CAPSNET_DATA_DIR, "crnn_error_analysis.json"), "w") as f:
            json.dump({'char_true': char_true, 'char_pred': char_pred}, f)

        # Generowanie macierzy pomyłek
        flat_true = [str(c) for t in char_true for c in t]
        flat_pred = []
        for p in char_pred:
            if isinstance(p, (list, tuple, str)):
                for c in p:
                    flat_pred.append(str(c))
            else:
                # Jeśli trafił się float lub coś innego, traktujemy to jako pusty znak
                continue

        plot_confusion_heatmap(flat_true, flat_pred, "Macierz pomyłek CRNN (Linie)", "confusion_matrix_lines.png", overwrite=True)

        # Zbieranie danych do raportu CER/WER i niepewności
        results_for_report = []
        with torch.no_grad():
            for batch in tqdm(export_loader, desc="Generowanie raportu"):
                if batch is None: continue
                imgs, lbls, _, _ = batch
                output = model(imgs.to(DEVICE))
                lp = output[0] if isinstance(output, (tuple, list)) else output
                preds_strings, _ = encoder.decode_greedy(lp)

                for gt, pred, logit in zip(lbls, preds_strings, lp.permute(1, 0, 2)):
                    results_for_report.append({
                        'gt': str(gt), 'pred': str(pred),
                        'dist': edit_distance(str(gt), str(pred)),
                        'uncertain': len(model.get_uncertainty_zones(logit.unsqueeze(1))) > 0
                    })

        # Wykresy
        try:
            # Pobieramy paczkę danych
            sample_batch = next(iter(export_loader))

            if sample_batch is not None and len(sample_batch) > 0:
                device = next(model.parameters()).device

                # Zakładamy, że batch[0] to obrazy, batch[1] to etykiety
                images = sample_batch[0]
                sample_img = images[0].unsqueeze(0).to(device)

                attn_save_path = os.path.join(VISUAL_DEBUG_DIR, "sample_attention_map.png")

                # Generowanie mapy
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

        # Generacja raportu końcowego
        generate_final_report(results=results_for_report,
                              output_path=os.path.join(CHECKPOINT_FOLDER, "final_thesis_report_lines.txt"),
                              plot_path=os.path.join(VISUAL_DEBUG_DIR, "uncertainty_coverage_chart_lines.png"))

        # Metryki TTA
        tqdm.write(f"[{now()}] Liczenie szczegółowych metryk (Character/Word Error Rate dla słów).")
        
        # Zmiana zbioru na val_dataset, aby poprawnie podzielić na short/medium/long
        detailed_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn_dynamic)
        detailed_results = evaluate_full_metrics(model, detailed_loader, DEVICE, encoder)

        with open(os.path.join(CHECKPOINT_FOLDER, "final_metrics_report_words.json"), "w") as f:
            json.dump(detailed_results, f)

        writer.close()
        tqdm.write(f"[{now()}] Wszystkie raporty, zbiory danych i wykresy gotowe. Trening CRNN zakończony.")

""" cd /home/marek/OCR/HandwrittenTextRecognition
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    docker system prune -a -f
    sudo docker compose down && sudo docker compose up -d
    docker build -t hcr-resnet-crnn .
    docker run -it --rm --name trening_test \
  --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -v /home/marek/OCR/HandwrittenTextRecognition/Data:/app/Data \
  -v /home/marek/OCR/HandwrittenTextRecognition/output_data:/app/output_data \
  -v /home/marek/OCR/HandwrittenTextRecognition/Models:/app/Models \
  --shm-size=16gb \
  hcr-resnet-crnn \
  python3 /app/Models/ResNetCRNNWordRecognition.py
  
    -it logi na żywo, --rm usuwanie kontenera od razu po przerwaniu
    --name nazwa kontenera, --gpus all wszystkie dostępne rdzenie GPU,   
    -e PYTHONBUFFERED = 1 wypisywanie logów w czasie rzeczywistym
    -v mapuje lokalny folder do kontenera pod podaną ścieżkę
    --shm-size - rozmiar pamięci współdzielonej, duży bo wątki przetwarzają np. cvl_lines
    python3... komenda startująca wewnątrz kontenera """