import os
import shutil
import json
import time
import random
import re
import numpy as np  # Required: 1.26.4
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from tqdm import tqdm
import difflib
import kenlm
from pyctcdecode import BeamSearchDecoderCTC, Alphabet, LanguageModel

# ==========================================
# 1. KONFIGURACJA ŚCIEŻEK (Sprawdź czy się zgadzają)
# ==========================================
CHECKPOINT_PATH = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"
DATA_ROOT = r"C:\OCR\iam_words\iam_words\words"
LM_PATH = r"C:\OCR\HandwrittenTextRecognition\lm_files\english_3gram.binary"
CAPSNET_DATA_DIR = r"C:\OCR\archive\iam_words\words_context"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16  # Zgodnie z Twoim wymogiem
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 568
CONFIDENCE_THRESHOLD = 0.80


# ==========================================
# 2. KLASY I ENKODER
# ==========================================
class HTREncoder:
    def __init__(self, char_list):
        self.char_list = sorted([c for c in char_list if c != ' '])
        self.char_to_num = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        self.blank_index = 0

    def get_num_classes(self): return len(self.char_list) + 1


class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer2[0].conv1.stride = (2, 1);
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1);
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1);
        resnet.layer4[0].downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                 resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5))
        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=0.5)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(3, 0, 1, 2).reshape(w, b, -1)
        features = self.projection(features)
        rnn_out, _ = self.lstm(features)
        return self.output(rnn_out).log_softmax(2)


# ==========================================
# 3. FUNKCJE POMOCNICZE
# ==========================================
def get_label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    label = re.sub(r'_\d+$', '', name)
    mapping = {'#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-', '#B': '(', '#K': ')', '#S': ';', '#L': ':',
               '#Q': '?', '#F': '/', '#G': '"'}
    for code, char in mapping.items(): label = label.replace(code, char)
    return label


def get_crop(image_np, t_center, target_size=64):
    h_orig, w_orig = image_np.shape
    pixel_x = int(t_center * 4)  # Skalowanie w oparciu o architekturę CNN
    x1 = max(0, min(pixel_x - target_size // 2, w_orig - target_size))
    y1 = (h_orig - target_size) // 2
    crop = image_np[y1:y1 + target_size, x1:x1 + target_size]
    return cv.resize(crop, (target_size, target_size)) if crop.shape != (64, 64) else crop


def align_prediction_to_ground_truth(gt_text, pred_text):
    matcher = difflib.SequenceMatcher(None, gt_text, pred_text)
    aligned_gt, aligned_pred = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            aligned_gt.extend(list(gt_text[i1:i2]));
            aligned_pred.extend(list(pred_text[j1:j2]))
        elif tag == 'replace':
            g_part, p_part = list(gt_text[i1:i2]), list(pred_text[j1:j2])
            max_l = max(len(g_part), len(p_part))
            aligned_gt.extend(g_part + ['[pusty]'] * (max_l - len(g_part)))
            aligned_pred.extend(p_part + ['[pusty]'] * (max_l - len(p_part)))
        elif tag in ('delete', 'insert'):
            max_l = max(i2 - i1, j2 - j1)
            aligned_gt.extend(list(gt_text[i1:i2]) + ['[pusty]'] * (max_l - (i2 - i1)))
            aligned_pred.extend(list(pred_text[j1:j2]) + ['[pusty]'] * (max_l - (j2 - j1)))
    return aligned_gt, aligned_pred


# ==========================================
# 4. GŁÓWNA LOGIKA EKSPORTU
# ==========================================
def force_error_analysis():
    print(f"[{time.strftime('%H:%M:%S')}] Inicjalizacja eksportu awaryjnego...")

    # Przygotowanie Enkodera i Dekodera
    base_chars = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    char_list = sorted(list(set(base_chars + ['.', ',', '!', '?', ':', ';', "'", '(', ')', '-', '/'])))
    encoder = HTREncoder(char_list)

    decoder = BeamSearchDecoderCTC(
        Alphabet([""] + encoder.char_list, is_bpe=False),
        LanguageModel(kenlm.Model(LM_PATH), alpha=0.15, beta=0.1)
    )

    # Ładowanie Modelu
    model = ResNetCRNN(encoder.get_num_classes()).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Przygotowanie Danych Walidacyjnych
    all_files = [os.path.join(r, f) for r, d, fs in os.walk(DATA_ROOT) for f in fs if f.endswith('.png')]
    val_files = all_files[int(0.9 * len(all_files)):]

    transform = alb.Compose([alb.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

    char_true, char_pred = [], []
    os.makedirs(CAPSNET_DATA_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Analiza {len(val_files)} obrazów...")

    with torch.no_grad():
        for f_path in tqdm(val_files):
            img_raw = cv.imread(f_path, cv.IMREAD_GRAYSCALE)
            if img_raw is None: continue

            # Preprocess dla CRNN
            h_o, w_o = img_raw.shape
            ratio = w_o / h_o
            new_w = min(int(64 * ratio), 568)
            img_resized = cv.resize(img_raw, (new_w, 64))
            if new_w < 568:
                img_resized = cv.copyMakeBorder(img_resized, 0, 0, 0, 568 - new_w, cv.BORDER_CONSTANT, value=255)

            input_tensor = transform(image=img_resized)['image'].unsqueeze(0).to(DEVICE)

            # Inference
            log_probs = model(input_tensor)
            probs_np = torch.exp(log_probs).permute(1, 0, 2).cpu().numpy()[0]
            pred_text = decoder.decode(probs_np).strip()
            gt_text = get_label_from_filename(os.path.basename(f_path))

            # Align i zapis do listy pomyłek
            a_gt, a_pred = align_prediction_to_ground_truth(gt_text, pred_text)
            char_true.extend(a_gt);
            char_pred.extend(a_pred)

            # Eksport wycinków (Punkt 2 - Hard Mining)
            # Wykorzystanie torch.long dla etykiet
            max_probs, indices = torch.max(torch.tensor(probs_np), dim=-1)
            prev_idx = -1
            for t, idx in enumerate(indices.tolist()):
                if idx != 0 and idx != prev_idx:
                    char = encoder.num_to_char.get(idx, '')
                    if char:
                        crop = get_crop(img_raw, t)
                        if 0.01 < (np.sum(crop < 200) / (64 * 64)) < 0.60:
                            char_dir = os.path.join(CAPSNET_DATA_DIR, char if char.isalnum() else f"sym_{ord(char)}")
                            os.makedirs(char_dir, exist_ok=True)
                            cv.imwrite(os.path.join(char_dir, f"crop_{t}_{idx}.png"), crop)
                prev_idx = idx

    # Zapis pliku JSON #1
    with open(os.path.join(CAPSNET_DATA_DIR, "crnn_error_analysis.json"), "w") as f:
        json.dump({'char_true': char_true, 'char_pred': char_pred}, f)

    print(f"[{time.strftime('%H:%M:%S')}] SUKCES: Plik crnn_error_analysis.json i wycinki gotowe w {CAPSNET_DATA_DIR}")


if __name__ == "__main__":
    force_error_analysis()
