import gc
import sys
import time

sys.path.append(r"C:\OCR\HandwrittenTextRecognition")
import os
import cv2 as cv
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import re
from Models.ResNetCRNNWordRecognition import ResNetCRNN

OUTPUT_CHARS_DIR = r"C:\OCR\archive\iam_words\letters_hard"
DATA_ROOT = r"C:\OCR\archive\iam_words\words"
CRNN_CHECKPOINT = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"

IMAGE_HEIGHT_WORD = 64
IMAGE_WIDTH_LINE = 2048
MODEL_DROPOUT = 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_CROP_WIDTH = 48
MIN_CROP_WIDTH = 20

# Musimy dopasować się do wagi (69 klas -> 68 unikalnych znaków + BLANK)
TARGET_NUM_CHARS = 68

# Mapowania
REVERSE_PUNCTUATION_MAP = {
    '#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-',
    '#B': '(', '#K': ')', '#S': ';', '#L': ':', '#U': '"'
}
FILENAME_MAP = {
    'dot': '.', 'comma': ',', 'question': '?', 'exclamation': '!',
    'colon': ':', 'semicolon': ';', 'quote': '"', 'apostrophe': "'",
    'lparen': '(', 'rparen': ')', 'hyphen': '-', 'space': ' '
}

# Wartości normalizacji dla EMNIST ByClass (62 klasy)
EMNIST_NORM = ((0.1307,), (0.3081,))


def get_emnist_char_list_byclass() -> list:
    """Zwraca listę 62 znaków z EMNIST 'byclass' (0-9, A-Z, a-z)."""
    char_list = []
    # 0-9
    for i in range(10): char_list.append(chr(48 + i))
    # A-Z
    for i in range(26): char_list.append(chr(65 + i))
    # a-z
    for i in range(26): char_list.append(chr(97 + i))
    return char_list


def apply_capsnet_augmentations(img_np: np.ndarray) -> Image.Image:
    """ Funkcja nie jest używana w tym Minerze, ale została pozostawiona dla kompletności,
        jeśli jest używana w innym miejscu modułu. """
    alpha = 120  # Siła deformacji
    sigma = 8  # Gładkość deformacji
    p_trans = 0.4  # Prawdopodobieństwo

    # Elastic Transform
    if np.random.rand() < p_trans:
        random_state = np.random.RandomState(None)
        shape = img_np.shape

        # Generowanie losowego pola przesunięć
        dx = cv.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # Aplikacja deformacji
        img_np = cv.remap(img_np, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=255)

    # Shift, Scale, Rotate (Afiniczne)
    if np.random.rand() < p_trans:
        h, w = img_np.shape
        
        # Losowanie parametrów
        angle = np.random.uniform(-5, 5)
        scale = np.random.uniform(0.9, 1.1)
        tx = np.random.uniform(-0.0625, 0.0625) * w
        ty = np.random.uniform(-0.0625, 0.0625) * h

        # Macierz transformacji
        M = cv.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        img_np = cv.warpAffine(img_np, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=255)

    # Coarse Dropout (Symulacja dziur/plam)
    if np.random.rand() < 0.3:
        h_img, w_img = img_np.shape
        
        # Wymiary dziury (max. 8 x 8)
        h_hole = np.random.randint(1, 9)
        w_hole = np.random.randint(1, 9)

        # Pozycja
        y1 = np.random.randint(0, h_img - h_hole)
        x1 = np.random.randint(0, w_img - w_hole)

        # Wstawienie białego kwadratu (255)
        img_np[y1:y1 + h_hole, x1:x1 + w_hole] = 255

    return Image.fromarray(img_np)


def smart_crop_char(word_img, center_x):
    """ Precyzyjne docięcie znaku. """
    h, w = word_img.shape
    half_width = MAX_CROP_WIDTH // 2

    x1, x2 = max(0, int(center_x - half_width)), min(w, int(center_x + half_width))
    crop = word_img[:, x1:x2].copy()

    if crop.shape[1] < MIN_CROP_WIDTH:
        return None

    try:
        # Jawne rzutowanie na float()
        mean_val = float(np.mean(crop))

        if mean_val > 127:
            # Tło jasne (IAM) -> inwersja i Otsu
            _, binary = cv.threshold(crop, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            # Tło ciemne (PHSF po inwersji) -> Otsu
            _, binary = cv.threshold(crop, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        coords = cv.findNonZero(binary)
        if coords is not None:
            _, y_box, _, h_box = cv.boundingRect(coords)
            y_start, y_end = max(0, y_box - 2), min(h, y_box + h_box + 2)

            final_crop = crop[y_start:y_end, :]

            if final_crop.shape[0] < 8 or final_crop.shape[1] < 8:
                return None

            return final_crop

    except (cv.error, AttributeError, TypeError, ValueError):
        return None

    return None


def get_label_from_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    label = re.sub(r'_\d+$', '', name_no_ext)
    for code, char in REVERSE_PUNCTUATION_MAP.items():
        if code in label: label = label.replace(code, char)
    label = FILENAME_MAP.get(label, label)
    return label


def decode_greedy(log_probs, encoder):
    """Dekodowanie zachłanne CTC."""
    probs = torch.exp(log_probs)
    indices = torch.argmax(probs, dim=2).squeeze(1).tolist()
    decoded = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx: decoded.append(encoder.num_to_char.get(idx, ''))
        prev_idx = idx
    return "".join(decoded)


def get_ctc_segments(log_probs, encoder):
    """Zwraca segmenty CTC (używane do lokalizacji znaków)."""
    probs = torch.exp(log_probs)
    max_probs, indices = torch.max(probs, dim=2)
    segments = []
    prev_idx = -1
    
    # Iteracja po krokach czasowych
    for t, idx in enumerate(indices.squeeze(1).tolist()):
        # Zapisujemy tylko pierwsze wystąpienie znaku po BLANK lub poprzednim znaku
        if idx != 0 and idx != prev_idx:
            segments.append({'char': encoder.num_to_char.get(idx, '?'), 'timestep': t})
        prev_idx = idx
    return segments


def load_word_for_inference(image_path, transform):
    try:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None: return None, None
        pil_img = Image.fromarray(img)
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        return tensor, img
    except (cv.error, AttributeError, TypeError, ValueError):
        return None


class HTREncoder:
    def __init__(self, char_list):
        # Spacja jest zawsze na końcu dla stabilności i porządku sortowania
        if ' ' in char_list:
            char_list.remove(' ')

        self.char_list = sorted(list(set(char_list)))
        self.char_list.append(' ')  # Spacja (jeśli jest) powinna być dodana po sortowaniu

        self.char_to_num = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        self.blank_index = 0

    def decode(self, indices):
        res = []
        for i in indices:
            if i != self.blank_index: res.append(self.num_to_char.get(i, ''))
        return "".join(res)


# Kopia klasy ResNetCRNN
class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Modyfikacja stride (zachowanie szerokości sekwencji)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.rnn_input_size = 512 * 2
        self.projection = nn.Sequential(
            nn.Linear(self.rnn_input_size, 256),
            nn.ReLU(),
            nn.Dropout(MODEL_DROPOUT)
        )

        self.lstm = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=False, dropout=MODEL_DROPOUT)
        self.output = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(3, 0, 1, 2).reshape(w, b, -1)
        features = self.projection(features)
        rnn_out, _ = self.lstm(features)
        return self.output(rnn_out).log_softmax(2)


# Mapowania i konfiguracja
POLISH_DATA_ROOT = r"C:\OCR\PHSF\phsf\inwokacja\png"
IAM_DATA_ROOT = r"C:\OCR\archive\iam_words\words"
OUTPUT_CHARS_DIR = r"C:\OCR\archive\iam_words\letters_hard"

# Tekst Inwokacji zmapowany na numery plików w PHSF
INWOKACJA_LINES = {
    "001": "Litwo Ojczyzno moja ty jesteś jak zdrowie",
    "002": "Ile cię trzeba cenić ten tylko się dowie",
    "003": "Kto cię stracił Dziś piękność twą w całej ozdobie",
    "004": "Widzę i opisuję bo tęsknię po tobie",
    "005": "Panno święta co Jasnej bronisz Częstochowy",
    "006": "I w Ostrej świecisz Bramie Ty co gród zamkowy",
    "007": "Nowogródzki ochraniasz z jego wiernym ludem",
    "008": "Jak mnie dziecko do zdrowia powróciłaś cudem",
    "009": "Gdy od płaczącej matki pod Twoją opiekę",
    "010": "Ofiarowany martwą podniosłem powiekę",
    "011": "I zaraz mogłem pieszo do Twych świątyń progu",
    "012": "Iść za wrócone życie podziękować Bogu",
    "013": "Tak nas powrócisz cudem na Ojczyzny łono"
}


def get_pl_char_list():
    """Zwraca polskie znaki diakrytyczne używane w CapsNet."""
    return list("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def get_combined_char_list():
    """Tworzy pełny alfabet: EMNIST + Symbole + Polskie znaki."""
    base = get_emnist_char_list_byclass()
    polish = get_pl_char_list()
    punct = ['.', ',', '!', '?', ':', ';', "'", '(', ')', '-', '/', ' ']
    full_list = sorted(list(set(base + polish + punct)))
    # Spacja na koniec dla stabilności CTC
    if ' ' in full_list:
        full_list.remove(' ')
    full_list.append(' ')
    return full_list


def get_pl_safe_name(char):
    """Mapuje znaki na nazwy folderów bezpieczne dla Windows."""
    mapping = {
        'ą': 'a_pl', 'ć': 'c_pl', 'ę': 'e_pl', 'ł': 'l_pl', 'ń': 'n_pl',
        'ó': 'o_pl', 'ś': 's_pl', 'ź': 'z_pl1', 'ż': 'z_pl2',
        'Ą': 'A_PL', 'Ć': 'C_PL', 'Ę': 'E_PL', 'Ł': 'L_PL', 'Ń': 'N_PL',
        'Ó': 'O_PL', 'Ś': 'S_PL', 'Ź': 'Z_PL1', 'Ż': 'Z_PL2',
        '.': 'dot', ',': 'comma', ':': 'colon', ';': 'semicolon',
        '?': 'question', '!': 'exclamation', "'": 'apostrophe', ' ': 'space'
    }
    return mapping.get(char, char if not char.isupper() else f"upper_{char}")


def now():
    return time.strftime('%H:%M:%S')


def extract_hard_cases_only():
    print(f"[{now()}] SMART MINING: Inicjalizacja hybrydowa (IAM English + PHSF Inwokacja)")

    # 1. PRZYGOTOWANIE ALFABETU (EN + PL + Symbole)
    char_list = get_combined_char_list()
    encoder = HTREncoder(char_list)
    num_classes = len(char_list) + 1  # +1 dla BLANK (index 0)

    # 2. INICJALIZACJA MODELU I ŁADOWANIE WAG
    # Używamy szerokiej rozdzielczości 2048, aby pomieścić pełne linie Inwokacji bez ściskania
    model = ResNetCRNN(num_classes=num_classes).to(DEVICE)

    if os.path.exists(CRNN_CHECKPOINT):
        try:
            ckpt = torch.load(CRNN_CHECKPOINT, map_location=DEVICE)
            state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt

            # Failsafe: Jeśli waga ma inną liczbę klas (np. 69) niż obecny alfabet (np. 87)
            if state_dict['output.bias'].shape[0] != num_classes:
                print(f"[{now()}] Wykryto zmianę alfabetu. Ładowanie selektywne wag CNN/RNN.")
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
                print(f"[{now()}] Załadowano pełne wagi.")
            model.eval()
        except Exception as e:
            print(f"Błąd krytyczny ładowania wag: {e}")
            return

    # 3. AGREGACJA ZADAŃ (IAM + Inwokacja)
    tasks = []

    # Angielskie IAM (Word Level)
    for root, _, files in os.walk(IAM_DATA_ROOT):
        for f in files:
            if f.endswith('.png'):
                label = get_label_from_filename(f)
                tasks.append({'path': os.path.join(root, f), 'label': label})

    # Polskie PHSF (Line Level - Inwokacja)
    if os.path.exists(POLISH_DATA_ROOT):
        for f in os.listdir(POLISH_DATA_ROOT):
            if f.endswith('.png'):
                match = re.search(r'(\d{3})', f)
                if match and match.group(1) in INWOKACJA_LINES:
                    tasks.append({
                        'path': os.path.join(POLISH_DATA_ROOT, f),
                        'label': INWOKACJA_LINES[match.group(1)]
                    })

    # Transformacja dla modelu (Stała wysokość 64, duża szerokość dla linii)
    mining_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT_WORD, IMAGE_WIDTH_LINE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    count_errors, count_chars = 0, 0
    print(f"[{now()}] Rozpoczynam analizę {len(tasks)} obrazów...")

    # 4. GŁÓWNA PĘTLA MININGU
    for task in tqdm(tasks, desc="Szukanie błędów wizyjnych"):
        img_np = cv.imread(task['path'], cv.IMREAD_GRAYSCALE)
        if img_np is None: continue

        true_label = task['label']
        if not all(c in char_list for c in true_label): continue

        # Inferencja
        pil_img = Image.fromarray(img_np)
        tensor = mining_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(tensor)  # Wynik CTC: [T, B, C]
            pred_text = decode_greedy(preds, encoder)

        # Jeśli model się pomylił (Hard Case) -> wycinamy znaki do douczenia CapsNet
        if pred_text != true_label:
            count_errors += 1
            segments = get_ctc_segments(preds, encoder)

            # SKALOWANIE: Obliczamy ile pikseli oryginału przypada na jeden krok czasowy modelu
            # preds.size(0) to liczba kroków czasowych (T) wygenerowanych przez CNN
            time_step_width = img_np.shape[1] / preds.size(0)

            for i, seg in enumerate(segments):
                char = seg['char']
                # Pomijamy spacje i znaki spoza alfabetu
                if char not in char_list or char == ' ': continue

                # Wyznaczamy środek znaku na oryginalnym obrazie
                cx = seg['timestep'] * time_step_width

                # Inteligentne wycięcie (pionowe + poziome)
                final_crop = smart_crop_char(img_np, cx)

                if final_crop is not None:
                    # Windows-Safe folder (np. 'ą' -> 'a_pl')
                    safe_folder = get_pl_safe_name(char)
                    save_dir = os.path.join(OUTPUT_CHARS_DIR, safe_folder)
                    os.makedirs(save_dir, exist_ok=True)

                    base = os.path.splitext(os.path.basename(task['path']))[0]
                    save_name = f"err_{count_errors}_{count_chars}_{base}.png"
                    cv.imwrite(os.path.join(save_dir, save_name), final_crop)
                    count_chars += 1

        # Oczyszczanie pamięci co 100 błędów
        if count_errors % 100 == 0: gc.collect()

    print(f"[{now()}] Wynik miningu błędów:")
    print(f"[{now()}] Błędy w słowach/liniach: {count_errors}")
    print(f"[{now()}] Wycięto znaków do CapsNet: {count_chars}")
    print(f"[{now()}] Folder wynikowy: {OUTPUT_CHARS_DIR}")

if __name__ == "__main__":
    extract_hard_cases_only()