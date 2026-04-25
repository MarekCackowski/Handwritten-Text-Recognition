import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_sauvola
import re
import difflib
import string
import kenlm
import enchant

# ==========================================
# 1. KONFIGURACJA
# ==========================================
INPUT_DIR_LINES = r"C:\OCR\cvl_dataset_words"
OUTPUT_DIR_LABELED = r"C:\OCR\cvl_auto_labeled_dataset"

# Ścieżki
MODEL_PATH = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"
DICTIONARY_FILE = r"C:\OCR\dictionary.txt"
LM_PATH = r"C:\Users\marek\OneDrive\Pulpit\HandwrittenTextRecognition\lm_files\english_3gram.binary"

CONFIDENCE_THRESHOLD = 0.75
MIN_WORD_AREA = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512

# OTHERS
IMAGE_HEIGHT_CHAR = 28
IMAGE_WIDTH_CHAR = 28
IMAGE_SIZE_CHAR = (IMAGE_HEIGHT_CHAR, IMAGE_WIDTH_CHAR)
EMNIST_NORM = ((0.1307,), (0.3081,))
VISUAL_CONFUSION_CHARS = {'l', '1', 'o', '0', 'i', 'u', 'v', 'I', 'J', 'S', '5', 'B', '8'}

REVERSE_PUNCTUATION_MAP = {
    '#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-',
    '#B': '(', '#K': ')', '#S': ';', '#L': ':', '#U': '"'
}


# ==========================================
# 2. HTR SPELL CORRECTOR (FULL LOGIC)
# ==========================================

class HTRSpellCorrector:
    PUNCTUATION_TO_SCRUB_CHARS = [',', '.', ';', '(', ')', '!', '?', '"']

    # Mapowania do generowania wariantów cyfr/liter
    CORRECTIONS_LOWER = {'0': 'o', '1': ['l', 'i'], '2': 'z', '5': 's', '6': 'b', 'c': ['e', 'o'], 'i': 'l', 'l': 'i',
                         'n': 'r', 'p': ['b', 'q'], 't': 'f', 'u': 'v', 'v': 'u'}
    CORRECTIONS_UPPER = {'0': 'O', '1': ['I', 'J'], '2': 'Z', '5': 'S', '8': 'B', '6': 'B', 'C': ['G', 'O'], 'E': 'F',
                         'F': 'E', 'I': 'L', 'L': 'I', 'P': ['R', 'B'], 'U': 'V', 'V': 'U'}

    def __init__(self, lm_path):
        try:
            self.model = kenlm.Model(lm_path) if kenlm and os.path.exists(lm_path) else None
        except:
            self.model = None

        try:
            self.d = enchant.Dict("en_US") if enchant else None
        except:
            self.d = None

        self.contraction_fixer = re.compile(r"([a-zA-Z])\s+(s|t|d|ll|m|re|ve)\b", re.IGNORECASE)

    @staticmethod
    def _check_for_vowel_fix(original_word, candidate):
        matcher = difflib.SequenceMatcher(None, original_word, candidate)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' and i2 - i1 == 1 and j2 - j1 == 1:
                if original_word[i1:i2] == "'" and candidate[j1:j2] in ['a', 'e', 'i']: return True
        return False

    def _scrub_mid_word_punctuation(self, text):
        new_text = text
        for char in self.PUNCTUATION_TO_SCRUB_CHARS:
            pattern = re.compile(r'([a-zA-Z0-9])' + re.escape(char) + r'([a-zA-Z0-9])')
            new_text = pattern.sub(r'\1\2', new_text)
        return new_text

    def smart_preprocess(self, text):
        return self.contraction_fixer.sub(r"\1'\2", text)

    @staticmethod
    def merge_two_to_one(raw_word):
        merged = []
        i = 0
        while i < len(raw_word):
            if i < len(raw_word) - 1:
                pair = raw_word[i:i + 2].lower()
                if pair in {'vv', 'vu', 'uv', 'uu'}:
                    merged.append('w')
                    i += 2
                    continue
                elif pair in {'rn', 'ri', 'nn'}:
                    merged.append('m')
                    i += 2
                    continue
                elif pair == 'cl':
                    merged.append('d')
                    i += 2
                    continue
            merged.append(raw_word[i])
            i += 1
        return ''.join(merged)

    @staticmethod
    def _is_digit_ambiguous(word):
        return sum(1 for c in word if c.isdigit()) == 1

    @staticmethod
    def _get_safe_digit_variants(word):
        if not HTRSpellCorrector._is_digit_ambiguous(word): return []
        candidates = set()
        confs = HTRSpellCorrector.CORRECTIONS_UPPER if any(
            c.isupper() for c in word) else HTRSpellCorrector.CORRECTIONS_LOWER
        for i, c in enumerate(word):
            if c.isdigit():
                opts = confs.get(c, [])
                if isinstance(opts, str): opts = [opts]
                for opt in opts:
                    if isinstance(opt, str) and opt.isalpha(): candidates.add(word[:i] + opt + word[i + 1:])
        return list(candidates)

    def get_enchant_candidates(self, word):
        if self.d and not self.d.check(word): return self.d.suggest(word)[:5]
        return []

    def check(self, word):
        # Sprawdzenie słownika dla zewnętrznego użytku (Semantic Merge)
        return self.d.check(word) if self.d else False

    def correct(self, word):
        # Uproszczona korekcja dla logiki Semantic Merge
        if self.d and self.d.check(word): return word

        # Używamy tylko pierwszej sugestii, jeśli jest
        sug = self.d.suggest(word) if self.d else []
        if sug:
            match = sug[0]
            if word and word[0].isupper(): return match.capitalize()
            return match
        return word

    def correct_text(self, raw_sentence):
        # Pełna korekcja zdaniowa (do użytku wewnętrznego)
        clean_sentence = self.smart_preprocess(raw_sentence)
        words = clean_sentence.split()
        final_words = []
        prev_word = "<s>"

        for word in words:
            candidates = set()
            orig = word
            if self.d and self.d.check(word):
                candidates.add(word)
            else:
                orig = self.merge_two_to_one(word); candidates.add(orig)

            if not (self.d and self.d.check(orig)):
                candidates.update(self._get_safe_digit_variants(word))

            best_cand = orig
            best_score = -float('inf')
            for cand in candidates:
                score = self.model.score(f"{prev_word} {cand}") if self.model else 0
                if self._check_for_vowel_fix(word, cand): score -= 100
                if self.d: score += 5.0 if self.d.check(cand) else -2.0
                score += 1.5 if cand == orig else -0.5
                if score > best_score: best_score = score; best_cand = cand

            final_words.append(best_cand)
            prev_word = best_cand
        return " ".join(final_words)


# ==========================================
# 3. SEGMENTACJA
# ==========================================
class Binarizer:
    def __init__(self, image):
        self.image = image.astype(np.uint8)

    def binarize(self):
        img = self.image
        adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 10)
        sauvola = ((img > threshold_sauvola(img, window_size=61, k=0.1)) * 255).astype(np.uint8)
        final = np.zeros_like(adaptive)
        n, l, s, _ = cv.connectedComponentsWithStats(sauvola, connectivity=8)
        for j in range(1, n):
            c = l == j
            if np.sum(c) > 0 and (np.sum(c & (adaptive == 255)) / np.sum(c)) >= 0.15: final[c] = 255
        return cv.morphologyEx(final, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))


class LineToWordSegmentor:
    def __init__(self):
        pass

    @staticmethod
    def clean_word_crop(img):
        _, b = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        n, l, s, _ = cv.connectedComponentsWithStats(b, 8)
        mask = np.zeros_like(b)
        h, w = img.shape
        for i in range(1, n):
            x, y, ww, hh, a = s[i]
            is_noise = (y <= 0 and (y + hh) < h * 0.4) or (y + hh >= h and y > h * 0.6) or (
                        (x <= 0 or x + ww >= w) and (ww < 5 or a < 20))
            if is_noise: mask[l == i] = 255
        c = img.copy()
        c[mask == 255] = 255
        return c

    def extract_atomic_crops(self, line_path):
        img = cv.imread(line_path, cv.IMREAD_GRAYSCALE)
        if img is None: return []  # Safety check

        # 1. Deskew & Deslant
        _, binarized = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        pts = cv.findNonZero(binarized)
        if pts is not None:
            angle = cv.minAreaRect(pts)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 0.5 < abs(angle) < 15:
                M = cv.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
                img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)

        h, w = img.shape
        best_s, max_v = 0, 0
        _, t = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        for s in np.linspace(-0.5, 0.5, 11):
            M = np.float32([[1, s, 0], [0, 1, 0]])
            v = np.var(np.sum(cv.warpAffine(t, M, (w + int(h * 0.5), h), flags=cv.INTER_NEAREST), axis=0))
            if v > max_v: max_v, best_s = v, s
        if abs(best_s) > 0.05:
            M = np.float32([[1, best_s, 0], [0, 1, 0]])
            img = cv.warpAffine(img, M, (w + abs(int(h * 0.5)), h), borderValue=255)

        # 2. Binarize & Contour
        binary = Binarizer(img).binarize()
        clean_binary = cv.morphologyEx(binary, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        inv = cv.bitwise_not(clean_binary)
        dilated = cv.dilate(inv, cv.getStructuringElement(cv.MORPH_RECT, (1, 1)), iterations=1)
        cnts, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        boxes = [cv.boundingRect(c) for c in cnts if cv.boundingRect(c)[2] * cv.boundingRect(c)[3] > MIN_WORD_AREA]
        boxes.sort(key=lambda b: b[0])

        if not boxes: return []
        gaps = [boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)]
        med = np.median([g for g in gaps if g > 0]) if gaps else 0
        thresh = max(10, img.shape[0] * 0.15, med * 4.5)

        merged = []
        curr = boxes[0]
        for i in range(1, len(boxes)):
            nxt = boxes[i]
            if (nxt[0] - (curr[0] + curr[2])) < thresh:
                x = min(curr[0], nxt[0])
                y = min(curr[1], nxt[1])
                curr = (x, y, max(curr[0] + curr[2], nxt[0] + nxt[2]) - x, max(curr[1] + curr[3], nxt[1] + nxt[3]) - y)
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

        crops = []
        h, w = img.shape
        pad = 4
        for (x, y, ww, hh) in merged:
            if ww > w * 0.98: continue

            c = img[max(0, y - pad):min(h, y + hh + pad), max(0, x - pad):min(w, x + ww + pad)]

            # --- SKALOWANIE DO DOCELOWEJ WYSOKOŚCI ---
            if c.shape[0] > 0 and c.shape[1] > 0:
                scale_factor = IMAGE_HEIGHT / c.shape[0]
                target_w = int(c.shape[1] * scale_factor)
                c_resized = cv.resize(c, (target_w, IMAGE_HEIGHT), interpolation=cv.INTER_LINEAR)
                crops.append(self.clean_word_crop(c_resized))
            else:
                continue

        return crops


# ==========================================
# 4. CRNN & UTILS
# ==========================================
def get_char_list():
    alnum = sorted(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
    raw_chars = alnum + ['.', ',', '?', '!', ':', ';', '"', "'", '(', ')', '-', ' ']
    char_list = sorted(list(set(raw_chars)))
    if ' ' in char_list: char_list.remove(' '); char_list.append(' ')
    return char_list


class HTRCharEncoder:
    def __init__(self, char_list):
        self.char_list = char_list
        self.num_to_char = {i + 1: c for i, c in enumerate(char_list)}


class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        r = models.resnet18(weights=None)
        r.layer2[0].conv1.stride = (2, 1)
        r.layer2[0].downsample[0].stride = (2, 1)
        r.layer3[0].conv1.stride = (2, 1)
        r.layer3[0].downsample[0].stride = (2, 1)
        r.layer4[0].conv1.stride = (2, 1)
        r.layer4[0].downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.rnn_input_size = 512 * 2
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25))
        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=0.25)
        self.output = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 3, 1, 1)
        f = self.cnn(x)
        b, c, h, w = f.size()
        f = f.permute(3, 0, 1, 2).reshape(w, b, -1)
        return self.output(self.lstm(self.projection(f))[0]).log_softmax(2)


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR_LINES): exit()
    dir_high = os.path.join(OUTPUT_DIR_LABELED, "high_confidence")
    dir_low = os.path.join(OUTPUT_DIR_LABELED, "review_hard_cases")
    os.makedirs(dir_high, exist_ok=True)
    os.makedirs(dir_low, exist_ok=True)

    # Init
    chars = get_char_list()
    encoder = HTRCharEncoder(chars)

    # 1. INICJALIZACJA CORRECTORA TUTAJ
    corrector = HTRSpellCorrector(LM_PATH)

    model = ResNetCRNN(len(chars) + 1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
    model.eval()

    transform = transforms.Compose(
        [transforms.Resize((64, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 2. INICJALIZACJA SEGMENTATORA
    segmentor = LineToWordSegmentor()

    files = []
    for r, d, f in os.walk(INPUT_DIR_LINES):
        for file in f:
            if file.endswith(('.png', '.jpg', '.tif')): files.append(os.path.join(r, file))

    print(f"Processing {len(files)} lines...")

    total_words_saved = 0

    # Mapa znaków dla nazw plików
    IAM_CODE_MAP = {'.': '#D', ',': '#C', "'": '#A', '!': '#E', '-': '#H', '(': '#B', ')': '#K', ';': '#S', ':': '#L',
                    '"': '#U', '?': '#Q'}

    for line_path in tqdm(files):
        try:
            words = segmentor.extract_atomic_crops(line_path)
            base = os.path.splitext(os.path.basename(line_path))[0]

            if not words: continue


            # --- Helper to score an image ---
            def score_image(img):
                # Pad to square-ish to help model? No, CRNN handles width.
                # Just add whitespace padding
                h, w = img.shape
                padded = cv.copyMakeBorder(img, 0, 0, 10, 10, cv.BORDER_CONSTANT, value=255)
                pil = Image.fromarray(padded)
                # FIX: Poprawiona kolejnosc resize arguments
                resized_img = cv.resize(padded, (IMAGE_WIDTH, IMAGE_HEIGHT))
                t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])(
                    Image.fromarray(resized_img)).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    preds = model(t)
                    probs = torch.exp(preds)
                    max_p, idxs = torch.max(probs, dim=2)
                    confs = max_p.squeeze(1).tolist()
                    decoded = []
                    prev = -1
                    sample_confs = []
                    for k, idx in enumerate(idxs.squeeze(1).tolist()):
                        if idx != 0 and idx != prev:
                            decoded.append(encoder.num_to_char.get(idx, ''))
                            sample_confs.append(confs[k])
                        prev = idx
                    text = "".join(decoded)
                    avg_conf = sum(sample_confs) / len(sample_confs) if sample_confs else 0.0
                return text, avg_conf


            # --- End Helper ---

            if not words: continue
            accumulator_img = words[0]
            curr_text, curr_conf = score_image(accumulator_img)

            i = 1
            while i < len(words):
                next_img = words[i]

                # ZABEZPIECZENIE WYSOKOŚCI: Używamy wysokości accumulator_img
                spacer = np.full((accumulator_img.width, 5), 255, dtype=np.uint8)

                if next_img.width != accumulator_img.width:
                    # BŁĄD: next_img.width, accumulator_img.width
                    target_width = int(next_img.height * (accumulator_img.width / next_img.width))
                    target_height = accumulator_img.width
                    next_img = cv.resize(next_img, (target_width, target_height), interpolation=cv.INTER_LINEAR)

                merged_img = np.hstack((accumulator_img, spacer, next_img))
                merged_text, merged_conf = score_image(merged_img)

                should_merge = False

                # Rule A: Dictionary Valid
                if len(merged_text) > 1 and corrector.check(merged_text) and not corrector.check(curr_text):
                    should_merge = True
                # Rule B: Confidence Boost
                elif merged_conf > (curr_conf + 0.1):
                    should_merge = True
                # Rule C: Rescue weak
                elif len(curr_text) < 2 and merged_conf > 0.6:
                    should_merge = True

                if should_merge:
                    accumulator_img = merged_img
                    curr_text = merged_text
                    curr_conf = merged_conf
                else:
                    # SPLIT -> Save Current
                    final_text = curr_text
                    final_conf = curr_conf

                    if len(final_text) > 1:
                        if not corrector.check(final_text):
                            fixed = corrector.correct(final_text)
                            if fixed != final_text: final_text = fixed; final_conf = 0.5  # Dictionary Fix

                    if len(final_text) > 0:
                        safe_label = "".join([c if c.isalnum() else IAM_CODE_MAP.get(c, '') for c in final_text])
                        if safe_label:
                            fname = f"{final_conf:.2f}___{safe_label}___{base}_{total_words_saved}.png"
                            target = dir_high if final_conf >= CONFIDENCE_THRESHOLD else dir_low
                            try:
                                cv.imwrite(os.path.join(target, fname), accumulator_img)
                                total_words_saved += 1
                            except:
                                pass

                    # Reset
                    accumulator_img = next_img
                    curr_text, curr_conf = score_image(accumulator_img)
                i += 1

            # Save last chunk
            if len(curr_text) > 0:
                final_text = curr_text
                final_conf = curr_conf
                if len(final_text) > 1:
                    if not corrector.check(final_text):
                        fixed = corrector.correct(final_text)
                        if fixed != final_text: final_text = fixed; final_conf = 0.5  # Dictionary Fix

                if len(final_text) > 0:
                    safe_label = "".join([c if c.isalnum() else IAM_CODE_MAP.get(c, '') for c in final_text])
                    if safe_label:
                        fname = f"{final_conf:.2f}___{safe_label}___{base}_{total_words_saved}.png"
                        target = dir_high if final_conf >= CONFIDENCE_THRESHOLD else dir_low
                        try:
                            cv.imwrite(os.path.join(target, fname), accumulator_img)
                            total_words_saved += 1
                        except:
                            pass


        except Exception as e:
            # print(f"CRITICAL ERROR on {os.path.basename(line_path)}: {e}")
            continue

    display_count = total_words_saved if total_words_saved is not None else 0
    print(f"\nDone. Total words extracted and labeled: {display_count}")
