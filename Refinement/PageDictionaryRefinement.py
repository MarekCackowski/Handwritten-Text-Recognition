import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_sauvola
from itertools import groupby
from App.SmartOCR import StrictSpellCorrector, PageBeamSearch

INPUT_DIR_LINES = r"C:\OCR\cvl_dataset_words"
OUTPUT_DIR_LABELED = r"C:\OCR\cvl_auto_labeled_dataset_semantic"
CRNN_PATH = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"
CAPS_PATH = r"output_data\checkpoints\hcr\CapsNet_char_level.pth"
LM_PATH = r"C:\Users\marek\OneDrive\Pulpit\HandwrittenTextRecognition\lm_files\english_3gram.binary"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512
IMAGE_SIZE_CHAR = (28, 28)
MIN_WORD_AREA = 30
ACCEPTANCE_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD_CHAR = 0.85
LOW_CONFIDENCE_CHAR_LIMIT = 0.60

IAM_CODE_MAP = {'.': '#D', ',': '#C', "'": '#A', '!': '#E', '-': '#H', '(': '#B', ')': '#K', ';': '#S', ':': '#L',
                '"': '#U', '?': '#Q'}


# Architektury
def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels=256, out_channels=32, dim_caps=8, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels * dim_caps, kernel_size=kernel_size, stride=stride)
        self.dim_caps = dim_caps
    def forward(self, x):
        outputs = self.conv2d(x)
        return squash(outputs.view(x.size(0), -1, self.dim_caps))

class DigitCaps(nn.Module):
    def __init__(self, num_capsules, num_routes=32 * 3 * 3, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
    def forward(self, x):
        batch_size = x.size(0)
        x = x[:, :, None, :, None]
        u_hat = torch.matmul(self.W, x)
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)
        v_j = torch.empty(0).to(x.device)
        for i in range(3):
            c_ij = func.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(3) * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, axis=-2)
            if i < 2:
                v_j_expanded = torch.cat([v_j] * self.num_routes, dim=1)
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j_expanded)
                b_ij = b_ij + a_ij.squeeze(4)
        return v_j.squeeze(1)

class CapsNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.primary_caps = PrimaryCaps(in_channels=256, out_channels=32, dim_caps=8, kernel_size=9, stride=2)
        self.digit_caps = DigitCaps(num_capsules=num_classes, num_routes=32 * 3 * 3, in_channels=8, out_channels=16)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.primary_caps(x)
        caps_output = self.digit_caps(output).squeeze(-1)
        return (caps_output ** 2).sum(dim=2) ** 0.5, None

class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        r = models.resnet18(weights=None)
        r.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        r.layer2[0].conv1.stride = (2, 1); r.layer2[0].downsample[0].stride = (2, 1)
        r.layer3[0].conv1.stride = (2, 1); r.layer3[0].downsample[0].stride = (2, 1)
        r.layer4[0].conv1.stride = (2, 1); r.layer4[0].downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25))
        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=0.25)
        self.output = nn.Linear(256 * 2, num_classes)
    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 2, 1, 1)
        f = self.cnn(x)
        b, c, h, w = f.size()
        f = f.permute(3, 0, 1, 2).reshape(w, b, -1)
        return self.output(self.lstm(self.projection(f))[0]).log_softmax(2)


# Helpery
def get_char_list():
    alnum = sorted(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
    raw_chars = alnum + ['.', ',', '?', '!', ':', ';', '"', "'", '(', ')', '-', ' ']
    char_list = sorted(list(set(raw_chars)))
    if ' ' in char_list: char_list.remove(' '); char_list.append(' ')
    return char_list

def get_extended_mapping():
    mapping = {}
    idx = 0
    for i in range(10): mapping[idx] = chr(48 + i); idx += 1
    for i in range(26): mapping[idx] = chr(65 + i); idx += 1
    for i in range(26): mapping[idx] = chr(97 + i); idx += 1
    return mapping

# Preprocessing i segmentacja
class LineToWordSegmentor:
    @staticmethod
    def clean_word_crop(img):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = np.asarray(img)
        img = clahe.apply(img)
        _, b = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        n, l, s, _ = cv.connectedComponentsWithStats(b, 8)
        mask = np.zeros_like(b)
        h, w = img.shape
        for i in range(1, n):
            x, y, ww, hh, a = s[i]
            is_noise = (y <= 0 and (y + hh) < h * 0.4) or (y + hh >= h and y > h * 0.6) or (
                        (x <= 0 or x + ww >= w) and (ww < 5 or a < 20))
            if is_noise: mask[l == i] = 255
        c = np.asarray(img).copy()
        c[mask == 255] = 255
        return c

    @staticmethod
    def deslant_img(img):
        def get_img_moments(img):
            _, b = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            return cv.moments(b)
        m = get_img_moments(img)
        if m['mu02'] != 0:
            skew = m['mu11'] / m['mu02']
            M = np.float32([[1, skew, -0.5 * int(img.shape[0]) * skew], [0, 1, 0]])
            img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR,
                                borderValue=255)
        return img

    def extract_atomic_crops(self, line_path):
        raw_img = cv.imread(line_path, cv.IMREAD_GRAYSCALE)
        if raw_img is None: return []
        img = np.asarray(self.deslant_img(raw_img))
        _, binarized = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        binary_raw = (img > threshold_sauvola(img, window_size=61, k=0.1)) * 255
        binary = np.asarray(binary_raw, dtype=np.uint8)
        clean_binary = cv.morphologyEx(binary, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        inv = cv.bitwise_not(clean_binary)
        dilated = cv.dilate(inv, cv.getStructuringElement(cv.MORPH_RECT, (1, 1)), iterations=1)
        cnts, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        boxes = [cv.boundingRect(c) for c in cnts if cv.boundingRect(c)[2] * cv.boundingRect(c)[3] > MIN_WORD_AREA]
        boxes.sort(key=lambda b: b[0])
        if not boxes: return []
        merged = []
        curr = boxes[0]
        for i in range(1, len(boxes)):
            nxt = boxes[i]
            if (nxt[0] - (curr[0] + curr[2])) < 10:
                x = min(curr[0], nxt[0]); y = min(curr[1], nxt[1])
                curr = (x, y, max(curr[0] + curr[2], nxt[0] + nxt[2]) - x, max(curr[1] + curr[3], nxt[1] + nxt[3]) - y)
            else: merged.append(curr); curr = nxt
        merged.append(curr)
        crops = []
        h, w = img.shape
        for (x, y, ww, hh) in merged:
            if ww > w * 0.98: continue
            c = img[max(0, y - 4):min(h, y + hh + 4), max(0, x - 4):min(w, x + ww + 4)]
            crops.append(self.clean_word_crop(c))
        return crops


# TTA
def preprocess_for_crnn(img):
    img = np.asarray(img)

    # Wymuszamy typ na wyniku z copyMakeBorder
    padded = np.asarray(cv.copyMakeBorder(img, 4, 4, 10, 10, cv.BORDER_CONSTANT, value=255))

    # Teraz linter jest pewny, że padded ma atrybut shape
    scale = IMAGE_HEIGHT / int(padded.shape[0])
    new_w = int(padded.shape[1] * scale)
    if new_w > IMAGE_WIDTH: new_w = IMAGE_WIDTH

    # Wymuszamy typ na wyniku z resize
    resized = np.asarray(cv.resize(padded, (new_w, IMAGE_HEIGHT)))

    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized

    img_tensor = torch.from_numpy(canvas).float()
    img_tensor = (img_tensor - 127.5) / 127.5
    return img_tensor.unsqueeze(0).unsqueeze(0)

def score_image_hybrid(img, model_crnn, model_caps, crnn_encoder, caps_mapping, caps_transform):
    t1 = preprocess_for_crnn(img).to(DEVICE)
    h, w = img.shape
    M_l = np.float32([[1, 0, -2], [0, 1, 0]])
    t2 = preprocess_for_crnn(cv.warpAffine(img, M_l, (w, h), borderValue=255)).to(DEVICE)
    M_r = np.float32([[1, 0, 2], [0, 1, 0]])
    t3 = preprocess_for_crnn(cv.warpAffine(img, M_r, (w, h), borderValue=255)).to(DEVICE)

    batch_t = torch.cat([t1, t2, t3], dim=0)
    preds = model_crnn(batch_t)
    probs = torch.exp(preds)
    avg_probs = torch.mean(probs, dim=0, keepdim=True)

    max_p, idxs = torch.max(avg_probs, dim=2)

    # Rzutowanie na Tensor z powodu struktury zwracanej przez torch.max
    decoded_indices = idxs.squeeze(1).cpu().long().tolist()
    confs = max_p.squeeze(1).cpu().tolist()

    final_chars = []; final_confs = []; prev = -1
    img_debug = None

    for k, idx in enumerate(decoded_indices):
        if idx != 0 and idx != prev:
            char = crnn_encoder.get(idx, '')
            conf = confs[k]
            if model_caps and conf < CONFIDENCE_THRESHOLD_CHAR:
                if img_debug is None: img_debug = cv.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                cx = int(k * 4)
                x1 = int(max(0, cx - 14)); x2 = int(min(IMAGE_WIDTH, cx + 14))
                if (x2 - x1) > 5 and x2 <= img_debug.shape[1]:
                    crop = img_debug[:, x1:x2]
                    if crop.size > 0:
                        t_caps = caps_transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
                        probs_c, _ = model_caps(t_caps)
                        conf_c, idx_c = torch.max(probs_c, dim=1)
                        alt_char = caps_mapping.get(idx_c.item(), '?')
                        alt_conf = conf_c.item()
                        if alt_conf > 0.9 and alt_char != char:
                            char = alt_char; conf = alt_conf
            final_chars.append(char); final_confs.append(conf)
        prev = idx
    text = "".join(final_chars)
    avg_conf = sum(final_confs) / len(final_confs) if final_confs else 0.0
    return text, avg_conf, final_confs


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR_LINES):
        print(f"Input dir not found: {INPUT_DIR_LINES}"); exit()

    dir_high = os.path.join(OUTPUT_DIR_LABELED, "high_confidence")
    dir_low = os.path.join(OUTPUT_DIR_LABELED, "review_hard_cases")
    os.makedirs(dir_high, exist_ok=True); os.makedirs(dir_low, exist_ok=True)

    chars = get_char_list()
    crnn_encoder = {i + 1: c for i, c in enumerate(chars)}
    caps_mapping = get_extended_mapping()

    corrector = StrictSpellCorrector(LM_PATH)
    page_solver = PageBeamSearch(corrector, beam_width=5)
    segmentor = LineToWordSegmentor()

    model_crnn = ResNetCRNN(len(chars) + 1).to(DEVICE)
    try:
        model_crnn.load_state_dict(torch.load(CRNN_PATH, map_location=DEVICE), strict=False); print("CRNN Loaded.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"CRNN Error: {e}")
        exit()
    model_crnn.eval()

    model_caps = CapsNet(len(caps_mapping)).to(DEVICE)
    try:
        model_caps.load_state_dict(torch.load(CAPS_PATH, map_location=DEVICE), strict=False)
    except (FileNotFoundError, RuntimeError):
        model_caps = None
    if model_caps: model_caps.eval()

    caps_transform = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE_CHAR), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Grupowanie po stronie
    all_files = []
    for r, d, f in os.walk(INPUT_DIR_LINES):
        for file in f:
            if file.endswith(('.png', '.jpg', '.tif')): all_files.append(os.path.join(r, file))

    all_files.sort()

    def get_page_id(filepath):
        fname = os.path.basename(filepath)
        parts = fname.split('-')
        if len(parts) >= 2: return "-".join(parts[:2])
        return fname


    total_words_saved = 0
    print(f"Preprocesowanie {len(all_files)} linii, grupowanych po stronie.")

    for page_id, page_files_iter in groupby(all_files, key=get_page_id):
        page_files = list(page_files_iter)
        page_buffer = []

        # Exktrakcja słów ze strony
        for [page_id, page_files_iter] in groupby(all_files, key=get_page_id):
            page_files = list(page_files_iter)
            page_buffer = []

            # Exktrakcja słów ze strony
            with torch.no_grad():
                for line_path in tqdm(page_files, leave=False, desc=f"Page {page_id}"):
                    try:
                        words = segmentor.extract_atomic_crops(line_path)
                        if not words: continue
                        base = os.path.splitext(os.path.basename(line_path))[0]

                        accumulator_img = words[0]
                        curr_text, curr_conf, curr_confs = score_image_hybrid(accumulator_img, model_crnn, model_caps,
                                                                              crnn_encoder, caps_mapping,
                                                                              caps_transform)
                        target_h = accumulator_img.shape[0]

                        i = 1
                        while i < len(words):
                            next_img = words[i]
                            if next_img.shape[0] != target_h:
                                scale_h = target_h / next_img.shape[0]
                                new_w_h = int(next_img.shape[1] * scale_h)
                                next_img = cv.resize(next_img, (new_w_h, target_h))

                            spacer = np.full((target_h, 8), 255, dtype=np.uint8)
                            merged_img = np.hstack((accumulator_img, spacer, next_img))
                            merged_text, merged_conf, merged_confs = score_image_hybrid(merged_img, model_crnn,
                                                                                        model_caps,
                                                                                        crnn_encoder, caps_mapping,
                                                                                        caps_transform)


                            def clean_check(t):
                                return t.strip(".,;:!?'\"()[]-")


                            is_merged_valid = len(clean_check(merged_text)) > 1 and corrector.is_valid(
                                clean_check(merged_text))
                            is_curr_valid = len(clean_check(curr_text)) > 1 and corrector.is_valid(
                                clean_check(curr_text))

                            should_merge = False
                            if is_merged_valid and not is_curr_valid:
                                should_merge = True
                            elif is_merged_valid and is_curr_valid:
                                if merged_conf > (curr_conf - 0.15): should_merge = True
                            elif merged_conf > (curr_conf + 0.1):
                                should_merge = True
                            elif len(curr_text) < 2 and merged_conf > 0.6:
                                should_merge = True

                            if should_merge:
                                accumulator_img = merged_img
                                curr_text = merged_text
                                curr_conf = merged_conf
                                curr_confs = merged_confs
                            else:
                                page_buffer.append({
                                    'img': accumulator_img, 'text': curr_text, 'confs': curr_confs, 'score': curr_conf,
                                    'base': base
                                })

                                # Wymuszamy typ na pierwszym wycinku
                                accumulator_img = np.asarray(words[0])
                                curr_text, curr_conf, curr_confs = score_image_hybrid(
                                    accumulator_img, model_crnn, model_caps, crnn_encoder, caps_mapping, caps_transform
                                )

                                # Linter jest pewny, że to ndarray i przypisujemy czysty int
                                target_h = int(accumulator_img.shape[0])

                                i = 1
                                while i < len(words):
                                    # Wymuszamy typ na kolejnych wycinkach
                                    next_img = np.asarray(words[i])

                                    if int(next_img.shape[0]) != target_h:
                                        scale_h = target_h / int(next_img.shape[0])
                                        new_w_h = int(next_img.shape[1] * scale_h)
                                        next_img = np.asarray(cv.resize(next_img, (new_w_h, target_h)))

                                    spacer = np.full((target_h, 8), 255, dtype=np.uint8)
                                    merged_img = np.hstack((accumulator_img, spacer, next_img))
                            i += 1

                        # Ostatnie słowo też
                        if len(curr_text) > 0:
                            page_buffer.append({
                                'img': accumulator_img, 'text': curr_text, 'confs': curr_confs, 'score': curr_conf,
                                'base': base
                            })

                    except (cv.error, RuntimeError, ValueError, IndexError):
                        continue

        if not page_buffer: continue

        corrected_sequence = page_solver.solve_page(page_buffer)

        # Zapis
        for k, packet in enumerate(page_buffer):
            final_text = corrected_sequence[k]
            final_conf = packet['score']

            if len(final_text) > 0:
                safe_label = "".join([c if c.isalnum() else IAM_CODE_MAP.get(c, '') for c in final_text])
                if len(safe_label) > 50: safe_label = safe_label[:50] + "_TRUNC"

                if safe_label:
                    target_dir = dir_high if final_conf >= ACCEPTANCE_THRESHOLD else dir_low
                    fname = f"{final_conf:.2f}___{safe_label}___{packet['base']}_w{k}.png"
                    try:
                        cv.imwrite(os.path.join(target_dir, fname), packet['img'])
                        total_words_saved += 1
                    except (cv.error, OSError): pass


    print(f"\nDone. Total words extracted: {total_words_saved}")
