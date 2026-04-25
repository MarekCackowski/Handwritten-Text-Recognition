import datetime
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_WORDS_DIR = r"C:\OCR\iam_words\iam_words"
# Root output directory for the CapsNet dataset
OUTPUT_ROOT = r"C:\OCR\archive\iam_words\letters_extracted"
MODEL_PATH = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"

TARGET_SIZE = (64, 64)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hardware and Smart Crop Settings
SCALE_FACTOR = 4  # 512 width / 128 feature width
BATCH_SIZE = 16  # Optimized for user hardware

# Mappings for filename parsing and safe folder creation
REVERSE_PUNCTUATION_MAP = {
    '#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-',
    '#B': '(', '#K': ')', '#S': ';', '#L': ':', '#U': '"', '#Q': '?'
}
FILENAME_MAP = {
    'dot': '.', 'comma': ',', 'question': '?', 'exclamation': '!',
    'colon': ':', 'semicolon': ';', 'quote': '"', 'apostrophe': "'",
    'lparen': '(', 'rparen': ')', 'hyphen': '-', 'space': ' '
}
PUNCT_FOLDER_MAP = {
    '.': 'dot', ',': 'comma', ':': 'colon', ';': 'semicolon',
    '?': 'question', '!': 'exclamation', '"': 'quote', "'": 'apostrophe',
    '(': 'lparen', ')': 'rparen', '-': 'hyphen'
}

def safe_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return list(zip(*batch))

# ==========================================
# 2. MODEL DEFINITION (STN + ResNetCRNN)
# ==========================================

class LocalizationNetwork(nn.Module):
    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.conv(x).view(batch_size, -1)
        theta = self.localization_fc2(self.localization_fc1(features))
        return theta.view(batch_size, self.F, 2)


class GridGenerator(nn.Module):
    def __init__(self, F, I_r_size):
        super(GridGenerator, self).__init__()
        self.F = F
        self.I_r_height, self.I_r_width = I_r_size
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, -1.0 * np.ones(int(F / 2))], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, 1.0 * np.ones(int(F / 2))], axis=1)
        self.target_ctrl_pts = torch.from_numpy(np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)).float()
        N = self.F
        dist = torch.norm(self.target_ctrl_pts.unsqueeze(1) - self.target_ctrl_pts.unsqueeze(0), p=2, dim=2)
        K = dist.pow(2) * torch.log(dist.pow(2) + 1e-6)
        P = torch.cat([torch.ones(N, 1), self.target_ctrl_pts], dim=1)
        L = torch.zeros(N + 3, N + 3)
        L[:N, :N], L[:N, N:], L[N:, :N] = K, P, P.t()
        self.inv_L = torch.inverse(L + torch.eye(N + 3) * 1e-6)

    def forward(self, source_ctrl_pts):
        batch_size = source_ctrl_pts.size(0)
        device = source_ctrl_pts.device
        Y = torch.cat([source_ctrl_pts, torch.zeros(batch_size, 3, 2).to(device)], dim=1)
        weights = torch.matmul(self.inv_L.expand(batch_size, -1, -1).to(device), Y)
        grid_h, grid_w = torch.meshgrid(torch.linspace(-1, 1, self.I_r_height, device=device),
                                        torch.linspace(-1, 1, self.I_r_width, device=device))
        P_grid = torch.cat([torch.ones_like(grid_h).unsqueeze(2), grid_w.unsqueeze(2), grid_h.unsqueeze(2)],
                           dim=2).view(-1, 3).unsqueeze(0).expand(batch_size, -1, -1)
        target_pts = self.target_ctrl_pts.to(device).unsqueeze(0).unsqueeze(0)
        pixel_pts = torch.cat([grid_w.unsqueeze(2), grid_h.unsqueeze(2)], dim=2).view(1, -1, 1, 2)
        dist = torch.norm(pixel_pts - target_pts, p=2, dim=3)
        K_grid = (dist.pow(2) * torch.log(dist.pow(2) + 1e-6)).expand(batch_size, -1, -1)
        grid = torch.matmul(torch.cat([K_grid, P_grid], dim=2), weights)
        return torch.clamp(grid.view(batch_size, self.I_r_height, self.I_r_width, 2), -1.5, 1.5)


class TPS_SpatialTransformerNetwork(nn.Module):
    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.LocalizationNetwork = LocalizationNetwork(F, I_channel_num)
        self.GridGenerator = GridGenerator(F, I_r_size)

    def forward(self, x):
        theta = self.LocalizationNetwork(x)
        generated_grid = self.GridGenerator(theta)
        return func.grid_sample(x, generated_grid, align_corners=True)


class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=None)

        # Zmiana na 1 kanał wejściowy i kernel 5 (zgodnie z błędem Size Mismatch)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3, bias=False)

        # Powrót do oryginalnych stride'ów dla zachowania 2560 cech wejściowych
        resnet.layer2[0].conv1.stride = (2, 2)
        resnet.layer2[0].downsample[0].stride = (2, 2)
        resnet.layer3[0].conv1.stride = (2, 2)
        resnet.layer3[0].downsample[0].stride = (2, 2)
        resnet.layer4[0].conv1.stride = (1, 2)
        resnet.layer4[0].downsample[0].stride = (1, 2)

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # Zgodnie z błędem: 2560 wejść do projekcji
        self.projection = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=0.25)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        features = features.permute(3, 0, 1, 2).reshape(w, b, -1)
        features = self.projection(features)
        rnn_out, _ = self.lstm(features)
        return self.output(rnn_out).log_softmax(2)

# ==========================================
# 3. EXTRACTION HELPERS & ENCODER
# ==========================================

class HTREncoder:
    def __init__(self, char_list):
        self.char_list = sorted(list(set(char_list))) + [' ']
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}


def get_label_from_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    label = re.sub(r'_\d+$', '', name_no_ext)
    for code, char in REVERSE_PUNCTUATION_MAP.items():
        if code in label: label = label.replace(code, char)
    return FILENAME_MAP.get(label, label)


def get_safe_segments(log_probs_sample, encoder, target_length):
    probs = torch.exp(log_probs_sample)
    max_probs, indices = torch.max(probs, dim=1)

    segments = []
    current_char = None
    start_t = -1

    for t, idx in enumerate(indices.tolist()):
        char = encoder.num_to_char.get(idx, None)
        if idx != 0:  # Nie blank
            if char != current_char:
                if current_char:
                    segments.append({
                        'char': current_char,
                        't_start': start_t,
                        't_end': t - 1,
                        'peak_t': start_t + torch.argmax(max_probs[start_t:t]).item()
                    })
                start_t, current_char = t, char
        elif current_char:  # Blank po literze
            segments.append({
                'char': current_char,
                't_start': start_t,
                't_end': t - 1,
                'peak_t': start_t + torch.argmax(max_probs[start_t:t]).item()
            })
            current_char, start_t = None, -1

    return segments if len(segments) == target_length else []


class IAMWordDataset(torch.utils.data.Dataset):
    def __init__(self, all_files, transform, char_list):
        self.all_files = all_files
        self.transform = transform
        self.char_list = char_list
        self.filtered_files = [f for f in all_files if
                               all(c in char_list for c in get_label_from_filename(os.path.basename(f)))]

    def __len__(self): return len(self.filtered_files)

    def __getitem__(self, idx):
        path = self.filtered_files[idx]
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None: return None
        img_res = cv.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return self.transform(Image.fromarray(img_res)), get_label_from_filename(os.path.basename(path)), img_res, path


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return None
    return list(zip(*batch))


# ==========================================
# 4. MAIN DUAL-STREAM EXTRACTION ENGINE
# ==========================================

def run_extraction():
    if os.path.exists(OUTPUT_ROOT):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cleaning the folder: {OUTPUT_ROOT}.")
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Dual-Stream High-Precision Extraction.")

    # 1. ALPHABET & MODEL SETUP (74 classes)
    alphabet_str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,'-!? \"()/:;"
    alphabet = sorted(list(set(alphabet_str)))
    encoder = HTREncoder(alphabet)

    model = ResNetCRNN(num_classes=74).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        print("Model załadowany pomyślnie.")
    else:
        print("CRNN path invalid.")
        return

    all_files = [os.path.join(r, f) for r, _, fs in os.walk(INPUT_WORDS_DIR) for f in fs if f.endswith('.png')]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = IAMWordDataset(all_files, transform, alphabet)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=safe_collate_fn
    )

    stats = {"Easy": 0, "Hard": 0}

    # 2. PROCESSING LOOP
    for batch in tqdm(loader, desc="Processing", disable=True):
        word_tensors, labels, word_imgs, paths = batch
        with torch.no_grad():
            preds = model(torch.stack(word_tensors).to(DEVICE)).permute(1, 0, 2).cpu()

        dynamic_scale = IMAGE_WIDTH / preds.shape[1]

        for i in range(len(labels)):
            segments = get_safe_segments(preds[i], encoder, len(labels[i]))
            if not segments: continue

            img_np = word_imgs[i]
            h_word, w_word = img_np.shape

            for j, seg in enumerate(segments):
                real_char = labels[i][j]
                if real_char == ' ': continue

                # Logic for stream sorting (Easy vs Hard)
                is_correct = (seg['char'].lower() == real_char.lower())
                activation_width = seg['t_end'] - seg['t_start']
                stream = "Easy" if is_correct else "Hard"

                # Map peak to pixel center
                center_x = int((seg['peak_t'] + 0.5) * dynamic_scale)

                # IMPROVED SEARCH MARGIN: Extra space for wide letters
                search_margin = 80 if real_char.lower() in ['m', 'w'] else 50
                x1_search = max(0, center_x - search_margin)
                x2_search = min(w_word, center_x + search_margin)
                raw_slice = img_np[:, x1_search:x2_search]

                # Inversion and vertical projection
                analysis_slice = 255 - raw_slice if np.mean(raw_slice) > 127 else raw_slice
                v_proj = np.sum(analysis_slice, axis=0)

                # Highly sensitive ink seeker (1.5% threshold for faint 'm' legs)
                ink_indices = np.where(v_proj > (np.max(v_proj) * 0.015))[0]

                if len(ink_indices) > 0:
                    # Find the continuous ink cluster closest to the predicted center
                    local_center = center_x - x1_search

                    # Logic to find the boundaries of the specific letter cluster
                    # We start at local_center and expand left/right until we hit a gap
                    # This prevents the 'cl' issue while keeping the whole 'm'
                    ink_left = ink_indices.min()
                    ink_right = ink_indices.max()

                    # Buffer: Wide letters need more room to not feel "cramped"
                    buffer = 45 if real_char.lower() in ['m', 'w'] else 35
                    width_needed = (ink_right - ink_left) + buffer

                    final_center_x = x1_search + (ink_left + ink_right) // 2
                    final_x1 = max(0, int(final_center_x - width_needed // 2))
                    final_x2 = min(w_word, int(final_center_x + width_needed // 2))
                else:
                    fallback = 40 if real_char.lower() in ['m', 'w'] else 30
                    final_x1, final_x2 = center_x - fallback, center_x + fallback

                crop = img_np[:, final_x1:final_x2]

                # 3. PROCESSING AND SCALING
                if crop.size == 0 or crop.shape[1] < 2: continue
                if np.mean(crop) > 127: crop = 255 - crop
                _, crop = cv.threshold(crop, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                hc, wc = crop.shape
                # Scale height to 75% for wide chars to give them horizontal room
                h_target_pct = 0.70 if real_char.lower() in ['m', 'w'] else 0.80
                scale = (TARGET_SIZE[1] * h_target_pct) / hc
                new_h, new_w = int(hc * scale), int(wc * scale)

                # Ensure width does not exceed canvas
                if new_w > TARGET_SIZE[0] * 0.95:
                    scale = (TARGET_SIZE[0] * 0.95) / wc
                    new_w, new_h = int(wc * scale), int(hc * scale)

                char_img = cv.resize(crop, (max(1, new_w), max(1, new_h)), interpolation=cv.INTER_AREA)

                # Final 64x64 canvas
                final_canvas = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
                y_off = (TARGET_SIZE[1] - char_img.shape[0]) // 2
                x_off = (TARGET_SIZE[0] - char_img.shape[1]) // 2
                final_canvas[y_off:y_off + char_img.shape[0], x_off:x_off + char_img.shape[1]] = char_img

                num_labels, labels_im, stats_cc, centroids = cv.connectedComponentsWithStats(final_canvas)

                # Środek płótna (zazwyczaj 32, 32)
                mid_y, mid_x = TARGET_SIZE[1] // 2, TARGET_SIZE[0] // 2
                center_label = labels_im[mid_y, mid_x]

                # Jeśli dokładnie w punkcie (32,32) jest pusto, szukamy klastra w promieniu 10px
                if center_label == 0:
                    for r in range(1, 10):
                        roi = labels_im[max(0, mid_y - r):min(64, mid_y + r), max(0, mid_x - r):min(64, mid_x + r)]
                        if np.any(roi > 0):
                            center_label = roi[roi > 0][0]
                            break

                # Jeśli znaleźliśmy centralną literę, usuwamy wszystko co nie jest nią
                if center_label > 0:
                    # Pobieramy zakres poziomy (X) centralnego elementu (np. laseczki 'i')
                    x_min = stats_cc[center_label, cv.CC_STAT_LEFT]
                    x_max = x_min + stats_cc[center_label, cv.CC_STAT_WIDTH]

                    # Czyścimy płótno selektywnie
                    for label_idx in range(1, num_labels):
                        l_left = stats_cc[label_idx, cv.CC_STAT_LEFT]
                        l_right = l_left + stats_cc[label_idx, cv.CC_STAT_WIDTH]

                        # Logika: Zostawiamy klastry, które są w tym samym pionowym pasie, co środek
                        # To pozwala zachować kropkę nad 'i' (label_idx != center_label, ale ten sam X)
                        if l_right < x_min or l_left > x_max:
                            final_canvas[labels_im == label_idx] = 0

                # 4. SAVE
                folder_char = PUNCT_FOLDER_MAP.get(real_char, real_char)
                if real_char.isupper() and real_char.isalpha(): folder_char = f"upper_{real_char}"

                out_path = os.path.join(OUTPUT_ROOT, stream, folder_char)
                os.makedirs(out_path, exist_ok=True)
                cv.imwrite(os.path.join(out_path, f"{stream}_{stats[stream]}.png"), final_canvas)
                stats[stream] += 1

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] EXTRACTION FINISHED")
    print(f"Easy Samples (Training base): {stats['Easy']}")
    print(f"Hard Samples (For Refinement): {stats['Hard']}")


if __name__ == "__main__":
    run_extraction()
