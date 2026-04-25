import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")
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
import difflib
from collections import defaultdict
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your RAW word images (IAM Words dataset)
INPUT_WORDS_DIR = r"C:\OCR\iam_words\iam_words\words"

# Where to save the cut-out letters (Hard Negatives)
OUTPUT_CHARS_DIR = r"C:\OCR\archive\iam_words\letters_extracted"
OUTPUT_CONTEXT_DIR = r"C:\OCR\archive\iam_words\words_context"  # New: Save full word for context

# Path to your TRAINED CRNN model
MODEL_PATH = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"

# Output size for the letter crops (CapsNet target)
TARGET_SIZE = (40, 40)

# CRNN Config (Must match training)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512
MODEL_DROPOUT = 0.25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SMART CROP SETTINGS
SCALE_FACTOR = 4  # 512 width image -> 128 width feature map (ResNet downsamples by 4)
CONTEXT_PADDING_PIXELS = 4  # Pixels to pad the crop
MIN_CROP_WIDTH = 5  # Minimum required width for a crop

BATCH_SIZE = 16

# Mappings (for filename parsing and saving)
REVERSE_PUNCTUATION_MAP = {
    '#D': '.', '#C': ',', '#A': "'", '#E': '!', '#H': '-',
    '#B': '(', '#K': ')', '#S': ';', '#L': ':', '#U': '"', '#Q': '?'
}
FILENAME_MAP = {
    'dot': '.', 'comma': ',', 'question': '?', 'exclamation': '!',
    'colon': ':', 'semicolon': ';', 'quote': '"', 'apostrophe': "'",
    'lparen': '(', 'rparen': ')', 'hyphen': '-', 'space': ' '
}
os.makedirs(OUTPUT_CHARS_DIR, exist_ok=True)
os.makedirs(OUTPUT_CONTEXT_DIR, exist_ok=True)


# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class LocalizationNetwork(nn.Module):
    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(self.I_channel_num, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
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
        ctrl_pts_y_top = -1.0 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = 1.0 * np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        self.target_ctrl_pts = torch.from_numpy(np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)).float()
        N = self.F
        target_ctrl_pts_1 = self.target_ctrl_pts.unsqueeze(1)
        target_ctrl_pts_2 = self.target_ctrl_pts.unsqueeze(0)
        dist = torch.norm(target_ctrl_pts_1 - target_ctrl_pts_2, p=2, dim=2)
        dist = dist + 1e-6
        K = dist.pow(2) * torch.log(dist.pow(2))
        P = torch.cat([torch.ones(N, 1), self.target_ctrl_pts], dim=1)
        L = torch.zeros(N + 3, N + 3)
        L[:N, :N], L[:N, N:], L[N:, :N] = K, P, P.t()
        L = L + torch.eye(N + 3) * 1e-6
        self.inv_L = torch.inverse(L)

    def forward(self, source_ctrl_pts):
        batch_size = source_ctrl_pts.size(0)
        device = source_ctrl_pts.device
        Y = torch.cat([source_ctrl_pts, torch.zeros(batch_size, 3, 2).to(device)], dim=1)
        weights = torch.matmul(self.inv_L.expand(batch_size, -1, -1).to(device), Y)
        grid_h, grid_w = torch.meshgrid(torch.linspace(-1, 1, self.I_r_height, device=device),
                                        torch.linspace(-1, 1, self.I_r_width, device=device))
        grid_h = grid_h.unsqueeze(2)
        grid_w = grid_w.unsqueeze(2)
        ones = torch.ones_like(grid_h)
        P_grid = torch.cat([ones, grid_w, grid_h], dim=2).view(-1, 3).unsqueeze(0).expand(batch_size, -1, -1)
        target_pts = self.target_ctrl_pts.to(device).unsqueeze(0).unsqueeze(0)
        pixel_pts = torch.cat([grid_w, grid_h], dim=2).view(1, -1, 1, 2)
        dist = torch.norm(pixel_pts - target_pts, p=2, dim=3)
        dist = dist + 1e-6
        K_grid = (dist.pow(2) * torch.log(dist.pow(2))).expand(batch_size, -1, -1)
        L_grid = torch.cat([K_grid, P_grid], dim=2)
        grid = torch.matmul(L_grid, weights)
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
        self.tps_stn = TPS_SpatialTransformerNetwork(F=20, I_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                     I_r_size=(IMAGE_HEIGHT, IMAGE_WIDTH), I_channel_num=1)
        resnet = models.resnet18(weights=None)
        self.resnet_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(
            self.resnet_conv1, resnet.bn1, nn.ReLU(True), resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Dropout(0.25))
        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=0.25)
        self.output = nn.Linear(256 * 2, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        B, C, H, W = x.size()
        if C == 3: x = x[:, 0:1, :, :]
        rectified_x = self.tps_stn(x)
        vertical_map = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        x_combined = torch.cat([rectified_x, vertical_map], dim=1)
        features = self.cnn(x_combined)
        b, c, h, w = features.size()
        features = features.permute(3, 0, 1, 2).reshape(w, b, -1)
        features = self.projection(features)
        rnn_out, _ = self.lstm(features)
        logits = self.output(rnn_out)
        return (logits / self.temperature).log_softmax(2)


# ==========================================
# 3. EXTRACTION LOGIC HELPERS
# ==========================================

def get_emnist_char_list_byclass() -> list:
    char_list = []
    for i in range(10): char_list.append(chr(48 + i))
    for i in range(26): char_list.append(chr(65 + i))
    for i in range(26): char_list.append(chr(97 + i))
    return char_list


def get_label_from_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]
    label = re.sub(r'_\d+$', '', name_no_ext)
    for code, char in REVERSE_PUNCTUATION_MAP.items():
        if code in label: label = label.replace(code, char)
    label = FILENAME_MAP.get(label, label)
    return label


def decode_greedy(log_probs, encoder):
    probs = torch.exp(log_probs)
    indices = torch.argmax(probs, dim=2)
    decoded_texts = []
    for sample_idx in range(indices.size(1)):
        sample_indices = indices[:, sample_idx].tolist()
        decoded = []
        prev_idx = -1
        for idx in sample_indices:
            if idx != 0 and idx != prev_idx:
                decoded.append(encoder.num_to_char.get(idx, ''))
            prev_idx = idx
        decoded_texts.append("".join(decoded))
    return decoded_texts


def get_ctc_segments(log_probs_sample, encoder):
    """
    Refined: Returns segments with start_t (first non-blank) and end_t (last non-blank).
    """
    probs = torch.exp(log_probs_sample)
    max_probs, indices = torch.max(probs, dim=1)

    segments = []
    current_char = None
    start_t = -1

    for t, idx in enumerate(indices.tolist()):
        idx = int(idx)
        char = encoder.num_to_char.get(idx, '?')

        if idx != 0:
            if char != current_char:
                # End previous segment if char changed
                if current_char is not None:
                    segments.append({
                        'char': current_char,
                        'start_t': start_t,
                        'end_t': t - 1,
                        'conf': max_probs[start_t].item()
                    })

                # Start new segment
                current_char = char
                start_t = t

        elif idx == 0 and current_char is not None:
            # End segment when blank is hit
            segments.append({
                'char': current_char,
                'start_t': start_t,
                'end_t': t - 1,
                'conf': max_probs[start_t].item()
            })
            current_char = None
            start_t = -1

    # Handle end of sequence if last segment was non-blank
    if current_char is not None:
        segments.append({
            'char': current_char,
            'start_t': start_t,
            'end_t': len(indices) - 1,
            'conf': max_probs[start_t].item()
        })

    return segments


def load_word_for_dataset(image_path, transform):
    try:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None: return None, None

        # We need the original image resized to (64x512) for pixel-space cropping.
        word_img_np = cv.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_LINEAR)

        pil_img = Image.fromarray(word_img_np)
        tensor = transform(pil_img)

        return tensor, word_img_np
    except Exception as e:
        return None, None


def align_prediction_to_ground_truth(gt_text, pred_text):
    """
    Maps the PREDICTED characters to the REAL characters using difflib SequenceMatcher.
    Returns: A list where index j corresponds to the j-th predicted character.
    """
    matcher = difflib.SequenceMatcher(None, gt_text, pred_text)
    alignment = [None] * len(pred_text)

    gt_idx = 0
    pred_idx = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        len_gt = i2 - i1
        len_pred = j2 - j1

        if tag == 'equal':
            for k in range(len_pred):
                alignment[pred_idx + k] = gt_text[gt_idx + k]
            pred_idx += len_pred
            gt_idx += len_gt

        elif tag == 'replace':
            for k in range(len_pred):
                if k < len_gt:
                    alignment[pred_idx + k] = gt_text[gt_idx + k]
                else:
                    # Model predicted extra chars during replacement
                    alignment[pred_idx + k] = 'HALLUCINATION'
            pred_idx += len_pred
            gt_idx += len_gt

        elif tag == 'insert':
            # Model hallucinated characters that don't map to GT
            for k in range(len_pred):
                alignment[pred_idx + k] = 'HALLUCINATION'
            pred_idx += len_pred

        elif tag == 'delete':
            # GT had a char model missed (no predicted char to assign label to)
            gt_idx += len_gt

    return alignment


# ==========================================
# 4. DATASET AND DATALOADER
# ==========================================

class HTREncoder:
    def __init__(self, char_list):
        if ' ' in char_list: char_list.remove(' ')
        self.char_list = sorted(list(set(char_list)))
        self.char_list.append(' ')  # Blank last
        self.char_to_num = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        self.blank_index = 0


class IAMWordDataset(torch.utils.data.Dataset):
    def __init__(self, all_files, transform, char_list):
        self.all_files = all_files
        self.transform = transform
        self.char_list = char_list
        self.filtered_files = []
        for p in self.all_files:
            try:
                true_label = get_label_from_filename(os.path.basename(p))
                # Only keep files where the GT label is in our defined character set
                if all(c in self.char_list for c in true_label):
                    self.filtered_files.append(p)
            except:
                pass

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        img_path = self.filtered_files[idx]
        true_label = get_label_from_filename(os.path.basename(img_path))
        word_tensor, word_img_np = load_word_for_dataset(img_path, self.transform)
        if word_tensor is None: return None
        return word_tensor, true_label, word_img_np, img_path


def collate_fn_ignore_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return None
    word_tensors, true_labels, word_img_nps, img_paths = zip(*batch)
    word_tensors = torch.stack(word_tensors, 0)
    return word_tensors, true_labels, word_img_nps, img_paths


# ==========================================
# 5. MAIN EXTRACTION LOGIC
# ==========================================

def extract_crops():
    # 1. Setup Alphabet and Model
    print(f"Extracting Hard/Context Characters")

    all_files = []
    for root, _, files in os.walk(INPUT_WORDS_DIR):
        for f in files:
            if f.endswith(('.png', '.jpg')): all_files.append(os.path.join(root, f))

    emnist_chars = get_emnist_char_list_byclass()
    final_char_set = set(emnist_chars)
    user_punct_list = ['.', ',', "'", '-', '!', '?', ' ']
    for symbol in user_punct_list:
        if symbol not in final_char_set: final_char_set.add(symbol)

    char_list = sorted(list(final_char_set))
    if ' ' in char_list: char_list.remove(' ')
    char_list.append(' ')  # Blank last

    encoder = HTREncoder(char_list)
    model = ResNetCRNN(num_classes=len(char_list) + 1).to(DEVICE)

    # 2. Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'model_state' in ckpt: ckpt = ckpt['model_state']

            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in ckpt.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            if not pretrained_dict: raise ValueError("No matching weights found in checkpoint.")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print(f"Weights loaded successfully. Num classes: {len(char_list) + 1}")
            model.eval()
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Checkpoint not found at {MODEL_PATH}")
        return

    # 3. Setup Data
    word_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = IAMWordDataset(all_files, word_transform, char_list)
    num_workers = min(os.cpu_count() or 4, 8 if DEVICE.type == 'cuda' else 4)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_ignore_none
    )

    print(f"Processing {len(dataset)} words.")

    count_errors_words = 0
    count_saved_chars = 0
    processed_word_files = set()  # To avoid saving context duplicates

    # 4. Inference Loop
    # Corrected tqdm initialization for immediate display
    total_steps = len(data_loader)
    for batch_data in tqdm(data_loader, total=total_steps, desc="Extracting Hard Chars", unit="batch", ncols=100):
        if batch_data is None: continue

        word_tensors, true_labels, word_img_nps, img_paths = batch_data
        word_tensors = word_tensors.to(DEVICE)

        with torch.no_grad():
            preds = model(word_tensors)  # [Time, Batch, Classes]

        pred_texts = decode_greedy(preds.cpu(), encoder)

        # Iterate over batch
        for i in range(len(true_labels)):
            true_label = true_labels[i]
            pred_text = pred_texts[i]
            img_path = img_paths[i]

            # We only proceed if the overall word prediction was wrong
            if pred_text == true_label:
                continue

            count_errors_words += 1

            # Get segmentation details
            log_probs_sample = preds[:, i, :].cpu()
            segments = get_ctc_segments(log_probs_sample, encoder)  # Now has start_t/end_t

            # Map Predicted Characters to Real GT Characters
            real_chars_map = align_prediction_to_ground_truth(true_label, pred_text)

            current_img_np = word_img_nps[i]
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            word_saved_flag = False

            # Iterate through the CTC predicted segments (crops)
            for j, seg in enumerate(segments):
                pred_char = seg['char']
                real_char = real_chars_map[j] if j < len(real_chars_map) else None

                # CRITICAL: Skip invalid labels
                if real_char in [None, 'HALLUCINATION', 'ERR']:
                    continue

                # --- EXTRACTION CRITERIA (Capture Ambiguity) ---
                is_error = (real_char != pred_char)
                is_low_conf = (seg['conf'] < 0.90)

                if not (is_error or is_low_conf):
                    continue

                # --- CROP LOGIC (Using precise CTC bounds) ---

                start_x_t = seg['start_t'] * SCALE_FACTOR
                end_x_t = (seg['end_t'] + 1) * SCALE_FACTOR

                start_x = max(0, int(start_x_t - CONTEXT_PADDING_PIXELS))
                end_x = min(IMAGE_WIDTH, int(end_x_t + CONTEXT_PADDING_PIXELS))

                if end_x - start_x < MIN_CROP_WIDTH: continue

                raw_crop = current_img_np[:, start_x:end_x]

                # --- SAVE LOGIC ---

                # 1. Filename sanitization remains unchanged
                folder_char = str(real_char)
                punct_map = {'.': 'dot', ',': 'comma', ':': 'colon', ';': 'semicolon',
                             '?': 'question', '!': 'exclamation', '"': 'quote', "'": 'apostrophe',
                             '(': 'lparen', ')': 'rparen', '-': 'hyphen', ' ': 'space'}

                safe_folder = punct_map.get(folder_char, folder_char)
                if folder_char.isupper() and folder_char.isalpha():
                    safe_folder = f"upper_{folder_char}"

                save_dir = os.path.join(OUTPUT_CHARS_DIR, safe_folder)
                os.makedirs(save_dir, exist_ok=True)

                # 2. Final Crop Preparation (FIXED LOGIC)
                h, w = raw_crop.shape
                if w > 0 and h > 0:

                    # Determine padding size: Use max(h, w) for a clean square canvas
                    # This ensures the character is NOT distorted by stretching, only shrunk proportionally.
                    pad_size = max(h, w)
                    pad_img = np.full((pad_size, pad_size), 255, dtype=np.uint8)

                    # Centering coordinates
                    start_x_pad = (pad_size - w) // 2
                    start_y_pad = (pad_size - h) // 2

                    # Place the raw crop onto the center of the square canvas
                    pad_img[start_y_pad:start_y_pad + h, start_x_pad:start_x_pad + w] = raw_crop

                    # Final Resize to CapsNet Target (40x40)
                    final_crop = cv.resize(pad_img, TARGET_SIZE, interpolation=cv.INTER_AREA)

                    # 3. Save files
                    safe_pred = punct_map.get(pred_char, pred_char)
                    filename = f"{base_filename}_R{safe_folder}_P{safe_pred}_C{int(seg['conf'] * 100)}_{count_saved_chars}.png"

                    try:
                        cv.imwrite(os.path.join(save_dir, filename), final_crop)
                        count_saved_chars += 1
                        word_saved_flag = True
                    except Exception as e:
                        pass

            # 5. Save the whole context image if any char was saved
            if word_saved_flag and img_path not in processed_word_files:
                try:
                    context_filename = f"{base_filename}_GT_WORD_{len(processed_word_files)}.png"
                    cv.imwrite(os.path.join(OUTPUT_CONTEXT_DIR, context_filename), current_img_np)
                    processed_word_files.add(img_path)
                except Exception:
                    pass

    print(f"\n--- DONE ---")
    print(f"Total Words with Errors processed: {count_errors_words}")
    print(f"Total Hard/Context Characters saved: {count_saved_chars}")
    print(f"Saved crops to: {OUTPUT_CHARS_DIR}")
    print(f"Saved context words to: {OUTPUT_CONTEXT_DIR}")


if __name__ == "__main__":
    extract_crops()
