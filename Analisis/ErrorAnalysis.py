import os
import json
import string
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2 as cv
import h5py
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from pyctcdecode import build_ctcdecoder
import difflib
from collections import Counter
from tqdm import tqdm
import re
from itertools import groupby

# --- SAFE IMPORTS ---
try:
    import kenlm
except ImportError:
    kenlm = None
try:
    import enchant
except ImportError:
    enchant = None
try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
BATCH_SIZE = 16
WORKERS = 4
MODEL_DROPOUT = 0.25
H5_DATABASE_PATH = "ocr_dataset_binary.h5"
CHECKPOINT_PATH = r"C:\OCR\HandwrittenTextRecognition\output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"
LM_PATH = r"C:\OCR\HandwrittenTextRecognition\output_data\english_3gram.binary"
VOCAB_CACHE_FILE = "vocab_cache.json"
OUTPUT_DIR = "output_data/visual_debug_output"
# CRITICAL: Corrections are FORBIDDEN if confidence is higher than this
LOW_CONFIDENCE_CHAR_LIMIT = 0.80  # Used by Corrector's Gatekeeper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
# 2. HELPER: PER-CHARACTER CONFIDENCE
# =========================================================================
def calculate_char_confidences(preds):
    probs = preds.cpu().exp()
    max_probs, max_indices = torch.max(probs, dim=2)
    max_probs = max_probs.permute(1, 0)
    max_indices = max_indices.permute(1, 0)

    batch_conf_lists = []
    for i in range(max_probs.size(0)):
        indices = max_indices[i]
        probs_seq = max_probs[i]
        conf_list = []
        for t in range(len(indices)):
            if indices[t] != 0:
                conf_list.append(probs_seq[t].item())
        batch_conf_lists.append(conf_list)
    return batch_conf_lists


# =========================================================================
# 3. STRICT SPELL CORRECTOR (Production Logic Replica)
# =========================================================================
class StrictSpellCorrector:
    def __init__(self, lm_path=None):
        self.lm = None
        self.pyspell = None
        self.d = None
        if kenlm and lm_path and os.path.exists(lm_path):
            try:
                self.lm = kenlm.Model(lm_path)
            except:
                pass
        if SpellChecker: self.pyspell = SpellChecker()
        if enchant:
            try:
                self.d = enchant.Dict("en_US")
            except:
                pass
        self.contraction_fixer = re.compile(r"([a-zA-Z])\s+(s|t|d|ll|m|re|ve)\b", re.IGNORECASE)

    def is_valid(self, word):
        if not word: return False
        if any(c.isdigit() for c in word): return True
        if len(word) == 1: return word.lower() in {'a', 'i'}
        if self.d: return self.d.check(word)
        if self.pyspell: return word.lower() in self.pyspell.known([word.lower()])
        return False

    @staticmethod
    def fix_punctuation(text):
        text = re.sub(r"([a-zA-Z]),([a-zA-Z])", r"\1'\2", text)
        match = re.search(r"([a-zA-RT-Z])'$", text)
        if match: text = text[:-1] + ","
        return text

    @staticmethod
    def is_candidate_strictly_valid(original, candidate, confidences):
        visual_sim = difflib.SequenceMatcher(None, original, candidate).ratio()

        if len(original) == len(candidate):
            limit = min(len(confidences), len(original))

            for i in range(limit):
                if original[i] != candidate[i]:
                    if confidences[i] >= LOW_CONFIDENCE_CHAR_LIMIT: return False

            if visual_sim > 0.75: return True
            return False

        else:
            # Length mismatch (e.g. rn -> m)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            if avg_conf < 0.70 < visual_sim: return True
            return False

    def get_valid_candidates(self, word, confs):
        """ Used by the Beam Search Solver. """
        word = self.fix_punctuation(word)
        candidates = {(word, 0.0)}

        sugs = []
        if not self.is_valid(word) or len(word) < 4:
            if self.d:
                sugs = self.d.suggest(word)
            elif self.pyspell:
                sugs = self.pyspell.candidates(word)

        if not sugs: return list(candidates)

        for s in list(sugs)[:5]:
            if s == word: continue

            if self.is_valid(s):
                if self.is_candidate_strictly_valid(word, s, confs):
                    candidates.add((s, 2.0))

        return list(candidates)


class AnalysisPageBeamSearch:
    """ Simulates the PageBeamSearch logic for error analysis. """

    def __init__(self, corrector, beam_width=5):
        self.corrector = corrector
        self.lm = corrector.lm
        self.beam_width = beam_width

    def solve_page(self, page_packets):
        """ Returns the best sequence of corrected words for the entire page. """
        beam = [([], 0.0)]

        for packet in page_packets:
            word = packet['text']
            confs = packet['confs']

            candidates = self.corrector.get_valid_candidates(word, confs)
            new_beam = []

            for (sent, score) in beam:
                context = ""
                if len(sent) > 0: context = sent[-1]
                if len(sent) > 1: context = sent[-2] + " " + sent[-1]
                if not context: context = "<s>"

                for (cand, bonus) in candidates:
                    lm_score = 0.0
                    if self.lm:
                        query = f"{context} {cand}"
                        lm_score = self.lm.score(query)

                    new_score = score + lm_score + bonus
                    new_beam.append((sent + [cand], new_score))

            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]

        if not beam: return [p['text'] for p in page_packets]
        return beam[0][0]


# =========================================================================
# 4. CLASS DEFINITIONS (DATASET/MODEL)
# =========================================================================

class HTRCharEncoder:
    def __init__(self, char_list):
        self.char_list = sorted(list(set(char_list)))
        self.char_to_num = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}

    def encode(self, text): return [self.char_to_num[c] for c in text if c in self.char_to_num]

    def decode(self, indices): return "".join([self.num_to_char.get(i, '') for i in indices if i != 0])

    def get_num_classes(self): return len(self.char_list) + 1


class ResNetCRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)
        self.cnn = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                                 resnet.layer3, resnet.layer4)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(MODEL_DROPOUT))
        self.lstm = nn.LSTM(256, 256, 2, bidirectional=True, batch_first=False, dropout=MODEL_DROPOUT)
        self.output = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        if x.size(1) == 1: x = x.repeat(1, 2, 1, 1)
        f = self.cnn(x)
        b, c, h, w = f.size()
        f = f.permute(3, 0, 1, 2).reshape(w, b, -1)
        f = self.projection(f)
        rnn_out, _ = self.lstm(f)
        return self.output(rnn_out).log_softmax(2)


class CTCBeamDecoder:
    def __init__(self, char_list, beam_width=5, lm_path_file=None, unigrams=None):
        self.labels = [""] + char_list
        if lm_path_file and os.path.exists(lm_path_file):
            self.decoder = build_ctcdecoder(self.labels, kenlm_model_path=lm_path_file, unigrams=unigrams, alpha=0.7,
                                            beta=3.0)
        else:
            self.decoder = build_ctcdecoder(self.labels, kenlm_model_path=None, alpha=0.5, beta=1.0)
        self.beam_width = beam_width

    def decode_batch(self, outputs):
        probs = outputs.permute(1, 0, 2).cpu().numpy()
        return [self.decoder.decode(prob, beam_width=self.beam_width) for prob in probs]


def re_binarize(image, **kwargs):
    _, binary_img = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if binary_img.ndim == 2: return np.expand_dims(binary_img, axis=-1)
    return binary_img


val_augmentations = alb.Compose([
    alb.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    alb.Lambda(image=re_binarize, name="ReBinarize"),
    alb.Normalize(mean=(0,), std=(1,), max_pixel_value=255.0),
    ToTensorV2(),
])


class HTRHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split, encoder):
        self.h5_path = h5_path
        self.encoder = encoder
        self.h5_file = None
        self.split = split
        with h5py.File(h5_path, "r") as f:
            self.labels = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in f[split + '/labels'][:]]
        self.length = len(self.labels)
        # Assuming filenames or an identifier are also stored if needed for proper page grouping,
        # but for analysis, consecutive items are treated as related.

    def __getitem__(self, idx):
        if self.h5_file is None: self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        img = self.h5_file[self.split + '/images'][idx] * 255
        if img.ndim == 2: img = np.expand_dims(img, axis=-1)
        aug = val_augmentations(image=img)
        # Return True Label, Raw Prediction, and a simulated Page ID for grouping
        # We simulate a "Page ID" based on index modulo 50 to process in chunks if true IDs aren't available.
        page_id = str(idx // 50)
        return aug['image'], self.labels[idx], page_id

    def __len__(self):
        return self.length


def simple_collate(batch):
    images, raw_labels, page_ids = zip(*batch)
    images = torch.stack(images)
    return images, list(raw_labels), list(page_ids)


# =========================================================================
# 5. ANALYSIS LOGIC
# =========================================================================

def count_errors(true_text, pred_text, counter_dict):
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            t_sub = true_text[i1:i2]
            p_sub = pred_text[j1:j2]
            if len(t_sub) == 1 and len(p_sub) == 1:
                counter_dict[(t_sub, p_sub)] += 1


def analyze_before_after(loader, model, decoder, device, corrector):
    model.eval()
    raw_subs = Counter()
    fixed_subs = Counter()
    all_raw_data = []

    tqdm.write("Running Raw Inference and Decoding...")

    # A. Decode and Collect Raw Data from DataLoader
    with torch.inference_mode():
        for imgs, raw_texts, page_ids in tqdm(loader, ncols=100, leave=False):
            imgs = imgs.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                preds = model(imgs)

            decoded_raw = decoder.decode_batch(preds)
            batch_char_confs = calculate_char_confidences(preds)

            for i in range(len(raw_texts)):
                all_raw_data.append({
                    'true_text': raw_texts[i],
                    'raw_pred': decoded_raw[i],
                    'confs': batch_char_confs[i],
                    'page_id': page_ids[i]
                })

    tqdm.write("Starting Page-Level Contextual Correction...")

    # B. Group by Simulated Page ID and Run Solver
    page_solver = AnalysisPageBeamSearch(corrector, beam_width=5)

    # Use groupby to process sequential batches as pages
    for page_id, page_data_iter in groupby(all_raw_data, key=lambda x: x['page_id']):
        page_data = list(page_data_iter)

        # Extract packets needed for the solver
        solver_packets = [{'text': p['raw_pred'], 'confs': p['confs']} for p in page_data]

        # CRITICAL: Solve the entire sequence using context
        corrected_sequence = page_solver.solve_page(solver_packets)

        # C. Analyze Errors (Word by Word)
        for i in range(len(page_data)):
            true_text = page_data[i]['true_text']
            raw_pred = page_data[i]['raw_pred']
            fixed_pred = corrected_sequence[i]

            count_errors(true_text, raw_pred, raw_subs)
            count_errors(true_text, fixed_pred, fixed_subs)

    return raw_subs, fixed_subs


def plot_confusion_heatmap(substitutions_counter, title, filename):
    if not substitutions_counter: return
    data = [{'True': t, 'Pred': p, 'Count': c} for (t, p), c in substitutions_counter.items()]
    df = pd.DataFrame(data)
    if df.empty: return

    top_n = 25
    top_true = df.groupby('True')['Count'].sum().nlargest(top_n).index
    top_pred = df.groupby('Pred')['Count'].sum().nlargest(top_n).index
    df_filtered = df[df['True'].isin(top_true) & df['Pred'].isin(top_pred)]

    heatmap_data = df_filtered.pivot_table(index='True', columns='Pred', values='Count', fill_value=0).astype(int)

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', square=True)
    plt.title(title)
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()


def print_diff_report(raw_subs, fixed_subs):
    print("\n" + "=" * 80)
    print(f"{'ERROR TYPE':<30} | {'BEFORE':<8} | {'AFTER':<8} | {'CHANGE'}")
    print("-" * 80)

    all_errors = set(raw_subs.keys()) | set(fixed_subs.keys())
    if not all_errors: print("No errors found!"); return

    sorted_fixes = sorted(all_errors, key=lambda k: (raw_subs.get(k, 0) - fixed_subs.get(k, 0)), reverse=True)
    sorted_regressions = sorted(all_errors, key=lambda k: (fixed_subs.get(k, 0) - raw_subs.get(k, 0)), reverse=True)

    print("--- TOP IMPROVEMENTS ---")
    count = 0
    for k in sorted_fixes:
        before = raw_subs.get(k, 0)
        after = fixed_subs.get(k, 0)
        if before > after:
            print(f"{str(k[0]) + ' -> ' + str(k[1]):<30} | {before:<8} | {after:<8} | -{before - after}")
            count += 1
            if count >= 15: break

    print("\n--- TOP REGRESSIONS ---")
    count = 0
    for k in sorted_regressions:
        before = raw_subs.get(k, 0)
        after = fixed_subs.get(k, 0)
        if after > before:
            print(f"{str(k[0]) + ' -> ' + str(k[1]):<30} | {before:<8} | {after:<8} | +{after - before}")
            count += 1
            if count >= 15: break

    total_fixed_count = sum(max(0, raw_subs[k] - fixed_subs.get(k, 0)) for k in all_errors)
    total_regressed_count = sum(max(0, fixed_subs.get(k, 0) - raw_subs.get(k, 0)) for k in all_errors)

    print("-" * 80)
    print("--- TOTAL SUMMARY ---")
    print(f"Errors Fixed:      {total_fixed_count}")
    print(f"Errors Created:    {total_regressed_count}")
    print(f"NET IMPROVEMENT:   {total_fixed_count - total_regressed_count}")
    print("=" * 80)


def get_char_list():
    alnum = sorted(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
    raw_chars = alnum + ['.', ',', '?', '!', ':', ';', '"', "'", '(', ')', '-', ' ']
    char_list = sorted(list(set(raw_chars)))
    if ' ' in char_list: char_list.remove(' '); char_list.append(' ')
    return char_list


# =========================================================================
# 6. MAIN
# =========================================================================
if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH): print("Checkpoint not found."); exit()

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    char_list = get_char_list()
    encoder = HTRCharEncoder(char_list)

    model = ResNetCRNN(len(char_list) + 1).to(DEVICE)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    unigrams = None
    if os.path.exists(VOCAB_CACHE_FILE):
        with open(VOCAB_CACHE_FILE, 'r', encoding='utf-8') as f: unigrams = json.load(f)

    decoder = CTCBeamDecoder(encoder.char_list, lm_path_file=LM_PATH, unigrams=unigrams)
    print("Initializing Strict Corrector for Page-Level Analysis...")
    corrector = StrictSpellCorrector(LM_PATH)

    # Note: We use the HDF5 dataset, assuming the data within it is somewhat ordered by page/sequence.
    val_dataset = HTRHDF5Dataset(H5_DATABASE_PATH, "val", encoder)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS,
                                             collate_fn=simple_collate)

    raw_counts, fixed_counts = analyze_before_after(val_loader, model, decoder, DEVICE, corrector)

    print_diff_report(raw_counts, fixed_counts)
    plot_confusion_heatmap(raw_counts, "Confusion Matrix (RAW MODEL)", "heatmap_BEFORE.png")
    plot_confusion_heatmap(fixed_counts, "Confusion Matrix (WITH STRICT CORRECTOR)", "heatmap_AFTER.png")
