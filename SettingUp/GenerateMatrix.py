import os
import cv2
import torch
import numpy as np
import difflib
from tqdm import tqdm
import torch.nn.functional as func
from ResNetCRNNWordRecognition import ResNetCRNN
from DeepCapsNetCharRecognition import CapsNet
from CRNNCNTROCR import CRNNInferencePipeline, CascadeRefinementNetwork


def update_matrix_with_alignment(matrix, true_text, pred_text, char_to_idx):
    """Używa SequenceMatcher zamiast Levenshteina, żeby uniknąć błędów bufora C.
    Następnie precyzyjnie mapuje poprawne trafienia i podmiany znaków."""
    t_text = str(true_text).strip()
    p_text = str(pred_text).strip()

    if not t_text or not p_text:
        return

    matcher = difflib.SequenceMatcher(None, t_text, p_text)

    # Poprawne
    for i, j, n in matcher.get_matching_blocks():
        for k in range(n):
            char = t_text[i + k]
            if char in char_to_idx:
                idx = char_to_idx[char]
                matrix[idx][idx] += 1

    # Zamienione
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Wycinamy fragmenty, które zostały zamienione
            true_chunk = t_text[i1:i2]
            pred_chunk = p_text[j1:j2]

            # Porównujemy znaki wewnątrz zamienionego bloku
            for t_char, p_char in zip(true_chunk, pred_chunk):
                if t_char in char_to_idx and p_char in char_to_idx:
                    matrix[char_to_idx[t_char]][char_to_idx[p_char]] += 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_DIR = r"C:\OCR\cvl_dataset_words"
    LABELS_FILE = os.path.join(DATASET_DIR, "labels.txt")
    OUTPUT_MATRIX = "confusion_matrix_post_capsnet.npy"

    CRNN_WEIGHTS = r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth"
    CAPS_WEIGHTS = r"output_data\checkpoints\hcr\capsnet_char_level.pth"

    CHAR_LIST = sorted(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?:;\"'()-/ "))
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_LIST)}

    print("Inicjalizacja kaskady wizualnej")
    model_crnn = ResNetCRNN(num_classes=len(CHAR_LIST) + 1).to(device).half().eval()
    checkpoint_crnn = torch.load(CRNN_WEIGHTS, map_location=device)

    # Sprawdzamy, czy wagi są ukryte pod kluczem 'model_state'
    if 'model_state' in checkpoint_crnn:
        model_crnn.load_state_dict(checkpoint_crnn['model_state'])
    else:
        model_crnn.load_state_dict(checkpoint_crnn)
    print("[+] Model CRNN wczytany pomyślnie.")

    # CapsNet
    model_caps = CapsNet(num_classes=62).to(device).half().eval()
    checkpoint_caps = torch.load(CAPS_WEIGHTS, map_location=device)

    if 'model_state' in checkpoint_caps:
        model_caps.load_state_dict(checkpoint_caps['model_state'])
    else:
        model_caps.load_state_dict(checkpoint_caps)
    print("[+] Model CapsNet wczytany pomyślnie.")

    pipeline = CRNNInferencePipeline(model_crnn, CHAR_LIST, device)
    pipeline.refiner = CascadeRefinementNetwork(model_caps, CHAR_LIST, pipeline)
    pipeline.transformer = None  # Chcemy błędy samej wizji

    # Macierz
    size = len(CHAR_LIST)
    conf_matrix = np.zeros((size, size), dtype=np.float32)

    if not os.path.exists(LABELS_FILE):
        print(f"[!] BŁĄD: Nie znaleziono pliku etykiet: {LABELS_FILE}")
        return

    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()

    print(f"[*] Przetwarzanie {len(data_lines)} linii tekstu.")

    for line in tqdm(data_lines):
        try:
            parts = line.strip().split('|')
            if len(parts) < 2: continue

            img_name, true_text = parts[0].strip(), parts[1].strip()
            img_path = os.path.join(DATASET_DIR, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # Preprocessing
            h, w = img.shape
            img_res = cv2.resize(img, (int(w * (64 / h)), 64))
            img_t = torch.from_numpy(img_res).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            img_t = ((img_t - 0.5) / 0.5).half()

            with torch.no_grad():
                _, visual_results = pipeline.predict_batch(img_t)
                pred_text = visual_results[0] if visual_results else ""

            update_matrix_with_alignment(conf_matrix, true_text, pred_text, CHAR_TO_IDX)

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            continue

    print("Normalizacja macierzy.")
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # Zastępujemy zera jedynkami w sumach, żeby uniknąć NaN
    norm_matrix = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix), where=row_sums != 0)

    np.save(OUTPUT_MATRIX, norm_matrix)
    print(f"Sukces! Znormalizowana macierz zapisana w: {OUTPUT_MATRIX}")


if __name__ == "__main__":
    main()
