import os
import cv2 as cv
import numpy as np
import time
from datetime import datetime
import string
import h5py
from sklearn.model_selection import train_test_split

# --- Hyperparameters ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 600
BASE_FOLDER = r"C:\OCR\words_data"
# PLIK Z ETYKIETAMI (Notatnik)
INDEX_FILE = r"C:\OCR\words_data\labels.txt"
OUTPUT_FILE = "ocr_dataset.h5"
BATCH_SIZE = 1000

# Alfabet do zapisania w metadanych HDF5
CHARACTERS = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + ' .,!?:;\"\'()-/')


def now():
    return datetime.now().strftime("%H:%M:%S")


def process_single_image(img_path, img_h, img_w):
    try:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None: return None
        h, w = img.shape
        scale = img_h / h
        new_w = int(w * scale)
        img_resized = cv.resize(img, (new_w, img_h), interpolation=cv.INTER_AREA)

        if new_w < img_w:
            pad_w = img_w - new_w
            img_resized = np.pad(img_resized, ((0, 0), (0, pad_w)), mode='constant', constant_values=255)
        else:
            img_resized = cv.resize(img_resized, (img_w, img_h), interpolation=cv.INTER_AREA)

        img_tensor = img_resized[np.newaxis, ...] / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor.astype(np.float32)
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")
        return None


def write_batch_to_hdf5(grp, images, labels, current_count):
    n_new = len(images)
    if n_new == 0: return current_count
    grp['images'].resize((current_count + n_new), axis=0)
    grp['labels'].resize((current_count + n_new), axis=0)
    grp['images'][current_count:] = np.array(images)
    grp['labels'][current_count:] = np.array(labels, dtype=object)
    return current_count + n_new


def process_and_save_split(hdf5_file, group_name, paths, labels_list):
    print(f"[{now()}] Start sekcji '{group_name}' ({len(paths)} obrazów)...")
    grp = hdf5_file.create_group(group_name)
    grp.create_dataset('images', shape=(0, 1, IMAGE_HEIGHT, IMAGE_WIDTH),
                       maxshape=(None, 1, IMAGE_HEIGHT, IMAGE_WIDTH),
                       dtype=np.float32, chunks=True)
    dt = h5py.special_dtype(vlen=str)
    grp.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=dt)

    buffer_imgs, buffer_labels = [], []
    total_saved = 0
    start_time = time.time()

    for i, (path, label) in enumerate(zip(paths, labels_list)):
        img_data = process_single_image(path, IMAGE_HEIGHT, IMAGE_WIDTH)
        if img_data is not None:
            buffer_imgs.append(img_data)
            buffer_labels.append(label)

        if len(buffer_imgs) >= BATCH_SIZE:
            total_saved = write_batch_to_hdf5(grp, buffer_imgs, buffer_labels, total_saved)
            buffer_imgs, buffer_labels = [], []
            if i % 5000 == 0:
                print(f"[{now()}] {group_name}: {i}/{len(paths)} zapisano.")

    if buffer_imgs:
        total_saved = write_batch_to_hdf5(grp, buffer_imgs, buffer_labels, total_saved)
    print(f"[{now()}] Gotowe '{group_name}'. Razem: {total_saved}")


def generate_dataset():
    if not os.path.exists(INDEX_FILE):
        print(f"[!] BŁĄD: Nie znaleziono pliku indeksu: {INDEX_FILE}")
        return

    print(f"[{now()}] Wczytywanie etykiet z pliku tekstowego...")
    all_png_paths = []
    all_labels = []

    # Oczekiwany format w labels.txt: nazwa_pliku.png|Etykieta z dowolnymi znakami ?!:
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line: continue

            filename, label = line.split("|", 1)
            full_path = os.path.join(BASE_FOLDER, filename)

            if os.path.exists(full_path):
                all_png_paths.append(full_path)
                all_labels.append(label)
            else:
                print(f"[?] Pominęto: brak pliku {filename}")

    print(f"[{now()}] Znaleziono {len(all_png_paths)} poprawnych par w indeksie.")

    X_train, X_val, y_train, y_val = train_test_split(all_png_paths, all_labels, test_size=0.15, random_state=42)

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.attrs['char_list'] = "".join(CHARACTERS)
        process_and_save_split(f, "train", X_train, y_train)
        process_and_save_split(f, "val", X_val, y_val)

    print(f"[{now()}] Dataset zapisany do {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
