import os
import h5py
import cv2
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_FOLDER = r"C:\OCR\archive\iam_words\words"
OUTPUT_H5 = "ocr_dataset_binary.h5"

# TARGET SIZE (Matches your optimized model)
IMG_HEIGHT = 64
IMG_WIDTH = 256


def extract_label_from_filename(filename):
    name_no_ext = os.path.splitext(filename)[0]

    # Try to strip _number suffix (e.g. word_1.png -> word)
    match = re.match(r"(.*)_\d+$", name_no_ext)
    if match:
        return match.group(1)

    # If no number, assume the whole name is the label
    return name_no_ext


def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    # 1. Binarizacja (Otsu) -> 0=Ink (Tusz), 255 = Background (Tło)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Inwersja -> 255=Ink (Tusz), 0 = Background (Tło)
    # Po inwersji nasze tło do paddingu to czarne piksele (0)
    img = cv2.bitwise_not(img)

    # 3. Skalowanie z zachowaniem proporcji (Aspect Ratio)
    h, w = img.shape
    if h == 0 or w == 0: return None

    # Obliczamy nową szerokość przy stałej wysokości IMG_HEIGHT (64)
    scale = IMG_HEIGHT / h
    new_width = int(w * scale)

    # Unikamy błędów, gdyby new_width wyszło 0
    new_width = max(1, new_width)

    # Przeskalowanie
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, (new_width, IMG_HEIGHT), interpolation=interp)

    # Ponowna binarizacja po skalowaniu (wygładza krawędzie)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 4. Padding (Wyrównanie do docelowej szerokości)
    # Tworzymy czarne płótno (tło po inwersji jest czarne)
    target = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    if new_width > IMG_WIDTH:
        # Jeśli słowo jest za długie -> Skalujemy je siłowo do IMG_WIDTH
        target = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        # Wklejamy przeskalowane słowo od lewej strony
        target[:, :new_width] = img

    # 5. Normalizacja do formatu 0/1 (uint8)
    # Dzięki temu plik H5 zajmuje 8x mniej miejsca
    target = (target > 127).astype(np.uint8)

    return target


def create_database():
    data = []
    labels = []
    chars = set()

    print(f"Scanning images in {SOURCE_FOLDER}.")

    # Walk through all subfolders to find .png files
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(SOURCE_FOLDER) for f in filenames if
             f.endswith('.png')]
    print(f"Found {len(files)} image files.")

    for file_path in tqdm(files):
        filename = os.path.basename(file_path)

        # EXTRACT LABEL FROM FILENAME
        label = extract_label_from_filename(filename)

        # Skip empty labels
        if len(label) < 1: continue

        img = process_image(file_path)
        if img is not None:
            data.append(img)
            labels.append(label)
            chars.update(list(label))

    print(f"Processed {len(data)} valid samples.")

    if len(data) == 0:
        print("Error: No data found. Check your SOURCE_FOLDER path.")
        return

    print("Splitting Data (90% Train, 10% Val).")
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)

    print(f"Saving to {OUTPUT_H5}.")
    with h5py.File(OUTPUT_H5, 'w') as f:
        # Train
        g_train = f.create_group('train')
        g_train.create_dataset('images', data=np.array(X_train, dtype=np.uint8), compression="gzip")
        dt = h5py.special_dtype(vlen=str)
        g_train.create_dataset('labels', data=np.array(y_train, dtype=object), dtype=dt)

        # Val
        g_val = f.create_group('val')
        g_val.create_dataset('images', data=np.array(X_val, dtype=np.uint8), compression="gzip")
        g_val.create_dataset('labels', data=np.array(y_val, dtype=object), dtype=dt)

        # Vocab
        char_list = "".join(sorted(list(chars)))
        f.attrs['char_list'] = char_list
        print(f"Done. Vocab Size: {len(char_list)}")
        print(f"Characters Found: {char_list}")


if __name__ == "__main__":
    create_database()
