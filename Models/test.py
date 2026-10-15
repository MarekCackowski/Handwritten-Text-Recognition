import math
import random
import sys
import os
from typing import List, Union, Tuple
import torch
from torchaudio.functional import edit_distance
from torchvision.utils import save_image
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import numpy as np
import h5py
from tqdm import tqdm
import re
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.ndimage import map_coordinates, gaussian_filter

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

IAM_MEAN = (0.8491,)
IAM_STD = (0.2259,)

class AdvancedHTRAugmentor:
    """ Augmentacje symulujące rzeczywisty ruch ręki przy pisaniu. """
    @staticmethod
    def ink_bleeding(image, **kwargs):
        """ Wykorzystuje ważoną dylatację i rozmycie Gaussa, uodparniając model na zmienną wilgotność papieru. """
        # Zabezpieczenie przed błędem wymiarów OpenCV dla obrazów (H, W, 1)
        is_expanded = False
        if image.ndim == 3 and image.shape[1] == 1:
            image = np.squeeze(image, axis=2)
            is_expanded = True

        strength = random.uniform(0.1, 0.25)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        dilated = cv.dilate(image, kernel, iterations=1)
        bleeded = cv.addWeighted(image, 1 - strength, dilated, strength, 0)
        
        k_size = 3
        blurred = cv.medianBlur(bleeded, k_size)
        
        # Przywrócenie wymiaru (H, W, 1) jeśli był obecny
        if is_expanded:
            blurred = np.expand_dims(blurred, axis=2)
            
        return blurred

    @staticmethod
    def variable_stroke(img, **kwargs):
        """ Symuluje zmienny nacisk narzędzia piszącego i różną grubość stalówki. """
        # Zabezpieczenie przed pustym obrazem
        if img is None or img.size == 0: 
            return img
        
        # Jeśli obraz jest 3-kanałowy, upewnijmy się, że pracujemy na kanale jasności
        if img.ndim == 3: 
            img = img[:, :, 0]
            
        # Jeśli średnia > 127, tło jest jasne -> odwracamy, by otrzymać biały tekst na czarnym tle
        binary = cv.bitwise_not(img) if np.mean(img) > 127 else img
        
        # Distance transform dla wyliczenia grubości kresek
        dist = cv.distanceTransform(binary, cv.DIST_L2, 3)
        
        # Modyfikacja grubości przez potęgowanie (factor < 1.0 -> cieńsza, factor > 1.0 -> grubsza)
        factor = np.random.uniform(0.7, 1.4)
        dist = np.power(dist, factor)
        
        # Normalizacja do zakresu [0, 255] i powrót do uint8
        cv.normalize(dist, dist, 0, 255, cv.NORM_MINMAX)
        return dist.astype(np.uint8)
    
    @staticmethod
    def phantom_elements(image, **kwargs):
        """ Wymusza na sieci wizualnej ignorowanie szumu z sąsiadujących wierszy. """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        phantom_h = int(h * 0.15)
        if phantom_h < 2:
            return image

        phantom = np.zeros((phantom_h, w), dtype=np.uint8)
        for _ in range(random.randint(1, 2)):
            x = random.randint(0, max(0, w - 30))
            cv.ellipse(phantom, (x, phantom_h), (random.randint(5, 12), 4), 
                       random.randint(0, 30), 0, 360, (255), -1)

        actual_ph = min(phantom_h, h)

        if image.ndim == 3:
            # Jeśli mamy 3 wymiary (H, W, C), operujemy na kanale 0
            if random.random() > 0.5:
                image[:actual_ph, :, 0] = cv.bitwise_or(image[:actual_ph, :, 0], phantom[:actual_ph, :])
            else:
                image[-actual_ph:, :, 0] = cv.bitwise_or(image[-actual_ph:, :, 0], np.flipud(phantom[:actual_ph, :]))
        else:
            # Jeśli mamy 2 wymiary (H, W)
            if random.random() > 0.5:
                image[:actual_ph, :] = cv.bitwise_or(image[:actual_ph, :], phantom[:actual_ph, :])
            else:
                image[-actual_ph:, :] = cv.bitwise_or(image[-actual_ph:, :], np.flipud(phantom[:actual_ph, :]))
            
        return image
    
    @staticmethod
    def create_realistic_space(height, width):
        """ Zastępuje idealnie czarną próżnię zaszumionym tłem charakterystycznym dla skanów. """
        if width <= 0:
            return np.zeros((height, 1), dtype=np.uint8)
        # Tworzymy delikatny szum o niskiej amplitudzie, który imituje fakturę papieru
        base_bg = np.random.randint(0, 18, (height, width), dtype=np.uint8)
        
        # Rozmywamy, aby przypominało fakturę papieru, a nie cyfrowy szum
        return cv.GaussianBlur(base_bg, (3, 3), 0)


class IAMWordDataset(Dataset):
    def __init__(self, h5_path, transform, char_list, name="Główny", split='train'):
        self.h5_path = h5_path
        self.transform = transform
        self.dictionary = set(char_list)
        self.name = name
        self.split = split
        self.IMAGE_HEIGHT = 64
        self.valid_indices = []
        self.valid_labels = []
        self.h5_file = None
        self.images_group_path = 'images'

        with h5py.File(self.h5_path, 'r') as f:
            if self.split in f:
                target_group = f[self.split]
                labels = target_group['labels'][:]
                self.images_group_path = f"{self.split}/images"
                indices = range(len(labels))
                tqdm.write(f"[{now()}] Wykryto strukturę H5. Używam grupy: {self.split}")
            else:
                labels = f['labels'][:]
                total = len(labels)
                np.random.seed(42)
                indices = np.random.permutation(total)
                train_cutoff = int(0.9 * total)
                indices = indices[:train_cutoff] if self.split == 'train' else indices[train_cutoff:]
                tqdm.write(f"[{now()}] Plik płaski. Stosuję manualny split.")

            for i in indices:
                lbl = labels[i]
                text = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text).strip()
                text = re.sub(r'_\d+$', '', text) 
                text = re.sub(r'-\d+$', '', text)
                
                if text and all(c in self.dictionary for c in text):
                    self.valid_indices.append(i)
                    self.valid_labels.append(text)

        self.dataset_len = len(self.valid_indices)
        tqdm.write(f"[{now()}] Zainicjalizowano {self.name}: {self.dataset_len} próbek.")
        
    def __getitem__(self, idx):
        h5_idx = self.valid_indices[idx]
        label = self.valid_labels[idx]
        img = self._load_single_h5_image(h5_idx)
        
        if img is None or img.ndim < 2 or img.size == 0:
            return torch.zeros((1, self.IMAGE_HEIGHT, 64)), label

        if np.mean(img) > 127:
            img = cv.bitwise_not(img)

        _, thresh = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        coords = cv.findNonZero(thresh)
        if coords is not None and coords.shape[0] > 0:
            x, y, w_bbox, h_bbox = cv.boundingRect(coords)
            if w_bbox > 1 and h_bbox > 1:
                pad = int(h_bbox * 0.15)
                x1 = max(0, x - pad)
                x2 = min(img.shape[1], x + w_bbox + pad)
                img = img[:, x1:x2]

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return torch.zeros((1, self.IMAGE_HEIGHT, 64)), label
            
        scale = self.IMAGE_HEIGHT / max(1, h)
        new_w = max(16, int(w * scale)) 
        img = cv.resize(img, (new_w, self.IMAGE_HEIGHT), interpolation=cv.INTER_AREA)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            
        augmented = self.transform(image=img)
        return augmented['image'], label
        
    def get_text(self, idx):
        if hasattr(self, 'word_list'):
            if not self.word_list: return "test"
            return self.word_list[idx % len(self.word_list)]
        if hasattr(self, 'valid_labels'):
             return self.valid_labels[idx]
        return ""
    
    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def _load_single_h5_image(self, h5_idx):
        f = self._get_h5_file()
        data = f[self.images_group_path][h5_idx]
        nparr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((64, 64), dtype=np.uint8)
        return img

    def __del__(self):
        try:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        except:
            pass

    def __len__(self):
        return self.dataset_len
    
class PolishCharStitcher:
    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.images = None
        self.labels = None
        self.char_map = None

    def _lazy_init(self):
        if self.char_map is None:
            data = np.load(self.npz_path, allow_pickle=True)
            raw_images = data['signs']
            self.labels = data['labels']
            self.char_map = {}
            for i, char in enumerate(self.labels):
                img = raw_images[i].copy()
                if np.mean(img) > 127: img = 255 - img
                if char not in self.char_map: self.char_map[char] = []
                self.char_map[char].append(img)

    def generate_word_image(self, text: str, target_height=64):
        self._lazy_init()
        word_imgs = []
        for char in text:
            if char in [" ", "_"]:
                word_imgs.append(np.zeros((target_height, random.randint(20, 40)), dtype=np.uint8))
                continue
            samples = self.char_map.get(char)
            if not samples:
                word_imgs.append(np.zeros((target_height, 10), dtype=np.uint8))
                continue
            char_img = random.choice(samples)
            h, w = char_img.shape
            M = cv.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), random.uniform(0.95, 1.05))
            char_img = cv.warpAffine(char_img, M, (w, h), borderValue=0)
            word_imgs.append(char_img)

        if not word_imgs: return np.zeros((target_height, target_height), dtype=np.uint8)
        combined_width = sum([img.shape[1] for img in word_imgs]) + 40
        final_img = np.zeros((target_height, combined_width), dtype=np.uint8)
        current_x = 15

        for char, img in zip(text, word_imgs):
            h, w = img.shape
            if h != target_height:
                w = int(w * (target_height / h))
                img = cv.resize(img, (w, target_height))
                h = target_height
            y_offset = random.randint(-2, 2)
            y_pos = max(0, min(target_height - h, (target_height - h) // 2 + y_offset))
            region = final_img[y_pos:y_pos + h, current_x:current_x + w]
            final_img[y_pos:y_pos + h, current_x:current_x + w] = np.maximum(region, img)
            current_x += w
            
        final_img = final_img[:, :current_x + 15]
        noise = np.random.randint(0, 11, final_img.shape, dtype=np.uint8)
        return cv.add(final_img, noise)

class PolishSyntheticDataset(Dataset):
    def __init__(self, npz_path, word_list, transform=None, num_samples=5000):
        self.npz_path = npz_path
        self.word_list = [w for w in word_list if len(w) > 0]
        self.transform = transform
        self.num_samples = num_samples
        self.stitcher = PolishCharStitcher(self.npz_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        word = self.word_list[idx % len(self.word_list)]
        img = self.stitcher.generate_word_image(word)
        img = cv.bitwise_not(img)
        
        if img is None or img.ndim < 2 or img.size == 0:
            img = np.zeros((64, 64), dtype=np.uint8)
            word = "error" 
        
        h, w = img.shape[:2]
        if h != 64:
            scale = 64 / max(1, h)
            new_w = max(16, int(w * scale))
            img = cv.resize(img, (new_w, 64), interpolation=cv.INTER_AREA)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, word

def collate_fn_dynamic(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    valid_batch = []
    for item in batch:
        if torch.is_tensor(item[0]) and item[0].dim() == 3:
            label_clean = str(item[1]).strip()
            if label_clean and ' ' not in label_clean: 
                new_item = list(item)
                new_item[1] = label_clean
                valid_batch.append(tuple(new_item))
                
    if not valid_batch: return None

    imgs = [item[0] for item in valid_batch]
    labels = [item[1] for item in valid_batch]
    categories = [item[2] if len(item) > 2 else 'short' for item in valid_batch]
    extras = list(zip(*[item[3:] for item in valid_batch])) if any(len(item) > 3 for item in valid_batch) else []

    bg_val = (0.0 - float(IAM_MEAN[0])) / float(IAM_STD[0])
    processed_imgs = []
    for img, label in zip(imgs, labels):
        current_w = int(img.shape[-1])
        min_w_needed = len(label) * 48
        if current_w < min_w_needed:
            pad_right = min_w_needed - current_w
            img = torch.nn.functional.pad(img, (0, pad_right), value=bg_val)
            current_w = min_w_needed
        if current_w > 2048:
            img = img[:, :, :2048]
        processed_imgs.append(img)

    batch_max_w = max(int(img.shape[-1]) for img in processed_imgs)
    original_widths = [int(img.shape[-1]) for img in processed_imgs]
    padded_imgs = []
    for img in processed_imgs:
        curr_w = int(img.shape[-1])
        pad_right = batch_max_w - curr_w
        if pad_right > 0:
            padded_img = torch.nn.functional.pad(img, (0, pad_right), value=bg_val)
        else:
            padded_img = img
        padded_imgs.append(padded_img)

    padded = torch.stack(padded_imgs)
    if extras:
        return padded, labels, categories, original_widths, *[list(field) for field in extras]
    return padded, labels, categories, original_widths

def load_sjp_dictionary(file_path, alphabet_set, num_desired=30000):
    polish_diacritics = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    with_diacritics = []
    standard = []
    encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1250', 'iso-8859-2']
    file_content = None

    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                file_content = f.read()
                if len(file_content.strip()) > 0: break
        except (UnicodeDecodeError, UnicodeError, FileNotFoundError):
            continue

    if not file_content or len(file_content.strip()) == 0:
        return ["zażółć", "gęślą", "jaźń", "kość", "awaryjny", "start"]

    words = re.findall(r'\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b', file_content)
    for word in words:
        if 3 < len(word) < 14 and all(c in alphabet_set for c in word):
            if any(char in polish_diacritics for char in word):
                with_diacritics.append(word)
            else:
                standard.append(word)

    with_diacritics = list(set(with_diacritics))
    standard = list(set(standard))
    random.shuffle(with_diacritics)
    random.shuffle(standard)
    count_diac = int(num_desired * 0.5)
    actual_diac = min(len(with_diacritics), count_diac)
    count_std = num_desired - actual_diac

    final_list = with_diacritics[:actual_diac] + standard[:count_std]
    random.shuffle(final_list)
    return final_list

def get_full_htr_char_list():
    return [
        ' ', '!', '"', "'", '(', ')', ',', '-', '.', '/', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        ':', ';', '?',                                    
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
        'U', 'V', 'W', 'X', 'Y', 'Z',                     
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
        'u', 'v', 'w', 'x', 'y', 'z',                     
        'Ó', 'ó', 'Ą', 'ą', 'Ć', 'ć', 'Ę', 'ę', 'Ł', 'ł', 
        'Ń', 'ń', 'Ś', 'ś', 'Ź', 'ź', 'Ż', 'ż'            
    ]


# =========================================================
# GŁÓWNY SKRYPT DETEKTYWISTYCZNY
# =========================================================
if __name__ == "__main__":
    print("\n[DEBUG] Rozpoczynam szybkie przewijanie zbioru walidacyjnego do kroku 2340...")

    DATA_ROOT = "/home/marek/OCR/HandwrittenTextRecognition/Data"
    debug_dir = "/home/marek/OCR/HandwrittenTextRecognition/output_data/debug_batch_2340"
    IAM_WORDS_H5_PATH = os.path.join(DATA_ROOT, "clean_dataset_processed.h5")
    OUTPUT_NPZ = os.path.join(DATA_ROOT, "dataset.npz")
    SJP_DICTIONARY = os.path.join(DATA_ROOT, "clean_corpus.txt") 

    os.makedirs(debug_dir, exist_ok=True)

    val_transform_crnn = alb.Compose([
            alb.Normalize(mean=IAM_MEAN, std=IAM_STD),
            ToTensorV2()
        ])

    char_list = get_full_htr_char_list()
    alphabet_set = set(char_list)
    polish_words_list = load_sjp_dictionary(SJP_DICTIONARY, alphabet_set, num_desired=30000)
    split_idx = int(0.9 * len(polish_words_list))
    val_polish_words = polish_words_list[split_idx:]

    val_iam = IAMWordDataset(h5_path=IAM_WORDS_H5_PATH, 
                                transform=val_transform_crnn,
                                char_list=char_list, 
                                name="IAM_Val",
                                split='val')
                                
    val_polish = PolishSyntheticDataset(char_map=char_list, 
                                            word_list=val_polish_words,
                                            transform=val_transform_crnn,
                                            num_samples=1333)
                                            
    val_dataset = ConcatDataset([val_iam, val_polish])

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,         
        shuffle=False,                 
        num_workers=0,      
        pin_memory=False,
        collate_fn=collate_fn_dynamic  
    )

    total_batches = len(val_loader)
    print(f"Całkowita liczba paczek w val_loader to: {total_batches}")

    if total_batches <= 2341:
        print(f"\n[BŁĄD FATALNY] Twój zbiór walidacyjny ma tylko {total_batches} paczek!")
        print("Nie mogę sprawdzić batcha 2340.")
        sys.exit(1)

    for i, batch in enumerate(val_loader):
        if i % 500 == 0:
            print(f"Przewinięto {i} paczek...")
            
        if i == 2341:
            print(f"\n[!!!] BATCH 2340 ZNALEZIONY [!!!]")
            if batch is None:
                print("Batch 2340 zwrócił 'None' (pusta paczka).")
            else:
                try:
                    images = batch[0]
                    text_labels = batch[1]
                    
                    save_image(images.float().cpu(), f"{debug_dir}/batch_2340_images.png", normalize=True)
                    
                    with open(f"{debug_dir}/batch_2340_labels.txt", "w", encoding="utf-8") as f:
                        for idx, text in enumerate(text_labels):
                            f.write(f"Zdjecie nr {idx}: '{text}'\n")
                            
                    print(f"SUKCES! Zapisano zdjęcia i etykiety do: {debug_dir}")
                    print(f"Podgląd etykiet w terminalu: {text_labels}")
                    
                except Exception as e:
                    print(f"Błąd krytyczny przy próbie zapisu batcha 2340: {e}")
            
            print("\nTest zakończony.")
            sys.exit(0)