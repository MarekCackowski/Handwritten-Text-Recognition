import os
import random
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

# Biblioteki do zaawansowanej analizy obrazu
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, convolve


# Segmentacja
from Preprocessing.OpticalLayoutRecognition import PageToLineSegmentor, LineToWordSegmentor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Binarizer:
    def __init__(self, image: np.ndarray):
        if image.dtype != np.uint8:
            self.image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            self.image = image

        self.overlap_threshold = 0.15
        self.n_win = 61
        self.n_k = 0.1
        self.block_size = 31

    @staticmethod
    def estimate_stroke_width(binary_image: np.ndarray) -> int:
        n_labels, _, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
        # Filtrujemy tło (indeks 0) i bardzo małe kropki
        heights = [stats[i, cv.CC_STAT_HEIGHT] for i in range(1, n_labels) if stats[i, cv.CC_STAT_HEIGHT] > 1]
        return int(np.median(heights)) if heights else 1

    @staticmethod
    def filter_small_cc(binary_image: np.ndarray, stroke_width: int, min_ratio: float = 0.5) -> np.ndarray:
        min_h = max(1, int(stroke_width * min_ratio))
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_image, connectivity=8)
        out = np.zeros_like(binary_image)
        for i in range(1, n_labels):
            if stats[i, cv.CC_STAT_HEIGHT] >= min_h:
                out[labels == i] = 255
        return out

    def binarize(self) -> np.ndarray:
        """ Główna logika binarizacji z jawnym typowaniem dla lintera. """
        # Wymuszamy typ uint8 i upewniamy się, że to ndarray
        img = np.asarray(self.image, dtype=np.uint8)

        # Rzutujemy na float32 przed std(), a wynik na float
        contrast_measure = float(np.std(img.astype(np.float32)))

        # Dynamiczne dopasowanie k do kontrastu obrazu
        current_k = float(self.n_k)
        if contrast_measure < 40.0:
            current_k = float(max(0.02, self.n_k * (contrast_measure / 50.0)))

        # OpenCV: rzutujemy wynik na np.ndarray, by linter widział .shape
        adaptive_bin = np.asarray(cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, self.block_size, 10
        ), dtype=np.uint8)

        sauvola_thresh = threshold_sauvola(img, window_size=self.n_win, k=current_k)
        sauvola_bin = np.where(img > sauvola_thresh, 255, 0).astype(np.uint8)

        final = np.zeros_like(adaptive_bin, dtype=np.uint8)
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(sauvola_bin, connectivity=8)

        for j in range(1, n_labels):
            comp = np.where(labels == j, 255, 0).astype(np.uint8)

            # type: ignore dla bitwise_and, bo stubs OpenCV mają błędy w definicji Unionów
            mask_overlap = cv.bitwise_and(comp, adaptive_bin)  # type: ignore

            # Rzutujemy na float, aby uniknąć błędów __truediv__ na typach generic
            comp_sum = float(np.sum(comp).item()) / 255.0

            if comp_sum <= 0.0:
                continue

            overlap_sum = float(np.sum(mask_overlap > 0))
            overlap = overlap_sum / comp_sum
            if overlap >= self.overlap_threshold:
                final[labels == j] = 255

        stroke_estimate = int(self.estimate_stroke_width(final))
        final = self.filter_small_cc(final, stroke_estimate, min_ratio=0.5)

        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        final = cv.morphologyEx(final, cv.MORPH_CLOSE, kernel_close)

        kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        return np.asarray(cv.dilate(final, kernel_dilate, iterations=1), dtype=np.uint8)


def estimate_background(image: np.ndarray, niblack_binary: np.ndarray) -> np.ndarray:
    """ Estymacja tła metodą inpaint-convolution z jawnym rzutowaniem. """
    dilate_size, inpaint_passes = 3, 5

    # Używamy np.where, by linter nie miał wątpliwości co do typu tablicy
    mask = np.where(niblack_binary == 0, 255, 0).astype(np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_size, dilate_size))
    mask_d = cv.dilate(mask, kernel, iterations=1)

    inpaint = image.astype(np.float32)
    conv_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

    passes = []
    for _ in range(4):
        img_p = inpaint.copy()
        m_p = mask_d.copy()
        for _ in range(inpaint_passes):
            # Zamiast .astype na masce, używamy opakowania np.float32()
            m_p_norm = np.float32(m_p > 0)

            # Obliczanie sumy i liczby sąsiadów (wykorzystujemy znane tło)
            neighbor_sum = convolve(img_p * (1.0 - m_p_norm), conv_kernel, mode='constant', cval=0.0)
            neighbor_count = convolve(1.0 - m_p_norm, conv_kernel, mode='constant', cval=0.0)

            # Unikanie dzielenia przez zero przy użyciu np.where
            neighbor_count = np.where(neighbor_count == 0, 1.0, neighbor_count)

            # Zastępujemy tylko piksele pod maską
            idx = m_p > 0
            img_p[idx] = neighbor_sum[idx] / neighbor_count[idx]
            m_p[idx] = 0
        passes.append(img_p)

    # Zwracamy najciemniejsze oszacowanie tła
    return np.min(np.stack(passes, axis=0), axis=0)


def normalize_image(image: np.ndarray, BG: np.ndarray) -> np.ndarray:
    """ Korekta oświetlenia (Flat-field) zwracająca zakres [-1, 1]. """
    # Obliczenia na float32
    I = image.astype(np.float32) + 1.0
    BGp = BG.astype(np.float32) + 1.0

    # Wykonujemy dzielenie (korekta tła)
    F = I / BGp

    # Używamy funkcji np.min/max zamiast metod obiektu - to ucisza lintera
    f_min = float(np.min(F))
    f_max = float(np.max(F))

    # Normalizacja do zakresu [0, 1], a potem przesunięcie do [-1, 1]
    if abs(f_max - f_min) < 1e-7:
        return np.zeros_like(F, dtype=np.float32)

    F_norm = (F - f_min) / (f_max - f_min)

    # Przekształcenie 0..1 -> -1..1
    final_res = F_norm * 2.0 - 1.0
    return np.asarray(final_res, dtype=np.float32)


class Preprocessing:
    def __init__(self):
        self.Binarizer = Binarizer
        self.kernel_size = 2
        self.iterations_dilate = 1
        self.iterations_erode = 1
        self.target_height = 64
        self.target_thickness = 3
        self.target_corpus_h = 32

    @staticmethod
    def balance_white(img: np.ndarray) -> np.ndarray:
        """ Korekcja balansu bieli oparta na 95. percentylu jasności. """
        p95 = float(np.percentile(img, 95)) or 1.0
        return np.clip(img.astype(np.float32) * (255.0 / p95), 0, 255).astype(np.uint8)

    @staticmethod
    def sharpen_ink(image: np.ndarray) -> np.ndarray:
        """ Ostrzenie krawędzi liter. """
        gaussian = cv.GaussianBlur(image, (0, 0), 3)
        return cv.addWeighted(image, 1.5, gaussian, -0.5, 0)

    @staticmethod
    def area_filter(binary_image: np.ndarray, min_area: int = 15) -> np.ndarray:
        """ Oczekuje formatu: białe obiekty (szum/litery) na czarnym tle. """
        work_bin = binary_image.copy()
        n_labels, labels, stats, _ = cv.connectedComponentsWithStats(work_bin, connectivity=8)
        # Zaczynamy od 1, by zignorować czarne tło (label 0)
        for i in range(1, n_labels):
            if stats[i, cv.CC_STAT_AREA] < min_area:
                work_bin[labels == i] = 0  # Kasujemy "pyłek" zamieniając go na tło
        return work_bin

    @staticmethod
    def denoise_fast(image: np.ndarray) -> np.ndarray:
        return cv.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def denoise_professional(image: np.ndarray) -> np.ndarray:
        """ NL-Means Denoising """
        return cv.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """ ML: Wykorzystanie Analizy Głównych Składowych (PCA) do prostowania strony. """
        # Szybka binaryzacja Otsu tylko po to, by znaleźć współrzędne atramentu
        _, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        y, x = np.nonzero(thresh)

        # Jeśli strona jest pusta, ignoruj
        if len(x) < 100:
            return image

        # Złożenie współrzędnych i obliczenie macierzy kowariancji (PCA)
        coords = np.vstack([x, y]).astype(np.float64).T
        mean, eigenvectors = cv.PCACompute(coords, mean=None)

        # Główny wektor własny definiuje naturalny kierunek układu tekstu.
        angle = float(np.degrees(np.arctan2(float(eigenvectors[0, 1]), float(eigenvectors[0, 0]))))

        # PCA może zwrócić wektor odwrócony, upewniamy się, że kąt mieści się w logicznych ramach
        if angle < -45.0:
            angle += 90.0
        elif angle > 45.0:
            angle -= 90.0

        if abs(angle) < 0.5:
            return image

        h, w = image.shape
        matrix = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv.warpAffine(image, matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    @staticmethod
    def enhance_contrast_kmeans(image: np.ndarray) -> np.ndarray:
        """ ML: Uczenie nienadzorowane (K-Means) do wyostrzania atramentu i tła.
            Automatycznie otwiera przestrzeń między najciemniejszymi a najjaśniejszymi pikselami. """
        # Rozdzielamy warunki, by zagwarantować linterowi zwrot typu np.ndarray
        if image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        if image.size == 0:
            return image

        # Spłaszczenie do 1D dla K-Means
        Z = image.reshape((-1, 1)).astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Klasteryzacja na 2 grupy (tło i atrament)
        _, labels, centers = cv.kmeans(Z, 2, None, criteria, 3, cv.KMEANS_PP_CENTERS)

        # To gwarantuje linterowi, że operujemy na typach numerycznych NumPy
        dark_val = float(np.min(centers))
        light_val = float(np.max(centers))

        # Ochrona przed dzieleniem przez zero dla pustych/jednolitych wycinków
        if light_val - dark_val < 5.0:
            return image

        # Matematyczne rozciągnięcie kontrastu na podstawie znalezionych klastrów
        img_float = image.astype(np.float32)
        sharpened = np.clip((img_float - dark_val) * 255.0 / (light_val - dark_val), 0, 255)

        return sharpened.astype(np.uint8)

    @staticmethod
    def marginal_noise_removal(image: np.ndarray, margin_ratio: float = 0.03, max_noise_size: int = 5) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        mh, mw = int(h * margin_ratio), int(w * margin_ratio)
        slices = [(slice(0, mh), slice(0, w)), (slice(h - mh, h), slice(0, w)),
                  (slice(0, h), slice(0, mw)), (slice(0, h), slice(w - mw, w))]
        for sy, sx in slices:
            crop = img[sy, sx]
            if crop.size == 0: continue
            binary_noise = np.where(crop < 128, 255, 0).astype(np.uint8)
            num, labels, stats, _ = cv.connectedComponentsWithStats(binary_noise, connectivity=8)
            for j in range(1, num):
                if stats[j, cv.CC_STAT_AREA] <= max_noise_size:
                    img[sy, sx][labels == j] = 255
        return img

    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """ Usuwa cienie i gradienty oświetlenia. """
        kernel_size = 45
        dilated = cv.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))
        bg_img = cv.medianBlur(dilated, 21)
        diff_img = 255 - cv.absdiff(image, bg_img)
        return cv.normalize(diff_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    @staticmethod
    def deslant(image: np.ndarray) -> np.ndarray:
        """ Korekta pochylenia liter (Vinciarelli). """
        img_work = np.asarray(image, dtype=np.uint8)

        # Sprawdzamy czy obraz nie jest pusty (rzutowanie sumy na int)
        if int(np.sum(img_work == 0)) == 0:
            return image

        # Inwersja, jeśli tło jest jasne (operujemy na czarnym tle dla statystyk)
        if float(np.mean(img_work)) > 127.0: # type: ignore
            img_work = cv.bitwise_not(img_work)

        def get_score(img: np.ndarray, angle: float) -> float:
            r, c = img.shape
            alpha = float(np.tan(angle))
            M = np.float32([[1, -alpha, 0], [0, 1, 0]]) # Macierz transformacji afinicznej (pochylenie)

            # type: ignore używamy, bo cv2 stubs nie radzą sobie z dynamicznym M
            sheared = cv.warpAffine(img, M, (c, r), flags=cv.INTER_NEAREST)  # type: ignore
            v_proj = np.sum(sheared, axis=0)
            return float(np.sum(v_proj ** 2))

        angles = np.deg2rad(np.arange(-30, 31, 2))
        # Obliczamy wyniki i rzutujemy argmax na int
        scores = [get_score(img_work, float(a)) for a in angles]
        best_angle = float(angles[int(np.argmax(scores))])

        if abs(best_angle) < 0.01:
            return image

        r, c = image.shape
        M_final = np.float32([[1, -np.tan(best_angle), 0], [0, 1, 0]])
        # Powrót do oryginalnej tonacji (borderValue=255 dla białego tła)
        res = cv.warpAffine(image, M_final, (c, r), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=255)  # type: ignore
        return np.asarray(res, dtype=np.uint8)

    @staticmethod
    def normalize_stroke_width(image: np.ndarray, target_width: int = 3) -> np.ndarray:
        """ Normalizacja grubości kreski pisma. """
        # Używamy prostszej binaryzacji do obliczeń
        binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        dist = cv.distanceTransform(binary, cv.DIST_L2, 3)  # Zmieniono 5 na 3 (szybciej)

        # Szybka erozja/dylatacja zamiast pętli warunkowych
        normalized = cv.threshold(dist, target_width / 2, 255, cv.THRESH_BINARY)[1]
        return normalized.astype(np.uint8)

    def connect_fragments(self, image: np.ndarray) -> np.ndarray:
        """ Łączy mikropęknięcia w literach wywołane słabym skanem. """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    @staticmethod
    def skeletonize_guohall(image: np.ndarray) -> np.ndarray:
        """ Zamienia litery na szkielet o grubości 1 piksela. Pomaga modelom HTR skupić się na kształcie zamiast na grubości atramentu. """
        # Rozdzielamy warunki, by zagwarantować linterowi zwrot typu np.ndarray
        if image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        if image.size == 0:
            return image

        try:
            binary_mask = (image < 128)
            skeleton = skeletonize(binary_mask)

            return np.where(skeleton, 0, 255).astype(np.uint8)

        except Exception as e:
            print(f"Error during skeletonization: {e}")
            return image

    @staticmethod
    def remove_bleed_through(image: np.ndarray) -> np.ndarray:
        """ Usuwa cienie liter przebijających z drugiej strony kartki.
            Wykorzystuje operację Black-Hat do izolacji tylko najciemniejszych struktur. """
        # Zmniejszono kernel z 15 na 11 dla szybkości
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
        blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
        res = cv.add(image, blackhat)
        return cv.normalize(res, None, 0, 255, cv.NORM_MINMAX)

    @staticmethod
    def apply_unsharp_mask(image: np.ndarray, sigma=1.0, strength=1.5):
        """ Zaawansowane ostrzenie krawędzi atramentu. """
        blurred = cv.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def standardize_ink_thickness(binary_inv_img, target_thickness=3):
        """ Eliminuje różnice między cienkim a grubym pismem. """
        # Zamiana na bool dla skimage
        binary_bool = binary_inv_img > 0

        # Szkieletowanie
        skeleton = skeletonize(binary_bool)
        skeleton_ubyte = (skeleton * 255).astype(np.uint8)

        # Kontrolowane pogrubienie
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (target_thickness, target_thickness))
        return cv.dilate(skeleton_ubyte, kernel, iterations=1)

    @staticmethod
    def enhance_archival_contrast(image: np.ndarray) -> np.ndarray:
        """ Adaptacyjne wyrównanie histogramu. Wydobywa wyblakły atrament bez prześwietlania tła. """
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def add_visual_padding(img, padding=4):
        """ Dodaje białe obramowanie, aby litery nie dotykały krawędzi wycinka. """
        return cv.copyMakeBorder(img, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)

    @staticmethod
    def remove_grid(img):
        """ Usuwa siatkę (kratkę/linie) przy użyciu filtrów morfologicznych, zachowując nienaruszony atrament pisma. """
        if img is None: return None

        # Progowanie adaptacyjne, aby wyodrębnić wszystkie ciemne elementy
        thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, 11, 2)

        # Wykrywanie linii poziomych
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        remove_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Wykrywanie linii pionowych
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))
        remove_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)

        # Łączymy maski linii
        grid_mask = cv.add(remove_horizontal, remove_vertical)

        # Delikatnie rozszerzamy maskę, by upewnić się, że usuniemy też krawędzie linii
        grid_mask = cv.dilate(grid_mask, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)

        # Wypełniamy miejsca pod maską kolorem tła
        result = cv.bitwise_or(img, grid_mask)

        return result

    @staticmethod
    def apply_sauvola_with_line_removal(img):
        """ Zaawansowana Sauvola z usuwaniem artefaktów linii. """
        img_float = img.astype(np.float32)

        # Standardowa Sauvola
        window_size = 25
        k = 0.2

        # Wymuszenie typu np.ndarray ucisza podejrzenia lintera o strukturę UMat
        mean = np.asarray(cv.blur(img_float, (window_size, window_size)), dtype=np.float32)
        sq_mean = np.asarray(cv.blur(img_float ** 2, (window_size, window_size)), dtype=np.float32)

        # Dodanie .0 do liczb wymusza operacje na czystych floatach
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
        threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
        binary = (img_float < threshold).astype(np.uint8) * 255

        # Usunięcie drobnych linii kratki
        kernel = np.ones((2, 2), np.uint8)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

        return binary

    @staticmethod
    def smart_binarization(img: np.ndarray) -> np.ndarray:
        """ ML: Używa K-Means do precyzyjnej binaryzacji lub adaptuje Sauvola dla dokumentów w kratkę. """
        # Wstępna analiza szumu (Canny + Hough), by wykryć linie
        edges = cv.Canny(img, 50, 150, apertureSize=3)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 5:
            # Brak linii -> Czysta kartka -> Używamy K-Means do separacji pikseli
            Z = img.reshape((-1, 1)).astype(np.float32)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv.kmeans(Z, 2, None, criteria, 3, cv.KMEANS_PP_CENTERS)

            # Identyfikacja, który klaster to atrament (ten ciemniejszy)
            ink_label = 0 if centers[0] < centers[1] else 1

            # Generowanie binarnej maski bezpośrednio z etykiet K-Means
            binary = np.where(labels.flatten() == ink_label, 0, 255).astype(np.uint8)
            return binary.reshape(img.shape)
        else:
            # Wykryto kratkę / linie ułożone w zeszyt — fallback do Sauvoli z usuwaniem linii
            return Preprocessing.apply_sauvola_with_line_removal(img)

    @staticmethod
    def preprocess_iam_dataset(raw_dir: str, output_dir: str, limit: int = None):
        """
        Automatyczne skanowanie i przetwarzanie bazy IAM.
        Parsuje strukturę folderów, czyta words.txt i przygotowuje dane pod CRNN/CapsNet.
        """
        from tqdm import tqdm
        import os

        # Ścieżka do głównego pliku etykiet IAM
        words_txt_path = os.path.join(raw_dir, "ascii", "words.txt")
        images_base_path = os.path.join(raw_dir, "words")

        if not os.path.exists(words_txt_path):
            print(f" BŁĄD: Nie znaleziono pliku etykiet w {words_txt_path}")
            return

        # Przygotowanie filtrów preprocessingu
        prep = Preprocessing()

        with open(words_txt_path, 'r') as f:
            lines = f.readlines()

        processed_count = 0
        # IAM words.txt zaczyna się od metadanych (linie na #)
        data_lines = [l for l in lines if not l.startswith("#")]

        if limit:
            data_lines = data_lines[:limit]

        print(f" Rozpoczynam przetwarzanie {len(data_lines)} próbek IAM...")

        for line in tqdm(data_lines):
            parts = line.strip().split(" ")
            if len(parts) < 9: continue

            # Format IAM: a01-000u-00-00 ok 154 408 768 27 51 AT A
            word_id = parts[0]
            status = parts[1]  # 'ok' lub 'err'
            label = parts[-1]

            if status != "ok": continue  # Pomijamy uszkodzone segmenty

            # Rozbicie ID na foldery: a01-000u-00-00 -> a01 / a01-000u / a01-000u-00-00.png
            id_parts = word_id.split("-")
            folder_l1 = id_parts[0]
            folder_l2 = f"{id_parts[0]}-{id_parts[1]}"
            img_path = os.path.join(images_base_path, folder_l1, folder_l2, f"{word_id}.png")

            if not os.path.exists(img_path):
                continue

            # 1. Wczytanie i wstępna obróbka
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None: continue

            # 2. Pipeline (używamy Twojej nowej logiki)
            # Możesz tu wybrać: full_pipeline lub konkretne kroki pod CRNN
            processed = prep.process_for_crnn(img)

            # 3. Zapis do nowej struktury (np. podział na znaki lub czyste słowa)
            # Tutaj decydujesz, czy idzie do folderu 'train' czy wg etykiet
            target_path = os.path.join(output_dir, f"{word_id}.png")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            cv.imwrite(target_path, processed)

            # Opcjonalnie: Zapis metadanych .npz, o których rozmawialiśmy
            # np.savez(target_path.replace(".png", ".npz"), gt=label, id=word_id)

            processed_count += 1

        print(f" Zakończono! Przetworzono pomyślnie {processed_count} plików.")

    @classmethod
    def deskew_image(cls, img):
        """  Prostuje pochylony znak przy użyciu momentów obrazu. """
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img

        # Obliczanie parametru pochylenia
        skew = m['mu11'] / m['mu02']

        # Tworzenie macierzy transformacji afinicznej (pochylenia)
        M = np.float32([[1, -skew, 0.5 * img.shape[0] * skew], [0, 1, 0]])

        # Zastosowanie transformacji
        img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
        return img

    @classmethod
    def process_and_size_norm(cls, crop_img, target_size=64, max_content=56):
        """ Lekkie formatowanie chroniące naturalną grubość atramentu.
            Skaluje znak tylko wtedy, gdy przekracza 56 pikseli.
            W przeciwnym razie zachowuje oryginalny rozmiar. """
        if crop_img is None or crop_img.size == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)

        # Próg globalny
        _, thresh = cv.threshold(crop_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Bounding box samego atramentu
        coords = cv.findNonZero(thresh)
        if coords is None:
            return np.zeros((target_size, target_size), dtype=np.uint8)

        x, y, w, h = cv.boundingRect(coords)
        ink = thresh[y:y + h, x:x + w]

        # Skalowanie warunkowe
        if w > max_content or h > max_content:
            scale = max_content / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            ink = cv.resize(ink, (nw, nh), interpolation=cv.INTER_AREA)
        else:
            nw, nh = w, h

        # Centrowanie na czystym czarnym płótnie
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        ox = (target_size - nw) // 2
        oy = (target_size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = ink

        return canvas

    @staticmethod
    def process_for_crnn(crop_img: np.ndarray, target_h: int = 64, target_w: int = 256) -> np.ndarray:
        """ Gwarantuje format: Czarny tekst, Białe tło, zachowane proporcje i centrowanie. """
        # Obsługa pustego wejścia
        if crop_img is None or crop_img.size == 0:
            return np.full((target_h, target_w), 255, dtype=np.uint8)

        # Wymuszenie typu i kopii
        img = np.asarray(crop_img, dtype=np.uint8)

        # Dynamiczna inwersja (bezpieczniejsza niż sprawdzanie jednego piksela [0,0])
        if float(np.mean(img).item()) < 127:
            img = cv.bitwise_not(img)

        h, w = img.shape[:2]

        # Obliczanie skali z zabezpieczeniem przed dzieleniem przez zero
        scale = float(target_h) / max(1, h)
        if (w * scale) > target_w:
            scale = float(target_w) / max(1, w)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # Skalowanie z jawnym rzutowaniem wymiarów
        resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

        # Inicjalizacja płótna
        canvas: np.ndarray = np.full((target_h, target_w), 255, dtype=np.uint8)

        # Centrowanie i wklejanie
        y_off = int((target_h - new_h) // 2)

        # Slice'owanie z jawnymi granicami
        canvas[y_off: y_off + new_h, 0: new_w] = resized

        return canvas

    @staticmethod
    def restore_broken_character(image: np.ndarray, gap_threshold: int = 3, iterations: int = 1) -> np.ndarray:
        """ Wykorzystuje mechanizm 'Balloon Force' do przywracania ciągłości przerwanych linii.
            Proces polega na generowaniu mapy odległości euklidesowej i wymuszaniu połączeń
            między segmentami atramentu znajdującymi się w bliskim sąsiedztwie. Pozwala to na
            domykanie pętli w literach (np. 'o', 'a', 'p') przy zachowaniu oryginalnej morfologii. """
        # Upewniamy się, że nie zwracamy typu None
        if image is None:
            return np.zeros((1, 1), dtype=np.uint8)

        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Binarizacja
        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        for _ in range(iterations):
            # Wyznaczamy dystans każdego piksela tła do najbliższego piksela atramentu
            dist_map = distance_transform_edt(binary == 0)

            #  Balloon Force: Łączymy przerwy mniejsze niż próg
            bridges = (dist_map <= gap_threshold).astype(np.uint8) * 255

            # Łączymy oryginalny atrament z nowymi połączeniami
            combined = cv.bitwise_or(binary, bridges)

            # Sprowadzamy połączoną strukturę do grubości 1px, aby usunąć zniekształcenia po dylatacji
            skeleton = skeletonize(combined > 0)
            binary = (skeleton * 255).astype(np.uint8)

        # Przywrócenie standardowej grubości dla lepszej jakości HCR
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        restored_fat = cv.dilate(binary, kernel, iterations=1)

        # Czarny tekst na Białym tle
        return cv.bitwise_not(restored_fat)

    def full_pipeline(self, page: np.ndarray) -> np.ndarray:
        """ Zintegrowany potok przetwarzania wstępnego z użyciem algorytmów ML. """
        # Wyostrzenie kontrastu oparte na klasteryzacji
        img = self.enhance_contrast_kmeans(page)

        # Wyeliminowanie gradientów i cieni
        img = self.remove_shadows(img)

        # Inteligentne prostowanie oparte na PCA
        img = self.correct_skew(img)

        # Usuwanie siatki z odbudową atramentu (inpaint)
        img = self.inpaint_line_removal(img)

        # Podbicie krawędzi atramentu
        img = self.apply_unsharp_mask(img)

        # Binaryzacja (K-Means dla czystych stron, Sauvola dla zeszytów)
        binary = self.smart_binarization(img)

        # Konwersja do białego tekstu na czarnym tle i normalizacja HTR
        binary_inv = cv.bitwise_not(binary)
        binary_inv = self.standardize_ink_thickness(binary_inv, target_thickness=3)
        binary_inv = cv.bitwise_not(self.area_filter(binary_inv, min_area=10))

        return cv.bitwise_not(binary_inv)

    @staticmethod
    def anisotropic_diffusion(img: np.ndarray, iterations: int = 5, kappa: int = 50, gamma: float = 0.1):
        """ Wygładzanie krawędziowe. Kappa steruje czułością na krawędzie, gamma szybkością dyfuzji. """
        img = img.astype(np.float32)
        for _ in range(iterations):
            # Obliczanie gradientów
            grad_n = np.roll(img, -1, axis=0) - img
            grad_s = np.roll(img, 1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img

            # Funkcja przewodności (Perona-Malik)
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)

            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def normalize_thickness(binary_inv_img: np.ndarray, target_thickness: int = 3):
        """ Normalizacja grubości kreski przy użyciu transformaty dystansowej. """
        dist_transform = cv.distanceTransform(binary_inv_img, cv.DIST_L2, 5)

        # Próg dobieramy tak, by uzyskać pożądaną grubość
        normalized = cv.threshold(dist_transform, target_thickness / 2, 255, cv.THRESH_BINARY)[1]
        return normalized.astype(np.uint8)

    @staticmethod
    def dewarp_baseline(binary_inv_img: np.ndarray):
        """ Prostowanie nieliniowe linii bazowej słowa. """
        h, w = binary_inv_img.shape
        # Wykrywanie punktów dolnej krawędzi
        baseline_points = []
        for x in range(0, w, 10):
            col = binary_inv_img[:, x]
            indices = np.where(col > 0)[0]
            if len(indices) > 0:
                baseline_points.append((x, indices[-1]))

        if len(baseline_points) < 2: return binary_inv_img

        pts = np.array(baseline_points)
        z = np.polyfit(pts[:, 0], pts[:, 1], 2)
        p = np.poly1d(z)

        # Przesuwanie pikseli w pionie zgodnie z krzywą
        corrected = np.zeros_like(binary_inv_img)
        target_y = int(np.median(pts[:, 1]))
        for x in range(w):
            shift = target_y - int(p(x))
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            corrected[:, x:x + 1] = cv.warpAffine(binary_inv_img[:, x:x + 1], M, (1, h))

        return corrected

    @staticmethod
    def inpaint_line_removal(image: np.ndarray):
        """ Adaptacyjne usuwanie kratki i linii. """
        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Szukamy tylko struktur, które stanowią np. 1/8 szerokości/wysokości całej strony, żeby nie usunąć np. 'l'.
        h_len = max(50, image.shape[1] // 8)
        v_len = max(50, image.shape[0] // 8)

        h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (h_len, 1))
        v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, v_len))

        # Wykrywanie gigantycznych linii
        h_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, h_kernel)
        v_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, v_kernel)

        # Jeśli strona ma tylko poziome linie, maska v_lines będzie po prostu pusta (czarna)
        full_mask = cv.add(h_lines, v_lines)

        # Poszerzamy delikatnie maskę, aby objęła poszarpane krawędzie ołówka/długopisu
        full_mask = cv.dilate(full_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Inpainting — inteligentne łatanie wyciętych dziur
        return cv.inpaint(image, full_mask, inpaintRadius=5, flags=cv.INPAINT_TELEA)

    @staticmethod
    def binarize_multiscale(img: np.ndarray):
        """ Binarizacja wieloskalowa dla trudnych dokumentów. """
        # Trzy skale okna
        scales = [31, 61, 121]
        results = []
        for s in scales:
            thresh = threshold_sauvola(img, window_size=s, k=0.1)
            results.append(np.where(img > thresh, 255, 0).astype(np.uint8))

        # Logika większościowa
        final = np.median(np.stack(results), axis=0).astype(np.uint8)
        return final

    @staticmethod
    def binarize_fast(img: np.ndarray) -> np.ndarray:
        """ Szybki zamiennik dla Binarize Multiscale.
            Adaptive Threshold (Gaussian) jest błyskawiczny (C++ backend). """
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 31, 10)

    @staticmethod
    def dilate_fn(image, **kwargs):
        """ Symulacja rozlanego tuszu lub mocnego nacisku. """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (random.randint(2, 3), random.randint(2, 3)))
        return cv.dilate(image, kernel, iterations=1)

    @staticmethod
    def erode_fn(image, **kwargs):
        """ Symulacja cienkiego pióra lub kończącego się tuszu. """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        return cv.erode(image, kernel, iterations=1)

    @classmethod
    def get_htr_augmentations(cls, phase="main"):
        """ Definiuje strategię augmentacji w zależności od fazy treningu (Curriculum Learning).
        1. Faza 'main': Agresywne zniekształcenia geometryczne (ElasticTransform, GridDistortion)
           oraz morfologiczne. Cel: Wymuszenie nauki niezmiennych cech kształtu (inwariantność).
        2. Faza 'fine_tune': Wyłączenie dystorsji geometrycznych. Pozostawienie jedynie
           szumu i rozmycia. Cel: Stabilizacja wag i douczenie na realistycznych artefaktach,
           bez ryzyka niszczenia czytelności liter. """
        if phase == "main":
            return alb.Compose([
                alb.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.1, rotate_limit=8, p=0.6,
                                     border_mode=cv.BORDER_CONSTANT),
                alb.OneOf([
                    alb.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
                    alb.ElasticTransform(alpha=1, sigma=40, p=1.0),
                ], p=0.4),
                alb.OneOf([
                    alb.Lambda(name="Dilation", image=cls.dilate_fn, p=1.0),
                    alb.Lambda(name="Erosion", image=cls.erode_fn, p=1.0),
                ], p=0.3),
                alb.GaussNoise(std_range=(10.0, 30.0), p=0.3),
                alb.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:  # fine_tune
            return alb.Compose([
                alb.ShiftScaleRotate(shift_limit=0.02, rotate_limit=3, p=0.3, border_mode=cv.BORDER_CONSTANT),
                alb.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                alb.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])


class CRAMBlock(nn.Module):
    """ Customized Residual Attention Module — odszumia tło i uwydatnia pismo. """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Uwaga kanałowa
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Uwaga przestrzenna
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = func.relu(self.bn1(self.conv1(x)))
        # Aplikacja uwagi
        x = x * self.ca(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([max_out, avg_out], dim=1))
        return x + res


def normalize_x_height(image: np.ndarray, target_corpus_h: int = 32) -> np.ndarray:
    """ Normalizacja wysokości korpusu pisma (x-height). """
    # Jawnie rzutujemy na ndarray, by linter nie miał wątpliwości
    img: np.ndarray = np.asarray(image, dtype=np.uint8)

    if int(np.sum(img == 0).item()) == 0:
        return img

    # Obliczamy projekcję
    work: np.ndarray = cv.bitwise_not(img) if float(np.mean(img).item()) > 127 else img.copy()
    h_proj: np.ndarray = np.sum(work, axis=1, dtype=np.int32)

    # .item() to jedyny sposób, by wyciągnąć "czysty" float/int, który linter zaakceptuje
    max_val = float(np.max(h_proj).item())
    thresh = max_val * 0.5

    indices = np.where(h_proj > thresh)[0]
    if indices.size < 2:
        return img

    # Obliczanie skali
    current_h = int(indices[-1].item()) - int(indices[0].item())
    scale = float(target_corpus_h) / (float(current_h) + 1e-6)

    # Wymiary muszą być jawnymi intami Pythona dla cv.resize
    nw = int(img.shape[1] * scale)
    nh = int(img.shape[0] * scale)

    # Ostateczne uciszenie lintera przez castowanie dsize
    return cv.resize(img, (max(1, nw), max(1, nh)), interpolation=cv.INTER_LINEAR)


def visualize_full_segmentation(original: np.ndarray, line_images: list[np.ndarray], word_segmentor):
    """ Wizualizacja hierarchiczna: Linie i wyodrębnione z nich słowa. """
    num_lines = len(line_images)
    if num_lines == 0:
        return

    # Tworzymy duży arkusz: każda linia to oddzielny wiersz na wykresie
    plt.figure(figsize=(20, 4 * num_lines))

    for i, line_raw in enumerate(line_images):
        # Upewniamy się, że linia jest w uint8 (ucisza lintera)
        line_img = np.asarray(line_raw, dtype=np.uint8)

        # Pobieramy wycinki słów
        word_crops = word_segmentor.extract_atomic_crops(line_img)

        # Obraz do  ramek
        line_rgb = cv.cvtColor(line_img, cv.COLOR_GRAY2RGB)
        for _, (x, y, w, h) in word_crops:
            # Jawne rzutowanie na int() dla cv2.rectangle
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv.rectangle(line_rgb, p1, p2, (0, 255, 0), 2)

        # Wyświetlanie linii z zaznaczonymi słowami
        plt.subplot(num_lines, 1, i + 1)
        plt.imshow(line_rgb)
        plt.title(f"Linia {i + 1}: Wykryto {len(word_crops)} jednostek atomowych")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_word_segmentation(lines, word_segmentor):
    """ Wyświetla wyprostowane linie z narysowanymi ramkami słów. """
    if not lines:
        return

    plt.figure(figsize=(18, 2.5 * len(lines)))

    for i, line_img in enumerate(lines):
        # Upewniamy się, że line_img to czysty uint8 dla segmentora
        img_input = np.asarray(line_img, dtype=np.uint8)
        words = word_segmentor.extract_atomic_crops(img_input)

        # Prostujemy i wymuszamy uint8, by cv.cvtColor nie krzyczało
        display_bg = np.asarray(word_segmentor.deslant_img(img_input), dtype=np.uint8)
        display_img = cv.cvtColor(display_bg, cv.COLOR_GRAY2RGB)

        # Rysowanie ramek - display_img jest już traktowane jako bezpieczny kontener
        for _, (x, y, w, h) in words:
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
            cv.rectangle(display_img, start_point, end_point, (0, 200, 0), 2)

        plt.subplot(len(lines), 1, i + 1)
        plt.title(f"Linia {i + 1} - Wykryto {len(words)} słów")
        plt.imshow(display_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Tylko dla CVL
def crop_cvl_content(img, padding=20):
    """ Odcina drukowany nagłówek CVL i przycina marginesy. """
    h, w = img.shape

    # Odcięcie górnych 33% strony (CVL ma zapiski na górze)
    cropped_top = img[int(h * 0.33):, :]

    # Standardowe przycięcie marginesów bocznych i dolnych
    _, thresh = cv.threshold(cropped_top, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = cv.medianBlur(thresh, 5)

    coords = cv.findNonZero(thresh)
    if coords is not None:
        x, y, bw, bh = cv.boundingRect(coords)
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + bw + padding), min(cropped_top.shape[0], y + bh + padding)
        return cropped_top[y1:y2, x1:x2]

    return cropped_top

if __name__ == "__main__":
    TEST_IMAGE_PATH = r"C:\OCR\cvl-database\testset\pages\0052-2.tif"

    # Inicjalizacja klas
    preprocessor = Preprocessing()
    line_seg = PageToLineSegmentor()
    word_seg = LineToWordSegmentor()

    print(f"Test na obrazie: {os.path.basename(TEST_IMAGE_PATH)}")

    # Wczytanie obrazu
    raw_img = cv.imread(TEST_IMAGE_PATH, cv.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("Nie znaleziono obrazu.")
    else:
        # Cięcie dla CVL
        cropped_img = crop_cvl_content(raw_img)

        # Preprocessing
        processed_inv = preprocessor.full_pipeline(cropped_img)

        # Segmentacja linii
        processed_norm = cv.bitwise_not(processed_inv)
        lines, seams = line_seg.extract_lines(processed_norm)

        # Ekstrakcja słów
        if lines:
            print(f"Wykryto {len(lines)} linii. Ekstrakcja słów.")
            words_line_1 = word_seg.extract_atomic_crops(lines[0])
            print(f"Sukces: Wycięto {len(words_line_1)} słów z pierwszej linii.")

        print("Generowanie wizualizacji.")

        # Wizualizacja wyciętych słów dla całej strony
        visualize_word_segmentation(lines, word_seg)