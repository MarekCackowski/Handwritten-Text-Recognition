import io
import os
import numpy as np
import cv2 as cv
from skimage import io, transform, measure, morphology, util
from skimage.util import img_as_ubyte
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from scipy.signal import find_peaks
from HandwrittenTextSegmentation.Preprocessing import Preprocessing
from HandwrittenTextSegmentation.OpticalLayoutRecognition import OpticalLayoutRecognition

FOLDER_PATH = r'C:\Users\marek\OneDrive\Pulpit\mg\test'
OUTPUT_DIR_NAME = 'segmented_words'

WORD_TARGET_HEIGHT = 64
WORD_TARGET_WIDTH = 256

CHAR_TARGET_HEIGHT = 28
CHAR_TARGET_WIDTH = 28

def load_images(path):
    if not os.path.exists(path):
        print("Podana ścieżka nie prowadzi do folderu")
        exit()
    imgs = [img for img in os.listdir(path) if img.lower().endswith('.png')]
    if len(imgs) == 0:
        print("Podany folder nie zawiera obrazów png")
        exit()
    return imgs


def draw_histogram_columns(img, pscs):
    height, width = img.shape
    draw = img.copy()
    for col in pscs:
        if 0 <= col < width:
            draw[:, col] = 0
    return draw


def show_img_and_columns(img, pscs):
    _, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(draw_histogram_columns(img, pscs), cmap='gray')
    plt.show()


def show_imgs(img1, img2):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')
    plt.show()


def show_words_on_page(words_imgs, words_segments, words_coords, words_shapes, whole_page):
    if len(whole_page.shape) == 2:
        page_with_words = cv.cvtColor(whole_page, cv.COLOR_GRAY2BGR)
    else:
        page_with_words = whole_page.copy()

    olr = OpticalLayoutRecognition()

    for w_img, s, (x, y), (h, w) in zip(words_imgs, words_segments, words_coords, words_shapes):
        x_start, y_start = int(x), int(y)
        x_end, y_end = int(x + w), int(y + h)

        cv.rectangle(page_with_words, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        seg_on_page = olr.flatten_to_ints(s)
        for col in seg_on_page:
            col_pos = x_start + col
            if 0 <= col_pos < whole_page.shape[1]:
                cv.line(page_with_words, (col_pos, y_start), (col_pos, y_end), (0, 0, 255), 1)

    plt.imshow(cv.cvtColor(page_with_words, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def sort_regions_reading_order(regions, line_tolerance=40):
    """ Logika sortowania: najpierw wiersze, potem kolumny. """
    if not regions:
        return []

    # Wstępne sortowanie po górnej krawędzi (Y)
    initial_sorted = sorted(regions, key=lambda r: r.bbox[0])

    final_sorted = []
    current_line = [initial_sorted[0]]
    #  Współrzędna Y środka ciężkości
    current_y = initial_sorted[0].centroid[0]

    for region in initial_sorted[1:]:
        y_center = region.centroid[0]

        if abs(y_center - current_y) < line_tolerance:
            current_line.append(region)
        else:
            # Sortujemy ukończoną linię od lewej do prawej (X)
            final_sorted.extend(sorted(current_line, key=lambda r: r.bbox[1]))
            current_line = [region]
            current_y = y_center

    # Dodanie ostatniej linii
    final_sorted.extend(sorted(current_line, key=lambda r: r.bbox[1]))
    return final_sorted


class DocumentProcessor:
    def __init__(self, window_size=25, k_niblack=-0.2, k_sauvola=0.2, r_dynamic=128):
        """ Inicjalizacja procesora dokumentów. """
        if window_size % 2 == 0:
            raise ValueError("Rozmiar okna musi być liczbą nieparzystą.")
        self.window_size = window_size
        self.k_niblack = k_niblack
        self.k_sauvola = k_sauvola
        self.r = r_dynamic

    @staticmethod
    def remove_lines(image):
        """ Usuwanie linii z wykorzystaniem inpaintingu. """
        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))
        h_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, h_kernel)
        v_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, v_kernel)
        line_mask = cv.add(h_lines, v_lines)
        line_mask = cv.dilate(line_mask, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        return cv.inpaint(image, line_mask, 3, cv.INPAINT_TELEA), line_mask  #

    def compute_threshold_maps(self, image):
        """ Oblicza progi lokalne dla Niblacka i Sauvola. """
        img_float = image.astype(np.float32)
        mean = cv.boxFilter(img_float, cv.CV_32F, (self.window_size, self.window_size))
        sq_mean = cv.boxFilter(img_float ** 2, cv.CV_32F, (self.window_size, self.window_size))
        std_dev = np.sqrt(np.clip(sq_mean - (mean ** 2), 0, None))

        t_niblack = mean + self.k_niblack * std_dev
        t_sauvola = mean * (1 + self.k_sauvola * ((std_dev / self.r) - 1))
        return t_niblack, t_sauvola  #

    def hybrid_binarization(self, image):
        """ Binaryzacja hybrydowa. """
        t_n, t_s = self.compute_threshold_maps(image)
        t_hybrid = (t_n + t_s) / 2.0
        return (image > t_hybrid).astype(np.uint8) * 255  #

    def full_pipeline(self, image_path):
        """ Pełny proces: wczytanie -> usuwanie linii -> binaryzacja. """
        original = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if original is None: return None
        cleaned, _ = self.remove_lines(original)
        return self.hybrid_binarization(cleaned)  #

def main():
    doc_processor = DocumentProcessor(window_size=25, k_niblack=-0.2, k_sauvola=0.2)
    olr = OpticalLayoutRecognition()

    images = load_images(FOLDER_PATH)

    if not images:
        print("Nie znaleziono obrazów w folderze.")
        return

    for image_name in images:
        image_path = os.path.join(FOLDER_PATH, image_name)
        print(f"Przetwarzanie: {image_name}.")

        try:
            # Pełny preprocessing
            page_clean = doc_processor.full_pipeline(image_path)
            if page_clean is None: continue

            # Segmentacja na wyrazy
            binary_text = (page_clean == 0)

            # Dylatacja pozioma do połączenia liter w słowa
            dilate_kernel = morphology.rectangle(3, 15)
            word_blobs = morphology.binary_dilation(binary_text, footprint=dilate_kernel)

            # Etykietowanie regionów
            label_image = measure.label(word_blobs)
            regions = measure.regionprops(label_image)

            # Filtrowanie małych artefaktów
            valid_regions = [r for r in regions if r.area > 150]

            # Sortowanie w kolejności czytania
            sorted_regions = sort_regions_reading_order(valid_regions, line_tolerance=40)

            # Zapis wyciętych słów
            image_base_name = os.path.splitext(image_name)[0]
            image_output_folder = os.path.join(FOLDER_PATH, image_base_name)
            os.makedirs(image_output_folder, exist_ok=True)

            for word_idx, region in enumerate(sorted_regions):
                y1, x1, y2, x2 = region.bbox

                # Padding dla bezpieczeństwa krawędzi
                pad = 4
                y1_p = max(0, y1 - pad)
                x1_p = max(0, x1 - pad)
                y2_p = min(page_clean.shape[0], y2 + pad)
                x2_p = min(page_clean.shape[1], x2 + pad)

                word_crop = page_clean[y1_p:y2_p, x1_p:x2_p]
                h, w = word_crop.shape
                if h == 0 or w == 0: continue

                # Skalowanie do docelowej wysokości modelu OCR
                scale = WORD_TARGET_HEIGHT / float(h)
                new_w = int(w * scale)

                word_resized = transform.resize(word_crop, (WORD_TARGET_HEIGHT, new_w),
                                                order=1, mode='constant', cval=1.0, anti_aliasing=True)
                word_resized = img_as_ubyte(word_resized)

                # Tworzenie kanwy o stałej szerokości
                final_word_img = np.ones((WORD_TARGET_HEIGHT, WORD_TARGET_WIDTH), dtype=np.uint8) * 255

                if new_w <= WORD_TARGET_WIDTH:
                    final_word_img[:, 0:new_w] = word_resized
                else:
                    # Jeśli słowo jest za długie — kompresujemy szerokość
                    word_compressed = transform.resize(word_resized, (WORD_TARGET_HEIGHT, WORD_TARGET_WIDTH), order=1, mode='constant', cval=1.0)
                    final_word_img = img_as_ubyte(word_compressed)

                save_filename = f'word_{word_idx + 1:03d}.png'
                io.imsave(os.path.join(image_output_folder, save_filename), final_word_img, check_contrast=False)

        except Exception as e:
            print(f"Błąd podczas przetwarzania {image_name}: {e}")
            continue

    print("Proces zakończenia segmentacji i preprocessingu powiódł się.")


if __name__ == "__main__":
    main()
