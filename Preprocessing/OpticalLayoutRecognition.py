import random
import numpy as np
import cv2 as cv
from typing import List, cast, Any
import pickle
import warnings
import os
warnings.filterwarnings("ignore")

MAX_WORD_WIDTH = 800  # Próg bezpieczeństwa dla okna niby 564px (z zapasem 576)

class SegmentationRefiner:
    def __init__(self, model_path='segment_rf.pkl'):
        self.model_path = model_path
        self.rf_model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"[Refiner] Ładowanie modelu RF z {self.model_path}.")
                with open(self.model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[Refiner] Error {e}")
                return None
        else:
            return None

    @staticmethod
    def _extract_refinement_features(segment_image: np.ndarray, cut_column: int) -> np.ndarray:
        """ Wyodrębnia lokalne cechy geometryczne dla danego kandydata cięcia. """
        H, W = segment_image.shape

        # Jawne rzutowanie na float() przed dzieleniem
        cut_density = float(np.sum(segment_image[:, cut_column] == 0)) / H

        # Gęstość pociągnięcia w strefie środkowej
        mid_zone_start = int(H * 0.25)
        mid_zone_end = int(H * 0.75)
        denom = float(mid_zone_end - mid_zone_start)
        if denom == 0: denom = 1.0

        mid_zone_density = float(np.sum(segment_image[mid_zone_start:mid_zone_end, cut_column] == 0)) / denom

        # Kąt pochylenia
        slant_angle = float(np.abs(np.tan(np.pi / 180 * 15)))

        # Wysokość elementu po prawej i masa po lewej
        right_window = segment_image[:, min(W - 1, cut_column + 1):min(W, cut_column + 11)]
        if right_window.size > 0:
            top_pixels = np.where(right_window == 0)[0]
            top_y = float(np.min(top_pixels)) if top_pixels.size > 0 else float(H)
            right_ascender_height = 1.0 - (top_y / H)
        else:
            right_ascender_height = 0.0

        left_window = segment_image[:, max(0, cut_column - 10):cut_column]
        if left_window.size > 0:
            left_mass = float(np.sum(left_window == 0)) / (H * 10.0)
        else:
            left_mass = 0.0

        return np.array([cut_density, mid_zone_density, slant_angle, right_ascender_height, left_mass], dtype=np.float32)

    def filter_cuts(self, word_image: np.ndarray, candidate_cuts: List[int]) -> List[int]:
        """ Filtruje kandydatów cięcia. ML weryfikuje wszystkie podejrzane i gęste połączenia. """
        final_cuts = []

        for cut_column in candidate_cuts:
            features = self._extract_refinement_features(word_image, cut_column)

            # Rzutowanie na float, aby uniknąć błędów _ScalarT
            cut_density = float(features[0])
            mid_zone_density = float(features[1])

            is_ambiguous_cut = (cut_density > 0.02) or (mid_zone_density > 0.4)

            if is_ambiguous_cut:
                if self.rf_model is not None:
                    # 'cast' wymusza na linterze traktowanie modelu jako obiektu posiadającego metody, to wycisza błąd
                    model_to_use = cast(Any, self.rf_model)
                    try:
                        prediction = model_to_use.predict(features.reshape(1, -1))
                        if int(prediction[0]) == 1:
                            final_cuts.append(cut_column)
                    except (AttributeError, ValueError, TypeError):
                        # W przypadku błędu modelu bezpieczniej jest zachować cięcie
                        final_cuts.append(cut_column)
                else:
                    # Fallback dla braku modelu
                    if random.random() < 0.8:
                        final_cuts.append(cut_column)
            else:
                # Wyraźna przerwa w atramencie
                final_cuts.append(cut_column)

        return final_cuts


class OpticalLayoutRecognition:
    def __init__(self):
        # Progi dla segmentacji linii
        self.separation_threshold = 0.8
        self.exclusion_threshold = 0.6

    @staticmethod
    def create_y_histogram(image):
        """ Rzut pionowy obrazu — podstawa wykrywania wierszy. """
        image = cv.bitwise_not(image)
        return np.sum(image, axis=1)

    def lines_segmentation_local(self, image, num_strips=5):
        """ Wykrywanie wierszy z podziałem na pionowe pasy (odporne na pochylenie tekstu).
            Zwraca binarny histogram, gdzie 1 oznacza obszar tekstu wiersza. """
        h, w = image.shape
        strip_width = w // num_strips
        all_bin_hists = np.zeros((num_strips, h), dtype=int)

        for i in range(num_strips):
            x_start = i * strip_width
            x_end = (i + 1) * strip_width if i < num_strips - 1 else w
            strip = image[:, x_start:x_end]

            hist_y = self.create_y_histogram(strip)
            # Progowanie linii
            threshold = np.mean(hist_y) * self.separation_threshold
            bin_hist = (hist_y > threshold).astype(int)

            # Usuwanie fałszywych (zbyt cienkich) linii
            bin_hist = self.false_line_exclusion(bin_hist)
            all_bin_hists[i] = bin_hist

        # Głosowanie między pasami i finalne wygładzenie
        combined_hist = (np.mean(all_bin_hists, axis=0) > 0.4).astype(int)
        return combined_hist

    def false_line_exclusion(self, binary_hist):
        """ Usuwa regiony, które są zbyt niskie, by być wierszem tekstu. """
        y_initial = np.where(np.diff(binary_hist, prepend=0) == 1)[0]
        y_final = np.where(np.diff(binary_hist, append=0) == -1)[0]

        if len(y_initial) == 0: return binary_hist

        avg_height = np.mean(y_final - y_initial)

        # .item() wyciąga czystą wartość ze skalara NumPy, żeby nie było warninga
        threshold = avg_height.item() * self.exclusion_threshold

        result_hist = binary_hist.copy()
        for start, end in zip(y_initial, y_final):
            if float(avg_height) < threshold:
                result_hist[start:end] = 0
        return result_hist

    def prepare_segments(self, image):
        """ Wykrywa wszystkie plamy atramentu (kontury) na stronie. """
        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        segments = []
        for i, contour in enumerate(contours):
            if i != 0:  # Pomijamy ramkę całej strony
                x, y, w, h = cv.boundingRect(contour)
                segments.append([x, y, w, h])

        # Filtrowanie konturów wewnętrznych (np. środek litery 'o')
        return self.delete_internal_segments(segments)

    @staticmethod
    def delete_internal_segments(segments):
        not_internal = []
        for i, s1 in enumerate(segments):
            x1, y1, w1, h1 = s1
            is_internal = False
            for j, s2 in enumerate(segments):
                if i == j: continue
                x2, y2, w2, h2 = s2
                if x1 > x2 and x1 + w1 < x2 + w2 and y1 > y2 and y1 + h1 < y2 + h2:
                    is_internal = True
                    break
            if not is_internal:
                not_internal.append(s1)
        return not_internal

    def group_in_words(self, segments_in_line):
        """ Grupuje pojedyncze znaki/plamy w całe słowa na podstawie odstępów. """
        if not segments_in_line: return []

        segments_in_line = sorted(segments_in_line, key=lambda s: s[0])
        words = []
        current_word = [segments_in_line[0]]

        # Obliczamy próg odstępu (dynamiczny dla danej linii)
        threshold = self.calculate_average_spacing(segments_in_line)

        for i in range(1, len(segments_in_line)):
            prev_seg = segments_in_line[i - 1]
            curr_seg = segments_in_line[i]

            # Przerwa między krawędzią poprzedniego znaku a początkiem obecnego segmentu
            gap = curr_seg[0] - (prev_seg[0] + prev_seg[2])

            if gap < threshold:
                current_word.append(curr_seg)
            else:
                words.append(current_word)
                current_word = [curr_seg]

        words.append(current_word)
        return words

    @staticmethod
    def calculate_average_spacing(segments):
        """ Szuka progu rozdzielającego litery od słów (statystyka przerw). """
        if len(segments) <= 1: return 20
        spacings = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1][0] - (segments[i][0] + segments[i][2])
            if gap > 0: spacings.append(gap)

        if not spacings: return 15
        median_gap = np.median(spacings)
        std_gap = np.std(spacings)
        return max(median_gap + (1.5 * std_gap), median_gap * 1.2)

    def get_word_boxes(self, image):
        """
        Główna funkcja: Page -> Lines -> Word Boxes.
        Zwraca listę [x, y, w, h] dla każdego słowa na stronie.
        """
        # 1. Wykryj linie (lokalnie)
        bin_hist = self.lines_segmentation_local(image)

        # 2. Wykryj wszystkie kontury
        all_segments = self.prepare_segments(image)

        # 3. Przypisz kontury do linii
        y_initial = np.where(np.diff(bin_hist, prepend=0) == 1)[0]
        y_final = np.where(np.diff(bin_hist, append=0) == -1)[0]

        all_word_boxes = []
        for i in range(len(y_initial)):
            line_start, line_end = y_initial[i], y_final[i]
            # Filtruj segmenty należące do tej linii
            line_segments = [s for s in all_segments if line_start < (s[1] + s[3] / 2) < line_end]

            # 4. Grupuj w słowa i wyznacz ramki otaczające
            words = self.group_in_words(line_segments)
            for word in words:
                x_min = min(s[0] for s in word)
                y_min = min(s[1] for s in word)
                x_max = max(s[0] + s[2] for s in word)
                y_max = max(s[1] + s[3] for s in word)
                all_word_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

        return all_word_boxes


class PageLayoutManager:
    """ Silnik analizy układu strony. Segmentuje obraz na linie i słowa, przygotowując paczki dla CRNN."""
    def __init__(self, yolo_model_path: str = None):
        self.separation_threshold = 0.8
        self.exclusion_threshold = 0.6
        self.yolo_path = yolo_model_path
        # Tutaj w przyszłości: self.yolo = YOLO(yolo_model_path)

    @staticmethod
    def create_y_histogram(image: np.ndarray) -> np.ndarray:
        """ Rzut pionowy — zakłada biały tekst na czarnym tle. """
        # Sprawdzamy czy obraz nie jest pusty
        if image.size == 0: return np.array([])
        return np.sum(image, axis=1)

    def lines_segmentation_local(self, image: np.ndarray, num_strips: int = 5) -> np.ndarray:
        """ Lokalna analiza histogramu na pionowych pasach. Pozwala na wykrycie linii nawet przy lekkim pofalowaniu tekstu. """
        h, w = image.shape[:2]
        strip_width = w // num_strips
        all_bin_hists = np.zeros((num_strips, h), dtype=int)

        # Inwersja do histogramu (tekst musi być "jasny")
        inv_img = cv.bitwise_not(image) if np.mean(image) > 127 else image

        for i in range(num_strips):
            x_start = i * strip_width
            x_end = (i + 1) * strip_width if i < num_strips - 1 else w
            strip = inv_img[:, x_start:x_end]

            hist_y = self.create_y_histogram(strip)
            if hist_y.size == 0: continue

            threshold = np.mean(hist_y) * self.separation_threshold
            all_bin_hists[i] = (hist_y > threshold).astype(int)

        combined_hist = (np.mean(all_bin_hists, axis=0) > 0.4).astype(int)
        return self._false_line_exclusion(combined_hist)

    def _false_line_exclusion(self, binary_hist: np.ndarray) -> np.ndarray:
        """ Eliminuje artefakty i szumy mniejsze niż ułamek średniej wysokości linii. """
        y_init = np.where(np.diff(binary_hist, prepend=0) == 1)[0]
        y_final = np.where(np.diff(binary_hist, append=0) == -1)[0]

        if len(y_init) == 0: return binary_hist

        avg_h = float(np.mean(y_final - y_init))
        res_hist = binary_hist.copy()
        threshold = avg_h * self.exclusion_threshold

        for s, e in zip(y_init, y_final):
            if int(e - s) < threshold:
                res_hist[s:e] = 0
        return res_hist

    def get_word_crops_for_ai(self, image: np.ndarray):
        """
        Główna metoda inferencyjna: wycina słowa z pełnej strony.
        Zwraca listę [crop, (x, y, w, h)].
        """
        # Przygotowanie obrazu pod kontury (musi być biały tekst na czarnym tle)
        work_img = cv.bitwise_not(image) if np.mean(image) > 127 else image.copy()

        # Znajdź linie
        bin_hist = self.lines_segmentation_local(image)

        # Znajdź plamy atramentu
        contours, _ = cv.findContours(work_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        segments = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 10: continue  # Ignoruj kropki i szum
            x, y, w, h = cv.boundingRect(cnt)
            segments.append([x, y, w, h])

        segments = self._delete_internal(segments)

        y_init = np.where(np.diff(bin_hist, prepend=0) == 1)[0]
        y_final = np.where(np.diff(bin_hist, append=0) == -1)[0]

        all_word_data = []
        for i in range(len(y_init)):
            l_start, l_end = y_init[i], y_final[i]
            # Kontury, których środek ciężkości leży wewnątrz wykrytej linii
            line_segments = [s for s in segments if l_start < (s[1] + s[3] / 2) < l_end]

            if not line_segments: continue

            words = self._group_in_words(line_segments)
            for word in words:
                x1, y1 = min(s[0] for s in word), min(s[1] for s in word)
                x2, y2 = max(s[0] + s[2] for s in word), max(s[1] + s[3] for s in word)

                pad = 4
                # Wycinek z ORYGINALNEGO obrazu
                crop = image[max(0, y1 - pad):min(image.shape[0], y2 + pad),
                max(0, x1 - pad):min(image.shape[1], x2 + pad)]

                all_word_data.append([crop, (x1, y1, x2 - x1, y2 - y1)])
        return all_word_data

    @staticmethod
    def _group_in_words(segments):
        if not segments: return []
        segments = sorted(segments, key=lambda s: s[0])
        words, current_word = [], [segments[0]]

        spacings = [segments[i + 1][0] - (segments[i][0] + segments[i][2])
                    for i in range(len(segments) - 1)]
        # Filtrujemy ujemne odstępy (nakładające się znaki)
        spacings = [s for s in spacings if s > 0]

        # Próg odległości między słowami: 1.5x mediana odstępów między znakami
        threshold = np.median(spacings) * 1.5 if spacings else 25

        for i in range(1, len(segments)):
            gap = segments[i][0] - (segments[i - 1][0] + segments[i - 1][2])
            if gap < threshold:
                current_word.append(segments[i])
            else:
                words.append(current_word)
                current_word = [segments[i]]
        words.append(current_word)
        return words

    @staticmethod
    def _delete_internal(segments):
        """ Usuwa kontury, które w całości zawierają się w innych (np. oczko w 'o'). """
        if not segments: return []
        res = []
        for i, s1 in enumerate(segments):
            is_int = False
            for j, s2 in enumerate(segments):
                if i != j:
                    # Sprawdź czy s1 zawiera się w s2
                    if (s1[0] >= s2[0] and s1[0] + s1[2] <= s2[0] + s2[2] and
                            s1[1] >= s2[1] and s1[1] + s1[3] <= s2[1] + s2[3]):
                        is_int = True
                        break
            if not is_int: res.append(s1)
        return res

class LineToWordSegmentor:
    @staticmethod
    def deslant_img(img):
        """ Niezawodna metoda prostowania kursywy (Vinciarelli). """
        if img is None or np.sum(img) == 0: return img

        _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        h, w = binary.shape

        best_var = 0
        best_angle = 0
        angles = np.arange(-40, 41, 2)

        for angle in angles:
            alpha = np.tan(np.deg2rad(angle))
            shift = abs(alpha) * h
            new_w = w + int(shift)
            offset = shift if alpha > 0 else 0
            M = np.float32([[1, -alpha, offset], [0, 1, 0]])
            sheared = cv.warpAffine(binary, M, (new_w, h), flags=cv.INTER_NEAREST)
            proj = np.sum(sheared, axis=0)

            var = np.var(proj)
            if var > best_var:
                best_var = var
                best_angle = angle

        if abs(best_angle) > 2:
            alpha = np.tan(np.deg2rad(best_angle))
            shift = abs(alpha) * h
            new_w = w + int(shift)
            offset = shift if alpha > 0 else 0

            M = np.float32([[1, -alpha, offset], [0, 1, 0]])
            img = cv.warpAffine(img, M, (new_w, h), flags=cv.INTER_LINEAR, borderValue=255)

        return img

    def extract_atomic_crops(self, img: np.ndarray):
        """ Ekstrakcja atomowych wycinków znaków do analizy przez CapsNet. """
        from skimage.filters import threshold_sauvola
        if img is None:
            return []

        # Wyprostowanie kursywy - wymuszamy typ ndarray, aby wykluczyć UMat
        img_clean = np.asarray(self.deslant_img(img))
        if img_clean is None or img_clean.size == 0:
            return []

        # Teraz linter ma 100% pewności, że .shape istnieje
        h_line, w_line = img_clean.shape

        # Przygotowanie do binaryzacji (Sauvola najlepiej czuje się na float32)
        img_float = img_clean.astype(np.float32)
        thresh = threshold_sauvola(img_float, window_size=31)
        binary_inv = (img_float <= thresh).astype(np.uint8) * 255

        # Pionowe domknięcie
        v_kernel_h = max(4, int(h_line * 0.25))
        v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, v_kernel_h))
        vertical_connected = cv.morphologyEx(binary_inv, cv.MORPH_CLOSE, v_kernel)

        # Bardzo lekkie poziome łatanie mikropęknięć
        h_kernel_w = max(3, int(h_line * 0.03))
        h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (h_kernel_w, 3))
        clean_map = cv.dilate(vertical_connected, h_kernel, iterations=1)

        # Wyciąganie konturów — teraz to będą litery lub grupy liter pisane ciągiem
        cnts, _ = cv.findContours(clean_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        min_w = max(2, int(h_line * 0.02))
        min_h = max(4, int(h_line * 0.1))

        boxes = []
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            if w > min_w and h > min_h:
                boxes.append([x, y, w, h])

        if not boxes: return []
        boxes.sort(key=lambda b: b[0])

        # Logika geometryczna (i tak CRNN utnie dłuższe słowa)
        merged_boxes = [boxes[0]]

        for i in range(1, len(boxes)):
            prev = merged_boxes[-1]
            curr = boxes[i]
            gap = curr[0] - (prev[0] + prev[2])

            # Sprawdzenie, czy potencjalne połączenie nie przekroczy okna
            potential_width = int(max(prev[0] + prev[2], curr[0] + curr[2])) - int(min(prev[0], curr[0]))

            # Klasyfikacja
            curr_is_apostrophe = (curr[1] < h_line * 0.35) and (curr[3] < h_line * 0.4)
            prev_is_word = prev[2] > h_line * 0.6  # Solidny rdzeń
            curr_is_word = curr[2] > h_line * 0.6

            merge = False

            # Nie łączymy, jak i tak CRNN potem utnie
            if potential_width > MAX_WORD_WIDTH:
                merge = False
            else:
                # Standardowe reguły przyciągania
                if gap <= 2:
                    merge = True
                elif curr_is_apostrophe and gap < h_line * 0.4:
                    merge = True
                elif prev_is_word and curr_is_word:
                    # Nawet jeśli są blisko, dwa duże bloki traktujemy jako osobne słowa
                    merge = (gap < h_line * 0.05)
                elif gap < h_line * 0.12:
                    merge = True

            if merge:
                # Rzutujemy na int(), aby linter wiedział, że operacje matematyczne są bezpieczne
                nx = int(min(prev[0], curr[0]))
                ny = int(min(prev[1], curr[1]))

                nw = int(max(prev[0] + prev[2], curr[0] + curr[2])) - nx
                nh = int(max(prev[1] + prev[3], curr[1] + curr[3])) - ny

                merged_boxes[-1] = [nx, ny, nw, nh]
            else:
                merged_boxes.append(curr)

        # Docinanie tła
        results = []
        dynamic_pad = max(4, int(h_line * 0.05))

        for (x, y, w, h) in merged_boxes:
            x1 = max(0, x - dynamic_pad)
            y1 = 0
            x2 = min(w_line, x + w + dynamic_pad)
            y2 = h_line

            # Łączenie fragmentów wyrazów, które zostały niepoprawnie podzielone (Używamy vc_arr w pętlach - linter widzi teraz czysty ndarray)
            vc_arr = np.asarray(vertical_connected)
            while y1 < y2 - 1 and int(np.sum(vc_arr[y1, x1:x2]).item()) == 0:
                y1 += 1

            while y2 > y1 + 1 and int(np.sum(vc_arr[y2 - 1, x1:x2]).item()) == 0:
                y2 -= 1

            y1 = max(0, y1 - dynamic_pad)
            y2 = min(h_line, y2 + dynamic_pad)

            crop = img_clean[y1:y2, x1:x2]
            results.append((crop, (x1, y1, x2 - x1, y2 - y1)))

        return results

    @staticmethod
    def _merge_group(group):
        if not group: return 0, 0, 0, 0
        min_x = min([b[0] for b in group])
        min_y = min([b[1] for b in group])
        max_x = max([b[0] + b[2] for b in group])
        max_y = max([b[1] + b[3] for b in group])
        return min_x, min_y, max_x - min_x, max_y - min_y

class PageToLineSegmentor:
    def __init__(self, min_line_height=15):
        self.min_line_height = min_line_height

    def extract_lines(self, img):
        """ Optymalizacja Seam Carving poprzez downscaling. """
        if img is None: return [], []

        h, w = img.shape
        if h < 50 or w < 50: return [img], []

        scale_factor = 512.0 / w if w > 512 else 1.0

        if scale_factor < 1.0:
            small_img = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            small_img = img

        # Progowanie
        binary_small = cv.adaptiveThreshold(small_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

        # Znajdowanie przerw
        proj = np.sum(binary_small, axis=1)
        gap_indices_small = self._find_line_gaps(proj, small_img.shape[0])
        if not gap_indices_small:
            return [img], []

        # Szukanie ścieżek
        seams_small = []
        for y_start in gap_indices_small:
            path = self._find_optimal_seam(binary_small, y_start)
            seams_small.append(path)

        seams_full = [np.zeros(w, dtype=int)]

        for s_small in seams_small:
            # Przygotowujemy dane wejściowe z jawnym typem
            s_input = s_small.astype(np.float32).reshape(1, -1)

            # Rzutujemy wynik resize na np.ndarray, co daje linterowi gwarancję dostępu do .flatten()
            resized = np.asarray(cv.resize(s_input, (w, 1), interpolation=cv.INTER_LINEAR))
            s_full = resized.flatten()

            # Skalowanie i rzutowanie na int (współrzędne pikseli)
            s_full = (s_full / scale_factor).astype(int)
            seams_full.append(np.clip(s_full, 0, h - 1))

        seams_full.append(np.full(w, h - 1, dtype=int))

        # Wycinanie linii
        lines = []
        for i in range(len(seams_full) - 1):
            top_seam = seams_full[i]
            bottom_seam = seams_full[i + 1]
            line_crop = self._extract_by_seams(img, top_seam, bottom_seam)

            if line_crop.shape[0] > self.min_line_height:
                # Ochrona przed pustymi liniami (fałszywe wykrycia na marginesach)
                if np.sum(line_crop == 0) > 30:
                    lines.append(line_crop)

        return lines, seams_full

    @staticmethod
    def _find_line_gaps(projection, height):
        """ Wykrywa środki pustych przestrzeni między wierszami. """
        # Przygotowujemy dane (rzutowanie i tak się przyda dla wydajności obliczeń)
        projection_float = np.asarray(projection, dtype=np.float32)
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smoothed = np.convolve(projection_float, kernel, mode='same')  # type: ignore

        threshold = float(np.mean(smoothed)) * 0.2
        gaps = np.where(smoothed < threshold)[0]

        if len(gaps) == 0:
            return []

        centers = []
        start = gaps[0]
        for i in range(1, len(gaps)):
            if gaps[i] - gaps[i - 1] > 10:
                centers.append(int((start + gaps[i - 1]) // 2))
                start = gaps[i]

        centers.append(int((start + gaps[-1]) // 2))
        return centers

    @staticmethod
    def _find_optimal_seam(binary, y_start):
        """ Znajduje optymalną ścieżkę przez obraz binarny przy użyciu programowania dynamicznego. """
        h, w = binary.shape

        # Dzięki temu szew zacznie omijać litery z większym wyprzedzeniem.
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
        cost_map_base = cv.dilate(binary.astype(np.float32), kernel)
        cost_map = np.where(cost_map_base > 0, 1000.0, 1.0).astype(np.float32)

        band = 60  # Zwiększamy zakres ruchu szwu w pionie
        min_y, max_y = max(0, y_start - band), min(h, y_start + band)
        band_h = max_y - min_y

        dp = np.full((band_h, w), np.inf, dtype=np.float32)
        parent = np.zeros((band_h, w), dtype=np.int8)

        local_start = np.clip(y_start - min_y, 0, band_h - 1)
        dp[:, 0] = np.abs(np.arange(band_h) - local_start) * 20.0
        dp[local_start, 0] = 0.0

        for x in range(1, w):
            prev = dp[:, x - 1]

            # Dodajemy karę (1.5) za ruch w pionie, by promować linie proste.
            cost_up = np.roll(prev, 1)
            cost_up[0] = np.inf
            cost_down = np.roll(prev, -1)
            cost_down[-1] = np.inf

            # Wybór najtańszego ruchu
            moves = np.vstack([prev, cost_up + 1.5, cost_down + 1.5])
            best_moves = np.argmin(moves, axis=0)

            # Aktualizacja kosztu
            dp[:, x] = np.min(moves, axis=0) + cost_map[min_y:max_y, x]
            parent[:, x] = best_moves

        # Rekonstrukcja ścieżki
        path = np.zeros(w, dtype=int)
        curr_y = np.argmin(dp[:, w - 1])
        path[w - 1] = curr_y + min_y
        for x in range(w - 1, 0, -1):
            move = parent[curr_y, x]
            if move == 1: # Ruch z góry
                curr_y -= 1
            elif move == 2: # Z dołu
                curr_y += 1
            # Zostajemy w wierszu
            path[x - 1] = curr_y + min_y

        return path

    @staticmethod
    def _extract_by_seams(img: np.ndarray, top_seam: np.ndarray, bottom_seam: np.ndarray) -> np.ndarray:
        """ Rektyfikacja: mapuje pofalowany obszar na idealny prostokąt. """
        img_work = np.asarray(img, dtype=np.uint8)
        h_orig, w_orig = img_work.shape

        # .item() wyciąga natywny float, co ucisza błąd 'float | complex'
        avg_dist = np.mean(bottom_seam - top_seam).item()
        target_h = max(1, int(float(avg_dist)))

        # Inicjalizacja płótna
        line_img: np.ndarray = np.full((target_h, w_orig), 255, dtype=np.uint8)

        # Przygotowujemy mapę X raz (zawsze 0 dla paska o szerokości 1px)
        map_x = np.zeros((target_h, 1), dtype=np.float32)

        for x in range(w_orig):
            # Generujemy punkty
            source_y = np.linspace(top_seam[x], bottom_seam[x], target_h, dtype=np.float32)
            map_y = np.reshape(source_y, (-1, 1))

            # Wycinamy pasek z jawnym rzutowaniem na ndarray
            column_slice: np.ndarray = np.asarray(img_work[:, x: x + 1], dtype=np.uint8)

            # Próbkowanie (dodajemy # type: ignore dla cv.remap przez błędy w stubs)
            remapped = cv.remap(column_slice, map_x, map_y, cv.INTER_LINEAR)  # type: ignore
            line_img[:, x] = remapped.flatten()

        return line_img