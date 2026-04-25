import os
import re
from tqdm import tqdm
from typing import Dict, Any

# ----------------------------
# 1. KONFIGURACJA ŚCIEŻEK
# ----------------------------
folder_path = r"C:\OCR\archive\iam_words\words"
labels_path = r"C:\OCR\archive\words_labels.txt"

# ----------------------------
# 2. LOGIKA SANITACJI (KROK 1)
# ----------------------------

# Co na co zamieniamy (Znak niedozwolony -> Bezpieczny Kod)
CHAR_REPLACEMENT: Dict[str, str] = {
    '.': '#D',  # Dot
    ',': '#C',  # Comma
    "'": '#A',  # Apostrophe
    '!': '#E',  # Exclamation
    '-': '#H',  # Hyphen (tylko w tekście, nie w ID)
    '(': '#B',  # Bracket (Left)
    ')': '#K',  # Bracket (Right)
    ';': '#S',  # Semicolon
    ':': '#L',  # Colon
    '"': '#U'  # Quote
}


def _sanitize_existing_filenames() -> int:
    """
    Krok 1: Zamienia problematyczne znaki w ISTNIEJĄCYCH nazwach plików (IAM IDs)
    na bezpieczne kody (#D, #C, itp.), aby uniknąć błędów systemu plików.
    """
    if not os.path.exists(folder_path):
        print(f"Błąd: Folder {folder_path} nie istnieje!")
        return 0

    print(f"--- Faza 1/2: Sanitacja nazw plików w {folder_path} ---")
    renamed_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in tqdm(files, desc="Sanitacja", leave=False):

            last_dot_idx = filename.rfind('.')
            if last_dot_idx == -1: continue  # Brak rozszerzenia

            name_part = filename[:last_dot_idx]
            ext_part = filename[last_dot_idx:]

            new_name_part = name_part
            needs_rename = False

            for char, code in CHAR_REPLACEMENT.items():
                if char in new_name_part:
                    new_name_part = new_name_part.replace(char, code)
                    needs_rename = True

            if needs_rename:
                old_path = os.path.join(root, filename)
                new_filename = new_name_part + ext_part
                new_path = os.path.join(root, new_filename)

                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Błąd przy zmianie {filename}: {e}")

    print(f"Sanitacja zakończona. Zmieniono nazw {renamed_count} plików.")
    return renamed_count


# ----------------------------
# 3. LOGIKA MAPOWANIA ETYKIET (KROK 2)
# ----------------------------

def clean_filename(text: str) -> str:
    """
    Usuwa znaki, które są niedozwolone w nazwach plików OS, ale zachowuje
    te, które zostały wcześniej zakodowane (np. #D).
    """
    # Znaki zakazane przez Windows/Linux
    invalid_chars = '<>:"/\|?*'

    # 1. Usuń znaki systemowe
    for char in invalid_chars:
        text = text.replace(char, '')

    # 2. Usuń kropkę kończącą (która nie jest częścią zakodowanej interpunkcji)
    return text.strip().rstrip('.')


def index_files(root_folder: str) -> Dict[str, str]:
    """
    Tworzy mapę: ID_pliku (bez rozszerzenia) -> Pełna ścieżka.
    ID pliku to teraz ID IAM z zakodowaną interpunkcją (np. a01-000u-00#D).
    """
    file_map = {}
    count = 0
    for root, _, files in os.walk(root_folder):
        for filename in files:
            name_no_ext = os.path.splitext(filename)[0]
            full_path = os.path.join(root, filename)
            file_map[name_no_ext] = full_path
            count += 1
    return file_map


def _cleanup_unlabeled_files(files_to_delete: Dict[str, str]):
    """
    Usuwa pliki, które pozostały w folderze po procesie mapowania (czyli są nienazwane).
    """
    print("\n--- Usuwanie nienazwanych plików ---")
    deleted_count = 0

    for _, full_path in files_to_delete.items():
        try:
            os.remove(full_path)
            deleted_count += 1
        except OSError as e:
            print(f"Nie można usunąć {full_path}: {e}")

    print(f"Usunięto: {deleted_count} nienazwanych plików.")


def process_labels_and_rename():
    """
    Krok 2: Odczytuje etykiety z pliku i nadaje plikom finalne, czytelne nazwy.
    """
    if not os.path.exists(folder_path) or not os.path.exists(labels_path):
        print("Błąd: Sprawdź ścieżki folderu lub etykiet.")
        return

    # Słownik śledzący unikalne nazwy słów i ich liczniki (np. "the": 1, "the": 2)
    word_counts: Dict[str, int] = {}

    # Indeksowanie plików na dysku (po sanitacji)
    files_location_map = index_files(folder_path)
    files_to_delete_after = files_location_map.copy()  # Kopią śledzimy, co usunąć

    print(f"\n--- Faza 2/2: Mapowanie {labels_path} na pliki ---")

    success_count = 0
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Mapowanie etykiet"):
        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split()

        # Oczekiwany format IAM: ID jest na [0], Label jest na końcu [-1]
        if len(parts) > 8:
            txt_id = parts[0]
            label_text = parts[-1]
        else:
            continue

        # Etykieta w pliku .txt może zawierać interpunkcję, którą OS by zakazał.
        # W IAM ID nie zawiera interpunkcji, ale musi pasować do klucza w files_location_map.

        # 1. Tworzymy ID pliku, które pasuje do klucza w files_location_map
        # (W IAM ID jest formatu a01-000u-00, nie zawiera interpunkcji)
        # Należy upewnić się, że txt_id jest kluczem bez rozszerzenia.

        if txt_id in files_location_map:
            current_full_path = files_location_map[txt_id]

            # 2. Czyścimy docelową nazwę (usuwamy znaki systemowe, ale zachowujemy tekst)
            safe_word = clean_filename(label_text)

            if not safe_word:
                # Jeśli etykieta jest pusta (np. zawiera tylko znak interpunkcyjny),
                # zostawiamy plik do usunięcia (nie ma sensu go nazywać).
                if txt_id in files_to_delete_after: del files_to_delete_after[txt_id]
                continue

            # 3. SMART INCREMENT LOOP (Unikalne nazwy)
            word_counts[safe_word] = word_counts.get(safe_word, 0) + 1
            count = word_counts[safe_word]

            dir_name = os.path.dirname(current_full_path)
            _, extension = os.path.splitext(current_full_path)

            new_filename = f"{safe_word}_{count}{extension}"
            new_full_path = os.path.join(dir_name, new_filename)

            # --- ZMIANA NAZWY ---
            try:
                os.rename(current_full_path, new_full_path)

                # Usuwamy plik z listy do usunięcia, bo został już nazwany
                if txt_id in files_to_delete_after:
                    del files_to_delete_after[txt_id]

                success_count += 1

            except OSError as e:
                print(f"Błąd przy zmianie nazwy {txt_id} na {new_filename}: {e}")
        else:
            # Jeśli ID nie znalezione (plik już nazwany/usunięty w innej fazie)
            pass

    print("-" * 30)
    print(f"Mapowanie DONE! Pomyślnie zmieniono nazw: {success_count}")

    # 4. KROK KOŃCOWY: SPRZĄTANIE
    _cleanup_unlabeled_files(files_to_delete_after)


def run_full_preprocessor():
    """Główna sekwencja wykonawcza."""
    if not os.path.exists(folder_path):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono folderu danych: {folder_path}")
        return

    # Faza 1: Sanitacja (Zabezpieczenie przed błędami w nazewnictwie)
    _sanitize_existing_filenames()

    # Faza 2: Mapowanie etykiet i nadawanie finalnych nazw
    process_labels_and_rename()


if __name__ == "__main__":
    run_full_preprocessor()
