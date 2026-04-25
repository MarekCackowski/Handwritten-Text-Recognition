import os
from tqdm import tqdm

# --- KONFIGURACJA ---
# Ścieżka do Twojego folderu z obrazami
DATA_ROOT = r"C:\OCR\archive\iam_words\words"

# Co zamieniamy (Znak niedozwolony -> Bezpieczny Kod)
# Używamy # bo jest dozwolony w nazwach plików i rzadki w tekście
CHAR_REPLACEMENT = {
    '.': '#D',  # Dot
    ',': '#C',  # Comma
    "'": '#A',  # Apostrophe
    '!': '#E',  # Exclamation
    '-': '#H',  # Hyphen
    '(': '#B',  # Bracket (Left)
    ')': '#K',  # Bracket (Right)
    ';': '#S',  # Semicolon
    ':': '#L',  # Colon
    '"': '#U'  # Quote
}


def sanitize_dataset():
    if not os.path.exists(DATA_ROOT):
        print(f"Błąd: Folder {DATA_ROOT} nie istnieje!")
        return

    print(f"Skanowanie: {DATA_ROOT}")
    renamed_count = 0

    # Rekurencyjne przejście przez wszystkie podfoldery
    for root, dirs, files in os.walk(DATA_ROOT):
        for filename in tqdm(files, desc="Sprawdzanie plików", leave=False):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Rozdziel nazwę i rozszerzenie (np. "Mr." i ".png")
            # Uwaga: os.path.splitext na pliku "Mr..png" może zgłupieć, dlatego robimy to ostrożnie
            if '.' in filename:
                # Znajdź ostatnią kropkę (rozszerzenie)
                last_dot_idx = filename.rfind('.')
                name_part = filename[:last_dot_idx]
                ext_part = filename[last_dot_idx:]
            else:
                name_part = filename
                ext_part = ""

            new_name_part = name_part
            needs_rename = False

            # Sprawdź, czy nazwa zawiera zakazane znaki
            for char, code in CHAR_REPLACEMENT.items():
                if char in new_name_part:
                    new_name_part = new_name_part.replace(char, code)
                    needs_rename = True

            # Wykonaj zmianę nazwy
            if needs_rename:
                old_path = os.path.join(root, filename)
                new_filename = new_name_part + ext_part
                new_path = os.path.join(root, new_filename)

                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                    # print(f"Zmieniono: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Błąd przy zmianie {filename}: {e}")

    print(f"\nZakończono! Zmieniono nazw {renamed_count} plików.")
    print("Teraz Twoje pliki są bezpieczne (np. 'Mr#D.png' zamiast 'Mr..png').")


if __name__ == "__main__":
    sanitize_dataset()
