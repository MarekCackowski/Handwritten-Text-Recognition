import os
import re
import time
from tqdm import tqdm

# Ścieżka do Twojego zbioru danych
DATA_ROOT = r"C:\OCR\iam_words\iam_words\words"
PATTERN = re.compile(r"^(#D|#C|#A|#S|#L|#U|#H)_\d+$")


def cleanup_specific_punctuation(root_dir):
    files_to_delete = []

    print(f"[{time.strftime('%H:%M:%S')}] Skanowanie katalogów w poszukiwaniu: #D, #C, #A, #S, #L, #Unie.")

    # Przeszukiwanie rekurencyjne (os.walk)
    for root, _, files in os.walk(root_dir):
        for f in files:
            name_no_ext = os.path.splitext(f)[0]

            # Weryfikacja czy nazwa pliku pasuje dokładnie do wzorca
            if PATTERN.match(name_no_ext):
                full_path = os.path.join(root, f)
                files_to_delete.append(full_path)

    if not files_to_delete:
        print("Nie znaleziono plików pasujących do tych konkretnych wzorców.")
        return

    print(f"\nZnaleziono {len(files_to_delete)} plików do usunięcia.")

    # Podgląd dla pewności
    for path in files_to_deleteTA:
        print(f" - Do usunięcia: {os.path.basename(path)}")

    confirm = input("\nCzy na pewno chcesz USUNĄĆ te pliki na stałe? (tak/nie): ").lower()

    if confirm == 'tak':
        for path in tqdm(files_to_delete, desc="Usuwanie"):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Błąd przy usuwaniu {path}: {e}")
        print(f"\nOperacja zakończona. Usunięto {len(files_to_delete)} plików.")
    else:
        print("Operacja anulowana.")


if __name__ == "__main__":
    cleanup_specific_punctuation(DATA_ROOT)
