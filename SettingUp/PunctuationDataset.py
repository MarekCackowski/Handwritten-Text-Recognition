import tarfile
import os
import io
from tqdm import tqdm

# --- 1. KONFIGURACJA ŚCIEŻEK ---
folder_pobrane = os.path.join(os.path.expanduser("~"), "Downloads")
target_root = r"C:\OCR\curated_clean"

# Definicja szukanej interpunkcji (ASCII Decimal)
# 33-47: ! " # $ % & ' ( ) * + , - . /
# 95: _ (underscore), 96: ` (backtick)
SELECTED_ASCII = set(list(range(33, 48)) + [95, 96])


# --- 2. KLASA STRUMIENIA (Łączy .01 i .02 w locie) ---
class MultiFileStream(io.RawIOBase):
    def __init__(self, files):
        self.files = files
        self.current_idx = 0
        self.current_file = open(self.files[0], 'rb')

    def read(self, size=-1):
        chunk = self.current_file.read(size)
        if not chunk and self.current_idx < len(self.files) - 1:
            self.current_file.close()
            self.current_idx += 1
            print(f"\n[Strumień] Przełączam na: {os.path.basename(self.files[self.current_idx])}")
            self.current_file = open(self.files[self.current_idx], 'rb')
            return self.read(size)
        return chunk

    def readable(self):
        return True

    def close(self):
        if hasattr(self, 'current_file'): self.current_file.close()
        super().close()


# --- 3. PRZYGOTOWANIE ---
if not os.path.exists(target_root):
    os.makedirs(target_root)

czesci_archiwum = [
    os.path.join(folder_pobrane, "curated.tar.gz.01"),
    os.path.join(folder_pobrane, "curated.tar.gz.02")
]

for p in czesci_archiwum:
    if not os.path.exists(p):
        print(f"BŁĄD: Nie znaleziono pliku {p}")
        exit()

# --- 4. PROCES ROZPAKOWYWANIA ---
stream = MultiFileStream(czesci_archiwum)

try:
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tqdm(tar, desc="Wypakowywanie interpunkcji (33-47, 95, 96)"):
            # Tylko pliki PNG, ignoruj metadane macOS
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue
            if os.path.basename(member.name).startswith("._"):
                continue

            path_parts = member.name.split('/')

            try:
                found_ascii = None
                for part in path_parts:
                    if part.isalnum():
                        # Próbujemy odczytać kod (dziesiętnie lub szesnastkowo)
                        try:
                            val = int(part)  # Próba dec
                        except ValueError:
                            try:
                                val = int(part, 16)  # Próba hex
                            except ValueError:
                                continue

                        # FILTR: Sprawdź czy to wybrana interpunkcja
                        if val in SELECTED_ASCII:
                            found_ascii = str(val)
                            break

                if found_ascii:
                    dest_path = os.path.join(target_root, found_ascii, os.path.basename(member.name))
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    extracted_file = tar.extractfile(member)
                    if extracted_file:
                        with open(dest_path, "wb") as f_out:
                            f_out.write(extracted_file.read())
            except Exception:
                continue
finally:
    stream.close()

print(f"\nSukces! Wyodrębniono wybraną interpunkcję do: {target_root}")
