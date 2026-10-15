import subprocess
import sys
import os

def run_step(script_name):
    """ Uruchamia wskazany skrypt Pythona i monitoruje jego status. """
    print(f"[ORKIESTRATOR] Uruchamianie modułu: {script_name}")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"[BŁĄD KRYTYCZNY] Moduł '{script_name}' zakończył się błędem (kod: {result.returncode}). Przerywam cały proces.")
        sys.exit(result.returncode)
    else:
        print(f"[SUKCES] Moduł '{script_name}' zakończył działanie pomyślnie.")

if __name__ == "__main__":
    print("Start orkiestratora projektu Handwritten Text Recognition - uruchamianie wszystkich modułów krok po kroku.")
    
    # KROK 1: Trening CRNN (Rozpoznawanie słów i ekstrakcja wycinków dla CapsNetu)
    if os.path.exists("ResNetCRNNWordRecognition.py"):
        run_step("ResNetCRNNWordRecognition.py")
    else:
        print("Brak pliku ResNetCRNNWordRecognition.py w katalogu projektu!")
        sys.exit(1)

    # KROK 2: Trening CapsNet (Weryfikacja geometryczna trudnych i niepewnych znaków)
    if os.path.exists("DeepCapsNetCharRecognition.py"):
        run_step("DeepCapsNetCharRecognition.py")
    else:
        print("Brak pliku DeepCapsNetCharRecognition.py w katalogu projektu!")
        sys.exit(1)

    # KROK 3: Trening Transformera ByT5 (Korekta semantyczna i budowa kaskady końcowej)
    if os.path.exists("TransformerDictionaryRefinement.py"):
        run_step("TransformerDictionaryRefinement.py")
    else:
        print("Brak pliku TransformerDictionaryRefinement.py w katalogu projektu!")
        sys.exit(1)

    print("Cały proces orkiestracji HCR zakończony pomyślnie! Wszystkie modele (CRNN -> CapsNet -> Transformer) zostały poprawnie wytrenowane.")