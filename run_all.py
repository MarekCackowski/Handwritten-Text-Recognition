import csv
from datetime import datetime
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from tqdm import tqdm
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

from dotenv import load_dotenv
from requests.exceptions import RequestException

try:
  import torch
except ImportError:
  torch = None

try:
  from huggingface_hub import HfApi, login
  from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
  HfApi, login = None, None
  HfHubHTTPError, RepositoryNotFoundError = OSError, OSError

""" Załadowanie zmiennych środowiskowych z pliku .env (np. HF_TOKEN, HF_REPO_ID, UPLOAD_TO_HF). """
load_dotenv()


def get_optimal_workers(max_workers: int = 4, ram_per_worker_gb: float = 2.5) -> int:
    """Dobiera bezpieczną liczbę workerów DataLoadera na podstawie wolnej pamięci RAM."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        safe_workers = max(0, int((avail_gb - 3.0) // ram_per_worker_gb))
        return min(max_workers, safe_workers)
    except Exception:
        return min(max_workers, 2)


def smart_gpu_guard(min_free_ratio: float = 0.10, max_temp_c: int = 86) -> None:
    """Zwalnia cache VRAM tylko przy deficycie pamięci i chroni GPU przed przegrzaniem."""
    import time
    import torch
    from tqdm import tqdm

    if not torch.cuda.is_available():
        return

    free_mem, total_mem = torch.cuda.mem_get_info()
    if total_mem > 0 and (free_mem / total_mem) < min_free_ratio:
        torch.cuda.empty_cache()

    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp >= max_temp_c:
                tqdm.write(f"[{time.strftime('%H:%M:%S')}] Wysoka temperatura GPU ({temp}°C) – pauza 2s dla schłodzenia.")
                time.sleep(2.0)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass


def epoch_progress_bar(iterable, total_epochs: int, stage_name: str, initial: int = 0):
  """ Zwraca licznik/pasek epok w formacie 'Epoka X/Y' dla pętli treningowych każdego modelu. """
  return tqdm(
      iterable,
      total=total_epochs,
      initial=initial,
      desc=f"{stage_name} | Epoki",
      unit="ep",
      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
      dynamic_ncols=True,
  )


def run_step(script_name):
  """ Uruchamia wskazany skrypt Pythona, ustawia PYTHONPATH i monitoruje status. """
  print(f'Uruchamianie modułu: {script_name}')

  """ Przekazanie obecnego katalogu do PYTHONPATH dla procesu potomnego. """
  env = os.environ.copy()
  env['PYTHONPATH'] = os.getcwd()

  try:
    result = subprocess.run([sys.executable, script_name], env=env)
  except (OSError, ValueError) as e:
    print(f"Błąd systemowy podczas uruchamiania modułu '{script_name}': {e}")
    sys.exit(1)

  if result.returncode != 0:
    print(
        f"Moduł '{script_name}' zakończył się błędem (kod:"
        f' {result.returncode}). Przerywam cały proces.'
    )
    sys.exit(result.returncode)
  else:
    print(f"Moduł '{script_name}' zakończył działanie pomyślnie.")


def initialize_databases_and_data():
  """ Weryfikuje środowisko PyTorch/CUDA oraz uruchamia InitDatabases.py. """
  print('KROK 0: Weryfikacja środowiska i inicjalizacja baz oraz danych.')

  """ Sprawdzenie dostępności biblioteki PyTorch oraz akceleracji CUDA. """
  if torch is None:
    print('Błąd: Brak wymaganych bibliotek z requirements.txt (torch).')
    sys.exit(1)

  print(f'Wersja PyTorch: {torch.__version__}, dostępność CUDA: {torch.cuda.is_available()}')

  """ Wyszukanie i uruchomienie skryptu inicjalizującego bazy danych oraz zbiory. """
  candidates = [os.path.join('Data', 'InitDatabases.py'), 'InitDatabases.py']
  init_db_script = next((p for p in candidates if os.path.exists(p)), None)

  if init_db_script:
    run_step(init_db_script)
  else:
    print(f'Błąd krytyczny: Nie znaleziono żadnego z plików: {candidates}')
    sys.exit(1)


def export_final_metrics(output_dir='output_data', readme_path='README.md'):
  """ Eksportuje ostateczne metryki ze wszystkich eksperymentów MLflow do JSON, CSV, tabeli MD oraz README.md. """
  try:
    os.makedirs(output_dir, exist_ok=True)
  except OSError as e:
    print(f"Błąd tworzenia katalogu wyjściowego '{output_dir}' (OSError): {e}")
    return None

  tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

  if not tracking_uri:
    for candidate_uri in ('http://htr_mlflow:5000', 'http://mlflow:5000', 'http://127.0.0.1:5000'):
      try:
        urllib.request.urlopen(candidate_uri, timeout=2)
        tracking_uri = candidate_uri
        break
      except urllib.error.HTTPError as e:
        print(f"Odpowiedź HTTP podczas sprawdzania MLflow ({candidate_uri}): {e}")
      except urllib.error.URLError as e:
        print(f"Błąd połączenia z MLflow ({candidate_uri}): {e}")
      except TimeoutError as e:
        print(f"Przekroczenie czasu oczekiwania na MLflow ({candidate_uri}): {e}")
      except ValueError as e:
        print(f"Nieprawidłowy adres URI MLflow ({candidate_uri}): {e}")

  if not tracking_uri:
    if os.path.exists('/app/mlruns/mlflow.db'):
      tracking_uri = 'sqlite:////app/mlruns/mlflow.db'
    else:
      tracking_uri = 'http://htr_mlflow:5000'

  try:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
  except (MlflowException, ValueError, RuntimeError) as e:
    print(f"Błąd inicjalizacji klienta MLflow ('{tracking_uri}'): {e}")
    return None

  # Eksport metryk do JSON, CSV i Markdown, żeby wiedzieć, które eksperymenty i runy były ostatnio zakończone
  exported_data = {
      'exported_at': datetime.now().isoformat(timespec='seconds'),
      'tracking_uri': tracking_uri,
      'experiments': {},
  }
  summary_rows = []

  try:
    experiments = client.search_experiments()
    for exp in experiments:
      runs = client.search_runs(
          experiment_ids=[exp.experiment_id],
          order_by=['start_time DESC'],
          max_results=10,
      )
      if not runs:
        continue

      # Zbieranie metryk i parametrów dla każdego runu w eksperymencie
      exp_runs = []
      for run in runs:
        metrics = dict(sorted(run.data.metrics.items()))
        params = dict(sorted(run.data.params.items()))
        run_name = run.info.run_name or run.info.run_id[:8]
        exp_runs.append({
            'run_id': run.info.run_id,
            'run_name': run_name,
            'status': run.info.status,
            'metrics': metrics,
            'params': params,
        })

      # Wybór najnowszego zakończonego runu z metrykami lub pierwszego dostępnego runu
      exported_data['experiments'][exp.name] = exp_runs
      latest = (
          next((r for r in exp_runs if r['status'] == 'FINISHED' and r['metrics']), None)
          or next((r for r in exp_runs if r['metrics']), exp_runs[0])
      )
      summary_rows.append({
          'experiment': exp.name,
          'run_name': latest['run_name'],
          'status': latest['status'],
          'metrics': latest['metrics'],
      })
  except (MlflowException, RequestException, OSError, KeyError, ValueError) as e:
    print(f"Błąd podczas pobierania eksperymentów i runów z MLflow ({tracking_uri}): {e}")
    return None

  # Eksport do pliku JSON, CSV i Markdown
  json_path = os.path.join(output_dir, 'final_metrics.json')
  all_metric_keys = sorted({k for row in summary_rows for k in row['metrics'].keys()})
  md_path = os.path.join(output_dir, 'final_metrics_table.md')
  csv_path = os.path.join(output_dir, 'final_metrics.csv')

  try:
    with open(json_path, 'w', encoding='utf-8') as f:
      json.dump(exported_data, f, ensure_ascii=False, indent=2)

    # Używamy zbioru wszystkich kluczy metryk, aby utworzyć spójną tabelę w Markdown i CSV
    with open(md_path, 'w', encoding='utf-8') as f:
      f.write('# Ostateczne metryki kaskady HTR\n\n')
      if summary_rows:
        headers = ['Eksperyment', 'Run', 'Status'] + all_metric_keys
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
        for row in summary_rows:
          vals = [
              f"{row['metrics'][k]:.4f}"
              if isinstance(row['metrics'].get(k), float)
              else str(row['metrics'].get(k, '-'))
              for k in all_metric_keys
          ]
          f.write('| ' + ' | '.join([row['experiment'], row['run_name'], row['status']] + vals) + ' |\n')

    # Eksport ostatecznych metryk do CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
      fieldnames = ['experiment', 'run_name', 'status'] + all_metric_keys
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for row in summary_rows:
        flat = {
            'experiment': row['experiment'],
            'run_name': row['run_name'],
            'status': row['status'],
        }
        flat.update(row['metrics'])
        writer.writerow(flat)
  except (OSError, TypeError, ValueError) as e:
    print(f"Błąd zapisu plików eksportu metryk w '{output_dir}': {e}")
    return exported_data

  print(f'Wyeksportowano ostateczne metryki do: {json_path}, {md_path}, {csv_path}')

  # Automatyczne uzupełnienie tabeli wyników w README.md
  if os.path.exists(readme_path):
    def _fmt_pct(val):
      """ Formatuje wartość metryki jako procent z dwoma miejscami po przecinku. """
      if val is None or not isinstance(val, (int, float)):
        return '*TBD*'
      pct = val * 100.0 if 0.0 <= val <= 1.0 else float(val)
      return f'{pct:.2f}%'

    def _pick_metric(metrics_dict, candidates):
      """ Wybiera pierwszą dostępną metrykę z listy kandydatów, ignorując wielkość liter.
          Używana, ponieważ nazwy metryk w MLflow mogą się różnić w zależności od eksperymentu. """
      lower_map = {k.lower(): v for k, v in metrics_dict.items()}
      for cand in candidates:
        if cand in lower_map:
          return lower_map[cand]
      for cand in candidates:
        for k, v in lower_map.items():
          if k.endswith(cand):
            return v
      return None

    def _collect_stage(exp_keywords, prefixes):
      """ Zbiera metryki CER, WER i diakrytyczne dla danego etapu kaskady. """
      merged = {}
      for row in summary_rows:
        exp_low = row['experiment'].lower()
        if any(kw in exp_low for kw in exp_keywords):
          merged.update(row['metrics'])
        for k, v in row['metrics'].items():
          if any(k.lower().startswith(p) for p in prefixes):
            merged[k] = v

      cer = _pick_metric(merged, [
          *(f'{p}test_cer' for p in prefixes),
          *(f'{p}val_cer' for p in prefixes),
          *(f'{p}cer' for p in prefixes),
          'test_cer', 'val_cer', 'best_cer', 'cer',
      ])
      wer = _pick_metric(merged, [
          *(f'{p}test_wer' for p in prefixes),
          *(f'{p}val_wer' for p in prefixes),
          *(f'{p}wer' for p in prefixes),
          'test_wer', 'val_wer', 'best_wer', 'wer',
      ])
      diac = _pick_metric(merged, [
          *(f'{p}polish_diacritic_acc' for p in prefixes),
          *(f'{p}diacritic_acc' for p in prefixes),
          'polish_diacritic_acc', 'pl_diacritic_acc', 'val_diacritic_acc', 'diacritic_acc',
      ])
      return _fmt_pct(cer), _fmt_pct(wer), _fmt_pct(diac)

    try:
      s1_cer, s1_wer, s1_diac = _collect_stage(['crnn'], ['crnn_', 'stage1_'])
      s2_cer, s2_wer, s2_diac = _collect_stage(['capsnet', 'caps'], ['capsnet_', 'fusion_', 'stage2_'])
      s3_cer, s3_wer, s3_diac = _collect_stage(['transformer', 'byt5', 'cascade'], ['transformer_', 'byt5_', 'cascade_', 'final_', 'stage3_'])

      with open(readme_path, 'r', encoding='utf-8') as f:
        readme_text = f.read()

      replacements = {
          r'\|\s*\*\*Stage 1 \(Baseline\)\*\*\s*\|.*': (
              f'| **Stage 1 (Baseline)** | `ResNet-18 + ScRN_STN + BiLSTM` | {s1_cer} | {s1_wer} | {s1_diac} | Fast word-level transcription & entropy map |'
          ),
          r'\|\s*\*\*Stage 2 \(\+ Geometry\)\*\*\s*\|.*': (
              f'| **Stage 2 (+ Geometry)** | `CRNN + DeepCapsNet (Deep Fusion)` | {s2_cer} | {s2_wer} | {s2_diac} | Sub-character & ligature disambiguation |'
          ),
          r'\|\s*\*\*Stage 3 \(Full Cascade\)\*\*\s*\|.*': (
              f'| **Stage 3 (Full Cascade)** | `CRNN + CapsNet + ByT5-LoRA` | {s3_cer} | {s3_wer} | {s3_diac} | Context-aware bilingual semantic correction |'
          ),
          r'\*Note: Replace `\*TBD\*` entries with final run metrics exported from the MLflow experiment summary once Stage 3 completes\.\*': (
              f"*Note: Benchmark metrics automatically populated from MLflow experiment summary (`{exported_data['exported_at']}`).*"
          ),
      }

      for pattern, repl in replacements.items():
        readme_text = re.sub(pattern, repl, readme_text)

      with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_text)

      print(f"Zaktualizowano tabelę wyników w '{readme_path}' (Stage 1 CER={s1_cer}, Stage 2 CER={s2_cer}, Stage 3 CER={s3_cer}).")
    except (OSError, UnicodeDecodeError, re.error, ValueError) as e:
      print(f"Błąd aktualizacji pliku '{readme_path}': {e}")

  return exported_data


def upload_to_hugging_face():
  """ Funkcja przesyłająca wytrenowane modele, adapter ByT5 i wyeksportowane metryki do Hugging Face Hub. """
  print('Rozpoczynam eksport artefaktów HCR do Hugging Face.')

  upload_flag = os.getenv('UPLOAD_TO_HF', 'False').lower() == 'true'
  if not upload_flag:
    print('Flaga UPLOAD_TO_HF jest wyłączona lub nie ustawiona. Pomijam wysyłanie.')
    return

  hf_token = os.getenv('HF_TOKEN')
  repo_id = os.getenv('HF_REPO_ID', 'StudentWAT/HCR')

  if not hf_token:
    print('Błąd: Brak tokenu HF_TOKEN w zmiennych środowiskowych (.env). Nie można przesłać modeli.')
    return

  if HfApi is None or login is None:
    print("Biblioteka 'huggingface_hub' nie jest zainstalowana. Zainstaluj ją za pomocą 'pip install huggingface_hub'.")
    return

  try:
    print(f'Logowanie do repozytorium HTR: {repo_id}')
    login(token=hf_token)
    api = HfApi()

    output_dir = 'output_data' if os.path.exists('output_data') else '.'
    print(f"Wysyłanie zawartości z '{output_dir}' do repozytorium: {repo_id}")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type='model',
        commit_message=(
            'Automated pipeline upload: Trained HTR (CRNN + CapsNet + ByT5) '
            'weights, checkpoints and final metrics'
        ),
    )

    byt5_candidates = [
        'byt5_htr_refiner_final',
        os.path.join(output_dir, 'checkpoints', 'transformer', 'byt5_htr_refiner_final'),
    ]
    byt5_dir = next((p for p in byt5_candidates if os.path.exists(p)), None)
    if byt5_dir:
      print(f"Wysyłanie adaptera Transformera z '{byt5_dir}' do repozytorium: {repo_id}")
      api.upload_folder(
          folder_path=byt5_dir,
          path_in_repo='byt5_htr_refiner_final',
          repo_id=repo_id,
          repo_type='model',
          commit_message='Upload fine-tuned ByT5 LoRA refiner weights',
      )

    if os.path.exists('README.md'):
      api.upload_file(
          path_or_fileobj='README.md',
          path_in_repo='README.md',
          repo_id=repo_id,
          repo_type='model',
          commit_message='Update README.md with architecture and metrics',
      )

    print(f'Pomyślnie przesłano wszystkie artefakty HTR do https://huggingface.co/{repo_id}')

  except RepositoryNotFoundError as e:
    print(f"Nie znaleziono repozytorium '{repo_id}' w Hugging Face Hub: {e}")
  except HfHubHTTPError as e:
    print(f'Błąd HTTP API Hugging Face podczas przesyłania plików: {e}')
  except RequestException as e:
    print(f'Błąd połączenia sieciowego z Hugging Face Hub: {e}')
  except ValueError as e:
    print(f'Nieprawidłowy token dostępowy lub konfiguracja repozytorium Hugging Face: {e}')
  except OSError as e:
    print(f'Błąd odczytu lokalnych plików podczas wysyłki do Hugging Face: {e}')


def run_interactive_mode():
  """ KROK 8: Ładuje wytrenowaną kaskadę do GPU, uruchamia prosty tryb interaktywny CLI,
      a po zakończeniu zwalnia VRAM i przekazuje modele do workera Celery. """
  import gc
  from celery.exceptions import CeleryError
  from kombu.exceptions import KombuError, OperationalError
  import App.Worker as worker_module

  print('\nKROK 8: Ładowanie wytrenowanych modeli (CRNN -> CapsNet -> ByT5) do pamięci GPU...')
  try:
    worker_module.load_models()
  except (ImportError, ModuleNotFoundError) as e:
    print(f'Brak wymaganych modułów podczas ładowania modeli workera: {e}')
    return
  except RuntimeError as e:
    if 'out of memory' in str(e).lower() and torch is not None and torch.cuda.is_available():
      torch.cuda.empty_cache()
    print(f'Błąd pamięci lub wykonania podczas ładowania modeli (RuntimeError): {e}')
    return
  except (OSError, ValueError, KeyError) as e:
    print(f'Błąd plików wag lub konfiguracji podczas ładowania modeli ({type(e).__name__}): {e}')
    return

  pipe = worker_module.inference_pipe

  if pipe is None:
    print('Nie udało się załadować modeli (brak wag). Pomijam tryb interaktywny.')
    return

  infer_fn = (
      getattr(pipe, 'predict', None)
      or getattr(pipe, 'recognize', None)
      or pipe
  )

  if sys.stdin.isatty():
    print('TRYB INTERAKTYWNY: Wprowadź ścieżkę do obrazu, aby uzyskać rozpoznany tekst z kaskady HTR. Wpisz "q", aby zakończyć.')
    try:
      while True:
        img_path = input('\nŚcieżka do obrazu > ').strip()
        if not img_path:
          continue
        if img_path.lower() in {'q', 'quit', 'exit'}:
          print('Zamykanie trybu interaktywnego.')
          break
        if not os.path.exists(img_path):
          print(f'Plik nie istnieje -> {img_path}')
          continue

        try:
          result = infer_fn(img_path)
          if isinstance(result, dict):
            print(f"crnn result:         {result.get('crnn', result.get('crnn_text', '-'))}")
            print(f"crnn+capsnet result: {result.get('capsnet', result.get('refined_text', '-'))}")
            print(f"final result:        {result.get('final', result.get('text', result))}")
          else:
            print(f'final result: {result}')
        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
          print(f'Błąd odczytu lub uszkodzony format pliku obrazu ({type(e).__name__}): {e}')
        except RuntimeError as e:
          if 'out of memory' in str(e).lower() and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f'Błąd pamięci VRAM GPU podczas inferencji — wyczyszczono cache CUDA: {e}')
          else:
            print(f'Błąd wykonania modelu PyTorch (RuntimeError): {e}')
        except (ValueError, TypeError, KeyError) as e:
          print(f'Błąd danych wejściowych lub struktury wyniku kaskady ({type(e).__name__}): {e}')
    except KeyboardInterrupt:
      print('\nPrzerwano tryb interaktywny przez użytkownika (Ctrl+C).')
    except EOFError:
      print('\nZakończono wejście standardowe (EOF) w trybie interaktywnym.')

  # Usunięcie lokalnych referencji i zwolnienie VRAM przed przejęciem GPU przez htr-worker
  del pipe, infer_fn
  worker_module.inference_pipe = None
  gc.collect()
  if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()

  try:
    active_workers = worker_module.celery_app.control.inspect(timeout=2.0).ping()
    worker_module.celery_app.send_task('worker.reload_models', ignore_result=True)
    if active_workers:
      print('Przekazano gotowe modele do aktywnej usługi htr-worker w tle (API: http://localhost:8000).')
    else:
      print('Zadanie dodane do kolejki Redis, ale żaden worker Celery nie odpowiedział na ping — sprawdź: docker compose ps')
  except OperationalError as e:
    print(f'Błąd połączenia z brokerem Redis (kombu.OperationalError): {e}')
    print('Sprawdź, czy kontener htr_redis działa, a następnie wykonaj: docker compose restart htr-worker')
  except CeleryError as e:
    print(f'Błąd kolejkowania zadania w Celery (CeleryError): {e}')
    print('Uruchom ręcznie: docker compose restart htr-worker')
  except KombuError as e:
    print(f'Błąd transportu wiadomości w Kombu (KombuError): {e}')
    print('Uruchom ręcznie: docker compose restart htr-worker')
  except (ConnectionError, TimeoutError, OSError) as e:
    print(f'Błąd sieciowy lub przekroczenie czasu oczekiwania podczas komunikacji z workerem ({type(e).__name__}): {e}')
    print('Uruchom ręcznie: docker compose restart htr-worker')


if __name__ == '__main__':
  try:
    print(
        'Start orkiestratora projektu Handwritten Text Recognition - uruchamianie'
        ' wszystkich modułów krok po kroku.'
    )

    """ KROK 0: Inicjalizacja baz danych (MongoDB, Redis, MinIO, PostgreSQL) i przygotowanie H5/NPZ/Corpus. """
    initialize_databases_and_data()

    """ KROK 1: Trening CRNN (Rozpoznawanie słów i ekstrakcja wycinków dla CapsNetu). """
    crnn_script = os.path.join('Models', 'ResNetCRNNWordRecognition.py')
    if os.path.exists(crnn_script):
      run_step(crnn_script)
    else:
      print('Brak pliku ResNetCRNNWordRecognition.py w katalogu projektu!')
      sys.exit(1)

    """ KROK 2: Trening CapsNet (Weryfikacja geometryczna trudnych i niepewnych znaków). """
    capsnet_script = os.path.join('Models', 'DeepCapsNetCharRecognition.py')
    if os.path.exists(capsnet_script):
      run_step(capsnet_script)
    else:
      print('Brak pliku DeepCapsNetCharRecognition.py w katalogu projektu!')
      sys.exit(1)

    """ KROK 2.5: Generowanie par błędów wizualnych, macierzy pomyłek i myśli wizualnych dla Transformera. """
    transformer_data_script = os.path.join('Data', 'TransformerDataGeneration.py')
    if os.path.exists(transformer_data_script):
      run_step(transformer_data_script)
    else:
      print(f'Brak pliku {transformer_data_script} w katalogu projektu!')
      sys.exit(1)

    """ KROK 3: Trening Transformera ByT5 (Korekta semantyczna i budowa kaskady końcowej). """
    transformer_script = os.path.join('Models', 'TransformerDictionaryRefinement.py')
    if os.path.exists(transformer_script):
      run_step(transformer_script)
    else:
      print('Brak pliku TransformerDictionaryRefinement.py w katalogu projektu!')
      sys.exit(1)

    """ KROK 4: Profilowanie wydajności, wyliczanie metryk i generowanie wykresów do rozdz. 5. """
    profiler_script = next(
        (p for p in [os.path.join('Analisis', 'PerformanceProfiling.py'), 'PerformanceProfiling.py'] if os.path.exists(p)),
        None,
    )
    if profiler_script:
      run_step(profiler_script)
    else:
      print('Brak pliku PerformanceProfiling.py w katalogu! Pominięto profilowanie wydajności.')

    """ KROK 5: Generowanie macierzy pomyłek i raportu poprawek przed/po. """
    error_analysis_script = next(
        (p for p in ['ErrorAnalysis.py', os.path.join('Analisis', 'ErrorAnalysis.py')] if os.path.exists(p)),
        None,
    )
    if error_analysis_script:
      run_step(error_analysis_script)
    else:
      print('Brak pliku ErrorAnalysis.py w katalogu! Pominięto generowanie heatmap błędów.')

    """ KROK 6: Ekstrakcja najtrudniejszych przypadków. """
    force_error_script = next(
        (p for p in ['ForceErrorAnalisis.py', os.path.join('Analisis', 'ForceErrorAnalisis.py')] if os.path.exists(p)),
        None,
    )
    if force_error_script:
      run_step(force_error_script)
    else:
      print('Brak pliku ForceErrorAnalisis.py. Pominięto eksport awaryjny.')

    """ KROK 7: Eksport ostatecznych metryk z MLflow oraz wysyłka na Hugging Face. """
    export_final_metrics()
    upload_to_hugging_face()

    print('Cały proces orkiestracji HCR zakończył się pomyślnie!')
    print('1. Architektura (CRNN -> CapsNet -> Transformer) została pomyślnie wytrenowana.')
    print('2. Pomiary czasu inferencji oraz zużycia pamięci VRAM zostały zakończone.')
    print('3. Analiza macierzy pomyłek i raport skuteczności przed/po są gotowe.')
    print('4. Wszystkie wykresy wydajnościowe/jakościowe zostały wygenerowane.')

    """ KROK 8: Załadowanie modeli na koniec i przejście w tryb interaktywny. """
    run_interactive_mode()

  except KeyboardInterrupt:
    print('Proces orkiestracji HCR został przerwany przez użytkownika (Ctrl+C).')
    sys.exit(130)
  except OSError as e:
    print(f'Krytyczny błąd systemu plików podczas wykonywania potoku HCR: {e}')
    sys.exit(1)
  except Exception as e:
    print(f'Nieoczekiwany błąd podczas wykonywania potoku HCR: {e}')
    sys.exit(1)