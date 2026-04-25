import os
import json
import torch
import cv2 as cv
import numpy as np
from datetime import datetime, timezone
from celery import Celery
import redis
from minio import Minio
from pymongo import MongoClient
from typing import Optional, Dict, Any, List, cast
import uuid
# Importy Twoich modeli
from Models.DeepCapsNetCharRecognition import CapsNet
from Models.ResNetCRNNWordRecognition import ResNetCRNN
from Preprocessing.Preprocessing import Preprocessing as ImagePreprocessor

# Poprawione importy klas z odpowiednich plików
from App.CRNNCNTROCR import (
    CRNNInferencePipeline,
    LineToWordSegmentor,
    PageToLineSegmentor
)
from Models.TransformerDictionaryRefinement import CascadeRefinementNetwork, TransformerRefiner

# Konfiguracja
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://172.189.176.192:27017")
MINIO_URL = os.getenv("MINIO_URL", "localhost:9000")

# Dynamiczne ścieżki dla Dockera
HTR_DATA_DIR = os.getenv("HTR_DATA_DIR", "/app/data")
CHECKPOINT_DIR = os.path.join(HTR_DATA_DIR, "output_data", "checkpoints")

CRNN_PATH = os.path.join(CHECKPOINT_DIR, "hwr", "WordLevelResNetCRNN.pth")
CAPS_PATH = os.path.join(CHECKPOINT_DIR, "hcr", "CharLevelCapsNet.pth")
TRANSFORMER_PATH = os.path.join(CHECKPOINT_DIR, "transformer", "byt5_htr_refiner_final")

# Klienci
celery_app = Celery('htr_tasks', broker=f'redis://{REDIS_HOST}:6379/0')
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
minio_client = Minio(MINIO_URL, access_key="admin", secret_key="admin123", secure=False)
mongo_client = MongoClient(MONGO_URL)
db = mongo_client["htr_database"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicjalizacja pustego potoku z poprawnym typowaniem
inference_pipe: Optional[CRNNInferencePipeline] = None


class CTCLabelEncoder:
    """ Enkoder dla Workera. """
    def __init__(self, chars: List[str]):
        self.char_to_idx = {char: i + 1 for i, char in enumerate(chars)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(chars)}
        self.idx_to_char[0] = ""
        self.vocab_size = len(chars) + 1


def load_models() -> None:
    """ Inicjalizacja kaskady modeli w pamięci GPU Workera. """
    global inference_pipe
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Inicjalizacja modeli w Celery Worker.")

    if not os.path.exists(CRNN_PATH):
        print("BŁĄD: Nie znaleziono wag CRNN!")
        return

    ckpt = torch.load(CRNN_PATH, map_location=device)
    char_list = ckpt.get("char_list", [])
    encoder = CTCLabelEncoder(char_list)

    model = ResNetCRNN(encoder.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])

    model_caps = CapsNet(num_classes=len(char_list)).to(device)
    if os.path.exists(CAPS_PATH):
        c_ckpt = torch.load(CAPS_PATH, map_location=device)
        model_caps.load_state_dict(c_ckpt.get('model_state', c_ckpt), strict=False)

    pipe = CRNNInferencePipeline(model, char_list, device)
    pipe.refiner = CascadeRefinementNetwork(model_caps, char_list, pipe)
    pipe.capsnet = model_caps
    pipe.preprocessor = ImagePreprocessor()

    if os.path.exists(TRANSFORMER_PATH):
        # The TransformerRefiner class now handles loading the base model and setting up PeftModel internally
        pipe.transformer = TransformerRefiner(TRANSFORMER_PATH, device, language="pl")

    inference_pipe = pipe
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Worker gotowy do pracy.")


# Inicjalizujemy modele przy starcie Workera
load_models()


@celery_app.task(name="worker.process_document")
def process_document_worker(task_id: str, object_name: str, user_id: str, language: str) -> None:
    """ Główne zadanie Celery przetwarzające dokument. """
    global inference_pipe

    pipe = inference_pipe
    if pipe is None or pipe.preprocessor is None:
        redis_client.set(f"task:{task_id}:status", "FAILED")
        print(f"Błąd Workera: Pipeline nie jest gotowy.")
        return

    # Sprawdzenie Transformera
    if pipe.transformer is not None:
        pipe.transformer.language = language

        # Ładowanie profilu LoRA odpowiedniego użytkownika
        pipe.transformer.set_user_expert(user_id)

    try:
        # Pobranie pliku z MinIO
        response = minio_client.get_object("htr-bucket", object_name)
        img_array = np.frombuffer(response.read(), np.uint8)
        raw_img = np.asarray(cv.imdecode(img_array, cv.IMREAD_GRAYSCALE))

        preprocessor_instance = cast(ImagePreprocessor, cast(Any, pipe.preprocessor))
        processed_raw = preprocessor_instance.full_pipeline(raw_img)

        processed_arr = np.asarray(processed_raw)
        processed = np.asarray(processed_arr * 255, dtype=np.uint8)

        if float(np.mean(processed)) < 127:
            processed = np.asarray(cv.bitwise_not(processed))

        line_seg = PageToLineSegmentor()
        word_seg = LineToWordSegmentor()
        lines = line_seg.extract_lines(processed)

        words_data: List[Dict[str, Any]] = []

        for l_idx, line_img in enumerate(lines):
            word_crops = word_seg.extract_atomic_crops(np.asarray(line_img))

            for (crop, (bx, by, bw, bh)) in word_crops:
                bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
                global_y = int(by) + (int(l_idx) * 75)

                crop_arr = np.asarray(crop)
                h, w = int(crop_arr.shape[0]), int(crop_arr.shape[1])
                new_w = max(16, int(w * (64.0 / h)))

                res = np.asarray(cv.resize(crop_arr, (new_w, 64), interpolation=cv.INTER_CUBIC))
                _, bin_img = cv.threshold(res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                bin_img_arr = np.asarray(bin_img)

                t_crnn = (torch.from_numpy(bin_img_arr).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
                bin_inv = np.asarray(cv.bitwise_not(bin_img_arr))
                caps_ready = ImagePreprocessor.standardize_ink_thickness(bin_inv, target_thickness=3)
                t_caps = (torch.from_numpy(caps_ready).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

                with torch.no_grad():
                    res_list = pipe.predict_batch([t_crnn.unsqueeze(0)], [t_caps.unsqueeze(0)])
                    if res_list:
                        r = res_list[0]
                        raw_mask = r.get('uncertainty_mask', [])

                        rel_zones = [
                            {'x': float(i * (1.0 / len(raw_mask))), 'w': float(1.0 / len(raw_mask))}
                            for i, val in enumerate(raw_mask) if val > 0
                        ] if raw_mask else []

                        # Generujemy unikalne ID dla każdego słowa na stronie
                        unique_word_id = f"word_{task_id}_{l_idx}_{uuid.uuid4().hex[:6]}"

                        words_data.append({
                            "word_id": unique_word_id,
                            "box": [bx, global_y, bw, bh],
                            "final_result": str(r.get('final_result', "")),
                            "crnn_result": str(r.get('crnn_result', "")),
                            "capsnet_result": str(r.get('capsnet_result', "")),
                            "confidence": float(r.get('hybrid_confidence', 0.0)),
                            "uncertain_zones": rel_zones
                        })

        final_result = {"document_id": object_name, "words": words_data}
        result_json = json.dumps(final_result)

        # Zapis do Redisa
        redis_client.set(f"task:{task_id}:result", result_json)
        redis_client.set(f"task:{task_id}:status", "SUCCESS")

        # Zapis do bazy danych MongoDB
        db.prediction_history.insert_one({
            "task_id": task_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "data": final_result
        })

        # Publikacja wiadomości przez Pub/Sub, żeby FastAPI kontynuowało pracę
        ws_message = json.dumps({"status": "SUCCESS", "result": final_result})
        redis_client.publish(f"channel:{task_id}", ws_message)

    except Exception as e:
        redis_client.set(f"task:{task_id}:status", "FAILED")

        # W razie błędu też informujemy UI, żeby nie czekało w nieskończoność
        redis_client.publish(f"channel:{task_id}", json.dumps({"status": "FAILED", "error": str(e)}))
        print(f"Błąd Workera: {str(e)}")