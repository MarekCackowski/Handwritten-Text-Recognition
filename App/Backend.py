import asyncio
import os
import io
import json
import uuid
import tempfile
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from typing import Dict, Any
import magic

import cv2 as cv
import numpy as np
import Levenshtein

import uvicorn
import redis.asyncio as redis_async
from celery import Celery

from prometheus_fastapi_instrumentator import Instrumentator

# Framework API i Chmura
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from minio import Minio
from fpdf import FPDF
from starlette.websockets import WebSocket, WebSocketDisconnect

# Obsługa wyjątków i logi
import json
from json import JSONDecodeError
from minio.error import S3Error
from pymongo.errors import PyMongoError
from redis.exceptions import RedisError
from fastapi import HTTPException
import logging
from passlib.context import CryptContext

# Logi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguracja szyfrowania haseł
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Konfiguracja
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongodb:27017")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
MINIO_URL = os.getenv("MINIO_URL", "minio:9000")
HTR_DATA_DIR = os.getenv("HTR_DATA_DIR", "/app/data/user_data")

# Klienci
redis_client = redis_async.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
minio_client = Minio(MINIO_URL, access_key="admin", secret_key="admin123", secure=False)

celery_app = Celery("htr_tasks", broker=f"redis://{REDIS_HOST}:6379/0", backend=f"redis://{REDIS_HOST}:6379/1")

if not minio_client.bucket_exists("htr-bucket"):
    minio_client.make_bucket("htr-bucket")

MAX_FILE_SIZE = 5 * 1024 * 1024 # Limit 5 MB na obraz
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

def validate_uploaded_image(file_stream):
    """ Weryfikuje nagłówki MIME oraz rozmiar pliku chroniąc vRAM i RAM serwera. """
    file_stream.seek(0, 2) 
    file_size = file_stream.tell()
    file_stream.seek(0) 
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Plik jest zbyt duży. Maksymalny rozmiar to 5 MB.")
        
    file_header = file_stream.read(2048)
    file_stream.seek(0) 
    
    mime_type = magic.from_buffer(file_header, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Niedozwolony format pliku ({mime_type}). Dozwolone to: JPG, PNG, WEBP.")
        
    return True

# Globalny stan API (Tylko baza)
state: Dict[str, Any] = {
    "db_client": None,
    "db": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Zwalnia zasoby API po zakończeniu działania. """
    # API łączy się z bazą asynchronicznie
    state["db_client"] = AsyncIOMotorClient(MONGO_URL)
    state["db"] = state["db_client"]["htr_database"]
    yield
    if state["db_client"]: state["db_client"].close()


app = FastAPI(lifespan=lifespan, title="Bilingual HTR API (Lightweight Gateway)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)


@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    """ Logowanie w oparciu o MongoDB. """
    try:
        user = await state["db"].users.find_one({"username": username})
    except Exception as e:
        logger.error(f"Błąd bazy danych podczas logowania: {e}")
        raise HTTPException(status_code=500, detail="Błąd wewnętrzny serwera")

    if not user:
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło")

    if not pwd_context.verify(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło")

    return {
        "access_token": str(user["username"]),
        "token_type": "bearer",
        "role": user.get("role", "user")
    }


@app.post("/process-document")
async def process_document(file: UploadFile = File(...), user_id: str = Form("default_user"),
                           language: str = Form("pl")):
    """ Przetwarza dokument i zwraca identyfikator zadania. Wzbogacone o twardą walidację i asynchroniczne I/O. """
    
    # Twarda weryfikacja bezpieczeństwa pliku
    validate_uploaded_image(file.file)

    task_id = str(uuid.uuid4())
    object_name = f"raw_scans/{task_id}_{file.filename}"

    fb = await file.read()
    
    # Zrzucenie synchronicznej operacji sieciowej MinIO do wątku pobocznego, aby nie blokować API
    await asyncio.to_thread(
        minio_client.put_object,
        "htr-bucket", object_name, io.BytesIO(fb), len(fb)
    )
    
    await redis_client.set(f"task:{task_id}:status", "PROCESSING")

    # Wysyłamy zadanie do wykonania
    celery_app.send_task("worker.process_document", args=[task_id, object_name, user_id, language])

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """ Zwraca status zadania. """
    status = await redis_client.get(f"task:{task_id}:status")
    if status == "SUCCESS":
        res = await redis_client.get(f"task:{task_id}:result")
        return {"status": status, "result": json.loads(res)}
    return {"status": status or "NOT_FOUND"}


@app.post("/submit-feedback")
async def submit_feedback(
        document_id: str = Form(...),
        word_id: str = Form(...),  # ID słowa
        box: str = Form(...),
        correct_text: str = Form(...),
        user_id: str = Form("default_user")
):
    try:
        bx, by, bw, bh = json.loads(box)

        # Asynchroniczne pobranie z MinIO i zamknięcie połączenia
        response = await asyncio.to_thread(minio_client.get_object, "htr-bucket", document_id)
        img_data = await asyncio.to_thread(response.read)
        response.close()
        response.release_conn()

        img_array = np.frombuffer(img_data, np.uint8)

        # Zrzucenie ciężkiego dekodowania OpenCV i zapisu dyskowego do wątku
        def process_and_save_crop():
            full_img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
            if full_img is None:
                raise ValueError("Błąd dekodowania wejściowego obrazu.")
                
            crop = full_img[by:by + bh, bx:bx + bw]
            path = os.path.join(HTR_DATA_DIR, user_id, "retrain_images")
            os.makedirs(path, exist_ok=True)
            fname = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            fpath = os.path.join(path, fname)
            cv.imwrite(fpath, crop)
            return fpath

        fpath = await asyncio.to_thread(process_and_save_crop)

        # Zapis do MongoDB
        await state["db"].user_corrections.insert_one({
            "user_id": user_id,
            "document_id": document_id,
            "word_id": word_id,
            "image_path": fpath,
            "ground_truth": correct_text,
            "used_for_training": False,
            "created_at": datetime.now(timezone.utc)
        })
        return {"status": "Feedback zapisany."}

    except ValueError as e:
        logger.error(f"Błąd przetwarzania obrazu: {e}")
        raise HTTPException(status_code=400, detail="Nie udało się zdekodować oryginału dokumentu z chmury.")
    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Błędny format pola 'box'. Oczekiwano JSON [x,y,w,h].")
    except S3Error as e:
        logger.error(f"Błąd MinIO: {e}")
        raise HTTPException(status_code=502, detail="Błąd pobierania obrazu z magazynu MinIO.")
    except (PyMongoError, RedisError) as e:
        logger.error(f"Błąd bazy/cache: {e}")
        raise HTTPException(status_code=500, detail="Błąd zapisu poprawki w bazie danych.")
    except OSError as e:
        logger.error(f"Błąd systemu plików: {e}")
        raise HTTPException(status_code=507, detail="Błąd zapisu wycinka obrazu na serwerze.")


@app.post("/generate-pdf")
async def generate_pdf(payload: Dict[str, Any]):
    """ Generuje raport PDF nakładający tekst (lub poprawki) na oryginał. """
    document_ids = payload.get("document_ids", [])
    user_id = payload.get("user_id", "default_user")

    if not document_ids:
        raise HTTPException(status_code=400, detail="Nie przekazano identyfikatorów dokumentów.")

    # Asynchroniczne zebranie wszystkich potrzebnych danych (Baza + Chmura)
    pages_data = []
    for doc_id in document_ids:
        try:
            page_record = await state["db"].prediction_history.find_one({
                "user_id": user_id,
                "data.document_id": doc_id
            })
            if not page_record:
                logger.info(f"Pominięto {doc_id}: brak rekordu w prediction_history.")
                continue

            corrections_cursor = state["db"].user_corrections.find({
                "user_id": user_id,
                "document_id": doc_id
            })
            corrections_map = {c["word_id"]: c["ground_truth"] async for c in corrections_cursor}

            # Asynchroniczne pobranie z MinIO i zamknięcie połączenia, żeby uniknąć wycieków/blokowania
            response = await asyncio.to_thread(minio_client.get_object, "htr-bucket", doc_id)
            img_data = await asyncio.to_thread(response.read)
            response.close()
            response.release_conn()

            pages_data.append({
                "img_data": img_data,
                "words": page_record["data"].get("words", []),
                "corrections_map": corrections_map
            })
        except Exception as e:
            logger.error(f"Błąd przy przygotowywaniu danych dla {doc_id}: {e}")
            continue

    if not pages_data:
        raise HTTPException(status_code=404, detail="Brak kompletnych danych do wygenerowania raportu.")

    # Synchroniczna, ciężka budowa pliku PDF i wysyłka do chmury (izolowana w wątku CPU, żeby uniknąć blokowania EventLoop)
    def build_and_upload_pdf():
        pdf = FPDF(unit="pt")
        try:
            pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', '', 12)
        except FileNotFoundError:
            logger.warning("Brak pliku DejaVuSans.ttf. PDF może nie obsługiwać polskich znaków.")
            pdf.set_font('Arial', '', 12)

        t_edit_dist, t_chars, t_gain, w_count = 0, 0, 0, 0

        for page in pages_data:
            img = cv.imdecode(np.frombuffer(page["img_data"], np.uint8), cv.IMREAD_GRAYSCALE)
            if img is None: continue
            
            h_img, w_img = img.shape
            pdf.add_page(format=(w_img, h_img))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(page["img_data"])
                tmp_img_path = tmp_img.name

            pdf.image(tmp_img_path, x=0, y=0, w=w_img, h=h_img)
            os.remove(tmp_img_path)

            pdf.set_text_color(0, 0, 255)

            for word in page["words"]:
                bx, by, bw, bh = word["box"]
                w_id = word.get("word_id")
                pred_text = word["final_result"]
                gt = page["corrections_map"].get(w_id)

                final_text = gt if gt else pred_text

                if gt:
                    t_edit_dist += Levenshtein.distance(pred_text, gt)
                    t_chars += len(gt)

                t_gain += word.get("confidence", 0)
                w_count += 1

                if final_text.strip():
                    pdf.set_font(pdf.font_family, size=max(6, bh * 0.75))
                    pdf.set_xy(bx, by)
                    pdf.cell(w=bw, h=bh, text=final_text, border=0)

        report_id = uuid.uuid4().hex
        pdf_name = f"reports/{user_id}/HTR_Report_{report_id}.pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            tmp_pdf_path = tmp_pdf.name

        minio_client.fput_object("htr-bucket", pdf_name, tmp_pdf_path)
        os.remove(tmp_pdf_path)

        f_cer = (t_edit_dist / max(1, t_chars)) * 100 if t_chars else 0.0
        a_conf = (t_gain / max(1, w_count)) * 100 if w_count else 0.0
        
        return pdf_name, f_cer, a_conf

    # Uruchomienie wyizolowanej funkcji w puli wątków
    try:
        pdf_name, final_cer, avg_confidence = await asyncio.to_thread(build_and_upload_pdf)
        
        # Generowanie linku też wymaga komunikacji HTTP z MinIO, więc przenosimy do wątku
        download_url = await asyncio.to_thread(
            minio_client.presigned_get_object, "htr-bucket", pdf_name, expires=timedelta(hours=1)
        )

        return {
            "status": "success",
            "download_url": download_url,
            "metrics": {
                "cer": round(final_cer, 2),
                "avg_confidence": round(avg_confidence, 2),
                "user_id": user_id
            }
        }
    except Exception as e:
        logger.error(f"Błąd zapisu/wysyłki raportu PDF: {e}")
        raise HTTPException(status_code=500, detail="Błąd przetwarzania raportu PDF w infrastrukturze chmurowej.")

@app.post("/trigger-adaptation/{user_id}")
async def trigger_adaptation(user_id: str, language: str = "pl"):
    """ Zleca uczenie do Workera. API samo nic nie uczy. """
    is_adapting = await redis_client.get(f"user:{user_id}:is_adapting")

    if is_adapting == "True":
        return {"status": "busy", "message": "Proces adaptacji już trwa."}

    await redis_client.set(f"user:{user_id}:is_adapting", "True")
    celery_app.send_task("worker.run_adaptation", args=[user_id, language])
    return {"status": "started", "task": "adaptation"}


@app.websocket("/ws/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    """ Zwraca status zadania w czasie rzeczywistym. """
    await websocket.accept()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"channel:{task_id}")

    try:
        # Używamy asynchronicznego generatora
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                await websocket.send_text(data)

                if "SUKCES" in data or "NIEPOWODZENIE" in data:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"channel:{task_id}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem"
    )