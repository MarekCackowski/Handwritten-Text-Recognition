import os
import shutil

import cv2 as cv
import time
import torch
import numpy as np
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.collection import Collection
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tran
from scipy.ndimage import gaussian_filter, map_coordinates

def now():
    return datetime.now().strftime("%H:%M:%S")

class UserCorrectionDataset(torch.utils.data.Dataset):
    """ Lekki dataset specjalnie do obsługi krotek (ścieżka, etykieta) z Active Learningu. """
    def __init__(self, samples_list, transform):
        self.samples = samples_list
        self.transform = transform
        self.TARGET_HEIGHT = 64
        self.MAX_WIDTH = 512

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None: return None

        if np.mean(img) < 127:
            img = cv.bitwise_not(img)

        h_orig, w_orig = img.shape
        if h_orig == 0 or w_orig == 0: return None

        length = len(label)
        category = 'short' if length < 5 else ('medium' if length < 9 else 'long')

        scale = self.TARGET_HEIGHT / h_orig
        new_w = int(w_orig * scale)
        new_h = self.TARGET_HEIGHT

        if new_w > self.MAX_WIDTH:
            scale = self.MAX_WIDTH / w_orig
            new_w = self.MAX_WIDTH
            new_h = int(h_orig * scale)

        new_w = max(8, new_w)
        new_h = max(8, new_h)
        img_resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

        # Zmuszamy lintera do zrozumienia, że to jest macierz ndarray
        final_img: np.ndarray = np.full((self.TARGET_HEIGHT, new_w), 255, dtype=np.uint8)
        y_offset = max(0, (self.TARGET_HEIGHT - new_h) // 2)

        # Teraz linter bez problemu zaakceptuje przypisanie do wycinka
        final_img[y_offset: y_offset + new_h, :] = img_resized

        augmented = self.transform(image=final_img)
        return augmented['image'], label, category, path


class ElasticTransform(object):
    """ Nieliniowe wyginanie obrazu symulujące ruchy ręki. """
    def __init__(self, alpha=35, sigma=5):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, tensor):
        image = tensor.cpu().detach().numpy().squeeze()
        shape = image.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        return torch.from_numpy(distorted).unsqueeze(0)


class FeedbackAugmentor:
    """ Zwraca zdeformowany obraz, symulując możliwe różnice w pismie odręcznym. """
    def __init__(self):
        self.elastic = ElasticTransform(alpha=30, sigma=5)  # wyginanie tekstu
        self.standard_aug = tran.Compose([
            tran.RandomApply([tran.RandomRotation(degrees=3)], p=0.5),
            tran.RandomApply([tran.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        ])

    def generate_variations(self, image_tensor, num_variations=5):
        variations = [image_tensor]
        for _ in range(num_variations):
            # Najpierw nieliniowe wygięcie
            aug_img = self.elastic(image_tensor.cpu())
            # Potem rotacja i szum
            aug_img = self.standard_aug(aug_img.to(image_tensor.device))
            noise = torch.randn_like(aug_img) * 0.015
            variations.append(aug_img + noise)
        return torch.cat(variations, dim=0)


class JointFeedbackFineTuner:
    """ Zaawansowany moduł dostrajający kaskadę HTR na żywo. Obsługuje jednoczesną optymalizację:
        1. CRNN (CTC Loss) - uczy się ogólnego wyglądu słów.
        2. CapsNet (Capsule Loss) - poprawia precyzję detekcji znak po znaku.
        3. Transformer ByT5 (Seq2Seq Loss) - adaptuje słownik i gramatykę do autora.
        Wykorzystuje akumulację gradientów, aby umożliwić trening na laptopach z małym VRAM. """
    def __init__(self, crnn_model, caps_model, transformer_model, encoder, device,
                 tokenizer=None, language="pl", accumulation_steps=16):
        self.crnn = crnn_model
        self.caps = caps_model
        self.transformer = transformer_model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        self.language = language
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

        # Bardzo małe współczynniki uczenia
        self.opt_crnn = optim.Adam(self.crnn.parameters(), lr=1e-5)

        self.opt_caps = None
        if self.caps:
            self.opt_caps = optim.Adam(self.caps.parameters(), lr=5e-7)

        self.opt_transformer = None
        if self.transformer:
            self.opt_transformer = optim.Adam(self.transformer.parameters(), lr=1e-6)

        # Funkcje straty
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # Zerowanie gradientów na starcie
        self.opt_crnn.zero_grad()
        if self.opt_caps is not None:
            self.opt_caps.zero_grad()
        if self.opt_transformer is not None:
            self.opt_transformer.zero_grad()

    def fine_tune_on_feedback(self, images_batch, original_pred_text, corrected_text, segmentor, prev_context=""):
        """ Wykonuje krok uczenia na podstawie poprawki wprowadzonej przez użytkownika. """
        self.crnn.train()
        if self.caps: self.caps.train()
        if self.transformer: self.transformer.train()

        batch_size = images_batch.size(0)

        # CRNN
        logits, features = self.crnn(images_batch.to(self.device), return_context=True)

        targets = [self.encoder.char_to_idx.get(c, 0) for c in corrected_text if c in self.encoder.char_to_idx]
        if targets:
            target_tensor = torch.IntTensor(targets).to(self.device)
            target_lengths = torch.IntTensor([len(targets)] * batch_size).to(self.device)

            log_probs = logits.log_softmax(2)
            input_lengths = torch.IntTensor([log_probs.size(0)] * batch_size).to(self.device)

            l_crnn = self.ctc_loss(log_probs, target_tensor.repeat(batch_size), input_lengths, target_lengths)
            l_crnn = l_crnn / self.accumulation_steps

            if not (torch.isnan(l_crnn) or torch.isinf(l_crnn)):
                l_crnn.backward(retain_graph=True)

        # CapsNet
        if self.caps and len(corrected_text) > 0:
            # Przetwarzamy pierwszy obraz z batcha (oryginał bez augmentacji) do detekcji znaków
            img_np = (images_batch[0].squeeze().cpu().detach().numpy() * 127.5 + 127.5).astype(np.uint8)
            char_crops = segmentor.extract_atomic_crops(img_np)

            # Jeśli segmentacja fizyczna zgadza się z liczbą znaków w poprawce
            if len(char_crops) == len(corrected_text):
                for i, (crop_img, _) in enumerate(char_crops):
                    label_idx = self.encoder.char_to_idx.get(corrected_text[i])
                    if label_idx is None: continue

                    crop_t = self._prepare_caps_crop(crop_img)

                    # Fuzja: bierzemy wektor kontekstu z CRNN dla środka znaku
                    t_idx = int((i + 0.5) * features.size(0) / len(corrected_text))
                    context_vec = features[t_idx, 0, :].unsqueeze(0).detach()

                    # CapsNet Forward (uproszczony dla fine-tuningu)
                    caps_out = self.caps(crop_t, word_context=context_vec)
                    probs = caps_out[0] if isinstance(caps_out, tuple) else caps_out

                    # Prosty Margin Loss dla CapsNet
                    target_one_hot = torch.eye(probs.size(-1)).to(self.device)[label_idx]
                    l_caps = torch.mean(torch.sum(target_one_hot * torch.relu(0.9 - probs) ** 2 + 0.5 * (1 - target_one_hot) * torch.relu(probs - 0.1) ** 2, dim=-1))

                    l_caps = l_caps / (self.accumulation_steps * len(char_crops))
                    l_caps.backward()

        # Zarządzanie krokiem optymalizacji
        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            self._apply_gradients()

        # Transformer
        if self.transformer and self.tokenizer and corrected_text != original_pred_text:
            # Budowa promptu zgodna z wybranym językiem
            if self.language == "pl":
                prompt = f"kontekst: {prev_context} popraw ocr: {original_pred_text}"
            else:
                prompt = f"context: {prev_context} fix ocr: {original_pred_text}"

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=384, truncation=True).to(self.device)
            labels = self.tokenizer(corrected_text, return_tensors="pt", max_length=256, truncation=True).input_ids.to(
                self.device)

            outputs = self.transformer(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            l_trans = outputs.loss / self.accumulation_steps

            if not (torch.isnan(l_trans) or torch.isinf(l_trans)):
                l_trans.backward(retain_graph=True)

    def _apply_gradients(self):
        """ Aplikuje skumulowane gradienty i czyści bufory. """
        torch.nn.utils.clip_grad_norm_(self.crnn.parameters(), 5.0)
        self.opt_crnn.step()
        self.opt_crnn.zero_grad()

        if self.opt_caps is not None:
            self.opt_caps.step()
            self.opt_caps.zero_grad()

        self.opt_transformer = None
        if self.transformer:
            # Wybieramy parametry, które LoRA odblokowała do treningu
            trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

            # LoRA uczy się tylko małej części modelu, więc  1e-4
            self.opt_transformer = optim.Adam(trainable_params, lr=1e-4)

    def finalize_session_learning(self):
        """ Wymusza aktualizację wag na koniec sesji, jeśli zostały jakieś resztki w buforze. """
        if self.current_step % self.accumulation_steps != 0:
            self._apply_gradients()
        print(f"[{time.strftime('%H:%M:%S')}] Pętla zwrotna: Wagi zaktualizowane pomyślnie.")

    def save_fine_tuned_weights(self, al_manager):
        """ Zapisuje modele .pth oraz automatycznie synchronizuje szybkie modele .onnx. """
        paths = al_manager.get_model_paths()

        # Zapis PyTorch
        torch.save({'model_state': self.crnn.state_dict(), 'char_list': self.encoder.chars}, paths['crnn'])
        if self.caps:
            torch.save({'model_state': self.caps.state_dict()}, paths['caps'])
        if self.transformer:
            self.transformer.save_pretrained(os.path.dirname(paths['trans']))

        # Synchronizacja ONNX dla natychmiastowej reakcji GUI
        print(f"[{time.strftime('%H:%M:%S')}] Eksportowanie nowych wag do ONNX.")
        al_manager.sync_onnx_weights(self.crnn, "crnn")
        if self.caps:
            al_manager.sync_onnx_weights(self.caps, "caps")

    def _prepare_caps_crop(self, crop_img):
        """ Przygotowuje mały obrazek znaku pod CapsNet. """
        if crop_img.shape != (64, 64):
            crop_img = cv.resize(crop_img, (64, 64))
        crop_t = torch.from_numpy(crop_img).float().unsqueeze(0).unsqueeze(0)
        crop_t = (crop_t / 255.0 - 0.5) / 0.5
        return crop_t.to(self.device)


class ActiveLearningManager:
    """ Zarządza procesem ciągłego uczenia i personalizacji modeli HTR. Odpowiada za komunikację z bazą MongoDB,
        zarządzanie ścieżkami do spersonalizowanych wag modeli (CRNN, CapsNet, Transformer) oraz przygotowywanie
        paczek treningowych opartych na mechanizmie Experience Replay. Wykorzystuje MC Dropout do oceny niepewności,
        filtrując tylko najtrudniejsze próbki do dotrenowania (zapobiegając zjawisku katastroficznego zapominania). """
    def __init__(self, user_id="default_user", storage_base="active_learning_data"):
        self.user_id = user_id

        # Ścieżki dyskowe do przechowywania samych plików graficznych
        self.user_path = os.path.join(storage_base, user_id)
        self.images_path = os.path.join(self.user_path, "retrain_images")
        os.makedirs(self.images_path, exist_ok=True)

        # Zdefiniowanie katalogu modeli dla danego użytkownika
        self.user_models_path = os.path.join("output_data", "checkpoints", "users", user_id)
        os.makedirs(self.user_models_path, exist_ok=True)

        # Ścieżki dla profili użytkownika (zapisywane modele po Fine-Tuningu)
        self.model_path_crnn = os.path.join("output_data/checkpoints/users", f"{user_id}_crnn.pth")
        self.model_path_caps = os.path.join("output_data/checkpoints/users", f"{user_id}_caps.pth")
        self.model_path_trans = os.path.join("output_data/checkpoints/users", f"{user_id}_trans.pth")

        # Ścieżki modeli bazowych
        self.base_paths = {
            "crnn": r"output_data\checkpoints\hwr\WordLevelResNetCRNN.pth",
            "caps": r"output_data\checkpoints\hcr\CharLevelCapsNet.pth",
            "trans": "./byt5_htr_refiner_final"
        }

        os.makedirs(os.path.dirname(self.model_path_crnn), exist_ok=True)

        # Synchroniczne połączenie z bazą MongoDB
        try:
            self.client = MongoClient("mongodb://172.189.176.192:27017", serverSelectionTimeoutMS=5000)
            # Szybki test połączenia
            self.client.server_info()
            self.db = self.client["htr_database"]
            self.collection = self.db["user_corrections"]
        except Exception as e:
            print(f"[{now()}] Błąd krytyczny: Nie można połączyć się z MongoDB! {e}")
            self.client = None
            self.db = None
            self.collection = None

        # Lista na _id dokumentów przetwarzanych w bieżącej sesji treningowej
        self.processed_ids = []

    def get_model_paths(self, version=None, use_onnx=False):
        """ Rozszerzona wersja obsługująca ONNX dla GUI. """
        paths = {}
        sub_dir = f"v_{version}" if version else "latest"
        version_path = os.path.join(self.user_models_path, sub_dir)

        model_types = ["crnn", "caps", "trans"]

        for m_type in model_types:
            # Wybór rozszerzenia: .onnx dla GUI, .pth dla Treningu
            ext = ""
            if m_type != "trans":
                ext = ".onnx" if use_onnx else ".pth"

            u_path = os.path.join(version_path, f"{m_type}{ext}")

            if os.path.exists(u_path):
                paths[m_type] = u_path
            else:
                # Fallback do bazy
                paths[m_type] = self.base_paths[m_type]
                if use_onnx:
                    paths[m_type] = paths[m_type].replace(".pth", ".onnx")

        return paths

    def capture_correction(self, word_img, user_correction, crnn_res, capsnet_res, final_res, language="pl"):
        """ Opcjonalna metoda do zapisu prosto z backendu (w razie potrzeby, choć API zapisuje to asynchronicznie) """
        if self.collection is None:
            return
        assert self.collection is not None

        coll = self.collection
        if coll is None: return

        is_error = final_res != user_correction

        if is_error:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            sample_id = f"{self.user_id}_{timestamp}"
            img_filename = f"{sample_id}.png"
            full_img_path = os.path.join(self.images_path, img_filename)

            cv.imwrite(full_img_path, word_img)

            document = {
                "user_id": self.user_id,
                "filename": img_filename,
                "image_path": full_img_path,
                "ground_truth": user_correction,
                "crnn_result": crnn_res,
                "crnn_capsnet_result": capsnet_res,
                "final_result": final_res,
                "used_for_training": False,
                "created_at": datetime.now(timezone.utc)
            }

            try:
                coll: Collection = self.collection
                coll.insert_one(document)
                print(
                    f"Zapisano błędną próbkę dla {self.user_id}: oczekiwano '{user_correction}', model zwrócił '{final_res}'")
            except Exception as e:
                print(f"Błąd zapisu w MongoDB: {e}")

    def prepare_finetune_batch(self, min_samples=50):
        """ Sprawdza, czy zebrano wystarczająco dużo nowych danych. """
        if self.collection is None:
            return False
        assert self.collection is not None

        query = {"user_id": self.user_id, "used_for_training": False}

        coll: Collection = self.collection
        new_samples_count = coll.count_documents(query)

        if new_samples_count >= min_samples:
            print(f"[{now()}] Gotowość do treningu! Zebrano {new_samples_count} nowych próbek dla {self.user_id}.")
            return True

        return False

    def mark_samples_as_trained(self):
        """ Oznacza próbki przetwarzane w obecnej sesji jako 'użyte w treningu'. """
        if self.collection is None or not self.processed_ids:
            return
        assert self.collection is not None

        try:
            coll: Collection = self.collection
            result = coll.update_many(
                {"_id": {"$in": self.processed_ids}},
                {"$set": {"used_for_training": True}}
            )
            print(f"[{now()}] Oznaczono {result.modified_count} próbek jako wytrenowane dla {self.user_id}.")
            self.processed_ids = []
        except Exception as e:
            print(f"[{now()}] Błąd podczas oznaczania próbek w bazie: {e}")

    def get_training_data(self, model=None, transform=None, top_k=50, num_passes=10, uncertainty_threshold=0.01):
        """ Zwraca dane do dotrenowania pobrane z MongoDB. Filtruje błędy przez MC Dropout
            i wybiera 'top_k' najbardziej niepewnych próbek, co zapobiega zapominaniu (experience replay). """
        if self.collection is None:
            return []
        assert self.collection is not None

        # Zbieramy dokumenty, których użyjemy
        coll = self.collection
        if coll is None: return []

        query = {"user_id": self.user_id, "used_for_training": False}

        coll: Collection = self.collection
        cursor = coll.find(query)

        samples = []
        raw_documents = []

        for doc in cursor:
            img_path = doc.get("image_path")
            ground_truth = doc.get("ground_truth")

            if img_path and os.path.exists(img_path):
                samples.append((img_path, ground_truth))
                raw_documents.append(doc)
            else:
                print(f"Brak pliku na dysku: {img_path}")

        # Jeśli brak modelu lub transformacji, lub mało próbek -> od razu zwracamy to co mamy
        if model is None or transform is None or len(samples) <= top_k:
            self.processed_ids.extend([doc["_id"] for doc in raw_documents])
            return samples

        # Aktywacja MC Dropout dla wyselekcjonowania najtrudniejszych próbek
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        device = next(model.parameters()).device
        uncertainties = []

        with torch.no_grad():
            for path, _ in samples:
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is None:
                    uncertainties.append(0.0)
                    continue

                # Normalizacja wymiarów pod przepuszczenie przez sieć
                h, w = img.shape
                new_w = max(8, int(w * (64.0 / h))) if h > 0 else 64
                img = cv.resize(img, (new_w, 64), interpolation=cv.INTER_AREA)

                # Formatowanie do Albumentations wymaga shape [H, W]
                augmented = transform(image=img)
                tensor_img = augmented['image'].unsqueeze(0).to(device)

                predictions = []
                for _ in range(num_passes):
                    output = model(tensor_img)
                    if isinstance(output, tuple): output = output[0]
                    predictions.append(output.unsqueeze(0))

                predictions = torch.cat(predictions, dim=0)
                probs = torch.softmax(predictions, dim=-1)
                variance = torch.var(probs, dim=0)
                uncertainties.append(torch.mean(variance).item())

        # Parowanie próbek z ich oceną trudności (niepewnością) i oryginalnym dokumentem bazy
        scored_samples = list(zip(samples, uncertainties, raw_documents))

        # Odrzucamy próbki, których model jest już pewien
        uncertain_samples = [item for item in scored_samples if item[1] > uncertainty_threshold]

        # Sortowanie po najwyższej niepewności (najtrudniejsze na początek)
        uncertain_samples.sort(key=lambda x: x[1], reverse=True)

        # Wybieramy top_k próbek
        final_selection = uncertain_samples[:top_k]

        # Zapisujemy ID wyselekcjonowanych dokumentów do oznaczenia po treningu
        self.processed_ids.extend([item[2]["_id"] for item in final_selection])

        return [item[0] for item in final_selection]

    def get_transformer_training_pairs(self):
        """ Zwraca pary tekstowe do douczania Transformera. """
        training_pairs = []
        if self.collection is None:
            return training_pairs
        assert self.collection is not None

        query = {"user_id": self.user_id, "used_for_training": False}

        coll: Collection = self.collection
        cursor = coll.find(query)

        for doc in cursor:
            # Próbujemy pobrać wynik po weryfikacji CapsNet, jeśli jest dostępny
            source_text = doc.get('crnn_capsnet_result') or doc.get('crnn_result', "")
            target_text = doc.get('ground_truth', "")

            if source_text and target_text and source_text != target_text:
                training_pairs.append((source_text, target_text))

        return training_pairs

    def save_optimized_cascade(self, crnn, caps=None, transformer=None, encoder=None):
        """ Zintegrowany zapis kaskady. Tworzy nową wersję w folderze użytkownika i synchronizuje ONNX. """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        version_dir = os.path.join(self.user_models_path, f"v_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)

        try:
            # Zapis CRNN + CharList (niezbędne do poprawnego dekodowania)
            crnn_path = os.path.join(version_dir, "crnn.pth")
            torch.save({
                'model_state': crnn.state_dict(),
                'char_list': encoder.char_list if encoder else []
            }, crnn_path)

            # Zapis CapsNet
            if caps:
                torch.save({'model_state': caps.state_dict()}, os.path.join(version_dir, "caps.pth"))

            # Zapis Transformera
            if transformer:
                trans_dir = os.path.join(version_dir, "transformer")
                transformer.save_pretrained(trans_dir)

            # Synchronizacja ONNX (dla szybkości GUI)
            self.sync_onnx_weights(crnn, model_type="crnn", target_path=crnn_path.replace(".pth", ".onnx"))

            # 5. Linkowanie jako 'latest'
            self._update_latest_link(version_dir)

            print(f"[{now()}] AL_MANAGE: Zarchiwizowano kaskadę jako wersję v_{timestamp}")
            return True
        except Exception as e:
            print(f"[{now()}] Błąd krytyczny AL_MANAGE: {e}")
            return False

    def _update_latest_link(self, version_dir):
        """ Aktualizuje folder 'latest', kopiując do niego najnowsze wytrenowane modele. """
        latest_dir = os.path.join(self.user_models_path, "latest")
        os.makedirs(latest_dir, exist_ok=True)

        try:
            # Pobieramy listę wszystkich plików i folderów z nowej wersji
            for file_name in os.listdir(version_dir):
                src_file = os.path.join(version_dir, file_name)
                dst_path = os.path.join(latest_dir, file_name)  # Ujednolicona nazwa zmiennej

                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_path) # Kopiowanie plików (.pth, .onnx, .json)
                elif os.path.isdir(src_file):
                    # Obsługa folderów (np. folder transformera)
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_file, dst_path)

            print(f"[{now()}] AL_MANAGE: Zaktualizowano folder 'latest' nową wersją z {os.path.basename(version_dir)}.")
        except Exception as e:
            print(f"[{now()}] Błąd podczas aktualizacji folderu latest: {e}")

    def sync_onnx_weights(self, model_pytorch, model_type="crnn", target_path=None):
        """ Konwertuje najnowsze wagi .pth na format ONNX, a następnie wykonuje
            dynamiczną kwantyzację do INT8, aby zoptymalizować działanie na CPU użytkownika. """
        # Jeśli nie podano ścieżki docelowej, użyj domyślnej
        if target_path is None:
            path_pth = self.get_model_paths()[model_type]
            target_path = path_pth.replace(".pth", "_int8.onnx")  # Celujemy od razu w INT8
            fp32_path = path_pth.replace(".pth", "_temp_fp32.onnx")
        else:
            fp32_path = target_path.replace(".onnx", "_temp_fp32.onnx")
            target_path = target_path.replace(".onnx", "_int8.onnx")

        # Zabezpieczenie przed błędem lokalizacji
        original_device = next(model_pytorch.parameters()).device
        model_pytorch.eval()

        # Eksport do ONNX jest najstabilniejszy na CPU
        model_pytorch.to('cpu')

        dummy_input = torch.randn(1, 1, 64, 576, requires_grad=True)

        try:
            # Eksport do formatu FP32
            torch.onnx.export(
                model_pytorch,
                dummy_input,
                fp32_path,
                export_params=True,
                opset_version=14,  # Podniesiono opset dla lepszej kompatybilności kwantyzacji
                do_constant_folding=True,
                input_names=['input_images'],
                output_names=['logits', 'context_maps'] if model_type == "crnn" else ['output'],
                dynamic_axes={
                    'input_images': {0: 'batch_size', 3: 'width'},
                    'logits': {0: 'time_steps', 1: 'batch_size'},
                    'context_maps': {0: 'time_steps', 1: 'batch_size'}
                } if model_type == "crnn" else {'input': {3: 'width'}, 'output': {0: 'timesteps'}}
            )
            print(f"[{now()}] Wygenerowano pośredni model ONNX (FP32).")

            # Dynamiczna Kwantyzacja do INT8
            quantize_dynamic(
                model_input=fp32_path,
                model_output=target_path,
                weight_type=QuantType.QUInt8
            )

            # Sprzątanie plików tymczasowych
            if os.path.exists(fp32_path):
                os.remove(fp32_path)

            print(f"[{now()}] Zsynchronizowano i SKWANTYZOWANO szybki model Edge AI: {target_path}")

        except Exception as e:
            print(f"[{now()}] Błąd podczas eksportu/kwantyzacji ONNX: {e}")
        finally:
            # Przywracamy model na oryginalne urządzenie, aby nie zepsuć kolejnych operacji
            model_pytorch.to(original_device)

    def __del__(self):
        """ Zamykanie połączenia z bazą przy destrukcji instancji klasy. """
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()