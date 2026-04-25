import os
import json
import torch
import gc
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from Models.ResNetCRNNWordRecognition import (
    ResNetCRNN, collate_fn_dynamic, get_augmentations
)
from Models import DeepCapsNetCharRecognition
from App.CRNNCNTROCR import CRNNInferencePipeline, LineToWordSegmentor
from Refinement.HybridBayesianActiveLearning import (
    ActiveLearningManager,
    JointFeedbackFineTuner,
    FeedbackAugmentor
)
from Models.TransformerDictionaryRefinement import (
    CascadeRefinementNetwork,
    run_transformer_adaptation
)
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def now():
    """ Obecna godzina w formacie HH:MM:SS. """
    return datetime.now().strftime("%H:%M:%S")


class CTCLabelEncoder:
    """ Encoder znaków. Zarządza mapowaniem znaków i dekodowaniem sekwencji CTC. """
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {char: i + 1 for i, char in enumerate(chars)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(chars)}
        self.idx_to_char[0] = ""  # CTC Blank
        self.vocab_size = len(chars) + 1

    def encode(self, text):
        return torch.LongTensor([self.char_to_idx[c] for c in text if c in self.char_to_idx])


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

        # Tworzymy białe tło
        final_img: np.ndarray = np.full((self.TARGET_HEIGHT, new_w), 255, dtype=np.uint8)

        y_offset = max(0, (self.TARGET_HEIGHT - new_h) // 2)

        # Wklejamy obrazek w tło
        final_img[y_offset: y_offset + new_h, :] = img_resized

        # Przekazujemy do augmentacji
        augmented = self.transform(image=final_img)
        return augmented['image'], label, category, path


def export_all_to_onnx(model_crnn, model_caps, output_dir):
    """ Eksportuje modele do formatu ONNX, a następnie wykonuje dynamiczną kwantyzację do INT8.
        Zabieg ten drastycznie redukuje rozmiar modelu (ok. 4-krotnie) i przyspiesza
        inferencję na procesorach CPU bez dedykowanej karty graficznej. """
    os.makedirs(output_dir, exist_ok=True)

    # Przełączamy modele w tryb ewaluacji i przenosimy na CPU.
    # Kwantyzacja ONNX Runtime jest najstabilniejsza, gdy proces odbywa się na procesorze.
    model_crnn.eval().cpu()
    model_caps.eval().cpu()

    tqdm.write(f"[{now()}] Rozpoczynam proces eksportu i kwantyzacji INT8.")

    # Eksport i kwantyzacja CRNN
    crnn_temp_fp32 = os.path.join(output_dir, "crnn_temp_fp32.onnx")
    crnn_final_int8 = os.path.join(output_dir, "crnn_user_int8.onnx")

    # Przykładowy tensor wejściowy (batch_size=1, kanały=1, H=64, W=256)
    sample_crnn_input = torch.randn(1, 1, 64, 256)

    # Standardowy eksport do ONNX
    torch.onnx.export(
        model_crnn,
        sample_crnn_input,
        crnn_temp_fp32,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_img'],
        output_names=['output_seq'],
        dynamic_axes={
            'input_img': {0: 'batch_size', 3: 'width'},
            'output_seq': {0: 'sequence_len', 1: 'batch_size'}
        }
    )

    # Kwantyzacja dynamiczna do INT8
    quantize_dynamic(
        model_input=crnn_temp_fp32,
        model_output=crnn_final_int8,
        weight_type=QuantType.QUInt8
    )

    # Usunięcie tymczasowego pliku FP32 w celu zaoszczędzenia miejsca
    if os.path.exists(crnn_temp_fp32):
        os.remove(crnn_temp_fp32)

    # Definiujemy ścieżkę tymczasową (FP32) i docelową (INT8)
    caps_temp_fp32 = os.path.join(output_dir, "caps_temp_fp32.onnx")
    caps_final_int8 = os.path.join(output_dir, "caps_user_int8.onnx")

    # Przykładowe tensory
    sample_caps_img = torch.randn(1, 1, 64, 64)
    sample_caps_ctx = torch.randn(1, 1024)

    # Standardowy eksport CapsNet
    torch.onnx.export(
        model_caps,
        (sample_caps_img, sample_caps_ctx),
        caps_temp_fp32,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['img_crop', 'context_vector'],
        output_names=['caps_output'],
        dynamic_axes={
            'img_crop': {0: 'batch_size'},
            'context_vector': {0: 'batch_size'},
            'caps_output': {0: 'batch_size'}
        }
    )

    # Kwantyzacja dynamiczna CapsNet
    quantize_dynamic(
        model_input=caps_temp_fp32,
        model_output=caps_final_int8,
        weight_type=QuantType.QUInt8
    )

    # Usunięcie pliku tymczasowego FP32
    if os.path.exists(caps_temp_fp32):
        os.remove(caps_temp_fp32)

    tqdm.write(f"[{now()}] Sukces: Wygenerowano zoptymalizowane modele INT8 w folderze: {output_dir}")

def run_integrated_pipeline():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USER_DATA_ROOT = "./user_data"
    FINE_COMPLETE_FILE = os.path.join(USER_DATA_ROOT, "fine_tuning_complete.flag")
    CHECKPOINT_DIR = r"C:\OCR\HandwrittenTextRecognition\output_data\checkpoints"
    ONNX_DIR = "./onnx_output"

    CRNN_BASE_WEIGHTS = os.path.join(CHECKPOINT_DIR, "hwr", "best_cer_model.pth")
    JSON_TRANSFORMER_PATH = "user_hard_cases.json"

    if os.path.exists(USER_DATA_ROOT) and os.path.exists(FINE_COMPLETE_FILE):
        tqdm.write(f"[{now()}] ROZPOCZYNAM ZINTEGROWANĄ ADAPTACJĘ BAYESOWSKĄ")

        # Przygotowanie Modelu Wizyjnego
        ckpt = torch.load(CRNN_BASE_WEIGHTS, map_location=DEVICE)
        char_list = ckpt.get("char_list", [])
        encoder = CTCLabelEncoder(char_list)

        model = ResNetCRNN(encoder.vocab_size).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

        # Aktywne uczenie
        al_manager = ActiveLearningManager(user_id="local_user", storage_base=USER_DATA_ROOT)
        bayes_samples = al_manager.get_training_data(model=model, top_k=200, num_passes=10)

        if len(bayes_samples) > 0:
            u_train_ds = UserCorrectionDataset(bayes_samples, get_augmentations("fine_tune"))
            u_train_loader = DataLoader(u_train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn_dynamic)

            # Załadowanie CapsNet
            model_caps = DeepCapsNetCharRecognition.CapsNet(num_classes=len(char_list)).to(DEVICE)
            caps_weights = os.path.join(CHECKPOINT_DIR, "CapsNet_Adapted_User.pth")
            if not os.path.exists(caps_weights):
                caps_weights = os.path.join(CHECKPOINT_DIR, "hcr", "CharLevelCapsNet.pth")

            try:
                model_caps.load_state_dict(torch.load(caps_weights, map_location=DEVICE).get('model_state',
                                                                                             torch.load(caps_weights,
                                                                                                        map_location=DEVICE)),
                                           strict=False)
            except Exception as e:
                tqdm.write(f"[{now()}] Uwaga przy ładowaniu CapsNet: {e}")

            # Joint Fine-Tuning
            joint_tuner = JointFeedbackFineTuner(model, model_caps, None, encoder, DEVICE)
            augmentor = FeedbackAugmentor()
            segmentor = LineToWordSegmentor()

            for batch in tqdm(u_train_loader, desc="Joint Fine-Tuning"):
                images, labels, *rest = batch

                # Generujemy szum, tylko jeśli wymiary się zgadzają
                if images.dim() == 4:
                    aug_imgs = torch.cat([
                        torch.cat([img.unsqueeze(0), augmentor.generate_variations(img.unsqueeze(0), 3)])
                        for img in images
                    ])
                    aug_labels = [lbl for lbl in labels for _ in range(4)]
                    joint_tuner.fine_tune_on_feedback(
                        images_batch=aug_imgs,
                        original_pred_text=labels,  # Tekst przed poprawką (uproszczenie)
                        corrected_text=labels,  # Tekst poprawiony
                        segmentor=segmentor,
                        prev_context=""
                    )


            joint_tuner.finalize_session_learning()

            tqdm.write(f"[{now()}] Generowanie logów dla Transformera i UI.")
            transformer_data = []

            # Musimy przygotować podwójny pipeline do ekstrakcji błędów
            inference_pipe = CRNNInferencePipeline(model, char_list, DEVICE)
            inference_pipe.refiner = CascadeRefinementNetwork(model_caps, char_list, inference_pipe)

            model.eval()
            with torch.no_grad():
                # Iterujemy po surowych ścieżkach do obrazów, aby zastosować poprawny potok "pod CapsNet" i "pod CRNN"
                for img_path, label in bayes_samples:
                    raw_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if raw_img is None: continue

                    # Wymuszenie czarnego na białym dla CRNN
                    if np.mean(raw_img) < 127: raw_img = cv.bitwise_not(raw_img)
                    h, w = raw_img.shape
                    new_w = max(16, int(w * (64.0 / h)))
                    crop_res = cv.resize(raw_img, (new_w, 64), interpolation=cv.INTER_CUBIC)
                    _, crop_bin = cv.threshold(crop_res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                    # Tensor CRNN
                    t_c = (torch.from_numpy(crop_res).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

                    # Tensor CapsNet
                    _, crop_bin = cv.threshold(crop_res, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                    caps_ready = prep_tool.standardize_ink_thickness(cv.bitwise_not(crop_bin), target_thickness=3)
                    t_k = (torch.from_numpy(caps_ready).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

                    # Predykcja na spiętych tensorach
                    res = inference_pipe.predict_batch([t_c], [t_k])

                    if res:
                        r = res[0]
                        transformer_data.append({
                            "metadata": {
                                "timestamp": now(),
                                "confidence_score": float(r.get('confidence', 0.0))
                            },
                            "vision_results": {
                                "step_1_crnn": r.get('final_result', ""),
                                "step_2_capsnet": r.get('final_result', "")
                            },
                            "transformer_results": {
                                "final_choice": r.get('final_result', ""),
                                "top_3_candidates": [r.get('final_result', "")]
                            },
                            "ui_elements": {
                                "uncertain_frames": r.get('mask', []),  # Zwracana maska z _decode_ctc
                                "ground_truth": label
                            }
                        })

            # Zapis do JSON
            with open(JSON_TRANSFORMER_PATH, "w", encoding="utf-8") as f:
                json.dump(transformer_data, f, ensure_ascii=False, indent=4)

            if len(transformer_data) >= 50:
                # Jeśli ten proces ma dostęp do ByT5:
                run_transformer_adaptation(json_data_path=JSON_TRANSFORMER_PATH, model_path="google/byt5-base",
                                           output_path="./byt5_adapted",
                                           matrix_path=os.path.join(CHECKPOINT_DIR, "confusion_matrix.npy"),
                                           encoder=encoder)

            # Eksport końcowy
            export_all_to_onnx(model, model_caps, ONNX_DIR)

            del model, model_caps, joint_tuner
            torch.cuda.empty_cache()
            gc.collect()

            tqdm.write(f"[{now()}] KOMPLETNA ADAPTACJA ZAKOŃCZONA.")
        else:
            tqdm.write(f"[{now()}] Brak nowych danych do adaptacji (Top-K nie znalazło wystarczająco dużo błędów).")

if __name__ == "__main__":
    run_integrated_pipeline()
