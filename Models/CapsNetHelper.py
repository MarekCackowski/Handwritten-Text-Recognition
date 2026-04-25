import os
import warnings
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from PIL import Image
from torchvision import transforms
from DeepCapsNetCharRecognition import CapsNet
from ResNetCRNNWordRecognition import ResNetCRNN
warnings.filterwarnings("ignore")

IMAGE_SIZE_CHAR = (64, 64)
IMAGE_HEIGHT_WORD = 64
IMAGE_WIDTH_WORD = 512
EMNIST_NORM = ((0.1307,), (0.3081,))
MAX_TIMESTEPS = 127  # Szerokość 512 / stride 4 minus padding


class CharLabelEncoder:
    """ Enkoder dla CapsNet. Wczytuje alfabet z pliku modelu. """

    def __init__(self, char_list=None):
        if char_list is None:
            self.idx_to_char = {}
            self.char_to_idx = {}
        else:
            self.idx_to_char = {i: c for i, c in enumerate(char_list)}
            self.char_to_idx = {c: i for i, c in enumerate(char_list)}

    def decode(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.idx_to_char.get(int(idx), '?')

    def encode(self, char):
        return self.char_to_idx.get(char, 0)

    def get_num_classes(self):
        return len(self.idx_to_char)


class CapsNetPredictor:
    """ Wrapper na model CapsNet z obsługą Deep Fusion (Kontekst). """

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {checkpoint_path}")

        print(f"Ładowanie CapsNet: {os.path.basename(checkpoint_path)}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        char_list = ckpt.get('char_list', [])
        self.encoder = CharLabelEncoder(char_list)

        # Inicjalizacja modelu z obsługą kontekstu (wymiar 512 z LSTM)
        num_classes = self.encoder.get_num_classes()
        self.model = CapsNet(num_classes=num_classes, context_dim=512).to(self.device)

        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CHAR),
            transforms.ToTensor(),
            transforms.Normalize(*EMNIST_NORM)
        ])

    def predict_char(self, img_input, context_vector=None):
        """ Przyjmuje obraz oraz opcjonalny wektor kontekstu z CRNN.
        Zwraca: (znak, pewność) """
        if img_input is None or img_input.size == 0:
            return "?", 0.0

        # Obsługa wejścia
        if isinstance(img_input, str):
            if os.path.exists(img_input):
                img = cv.imread(img_input, cv.IMREAD_GRAYSCALE)
            else:
                return "?", 0.0
        else:
            img = img_input

        if img is None: return "?", 0.0

        # Transformacja obrazu
        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Obsługa kontekstu
        if context_vector is not None:
            if not torch.is_tensor(context_vector):
                context_vector = torch.tensor(context_vector)
            context_vector = context_vector.to(self.device)
            if context_vector.dim() == 1:
                context_vector = context_vector.unsqueeze(0)

        with torch.no_grad():
            # Przekazujemy word_context do modelu CapsNet!
            probs, _, _, _ = self.model(img_tensor, word_context=context_vector)

            conf, idx = torch.max(probs, dim=1)
            char = self.encoder.decode(idx.item())

            return char, conf.item()


class HTRCharEncoder:
    """ Enkoder CRNN z obliczaniem entropii. """

    def __init__(self, char_list):
        self.char_list = sorted(list(set(char_list)))
        self.num_to_char = {i + 1: c for i, c in enumerate(self.char_list)}
        self.num_to_char[0] = ''

    def decode_detailed(self, log_probs_tensor):
        probs = torch.exp(log_probs_tensor)

        # Entropia Shannona jako miara niepewności
        entropy_tensor = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

        max_probs, indices = torch.max(probs, dim=1)

        indices = indices.tolist()
        max_probs = max_probs.tolist()
        entropies = entropy_tensor.tolist()

        result = []
        prev_idx = -1

        for t, idx in enumerate(indices):
            if idx != 0 and idx != prev_idx:
                char = self.num_to_char.get(idx, '?')
                result.append({
                    'char': char,
                    'conf': max_probs[t],
                    'entropy': entropies[t],
                    'timestep': t
                })
            prev_idx = idx

        return result


class HybridRefiner:
    """ Główny Integrator (Orkiestrator).
        Zarządza przepływem: Obraz -> CRNN -> (Kontekst + Wycinek) -> CapsNet -> Wynik. """

    def __init__(self, crnn_checkpoint_path, caps_checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gap_threshold = 12
        self.entropy_threshold = 0.6

        if not os.path.exists(crnn_checkpoint_path):
            raise FileNotFoundError(f"Brak modelu CRNN: {crnn_checkpoint_path}")

        print(f"Wczytywanie CRNN: {os.path.basename(crnn_checkpoint_path)}")
        crnn_ckpt = torch.load(crnn_checkpoint_path, map_location=self.device)
        self.crnn_char_list = crnn_ckpt.get('char_list')
        self.word_encoder = HTRCharEncoder(self.crnn_char_list)

        self.crnn = ResNetCRNN(len(self.crnn_char_list) + 1).to(self.device)
        state_dict = crnn_ckpt['model_state'] if 'model_state' in crnn_ckpt else crnn_ckpt
        self.crnn.load_state_dict(state_dict, strict=False)
        self.crnn.eval()

        print(f"Wczytywanie CapsNet: {os.path.basename(caps_checkpoint_path)}")
        self.caps_predictor = CapsNetPredictor(caps_checkpoint_path, device=self.device)

        self.word_transform = transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT_WORD, IMAGE_WIDTH_WORD)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def recognize_word_hybrid(self, word_image_np):
        if word_image_np is None: return ""

        # Preprocessing
        processed_img = cv.resize(word_image_np, (IMAGE_WIDTH_WORD, IMAGE_HEIGHT_WORD))
        pil_img = Image.fromarray(processed_img)
        word_tensor = self.word_transform(pil_img).unsqueeze(0).to(self.device)

        # Inferencja CRNN
        with torch.no_grad():
            preds, context_map = self.crnn(word_tensor, return_context=True)
            preds = preds.squeeze(1)
            context_map = context_map.squeeze(0)

        decoded_data = self.word_encoder.decode_detailed(preds)

        final_text_builder = []
        prev_timestep = 0

        def get_context_vec(t):
            t = min(max(0, t), context_map.size(0) - 1)
            return context_map[t]

        for item in decoded_data:
            char = item['char']
            conf = item['conf']
            entropy = item['entropy']
            curr_timestep = item['timestep']

            # Wypełnianie luk
            if (curr_timestep - prev_timestep) > self.gap_threshold:
                gap_t = (curr_timestep + prev_timestep) // 2
                gap_crop = self.get_dynamic_crop(processed_img, gap_t)

                if np.mean(gap_crop) > 20:
                    p_char, p_conf = self.caps_predictor.predict_char(gap_crop, context_vector=None)
                    if p_conf > 0.90 and p_char in ".,-':;\"!ilI":
                        final_text_builder.append(p_char)

            if entropy < self.entropy_threshold:
                # CRNN pewny
                final_text_builder.append(char)
            else:
                # CRNN niepewny
                crop = self.get_dynamic_crop(processed_img, curr_timestep)
                vec = get_context_vec(curr_timestep)  # Pobieramy też myśl
                alt_char, alt_conf = self.caps_predictor.predict_char(crop, context_vector=vec)

                # Logika podejmowania decyzji
                if alt_char == char:
                    final_text_builder.append(char)
                elif alt_conf > 0.8:
                    final_text_builder.append(alt_char)
                else:
                    # CapsNet ma lekki bonus za specjalizację
                    if (alt_conf + 0.15) > conf:
                        final_text_builder.append(alt_char)
                    else:
                        final_text_builder.append(char)

            prev_timestep = curr_timestep

        # Sprawdzenie końcówki słowa
        if (MAX_TIMESTEPS - prev_timestep) > self.gap_threshold:
            end_t = (prev_timestep + MAX_TIMESTEPS) // 2
            end_crop = self.get_dynamic_crop(processed_img, end_t)
            if np.mean(end_crop) > 20:
                e_char, e_conf = self.caps_predictor.predict_char(end_crop)
                if e_conf > 0.85 and e_char in ".,?!":
                    final_text_builder.append(e_char)

        return "".join(final_text_builder)

    @staticmethod
    def get_dynamic_crop(word_img, timestep):
        center_x = int((timestep / MAX_TIMESTEPS) * word_img.shape[1])
        crop_target_size = word_img.shape[0]
        half_size = crop_target_size // 2

        x1 = int(max(0, center_x - half_size))
        x2 = int(min(word_img.shape[1], center_x + half_size))

        crop = word_img[:, x1:x2]
        h, w = crop.shape
        if w < crop_target_size:
            pad_needed = crop_target_size - w
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            crop = cv.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv.BORDER_CONSTANT, value=0)

        crop = cv.resize(crop, (crop_target_size, crop_target_size))
        return crop
