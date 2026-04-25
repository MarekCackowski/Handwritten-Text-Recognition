import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 as cv

IMAGE_SIZE_CHAR = (64, 64)
EMNIST_NORM = ((0.1307,), (0.3081,))

def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 2 kanały: MaxPool i AvgPool
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv(res)
        return self.sigmoid(res)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels=256, out_channels=32, dim_caps=8, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels * dim_caps, kernel_size=kernel_size, stride=stride)
        self.dim_caps = dim_caps

    def forward(self, x):
        outputs = self.conv2d(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


class DigitCaps(nn.Module):
    def __init__(self, num_capsules, num_routes=32 * 12 * 12, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        # Inicjalizacja wag W
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)

    def forward(self, x):
        batch_size = x.size(0)
        # Rozszerzenie x do mnożenia: [B, R, 1, In, 1]
        x = x[:, :, None, :, None]
        u_hat = torch.matmul(self.W, x)

        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)
        v_j = None
        for i in range(2):
            c_ij = func.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(3) * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, axis=-2)
            if i < 1:
                # Rozszerzenie v_j do uaktualnienia b_ij
                v_j_expanded = v_j.expand(batch_size, self.num_routes, self.num_capsules, 16, 1)
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j_expanded)
                b_ij = b_ij + a_ij.squeeze(4)
        return v_j.squeeze(1)


class CapsNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes

        # Backbone
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=1)

        # Atencja
        self.attention = SpatialAttention(kernel_size=7)

        # Kapsułki (siatka 12x12)
        self.primary_caps = PrimaryCaps(in_channels=256, out_channels=32, dim_caps=8)
        self.digit_caps = DigitCaps(num_capsules=num_classes, num_routes=32 * 12 * 12)

        # Decoder 4096 (dla 64x64)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 4096), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Maska atencji
        attn_map = self.attention(x)
        x = x * attn_map

        x = self.primary_caps(x)
        x = self.digit_caps(x)
        x = x.squeeze(-1)
        classes_probs = (x ** 2).sum(dim=2) ** 0.5
        return classes_probs


class CapsNetPredictor:
    """ Klasa pomocnicza do ładowania modelu i predykcji pojedynczych znaków. """
    def __init__(self, checkpoint_path, encoder, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder

        # Inicjalizacja modelu z poprawną liczbą klas
        self.model = CapsNet(num_classes=encoder.get_num_classes()).to(self.device)
        self.model.eval()

        try:
            print(f"[CapsNet] Loading weights from {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location=self.device)
            
            # Strict = False pozwala pominąć ewentualne niezgodności w buforach, ale wagi muszą pasować
            self.model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[CapsNet] Error loading weights: {e}")

        # Transformacja dokładnie taka sama jak przy walidacji/treningu
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_CHAR),
            transforms.ToTensor(),
            transforms.Normalize(*EMNIST_NORM)
        ])

    def predict_char(self, img_input):
        if isinstance(img_input, str):
            img = cv.imread(img_input, cv.IMREAD_GRAYSCALE)
        else:
            img = img_input

        if img is None: return "?", 0.0

        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model(img_tensor)
            conf, idx = torch.max(probs, dim=1)

            char_idx = idx.to(torch.long).item()
            char = self.encoder.decode(char_idx)

            return char, conf.item()
