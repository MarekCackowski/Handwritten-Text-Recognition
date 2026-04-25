FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Najpierw ciężkie paczki, potem reszta
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Kopiujemy requirements.txt i instalujemy wszystkie pozostałe
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Kopiujemy resztę kodu
COPY . .

# Konfiguracja środowiska
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python3", "Models/ResNetCRNNWordRecognition.py"]