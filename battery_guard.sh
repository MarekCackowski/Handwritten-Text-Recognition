#!/usr/bin/env bash

PROJECT_DIR="$HOME/HCR"
cd "$PROJECT_DIR" || exit 1

# Odświeżenie uprawnień sudo (będą podtrzymywane w pętli co 15s)
sudo -v || exit 1
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
sudo nvidia-smi -pl 45 >/dev/null

# Wczytanie zmiennych środowiskowych z .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

HF_TOKEN_VAL="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
ENV_FLAG=""
[ -f "$PROJECT_DIR/.env" ] && ENV_FLAG="--env-file $PROJECT_DIR/.env"

docker rm -f trening_htr 2>/dev/null || true
docker compose up -d

PAUSED=0
CURRENT_PL=45
LOG_PID=""

cleanup() {
    echo -e "\nZatrzymywanie i przywracanie ustawień fabrycznych (GPU 60W, Turbo ON)..."
    [ -n "$LOG_PID" ] && kill "$LOG_PID" 2>/dev/null
    docker unpause trening_htr 2>/dev/null || true
    docker rm -f trening_htr 2>/dev/null || true
    sudo nvidia-smi -pl 60 >/dev/null
    echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
    exit 0
}
trap cleanup INT TERM

# Uruchomienie kontenera treningowego w tle z pełną konfiguracją środowiska HTR
docker run -d --rm --name trening_htr \
    --gpus all \
    --ipc=host \
    --network hcr_htr-network \
    --shm-size=16gb \
    $ENV_FLAG \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/app \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_TOKEN="$HF_TOKEN_VAL" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL" \
    -e HF_HUB_VERBOSITY=error \
    -e HF_HUB_DISABLE_IMPLICIT_TOKEN=0 \
    -e MONGO_URL="mongodb://htr_mongo:27017" \
    -e REDIS_HOST="htr_redis" \
    -e REDIS_URL="redis://htr_redis:6379/0" \
    -e MINIO_URL="minio:9000" \
    -v "$HOME/.cache/torch:/root/.cache/torch" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PROJECT_DIR:/app" \
    -v "$PROJECT_DIR/Data:/app/data:rw" \
    htr-worker:v6 \
    python3 -B run_all.py >/dev/null

# Podgląd logów treningu na żywo w bieżącym terminalu
docker logs -f trening_htr &
LOG_PID=$!

while [ "$(docker inspect -f '{{.State.Running}}' trening_htr 2>/dev/null)" == "true" ]; do
    # Podtrzymanie sesji sudo, aby nie wygasła podczas wielogodzinnego treningu
    sudo -v 2>/dev/null

    BAT=$(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -n 1)
    STATUS=$(cat /sys/class/power_supply/BAT*/status 2>/dev/null | head -n 1)

    if [[ -n "$BAT" ]]; then
        # Krytyczny poziom (<=15%) -> zamrożenie kontenera i ładowanie do 45%
        if [[ "$PAUSED" -eq 0 && "$BAT" -le 15 && "$STATUS" == "Discharging" ]]; then
            echo -e "\n[BEZPIECZNIK] Bateria ${BAT}% — wstrzymuję kontener do osiągnięcia 45%..."
            docker pause trening_htr >/dev/null
            PAUSED=1
        elif [[ "$PAUSED" -eq 1 && "$BAT" -ge 45 ]]; then
            echo -e "\n[BEZPIECZNIK] Bateria naładowana do ${BAT}% — wznawiam trening!"
            sudo nvidia-smi -pl 40 >/dev/null
            CURRENT_PL=40
            docker unpause trening_htr >/dev/null
            PAUSED=0
        # Ostrzegawczy poziom (<=30%) -> redukcja mocy GPU do 30W
        elif [[ "$PAUSED" -eq 0 && "$BAT" -le 30 && "$STATUS" == "Discharging" && "$CURRENT_PL" -gt 30 ]]; then
            echo -e "\n[BEZPIECZNIK] Bateria ${BAT}% — obniżam limit GPU do 30W."
            sudo nvidia-smi -pl 30 >/dev/null
            CURRENT_PL=30
        elif [[ "$PAUSED" -eq 0 && "$BAT" -ge 60 && "$CURRENT_PL" -lt 45 ]]; then
            echo -e "\n[BEZPIECZNIK] Bateria ${BAT}% — przywracam limit GPU 45W."
            sudo nvidia-smi -pl 45 >/dev/null
            CURRENT_PL=45
        fi
    fi
    sleep 15
done

cleanup