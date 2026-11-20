# Cascaded Handwritten Text Recognition System

An advanced Handwritten Text Recognition system designed for recognizing and correcting difficult handwriting, including Polish with full diacritic support. The system utilizes a hybrid approach combining computer vision, geometric character verification, and semantic linguistic analysis. The project is optimized for training and deployment on a local machine and utilizes MLflow for experiment tracking and logging.

## System Architecture

The system processes images through a cascaded architecture, which is divided into the following integrated modules:

### Sequential Module: CRNN - Vision Baseline

Responsible for the baseline visual recognition of word images. Utilizes a ResNet-18 backbone optimized to preserve the vertical proportions of characters. Features a built-in geometric rectification network, ScRN_STN, that straightens skewed text, alongside a Visual Attention module with a BiLSTM network. Outputs initial predictions along with a probability map featuring Shannon entropy and a 1024-dimensional deep context vector.

### Capsule Network: CapsNet - Ambiguity Verification

Designed to analyze uncertain regions, so-called "Dubious Regions", detected by the sequential module. Verifies edge cases, ligatures, and diacritic misclassifications. Employs a Deep Fusion mechanism combining the raw image crop with the 1024-dimensional context vector derived from the CRNN. Incorporates reconstruction algorithms like GVFBalloonFilter and SSIM Loss to handle degraded ink.

### Language Module & Semantic Correction: Transformer

Utilizes a ByT5 model at the byte or character level fine-tuned with LoRA adapters for final semantic correction. Receives text from the vision and capsule models containing special tokens, such as the `<unc>` uncertainty tag, and reconstructs the correct word based on sentence context. Operates bilingually supporting `<pl>` and `<en>` tags, correcting both OCR errors and analyzing the probability of a word occurring within the broader context of the text line.

### Results Analysis, Experiments, & Integration

The final module aggregates metrics like CER and WER, visualizes XAI attention maps, and integrates the individual cascades into a single, seamless prediction pipeline.

## Key Features & User Interface

* **Dynamic Model Fusion:** The system analyzes the confidence of the CRNN output. If a word is deemed suspicious, the system crops the difficult characters and routes them to the CapsNet. If the vision hybrid is still uncertain, the Transformer takes over.

* **Interactive User Interface:** Within the graphical application, the system highlights all uncertain fragments, whether entire words or sub-parts, with bounding boxes. By hovering the cursor over a selected word, the user can trace the algorithm's entire decision-making process via a tooltip showing: `crnn result:` for raw sequential prediction, `crnn+capsnet result:` for the result after CapsNet verification, and `final result:` for the ultimate output after Transformer correction.

* **Explainable AI:** Employs Integrated Gradients to generate heatmaps, visually explaining which pixels the CapsNet focused on when making its decision.

* **Advanced Augmentations & Curriculum Learning:** The module can dynamically increase the difficulty of the learning process by simulating degraded paper, creases, ink blots, or segmentation errors like cutting through ligatures.

## Environment and Execution

The project is configured to run directly on a local machine in an environment with a CUDA-enabled GPU.

**Access to the Service and Metrics:**
The public IP address of the virtual machine `vm-app` for communicating with the deployed system and the MLflow experiment tracking interface running on default port 5000.

### Training the Modules

To train the complete pipeline sequentially, execute the orchestrator script:

```bash
PROJECT_DIR="$HOME/HCR"
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

HF_TOKEN_VAL="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
ENV_FLAG=""
[ -f "$PROJECT_DIR/.env" ] && ENV_FLAG="--env-file $PROJECT_DIR/.env"

docker rm -f trening_htr mlflow_ui 2>/dev/null || true
docker compose up -d

gnome-terminal --title="Monitor GPU & Docker" -- bash -c "watch -n 1 'nvidia-smi; echo \"\"; docker ps --format \"table {{.Names}}\t{{.Status}}\"'; exec bash" &
gnome-terminal --title="MLflow UI" -- bash -c "docker run --rm --name mlflow_ui --network hcr_htr-network -p 5000:5000 $ENV_FLAG -v \"$PROJECT_DIR:/app\" -w /app htr-worker:v6 mlflow ui --host 0.0.0.0 --port 5000; exec bash" &

docker run --rm --name trening_htr \
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
    python3 -B run_all.py
```

This script handles the verification and execution of each training phase step by step:

1. **CRNN Training:** `ResNetCRNNWordRecognition.py`
2. **CapsNet Training:** `DeepCapsNetCharRecognition.py`
3. **Transformer Training:** `TransformerDictionaryRefinement.py`

Individual scripts can also be run independently if a specific stage needs retraining.

## Technologies Used

* **PyTorch:** Models, Tensor operations, AMP

* **OpenCV, Albumentations:** Image processing, Deskewing, Augmentations

* **HuggingFace Transformers, PEFT/LoRA:** ByT5, LLM Optimization

* **MLflow:** Model lifecycle management and metric logging

* **Celery, Redis:** Asynchronous task queueing for HTR
