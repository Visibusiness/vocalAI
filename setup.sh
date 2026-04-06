#!/bin/bash
# =============================================================
# VocalAI — RunPod / bare-metal Linux bootstrap
# Run once on a fresh machine after cloning the repo.
# Requires: CUDA GPU, internet access, Ubuntu/Debian base.
# =============================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# -----------------------------------------------------------
# Required env vars (can be pre-set or passed at runtime)
# e.g.: GOOGLE_CALENDAR_ID=".." TWILIO_ACCOUNT_SID=".." TWILIO_AUTH_TOKEN=".." ./setup.sh
# -----------------------------------------------------------
: "${GOOGLE_CALENDAR_ID:?ERROR: GOOGLE_CALENDAR_ID is not set. Export it before running setup.sh}"
: "${TWILIO_ACCOUNT_SID:?ERROR: TWILIO_ACCOUNT_SID is not set. Export it before running setup.sh}"
: "${TWILIO_AUTH_TOKEN:?ERROR: TWILIO_AUTH_TOKEN is not set. Export it before running setup.sh}"

echo ""
echo "============================================"
echo "  VocalAI — Bootstrap"
echo "============================================"
echo ""

# -----------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------
echo "[1/6] Installing system packages..."
apt-get update -y -q
apt-get install -y -q curl git python3-pip python3-dev redis-server ffmpeg lsof zstd pciutils

# -----------------------------------------------------------
# 2. Ollama
# -----------------------------------------------------------
echo "[2/6] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  Ollama already installed, skipping."
fi

# -----------------------------------------------------------
# 3. Python dependencies
# -----------------------------------------------------------
echo "[3/6] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# -----------------------------------------------------------
# 4. Start Redis (daemonized)
# -----------------------------------------------------------
echo "[4/6] Starting Redis..."
if redis-cli ping &>/dev/null 2>&1; then
    echo "  Redis already running."
else
    redis-server --daemonize yes --loglevel warning
    sleep 1
    redis-cli ping > /dev/null
    echo "  Redis started."
fi

# -----------------------------------------------------------
# 5. Start Ollama + pull model
# -----------------------------------------------------------
echo "[5/6] Starting Ollama and pulling model..."
echo "  NOTE: First pull of gemma4:26b is ~17 GB — this will take a while."

# Start ollama serve in background if not already running
if ! pgrep -x ollama &>/dev/null; then
    ollama serve > /var/log/ollama.log 2>&1 &
    echo "  Waiting for Ollama to be ready..."
    for i in $(seq 1 30); do
        if ollama list &>/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
else
    echo "  Ollama already running."
fi

# Convert Romanian fine-tuned Whisper to CTranslate2 format (required by faster-whisper)
# The model was saved with torch.compile() which adds _orig_mod.model. prefix to all
# weight keys — we strip those before converting.
WHISPER_RO_PATH="/workspace/whisper-ro-turbo"
WHISPER_RO_FIXED="/tmp/whisper-ro-fixed"
if [ ! -d "$WHISPER_RO_PATH" ]; then
    echo "  Downloading and fixing IonGrozea/whisper-large-v3-ro-turbo weight keys..."
    pip install -q ctranslate2 transformers safetensors
    python3 - <<'PYEOF'
import os, torch
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperFeatureExtractor

model_id = "IonGrozea/whisper-large-v3-ro-turbo"
fixed_dir = "/tmp/whisper-ro-fixed"

print("  Loading state dict from HuggingFace...")
try:
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_id, "model.safetensors")
    state_dict = safetensors.torch.load_file(path)
except Exception:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_id, "pytorch_model.bin")
    state_dict = torch.load(path, map_location="cpu")

print("  Stripping _orig_mod.model. prefix from weight keys...")
prefix = "_orig_mod.model."
fixed = {}
for k, v in state_dict.items():
    fixed[k[len(prefix):] if k.startswith(prefix) else k] = v

print("  Loading into fresh WhisperForConditionalGeneration...")
config = WhisperConfig.from_pretrained(model_id)
model = WhisperForConditionalGeneration(config)
model.load_state_dict(fixed, strict=True)
model.save_pretrained(fixed_dir)

fe = WhisperFeatureExtractor.from_pretrained(model_id)
fe.save_pretrained(fixed_dir)
print(f"  Fixed model saved to {fixed_dir}")
PYEOF

    echo "  Converting fixed model to CTranslate2 format..."
    ct2-transformers-converter \
        --model "$WHISPER_RO_FIXED" \
        --output_dir "$WHISPER_RO_PATH" \
        --quantization float16 \
        --force
    cp "$WHISPER_RO_FIXED/preprocessor_config.json" "$WHISPER_RO_PATH/"
    rm -rf "$WHISPER_RO_FIXED"
    echo "  Romanian Whisper model ready at $WHISPER_RO_PATH"
else
    echo "  Romanian Whisper model already converted, skipping."
fi

# Pull the model directly
echo "  Pulling gemma4:26b (downloads ~17 GB on first run)..."
ollama pull gemma4:26b
echo "  Ollama model ready."

# -----------------------------------------------------------
# 6. Start FastAPI server
# -----------------------------------------------------------
# Add nvidia pip package lib paths to LD_LIBRARY_PATH so ctranslate2 can find libcudart
NVIDIA_LIBS=$(python3 -c "
import os, sys
base = os.path.join(sys.prefix, 'lib', 'python' + sys.version[:4], 'dist-packages', 'nvidia')
if not os.path.isdir(base):
    base = '/usr/local/lib/python3.11/dist-packages/nvidia'
paths = []
if os.path.isdir(base):
    for pkg in os.listdir(base):
        lib = os.path.join(base, pkg, 'lib')
        if os.path.isdir(lib):
            paths.append(lib)
print(':'.join(paths))
" 2>/dev/null)
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
    echo "  CUDA lib paths: $NVIDIA_LIBS"
fi

# Empty CUDA_VISIBLE_DEVICES hides all GPUs — unset it so CUDA can see the GPU
unset CUDA_VISIBLE_DEVICES

echo "[6/6] Starting VocalAI server on port 8000..."
echo ""
echo "============================================"
echo "  Server starting at http://0.0.0.0:8000"
echo "  POST /voice        — WAV in, MP3 out"
echo "  POST /twilio/incoming — Twilio webhook"
echo "============================================"
echo ""

GOOGLE_CALENDAR_ID="$GOOGLE_CALENDAR_ID" \
TWILIO_ACCOUNT_SID="$TWILIO_ACCOUNT_SID" \
TWILIO_AUTH_TOKEN="$TWILIO_AUTH_TOKEN" \
uvicorn app.main:app --host 0.0.0.0 --port 8000
