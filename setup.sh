#!/bin/bash
# =============================================================
# VocalAI — RunPod / bare-metal Linux bootstrap
# Run once on a fresh machine after cloning the repo.
# Requires: CUDA GPU, internet access, Ubuntu/Debian base.
# =============================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

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
echo "  NOTE: First pull of gemma-3-27b-it-GGUF:Q4_K_M is ~17 GB — this will take a while."

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

# Pre-download Whisper model weights (avoids cold-start delay on first request)
echo "  Pre-downloading Whisper medium model weights (CPU, weights-only)..."
python3 - <<'PYEOF'
# Download weights only — no GPU needed, avoids CUDA fork issues
from faster_whisper import WhisperModel
WhisperModel("medium", device="cpu", compute_type="int8")
print("  Whisper weights cached.")
PYEOF

# Pull the model directly
echo "  Pulling gemma-3-27b-it-GGUF:Q4_K_M (downloads ~17 GB on first run)..."
ollama pull hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M
echo "  Ollama model ready."

# -----------------------------------------------------------
# 6. Start FastAPI server
# -----------------------------------------------------------
# Warm up the CUDA driver so Whisper doesn't hit "unknown error" on first init
echo "  Warming up CUDA driver..."
nvidia-smi > /dev/null 2>&1 || true
sleep 2

echo "[6/6] Starting VocalAI server on port 8000..."
echo ""
echo "============================================"
echo "  Server starting at http://0.0.0.0:8000"
echo "  POST /voice  — WAV in, MP3 out"
echo "  Set GOOGLE_CALENDAR_ID env var and place"
echo "  credentials.json in the project root."
echo "============================================"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000
