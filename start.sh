#!/bin/bash

set -e

echo "🔎 Checking Ollama..."

# 1️⃣ Dacă ollama nu există, îl instalăm
if ! command -v ollama &> /dev/null
then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "🚀 Starting Ollama..."
ollama serve &

sleep 5

# 2️⃣ Dacă modelul nu există, îl creăm
if ! ollama list | grep -q "visi-ro"; then
    echo "⬇️ Creating model visi-ro..."
    ollama create visi-ro -f Modelfile
fi

# echo "🔥 Starting FastAPI..."
# uvicorn app.main:app --host 0.0.0.0 --port 8001