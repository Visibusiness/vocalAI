#!/bin/bash

# Oprim execuția dacă o comandă importantă dă eroare
set -e

echo "🔄 1. Actualizăm sistemul (apt update & upgrade)..."
apt-get update -y
apt-get upgrade -y

echo "📦 2. Instalăm pachetele de sistem necesare..."
# Ne asigurăm că avem Redis, curl și ffmpeg (pentru Whisper/audio)
apt-get install -y redis-server curl ffmpeg python3-pip

echo "🗄️ 3. Pornim baza de date Redis în fundal..."
redis-server --daemonize yes

echo "🐍 4. Instalăm pachetele Python din requirements.txt..."
# Ne asigurăm că ai redis și python-multipart în requirements.txt
pip install --no-cache-dir -r requirements.txt

echo "🔎 5. Verificăm și instalăm Ollama..."
if ! command -v ollama &> /dev/null
then
    echo "⬇️ Ollama nu există. Se instalează acum..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "🚀 6. Pornim Ollama în fundal..."
ollama serve &

# Îi dăm 5 secunde să deschidă portul înainte să îi cerem să facă modelul
echo "⏳ Așteptăm inițializarea Ollama..."
sleep 5

echo "🧠 7. Verificăm modelul AI..."
if ! ollama list | grep -q "visi-ro"; then
    echo "⚙️ Modelul visi-ro lipsește. Îl construim din Modelfile..."
    ollama create visi-ro -f Modelfile
fi

echo "🔥 8. Totul e gata! Pornim FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8001