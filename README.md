# 🎙️ VocalAI – STT + LLM + TTS Server

Server vocal AI construit cu:

- 🎤 Faster-Whisper (Speech-to-Text)
- 🧠 Ollama + RoLlama3.1-8b GGUF (LLM)
- 🔊 edge-tts (Text-to-Speech)
- 🚀 FastAPI (API HTTP)

---

# ⚠️ IMPORTANT (RunPod & Port)

RunPod rămâne blocat în `Initializing` dacă aplicația NU ascultă pe portul expus.

Serverul TREBUIE pornit pe:

0.0.0.0

NU pe:

127.0.0.1

FastAPI este pornit corect prin:

uvicorn app.main:app --host 0.0.0.0 --port 8001

---

# 📦 VARIANTA 1 – Docker (recomandat)

## Cerințe

- Docker
- GPU NVIDIA (opțional dar recomandat)
- NVIDIA Container Toolkit (pentru GPU)

---

## 1️⃣ Build imagine

docker build -t vocalai .

---

## 2️⃣ Rulează containerul

docker run --gpus all \
  -v ollama_data:/root/.ollama \
  -p 8001:8001 \
  vocalai

Ce fac opțiunile:

- --gpus all → activează GPU
- -v ollama_data:/root/.ollama → persistă modelul (NU îl mai descarcă la fiecare restart)
- -p 8001:8001 → expune serverul

---

## 🌐 Acces API

http://localhost:8001/docs

---

# ☁️ VARIANTA 2 – RunPod (test rapid)

## Cerințe

- Pod cu GPU
- HTTP Port expus (ex: 8001)
- Recomandat: Volume Disk activat (pentru persistență model)

---

## 1️⃣ Clone repo

git clone <repo_url>
cd vocalAI

---

## 2️⃣ Instalare dependențe

apt update
apt install -y python3-pip ffmpeg curl
pip3 install -r requirements.txt

---

## 3️⃣ Pornește serverul

chmod +x start.sh
./start.sh

---

# 🔄 Ce face start.sh automat

Scriptul:

- ✔ Instalează Ollama (dacă nu există)
- ✔ Pornește Ollama
- ✔ Creează modelul `visi-ro` din `Modelfile`
- ✔ Pornește FastAPI pe `0.0.0.0`

Am adăugat în script:

ollama create visi-ro -f Modelfile

Asta înseamnă că identitatea modelului (System Prompt-ul) este aplicată automat la fiecare Pod nou.
Nu mai trebuie făcut nimic manual.

---

# 🧠 REPARAȚIE MODELFILE (IMPORTANT)

Versiunea veche avea template de tip „Safety Assessment” (Llama-Guard).

Problema:
Modelul răspundea cu:
safe
unsafe

în loc să vorbească normal.

Am rescris TEMPLATE-ul în `Modelfile` folosind format standard Chat:

- system
- user
- assistant

Am adăugat și STOP TOKENS pentru Llama 3.1.

Acum modelul răspunde normal, ca recepționeră.

---

# 📁 Structura proiectului

vocalAI/
│
├── app/
│   └── main.py
│
├── requirements.txt
├── Dockerfile
├── Modelfile
├── start.sh
└── .dockerignore

IMPORTANT:

main.py este în folderul app/.

De aceea comanda corectă este:

uvicorn app.main:app

NU:

uvicorn main:app

---

# 💾 Persistență model (foarte important)

Modelul are ~5GB.

Fără volum persistent:
→ se descarcă la fiecare restart.

Docker:
-v ollama_data:/root/.ollama

RunPod:
Activează Volume Disk.

---

# ⚙️ Model folosit

mradermacher/RoLlama3.1-8b-Instruct-DPO-GGUF
Quantizare: Q4_K_M

---

# 🖥️ Optimizare GPU

Testat pe RTX 2000 Ada:

- Model Q4_K_M ocupă ~5GB VRAM
- Încap și Whisper + modelul LLM simultan
- Rulează fără probleme

---

# 🎤 Endpoint principal

POST /voice

Trimite fișier .wav
Primești răspuns .mp3

---

# 🔍 Debug rapid

Verificare GPU:
nvidia-smi

Verificare modele Ollama:
ollama list

---

# 🏁 Server pornit corect când vezi:

Application startup complete.
Uvicorn running on http://0.0.0.0:8001
