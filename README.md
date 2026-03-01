# 🎙️ VocalAI – STT + LLM + TTS Server

Acest proiect rulează un server vocal AI folosind:

* 🎤 **Faster-Whisper** (STT)
* 🧠 **Ollama + RoLlama3.1-8b (GGUF)** (LLM)
* 🔊 **edge-tts** (TTS)
* 🚀 **FastAPI** (HTTP API)

---

# 📦 VARIANTA 1 – Rulare cu Docker (Recomandat pentru producție)

## 🔧 Cerințe

* Docker
* NVIDIA GPU (opțional, dar recomandat)
* NVIDIA Container Toolkit (pentru GPU)

---

## 🚀 1. Build imaginea

```bash
docker build -t vocalai .
```

---

## 🚀 2. Rulează containerul

```bash
docker run --gpus all \
  -v ollama_data:/root/.ollama \
  -p 8001:8001 \
  vocalai
```

### 🔹 Explicații:

* `--gpus all` → activează GPU
* `-v ollama_data:/root/.ollama` → salvează modelul persistent
* `-p 8001:8001` → expune serverul

---

## 🌐 Acces server

```
http://localhost:8001/docs
```

---

# ☁️ VARIANTA 2 – Rulare direct în RunPod (Test Rapid)

## 🔧 Cerințe

* Pod cu GPU
* HTTP Port expus (ex: 8010)

---

## 🚀 1. Clone repo

```bash
git clone <repo_url>
cd vocalAI
git checkout <docker_branch>
```

---

## 🚀 2. Instalează dependențe

```bash
apt update
apt install -y python3-pip ffmpeg curl
pip3 install -r requirements.txt
```

---

## 🚀 3. Rulează serverul

```bash
chmod +x start.sh
./start.sh
```

Scriptul face automat:

* ✔ Instalează Ollama dacă nu există
* ✔ Pornește Ollama
* ✔ Creează modelul `visi-ro` dacă nu există
* ✔ Pornește FastAPI

---

## 🌐 Acces server

Din UI RunPod → HTTP Service → portul ales (ex: 8010)

Exemplu:

```
https://xxxxx-8010.proxy.runpod.net/docs
```

---

# 🎤 Endpoint

```
POST /voice
```

Trimite fișier `.wav` și primești răspuns `.mp3`.

---

# 📁 Structura proiectului

```
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
```

---

# ⚙️ Model utilizat

```
mradermacher/RoLlama3.1-8b-Instruct-DPO-GGUF
Quant: Q4_K_M
```

Modelul este descărcat automat de Ollama la prima rulare.

---

# 🧠 Note importante

* Prima pornire va descărca modelul (~5GB).
* Folosiți Volume Disk în RunPod pentru persistență.
* Serverul trebuie rulat pe `0.0.0.0`.

---

# 🔥 Debug rapid

Verificare GPU:

```bash
nvidia-smi
```

Verificare model Ollama:

```bash
ollama list
```

---

# 🏁 Status

Server ready când vezi:

```
Application startup complete.
Uvicorn running on http://0.0.0.0:XXXX
```

---

Made with ❤️ for rapid AI prototyping.
