# VocalAI — Voice Receptionist

A Romanian-language AI receptionist for a clinic. Patients speak into a microphone, the AI understands them, holds a conversation, and books appointments directly into Google Calendar.

## How it works

```
Patient speaks (WAV)
       ↓
Whisper (speech-to-text)
       ↓
Redis (conversation memory)
       ↓
Ollama / Gemma-3-27B (understands + replies in Romanian)
       ↓
Google Calendar (books appointment if confirmed)
       ↓
Edge-TTS (text-to-speech → MP3)
       ↓
Patient hears the response
```

The assistant collects the patient's **name**, **date**, and **time** through natural conversation. When the patient confirms, it creates a 1-hour event in Google Calendar automatically.

---

## Running the server

**Full bootstrap on a fresh Linux server (RunPod / bare metal):**

```bash
./setup.sh
```

Then start with your calendar ID:

```bash
GOOGLE_CALENDAR_ID="your-email@gmail.com" uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Requirements:**
- NVIDIA GPU with CUDA
- Redis running on `localhost:6379`
- Ollama running on `localhost:11434` with the Gemma-3 model pulled
- Internet access (Edge-TTS calls Microsoft servers)

---

## Testing from your laptop

```bash
pip install sounddevice soundfile httpx pydub
sudo apt install portaudio19-dev ffmpeg

python client.py          # records 6s of mic audio, sends to server, plays response
python client.py --new    # start a fresh conversation session
```

Update `SERVER_URL` in `client.py` to point to your running server.

---

## Google Calendar setup

See the **Google Calendar Setup** section in `CLAUDE.md` for full steps.

Short version:
1. Create a **service account** in Google Cloud Console → download the JSON key
2. Share your Google Calendar with the service account email (`Make changes to events`)
3. Copy the JSON key to the server as `credentials.json`
4. Start the server with `GOOGLE_CALENDAR_ID="your-email@gmail.com"`

Steps 1 and 2 are one-time. Only steps 3 and 4 repeat on a new server.

---

## Key files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI server — single `/voice` endpoint |
| `app/appointment_parser.py` | Extracts booking data from LLM reply |
| `app/calendar_service.py` | Creates Google Calendar events |
| `client.py` | Local test client (mic → server → speakers) |
| `setup.sh` | Full server bootstrap script |
| `credentials.json` | Service account key (not committed to git) |

---

## Customization

| What | Where |
|---|---|
| AI name / behavior | `SYSTEM_PROMPT` in `app/main.py` |
| TTS voice | `"ro-RO-AlinaNeural"` in `app/main.py` |
| LLM model | Model name string in `app/main.py` + `ollama pull <model>` |
| Calendar ID | `GOOGLE_CALENDAR_ID` env var at server start |
