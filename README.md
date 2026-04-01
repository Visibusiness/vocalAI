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

The assistant collects the patient's **name**, **preferred doctor**, **date**, and **time** through natural conversation. When the patient confirms, it creates a 1-hour event in Google Calendar automatically. It also handles conflict detection, cancellations, appointment listings, and respects clinic business hours.

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

**Text-based test client (no microphone needed):**

```bash
python test_client.py --text "Vreau o programare la Dr. Ionescu pe 5 mai la 10:00"
python test_client.py --text "Ce programări am?" --phone +40721000000
python test_client.py --new --text "Bună ziua"   # fresh session
```

**Seed the calendar with demo appointments:**

```bash
GOOGLE_CALENDAR_ID="your-email@gmail.com" python demo_seed.py
```

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
| `app/main.py` | FastAPI server — single `/voice` endpoint, streaming pipeline |
| `app/appointment_parser.py` | Extracts booking JSON from LLM reply |
| `app/calendar_service.py` | Google Calendar CRUD — DST-aware, anonymizes other patients |
| `client.py` | Mic-based local test client (records → server → plays) |
| `test_client.py` | Text-based test client (no mic needed, uses Edge-TTS for input) |
| `demo_seed.py` | Seeds calendar with demo appointments before a presentation |
| `setup.sh` | Full server bootstrap script |
| `credentials.json` | Service account key (not committed to git) |

---

## Customization

| What | Where |
|---|---|
| AI name / behavior | `build_system_prompt()` in `app/main.py` |
| Doctors list | `build_system_prompt()` in `app/main.py` |
| Business hours | `BUSINESS_HOURS` dict in `app/main.py` |
| TTS voice | `VOICE` constant in `app/main.py` |
| LLM model | `MODEL` constant in `app/main.py` + `ollama pull <model>` |
| Calendar ID | `GOOGLE_CALENDAR_ID` env var at server start |
| Session timeout | `redis_client.setex(..., 3600, ...)` in `app/main.py` |
