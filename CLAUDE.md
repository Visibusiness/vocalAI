# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

VocalAI is a voice-based conversational AI receptionist ("TestAi") for a clinic called TestClinic. It accepts a WAV audio file, transcribes it, passes it through an LLM, and returns an MP3 response. All conversation is in Romanian.

**Pipeline:** WAV → Faster-Whisper (STT) → Redis (history) → Ollama/Gemma-3-27B (LLM) → Edge-TTS → MP3

## Running the Server

**Via Docker (recommended):**

```bash
docker build -t vocalai .
docker run --gpus all -v ollama_data:/root/.ollama -p 8000:8000 vocalai
```

**Manually (requires Redis + Ollama running):**

```bash
redis-server --daemonize yes
ollama serve &
ollama pull hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M   # only needed once (~17 GB)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Full bootstrap (RunPod / bare metal):**

```bash
./setup.sh
```

## Testing

Test client (records 5s of mic audio, sends to server, plays response):

```bash
python client.py
```

The `SERVER_URL` in `client.py` must be updated to point to the running server (currently hardcoded to a RunPod proxy URL).

Manual API test:

```bash
curl -X POST http://localhost:8000/voice \
  -F "file=@audio.wav" \
  -F "session_id=test_session"
```

## Architecture

### `app/main.py` — Single-file FastAPI server

- Whisper model is loaded once at startup (`@app.on_event("startup")`) into global `stt_model`
- The only endpoint is `POST /voice`: receives `file` (WAV) + `session_id` (form fields)
- Conversation history is stored in Redis under the `session_id` key as a JSON array of `{role, content}` messages with a 600-second TTL
- Blocking operations (Whisper transcription, Ollama chat) run via `run_in_threadpool` to avoid blocking the async event loop
- Temp files (`in_<uuid>.wav`, `out_<uuid>.mp3`) are created per-request and deleted by a background task 2 seconds after the response is sent

### LLM model

- Model: `hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M` pulled directly via Ollama (no alias)
- Called directly by name in `app/main.py` with `temperature=0.3`

### Key runtime dependencies

- **Redis** must be running on `localhost:6379` before the FastAPI app starts
- **Ollama** must be running on `localhost:11434` with `hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M` pulled
- **CUDA GPU** is required — Whisper loads with `device="cuda", compute_type="float16"`
- **Edge-TTS** makes outbound HTTPS calls to Microsoft; requires internet access

## Changing the AI Persona or Voice

- System prompt: `SYSTEM_PROMPT` constant in `app/main.py` — injected as the first message in Redis history for new sessions
- TTS voice: `"ro-RO-AlinaNeural"` in `app/main.py` — change to any Edge-TTS Romanian voice
- LLM model: update the model name string in `app/main.py` and re-pull via `ollama pull <model>`

---

## Session Summary — 2026-03-24

### Features implemented

- **Appointment booking pipeline** — the assistant now collects patient name, date, and time through natural Romanian conversation and books a real Google Calendar event when the user confirms.
- **Structured JSON output** — the LLM is instructed to append a `\`\`\`json { "action": "schedule", ... }\`\`\`` block to its reply only upon confirmation. The JSON is parsed in Python, never read aloud.
- **Updated persona** — receptionist name changed to "TestRec", clinic name to "TestClinic". System prompt rewritten in Romanian with explicit booking instructions.

### Files created

| File | Purpose |
|---|---|
| `app/appointment_parser.py` | Extracts and validates the JSON booking block from the LLM reply; returns clean text for TTS |
| `app/calendar_service.py` | Authenticates with Google Calendar (service account) and creates a 1-hour event |
| `credentials.json.example` | Template showing the shape of the required Google service account JSON file |

### Files modified

| File | What changed |
|---|---|
| `app/main.py` | New imports; `SYSTEM_PROMPT` rewritten; post-LLM step calls parser + calendar; TTS uses clean text |
| `requirements.txt` | Added `google-api-python-client` and `google-auth`; removed duplicate `python-multipart` |
| `.gitignore` | Added `credentials.json` to prevent accidental credential commits |

### What the system can do now

1. Receive a WAV audio file and transcribe it (Whisper)
2. Hold a multi-turn conversation in Romanian with session memory (Redis)
3. Collect patient name, date, and time through natural dialogue
4. Detect when an appointment is confirmed and parse structured data from the LLM reply
5. Create a Google Calendar event automatically (1 hour, `Europe/Bucharest` timezone)
6. Return only the natural Romanian text as audio to the user (JSON never spoken)

---

## Session Summary — 2026-03-24 (bootstrap & cleanup)

### Changes made

- **`setup.sh` created** — single-command bootstrap for RunPod / bare-metal Linux. Installs system packages, Ollama, Python deps, pre-downloads Whisper `medium` weights, pulls the Gemma-3 model, and starts the FastAPI server.
- **Modelfile removed** — no longer needed. Model is pulled directly via `ollama pull`; no custom alias.
- **`app/main.py`** — model reference changed from `"visi-ro"` to `"hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M"`.
- **`requirements.txt`** — removed `scipy` and `sounddevice` (only used in deleted scripts); added `fastapi` and `uvicorn[standard]` which were missing.
- **`stt.py` deleted** — old standalone mic-based script (Piper TTS, webrtcvad); replaced entirely by the FastAPI server.
- **`calendar_tool.py` deleted** — old OAuth-based calendar helper; replaced by `app/calendar_service.py` (service account).

---

---

## Session Summary — 2026-03-24 (client, memory, calendar fixes)

### What was built

- **`client.py` created** — local laptop client that records mic audio (6s default), sends to `/voice`, prints AI reply text in terminal, and plays back the MP3 response through speakers.
  - Uses `sounddevice` for both recording and playback (pydub's `ffplay` backend had no audio output on Linux desktop)
  - Session ID is persisted to `.session_id` file — survives restarts, memory continues across runs
  - `--new` flag starts a fresh session; `--session <id>` overrides manually
  - Requires: `pip install sounddevice soundfile httpx pydub` + `sudo apt install portaudio19-dev ffmpeg`
  - Run from `.venv`: `source .venv/bin/activate && python3 client.py`

- **`app/main.py` updated**:
  - Returns AI reply text as URL-encoded `X-AI-Text` response header (Romanian characters can't go in headers as raw UTF-8)
  - Added `flush=True` to all print statements (stdout was buffering on RunPod)
  - Added Redis hit/miss debug logging per request: `[redis] Loaded N messages for session X`

### Bugs fixed

| Bug | Cause | Fix |
|---|---|---|
| No audio playback on laptop | pydub used `ffplay` which had no output device | Switched to `sounddevice` + pydub decode |
| Romanian chars crashed header | HTTP headers are Latin-1 only | `urllib.parse.quote/unquote` on `X-AI-Text` |
| Memory lost on client restart | Random session ID generated each run | Persist session ID to `.session_id` file |
| Server logs not flushing | Python stdout buffering | Added `flush=True` to prints |
| `os` not imported in client | Missing import | Fixed |

### RunPod deployment

- **SSH**: `ssh root@195.26.232.186 -p 25154 -i ~/.ssh/id_ed25519`
- **SCP a file**: `scp -P 25154 -i ~/.ssh/id_ed25519 <local_file> root@195.26.232.186:/workspace/vocalAI/<path>`
- **Proxy URL**: `https://gj4u6gqf1pn91j-8000.proxy.runpod.net/`
- **Restart server**: `pkill -f uvicorn && cd /workspace/vocalAI && GOOGLE_CALENDAR_ID="calinanicolas91@gmail.com" uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Google Calendar status

- `credentials.json` — copied to `/workspace/vocalAI/credentials.json` on RunPod ✅
- `CALENDAR_ID` — hardcoded to `calinanicolas91@gmail.com` in `app/calendar_service.py` ✅
- Appointment parsing — working correctly (JSON extracted, name/date/time parsed) ✅
- Calendar event creation — **FAILING with 404 Not Found** ❌

**Root cause of 404**: The Google Calendar has not been shared with the service account email.

**Fix needed (one-time, in Google Calendar UI)**:
1. Find service account email: `python3 -c "import json; d=json.load(open('/workspace/vocalAI/credentials.json')); print(d['client_email'])"`
2. Open Google Calendar → 3 dots next to calendar → Settings and sharing → Share with specific people
3. Add the service account email with **"Make changes to events"** permission
4. Restart the server

---

## Google Calendar Setup (new server / first time)

The app uses a **Google Service Account** to write events to your calendar. Do this once per Google account — the calendar sharing persists permanently.

### Step 1 — Create service account credentials (one-time)

1. Go to [Google Cloud Console](https://console.cloud.google.com) → your project
2. Enable the **Google Calendar API** if not already enabled
3. Go to **IAM & Admin → Service Accounts → Create Service Account**
4. Give it any name, click through
5. Go to **Keys → Add Key → Create new key → JSON** → download it

### Step 2 — Share your calendar with the service account (one-time)

1. Open the downloaded JSON and copy the `client_email` value (looks like `xxx@your-project.iam.gserviceaccount.com`)
2. Open **Google Calendar** → 3 dots next to your calendar → **Settings and sharing** → **Share with specific people**
3. Add the `client_email` with **"Make changes to events"** permission

### Step 3 — Copy credentials to the server

```bash
scp -P <port> -i ~/.ssh/id_ed25519 ~/Downloads/your-key.json root@<server-ip>:/workspace/vocalAI/credentials.json
```

### Step 4 — Start the server with your calendar ID

```bash
pkill -f uvicorn
cd /workspace/vocalAI
GOOGLE_CALENDAR_ID="your-email@gmail.com" uvicorn app.main:app --host 0.0.0.0 --port 8000
```

That's it. Steps 1 and 2 never need to be repeated — only steps 3 and 4 when moving to a new server.

---

## Next Steps

- [x] **Fix Google Calendar 404** — calendar shared with service account, booking confirmed working ✅
- [ ] **LLM reliability testing** — check if Gemma-3 consistently outputs the JSON block in the right format; adjust the system prompt if needed
- [ ] **Date parsing robustness** — users may say "mâine" or "vineri"; consider adding a date normalization step before passing to the calendar API
- [ ] **Conflict checking** — before creating an event, query the calendar for existing events at that time slot and inform the patient if it's taken
- [ ] **Confirmation SMS/email** — after booking, notify the patient via an external service (Twilio, SendGrid, etc.)
- [ ] **Docker update** — add `GOOGLE_CALENDAR_ID` env var to the `docker run` command and ensure `credentials.json` is mounted into the container
