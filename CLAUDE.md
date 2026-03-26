# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

VocalAI is a voice-based conversational AI receptionist ("TestRec") for a clinic called TestClinic. It accepts a WAV audio file, transcribes it, passes it through an LLM, and returns an MP3 response. All conversation is in Romanian.

**Pipeline:** WAV → Faster-Whisper (STT) → Redis (history) → Ollama/Gemma-3-27B (LLM) → Google Calendar API → Edge-TTS → MP3

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

### `app/main.py` — FastAPI server

- Whisper and Ollama are pre-loaded at startup to eliminate cold start on first request
- The only endpoint is `POST /voice`: receives `file` (WAV) + `session_id` (form fields)
- `session_id` is the patient's phone number when using `--phone` flag, otherwise a random ID
- Conversation history stored in Redis under `session_id` as JSON array of `{role, content}` messages with 600s TTL
- Blocking operations (Whisper, Ollama) run via `run_in_threadpool` to avoid blocking the async event loop
- After LLM reply, `extract_appointment()` parses a JSON action block; the server routes to the appropriate handler
- Temp files (`in_<uuid>.wav`, `out_<uuid>.mp3`) deleted by background task 2 seconds after response

### JSON action system

The LLM appends a fenced ` ```json ``` ` block to trigger server-side actions. The block is stripped before TTS.

| Action | Required fields | What server does |
|---|---|---|
| `schedule` | `name`, `date`, `time` | Checks conflict → creates Google Calendar event |
| `cancel` | `date`, `time` | Finds and deletes the event at that slot |
| `check` | `date`, `time` | Queries calendar, injects result, re-runs LLM |
| `list` | _(none)_ | Lists all future events matching phone number |

### `app/appointment_parser.py`

Extracts and validates the JSON block from LLM replies. Rejects blocks with `null` values (AI sent too early).

### `app/calendar_service.py`

Google Calendar service account integration. Functions: `create_appointment`, `check_conflict`, `cancel_appointment`, `get_appointments`, `list_appointments`.

### LLM model

- Model: `hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M` via Ollama, `temperature=0.3`, `num_ctx=8192`
- System prompt built dynamically by `build_system_prompt()` — injects today's date so AI knows the current year

### Key runtime dependencies

- **Redis** must be running on `localhost:6379`
- **Ollama** must be running on `localhost:11434` with the Gemma-3 model pulled
- **CUDA GPU** required — Whisper loads with `device="cuda", compute_type="float16"`
- **Edge-TTS** makes outbound HTTPS calls to Microsoft; requires internet access
- **`credentials.json`** — Google service account key must be present in project root

## Clients

- **`client.py`** — mic-based client; `--phone +40xxx` uses phone as session ID, `--new` resets session
- **`test_client.py`** — text-based test client; `--text "..."` converts text to WAV via Edge-TTS and sends to server

## Changing the AI Persona or Voice

- System prompt: `build_system_prompt()` in `app/main.py`
- TTS voice: `"ro-RO-AlinaNeural"` in `app/main.py`
- LLM model: update model name string in `app/main.py` and re-pull via `ollama pull <model>`

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

- `credentials.json` — service account key present on RunPod ✅
- `CALENDAR_ID` — passed via `GOOGLE_CALENDAR_ID` env var at server start ✅
- Calendar shared with service account email ✅
- Booking, cancellation, conflict check, list — all working ✅

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

## Session Summary — 2026-03-26

### Features implemented

- **Conflict checking** — before booking, server queries Google Calendar for the 1-hour slot; if taken, LLM re-run proposes another time
- **Appointment cancellation** — `cancel` action deletes the event; if not found, LLM re-run responds naturally
- **Appointment check** — `check` action queries calendar and injects real result back before LLM responds
- **Past date validation** — server rejects booking dates in the past, re-runs LLM to ask for future date
- **Null JSON guard** — `appointment_parser.py` rejects blocks with any `null` values so incomplete JSON never triggers an action
- **Current date in system prompt** — `build_system_prompt()` injects today's date; AI no longer asks patients for the year
- **Session ID debug print** — `client.py` prints `[session] <id>` before each request

### Known limitations

- **Check by full day** — patient asking "do I have anything on March 25" (no specific time) causes AI to use `00:00`, always returning empty. Full-day scan not yet implemented.
- **Vague dates** — "mâine", "vineri viitoare" work if LLM resolves them correctly; no server-side normalization fallback.

---

## Next Steps

- [x] **Fix Google Calendar 404** — calendar shared with service account, booking confirmed working ✅
- [x] **Conflict checking** — checks for existing events before booking, informs patient if slot is taken ✅
- [x] **Date parsing robustness** — system prompt enforces current year, handles "mâine"/"azi" etc. ✅
- [x] **Phone number as session ID** — `--phone` arg on both clients; stored in calendar event description ✅
- [x] **List appointments action** — patient can ask "ce programare am?" and system queries calendar by phone ✅
- [x] **Confirmation gate fixed** — AI now asks confirmation in a separate turn before sending schedule/cancel JSON ✅
- [ ] **Confirmation SMS/email** — after booking, notify the patient via Twilio/SendGrid
- [ ] **Twilio integration** — replace manual `--phone` arg with real inbound call handler
- [ ] **Docker update** — add `GOOGLE_CALENDAR_ID` env var and mount `credentials.json` into container
- [ ] **LLM reliability** — stress-test edge cases (interruptions mid-booking, ambiguous confirmations)

---

## Session Summary — 2026-03-25 (bug fixes & list action)

### Problems found during testing

| Issue | Root cause | Fix |
|---|---|---|
| Appointment created before confirmation | AI sent JSON in same turn as confirmation question | System prompt: explicit rule to never send JSON in same turn as confirmation |
| AI announced date deductions aloud | No rule against explaining reasoning | System prompt: "do not explain deductions, use them silently" |
| "Este liber mâine la 10?" didn't trigger check | AI gave generic greeting instead | System prompt: clearer rule that date+time present → send check JSON immediately |
| "Ce programare am?" not handled | No `list` action existed | Added `list` action end-to-end |
| f-string ValueError on startup | JSON example blocks used bare `{}` in f-string | Escaped all literal braces as `{{}}` |
| LLM used year 2024 instead of current | Weak year rule in system prompt | Hardened: "ALWAYS use {today.year}, NEVER another year" |

### New files / changes

| File | What changed |
|---|---|
| `app/calendar_service.py` | Added `list_appointments(phone)` — queries future events by phone via Google Calendar `q` param |
| `app/appointment_parser.py` | Added `"list"` action (only requires `action` key) |
| `app/main.py` | Import `list_appointments`; handle `list` action; rewrote system prompt behavior rules |
| `test_client.py` | New text-based test client (edge-tts → WAV → POST); `--phone` and `--text` args |
| `client.py` | Added `--phone` arg (uses phone as session ID) |

### How `list` works

1. Patient says "ce programare am?" / "am vreo programare?"
2. LLM emits `` ```json {"action": "list"} ``` ``
3. Server calls `list_appointments(session_id)` — queries Google Calendar with `q=<phone>`
4. Results injected back as system message → LLM replies naturally
5. Clean text returned as audio
