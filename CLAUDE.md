# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

VocalAI is a voice-based conversational AI receptionist ("TestRec") for a clinic called TestClinic. It accepts a WAV audio file, transcribes it, passes it through an LLM, and returns an MP3 response. All conversation is in Romanian.

**Pipeline:** WAV → Faster-Whisper large-v3-turbo (STT) → Redis (history) → Ollama/Gemma-3-27B (LLM) → Google Calendar API → Edge-TTS → streaming MP3

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
- The only endpoint is `POST /voice`: receives `file` (WAV) + `session_id` (form fields), returns a `StreamingResponse`
- `session_id` is the patient's phone number when using `--phone` flag, otherwise a random ID
- Conversation history stored in Redis under `session_id` as JSON array of `{role, content}` messages with 600s TTL
- WAV bytes are passed directly to Whisper via `io.BytesIO` — no disk write
- LLM streams tokens via `ollama.AsyncClient`; complete sentences are sent to Edge-TTS as they arrive
- Audio is returned as a stream of length-prefixed MP3 chunks (4-byte LE uint32 + MP3 bytes); `\x00\x00\x00\x00` = end
- After LLM reply, `extract_appointment()` parses a JSON action block; `handle_calendar_action()` routes to the appropriate handler
- No temp files — all audio stays in memory end-to-end

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

### STT model

- Model: `large-v3-turbo` via faster-whisper, `device="cuda"`, `compute_type="float16"`, `vad_filter=True`
- `large-v3-turbo` = same encoder as large-v3 (better than medium for Romanian), pruned decoder → ~2x faster than medium, ~1.6 GB VRAM
- `vad_filter=True` prevents hallucinations on silence/noise (required for large-v3 based models)

### LLM model

- Model: `hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M` via Ollama, `temperature=0.3`, `num_ctx=8192`
- Streamed via `ollama.AsyncClient` — tokens arrive as async generator, no blocking
- System prompt built dynamically by `build_system_prompt()` — injects today's date so AI knows the current year

### Key runtime dependencies

- **Redis** must be running on `localhost:6379`
- **Ollama** must be running on `localhost:11434` with the Gemma-3 model pulled
- **CUDA GPU** required — Whisper loads with `device="cuda", compute_type="float16"`
- **Edge-TTS** makes outbound HTTPS calls to Microsoft; requires internet access
- **`credentials.json`** — Google service account key must be present in project root

## Clients

- **`client.py`** — mic-based client; `--phone +40xxx` uses phone as session ID, `--new` resets session
  - Uses `httpx.stream()` + length-prefixed protocol to receive and play audio sentence by sentence
  - Playback queue + background thread: next sentence decoded while current one plays (no gap)
  - Session ID persisted to `.session_id` file across runs
- **`test_client.py`** — text-based test client; `--text "..."` converts text to WAV via Edge-TTS and sends to server
  - Note: `test_client.py` still uses the old non-streaming response format and needs updating

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

- [x] **Fix Google Calendar 404** ✅
- [x] **Conflict checking** ✅
- [x] **Date parsing robustness** ✅
- [x] **Phone number as session ID** ✅
- [x] **List appointments action** ✅
- [x] **Confirmation gate fixed** ✅
- [x] **Streaming LLM → TTS pipeline** ✅ — first word heard ~2s after speaking
- [x] **Skip disk write for STT** ✅ — WAV bytes passed directly via io.BytesIO
- [x] **Whisper large-v3-turbo** ✅ — faster + more accurate than medium for Romanian
- [x] **Twilio integration** ✅ — real inbound phone calls working end-to-end
- [ ] **Latency optimization** — CRITICAL. Current round-trip is ~15-20s per turn (Twilio record → download → Whisper → LLM → TTS → upload). Must get under 3s for usable conversation. See dedicated section below.
- [ ] **Twilio Media Streams** — replace record+download with real-time WebSocket audio to eliminate the biggest latency bottleneck (~3-5s saved)
- [ ] **Barge-in / interruption** — user should be able to speak while AI is talking and interrupt it; requires Media Streams
- [ ] **End-of-speech detection** — currently uses 3s silence timeout in Twilio `<Record>`; should use VAD (voice activity detection) on the stream for faster cutoff (~0.5s instead of 3s)
- [ ] **SMS confirmation** — after booking, send patient a confirmation SMS via Twilio
- [ ] **PostgreSQL database** — replace Redis with persistent DB for call history, appointment audit trail, analytics
- [ ] **Multi-doctor scheduling** — each doctor has their own calendar; route by specialty or availability
- [ ] **Full-day calendar scan** — "am ceva pe 25 martie?" currently sends 00:00 and returns empty; needs day-range query
- [ ] **Docker update** — mount `credentials.json` + env vars in Dockerfile
- [ ] **Update `test_client.py`** — still uses old non-streaming response format, needs updating to length-prefixed protocol
- [ ] **Fine-tune STT** — train Whisper on Romanian medical vocabulary (worth doing once real call data exists)
- [ ] **Fine-tune LLM** — train on real receptionist conversations for more consistent booking flow

---

## Latency Problem & Solutions

Current pipeline per turn (Twilio record+respond flow):

| Step | Time |
|---|---|
| Caller speaks + 3s silence timeout | 3s (fixed overhead) |
| Twilio processes + POSTs to server | ~0.5s |
| Download WAV from Twilio API | ~1s |
| Whisper transcription | ~1.5s |
| LLM response | ~1-2s |
| TTS generation (full response) | ~1-2s |
| Twilio fetches MP3 + plays | ~0.5s |
| **Total** | **~9-13s per turn** |

This is too slow for natural conversation. Target is <3s end-to-end.

### Solution: Twilio Media Streams (WebSocket)

Replace the current record+download approach with a real-time WebSocket connection. Twilio streams raw audio to the server as it's being spoken, eliminating the 3s silence wait and the download step.

Architecture:
```
Caller speaks
→ Twilio streams audio chunks via WebSocket to /twilio/stream
→ Server runs VAD on chunks (detect end of speech ~0.5s of silence)
→ Server pipes audio into Whisper in real-time
→ LLM starts as soon as transcription is ready
→ TTS streams first sentence back through WebSocket while LLM generates the rest
→ Caller hears first word in ~2s
```

Key changes needed:
- Add WebSocket endpoint `/twilio/stream`
- Implement VAD (silero-vad or webrtcvad) for end-of-speech detection
- Replace `<Record>` with `<Connect><Stream>` TwiML
- Handle barge-in: if new audio arrives while TTS is playing, cancel current playback
- Audio format: Twilio sends mulaw 8kHz; needs conversion to 16kHz PCM for Whisper

---

## Session Summary — 2026-03-29 (Twilio integration)

### What was built

- **Twilio inbound call handling** — real phone calls now route to the AI end-to-end
- **Three new endpoints** added to `app/main.py`:
  - `POST /twilio/incoming` — answers call, plays Romanian greeting via Edge-TTS, starts recording
  - `POST /twilio/process` — downloads Twilio WAV recording, runs full STT→LLM→TTS pipeline, returns TwiML `<Play>`
  - `GET /audio/{id}` — serves cached MP3 chunks to Twilio (in-memory dict, 5-min TTL)
- **`setup.sh` updated** — now requires and passes `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `GOOGLE_CALENDAR_ID` as env vars; fails fast with clear error if any is missing

### Bugs fixed during integration

| Bug | Cause | Fix |
|---|---|---|
| 401 on recording download | `TWILIO_AUTH_TOKEN` env var empty (shell variable not exported) | Use `export` before running setup.sh; validate at startup |
| Twilio error voice on hangup | 404 on final silence recording after caller hangs up | Catch 404 specifically, return empty `<Response>` |
| httpx stripping auth on redirect | httpx security feature strips auth headers cross-domain | Switched to `requests` library for recording download |

### How to start the server

```bash
export GOOGLE_CALENDAR_ID="your-email@gmail.com"
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"
./setup.sh
```

### Twilio Console configuration

- Voice webhook: `https://your-runpod-url/twilio/incoming` — HTTP POST
- Account SID + Auth Token: Twilio Console dashboard → Account Info

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

---

## Session Summary — 2026-03-27 (performance & streaming)

### Changes made

| Change | Detail |
|---|---|
| **Skip disk write for STT** | WAV bytes passed directly to Whisper via `io.BytesIO` — no temp file written/read/deleted |
| **Whisper `medium` → `large-v3-turbo`** | ~2x faster than medium, better Romanian accuracy (large-v3 encoder), ~1.6 GB VRAM |
| **`vad_filter=True`** | Required for large-v3 based models — prevents hallucinations on silence and background noise |
| **Streaming LLM → TTS pipeline** | LLM tokens streamed via `ollama.AsyncClient`; each complete sentence sent to Edge-TTS immediately; audio chunks streamed to client as they're ready |
| **Length-prefixed streaming protocol** | Response is `application/octet-stream`: 4-byte LE uint32 length + MP3 bytes per sentence; `\x00\x00\x00\x00` = end of stream |
| **No more temp output files** | Edge-TTS audio goes directly to bytes in memory; `cleanup_output()` removed entirely |
| **`handle_calendar_action()` extracted** | Action handling logic moved to a dedicated async function; `llm_call()` helper for non-streaming follow-up LLM calls |
| **`client.py` streaming playback** | Uses `httpx.stream()` + `iter_audio_chunks()` to parse protocol; playback queue + background thread plays each sentence immediately, next decoded while current plays |

### Latency improvement

- **Before:** wait for full LLM response (~8–15s) + full TTS (~2–3s) → first word heard after ~15s
- **After:** first sentence from LLM (~1–2s) + TTS that sentence (~300ms) → first word heard after ~2s

### Key implementation details

- `SENTENCE_END = re.compile(r'(?<=[.!?])\s')` — splits on sentence boundaries mid-stream
- JSON block detection: `"```json" in full_response` — when detected, remaining tokens consumed silently, pre-JSON text already spoken
- Action responses (check/list/conflict/cancel-not-found) also streamed sentence by sentence after the calendar API call
- `ollama.AsyncClient` used for both streaming and non-streaming calls (no `run_in_threadpool` for LLM)

### Known issues

- `test_client.py` still uses the old `FileResponse` format — it will break against the new streaming endpoint and needs updating
