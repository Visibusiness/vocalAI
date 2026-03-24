# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

VocalAI is a voice-based conversational AI receptionist ("TestAi") for a clinic called TestClinic. It accepts a WAV audio file, transcribes it, passes it through an LLM, and returns an MP3 response. All conversation is in Romanian.

**Pipeline:** WAV → Faster-Whisper (STT) → Redis (history) → Ollama/Gemma-3 (LLM) → Edge-TTS → MP3

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
ollama create visi-ro -f Modelfile   # only needed once
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Full bootstrap (RunPod / bare metal):**

```bash
./start.sh
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

### `Modelfile` — Ollama model definition

- Pulls `gemma-3-27b-it-GGUF:Q4_K_M` from Hugging Face via Unsloth
- Registers it as the `visi-ro` model in Ollama
- Sets the system prompt, Gemma-3 chat template, stop tokens, and inference parameters (temperature 0.3, ctx 8192)
- To change the persona or model, edit this file and re-run `ollama create visi-ro -f Modelfile`

### Key runtime dependencies

- **Redis** must be running on `localhost:6379` before the FastAPI app starts
- **Ollama** must be running on `localhost:11434` with the `visi-ro` model created
- **CUDA GPU** is required — Whisper loads with `device="cuda", compute_type="float16"`
- **Edge-TTS** makes outbound HTTPS calls to Microsoft; requires internet access

## Changing the AI Persona or Voice

- System prompt: defined both in `Modelfile` (Ollama-level) and `app/main.py` (`SYSTEM_PROMPT` constant, used as the first message in Redis history when a session is new)
- TTS voice: `"ro-RO-AlinaNeural"` in `app/main.py` — change to any Edge-TTS Romanian voice
- LLM model: update `FROM` in `Modelfile` and re-create the Ollama model

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

## Next Steps

- [ ] **Set up Google credentials** — follow the checklist in `app/calendar_service.py` to create a service account, download `credentials.json`, and share the calendar with the service account email
- [ ] **Set `CALENDAR_ID`** — export `GOOGLE_CALENDAR_ID` env var or edit the constant in `calendar_service.py`
- [ ] **End-to-end test** — run the server manually and call `POST /voice` with a Romanian audio file that confirms an appointment; verify the event appears in Google Calendar
- [ ] **LLM reliability testing** — check if Gemma-3 consistently outputs the JSON block in the right format; adjust the system prompt if needed
- [ ] **Date parsing robustness** — users may say "mâine" or "vineri"; consider adding a date normalization step before passing to the calendar API
- [ ] **Conflict checking** — before creating an event, query the calendar for existing events at that time slot and inform the patient if it's taken
- [ ] **Confirmation SMS/email** — after booking, notify the patient via an external service (Twilio, SendGrid, etc.)
- [ ] **Docker update** — add `GOOGLE_CALENDAR_ID` env var to the `docker run` command and ensure `credentials.json` is mounted into the container
