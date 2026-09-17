# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

VocalAI is a voice-based conversational AI receptionist ("Sara") for a clinic called TestClinic. It accepts real phone calls via Twilio, transcribes speech, drives an LLM conversation in Romanian, books/cancels/checks Google Calendar appointments, and speaks back to the caller in real time.

**Pipeline:** Twilio mulaw 8kHz → Silero VAD → Whisper large-v3-turbo (STT) → Redis (history) → Ollama/Gemma-4-26B (LLM) → Google Calendar API → Edge-TTS → mulaw 8kHz → Twilio

## Running the Server

**Full bootstrap (RunPod / bare metal):**

```bash
./setup.sh
```

**Manual start (requires Redis + Ollama already running):**

```bash
export BASE_URL="https://<pod-id>-8000.proxy.runpod.net"
export GOOGLE_CALENDAR_ID="your-email@gmail.com"
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"
pkill -f uvicorn
cd /workspace/vocalAI
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Twilio Console: Voice webhook → `https://<pod-url>/twilio/incoming` — HTTP POST

**Current RunPod proxy URL:** `https://iah5ngs3tm7wbk-8000.proxy.runpod.net`

## Branches

| Branch | Purpose |
|---|---|
| `nicubranci` | Production — Twilio Media Streams, Silero VAD, barge-in |
| `nicubranci-bargein` | Active development — barge-in with grace period fix (newest) |
| `nicubranci_no_twilio` | Laptop testing — streaming pipeline + `client.py`, no Twilio |

## Testing

**Text-based test (no mic needed):**

```bash
python test_client.py --text "Vreau să fac o programare" --phone +40721000000
```

**Mic-based test client:**

```bash
python client.py --phone +40721000000
```

**Manual API test:**

```bash
curl -X POST http://localhost:8000/voice \
  -F "file=@audio.wav" \
  -F "session_id=+40721000000"
```

> Note: `test_client.py` uses the old non-streaming response format and needs updating to the length-prefixed protocol.

## Architecture

### `app/main.py` — FastAPI server

- Whisper, Silero VAD, Ollama, and greeting audio are all pre-loaded at startup (zero cold start on first call)
- **Twilio path** (production): `POST /twilio/incoming` → `<Connect><Stream>` TwiML → `WS /twilio/stream` handles full call
- **Direct path** (testing): `POST /voice` receives WAV + `session_id`, returns streaming MP3
- `session_id` = caller's phone number (`From` field from Twilio start event) — memory persists across calls from the same patient
- Conversation history stored in Redis under `session_id` as JSON array of `{role, content}` with 600s TTL
- No temp files — all audio stays in memory end-to-end

### `WS /twilio/stream` — Media Streams WebSocket handler

Two concurrent coroutines via `asyncio.gather`:
- **`receive_loop`**: reads Twilio JSON frames, decodes base64 mulaw chunks → `asyncio.Queue`
- **`process_loop`**: Silero VAD state machine → end-of-speech detection → `handle_utterance` task

Audio pipeline per utterance:
```
Twilio mulaw 8kHz (160 bytes/chunk = 20ms)
→ Silero VAD end-of-speech detection (500ms silence = 25 frames)
→ audioop.ulaw2lin → scipy sinc resample 8kHz→16kHz → pre-emphasis filter → normalize amplitude → WAV
→ Whisper large-v3-turbo (beam_size=10, dynamic initial_prompt from CLINIC CONFIGURATION)
→ LLM streaming → per-sentence Edge-TTS
→ pydub MP3→PCM → audioop.lin2ulaw → mulaw 8kHz
→ send back through WebSocket to Twilio
```

Key constants (all in `twilio_stream()`):

| Constant | Value | Purpose |
|---|---|---|
| `SILENCE_FRAMES` | 25 | 500ms silence triggers end-of-speech |
| `MIN_SPEECH_FRAMES` | 15 | Ignore utterances < 300ms (noise filter) |
| `PRE_SPEECH_FRAMES` | 10 | 200ms pre-roll before first voiced frame |
| `SILERO_THRESHOLD` | 0.5 | Silero confidence to count as speech (normal) |
| `SILERO_CHUNK_SAMPLES` | 256 | 32ms per Silero inference @ 8kHz |
| `BARGE_IN_THRESHOLD` | 0.7 | Higher confidence required to interrupt AI |
| `BARGE_IN_FRAMES` | 3 | ~96ms of confirmed speech to trigger barge-in |
| `BARGE_IN_GRACE` | 1.5s | Grace period after first AI audio plays before barge-in arms |
| Echo cooldown | `audio_secs + 0.4s` | VAD off while AI audio plays + 0.4s buffer |

### Barge-in

When `is_processing=True` (AI is generating/playing response), Silero still runs on incoming audio at the higher `BARGE_IN_THRESHOLD`. After a `BARGE_IN_GRACE` period (1.5s), if `BARGE_IN_FRAMES` consecutive voiced chunks are detected:

1. `current_utterance_task.cancel()` — cancels the in-flight STT→LLM→TTS coroutine
2. `{"event": "clear", "streamSid": ...}` sent to Twilio — stops audio playback immediately
3. `is_processing=False`, `in_speech=True`, `speech_frames` seeded with pre-speech buffer
4. `handle_utterance` catches `CancelledError` and skips the post-play sleep/drain

The grace period is armed from when the first AI audio chunk is actually sent (not from when processing starts), so STT+LLM latency (~3-4s) doesn't cause false triggers before Sara has said anything.

### JSON action system

The LLM appends a fenced ` ```json ``` ` block to trigger server-side actions. The block is stripped before TTS and never spoken.

| Action | Required fields | What server does |
|---|---|---|
| `schedule` | `name`, `date`, `time` | Validates date/time/business hours → conflict check → creates Google Calendar event |
| `cancel` | `date`, `time` | Finds and deletes the event at that slot |
| `check` | `date`, `time` | Queries calendar, injects result as system message, re-runs LLM |
| `list` | _(none)_ | Lists all future events matching the caller's phone number |
| `hangup` | _(none)_ | Speaks goodbye, sends `HANGUP_MARKER`, closes WebSocket |

Pre-JSON audio buffering: text before the JSON block is buffered (not spoken) until after the calendar action resolves. If there's a conflict or error, the buffered "confirmed" audio is discarded. On success it's played; on hangup it's played then the call ends.

### `app/appointment_parser.py`

Extracts and validates the JSON block from LLM replies. Rejects blocks with any `null` values (AI sent incomplete data). Recognized actions: `schedule`, `cancel`, `check`, `list`, `hangup`.

### `app/calendar_service.py`

Google Calendar service account integration. All slot queries use `zoneinfo.ZoneInfo("Europe/Bucharest")` for DST-aware offsets (EET +02:00 in winter, EEST +03:00 in summer).

Functions: `create_appointment`, `check_conflict`, `cancel_appointment`, `get_appointments`, `list_appointments`.

### STT model

- Model: `large-v3-turbo` via faster-whisper, `device="cuda"`, `compute_type="float16"`, `vad_filter=True`, `beam_size=10`
- `large-v3-turbo` = same encoder as large-v3 (better than medium for Romanian), pruned decoder → ~2x faster, ~1.6 GB VRAM
- `vad_filter=True` prevents hallucinations on silence/noise (required for large-v3 based models)
- `beam_size=10` (up from default 5) — more accurate transcription at ~30% extra compute, still fast on GPU
- `initial_prompt` built dynamically by `_build_whisper_prompt()` from the **CLINIC CONFIGURATION** block — includes clinic name, doctor names, specialties, services, common patient names, Romanian date/time vocabulary, and explicit diacritics words (ă â î ș ț) to prevent stripping
- Audio pre-processing pipeline (`_build_wav_from_mulaw`):
  1. `audioop.ulaw2lin` — mulaw decode
  2. `scipy.signal.resample_poly(up=2, down=1)` — polyphase sinc resampling 8kHz→16kHz (replaces linear interpolation; better transient preservation)
  3. Pre-emphasis filter `y[n] = x[n] - 0.97·x[n-1]` — boosts high frequencies attenuated by the 300Hz–3400Hz phone passband; improves consonant recognition (ș ț s f v)
  4. Normalize to 90% peak amplitude

### VAD model

- Model: Silero VAD via `torch.hub.load("snakers4/silero-vad", onnx=True)` — ONNX backend avoids NNPACK warnings on VMs
- Runs on CPU; 256-sample chunks @ 8kHz = 32ms per inference
- Falls back to RMS energy threshold if Silero fails to load
- Loaded at startup alongside Whisper

### LLM model

- Model: `gemma4:26b` via Ollama, `temperature=0.3`, `num_ctx=8192`
- MoE architecture — runs at ~4B speed while matching or beating Gemma 3 27B quality
- Streamed via `ollama.AsyncClient` — tokens arrive as async generator, no blocking
- System prompt built dynamically by `build_system_prompt()` — injects today's date, business hours, and explicit rules for JSON action format

### Key runtime dependencies

- **Redis** — `localhost:6379`
- **Ollama** — `localhost:11434` with `gemma4:26b` pulled
- **CUDA GPU** — Whisper requires `device="cuda", compute_type="float16"`
- **Edge-TTS** — outbound HTTPS to Microsoft; requires internet
- **`credentials.json`** — Google service account key in project root

## Clients

- **`client.py`** — mic-based; `--phone +40xxx` uses phone as session ID, `--new` resets session
  - Streaming playback: `httpx.stream()` + length-prefixed protocol; playback queue + background thread
- **`test_client.py`** — text-based; `--text "..."` converts to WAV via Edge-TTS, sends to server
  - Still uses old non-streaming format — needs updating to length-prefixed protocol

## Clinic Configuration (plug-and-play)

All clinic-specific settings live in the **CLINIC CONFIGURATION** block near the top of `app/main.py`. Every field has a `# TODO` comment. Update these before deploying to a real clinic:

| Variable | Default (test) | What it controls |
|---|---|---|
| `CLINIC_NAME` | `"TestClinic"` | Used in Sara's system prompt and Whisper prompt |
| `CLINIC_CITY` | `"București"` | Whisper prompt context |
| `CLINIC_DOCTORS` | `["Dr. Popescu", "Dr. Ionescu"]` | Whisper prompt + system prompt |
| `CLINIC_SPECIALTIES` | `["stomatologie"]` | Whisper prompt |
| `CLINIC_SERVICES` | detartraj, plombă, etc. | Whisper prompt vocabulary |
| `CLINIC_COMMON_SURNAMES` | Romanian surnames | Whisper spelling bias |
| `CLINIC_COMMON_FIRSTNAMES` | Romanian first names | Whisper spelling bias |

The Whisper `initial_prompt` is built automatically from these by `_build_whisper_prompt()` — no manual editing needed.

## Changing the AI Persona or Voice

- System prompt: `build_system_prompt()` in `app/main.py` — persona is "Sara", includes tone/style rules, sentiment mirroring, and few-shot dialogue examples
- TTS voice: `"ro-RO-AlinaNeural"` in `app/main.py`
- Greeting text: `_GREETING_VARIANTS` list in `app/main.py` — one variant is picked randomly at startup each time the server starts
- LLM model: update model name string in `app/main.py` and `ollama pull <model>`

## Business Hours

Configured in `BUSINESS_HOURS` dict in `app/main.py`. Currently Mon–Fri 09:00–17:00, closed weekends. The system prompt is dynamically built from this dict so the LLM knows the schedule. Server-side validation also blocks bookings outside hours before they reach the calendar.

---

## Google Calendar Setup (new server / first time)

The app uses a **Google Service Account** to write events. Do steps 1–2 once per Google account — they persist permanently.

### Step 1 — Create service account credentials (one-time)

1. Google Cloud Console → your project → enable **Google Calendar API**
2. IAM & Admin → Service Accounts → Create Service Account
3. Keys → Add Key → Create new key → JSON → download it

### Step 2 — Share your calendar (one-time)

1. Copy `client_email` from the downloaded JSON
2. Google Calendar → 3 dots → Settings and sharing → Share with specific people
3. Add `client_email` with **"Make changes to events"** permission

### Step 3 — Copy credentials to server

```bash
scp -P <port> -i ~/.ssh/id_ed25519 ~/Downloads/key.json root@<server>:/workspace/vocalAI/credentials.json
```

### Step 4 — Start server with your calendar ID

```bash
GOOGLE_CALENDAR_ID="your-email@gmail.com" uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Latency (current state)

| Step | Time |
|---|---|
| Caller speaks + VAD silence detection | ~0.5s |
| Whisper transcription | ~1.5s |
| LLM first sentence | ~1–2s |
| TTS first sentence | ~0.3s |
| **First word heard** | **~3–4s after caller stops speaking** |

Main bottleneck is Whisper (~1.5s). Options: streaming Whisper (not yet in faster-whisper), or Deepgram streaming STT.

---

## Known Limitations

- **8kHz phone audio** — mulaw cuts frequencies above 4kHz; Romanian names still occasionally garbled. Pre-emphasis filter, sinc resampling, `beam_size=10`, and rich `initial_prompt` all help but this is an inherent codec limit.
- **Short utterance STT** — clips under ~2s (e.g. "Da. Confirm.") are more likely to be misrecognized.
- **Echo cooldown is estimated** — playback duration inferred from byte count; slightly off if Twilio's buffer adds delay.
- **Full-day calendar scan** — "am ceva pe 25 martie?" (no specific time) sends 00:00 and returns empty. Day-range query not yet implemented.
- **Digi Romania** — Twilio US/UK numbers may not be reachable from Digi. Fix: get a +40 Romanian Twilio number (requires regulatory bundle).
- **test_client.py outdated** — still uses the old non-streaming response format.

---

## Next Steps

- [x] Appointment booking, cancellation, conflict check, list ✅
- [x] Streaming LLM → TTS pipeline (~2s first word) ✅
- [x] Whisper large-v3-turbo ✅
- [x] Twilio Media Streams (full-duplex WebSocket) ✅
- [x] Silero VAD (replaces webrtcvad) ✅
- [x] Echo prevention (duration-based cooldown) ✅
- [x] Pre-speech ring buffer (200ms pre-roll) ✅
- [x] Phone number as session ID (memory across calls) ✅
- [x] Hangup action ✅
- [x] DST-aware timezone fix (Europe/Bucharest via zoneinfo) ✅
- [x] Barge-in (caller interrupts AI mid-response) ✅
- [x] Barge-in grace period (prevents false triggers on noise) ✅
- [x] **Romanian +40 Twilio number** — acquired, Digi reachability resolved ✅
- [x] **Humanized AI persona** — renamed to Sara, natural tone/style rules, sentiment mirroring, randomized greeting variants, few-shot dialogue examples in system prompt ✅
- [x] **Upgrade LLM to Gemma 4 26B MoE** — faster inference, better quality than Gemma 3 27B ✅
- [x] **Fix cancel_appointment bug** — was called with 3 args but accepts 2; cancellations now work ✅
- [x] **Improve 8kHz phone audio STT quality** — sinc resampling, pre-emphasis filter, beam_size=10, rich dynamic initial_prompt ✅
- [x] **Plug-and-play CLINIC CONFIGURATION block** — one place to set doctors, specialties, names, city for any clinic ✅
- [ ] **SMS/WhatsApp confirmation** — skipped for now (Twilio number is voice-only; WhatsApp API not free)
- [ ] **Full-day calendar scan** — "am ceva pe 25 martie?" needs day-range query, not just HH:MM slot
- [ ] **Slot suggestion** — Sara can't answer "când ești liber?"; needs `find_free_slots` calendar query
- [ ] **Update test_client.py** — needs length-prefixed streaming protocol
- [ ] **Longer Redis TTL** — 600s loses history if patient calls back after 10min; should be hours
- [ ] **Cache Google Calendar service object** — `_get_service()` re-authenticates on every action; should be a module-level singleton
- [ ] **PostgreSQL** — replace Redis with persistent DB for call history + audit trail
- [ ] **Multi-doctor scheduling** — each doctor has own calendar; route by specialty or availability
- [ ] **Fine-tune STT** — Whisper on Romanian medical vocab (worth doing once real call data exists)
- [ ] **Fine-tune LLM** — train on real receptionist conversations for more consistent booking flow
