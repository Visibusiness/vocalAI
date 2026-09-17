# VocalAI — Voice Receptionist

A Romanian-language AI receptionist for a clinic. Patients call a real phone number, the AI understands them, holds a conversation, and books appointments directly into Google Calendar.

## How it works

```
Patient calls Twilio number
       ↓
Twilio opens WebSocket → WS /twilio/stream
       ↓
Silero VAD (end-of-speech detection)
       ↓
Whisper large-v3-turbo (speech-to-text)
       ↓
Redis (conversation memory per caller)
       ↓
Ollama / Gemma 4 26B (understands + replies in Romanian)
       ↓
Google Calendar (books/cancels/checks appointments)
       ↓
Edge-TTS → mulaw audio streamed back to Twilio
       ↓
Patient hears the response
```

The assistant collects the patient's **name**, **date**, and **time** through natural conversation. When the patient confirms, it creates a 1-hour event in Google Calendar automatically.

---

## Running the server

**Full bootstrap on a fresh Linux server (RunPod / bare metal):**

```bash
export GOOGLE_CALENDAR_ID="your-email@gmail.com"
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"
./setup.sh
```

`setup.sh` installs all dependencies, starts Redis and Ollama, pulls the LLM model, and launches the FastAPI server. All three env vars are required — the script will error immediately if any is missing.

**Requirements:**
- NVIDIA GPU with CUDA
- Internet access (Edge-TTS calls Microsoft servers, Twilio downloads recordings)

---

## Twilio setup (one-time)

1. Create a free account at [twilio.com](https://twilio.com) and get a phone number
2. In **Twilio Console → Phone Numbers → your number → Voice webhook**, set:
   - URL: `https://your-server/twilio/incoming`
   - Method: `HTTP POST`
3. Find your **Account SID** and **Auth Token** on the Twilio Console dashboard
4. Pass them as env vars when starting the server (see above)

Your `BASE_URL` defaults to the RunPod proxy URL hardcoded in `app/main.py`. Update it via the `BASE_URL` env var if your server URL changes.

---

## Testing from your laptop (mic client)

```bash
pip install sounddevice soundfile httpx pydub
sudo apt install portaudio19-dev ffmpeg

python client.py                    # records 6s of mic audio, sends to server, plays response
python client.py --new              # start a fresh conversation session
python client.py --phone +40721000  # use phone number as session ID
```

Update `SERVER_URL` in `client.py` to point to your running server.

---

## Google Calendar setup

Short version:
1. Create a **service account** in Google Cloud Console → download the JSON key
2. Share your Google Calendar with the service account email (`Make changes to events`)
3. Copy the JSON key to the server as `credentials.json`
4. Pass `GOOGLE_CALENDAR_ID="your-email@gmail.com"` at server start

See the **Google Calendar Setup** section in `CLAUDE.md` for full steps.

---

## Key files

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI server — `/voice`, `/twilio/incoming`, `WS /twilio/stream`, `/audio/{id}` |
| `app/appointment_parser.py` | Extracts booking JSON from LLM reply |
| `app/calendar_service.py` | Google Calendar integration |
| `client.py` | Local mic test client |
| `setup.sh` | Full server bootstrap (requires env vars) |
| `credentials.json` | Google service account key (not in git) |

---

## Customization

| What | Where |
|---|---|
| **Clinic name, city, doctors, specialties, services, patient names** | `CLINIC CONFIGURATION` block at the top of `app/main.py` — all marked with `# TODO` |
| AI name / behavior | `build_system_prompt()` in `app/main.py` |
| TTS voice | `VOICE = "ro-RO-AlinaNeural"` in `app/main.py` |
| LLM model | `MODEL` in `app/main.py` — `setup.sh` pulls it automatically |
| Greeting message | `_GREETING_VARIANTS` list in `app/main.py` (random per startup) |
| Business hours | `BUSINESS_HOURS` dict in `app/main.py` |
| Calendar ID | `GOOGLE_CALENDAR_ID` env var |
| Server public URL | `BASE_URL` env var |
