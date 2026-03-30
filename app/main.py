import audioop
import base64
import os
import io
import asyncio
import json
import re
import struct
import time as time_mod
import uuid
import wave
from datetime import date as date_type, datetime as dt, timedelta
from urllib.parse import quote

import httpx
import requests as req_lib
import torch  # must be imported before faster_whisper to init CUDA lib paths
import redis
import ollama
import edge_tts
import webrtcvad
from pydub import AudioSegment

from fastapi import FastAPI, UploadFile, File, Form, Request, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from ollama import AsyncClient

from app.appointment_parser import extract_appointment
from app.calendar_service import (
    create_appointment, check_conflict, cancel_appointment,
    get_appointments, list_appointments,
)

app = FastAPI()
stt_model = None
ollama_async = AsyncClient()

# --- TWILIO CONFIG ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
BASE_URL           = os.environ.get("BASE_URL", "https://bx4juhqyhsvr7y-8000.proxy.runpod.net")
GREETING_TEXT      = "Bună ziua, ați ajuns la TestClinic. Cu ce vă pot ajuta?"

# In-memory audio cache: {audio_id: (mp3_bytes, created_at)}
_audio_cache: dict[str, tuple[bytes, float]] = {}

MODEL = "hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M"
VOICE = "ro-RO-AlinaNeural"
SENTENCE_END = re.compile(r'(?<=[.!?])\s')

# --- CONECTARE REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# --- ORE DE PROGRAM ---
# weekday() returns 0=Monday … 6=Sunday; None means closed
BUSINESS_HOURS: dict[int, tuple[str, str] | None] = {
    0: ("09:00", "17:00"),
    1: ("09:00", "17:00"),
    2: ("09:00", "17:00"),
    3: ("09:00", "17:00"),
    4: ("09:00", "17:00"),
    5: None,
    6: None,
}

_DAY_NAMES_RO = {
    0: "luni", 1: "marți", 2: "miercuri", 3: "joi",
    4: "vineri", 5: "sâmbătă", 6: "duminică",
}


def _business_hours_text() -> str:
    lines = []
    for day_idx, hours in BUSINESS_HOURS.items():
        name = _DAY_NAMES_RO[day_idx].capitalize()
        if hours:
            lines.append(f"- {name}: {hours[0]} – {hours[1]}")
        else:
            lines.append(f"- {name}: închis")
    return "\n".join(lines)


def is_within_business_hours(date_str: str, time_str: str) -> bool:
    """Return True if the given date+time falls within configured business hours."""
    try:
        appt_dt = dt.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False
    day_hours = BUSINESS_HOURS.get(appt_dt.weekday())
    if day_hours is None:
        return False
    open_t = dt.strptime(day_hours[0], "%H:%M").time()
    close_t = dt.strptime(day_hours[1], "%H:%M").time()
    return open_t <= appt_dt.time() < close_t


def build_system_prompt() -> str:
    today = date_type.today()
    today_str = today.strftime("%-d %B %Y")  # e.g. "25 martie 2026"
    return f"""
Ești TestRec, recepționera clinicii TestClinic.
Rolul tău principal este să ajuți pacienții să programeze, să anuleze sau să verifice consultații.

Data de astăzi este: {today_str}. Anul curent este {today.year}.
REGULI STRICTE PENTRU AN:
- Dacă pacientul nu specifică anul, folosește ÎNTOTDEAUNA {today.year}. NICIODATĂ alt an (ex: 2024, 2025).
- NU întreba pacientul despre an în nicio situație. Deduce singur: dacă data a trecut deja în {today.year}, folosește {today.year + 1}.
- NU programa sau verifica date din trecut. Dacă data este anterioară față de astăzi ({today_str}), informează pacientul că nu este posibil.

Orarul clinicii:
{_business_hours_text()}
REGULI STRICTE PENTRU ORAR:
- NU accepta programări în afara orelor de program de mai sus.
- Dacă pacientul propune o zi închisă sau o oră în afara programului, informează-l politicos și propune o alternativă.
- NU trimite blocul JSON schedule pentru ore sau zile în afara programului.

Cum să te comporți:
- Vorbește natural, politicos și prietenos în limba română.
- NU cere niciodată numărul de telefon al pacientului — îl avem deja în sistem.
- NU repeta sau explica cum ai dedus data — folosește-o direct în confirmare.
- NU trimite blocul JSON dacă oricare câmp este necunoscut — mai întâi colectează toate informațiile.

REGULI CRITICE DE CONFIRMARE:
- Pentru PROGRAMARE: colectează numele complet, data, ora. Când le ai pe toate, cere confirmare
  într-un mesaj separat (ex: "Confirmați programarea pentru Ion Popescu pe 26 martie la 10:00?").
  Trimite blocul JSON DOAR după ce pacientul răspunde explicit cu "da", "confirm", "corect" etc.
  NICIODATĂ nu trimite blocul JSON în același mesaj în care ceri confirmarea.
- Pentru ANULARE: colectează data și ora. Cere confirmare. Trimite blocul JSON DOAR după "da".
  NICIODATĂ nu trimite blocul JSON în același mesaj în care ceri confirmarea.
- Pentru VERIFICARE: când ai data și ora, trimite IMEDIAT blocul JSON cu action "check".
  Nu mai cere confirmare — este doar o interogare.
- Pentru LISTA PROGRAMĂRI: când pacientul întreabă "ce programare am", "am programări", "când am programare"
  sau orice întrebare despre programările sale fără să specifice o dată anume, trimite IMEDIAT blocul JSON
  cu action "list". Nu cere date suplimentare.

Când ai TOATE informațiile și confirmarea necesară, adaugă UN SINGUR bloc JSON exact în formatele de mai jos.

Format JSON pentru programare (după ce pacientul confirmă):
```json
{{
  "action": "schedule",
  "name": "Numele Pacientului",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Format JSON pentru anulare (după ce pacientul confirmă anularea):
```json
{{
  "action": "cancel",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Format JSON pentru verificare (imediat ce ai data și ora, fără să mai aștepți confirmare):
```json
{{
  "action": "check",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Format JSON pentru lista programărilor pacientului (imediat, fără să ceri alte informații):
```json
{{"action": "list"}}
```

REGULI STRICTE — TREBUIE RESPECTATE ÎNTOTDEAUNA:
1. NU știi ce programări există în calendar. Nu ai acces direct. Nu presupune nimic.
2. Dacă pacientul întreabă dacă există o programare la o anumită dată și oră, NU răspunde din memorie.
   Trebuie să trimiți OBLIGATORIU blocul JSON cu action "check" pentru ca sistemul să verifice.
3. Dacă pacientul întreabă dacă un interval este liber sau ocupat, NU răspunde din memorie.
   Trebuie să trimiți OBLIGATORIU blocul JSON cu action "check".
4. Fără blocul JSON, nicio acțiune nu se execută — nici programare, nici anulare, nici verificare.
5. Nu spune niciodată "este ocupat", "este liber", "am găsit", "nu am găsit" fără să fi primit
   rezultatul verificării din sistem (adică fără să fi trimis blocul check și să fi primit răspuns).
6. Răspunsul natural vine ÎNAINTE de blocul JSON.

Exemplu corect când pacientul întreabă dacă are o programare:
"Verificăm imediat în sistem..."
```json
{{"action": "check", "date": "{(date_type.today() + timedelta(days=7)).strftime('%Y-%m-%d')}", "time": "15:00"}}
```
"""




@app.on_event("startup")
async def load_models():
    global stt_model
    print("Loading Whisper large-v3-turbo...", flush=True)
    stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("Whisper ready.", flush=True)

    print("Warming up Ollama (loading model into VRAM)...", flush=True)
    await run_in_threadpool(
        ollama.chat,
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 1},
    )
    print("Ollama ready.", flush=True)


def get_stt_model():
    return stt_model


async def tts_chunk(text: str) -> bytes:
    """Convert text to a length-prefixed MP3 chunk. Returns b'' if text is empty."""
    text = text.strip()
    if not text:
        return b""
    communicate = edge_tts.Communicate(text, VOICE)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
    if not mp3_data:
        return b""
    return struct.pack('<I', len(mp3_data)) + mp3_data


async def llm_call(messages: list) -> str:
    """Non-streaming LLM call used for action follow-up responses."""
    response = await ollama_async.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": 0.3, "num_ctx": 8192},
    )
    return response["message"]["content"].strip()


async def handle_calendar_action(
    appointment: dict, messages: list, session_id: str
) -> str | None:
    """
    Execute a calendar action and return the text to speak, or None if
    the pre-JSON text already covers the response (schedule/cancel success).
    """
    action = appointment.get("action")
    try:
        if action == "schedule":
            appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
            if appt_date < date_type.today():
                print(f"[main] Data in trecut: {appointment['date']}", flush=True)
                messages.append({"role": "system", "content": (
                    f"Data {appointment['date']} este în trecut. "
                    "Informează pacientul că nu se pot face programări pentru date trecute "
                    "și roagă-l să aleagă o dată viitoare. Nu include niciun bloc JSON în răspuns."
                )})
                reply = await llm_call(messages)
                return reply or "Nu se pot face programări pentru date trecute. Vă rog să alegeți o dată viitoare."

            if not is_within_business_hours(appointment["date"], appointment["time"]):
                day_name = _DAY_NAMES_RO[dt.strptime(appointment["date"], "%Y-%m-%d").weekday()]
                messages.append({"role": "system", "content": (
                    f"Programarea solicitată ({day_name} {appointment['date']} ora {appointment['time']}) "
                    "este în afara orelor de program ale clinicii. "
                    "Informează pacientul politicos și propune o alternativă în orele de program. "
                    "Nu include niciun bloc JSON în răspuns."
                )})
                reply = await llm_call(messages)
                return reply or (
                    f"Îmi pare rău, clinica este închisă {day_name} sau ora {appointment['time']} "
                    "este în afara programului. Vă rog să alegeți o altă zi sau oră."
                )

            conflict = await run_in_threadpool(
                check_conflict, appointment["date"], appointment["time"]
            )
            if conflict:
                print(f"[main] Conflict: {appointment['date']} {appointment['time']}", flush=True)
                messages.append({"role": "system", "content": (
                    f"Intervalul {appointment['time']} din {appointment['date']} este deja ocupat. "
                    "Informează pacientul politicos că acel interval nu este disponibil și propune-i "
                    "să aleagă o altă oră sau zi. Nu include niciun bloc JSON în răspuns."
                )})
                reply = await llm_call(messages)
                return reply or (
                    f"Îmi pare rău, intervalul {appointment['time']} din {appointment['date']} "
                    "este ocupat. Doriți să alegeți altă oră?"
                )
            event_link = await run_in_threadpool(
                create_appointment,
                appointment["name"], appointment["date"], appointment["time"], session_id,
            )
            print(f"[main] Programare creata: {event_link}", flush=True)
            return None  # pre-JSON confirmation text was already streamed

        elif action == "cancel":
            appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
            if appt_date < date_type.today():
                return (
                    f"Data de {appointment['date']} este în trecut. "
                    "Nu există programări active pentru date trecute."
                )
            deleted = await run_in_threadpool(
                cancel_appointment, appointment["date"], appointment["time"], session_id
            )
            if not deleted:
                print(f"[main] Anulare: nimic gasit pentru {appointment['date']} {appointment['time']}", flush=True)
                messages.append({"role": "system", "content": (
                    f"Nu a fost găsită nicio programare pe data de {appointment['date']} "
                    f"la ora {appointment['time']}. "
                    "Informează pacientul politicos și întreabă dacă dorește să verifice altă dată sau oră. "
                    "Nu include niciun bloc JSON în răspuns."
                )})
                reply = await llm_call(messages)
                return reply or (
                    f"Nu am găsit nicio programare pe {appointment['date']} la {appointment['time']}."
                )
            print(f"[main] Programare anulata: {appointment['date']} {appointment['time']}", flush=True)
            return None  # pre-JSON confirmation text was already streamed

        elif action == "check":
            appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
            if appt_date < date_type.today():
                return (
                    f"Data de {appointment['date']} este în trecut. "
                    "Puteți verifica doar programări viitoare."
                )
            events = await run_in_threadpool(
                get_appointments, appointment["date"], appointment["time"]
            )
            if events:
                result_msg = (
                    f"Rezultat verificare din sistem: există o programare pe {appointment['date']} "
                    f"la ora {appointment['time']}: {', '.join(events)}. "
                    "Informează pacientul și întreabă dacă dorește să o anuleze sau dacă mai are alte întrebări. "
                    "Nu include niciun bloc JSON în răspuns."
                )
            else:
                result_msg = (
                    f"Rezultat verificare din sistem: nu există nicio programare pe {appointment['date']} "
                    f"la ora {appointment['time']}. "
                    "Informează pacientul și întreabă dacă dorește să programeze o consultație. "
                    "Nu include niciun bloc JSON în răspuns."
                )
            print(f"[main] Check result: {result_msg}", flush=True)
            messages.append({"role": "system", "content": result_msg})
            reply = await llm_call(messages)
            return reply or ("Există o programare." if events else "Nu există nicio programare.")

        elif action == "list":
            appts = await run_in_threadpool(list_appointments, session_id)
            if appts:
                appt_lines = "\n".join(f"- {a['summary']} la {a['start']}" for a in appts)
                result_msg = (
                    f"Rezultat din sistem: pacientul are următoarele programări viitoare:\n{appt_lines}\n"
                    "Informează pacientul politicos și întreabă dacă dorește să modifice ceva. "
                    "Nu include niciun bloc JSON în răspuns."
                )
            else:
                result_msg = (
                    "Rezultat din sistem: pacientul nu are nicio programare viitoare înregistrată. "
                    "Informează pacientul și întreabă dacă dorește să programeze o consultație. "
                    "Nu include niciun bloc JSON în răspuns."
                )
            print(f"[main] List result: {result_msg}", flush=True)
            messages.append({"role": "system", "content": result_msg})
            reply = await llm_call(messages)
            return reply or (
                "Aveți programările: ..." if appts else "Nu aveți nicio programare viitoare înregistrată."
            )

    except Exception as e:
        print(f"[main] Eroare la acțiune: {e}", flush=True)

    return None


async def voice_stream(session_id: str, audio_buffer: io.BytesIO):
    """
    Main pipeline as an async generator:
      STT → streaming LLM → per-sentence TTS → yield length-prefixed MP3 chunks

    Protocol:
      Each chunk  = 4-byte little-endian uint32 (length) + <length> bytes of MP3
      End marker  = 4 zero bytes
    """
    try:
        async for chunk in _voice_stream_inner(session_id, audio_buffer):
            yield chunk
    except Exception as e:
        print(f"[main] Eroare neasteptata [{session_id}]: {e}", flush=True)
        fallback = await tts_chunk("Îmi pare rău, a apărut o eroare tehnică. Vă rugăm să sunați din nou.")
        if fallback:
            yield fallback
        yield struct.pack('<I', 0)


async def _voice_stream_inner(session_id: str, audio_buffer: io.BytesIO):
    # --- STT ---
    segments, _ = await run_in_threadpool(
        get_stt_model().transcribe, audio_buffer, language="ro", vad_filter=True
    )
    user_text = " ".join([s.text for s in segments]).strip()
    print(f"User [{session_id}]: {user_text}", flush=True)

    if not user_text:
        audio = await tts_chunk("Nu am înțeles. Vă rog să repetați.")
        if audio:
            yield audio
        yield struct.pack('<I', 0)
        return

    # --- Load session history ---
    history_json = redis_client.get(session_id)
    if history_json:
        messages = json.loads(history_json)
        print(f"[redis] Loaded {len(messages)} messages for {session_id}", flush=True)
    else:
        messages = [{"role": "system", "content": build_system_prompt()}]
        print(f"[redis] New session for {session_id}", flush=True)
    messages.append({"role": "user", "content": user_text})

    # --- Stream LLM, TTS each sentence as it completes ---
    full_response = ""
    sentence_buffer = ""
    json_detected = False

    async for chunk in await ollama_async.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_ctx": 8192},
    ):
        token = chunk["message"]["content"]
        full_response += token

        if json_detected:
            continue  # keep consuming to get the full response; no more TTS

        # Detect start of JSON action block
        if "```json" in full_response or "```\n{" in full_response:
            json_detected = True
            # Speak whatever was buffered before the backticks
            pre_json = sentence_buffer.split("```")[0].strip()
            audio = await tts_chunk(pre_json)
            if audio:
                yield audio
            sentence_buffer = ""
            continue

        sentence_buffer += token

        # Yield a chunk whenever a sentence boundary is reached
        m = SENTENCE_END.search(sentence_buffer)
        if m:
            to_speak = sentence_buffer[:m.end()].strip()
            sentence_buffer = sentence_buffer[m.end():]
            audio = await tts_chunk(to_speak)
            if audio:
                yield audio

    # Yield any remaining text (no JSON block in this turn)
    if not json_detected and sentence_buffer.strip():
        audio = await tts_chunk(sentence_buffer.strip())
        if audio:
            yield audio

    ai_reply = full_response.strip()
    print(f"AI [{session_id}]: {ai_reply}", flush=True)

    # --- Handle calendar action if present ---
    _, appointment = extract_appointment(ai_reply)
    if appointment:
        action_text = await handle_calendar_action(appointment, messages, session_id)
        if action_text:
            # Stream the action response sentence by sentence
            for sentence in [s.strip() for s in SENTENCE_END.split(action_text) if s.strip()]:
                audio = await tts_chunk(sentence)
                if audio:
                    yield audio

    # --- Persist conversation to Redis ---
    messages.append({"role": "assistant", "content": ai_reply})
    redis_client.setex(session_id, 600, json.dumps(messages))

    # End-of-stream marker
    yield struct.pack('<I', 0)


async def _tts_raw(text: str) -> bytes:
    """Return raw MP3 bytes (no length prefix)."""
    chunk = await tts_chunk(text)
    return chunk[4:] if len(chunk) > 4 else b""


def _store_audio(mp3_bytes: bytes) -> str:
    """Cache MP3 bytes, evict entries older than 5 minutes, return cache key."""
    now = time_mod.time()
    stale = [k for k, (_, ts) in _audio_cache.items() if now - ts > 300 and k != "greeting"]
    for k in stale:
        del _audio_cache[k]
    audio_id = str(uuid.uuid4())
    _audio_cache[audio_id] = (mp3_bytes, now)
    return audio_id


async def voice_to_mp3(session_id: str, audio_buffer: io.BytesIO) -> bytes:
    """Run the full voice pipeline and return concatenated raw MP3 bytes."""
    mp3_parts = []
    async for chunk in _voice_stream_inner(session_id, audio_buffer):
        if len(chunk) > 4:           # skip end marker (4 zero bytes)
            mp3_parts.append(chunk[4:])   # strip 4-byte length prefix
    return b"".join(mp3_parts)


def _build_wav_from_mulaw(mulaw_data: bytes) -> io.BytesIO:
    """Convert raw mulaw 8kHz bytes → WAV BytesIO at 16kHz (for Whisper)."""
    pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_16k)
    buf.seek(0)
    return buf


def _mp3_to_mulaw_chunks(mp3_bytes: bytes) -> list[bytes]:
    """Decode MP3 → mulaw 8kHz mono, return list of 160-byte chunks."""
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    mulaw = audioop.lin2ulaw(seg.raw_data, 2)
    chunks = []
    for i in range(0, len(mulaw), 160):
        frame = mulaw[i:i + 160]
        if len(frame) < 160:
            frame = frame + bytes(160 - len(frame))
        chunks.append(frame)
    return chunks


@app.get("/audio/{audio_id}")
async def serve_audio(audio_id: str):
    entry = _audio_cache.get(audio_id)
    if not entry:
        return Response(status_code=404)
    mp3 = entry[0]
    return Response(content=mp3, media_type="audio/mpeg", headers={"Content-Length": str(len(mp3))})


@app.post("/twilio/incoming")
async def twilio_incoming():
    """Answer inbound call and connect Twilio Media Streams WebSocket."""
    wss_url = BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{wss_url}/twilio/stream" />
    </Connect>
</Response>"""
    return Response(content=xml, media_type="text/xml")


@app.post("/twilio/process")
async def twilio_process(request: Request):
    """Twilio posts here after recording. Download audio, run pipeline, return TwiML."""
    form = await request.form()
    recording_url = str(form.get("RecordingUrl", ""))
    call_sid      = str(form.get("CallSid", "unknown"))

    print(f"[twilio] Call {call_sid}, recording: {recording_url}", flush=True)
    print(f"[twilio] Auth SID={TWILIO_ACCOUNT_SID[:8]}... Token={'set' if TWILIO_AUTH_TOKEN else 'EMPTY'}", flush=True)

    try:
        def _download_wav() -> bytes:
            r = req_lib.get(
                f"{recording_url}.wav",
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                timeout=10,
            )
            r.raise_for_status()
            return r.content

        try:
            wav_bytes = await run_in_threadpool(_download_wav)
        except req_lib.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"[twilio] Recording not found (caller hung up): {call_sid}", flush=True)
                return Response(content="<?xml version='1.0'?><Response></Response>", media_type="text/xml")
            raise
        mp3_bytes = await voice_to_mp3(call_sid, io.BytesIO(wav_bytes))
        print(f"[twilio] MP3 size: {len(mp3_bytes)} bytes", flush=True)

        if not mp3_bytes:
            raise RuntimeError("TTS returned empty audio")

        audio_id  = _store_audio(mp3_bytes)
        audio_url = f"{BASE_URL}/audio/{audio_id}"
        print(f"[twilio] Audio URL: {audio_url}", flush=True)

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Record action="{BASE_URL}/twilio/process" method="POST" maxLength="30" timeout="3" playBeep="false" />
</Response>"""
    except Exception as e:
        print(f"[twilio] Eroare procesare {call_sid}: {e}", flush=True)
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ro-RO">Îmi pare rău, a apărut o eroare. Vă rugăm să sunați din nou.</Say>
</Response>"""

    return Response(content=xml, media_type="text/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """
    Twilio Media Streams WebSocket handler.

    Twilio streams mulaw 8kHz audio (160 bytes = 20ms per chunk, base64-encoded).
    We run VAD to detect end-of-speech, then pipe through STT → LLM → TTS,
    convert the response to mulaw 8kHz, and send it back.
    """
    await ws.accept()

    stream_sid: str = ""
    call_sid: str = ""

    # VAD settings
    VAD_AGGRESSIVENESS = 2    # 0=least strict, 3=most strict
    SILENCE_FRAMES     = 25   # 25 × 20ms = 500ms silence → end of speech
    MIN_SPEECH_FRAMES  = 5    # ignore utterances shorter than 100ms

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    queue: asyncio.Queue = asyncio.Queue()

    async def send_mulaw(mulaw_chunks: list[bytes]):
        """Send mulaw chunks at real-time pace (20ms/chunk) so callers hear smooth audio."""
        for chunk in mulaw_chunks:
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode()},
            }))
            await asyncio.sleep(0.02)  # 160 bytes = 20ms of audio at 8kHz

    async def receive_loop():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                ev = msg.get("event")
                if ev == "start":
                    await queue.put(("start", msg))
                elif ev == "media":
                    mulaw = base64.b64decode(msg["media"]["payload"])
                    await queue.put(("media", mulaw))
                elif ev == "stop":
                    await queue.put(("stop", None))
                    break
        except (WebSocketDisconnect, Exception) as e:
            print(f"[stream] receive_loop exit: {type(e).__name__}", flush=True)
        finally:
            await queue.put(("stop", None))

    async def process_loop():
        nonlocal stream_sid, call_sid

        speech_frames: list[bytes] = []  # accumulated mulaw frames
        silence_count = 0
        in_speech = False
        is_processing = False  # True while STT→LLM→TTS is running

        async def handle_utterance(frames: list[bytes]):
            nonlocal is_processing
            try:
                mulaw_data = b"".join(frames)
                wav_buf = await run_in_threadpool(_build_wav_from_mulaw, mulaw_data)
                end_marker = struct.pack("<I", 0)
                async for chunk in _voice_stream_inner(call_sid, wav_buf):
                    if chunk == end_marker:
                        break
                    if len(chunk) > 4:
                        mp3_data = chunk[4:]  # strip 4-byte length prefix
                        mulaw_chunks = await run_in_threadpool(_mp3_to_mulaw_chunks, mp3_data)
                        await send_mulaw(mulaw_chunks)
            except Exception as e:
                print(f"[stream] handle_utterance error [{call_sid}]: {e}", flush=True)
            finally:
                # Keep is_processing=True during cooldown so VAD stays off while
                # the caller's phone echo decays (audio still playing + ~1s buffer)
                await asyncio.sleep(1.0)
                is_processing = False
                # Drain audio buffered while we were processing + cooldown
                drained = 0
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    print(f"[stream] Drained {drained} queued frames after processing", flush=True)

        while True:
            try:
                ev, data = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                print(f"[stream] Timeout on call {call_sid}", flush=True)
                break

            if ev == "stop":
                break

            elif ev == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid   = data["start"]["callSid"]
                print(f"[stream] Call started: {call_sid}", flush=True)
                # Send greeting (real-time paced via send_mulaw)
                try:
                    if "greeting" not in _audio_cache:
                        mp3 = await _tts_raw(GREETING_TEXT)
                        _audio_cache["greeting"] = (mp3, time_mod.time())
                    greeting_chunks = await run_in_threadpool(
                        _mp3_to_mulaw_chunks, _audio_cache["greeting"][0]
                    )
                    await send_mulaw(greeting_chunks)
                    # Cooldown: let caller's phone echo decay before VAD opens
                    await asyncio.sleep(1.0)
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception as e:
                    print(f"[stream] Greeting error: {e}", flush=True)

            elif ev == "media":
                if is_processing:
                    continue  # drain audio during pipeline run

                mulaw_frame: bytes = data
                if len(mulaw_frame) != 160:
                    continue  # webrtcvad requires exactly 160 mulaw bytes (20ms@8kHz)

                pcm_frame = audioop.ulaw2lin(mulaw_frame, 2)  # → 320 bytes PCM

                try:
                    is_speech = vad.is_speech(pcm_frame, 8000)
                except Exception:
                    # Fallback to energy-based detection
                    is_speech = audioop.rms(pcm_frame, 2) > 300

                if is_speech:
                    speech_frames.append(mulaw_frame)
                    silence_count = 0
                    in_speech = True
                elif in_speech:
                    speech_frames.append(mulaw_frame)  # keep trailing silence
                    silence_count += 1
                    if silence_count >= SILENCE_FRAMES:
                        if len(speech_frames) >= MIN_SPEECH_FRAMES:
                            print(
                                f"[stream] Utterance detected: "
                                f"{len(speech_frames)} frames ({len(speech_frames) * 20}ms)",
                                flush=True,
                            )
                            is_processing = True
                            frames_copy = list(speech_frames)
                            asyncio.create_task(handle_utterance(frames_copy))
                        speech_frames = []
                        silence_count = 0
                        in_speech = False

        print(f"[stream] Call ended: {call_sid}", flush=True)

    try:
        await asyncio.gather(receive_loop(), process_loop())
    except Exception as e:
        print(f"[stream] Fatal error: {e}", flush=True)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.post("/voice")
async def voice_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...),  # patient phone number (e.g. +40721000000)
):
    content = await file.read()
    audio_buffer = io.BytesIO(content)
    return StreamingResponse(
        voice_stream(session_id, audio_buffer),
        media_type="application/octet-stream",
    )
