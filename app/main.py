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

import numpy as np
import httpx
import requests as req_lib
import torch  # must be imported before faster_whisper to init CUDA lib paths
import redis
import ollama
import edge_tts
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
silero_vad_model = None   # loaded at startup; replaces webrtcvad
ollama_async = AsyncClient()
_greeting_mulaw: list[bytes] = []   # pre-generated at startup, ready for first call

# Sentinel yielded by _voice_stream_inner to signal the caller should be hung up
HANGUP_SIGNAL = "__HANGUP__"
HANGUP_MARKER = struct.pack('<I', 0xFFFFFFFF)  # 4-byte sentinel, distinct from end marker

# --- TWILIO CONFIG ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
BASE_URL           = os.environ.get("BASE_URL", "https://5ipahenri81624-8000.proxy.runpod.net")

_GREETING_VARIANTS = [
    "Bună ziua, ați ajuns la TestClinic. Cu ce vă pot ajuta?",
    "Bună ziua! TestClinic, vă ascult.",
    "Bună ziua, aici TestClinic. Cum vă pot fi de folos?",
    "TestClinic, bună ziua! Cu ce vă pot ajuta?",
]
GREETING_TEXT      = __import__('random').choice(_GREETING_VARIANTS)

# In-memory audio cache: {audio_id: (mp3_bytes, created_at)}
_audio_cache: dict[str, tuple[bytes, float]] = {}

MODEL = "qwen3.5:27b-q4_K_M"
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
    return f"""Ești Sara, recepționera clinicii TestClinic. Ești caldă, eficientă și vorbești ca un om real — nu ca un robot. Ai lucrat la recepție ani de zile și știi cum să pui pacienții la ușurință.

Data de azi: {today_str}. Anul curent: {today.year}.

## TON ȘI STIL — CELE MAI IMPORTANTE REGULI

Vorbești ca o recepționeră experimentată, nu ca un sistem automat. Asta înseamnă:
- Răspunsuri scurte și naturale. Nu explica mai mult decât e necesar.
- Folosești cuvinte de legătură românești firești: "sigur", "bineînțeles", "desigur", "cu plăcere", "înțeleg", "păi", "deci", "hai", "ok, deci".
- Variezi formulările — nu spui niciodată același lucru de două ori la rând.
- NU începi niciodată cu "Bună ziua" după primul mesaj — salutul a fost deja făcut.
- NU repeta ce a spus pacientul înainte să răspunzi (ex: nu "Ați spus că doriți...").
- NU folosești fraze robotice ca "Cu ce vă pot ajuta astăzi?", "Vă mulțumesc pentru răbdare", "Desigur, voi procesa solicitarea dumneavoastră".

## ADAPTARE LA STAREA PACIENTULUI

Citești tonul din mesajul pacientului și te adaptezi:
- Pacient grăbit (mesaje scurte, directe) → fii concisă, nu bate câmpii.
- Pacient anxios sau nesigur → fii mai caldă, liniștitoare: "Nu vă faceți griji, rezolvăm noi."
- Pacient confuz → simplifică, o întrebare la un moment dat.
- Pacient mulțumit, relaxat → poți fi mai prietenoasă, mai informală.
- Dacă ceva nu merge (conflict de orar, eroare) → recunoaște cu empatie: "Îmi pare rău, se pare că acel interval e ocupat..."

## COLECTARE INFORMAȚII

Când ai nevoie de date (nume, dată, oră), întrebi natural, câte una:
- "Și pe ce nume facem programarea?"
- "La ce dată vă gândeați?"
- "La ce oră vă convine?"
Nu întreba mai mult de un lucru odată.

## REGULI PENTRU DATE ȘI ANI

- Dacă pacientul nu specifică anul, folosești ÎNTOTDEAUNA {today.year}. Niciodată 2024 sau 2025.
- Nu întreba niciodată pacientul despre an. Dacă data a trecut în {today.year}, folosești {today.year + 1}.
- Nu programa sau verifica date din trecut. Dacă e o dată anterioară față de {today_str}, spui că nu e posibil.
- NU repeta sau explica cum ai dedus data — o folosești direct în confirmare.

## ORARUL CLINICII

{_business_hours_text()}

- NU accepta programări în afara acestor ore.
- Dacă pacientul propune o zi închisă sau oră nepotrivită, spui politicos și propui o alternativă concretă.
- NU trimite blocul JSON schedule în afara programului.

## CONFIRMARE ÎNAINTE DE ACȚIUNE

- PROGRAMARE: colectezi numele complet, data, ora. Când le ai pe toate, ceri confirmare într-un singur mesaj clar (ex: "Deci facem pe Ion Popescu, marți 26 martie la 10:00 — confirmați?"). Trimiți JSON DOAR după "da" / "confirm" / "corect". NICIODATĂ în același mesaj cu întrebarea de confirmare.
- ANULARE: colectezi data și ora. Ceri confirmare. Trimiți JSON DOAR după "da". Nu în același mesaj.
- VERIFICARE: când ai data și ora, trimiți IMEDIAT blocul JSON check. Nu ceri confirmare — e doar o căutare.
- LISTA: dacă pacientul întreabă "ce programări am", "când am programare" etc. fără să specifice o dată, trimiți IMEDIAT blocul JSON list.

Nu cere niciodată numărul de telefon — îl avem deja în sistem.

## REGULI CRITICE — CALENDAR

1. Nu știi ce există în calendar. Nu presupune nimic.
2. Dacă pacientul întreabă de o programare sau dacă un interval e liber, trimiți OBLIGATORIU blocul JSON check — nu răspunzi din memorie.
3. Fără JSON, nicio acțiune nu se execută.
4. Nu spune "este ocupat / liber / am găsit / nu am găsit" fără să fi primit rezultatul din sistem.
5. Răspunsul natural vine ÎNAINTE de blocul JSON.

## FORMATUL JSON

Programare (după confirmare):
```json
{{
  "action": "schedule",
  "name": "Numele Pacientului",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Anulare (după confirmare):
```json
{{
  "action": "cancel",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Verificare (imediat, fără confirmare):
```json
{{
  "action": "check",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}}
```

Lista programărilor (imediat):
```json
{{"action": "list"}}
```

Încheierea convorbirii (DOAR când pacientul și-a luat rămas bun):
```json
{{"action": "hangup"}}
```
Mesajul de rămas bun vine ÎNAINTE de JSON. Nu încheia înainte să fi rezolvat solicitarea.

## EXEMPLE DE DIALOG NATURAL

Exemplu 1 — programare:
Pacient: "Bună ziua, aș vrea să fac o programare."
Sara:"Sigur! Pe ce nume facem programarea?"
Pacient: "Maria Ionescu."
Sara:"Și la ce dată vă gândeați?"
Pacient: "Joi, 10 aprilie."
Sara:"La ce oră vă convine?"
Pacient: "La 11."
Sara:"Deci facem pe Maria Ionescu, joi 10 aprilie la 11:00 — confirmați?"
Pacient: "Da."
Sara:"Perfect, am înregistrat programarea! Mai pot face ceva pentru dumneavoastră?"

Exemplu 2 — pacient grăbit:
Pacient: "Vreau să anulez programarea de mâine la 9."
Sara:"Sigur. Anulăm programarea de mâine, {(date_type.today() + timedelta(days=1)).strftime('%-d %B')}, la 9:00 — confirmați?"
Pacient: "Da."
Sara:"Gata, am anulat."
"""




@app.on_event("startup")
async def load_models():
    global stt_model, silero_vad_model
    print("Loading Whisper large-v3-turbo...", flush=True)
    stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("Whisper ready.", flush=True)

    print("Loading Silero VAD...", flush=True)
    try:
        _model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            onnx=True,   # use onnxruntime backend — avoids NNPACK warnings on VMs
        )
        silero_vad_model = _model
        print("Silero VAD ready.", flush=True)
    except Exception as e:
        print(f"Silero VAD load failed ({e}); will fall back to energy-based detection.", flush=True)

    print("Warming up Ollama (loading model into VRAM)...", flush=True)
    await run_in_threadpool(
        ollama.chat,
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 1, "think": False},
    )
    print("Ollama ready.", flush=True)

    print("Pre-generating greeting audio...", flush=True)
    global _greeting_mulaw
    mp3 = await _tts_raw(GREETING_TEXT)
    _audio_cache["greeting"] = (mp3, time_mod.time())
    _greeting_mulaw = await run_in_threadpool(_mp3_to_mulaw_chunks, mp3)
    print("Greeting ready.", flush=True)


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
        options={"temperature": 0.3, "num_ctx": 8192, "think": False},
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

        elif action == "hangup":
            print(f"[main] Hangup requested for {session_id}", flush=True)
            return HANGUP_SIGNAL

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
        get_stt_model().transcribe,
        audio_buffer,
        language="ro",
        vad_filter=True,
        initial_prompt=(
            "Consultație medicală. Programare la clinică. "
            "Prenume și nume de familie. Pacient. Doctor. "
            "Data și ora programării. Confirmare. Anulare."
        ),
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
    pre_json_audio = b""  # buffered until after conflict check

    in_think_block = False  # strip <think>...</think> from Qwen3 thinking mode

    async for chunk in await ollama_async.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_ctx": 8192, "think": False},
    ):
        token = chunk["message"]["content"]
        full_response += token

        # Strip Qwen3 thinking blocks — they must never reach TTS
        if "<think>" in token:
            in_think_block = True
        if in_think_block:
            if "</think>" in token:
                in_think_block = False
            continue

        if json_detected:
            continue  # keep consuming to get the full response; no more TTS

        # Detect start of JSON action block (case-insensitive; also catches
        # raw JSON without backticks in case the LLM omits the fence)
        _fl = full_response.lower()
        if "```json" in _fl or "```\n{" in full_response or '{"action"' in full_response:
            json_detected = True
            # Buffer (don't yield yet) the text before the backticks —
            # we need to check the calendar action first; if there's a conflict
            # this confirmation text must be suppressed.
            pre_json_text = sentence_buffer.split("```")[0].split('{"action"')[0].strip()
            pre_json_audio = await tts_chunk(pre_json_text)
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

    # Yield any remaining text (no JSON block in this turn).
    # Safety-strip any JSON that slipped past the streaming detector.
    if not json_detected and sentence_buffer.strip():
        safe_text = re.sub(r'`{3}.*?`{3}|\{"action".*', '', sentence_buffer,
                           flags=re.DOTALL | re.IGNORECASE).strip()
        if safe_text:
            audio = await tts_chunk(safe_text)
            if audio:
                yield audio

    ai_reply = full_response.strip()
    print(f"AI [{session_id}]: {ai_reply}", flush=True)

    # --- Handle calendar action if present ---
    _, appointment = extract_appointment(ai_reply)
    if appointment:
        action_text = await handle_calendar_action(appointment, messages, session_id)
        if action_text == HANGUP_SIGNAL:
            # Speak the goodbye (pre-JSON text), then signal the stream to hang up.
            if pre_json_audio:
                yield pre_json_audio
            yield HANGUP_MARKER
        elif action_text:
            # Calendar returned an override (conflict, not found, check result, etc.)
            # Discard pre_json_audio — the "confirmed" text was premature.
            for sentence in [s.strip() for s in SENTENCE_END.split(action_text) if s.strip()]:
                audio = await tts_chunk(sentence)
                if audio:
                    yield audio
        else:
            # Success — the pre-JSON confirmation text is correct, speak it now.
            if pre_json_audio:
                yield pre_json_audio

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
    """Convert raw mulaw 8kHz bytes → WAV BytesIO at 16kHz (for Whisper).

    Uses numpy linear interpolation for upsampling — better quality than audioop.ratecv
    while avoiding a scipy dependency.
    """
    pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
    samples_8k = np.frombuffer(pcm_8k, dtype=np.int16).astype(np.float32)
    # Upsample 8kHz → 16kHz via linear interpolation
    n_out = len(samples_8k) * 2
    x_in  = np.arange(len(samples_8k))
    x_out = np.linspace(0, len(samples_8k) - 1, n_out)
    samples_16k = np.interp(x_out, x_in, samples_8k)
    # Normalize amplitude so Whisper gets a consistent signal level
    peak = np.abs(samples_16k).max()
    if peak > 0:
        samples_16k = samples_16k / peak * 32767 * 0.9
    samples_16k = samples_16k.astype(np.int16)
    pcm_16k = samples_16k.tobytes()
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
async def twilio_incoming(request: Request):
    """Answer inbound call and connect Twilio Media Streams WebSocket."""
    form = await request.form()
    caller_phone = str(form.get("From", "")) or "unknown"
    wss_url = BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{wss_url}/twilio/stream">
            <Parameter name="caller_phone" value="{caller_phone}"/>
        </Stream>
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
    caller_phone: str = ""   # set from Twilio start event; used as Redis session key

    # VAD / speech detection settings
    SILENCE_FRAMES       = 25    # 25 × 20ms = 500ms silence → end of speech
    MIN_SPEECH_FRAMES    = 15    # ignore utterances shorter than 300ms (noise/echo filter)
    PRE_SPEECH_FRAMES    = 10    # frames to prepend before first voiced frame (word onset)
    SILERO_THRESHOLD     = 0.5   # Silero speech confidence threshold (0–1)
    SILERO_CHUNK_SAMPLES = 256   # 256 samples @ 8kHz = 32ms per Silero inference
    BARGE_IN_THRESHOLD   = 0.7   # higher confidence required to interrupt AI speech
    BARGE_IN_FRAMES      = 3     # consecutive voiced Silero chunks (~96ms) to confirm barge-in

    queue: asyncio.Queue = asyncio.Queue()

    async def send_mulaw(mulaw_chunks: list[bytes]) -> float:
        """Send mulaw chunks as fast as possible. Returns audio duration in seconds."""
        for chunk in mulaw_chunks:
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode()},
            }))
        # mulaw is 1 byte per sample at 8kHz → duration = bytes / 8000
        return sum(len(c) for c in mulaw_chunks) / 8000

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
        nonlocal stream_sid, call_sid, caller_phone

        speech_frames: list[bytes] = []  # accumulated mulaw frames for current utterance
        pre_speech: list[bytes] = []     # rolling pre-speech buffer (captures word onset)
        silero_buf = b""                 # PCM int16 accumulator for Silero chunks
        last_is_speech = False           # last Silero result, reused while accumulating
        silence_count = 0
        in_speech = False
        is_processing = False            # True while STT→LLM→TTS is running
        current_utterance_task: asyncio.Task | None = None
        barge_in_count = 0               # consecutive Silero-voiced chunks during processing
        barge_in_armed_at: float = 0.0   # time.monotonic() when barge-in was armed
        BARGE_IN_GRACE = 1.5             # seconds after processing starts before barge-in arms
                                          # (prevents noise/echo immediately after user speech)

        async def handle_utterance(frames: list[bytes]):
            nonlocal is_processing, barge_in_armed_at
            total_audio_secs = 0.0
            should_hangup = False
            was_cancelled = False
            first_audio_sent = False
            session_id = caller_phone  # phone number → memory persists across calls
            end_marker = struct.pack("<I", 0)
            try:
                mulaw_data = b"".join(frames)
                wav_buf = await run_in_threadpool(_build_wav_from_mulaw, mulaw_data)
                async for chunk in _voice_stream_inner(session_id, wav_buf):
                    if chunk == end_marker:
                        break
                    if chunk == HANGUP_MARKER:
                        should_hangup = True
                        break
                    if len(chunk) > 4:
                        mp3_data = chunk[4:]  # strip 4-byte length prefix
                        mulaw_chunks = await run_in_threadpool(_mp3_to_mulaw_chunks, mp3_data)
                        total_audio_secs += await send_mulaw(mulaw_chunks)
                        if not first_audio_sent:
                            first_audio_sent = True
                            # Re-arm grace period from when AI audio actually starts playing,
                            # not from when STT/LLM processing began. This prevents false
                            # barge-in triggers during the STT+LLM latency window.
                            barge_in_armed_at = time_mod.monotonic() + BARGE_IN_GRACE
            except asyncio.CancelledError:
                was_cancelled = True
                print(f"[stream] handle_utterance cancelled (barge-in) [{call_sid}]", flush=True)
                raise  # propagate so the task is properly marked cancelled
            except Exception as e:
                print(f"[stream] handle_utterance error [{call_sid}]: {e}", flush=True)
            finally:
                if not was_cancelled:
                    # Normal completion: wait for audio to finish + 0.4s echo decay,
                    # then drain any audio queued during playback.
                    await asyncio.sleep(total_audio_secs + 0.4)
                    is_processing = False
                    drained = 0
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        print(f"[stream] Drained {drained} queued frames after processing", flush=True)
                # On barge-in cancellation, process_loop already set is_processing=False
                # and started collecting speech — no sleep or drain needed here.
                if should_hangup:
                    print(f"[stream] Hanging up {call_sid}", flush=True)
                    try:
                        await ws.close()
                    except Exception:
                        pass

        while True:
            try:
                ev, data = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                print(f"[stream] Timeout on call {call_sid}", flush=True)
                break

            if ev == "stop":
                break

            elif ev == "start":
                stream_sid   = data["start"]["streamSid"]
                call_sid     = data["start"]["callSid"]
                caller_phone = data["start"].get("customParameters", {}).get("caller_phone", "") or call_sid
                print(f"[stream] Call started: {call_sid}, session: {caller_phone}", flush=True)
                # Send greeting — use pre-generated mulaw chunks (zero latency)
                try:
                    greeting_chunks = _greeting_mulaw
                    greeting_secs = await send_mulaw(greeting_chunks)
                    # Wait for greeting to finish playing + 0.4s echo decay
                    await asyncio.sleep(greeting_secs + 0.4)
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception as e:
                    print(f"[stream] Greeting error: {e}", flush=True)

            elif ev == "media":
                mulaw_frame: bytes = data
                if len(mulaw_frame) != 160:
                    continue  # Twilio sends exactly 160 mulaw bytes (20ms @ 8kHz)

                # Convert mulaw → PCM before the is_processing check so both
                # the barge-in path and the normal VAD path can use pcm_frame.
                pcm_frame = audioop.ulaw2lin(mulaw_frame, 2)  # → 320 bytes PCM

                if is_processing:
                    # Keep pre-speech ring buffer fresh so caller's first words
                    # aren't cut off when they speak right after the AI finishes.
                    pre_speech.append(mulaw_frame)
                    if len(pre_speech) > PRE_SPEECH_FRAMES:
                        pre_speech.pop(0)

                    # --- Barge-in detection ---
                    # Run Silero with a higher threshold while AI is speaking so we
                    # don't mistake echo or background noise for a barge-in.
                    # Grace period: skip detection until AI has had time to start playing.
                    silero_buf += pcm_frame
                    if time_mod.monotonic() < barge_in_armed_at:
                        continue  # still in grace period — discard frame and move on
                    if len(silero_buf) >= SILERO_CHUNK_SAMPLES * 2:
                        chunk_pcm = silero_buf[:SILERO_CHUNK_SAMPLES * 2]
                        silero_buf = silero_buf[SILERO_CHUNK_SAMPLES * 2:]
                        voiced = False
                        try:
                            if silero_vad_model is not None:
                                pcm_np = np.frombuffer(chunk_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                                audio_t = torch.from_numpy(pcm_np)
                                with torch.no_grad():
                                    voiced = silero_vad_model(audio_t, 8000).item() > BARGE_IN_THRESHOLD
                            else:
                                voiced = audioop.rms(chunk_pcm, 2) > 600
                        except Exception:
                            pass

                        if voiced:
                            barge_in_count += 1
                            if barge_in_count >= BARGE_IN_FRAMES:
                                print(f"[stream] Barge-in detected ({barge_in_count} frames) [{call_sid}]", flush=True)
                                # Cancel in-flight response task
                                if current_utterance_task and not current_utterance_task.done():
                                    current_utterance_task.cancel()
                                # Tell Twilio to stop playing the current audio
                                try:
                                    await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
                                except Exception:
                                    pass
                                # Transition immediately to speech collection
                                is_processing = False
                                in_speech = True
                                speech_frames = list(pre_speech)
                                silence_count = 0
                                barge_in_count = 0
                                silero_buf = b""
                                last_is_speech = True
                                # Fall through to normal VAD logic below with in_speech=True
                        else:
                            barge_in_count = 0

                    if is_processing:
                        continue  # still in processing and no barge-in yet
                    # Barge-in triggered: is_processing=False, in_speech=True,
                    # speech_frames seeded with pre_speech — fall through to speech
                    # accumulation below with is_speech=True already set.
                    is_speech = True
                else:
                    # --- Silero VAD (normal path — not processing) ---
                    # Accumulate PCM until we have a full 256-sample chunk (32ms @ 8kHz),
                    # then run the neural VAD. Reuse the last result between chunks.
                    silero_buf += pcm_frame
                    if len(silero_buf) >= SILERO_CHUNK_SAMPLES * 2:  # *2: int16 = 2 bytes/sample
                        chunk_pcm = silero_buf[:SILERO_CHUNK_SAMPLES * 2]
                        silero_buf = silero_buf[SILERO_CHUNK_SAMPLES * 2:]
                        try:
                            if silero_vad_model is not None:
                                pcm_np = np.frombuffer(chunk_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                                audio_t = torch.from_numpy(pcm_np)
                                with torch.no_grad():
                                    last_is_speech = silero_vad_model(audio_t, 8000).item() > SILERO_THRESHOLD
                            else:
                                last_is_speech = audioop.rms(chunk_pcm, 2) > 300
                        except Exception:
                            last_is_speech = audioop.rms(chunk_pcm, 2) > 300
                    is_speech = last_is_speech

                if not in_speech:
                    # Roll pre-speech buffer so we capture the onset of each word
                    pre_speech.append(mulaw_frame)
                    if len(pre_speech) > PRE_SPEECH_FRAMES:
                        pre_speech.pop(0)
                    if is_speech:
                        in_speech = True
                        speech_frames = list(pre_speech)  # include pre-roll
                        silence_count = 0
                else:
                    speech_frames.append(mulaw_frame)  # keep trailing silence too
                    if not is_speech:
                        silence_count += 1
                        if silence_count >= SILENCE_FRAMES:
                            if len(speech_frames) >= MIN_SPEECH_FRAMES:
                                print(
                                    f"[stream] Utterance detected: "
                                    f"{len(speech_frames)} frames ({len(speech_frames) * 20}ms)",
                                    flush=True,
                                )
                                is_processing = True
                                barge_in_count = 0
                                barge_in_armed_at = float('inf')  # disarmed until first audio plays
                                frames_copy = list(speech_frames)
                                current_utterance_task = asyncio.create_task(handle_utterance(frames_copy))
                            speech_frames = []
                            pre_speech = []
                            silero_buf = b""
                            last_is_speech = False
                            in_speech = False
                            silence_count = 0
                    else:
                        silence_count = 0

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
