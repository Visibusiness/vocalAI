import os
import uuid
import asyncio
import json
from urllib.parse import quote, unquote
import torch  # must be imported before faster_whisper to init CUDA lib paths
import redis
import ollama
import edge_tts

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

from app.appointment_parser import extract_appointment
from app.calendar_service import create_appointment, check_conflict, cancel_appointment

app = FastAPI()
stt_model = None

# --- CONECTARE REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

SYSTEM_PROMPT = """
Ești TestRec, recepționera clinicii TestClinic.
Rolul tău principal este să ajuți pacienții să programeze sau să anuleze consultații.

Cum să te comporți:
- Vorbește natural, politicos și prietenos în limba română.
- Pentru PROGRAMARE, colectează:
    1. Numele complet al pacientului
    2. Data dorită (zi, lună, an)
    3. Ora dorită
- Pentru ANULARE, colectează:
    1. Data programării (zi, lună, an)
    2. Ora programării
- Dacă lipsește vreo informație, întreabă politicos.
- Când ai toate informațiile și pacientul confirmă, răspunde natural
  și adaugă UN SINGUR bloc JSON la sfârșitul mesajului, exact în formatele de mai jos.

Format JSON pentru programare (doar când pacientul confirmă):
```json
{
  "action": "schedule",
  "name": "Numele Pacientului",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}
```

Format JSON pentru anulare (doar când pacientul confirmă anularea):
```json
{
  "action": "cancel",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}
```

Important:
- Nu include blocul JSON decât dacă pacientul a confirmat explicit acțiunea.
- Nu inventa informații — dacă nu știi data sau ora, întreabă.
- Răspunsul natural vine ÎNAINTE de blocul JSON.
"""

@app.on_event("startup")
async def load_models():
    global stt_model
    print("Loading Whisper medium...", flush=True)
    stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
    print("Whisper ready.", flush=True)

    print("Warming up Ollama (loading model into VRAM)...", flush=True)
    await run_in_threadpool(
        ollama.chat,
        model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 1},
    )
    print("Ollama ready.", flush=True)

def get_stt_model():
    return stt_model

@app.post("/voice")
async def voice_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    session_id: str = Form(...) # primim ID-ul sesiunii (ex: numar telefon)
):
    unique_id = uuid.uuid4().hex
    input_path = f"in_{unique_id}.wav"
    output_path = f"out_{unique_id}.mp3"

    try:
        # salvam fisierul audio primit
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 2. STT (Audio -> Text)
        segments, _ = await run_in_threadpool(
            get_stt_model().transcribe, input_path, language="ro"
        )
        user_text = " ".join([s.text for s in segments]).strip()
        print(f"User [{session_id}]:", user_text, flush=True)

        # recuperam istoricul din Redis
        history_json = redis_client.get(session_id)

        if history_json:
            messages = json.loads(history_json)
            print(f"[redis] Loaded {len(messages)} messages for session {session_id}", flush=True)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"[redis] New session {session_id}", flush=True)

        # adaugam ce a spus clientul ACUM
        messages.append({"role": "user", "content": user_text})

        # LLM
        response = await run_in_threadpool(
            ollama.chat,
            model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
            messages=messages,
            options={"temperature": 0.3, "num_ctx": 8192}
        )

        ai_reply = response["message"]["content"].strip()
        print(f"AI [{session_id}]:", ai_reply, flush=True)

        # --- EXTRAGE PROGRAMAREA DIN RASPUNS (daca exista) ---
        clean_text, appointment = extract_appointment(ai_reply)

        if appointment:
            action = appointment.get("action")
            try:
                if action == "schedule":
                    conflict = await run_in_threadpool(
                        check_conflict,
                        appointment["date"],
                        appointment["time"],
                    )
                    if conflict:
                        print(f"[main] Conflict detectat pentru {appointment['date']} {appointment['time']}", flush=True)
                        messages.append({
                            "role": "system",
                            "content": (
                                f"Intervalul {appointment['time']} din {appointment['date']} este deja ocupat. "
                                "Informează pacientul politicos că acel interval nu este disponibil și propune-i "
                                "să aleagă o altă oră sau zi. Nu include niciun bloc JSON în răspuns."
                            ),
                        })
                        conflict_response = await run_in_threadpool(
                            ollama.chat,
                            model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
                            messages=messages,
                            options={"temperature": 0.3, "num_ctx": 8192},
                        )
                        ai_reply = conflict_response["message"]["content"].strip()
                        if not ai_reply:
                            ai_reply = (
                                f"Îmi pare rău, intervalul de la ora {appointment['time']} "
                                f"din data de {appointment['date']} este deja ocupat. "
                                "Doriți să alegeți o altă oră sau zi?"
                            )
                        clean_text = ai_reply
                        print(f"AI [{session_id}] (conflict):", ai_reply, flush=True)
                    else:
                        event_link = await run_in_threadpool(
                            create_appointment,
                            appointment["name"],
                            appointment["date"],
                            appointment["time"],
                        )
                        print(f"[main] Programare creata: {event_link}", flush=True)

                elif action == "cancel":
                    deleted = await run_in_threadpool(
                        cancel_appointment,
                        appointment["date"],
                        appointment["time"],
                    )
                    if not deleted:
                        print(f"[main] Anulare: nicio programare gasita pentru {appointment['date']} {appointment['time']}", flush=True)
                        messages.append({
                            "role": "system",
                            "content": (
                                f"Nu a fost găsită nicio programare pe data de {appointment['date']} "
                                f"la ora {appointment['time']}. "
                                "Informează pacientul politicos și întreabă dacă dorește să verifice altă dată sau oră. "
                                "Nu include niciun bloc JSON în răspuns."
                            ),
                        })
                        not_found_response = await run_in_threadpool(
                            ollama.chat,
                            model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
                            messages=messages,
                            options={"temperature": 0.3, "num_ctx": 8192},
                        )
                        ai_reply = not_found_response["message"]["content"].strip()
                        if not ai_reply:
                            ai_reply = (
                                f"Nu am găsit nicio programare pe data de {appointment['date']} "
                                f"la ora {appointment['time']}. "
                                "Doriți să verificați o altă dată sau oră?"
                            )
                        clean_text = ai_reply
                        print(f"AI [{session_id}] (cancel-not-found):", ai_reply, flush=True)
                    else:
                        print(f"[main] Programare anulata: {appointment['date']} {appointment['time']}", flush=True)

            except Exception as e:
                print(f"[main] Eroare la procesare programare: {e}", flush=True)

        # MEMORIA: Salvam istoricul actualizat in Redis ---
        # Salvam raspunsul complet (cu JSON) in istoric, dar trimitem doar textul curat la TTS
        messages.append({"role": "assistant", "content": ai_reply})

        # Salvam in Redis cu o durata de viața de 600 secunde (10 minute)
        redis_client.setex(session_id, 600, json.dumps(messages))

        # TTS (Text -> Audio) — folosim textul fara blocul JSON
        communicate = edge_tts.Communicate(clean_text, "ro-RO-AlinaNeural")
        await communicate.save(output_path)

        background_tasks.add_task(cleanup_files, input_path, output_path)

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            headers={"X-AI-Text": quote(clean_text)},
        )

    except Exception as e:
        return {"error": str(e)}

async def cleanup_files(i, o):
    await asyncio.sleep(2)
    if os.path.exists(i):
        os.remove(i)
    if os.path.exists(o):
        os.remove(o)