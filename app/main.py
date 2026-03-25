import os
import uuid
import asyncio
import json
from datetime import date as date_type, datetime as dt
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
from app.calendar_service import create_appointment, check_conflict, cancel_appointment, get_appointments

app = FastAPI()
stt_model = None

# --- CONECTARE REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def build_system_prompt() -> str:
    today = date_type.today()
    today_str = today.strftime("%-d %B %Y")  # e.g. "25 martie 2026"
    return f"""
Ești TestRec, recepționera clinicii TestClinic.
Rolul tău principal este să ajuți pacienții să programeze, să anuleze sau să verifice consultații.

Data de astăzi este: {today_str}. Anul curent este {today.year}.
Dacă pacientul nu specifică anul, folosește {today.year} (sau {today.year + 1} dacă data menționată a trecut deja în {today.year}).
NU întreba pacientul despre an dacă nu este necesar — deduce-l din context.

Cum să te comporți:
- Vorbește natural, politicos și prietenos în limba română.
- Pentru PROGRAMARE, colectează: numele complet, data (zi, lună), ora.
- Pentru ANULARE, colectează: data programării (zi, lună), ora programării.
- Pentru VERIFICARE, colectează: data (zi, lună), ora.
- Dacă lipsește vreo informație, întreabă politicos.
- NU trimite blocul JSON dacă oricare câmp este necunoscut — mai întâi colectează toate informațiile.
- Când ai TOATE informațiile necesare și completate, răspunde natural și adaugă UN SINGUR bloc JSON,
  exact în formatele de mai jos.

Format JSON pentru programare (după ce pacientul confirmă):
```json
{
  "action": "schedule",
  "name": "Numele Pacientului",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}
```

Format JSON pentru anulare (după ce pacientul confirmă anularea):
```json
{
  "action": "cancel",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}
```

Format JSON pentru verificare (imediat ce ai data și ora, fără să mai aștepți confirmare):
```json
{
  "action": "check",
  "date": "YYYY-MM-DD",
  "time": "HH:MM"
}
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
{"action": "check", "date": "2026-03-25", "time": "15:00"}
```
"""

SYSTEM_PROMPT = build_system_prompt()

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
                    # Reject past dates
                    appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
                    if appt_date < date_type.today():
                        print(f"[main] Data in trecut: {appointment['date']}", flush=True)
                        messages.append({
                            "role": "system",
                            "content": (
                                f"Data {appointment['date']} este în trecut. "
                                "Informează pacientul că nu se pot face programări pentru date trecute "
                                "și roagă-l să aleagă o dată viitoare. Nu include niciun bloc JSON în răspuns."
                            ),
                        })
                        past_response = await run_in_threadpool(
                            ollama.chat,
                            model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
                            messages=messages,
                            options={"temperature": 0.3, "num_ctx": 8192},
                        )
                        ai_reply = past_response["message"]["content"].strip()
                        if not ai_reply:
                            ai_reply = "Nu se pot face programări pentru date trecute. Vă rog să alegeți o dată viitoare."
                        clean_text = ai_reply
                        print(f"AI [{session_id}] (past-date):", ai_reply, flush=True)
                    else:
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

                elif action == "check":
                    events = await run_in_threadpool(
                        get_appointments,
                        appointment["date"],
                        appointment["time"],
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
                    print(f"[main] Check: {result_msg}", flush=True)
                    messages.append({"role": "system", "content": result_msg})
                    check_response = await run_in_threadpool(
                        ollama.chat,
                        model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
                        messages=messages,
                        options={"temperature": 0.3, "num_ctx": 8192},
                    )
                    ai_reply = check_response["message"]["content"].strip()
                    if not ai_reply:
                        ai_reply = (
                            f"Am verificat în sistem. "
                            + (f"Există o programare pe {appointment['date']} la ora {appointment['time']}."
                               if events else
                               f"Nu există nicio programare pe {appointment['date']} la ora {appointment['time']}.")
                        )
                    clean_text = ai_reply
                    print(f"AI [{session_id}] (check):", ai_reply, flush=True)

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