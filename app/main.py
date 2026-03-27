import os
import io
import uuid
import asyncio
import json
from datetime import date as date_type, datetime as dt, timedelta
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
from app.calendar_service import create_appointment, check_conflict, cancel_appointment, get_appointments, list_appointments

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
REGULI STRICTE PENTRU AN:
- Dacă pacientul nu specifică anul, folosește ÎNTOTDEAUNA {today.year}. NICIODATĂ alt an (ex: 2024, 2025).
- NU întreba pacientul despre an în nicio situație. Deduce singur: dacă data a trecut deja în {today.year}, folosește {today.year + 1}.
- NU programa sau verifica date din trecut. Dacă data este anterioară față de astăzi ({today_str}), informează pacientul că nu este posibil.

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
{{"action": "check", "date": "{(today + timedelta(days=7)).strftime('%Y-%m-%d')}", "time": "15:00"}}
```
"""

SYSTEM_PROMPT = build_system_prompt()

@app.on_event("startup")
async def load_models():
    global stt_model
    print("Loading Whisper small...", flush=True)
    stt_model = WhisperModel("small", device="cuda", compute_type="float16")
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
    session_id: str = Form(...) # numarul de telefon al pacientului (ex: +40721000000)
):
    unique_id = uuid.uuid4().hex
    output_path = f"out_{unique_id}.mp3"

    try:
        # read uploaded audio into memory — no disk write needed
        content = await file.read()
        audio_buffer = io.BytesIO(content)

        # STT (Audio -> Text)
        segments, _ = await run_in_threadpool(
            get_stt_model().transcribe, audio_buffer, language="ro"
        )
        user_text = " ".join([s.text for s in segments]).strip()
        print(f"User [{session_id}]:", user_text, flush=True)

        # recuperam istoricul din Redis
        history_json = redis_client.get(session_id)

        if history_json:
            messages = json.loads(history_json)
            print(f"[redis] Loaded {len(messages)} messages for phone {session_id}", flush=True)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"[redis] New session for phone {session_id}", flush=True)

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
                                session_id,
                            )
                            print(f"[main] Programare creata: {event_link}", flush=True)

                elif action == "cancel":
                    appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
                    if appt_date < date_type.today():
                        print(f"[main] Cancel ignorat — data in trecut: {appointment['date']}", flush=True)
                        clean_text = f"Data de {appointment['date']} este în trecut. Nu există programări active pentru date trecute."
                        messages.append({"role": "assistant", "content": clean_text})
                        redis_client.setex(session_id, 600, json.dumps(messages))
                        communicate = edge_tts.Communicate(clean_text, "ro-RO-AlinaNeural")
                        await communicate.save(output_path)
                        background_tasks.add_task(cleanup_output, output_path)
                        return FileResponse(output_path, media_type="audio/mpeg", headers={"X-AI-Text": quote(clean_text)})
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
                    appt_date = dt.strptime(appointment["date"], "%Y-%m-%d").date()
                    if appt_date < date_type.today():
                        print(f"[main] Check ignorat — data in trecut: {appointment['date']}", flush=True)
                        clean_text = f"Data de {appointment['date']} este în trecut. Puteți verifica doar programări viitoare."
                        messages.append({"role": "assistant", "content": clean_text})
                        redis_client.setex(session_id, 600, json.dumps(messages))
                        communicate = edge_tts.Communicate(clean_text, "ro-RO-AlinaNeural")
                        await communicate.save(output_path)
                        background_tasks.add_task(cleanup_output, output_path)
                        return FileResponse(output_path, media_type="audio/mpeg", headers={"X-AI-Text": quote(clean_text)})
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

                elif action == "list":
                    appts = await run_in_threadpool(list_appointments, session_id)
                    if appts:
                        appt_lines = "\n".join(
                            f"- {a['summary']} la {a['start']}" for a in appts
                        )
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
                    print(f"[main] List: {result_msg}", flush=True)
                    messages.append({"role": "system", "content": result_msg})
                    list_response = await run_in_threadpool(
                        ollama.chat,
                        model="hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
                        messages=messages,
                        options={"temperature": 0.3, "num_ctx": 8192},
                    )
                    ai_reply = list_response["message"]["content"].strip()
                    if not ai_reply:
                        ai_reply = (
                            "Aveți următoarele programări viitoare: " + ", ".join(a["summary"] for a in appts)
                            if appts else "Nu aveți nicio programare viitoare înregistrată."
                        )
                    clean_text = ai_reply
                    print(f"AI [{session_id}] (list):", ai_reply, flush=True)

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

        background_tasks.add_task(cleanup_output, output_path)

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            headers={"X-AI-Text": quote(clean_text)},
        )

    except Exception as e:
        return {"error": str(e)}

async def cleanup_output(path):
    await asyncio.sleep(2)
    if os.path.exists(path):
        os.remove(path)