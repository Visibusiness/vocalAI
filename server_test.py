import os
import uuid
import ollama
import asyncio
import torch
import edge_tts
import json

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from database import init_db

conversation_state = {
    "state": "START",
    "barber": None,
    "date": None,
    "hour": None,
    "phone": None,
    "client_name": None
}

BARBERS = ["Andrei", "Bogdan", "Cristi"]

# ==========================
# PYTORCH FIX
# ==========================
_original_load = torch.load
def safe_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

app = FastAPI()

# ==========================
# PROMPT
# ==========================

def get_system_prompt():
    return """
Ești Visi, recepționera salonului Elite.

Reguli:
1. Dacă clientul vrea programare, cere ora dacă nu a fost specificată.
2. Dacă a fost specificată ora și frizerul, confirmă politicos programarea.
3. Dacă lipsesc informații, cere-le clar.
4. Răspunde scurt și natural.
"""

async def extract_barber(text):
    prompt = f"""
Extrage doar numele frizerului din mesaj.

Frizeri disponibili: Andrei, Bogdan, Cristi.

Mesaj: "{text}"

Răspunde STRICT în format JSON:
{{
    "barber": "Andrei" | "Bogdan" | "Cristi" | null
}}
"""

    response = await run_in_threadpool(
        ollama.chat,
        model="visi-ro",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0}
    )
    print("RAW LLM RESPONSE:", response["message"]["content"])
    return json.loads(response["message"]["content"])

# ==========================
# VOICE ENDPOINT
# ==========================
stt_model = None

@app.on_event("startup")
async def load_models():
    init_db()
    global stt_model
    
    print("Încărcare model Whisper în memorie...")
    stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    print("Încălzire Ollama (visi-ro)...")
    try:
        # Trimitem o cerere scurtă și falsă către Ollama pentru a forța urcarea modelului în VRAM
        await run_in_threadpool(
            ollama.chat,
            model="visi-ro",
            messages=[{"role": "user", "content": "wake up"}],
            options={"temperature": 0.0} # Temperatură mică, ca să răspundă scurt și predictibil
        )
        print("Modelele sunt gata și încărcate!")
    except Exception as e:
        print(f"Eroare la conectarea cu Ollama: {e}")

@app.post("/voice")
async def voice_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    input_path = f"in_{unique_id}.wav"
    output_path = f"out_{unique_id}.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        segments, _ = await run_in_threadpool(
            stt_model.transcribe,
            input_path,
            language="ro"
        )

        user_text = " ".join([s.text for s in segments]).strip()
        print(f"🎤 User: {user_text}")

        global conversation_state
        print("STATE CURENT:", conversation_state)
        # 🔄 Reset conversație dacă userul spune "gata"
        if "gata" in user_text.lower():
            conversation_state = {
                "state": "START",
                "barber": None,
                "date": None,
                "hour": None,
                "phone": None,
                "client_name": None
            }

            ai_reply = "Conversația a fost resetată."

            print("🔁 Conversație resetată.")

            communicate = edge_tts.Communicate(
                ai_reply,
                "ro-RO-AlinaNeural",
                rate="-10%",
                pitch="-5Hz"
            )
            await communicate.save(output_path)

            background_tasks.add_task(cleanup_files, input_path, output_path)

            return FileResponse(output_path, media_type="audio/mpeg")
        
        # 🔹 START state
        if conversation_state["state"] == "START":
            normalized = user_text.lower()

            if "programare" in normalized or "rezerv" in normalized:
                conversation_state["state"] = "ASK_BARBER"

                ai_reply = "Cu ce frizer doriți? Andrei, Bogdan sau Cristi?"

                print("➡️ Trecem în ASK_BARBER")

                communicate = edge_tts.Communicate(
                    ai_reply,
                    "ro-RO-AlinaNeural",
                    rate="-10%",
                    pitch="-5Hz"
                )
                await communicate.save(output_path)

                background_tasks.add_task(cleanup_files, input_path, output_path)

                return FileResponse(output_path, media_type="audio/mpeg")
        # 🔹 ASK_BARBER state
        if conversation_state["state"] == "ASK_BARBER":
            data = await extract_barber(user_text)

            barber = data.get("barber")

            if barber and barber in BARBERS:
                conversation_state["barber"] = barber
                conversation_state["state"] = "ASK_DATE"

                ai_reply = f"Perfect. Pentru ce dată doriți programarea cu {barber}?"

                print(f"✔️ Frizer selectat: {barber}")
                print("➡️ Trecem în ASK_DATE")
            else:
                ai_reply = "Nu am înțeles frizerul. Vă rog alegeți între Andrei, Bogdan sau Cristi."

            communicate = edge_tts.Communicate(
                ai_reply,
                "ro-RO-AlinaNeural",
                rate="-10%",
                pitch="-5Hz"
            )
            await communicate.save(output_path)

            background_tasks.add_task(cleanup_files, input_path, output_path)

            return FileResponse(output_path, media_type="audio/mpeg")

        # Răspuns LLM (fără istoric)
        response = await run_in_threadpool(
            ollama.chat,
            model="visi-ro",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_text}
            ],
            options={"temperature": 0.3}
        )

        ai_reply = response["message"]["content"].strip()
        print(f"🤖 AI: {ai_reply}")

        communicate = edge_tts.Communicate(ai_reply, "ro-RO-AlinaNeural", rate="-10%" , pitch ="-5Hz")
        await communicate.save(output_path)

        background_tasks.add_task(cleanup_files, input_path, output_path)

        return FileResponse(output_path, media_type="audio/mpeg")

    except Exception as e:
        return {"error": str(e)}

async def cleanup_files(i, o):
    await asyncio.sleep(2)
    if os.path.exists(i):
        os.remove(i)
    if os.path.exists(o):
        os.remove(o)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)