import os
import uuid
import asyncio
import json
import redis
import ollama
import edge_tts
import io

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

from datetime import datetime, timedelta
from app.database import SessionLocal, CallLog, init_db

app = FastAPI()
stt_model = None

# --- CONECTARE REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

SYSTEM_PROMPT = """
Ești Visi, recepționera salonului Elite.
Răspunde natural și concis în română.
"""

def save_convo_to_db(session_id: str, messages_json: str):
    db = SessionLocal()
    try:
        # Secretul aici: Căutăm o conversație pentru acest număr care 
        # a fost actualizată în ultimele 15 minute.
        fifteen_mins_ago = datetime.utcnow() - timedelta(minutes=5)
        
        log = db.query(CallLog).filter(
            CallLog.session_id == session_id,
            CallLog.updated_at >= fifteen_mins_ago
        ).order_by(CallLog.id.desc()).first()

        if log:
            # Apelul este încă activ! Actualizăm jurnalul cu noile mesaje.
            log.messages = messages_json
        else:
            # Este un apel nou. Creăm un rând nou în tabel.
            new_log = CallLog(
                session_id=session_id,
                messages=messages_json
            )
            db.add(new_log)
            
        db.commit()
    except Exception as e:
        print(f"❌ Eroare la salvarea în baza de date: {e}")
    finally:
        db.close()

@app.on_event("startup")
async def load_models():
    global stt_model
    print("Loading Whisper medium...")
    
    # 1. Încărcăm Whisper
    stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
    print("Whisper ready.")

    # 2. Încălzim Ollama
    print("⏳ Încălzim modelul Ollama (Gemma-3 27B)... Asta va dura 1-2 minute.")
    try:
        # Rulăm sincron, deoarece suntem în faza de startup
        response = ollama.chat(
            model="visi-ro",
            messages=[{"role": "user", "content": "Salut. Ești gata?"}]
        )
        print("✅ Ollama este încălzit și încărcat în VRAM! Răspuns test:", response["message"]["content"].strip())
    except Exception as e:
        print(f"❌ Eroare la încălzirea Ollama: {e}")

    init_db()
    print("🗄️ Baza de date SQLite a fost inițializată (salon.db).")

@app.post("/voice")
async def voice_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    session_id: str = Form(...) # Primim ID-ul sesiunii (ex: numar telefon)
):
    unique_id = uuid.uuid4().hex

    try:
        # 1. Salvam fisierul audio primit
        content = await file.read()

        audio_stream = io.BytesIO(content)
        
        # 2. STT (Audio -> Text)
        segments, _ = await run_in_threadpool(
            stt_model.transcribe, audio_stream, language="ro"
        )
        user_text = " ".join([s.text for s in segments]).strip()
        print(f"User [{session_id}]:", user_text)

        # --- 3. MEMORIA: Recuperăm istoricul din Redis ---
        history_json = redis_client.get(session_id)
        
        if history_json:
            # Daca exista, il transformam din text inapoi in lista de Python
            messages = json.loads(history_json)
        else:
            # Daca nu, incepem o conversatie noua doar cu promptul
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Adăugăm ce a spus clientul ACUM
        messages.append({"role": "user", "content": user_text})

        # 4. LLM (Trimitem tot istoricul către Ollama)
        response = await run_in_threadpool(
            ollama.chat,
            model="visi-ro",
            messages=messages, 
            options={"temperature": 0.3}
        )

        ai_reply = response["message"]["content"].strip()
        print(f"AI [{session_id}]:", ai_reply)

        # --- 5. MEMORIA: Salvăm istoricul actualizat în Redis ---
        # Adăugăm răspunsul lui Visi
        messages.append({"role": "assistant", "content": ai_reply})

        messages_json = json.dumps(messages)
        
        # Salvăm în Redis cu o durată de viață de 600 secunde (10 minute)
        redis_client.setex(session_id, 600, messages_json)

        background_tasks.add_task(save_convo_to_db, session_id, messages_json)

        # 6. TTS (Text -> Audio) In-Memory (Streaming)
        async def audio_stream_generator():
            communicate = edge_tts.Communicate(ai_reply, "ro-RO-AlinaNeural")
            # Iterăm prin bucățile de date pe măsură ce Edge-TTS le generează
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"] # Trimitem doar byții de sunet

        # Returnăm fluxul continuu către client
        return StreamingResponse(audio_stream_generator(), media_type="audio/mpeg")

    except Exception as e:
        return {"error": str(e)}
