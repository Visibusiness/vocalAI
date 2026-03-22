import os
import uuid
import asyncio
import json
import redis
import ollama
import edge_tts

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

app = FastAPI()
stt_model = None

# --- CONECTARE REDIS ---
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

SYSTEM_PROMPT = """
Ești Visi, recepționera salonului Elite.
Răspunde natural și concis în română.
"""

@app.on_event("startup")
async def load_models():
    global stt_model
    print("Loading Whisper medium...")
    stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
    print("Whisper ready.")

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
            stt_model.transcribe, input_path, language="ro"
        )
        user_text = " ".join([s.text for s in segments]).strip()
        print(f"User [{session_id}]:", user_text)

        # recuperam istoricul din Redis
        history_json = redis_client.get(session_id)
        
        if history_json:
            messages = json.loads(history_json)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # adaugam ce a spus clientul ACUM
        messages.append({"role": "user", "content": user_text})

        # LLM
        response = await run_in_threadpool(
            ollama.chat,
            model="visi-ro",
            messages=messages, 
            options={"temperature": 0.3}
        )

        ai_reply = response["message"]["content"].strip()
        print(f"AI [{session_id}]:", ai_reply)

        # MEMORIA: Salvam istoricul actualizat in Redis ---
        messages.append({"role": "assistant", "content": ai_reply})
        
        # Salvam in Redis cu o durata de viața de 600 secunde (10 minute)
        redis_client.setex(session_id, 600, json.dumps(messages))

        # TTS (Text -> Audio)
        communicate = edge_tts.Communicate(ai_reply, "ro-RO-AlinaNeural")
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
