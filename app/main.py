import os
import uuid
import asyncio
import ollama
import edge_tts

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

app = FastAPI()
stt_model = None

SYSTEM_PROMPT = """
Ești Visi, recepționera salonului Elite.
Răspunde natural și concis în română.
"""

@app.on_event("startup")
async def load_models():
    global stt_model

    print("Loading Whisper medium...")
    stt_model = WhisperModel(
        "medium",
        device="cuda",
        compute_type="float16"
    )

    print("Whisper ready.")

@app.post("/voice")
async def voice_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    unique_id = uuid.uuid4().hex
    input_path = f"in_{unique_id}.wav"
    output_path = f"out_{unique_id}.mp3"

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # STT
        segments, _ = await run_in_threadpool(
            stt_model.transcribe,
            input_path,
            language="ro"
        )

        user_text = " ".join([s.text for s in segments]).strip()
        print("User:", user_text)

        # LLM (Ollama local model)
        response = await run_in_threadpool(
            ollama.chat,
            model="visi-ro",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            options={"temperature": 0.3}
        )

        ai_reply = response["message"]["content"].strip()
        print("AI:", ai_reply)

        # TTS
        communicate = edge_tts.Communicate(
            ai_reply,
            "ro-RO-AlinaNeural"
        )
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