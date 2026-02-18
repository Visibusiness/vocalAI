import os
import uuid
import ollama
import asyncio
import datetime
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel
from TTS.api import TTS

# === FIX PENTRU PYTORCH 2.6+ ===
# Acest bloc previne eroarea "Weights only load failed"
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load
# ===============================

app = FastAPI()

# === MODELUL ROMÂNESC PUR (VITS) ===
# Acesta este modelul oficial Coqui pentru limba română (Common Voice)
# Este o voce feminină standard, foarte clară.
MODEL_NAME = "tts_models/ro/cv/vits"

CHAT_HISTORY = []

print("⏳ [Server] Încărcare Whisper Large-v3...")
stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

print(f"⏳ [Server] Încărcare Model Românesc ({MODEL_NAME})...")
# Prima rulare va dura puțin (descarcă modelul), apoi e instant.
tts_model = TTS(MODEL_NAME).to("cuda")
print("✅ Totul încărcat! Visi vorbește românește.")

now = datetime.datetime.now()
today_str = now.strftime("%d-%m-%Y %H:%M")

SYSTEM_PROMPT = f"""
Ești Visi, recepționera salonului 'Elite'. Azi este {today_str}.
Vorbește EXCLUSIV în limba română.
Fii politicoasă, dar scurtă și eficientă.
Scop: Obține Numele, Serviciul și Ora.
"""

async def cleanup_files(input_f: str, output_f: str):
    await asyncio.sleep(1)
    if os.path.exists(input_f): os.remove(input_f)
    if os.path.exists(output_f): os.remove(output_f)

@app.post("/voice")
async def voice_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global CHAT_HISTORY
    unique_id = uuid.uuid4().hex
    input_path = f"temp_in_{unique_id}.wav"
    output_path = f"temp_out_{unique_id}.wav"

    try:
        content = await file.read()
        with open(input_path, "wb") as buffer:
            buffer.write(content)

        # 1. STT (Whisper)
        segments, _ = await run_in_threadpool(stt_model.transcribe, input_path, language="ro", beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        print(f"🎤 User: {text}")

        if not text:
            if os.path.exists(input_path): os.remove(input_path)
            return {"error": "Liniște detectată."}

        CHAT_HISTORY.append({'role': 'user', 'content': text})
        
        # 2. LLM (Gemma 2 9B)
        # Folosim context scurt (ultimele 10 replici)
        response = await run_in_threadpool(
            ollama.chat,
            model="gemma2:9b",
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT}] + CHAT_HISTORY[-10:],
            options={'temperature': 0.2}
        )
        
        ai_reply = response['message']['content'].replace("*", "").strip()
        CHAT_HISTORY.append({'role': 'assistant', 'content': ai_reply})
        print(f"🤖 AI: {ai_reply}")

        # 3. TTS (VITS STANDARD)
        # Aici e simplificarea majoră: nu mai avem speaker_wav sau language.
        # Modelul știe doar română.
        await run_in_threadpool(
            lambda: tts_model.tts_to_file(
                text=ai_reply,
                file_path=output_path
            )
        )
        
        if os.path.exists(output_path):
            background_tasks.add_task(cleanup_files, input_path, output_path)
            return FileResponse(output_path, media_type="audio/wav")
        else:
            raise HTTPException(status_code=500, detail="Eroare generare audio VITS.")

    except Exception as e:
        print(f"❌ EROARE: {str(e)}")
        if os.path.exists(input_path): os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)