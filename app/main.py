import os, uuid, ollama, asyncio, datetime, torch, edge_tts, json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

# FIX PYTORCH
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

app = FastAPI()

# --- DATABASE & MEMORY ---
DB_PATH = "clinica_db.json"
# Memoria sesiunii: reține medicul, specializarea, data și ora
STATE = {"medic": None, "specializare": None, "data": None, "ora": None, "pacient": None}

def load_db():
    if not os.path.exists(DB_PATH):
        # Default mock data if file is missing
        data = {
            "Dr. Ionescu": {"specializare": "Cardiologie", "program": []},
            "Dr. Popescu": {"specializare": "Dermatologie", "program": []}
        }
        with open(DB_PATH, "w") as f: json.dump(data, f)
    with open(DB_PATH, "r") as f: return json.load(f)

def save_db(db):
    with open(DB_PATH, "w") as f: json.dump(db, f, indent=4)

# --- LOGICA MEDICALĂ ---
def process_medical_logic(extracted):
    global STATE
    db = load_db()
    
    # Actualizăm STATE cu ce am extras nou
    if extracted.get("nume_medic"): STATE["medic"] = extracted["nume_medic"]
    if extracted.get("specializare"): STATE["specializare"] = extracted["specializare"]
    if extracted.get("data"): STATE["data"] = extracted["data"]
    if extracted.get("ora"): STATE["ora"] = extracted["ora"]
    if extracted.get("nume_pacient"): STATE["pacient"] = extracted["nume_pacient"]

    # 1. Identificăm medicul dacă s-a dat doar specializarea
    if STATE["specializare"] and not STATE["medic"]:
        for nume, info in db.items():
            if STATE["specializare"].lower() in info["specializare"].lower():
                STATE["medic"] = nume
                break

    # 2. Verificăm ce lipsește
    missing = []
    if not STATE["medic"]: missing.append("specializarea sau numele medicului")
    if not STATE["data"]: missing.append("data (ziua)")
    if not STATE["ora"]: missing.append("ora")
    if not STATE["pacient"]: missing.append("numele dumneavoastră")

    if missing:
        return f"SISTEM: Avem nevoie de {', '.join(missing)}."

    # 3. Verificăm disponibilitatea (Data + Ora)
    programare_cheie = f"{STATE['data']} {STATE['ora']}"
    if programare_cheie in db[STATE["medic"]]["program"]:
        return f"SISTEM: Dr. {STATE['medic']} este ocupat pe {STATE['data']} la ora {STATE['ora']}. Alegeți alt moment."

    # 4. SALVARE (BOOK)
    db[STATE["medic"]]["program"].append(programare_cheie)
    save_db(db)
    res = f"SUCCES: Programare confirmată: Pacient {STATE['pacient']}, la {STATE['medic']} ({db[STATE['medic']]['specializare']}), pe data de {STATE['data']} la ora {STATE['ora']}."
    
    # Resetăm pentru următorul pacient
    STATE = {"medic": None, "specializare": None, "data": None, "ora": None, "pacient": None}
    return res

# --- AI MODELS ---
stt_model = None
@app.on_event("startup")
async def startup():
    global stt_model
    stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

async def extract_medical_data(text):
    acum = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    prompt = f"""Ești asistent medical. Azi e {acum}.
Analizează cererea: "{text}"
Extrage STRICT JSON:
{{
  "nume_pacient": string | null,
  "nume_medic": string | null,
  "specializare": "Cardiologie" | "Dermatologie" | "Pediatrie" | null,
  "data": "DD-MM" | null,
  "ora": "HH:MM" | null
}}"""
    res = await run_in_threadpool(ollama.chat, model="qwen2.5:1.5b-instruct", 
                                  messages=[{"role":"user","content":prompt}], 
                                  format="json", options={"temperature":0})
    return json.loads(res["message"]["content"])

def get_system_prompt(info):
    return f"""Ești Visi, recepționera Clinicii Elite. 
INFO SISTEM: {info}
REGULI:
1. Dacă sistemul cere date, întreabă politicos pacientul.
2. NU inventa medici sau ore. 
3. Dacă programarea e gata (SUCCES), confirmă toate detaliile clar.
4. Răspunde scurt și profesionist."""

@app.post("/voice")
async def voice_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    in_p, out_p = f"in_{uid}.wav", f"out_{uid}.mp3"
    with open(in_p, "wb") as f: f.write(await file.read())
    
    try:
        # STT
        segments, _ = await run_in_threadpool(stt_model.transcribe, in_p, language="ro")
        user_text = " ".join([s.text for s in segments]).strip()
        
        # LOGICĂ
        extracted = await extract_medical_data(user_text)
        info_sistem = process_medical_logic(extracted)
        
        # LLM
        res = await run_in_threadpool(ollama.chat, model="visi-ro", 
                                      messages=[{"role":"system", "content": get_system_prompt(info_sistem)}, 
                                                {"role":"user", "content": user_text}],
                                      options={"temperature": 0.1})
        ai_reply = res["message"]["content"]
        
        # TTS
        await edge_tts.Communicate(ai_reply, "ro-RO-AlinaNeural").save(out_p)
        background_tasks.add_task(lambda: (os.remove(in_p), os.remove(out_p)) if os.path.exists(in_p) else None)
        return FileResponse(out_p, media_type="audio/mpeg")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)