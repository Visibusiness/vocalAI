import os
import uuid
import asyncio
import datetime
import json
import re
import edge_tts
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from faster_whisper import WhisperModel

# ==========================
# CONFIGURARE FASTAPI
# ==========================
app = FastAPI()

# ==========================
# CONFIGURARE DB
# ==========================
DB_PATH = "salon_db.json"
FRIZERI_VALIZI = ["Andrei", "Bogdan", "Cristi"]
ORE_VALIDE = [f"{h:02d}:00" for h in range(9, 21)]

# Stările posibile
STARI = ["START", "ASTEPT_BARBER", "ASTEPT_DATA", "ASTEPT_ORA", "ASTEPT_NUME", "ASTEPT_TELEFON", "CONFIRMARE"]

# Starea globală a conversației (se menține între request-uri)
conversation_state = {
    "state": "START",
    "barber": None,
    "date": None,
    "hour": None,
    "phone": None,
    "client_name": None
}

# ==========================
# FUNCȚII DB
# ==========================
def load_db():
    """Încarcă baza de date"""
    if not os.path.exists(DB_PATH):
        db = {frizer: {} for frizer in FRIZERI_VALIZI}
        with open(DB_PATH, "w") as f:
            json.dump(db, f, indent=4)
        return db
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    """Salvează baza de date"""
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

def save_appointment():
    """Salvează programarea în baza de date"""
    global conversation_state
    required_fields = ['barber', 'date', 'hour', 'client_name', 'phone']
    for field in required_fields:
        if not conversation_state.get(field):
            print(f"Eroare: lipsă {field} la salvare")
            return False

    db = load_db()
    if conversation_state["barber"] not in db:
        db[conversation_state["barber"]] = {}
    if conversation_state["date"] not in db[conversation_state["barber"]]:
        db[conversation_state["barber"]][conversation_state["date"]] = []
    if conversation_state["hour"] in db[conversation_state["barber"]][conversation_state["date"]]:
        print(f"Eroare: ora {conversation_state['hour']} este deja ocupată")
        return False

    db[conversation_state["barber"]][conversation_state["date"]].append(conversation_state["hour"])
    save_db(db)

    # Istoric
    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "client": conversation_state["client_name"],
        "phone": conversation_state["phone"],
        "barber": conversation_state["barber"],
        "date": conversation_state["date"],
        "hour": conversation_state["hour"]
    }
    history_path = "programari_istorice.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(history_entry)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Programare salvată: {conversation_state}")
    return True

def check_availability(barber, date, hour):
    """Verifică dacă o oră e disponibilă"""
    db = load_db()
    if barber in db and date in db[barber]:
        return hour not in db[barber][date]
    return True

def get_available_hours(barber, date):
    """Obține orele disponibile pentru un frizer la o dată"""
    db = load_db()
    if barber in db and date in db[barber]:
        ocupate = set(db[barber][date])
        return [ora for ora in ORE_VALIDE if ora not in ocupate]
    return ORE_VALIDE.copy()

# ==========================
# FUNCȚII DE VALIDARE
# ==========================
def valideaza_data(text):
    text_lower = text.lower()
    azi = datetime.datetime.now()

    if "maine" in text_lower or "mâine" in text_lower:
        return (azi + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if "poimaine" in text_lower or "poimâine" in text_lower:
        return (azi + datetime.timedelta(days=2)).strftime("%Y-%m-%d")

    match = re.search(r'(\d{1,2})[./-](\d{1,2})(?:[./-](\d{4}))?', text)
    if match:
        zi, luna, an = int(match.group(1)), int(match.group(2)), int(match.group(3)) if match.group(3) else azi.year
        try:
            data = datetime.datetime(an, luna, zi)
            if data.date() >= azi.date():
                return data.strftime("%Y-%m-%d")
        except:
            pass

    luni_romana = {
        'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4,
        'mai': 5, 'iunie': 6, 'iulie': 7, 'august': 8,
        'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12
    }
    for nume_luna, nr_luna in luni_romana.items():
        if nume_luna in text_lower:
            match_zi = re.search(r'(\d{1,2})\s*' + nume_luna, text_lower)
            if match_zi:
                zi = int(match_zi.group(1))
                try:
                    data = datetime.datetime(azi.year, nr_luna, zi)
                    if data.date() >= azi.date():
                        return data.strftime("%Y-%m-%d")
                    else:
                        data = datetime.datetime(azi.year + 1, nr_luna, zi)
                        return data.strftime("%Y-%m-%d")
                except:
                    pass
    return None

def valideaza_ora(text):
    patterns = [
        r'(?:ora|la)\s+(\d{1,2})(?::(\d{2}))?',
        r'(\d{1,2}):(\d{2})',
        r'\b(\d{1,2})\b(?![:\d])'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if ':' in pattern or (len(match.groups()) > 1 and match.group(2) is not None):
                ora = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
            else:
                ora = int(match.group(1))
                minute = 0
            if 9 <= ora <= 20 and minute == 0:
                ora_formatted = f"{ora:02d}:00"
                if ora_formatted in ORE_VALIDE:
                    return ora_formatted
    return None

def valideaza_frizer(text):
    text_lower = text.lower()
    for frizer in FRIZERI_VALIZI:
        if frizer.lower() in text_lower:
            return frizer
    return None

def valideaza_telefon(text):
    digits = re.sub(r'\D', '', text)
    if len(digits) == 10:
        return digits
    elif len(digits) == 9:
        return "0" + digits
    return None

# ==========================
# LOGICA PRINCIPALĂ A CHAT-ULUI
# ==========================
def get_visi_response(user_input):
    """Generează răspunsul lui Visi pe baza stării curente"""
    global conversation_state

    # START – inițiere programare
    if conversation_state["state"] == "START":
        if any(word in user_input.lower() for word in ["vreau programare", "as vrea programare", "programare", "rezervare"]):
            conversation_state["state"] = "ASTEPT_BARBER"
            return "Bună! Cu drag te ajut. La care dintre frizeri dorești programare? Îl avem pe Andrei, Bogdan sau Cristi."

    # ASTEPT_BARBER
    if conversation_state["state"] == "ASTEPT_BARBER":
        barber = valideaza_frizer(user_input)
        if barber:
            conversation_state["barber"] = barber
            conversation_state["state"] = "ASTEPT_DATA"
            return f"Perfect, ai ales pe {barber}. Pentru ce dată dorești programarea? (de exemplu: mâine, sau 15 martie)"
        else:
            return f"Îmi pare rău, nu am înțeles. Îl poți alege pe Andrei, Bogdan sau Cristi?"

    # ASTEPT_DATA – așteptăm data (și eventual ora)
    if conversation_state["state"] == "ASTEPT_DATA":
        data = valideaza_data(user_input)
        ora = valideaza_ora(user_input)

        if data:
            conversation_state["date"] = data
            if ora:
                if check_availability(conversation_state["barber"], data, ora):
                    conversation_state["hour"] = ora
                    conversation_state["state"] = "ASTEPT_NUME"
                    return f"Ora {ora} e liberă. Cum te numești, te rog?"
                else:
                    ore_disponibile = get_available_hours(conversation_state["barber"], data)
                    if ore_disponibile:
                        ore_text = ', '.join(ore_disponibile[:5])
                        return f"Ora {ora} este deja ocupată. Pentru {conversation_state['barber']} în data de {data} sunt disponibile orele: {ore_text}. Ce oră ți-ar conveni?"
                    else:
                        return f"Nu mai sunt ore disponibile pentru {conversation_state['barber']} în data de {data}. Poți alege o altă dată?"
            else:
                conversation_state["state"] = "ASTEPT_ORA"
                ore_disponibile = get_available_hours(conversation_state["barber"], data)
                if ore_disponibile:
                    ore_text = ', '.join(ore_disponibile[:5])
                    return f"Pentru {conversation_state['barber']} în data de {data} sunt disponibile orele: {ore_text}. Ce oră ți-ar conveni?"
                else:
                    return f"Nu mai sunt ore disponibile pentru {conversation_state['barber']} în data de {data}. Poți alege o altă dată?"
        else:
            return "Nu am înțeles data. Poți spune 'mâine' sau o dată exactă, de exemplu 25 martie?"

    # ASTEPT_ORA – așteptăm ora
    if conversation_state["state"] == "ASTEPT_ORA":
        ora = valideaza_ora(user_input)
        if ora:
            if check_availability(conversation_state["barber"], conversation_state["date"], ora):
                conversation_state["hour"] = ora
                conversation_state["state"] = "ASTEPT_NUME"
                return f"Ora {ora} e liberă. Cum te numești, te rog?"
            else:
                ore_disponibile = get_available_hours(conversation_state["barber"], conversation_state["date"])
                if ore_disponibile:
                    ore_text = ', '.join(ore_disponibile[:5])
                    return f"Ora {ora} este deja ocupată. Poți alege dintre orele disponibile: {ore_text}"
                else:
                    return f"Nu mai sunt ore disponibile pentru {conversation_state['barber']} în această zi. Poți alege o altă dată?"
        else:
            return "Te rog să alegi o oră între 9 dimineața și 8 seara, de exemplu 14:00."

    # ASTEPT_NUME – așteptăm numele
    if conversation_state["state"] == "ASTEPT_NUME":
        if len(user_input.strip()) >= 2:
            conversation_state["client_name"] = user_input.strip().title()
            conversation_state["state"] = "ASTEPT_TELEFON"
            return "Îți mai trebuie un număr de telefon ca să pot confirma programarea. Îl poți da?"
        else:
            return "Scuze, nu am înțeles numele. Poți repeta, te rog?"

    # ASTEPT_TELEFON – așteptăm telefonul și salvăm
    if conversation_state["state"] == "ASTEPT_TELEFON":
        telefon = valideaza_telefon(user_input)
        if telefon:
            conversation_state["phone"] = telefon
            if save_appointment():
                barber = conversation_state["barber"]
                client_name = conversation_state["client_name"]
                date = conversation_state["date"]
                hour = conversation_state["hour"]
                phone = conversation_state["phone"]
                data_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y")
                confirmare = f"""Perfect, {client_name}! Programarea ta a fost confirmată:

📅 Data: {data_formatted}
⏰ Ora: {hour}
💇 Frizer: {barber}
📞 Telefon: {phone}

Te așteptăm cu drag la salon! Dacă mai ai nevoie de ajutor, sunt aici pentru tine."""

                # Resetăm starea pentru următoarea conversație
                conversation_state = {
                    "state": "START",
                    "barber": None,
                    "date": None,
                    "hour": None,
                    "phone": None,
                    "client_name": None
                }
                return confirmare
            else:
                return "A apărut o eroare la salvarea programării. Te rog încearcă din nou."
        else:
            return "Te rog să introduci un număr de telefon valid cu 10 cifre."

    # Verificare disponibilitate explicită (indiferent de stare)
    if "disponibil" in user_input.lower() or "ce ore" in user_input.lower():
        barber = valideaza_frizer(user_input)
        data = valideaza_data(user_input) or datetime.datetime.now().strftime("%Y-%m-%d")
        data_formatted = datetime.datetime.strptime(data, "%Y-%m-%d").strftime("%d.%m.%Y")
        if barber:
            ore = get_available_hours(barber, data)
            if ore:
                ore_text = ', '.join(ore)
                return f"{barber} are liber în data de {data_formatted} la orele: {ore_text}. Dorești să programezi la una dintre ele?"
            else:
                return f"{barber} nu mai are ore libere în data de {data_formatted}. Poți verifica o altă zi?"
        else:
            disponibilitate = []
            for f in FRIZERI_VALIZI:
                ore = get_available_hours(f, data)
                if ore:
                    disponibilitate.append(f"{f} ({len(ore)} ore libere)")
            if disponibilitate:
                return f"Pentru data de {data_formatted}: {', '.join(disponibilitate)}. Vrei să programezi la cineva anume?"
            else:
                return f"În data de {data_formatted} toți frizerii sunt ocupați. Încearcă o altă zi."

    # Comandă reset
    if "reset" in user_input.lower():
        conversation_state = {
            "state": "START",
            "barber": None,
            "date": None,
            "hour": None,
            "phone": None,
            "client_name": None
        }
        return "Am resetat conversația. Cu ce te pot ajuta?"

    # Răspuns implicit
    return "Cu ce te pot ajuta? Poți să îmi spui 'vreau o programare' sau să întrebi de disponibilitate."

# ==========================
# CONFIGURARE SERVER
# ==========================
stt_model = None

@app.on_event("startup")
async def startup():
    global stt_model
    print("⏳ Pornire Whisper Large-v3...")
    stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("✅ Whisper gata. Server activ.")

# ==========================
# ENDPOINT PRINCIPAL
# ==========================
@app.post("/voice")
async def voice_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    in_p, out_p = f"in_{uid}.wav", f"out_{uid}.mp3"

    with open(in_p, "wb") as f:
        f.write(await file.read())

    # 1. Voice to Text (Whisper)
    segments, _ = await run_in_threadpool(stt_model.transcribe, in_p, language="ro")
    user_text = " ".join([s.text for s in segments]).strip()
    print(f"🎤 Client: {user_text}")

    # 2. Obține răspunsul de la Visi (bazat pe starea conversației)
    visi_reply = get_visi_response(user_text)
    print(f"🤖 Visi: {visi_reply}")

    # 3. Text to Speech (Edge-TTS)
    await edge_tts.Communicate(visi_reply, "ro-RO-AlinaNeural").save(out_p)

    background_tasks.add_task(lambda: (os.remove(in_p), os.remove(out_p)) if os.path.exists(in_p) else None)
    return FileResponse(out_p, media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)