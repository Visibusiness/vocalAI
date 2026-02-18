import os
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama
import wave
import subprocess
import json
import datetime
from piper import PiperVoice

# === IMPORTĂM FUNCȚIA DE CALENDAR ===
# Asigură-te că fișierul calendar_tool.py este în același folder!
try:
    from calendar_tool import create_appointment
except ImportError:
    print("⚠️ ATENȚIE: Nu am găsit calendar_tool.py! Funcția de calendar nu va merge.")
    def create_appointment(s, d): return False

# === CONFIGURARE NVIDIA ===
def setup_nvidia_libs():
    venv_base = os.environ.get('VIRTUAL_ENV', os.path.join(os.getcwd(), 'venv'))
    found_libs = []
    if os.path.exists(venv_base):
        for root, dirs, files in os.walk(venv_base):
            if 'nvidia' in root and 'lib' in dirs:
                found_libs.append(os.path.join(root, 'lib'))
    if found_libs:
        os.environ['LD_LIBRARY_PATH'] = ":".join(found_libs) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

setup_nvidia_libs()

# === SETĂRI ===
SAMPLE_RATE = 16000          
DURATION = 10                 
WHISPER_MODEL = "small"
LLM_MODEL = "llama3.2"
PIPER_MODEL = "ro_RO-mihai-medium.onnx"
OUTPUT_FILENAME = "raspuns_visi.wav"

print("⌛ Încărcare modele (Whisper + Piper)...")
stt_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
voice = PiperVoice.load(PIPER_MODEL)

# === CONFIGURARE PROMPT AGENT ===
# Calculăm data și ora curentă pentru ca Visi să știe ce zi e azi
now = datetime.datetime.now()
today_str = now.strftime("%Y-%m-%d %H:%M") # Format: 2023-10-27 14:30
day_name = now.strftime("%A")

chat_history = [
    {
        'role': 'system', 
        'content': (
            f'Ești Visi, recepționer virtual la un salon. Azi suntem în data de {today_str} ({day_name}). '
            'REGULI: '
            '1. Vorbește EXCLUSIV Română (cu diacritice). '
            '2. Scopul tău este să faci o programare. Cere pe rând: Serviciul, Numele, Data și Ora. '
            '3. IMPORTANT: Când ai TOATE detaliile și clientul confirmă, NU mai genera text obișnuit. '
            'GENEREAZĂ DOAR ACEST JSON: '
            '{"action": "book", "nume": "Nume Client", "data": "YYYY-MM-DDTHH:MM:00"} '
            '4. Asigură-te că data din JSON este în formatul ISO corect (an-luna-ziTora:minut:00).'
        )
    }
]

# === DETECTARE PLAYER AUDIO ===
def get_audio_player():
    for player in ["paplay", "aplay", "ffplay"]:
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            return player
    return None

AUDIO_PLAYER = get_audio_player()

# === FUNCȚIE SINTEZĂ VOCALĂ (FIXED) ===
def speak(text):
    try:
        print(f"🤖 Agent: {text}")
        
        # Dacă textul e gol (poate a fost doar o comandă internă), nu zicem nimic
        if not text:
            return

        with wave.open(OUTPUT_FILENAME, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(voice.config.sample_rate)
            
            stream = voice.synthesize(text)
            has_audio = False
            
            for chunk in stream:
                # Fixul pentru audio_int16_bytes
                if hasattr(chunk, 'audio_int16_bytes'):
                    wf.writeframes(chunk.audio_int16_bytes)
                    has_audio = True
                elif hasattr(chunk, 'bytes'):
                     wf.writeframes(chunk.bytes)
                     has_audio = True
            
        if AUDIO_PLAYER and has_audio:
            subprocess.run([AUDIO_PLAYER, OUTPUT_FILENAME], check=True)

    except Exception as e:
        print(f"❌ Eroare TTS: {e}")

# === FUNCȚIE PRINCIPALĂ ===
def main():
    print(f"\n🚀 AGENT ACTIV | Data: {today_str}")
    print("🎤 Te ascult... (Ctrl+C pentru oprire)\n")

    while True:
        try:
            print("[Ascult...]", end="", flush=True)

            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            print("\r[Procesez...]", end="", flush=True)

            audio_data = recording.flatten().astype(np.float32) / 32768.0

            segments, _ = stt_model.transcribe(
                audio_data,
                language="ro",
                beam_size=5,
                vad_filter=True
            )
            user_text = "".join([s.text for s in segments]).strip()

            if user_text:
                print(f"\r👤 Client: {user_text}          ")

                # 1. Adăugăm ce a zis userul în istoric
                chat_history.append({'role': 'user', 'content': user_text})

                # 2. Întrebăm LLM-ul
                response = ollama.chat(model=LLM_MODEL, messages=chat_history)
                ai_response = response['message']['content']

                # 3. VERIFICĂM DACA AI-ul VREA SĂ FACĂ PROGRAMARE (JSON)
                if "{" in ai_response and "action" in ai_response:
                    try:
                        # Extragem JSON-ul din răspuns (în caz că mai are text pe lângă)
                        start = ai_response.find("{")
                        end = ai_response.rfind("}") + 1
                        json_str = ai_response[start:end]
                        
                        data = json.loads(json_str)
                        
                        if data.get("action") == "book":
                            print(f"📅 DETECTAT: Programare pentru {data['nume']} la {data['data']}")
                            
                            # === APELĂM GOOGLE CALENDAR ===
                            succes = create_appointment(f"Programare: {data['nume']}", data['data'])
                            
                            if succes:
                                ai_response = f"Gata {data['nume']}, am notat programarea în calendar. O zi bună!"
                                # Opțional: Resetăm istoria după programare reușită
                                # chat_history = [chat_history[0]] 
                            else:
                                ai_response = "Am o problemă tehnică și nu pot accesa calendarul momentan."

                    except Exception as e:
                        print(f"❌ Eroare procesare comandă: {e}")
                        ai_response = "Nu am înțeles exact data. Poți repeta, te rog?"

                # 4. Salvăm răspunsul final în istoric
                chat_history.append({'role': 'assistant', 'content': ai_response})

                # 5. Vorbim
                speak(ai_response)

            else:
                print("\r" + " " * 30 + "\r", end="", flush=True)

        except KeyboardInterrupt:
            print("\n✅ La revedere!")
            break
        except Exception as e:
            print(f"\n❌ Eroare: {e}")
            print("🔄 Reîncep ascultarea...\n")

if __name__ == "__main__":
    main()