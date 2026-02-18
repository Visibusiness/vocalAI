import os
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama
import subprocess
import json
import datetime
import asyncio
import edge_tts
import webrtcvad

# === IMPORT CALENDAR ===
try:
    from calendar_tool import create_appointment
except ImportError:
    print("⚠️ Nu am găsit calendar_tool.py!")
    def create_appointment(s, d): return False

# === CONFIG NVIDIA ===
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

# === SETTINGS ===
SAMPLE_RATE = 16000
WHISPER_MODEL = "distil-large-v3" # Recomandat pentru viteză
LLM_MODEL = "qwen2.5:7b"          # Recomandat pentru română (sau llama3.2)
VOICE_NAME = "ro-RO-AlinaNeural"  # Vocea ALINA
OUTPUT_FILENAME = "raspuns_visi.mp3"

print(f"⌛ Încărcare Whisper ({WHISPER_MODEL})...")
stt_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

# === SYSTEM PROMPT ===
now = datetime.datetime.now()
today_str = now.strftime("%Y-%m-%d %H:%M")
day_name = now.strftime("%A")

chat_history = [{
    'role': 'system',
    'content': (
        f'Ești Visi, recepționer virtual la un salon. Azi este {today_str} ({day_name}). '
        'Vorbește doar Română. '
        'Cere: Serviciu, Nume, Dată, Oră. '
        'Când e confirmat, generează DOAR JSON: '
        '{"action": "book", "nume": "Nume", "data": "YYYY-MM-DDTHH:MM:00"}'
    )
}]

# === AUDIO PLAYER (MP3 Support) ===
def get_mp3_player():
    # Căutăm playere capabile de MP3
    for player in ["ffplay", "mpv", "mpg123"]:
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            return player
    return None

AUDIO_PLAYER = get_mp3_player()

# =========================================================
# =====================  TTS + BARGE IN  ==================
# =========================================================

# Funcție helper pentru async TTS
async def _generate_audio(text):
    communicate = edge_tts.Communicate(text, VOICE_NAME)
    await communicate.save(OUTPUT_FILENAME)

def speak(text):
    try:
        print(f"👩 Agent (Alina): {text}")
        if not text:
            return

        # 1. Generare MP3 (Edge TTS)
        asyncio.run(_generate_audio(text))

        if not AUDIO_PLAYER:
            print("⚠️ Nu am găsit player audio (instalează ffmpeg)!")
            return

        # 2. Configurare comandă player
        cmd = []
        if AUDIO_PLAYER == "ffplay":
            # Ascundem fereastra și log-urile inutile
            cmd = [AUDIO_PLAYER, "-nodisp", "-autoexit", "-loglevel", "quiet", OUTPUT_FILENAME]
        else:
            cmd = [AUDIO_PLAYER, OUTPUT_FILENAME]

        # 3. Pornim player NON blocking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # === BARGE-IN STABIL ===
        # Ascultăm microfonul în timp ce vorbește Alina
        vad = webrtcvad.Vad(3)
        frame_duration = 20 # ms
        frame_size = int(SAMPLE_RATE * frame_duration / 1000) # 320 samples

        speech_frames_needed = 10   # Cât de repede întrerupe
        speech_counter = 0

        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype='int16') as stream:

            while process.poll() is None:
                # Citim din microfon
                frame, overflow = stream.read(frame_size)
                if overflow:
                    continue
                    
                frame_bytes = frame.tobytes()

                # 🔹 Filtru volum (ca să nu se audă pe ea însăși prea ușor)
                # Dacă ai difuzoare puternice, crește pragul (ex: 300, 500)
                volume = np.abs(frame).mean()
                if volume < 250: 
                    speech_counter = 0
                    continue

                try:
                    is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
                except:
                    continue

                if is_speech:
                    speech_counter += 1
                else:
                    speech_counter = 0

                # Dacă utilizatorul vorbește peste AI
                if speech_counter >= speech_frames_needed:
                    print("🔴 Întrerupt de utilizator!")
                    process.kill() # Oprim playerul
                    break

    except Exception as e:
        print(f"Eroare TTS: {e}")

# =========================================================
# =====================  RECORD VAD  ======================
# =========================================================

def record_until_silence(sample_rate=16000,
                         silence_duration=1.0, # Am crescut puțin timpul de așteptare
                         max_record_time=15):

    vad = webrtcvad.Vad(3)
    frame_duration = 20
    frame_size = int(sample_rate * frame_duration / 1000)

    silence_frames_needed = int(silence_duration * 1000 / frame_duration)
    max_total_frames = int(max_record_time * 1000 / frame_duration)

    print("[Aștept să vorbești...]", end="", flush=True)

    audio_buffer = []
    silence_counter = 0
    speech_detected = False
    total_frames = 0

    with sd.InputStream(samplerate=sample_rate,
                        channels=1,
                        dtype='int16') as stream:

        while True:
            frame, _ = stream.read(frame_size)
            frame_bytes = frame.tobytes()

            volume = np.abs(frame).mean()
            try:
                is_speech = vad.is_speech(frame_bytes, sample_rate)
            except:
                is_speech = False

            total_frames += 1

            # Așteptăm prima voce
            if not speech_detected:
                if is_speech and volume > 200:
                    print("\r[Ascult...]", end="", flush=True)
                    speech_detected = True
                    audio_buffer.append(frame.copy())
                continue

            # După ce începe vorbirea
            audio_buffer.append(frame.copy())

            if is_speech:
                silence_counter = 0
            else:
                silence_counter += 1

            if silence_counter > silence_frames_needed:
                break

            if total_frames > max_total_frames:
                break

    if len(audio_buffer) == 0:
        print("\r" + " " * 40 + "\r", end="")
        return None

    print("\r[Procesez...]              ")

    audio = np.concatenate(audio_buffer, axis=0)
    return audio

# =========================================================
# =====================  MAIN LOOP  =======================
# =========================================================

def main():
    if not AUDIO_PLAYER:
        print("\n❌ EROARE: Nu am găsit 'ffplay' sau 'mpv'.")
        print("Instalează: sudo apt install ffmpeg\n")
        return

    print(f"\n🚀 AGENT ACTIV (Alina) | {today_str}")
    print("🎤 Vorbește liber...\n")

    while True:
        try:
            recording = record_until_silence()
            if recording is None:
                continue

            audio_data = recording.flatten().astype(np.float32) / 32768.0

            segments, _ = stt_model.transcribe(
                audio_data,
                language="ro",
                beam_size=5
            )

            user_text = "".join([s.text for s in segments]).strip()
            if not user_text:
                continue

            print(f"👤 Client: {user_text}")

            chat_history.append({'role': 'user', 'content': user_text})
            response = ollama.chat(model=LLM_MODEL, messages=chat_history)
            ai_response = response['message']['content']

            # Detect booking JSON
            if "{" in ai_response and "action" in ai_response:
                try:
                    start = ai_response.find("{")
                    end = ai_response.rfind("}") + 1
                    json_str = ai_response[start:end]
                    data = json.loads(json_str)

                    if data.get("action") == "book":
                        succes = create_appointment(
                            f"Programare: {data['nume']}",
                            data['data']
                        )

                        if succes:
                            ai_response = f"Gata {data['nume']}, programarea a fost înregistrată."
                        else:
                            ai_response = "Nu pot accesa calendarul momentan."

                except:
                    ai_response = "Nu am înțeles data. Poți repeta?"

            chat_history.append({'role': 'assistant', 'content': ai_response})
            speak(ai_response)

        except KeyboardInterrupt:
            print("\n✅ La revedere!")
            break
        except Exception as e:
            print(f"\nEroare: {e}")
            print("Reîncep...\n")

if __name__ == "__main__":
    main()