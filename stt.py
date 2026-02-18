import os
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama
<<<<<<< Updated upstream
import tempfile
import wave
import subprocess
from piper import PiperVoice
=======
import subprocess
import json
import datetime
import asyncio
import edge_tts
import webrtcvad
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
# === SETĂRI ===
SAMPLE_RATE = 16000          # pentru Whisper STT
PIPER_SAMPLE_RATE = 22050    # pentru Piper TTS (schimbă la 16000 dacă modelul tău e 16k)
DURATION = 5                 # secunde înregistrare
WHISPER_MODEL = "large-v3"
LLM_MODEL = "llama3.2"
PIPER_MODEL = "ro_RO-mihai-medium.onnx"

print("⌛ Încărcare modele pe RTX 2060 Super...")
stt_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
voice = PiperVoice.load(PIPER_MODEL)
=======
# === SETTINGS ===
SAMPLE_RATE = 16000
WHISPER_MODEL = "small" # Recomandat pentru viteză
LLM_MODEL = "qwen2.5:7b"          # Recomandat pentru română (sau llama3.2)
VOICE_NAME = "ro-RO-AlinaNeural"  # Vocea ALINA
OUTPUT_FILENAME = "raspuns_visi.mp3"

print(f"⌛ Încărcare Whisper ({WHISPER_MODEL})...")
stt_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
>>>>>>> Stashed changes

print([m for m in dir(voice) if not m.startswith('_')])

# Prompt personalizat pentru Visi
chat_history = [
    {'role': 'system', 'content': 'Ești un asistent vocal prietenos. Utilizatorul se numește Visi. Răspunde scurt în română.'}
]

<<<<<<< Updated upstream
# === DETECTARE PLAYER AUDIO ===
def get_audio_player():
    """Detectează automat playerul audio disponibil."""
    for player in ["paplay", "aplay", "ffplay"]:
        result = subprocess.run(["which", player], capture_output=True, text=True)
        if result.returncode == 0:
            return player
    return None

AUDIO_PLAYER = get_audio_player()
if not AUDIO_PLAYER:
    print("⚠️ Niciun player audio găsit! Instalează pulseaudio-utils: sudo apt install pulseaudio-utils")

# === FUNCȚIE SINTEZĂ VOCALĂ ===
=======
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

>>>>>>> Stashed changes
def speak(text):
    output_file = None
    try:
<<<<<<< Updated upstream
        print(f"🤖 Agent: {text}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            output_file = tmpfile.name

        # Colectăm raw PCM bytes din generator
        raw_audio = b"".join(voice.synthesize_stream_raw(text))

        if not raw_audio:
            print("⚠️ Piper nu a generat audio.")
            return

        # Scriem manual WAV
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(PIPER_SAMPLE_RATE)
            wf.writeframes(raw_audio)

        sd.stop()
        subprocess.run([AUDIO_PLAYER, output_file], check=True)

    except Exception as e:
        print(f"❌ Eroare: {e}")
    finally:
        if output_file and os.path.exists(output_file):
            os.unlink(output_file)
=======
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
>>>>>>> Stashed changes

# === FUNCȚIE PRINCIPALĂ ===
def main():
<<<<<<< Updated upstream
    print(f"\n🚀 AGENT ACTIV | Salut, Visi!")
    print(f"🔊 Player audio: {AUDIO_PLAYER}")
    print("🎤 Te ascult... (Ctrl+C pentru oprire)\n")
=======
    if not AUDIO_PLAYER:
        print("\n❌ EROARE: Nu am găsit 'ffplay' sau 'mpv'.")
        print("Instalează: sudo apt install ffmpeg\n")
        return

    print(f"\n🚀 AGENT ACTIV (Alina) | {today_str}")
    print("🎤 Vorbește liber...\n")
>>>>>>> Stashed changes

    while True:
        try:
            print("[Ascult...]", end="", flush=True)

            # Înregistrare voce
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            print("\r[Procesez...]", end="", flush=True)

            # Conversie pentru Whisper (float32 normalizat)
            audio_data = recording.flatten().astype(np.float32) / 32768.0

            segments, _ = stt_model.transcribe(
                audio_data,
                language="ro",
                beam_size=5,
                vad_filter=True
            )
            user_text = "".join([s.text for s in segments]).strip()

            if user_text:
                print(f"\r👤 Visi: {user_text}          ")

                chat_history.append({'role': 'user', 'content': user_text})

                response = ollama.chat(model=LLM_MODEL, messages=chat_history)
                ai_response = response['message']['content']

                chat_history.append({'role': 'assistant', 'content': ai_response})

                speak(ai_response)
            else:
                # Nimic detectat, ștergem linia
                print("\r" + " " * 30 + "\r", end="", flush=True)

        except KeyboardInterrupt:
            print("\n✅ La revedere, Visi!")
            break
        except Exception as e:
            print(f"\n❌ Eroare neașteptată: {e}")
            print("🔄 Reîncep ascultarea...\n")

if __name__ == "__main__":
    main()