import os
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ollama
import tempfile
import wave
import subprocess
from piper import PiperVoice

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
SAMPLE_RATE = 16000          # pentru Whisper STT
PIPER_SAMPLE_RATE = 22050    # pentru Piper TTS (schimbă la 16000 dacă modelul tău e 16k)
DURATION = 5                 # secunde înregistrare
WHISPER_MODEL = "large-v3"
LLM_MODEL = "llama3.2"
PIPER_MODEL = "ro_RO-mihai-medium.onnx"

print("⌛ Încărcare modele pe RTX 2060 Super...")
stt_model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
voice = PiperVoice.load(PIPER_MODEL)

print([m for m in dir(voice) if not m.startswith('_')])

# Prompt personalizat pentru Visi
chat_history = [
    {'role': 'system', 'content': 'Ești un asistent vocal prietenos. Utilizatorul se numește Visi. Răspunde scurt în română.'}
]

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
def speak(text):
    output_file = None
    try:
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

# === FUNCȚIE PRINCIPALĂ ===
def main():
    print(f"\n🚀 AGENT ACTIV | Salut, Visi!")
    print(f"🔊 Player audio: {AUDIO_PLAYER}")
    print("🎤 Te ascult... (Ctrl+C pentru oprire)\n")

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