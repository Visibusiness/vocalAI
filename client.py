"""
client.py

Records audio from the local microphone, sends it to the VocalAI /voice endpoint,
and plays back the MP3 response.

Requirements:
    pip install sounddevice soundfile httpx pydub
    sudo apt install ffmpeg   # or: brew install ffmpeg (macOS)

Usage:
    python client.py
    python client.py --session my_session_id
    python client.py --duration 8
"""

import argparse
import io
import os
import tempfile
import uuid
import wave
from urllib.parse import unquote

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

SERVER_URL = "https://gj4u6gqf1pn91j-8000.proxy.runpod.net/voice"

SAMPLE_RATE = 16000   # Hz — Whisper works best at 16 kHz
CHANNELS = 1
RECORD_SECONDS = 6    # how long to record per turn


def record_audio(duration: int) -> bytes:
    """Record from the default microphone and return raw WAV bytes."""
    print(f"🎙  Recording for {duration} seconds... (speak now)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("   Done recording.")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def send_and_play(wav_bytes: bytes, session_id: str) -> None:
    """Send WAV bytes to the server and play the MP3 response."""
    print(f"   Sending to server (session={session_id})...")
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"session_id": session_id},
        )

    if response.status_code != 200:
        print(f"Server error {response.status_code}: {response.text}")
        return

    content_type = response.headers.get("content-type", "")
    if "audio" not in content_type:
        print(f"Unexpected response (content-type: {content_type}):")
        print(response.text[:500])
        return

    # Print the AI text response
    ai_text = response.headers.get("x-ai-text", "")
    if ai_text:
        print(f"AI: {unquote(ai_text)}")

    # Save MP3 to a temp file (ffmpeg needs a seekable file, not a pipe)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        # Decode MP3 to raw PCM via pydub, then play with sounddevice
        audio_segment = AudioSegment.from_mp3(tmp_path)
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (audio_segment.sample_width * 8 - 1)  # normalise to [-1, 1]
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
        print("Playing response...")
        sd.play(samples, samplerate=audio_segment.frame_rate)
        sd.wait()
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="VocalAI voice client")
    parser.add_argument(
        "--session",
        default=uuid.uuid4().hex[:8],
        help="Session ID (default: random). Use the same ID to continue a conversation.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=RECORD_SECONDS,
        help=f"Recording duration in seconds (default: {RECORD_SECONDS})",
    )
    args = parser.parse_args()

    print(f"\nVocalAI Client  |  session={args.session}  |  server={SERVER_URL}")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            input("Press Enter to record...")
            wav_bytes = record_audio(args.duration)
            send_and_play(wav_bytes, args.session)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
