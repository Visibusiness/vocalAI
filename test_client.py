# -*- coding: utf-8 -*-
"""
test_client.py

Text-based test client for VocalAI. Converts a text string to WAV via edge-tts,
sends it to the /voice endpoint, and prints the AI reply. No microphone needed.

Requirements (already in requirements.txt + client.py deps):
    pip install edge-tts httpx pydub sounddevice

Usage:
    python test_client.py --text "Vreau să fac o programare"
    python test_client.py --text "Anulează programarea de mâine la 10" --session abc123
    python test_client.py --text "Am o programare joi la 14:00?" --new
"""

import argparse
import asyncio
import io
import os
import tempfile
import uuid
import wave
from urllib.parse import unquote

import edge_tts
import httpx
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

SERVER_URL = "https://enwpc4da4zg5nw-8000.proxy.runpod.net/voice"
SESSION_FILE = os.path.join(os.path.dirname(__file__), ".session_id")

# Romanian male voice for test input (sounds different from AI response voice)
TTS_VOICE = "ro-RO-EmilNeural"
SAMPLE_RATE = 16000


async def text_to_wav(text: str) -> bytes:
    """Synthesize text to MP3 via edge-tts, then convert to 16kHz mono WAV."""
    mp3_buf = io.BytesIO()

    communicate = edge_tts.Communicate(text, TTS_VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])

    mp3_buf.seek(0)
    audio = AudioSegment.from_mp3(mp3_buf)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)

    wav_buf = io.BytesIO()
    audio.export(wav_buf, format="wav")
    return wav_buf.getvalue()


def send(wav_bytes: bytes, session_id: str) -> str:
    """POST WAV to /voice, print and return the AI text reply."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"session_id": session_id},
        )

    if response.status_code != 200:
        print(f"Server error {response.status_code}: {response.text}")
        return ""

    ai_text = unquote(response.headers.get("x-ai-text", ""))
    print(f"AI: {ai_text}")
    return ai_text


def play_response(mp3_bytes: bytes) -> None:
    """Play the MP3 response through speakers."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp_path = tmp.name
    try:
        audio = AudioSegment.from_mp3(tmp_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (audio.sample_width * 8 - 1)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    finally:
        os.unlink(tmp_path)


def load_or_create_session(new: bool, override: str | None) -> str:
    if override:
        return override
    if new or not os.path.exists(SESSION_FILE):
        session_id = uuid.uuid4().hex[:8]
        with open(SESSION_FILE, "w") as f:
            f.write(session_id)
        return session_id
    with open(SESSION_FILE) as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(description="VocalAI text-based test client")
    parser.add_argument("--text", required=True, help="Romanian text to send as patient input")
    parser.add_argument("--session", default=None, help="Session ID (default: reuse last)")
    parser.add_argument("--new", action="store_true", help="Start a fresh session")
    parser.add_argument("--play", action="store_true", help="Also play the audio response")
    args = parser.parse_args()

    session_id = load_or_create_session(args.new, args.session)
    print(f"session={session_id} | text=\"{args.text}\"")

    print("Synthesizing input audio...")
    wav_bytes = asyncio.run(text_to_wav(args.text))

    print("Sending to server...")
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"session_id": session_id},
        )

    if response.status_code != 200:
        print(f"Server error {response.status_code}: {response.text}")
        return

    ai_text = unquote(response.headers.get("x-ai-text", ""))
    print(f"\nAI: {ai_text}\n")

    if args.play:
        play_response(response.content)


if __name__ == "__main__":
    main()
