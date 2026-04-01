# -*- coding: utf-8 -*-
"""
test_client.py

Text-based test client for VocalAI. Converts a text string to WAV via edge-tts,
sends it to the /voice endpoint, and streams + plays back the MP3 response.

Requirements (already in requirements.txt + client.py deps):
    pip install edge-tts httpx pydub sounddevice

Usage:
    python test_client.py --text "Vreau să fac o programare"
    python test_client.py --text "Anulează programarea de mâine la 10" --session abc123
    python test_client.py --text "Am o programare joi la 14:00?" --new
    python test_client.py --text "Ce programare am?" --phone +40721000000 --play
"""

import argparse
import asyncio
import io
import os
import queue
import struct
import threading
import uuid

import edge_tts
import httpx
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

SERVER_URL = "https://w36j0lsgy0qk93-8000.proxy.runpod.net/voice"
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


def iter_audio_chunks(response):
    """
    Parse length-prefixed MP3 chunks from a streaming response.

    Protocol: 4-byte little-endian uint32 length + <length> MP3 bytes.
    A length of 0 signals end of stream.
    """
    buf = b""
    for data in response.iter_bytes():
        buf += data
        while len(buf) >= 4:
            length = struct.unpack('<I', buf[:4])[0]
            if length == 0:
                return
            if len(buf) < 4 + length:
                break
            yield buf[4:4 + length]
            buf = buf[4 + length:]


def send_and_play(wav_bytes: bytes, session_id: str, play: bool) -> None:
    """POST WAV to /voice, print AI text, and optionally play each audio chunk."""
    audio_queue = queue.Queue()

    def playback_worker():
        while True:
            item = audio_queue.get()
            if item is None:
                break
            samples, rate = item
            sd.play(samples, samplerate=rate)
            sd.wait()

    if play:
        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

    try:
        with httpx.stream(
            "POST", SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"session_id": session_id},
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                print(f"Server error {response.status_code}: {response.text}")
                return

            first = True
            for mp3_data in iter_audio_chunks(response):
                if first:
                    print("Receiving response...")
                    first = False
                if play:
                    seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    samples /= 2 ** (seg.sample_width * 8 - 1)
                    if seg.channels == 2:
                        samples = samples.reshape((-1, 2))
                    audio_queue.put((samples, seg.frame_rate))
    finally:
        if play:
            audio_queue.put(None)
            playback_thread.join()


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
    parser.add_argument("--phone", default=None, help="Patient phone number — used as session ID (e.g. +40721000000)")
    parser.add_argument("--session", default=None, help="Session ID (default: reuse last)")
    parser.add_argument("--new", action="store_true", help="Start a fresh session")
    parser.add_argument("--play", action="store_true", help="Play the audio response through speakers")
    args = parser.parse_args()

    if args.phone:
        session_id = args.phone
    else:
        session_id = load_or_create_session(args.new, args.session)

    print(f"session={session_id} | text=\"{args.text}\"")
    print("Synthesizing input audio...")
    wav_bytes = asyncio.run(text_to_wav(args.text))

    print("Sending to server...")
    send_and_play(wav_bytes, session_id, play=args.play)


if __name__ == "__main__":
    main()
