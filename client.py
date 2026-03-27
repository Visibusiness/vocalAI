"""
client.py

Records audio from the local microphone, sends it to the VocalAI /voice endpoint,
and plays back the streamed MP3 response sentence by sentence.

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
import queue
import struct
import threading
import uuid
import wave

import httpx
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

SERVER_URL = "https://8j0nj9eqagvpet-8000.proxy.runpod.net/voice"

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


def send_and_play(wav_bytes: bytes, session_id: str) -> None:
    """Send WAV bytes to the server and play each sentence chunk as it arrives."""
    print(f"   Sending to server (session={session_id})...")

    # Playback queue: main thread feeds decoded PCM, worker thread plays it
    audio_queue = queue.Queue()

    def playback_worker():
        while True:
            item = audio_queue.get()
            if item is None:
                break
            samples, rate = item
            sd.play(samples, samplerate=rate)
            sd.wait()

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    try:
        with httpx.stream(
            "POST", SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"session_id": session_id},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                print(f"Server error {response.status_code}: {response.text}")
                return

            first_chunk = True
            for mp3_data in iter_audio_chunks(response):
                if first_chunk:
                    print("Playing response...")
                    first_chunk = False
                audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                samples /= 2 ** (audio_segment.sample_width * 8 - 1)  # normalise to [-1, 1]
                if audio_segment.channels == 2:
                    samples = samples.reshape((-1, 2))
                audio_queue.put((samples, audio_segment.frame_rate))
    finally:
        audio_queue.put(None)
        playback_thread.join()


SESSION_FILE = os.path.join(os.path.dirname(__file__), ".session_id")


def load_or_create_session(new: bool) -> str:
    if new or not os.path.exists(SESSION_FILE):
        session_id = uuid.uuid4().hex[:8]
        with open(SESSION_FILE, "w") as f:
            f.write(session_id)
        return session_id
    with open(SESSION_FILE) as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser(description="VocalAI voice client")
    parser.add_argument(
        "--phone",
        default=None,
        help="Patient phone number — used as session ID (e.g. +40721000000)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID (default: reuse last session from .session_id file)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a brand new session (forget conversation history)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=RECORD_SECONDS,
        help=f"Recording duration in seconds (default: {RECORD_SECONDS})",
    )
    args = parser.parse_args()

    if args.phone:
        session_id = args.phone
    elif args.session:
        session_id = args.session
    else:
        session_id = load_or_create_session(args.new)

    print(f"\nVocalAI Client  |  session={session_id}  |  server={SERVER_URL}")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            input("Press Enter to record...")
            wav_bytes = record_audio(args.duration)
            print(f"[session] {session_id}")
            send_and_play(wav_bytes, session_id)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
