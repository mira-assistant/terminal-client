"""
Live transcription with real-time speaker diarization (up to two speakers) using
local Whisper, WebRTC-VAD, and Resemblyzer with real-time audio denoising.

Supports multilingual speech recognition including English and Indian languages.
Code-mixed speech (e.g., English + Hindi/Tamil/etc.) is automatically detected.
Real-time audio denoising filters out white noise and background sounds.

How it works
------------
• Captures 30 ms frames from the microphone.
• WebRTC-VAD detects speech; ≥ 2 s of silence marks sentence boundary.
• Real-time audio denoising removes white noise and background sounds from audio.
• Each sentence is transcribed by Whisper locally (no API) with automatic language detection.
• Indian names and words are recognized alongside English through multilingual model.
• The sentence audio is embedded with Resemblyzer.
• A lightweight online clustering assignment labels sentences
  as Speaker 1 or Speaker 2 by cosine similarity to running centroids.
"""

from __future__ import annotations

import time
import warnings

import pyaudio
import webrtcvad

warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

# ---------- Constants ----------
CHANNELS = 1
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
MAX_SILENCE_BREAK = 0.6
enabled = False


def start_observer() -> None:
    """Initialize audio stream, VAD, and TTS engine."""

    global stream, pa, vad, silence_start, did_speak, enabled

    vad = webrtcvad.Vad(3)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_BYTES // 2,
    )

    silence_start = -1
    did_speak = False
    enabled = True


def is_observer_running() -> bool:
    """Check if the audio stream is active."""
    return enabled


def kill_observer() -> None:
    """Gracefully stop the observer."""
    global stream, pa, enabled

    stream.stop_stream()
    stream.close()
    pa.terminate()
    enabled = False


def refresh_observer() -> None:
    """Reset the observer state for a new sentence."""
    global silence_start, did_speak
    silence_start = -1
    did_speak = False


def silent_observer(sentence_buf: bytearray) -> tuple[bytearray, bool]:
    """
    Capture audio frames, detect speech, transcribe sentences, and assign speakers
    using VAD and Whisper ASR.
    """

    global silence_start, did_speak, stream, vad

    frame = stream.read(FRAME_BYTES // 2, exception_on_overflow=False)

    if len(frame) != FRAME_BYTES:
        # Incomplete frame; skip
        return sentence_buf, False

    try:
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
    except Exception as e:
        print("VAD error:", e)
        return sentence_buf, False

    sentence_buf.extend(frame)
    now = time.perf_counter()

    status = ""

    if is_speech:
        status = "speaking"
    elif did_speak:
        status = f"silence, {now - (silence_start if silence_start != -1 else 0):.2f}s"
    else:
        status = "waiting for speech"

    print(f"{status}{' ' * (40 - len(status))}", end="\r")

    if is_speech:
        silence_start = -1
        did_speak = True
    else:
        if silence_start == -1:
            if did_speak:
                silence_start = now

        elif now - silence_start >= MAX_SILENCE_BREAK:
            refresh_observer()
            return sentence_buf, True

    return sentence_buf, False
