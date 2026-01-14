import logging
import os
import wave
from typing import List, Tuple

import pyaudio
import webrtcvad

from .config import AUDIO_PLAYBACK_CHUNK_SIZE, CHANNELS, FORMAT, FRAME_SIZE, RATE, SAMPLE_WIDTH, VAD_MODE

logger = logging.getLogger(__name__)


def init_audio_stream() -> Tuple[pyaudio.PyAudio, webrtcvad.Vad, pyaudio.Stream]:
    """Initialize audio input stream with VAD (Voice Activity Detection)."""
    pa = pyaudio.PyAudio()
    vad = webrtcvad.Vad(VAD_MODE)
    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )
        return pa, vad, stream
    except OSError as e:
        pa.terminate()
        raise RuntimeError(f"Failed to open audio input stream: {e}") from e
    except Exception as e:
        pa.terminate()
        raise RuntimeError(f"Audio stream initialization failed: {e}") from e


def save_wave(frames: List[bytes], filename: str) -> None:
    """Save PCM frames to WAV file."""
    if not frames:
        raise ValueError("frames cannot be empty")
    
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
    except Exception as e:
        logger.exception("Failed to save WAV file %s: %s", filename, e)
        raise


def play_audio_file(filepath: str) -> None:
    """Play WAV file through audio output."""
    pa = None
    stream = None
    try:
        with wave.open(filepath, "rb") as wf:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            
            data = wf.readframes(AUDIO_PLAYBACK_CHUNK_SIZE)
            while data:
                stream.write(data)
                data = wf.readframes(AUDIO_PLAYBACK_CHUNK_SIZE)
            stream.stop_stream()
    except FileNotFoundError:
        logger.warning("Audio file not found: %s", filepath)
    except Exception as e:
        logger.exception("Failed to play audio file %s: %s", filepath, e)
        raise RuntimeError(f"Audio playback failed: {e}") from e
    finally:
        if stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa:
            try:
                pa.terminate()
            except Exception:
                pass