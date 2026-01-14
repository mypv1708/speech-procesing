import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from eff_word_net import RATE as EFF_WORD_NET_RATE
from eff_word_net.engine import HotwordDetector

from .config import (
    HOTWORD_BLOCKSIZE,
    HOTWORD_NAME,
    HOTWORD_SLIDING_WINDOW_SECS,
    HOTWORD_WINDOW_LENGTH_SECS,
    PIP_SOUND_FILE,
    WAKE_WORD_CHANNELS,
    WAKE_WORD_DTYPE,
    WAKE_WORD_MIN_AUDIO_LEVEL,
    WAKE_WORD_SLEEP_MS,
)
from .audio import play_audio_file

logger = logging.getLogger(__name__)


def wait_for_wake_word(detector: HotwordDetector) -> bool:
    try:
        window_samples = int(HOTWORD_WINDOW_LENGTH_SECS * EFF_WORD_NET_RATE)  # 1.5s window
        step_samples = int(HOTWORD_SLIDING_WINDOW_SECS * EFF_WORD_NET_RATE)  # 0.75s step

        audio_buffer = deque(maxlen=window_samples)
        samples_since_last = 0
        wake_word_detected = threading.Event()

        logger.info("Listening for wake word '%s'...", HOTWORD_NAME.replace("_", " "))

        def audio_callback(indata, frames, time_info, status):
            nonlocal samples_since_last
            if status:
                logger.warning("Audio stream status: %s", status)
            if wake_word_detected.is_set():
                return
            
            mono_audio = indata[:, 0] if indata.ndim > 1 else indata
            audio_buffer.extend(mono_audio)
            samples_since_last += len(mono_audio)
            
            if samples_since_last >= step_samples and len(audio_buffer) >= window_samples:
                samples_since_last = 0
                frame = np.fromiter(audio_buffer, dtype=np.float32, count=len(audio_buffer))
                
                if np.max(np.abs(frame)) > WAKE_WORD_MIN_AUDIO_LEVEL:  # Skip if too quiet
                    result = detector.scoreFrame(frame)
                    if result and result.get("match", False):
                        confidence = result.get("confidence", 0.0)
                        logger.info("HOTWORD DETECTED | confidence=%.3f", confidence)
                        wake_word_detected.set()
                        def play_pip():
                            try:
                                play_audio_file(PIP_SOUND_FILE)
                            except Exception as e:
                                logger.warning("Failed to play pip sound: %s", e)
                        threading.Thread(target=play_pip, daemon=True).start()

        try:
            with sd.InputStream(
                samplerate=EFF_WORD_NET_RATE,
                channels=WAKE_WORD_CHANNELS,
                dtype=WAKE_WORD_DTYPE,
                blocksize=HOTWORD_BLOCKSIZE,
                callback=audio_callback,
            ):
                while not wake_word_detected.is_set():
                    sd.sleep(WAKE_WORD_SLEEP_MS)
            return True
        except sd.PortAudioError as e:
            logger.error("PortAudio error during wake word detection: %s", e)
            raise RuntimeError(f"Audio stream error: {e}") from e
    except KeyboardInterrupt:
        logger.info("Wake word detection interrupted by user")
        return False
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Wake word detection failed: %s", e)
        raise RuntimeError(f"Wake word detection failed: {e}") from e


