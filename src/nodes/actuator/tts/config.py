"""
TTS (Text-to-Speech) configuration.
"""
from pathlib import Path

# Model Configuration
TTS_MODEL_DIR = Path(__file__).parent / "model"
TTS_DEFAULT_LANGUAGE = "en"
TTS_DEFAULT_VOICE = "en_US-lessac-low"
TTS_DEFAULT_MODEL_NAME = f"{TTS_DEFAULT_VOICE}.onnx"

# Model Paths
TTS_MODEL_SEARCH_PATHS = [
    str(TTS_MODEL_DIR),
    "./text-to-speech/model",
    "./models/tts",
]

# Audio Playback Configuration
AUDIO_PLAYBACK_TIMEOUT = 30 
AUDIO_CHUNK_SIZE = 1024

# System Audio Players (in order of preference)
AUDIO_PLAYERS = ["aplay", "paplay", "play"]

# CUDA Configuration
TTS_USE_CUDA_DEFAULT = True
TTS_CUDA_FALLBACK_TO_CPU = True

# Model Quality Options
TTS_VOICE_OPTIONS = {
    "high": "en_US-lessac-high",
    "medium": "en_US-lessac-medium",
    "low": "en_US-lessac-low",
}

