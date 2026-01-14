"""
Text-to-Speech module using Piper TTS.
"""

from .tts_helper import speak_text
from .synthesizer import TTSSynthesizer
from .audio_player import play_audio_bytes
from .model_loader import (
    load_tts_synthesizer,
    find_tts_model,
    get_default_model_path,
    preload_tts_model,
    clear_cache,
)

__all__ = [
    'speak_text',
    'TTSSynthesizer',
    'play_audio_bytes',
    'load_tts_synthesizer',
    'find_tts_model',
    'get_default_model_path',
    'preload_tts_model',
    'clear_cache',
]

