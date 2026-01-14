"""
Text-to-Speech helper utilities for intent responses.
"""
import logging
from typing import Optional, Callable, TYPE_CHECKING

from .model_loader import load_tts_synthesizer

if TYPE_CHECKING:
    from .synthesizer import TTSSynthesizer

logger = logging.getLogger(__name__)

# Global TTS synthesizer instance (lazy loaded)
_tts_synthesizer: Optional["TTSSynthesizer"] = None
_play_audio_bytes: Optional[Callable[[bytes], None]] = None


def _get_tts_synthesizer(use_cuda: Optional[bool] = None) -> Optional["TTSSynthesizer"]:
    """
    Get or initialize the global TTS synthesizer instance.
    
    Args:
        use_cuda: Whether to use CUDA.
        
    Returns:
        TTSSynthesizer instance or None if initialization fails.
    """

    global _tts_synthesizer
    
    if _tts_synthesizer is not None:
        return _tts_synthesizer
    
    try:
        _tts_synthesizer = load_tts_synthesizer(use_cuda=use_cuda)
        return _tts_synthesizer
    except Exception as e:
        logger.warning(f"Failed to initialize TTS synthesizer: {e}", exc_info=True)
        return None


def _get_play_audio_function() -> Optional[Callable[[bytes], None]]:

    global _play_audio_bytes
    
    if _play_audio_bytes is not None:
        return _play_audio_bytes
    
    try:
        from .audio_player import play_audio_bytes
        _play_audio_bytes = play_audio_bytes
        return _play_audio_bytes
    except Exception as e:
        logger.warning(f"Failed to load play_audio_bytes: {e}", exc_info=True)
        return None


def speak_text(text: str, use_cuda: bool = False, verbose: bool = False) -> bool:
    if not text or not text.strip():
        if verbose:
            logger.warning("Empty text provided to speak_text")
        return False
    
    try:
        synthesizer = _get_tts_synthesizer(use_cuda=use_cuda)
        if synthesizer is None:
            if verbose:
                logger.warning("TTS synthesizer not available")
            return False
        
        if verbose:
            logger.info(f"Speaking: {text}")
        
        # Synthesize text to audio bytes
        audio_bytes = synthesizer.synthesize_to_bytes(text)
        
        # Play audio
        play_audio_func = _get_play_audio_function()
        if play_audio_func is None:
            if verbose:
                logger.warning("play_audio_bytes function not available")
            return False
        
        play_audio_func(audio_bytes)
        
        if verbose:
            logger.info("✓ Audio played successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to speak text: {e}", exc_info=verbose)
        return False
