"""
Model loader for speech recognition - loads all models upfront.
"""
import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from speechbrain.inference.speaker import SpeakerRecognition

from .config import SPEAKER_DEVICE, STT_DEVICE

if TYPE_CHECKING:
    from .speech_to_text import SpeechToTextEngine

logger = logging.getLogger(__name__)


def load_speaker_model() -> Optional[SpeakerRecognition]:
    """
    Load speaker recognition model.
    
    Returns:
        SpeakerRecognition model or None if loading fails.
    """
    device = SPEAKER_DEVICE if SPEAKER_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info("Loading Speaker Recognition model on %s...", device)
        start_time = time.perf_counter()
        
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        
        load_time = time.perf_counter() - start_time
        logger.info("Speaker Recognition model loaded on %s in %.2f seconds", device, load_time)
        return model
    except Exception as e:
        logger.warning("Failed to load Speaker Recognition model (speaker verification disabled): %s", e)
        return None


def load_stt_model(preload: bool = True) -> "SpeechToTextEngine":
    """
    Load Speech-to-Text model.
    
    Args:
        preload: Whether to preload the model immediately (default: True).
        
    Returns:
        SpeechToTextEngine instance.
        
    Raises:
        RuntimeError: If model loading fails.
    """
    logger.info("Initializing Speech-to-Text engine...")
    if STT_DEVICE:
        device = STT_DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from .speech_to_text import SpeechToTextEngine

        engine = SpeechToTextEngine(device)
        if preload:
            logger.info("Preloading STT model to avoid first-use delay...")
            start_time = time.perf_counter()
            engine.preload()
            load_time = time.perf_counter() - start_time
            logger.info("Speech-to-Text engine ready (preloaded on %s in %.2f seconds)", device, load_time)
        else:
            logger.info("Speech-to-Text engine ready (lazy loading on %s)", device)
        return engine
    except Exception as e:
        logger.exception("Failed to initialize STT engine: %s", e)
        raise RuntimeError(f"STT engine initialization failed: {e}") from e


def preload_all_models() -> Tuple[Optional[SpeakerRecognition], "SpeechToTextEngine"]:
    """
    Preload all speech recognition models upfront.
    
    This function loads both speaker recognition and STT models
    to avoid delays during runtime.
    
    Returns:
        Tuple of (speaker_model, stt_engine).
        speaker_model may be None if loading fails.
        
    Raises:
        RuntimeError: If STT model loading fails (STT is required).
    """
    logger.info("=" * 60)
    logger.info("Preloading all Speech Recognition models...")
    logger.info("=" * 60)
    
    total_start = time.perf_counter()
    
    # Load speaker model
    speaker_model = load_speaker_model()
    
    # Load STT model (required, will raise if fails)
    stt_engine = load_stt_model(preload=True)
    
    total_time = time.perf_counter() - total_start
    
    logger.info("=" * 60)
    logger.info("All Speech Recognition models loaded in %.2f seconds", total_time)
    logger.info("=" * 60)
    
    return speaker_model, stt_engine
