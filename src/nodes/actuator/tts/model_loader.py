"""
TTS Model Loader - Load and manage TTS models
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .synthesizer import TTSSynthesizer

from .config import (
    TTS_MODEL_DIR,
    TTS_DEFAULT_LANGUAGE,
    TTS_DEFAULT_VOICE,
    TTS_DEFAULT_MODEL_NAME,
    TTS_MODEL_SEARCH_PATHS,
    TTS_USE_CUDA_DEFAULT,
    TTS_CUDA_FALLBACK_TO_CPU,
    TTS_VOICE_OPTIONS,
)

logger = logging.getLogger(__name__)

_tts_model_cache: Dict[str, Any] = {}


def find_tts_model(
    voice: Optional[str] = None,
    language: Optional[str] = None,
    quality: Optional[str] = None,
) -> Optional[Path]:
    """
    Find TTS model file in search paths.
    """
    # Determine model name
    if quality and quality in TTS_VOICE_OPTIONS:
        model_name = f"{TTS_VOICE_OPTIONS[quality]}.onnx"
    elif voice:
        model_name = f"{voice}.onnx" if not voice.endswith(".onnx") else voice
    else:
        model_name = TTS_DEFAULT_MODEL_NAME
    
    # Determine language directory
    lang = language or TTS_DEFAULT_LANGUAGE
    
    # Search in all paths
    for search_path in TTS_MODEL_SEARCH_PATHS:
        search_dir = Path(search_path)
        
        # Try language subdirectory first
        model_path = search_dir / lang / model_name
        if model_path.exists():
            logger.debug(f"Found TTS model: {model_path}")
            return model_path
        
        # Try direct path
        model_path = search_dir / model_name
        if model_path.exists():
            logger.debug(f"Found TTS model: {model_path}")
            return model_path
    
    logger.warning(f"TTS model not found: {model_name} (language: {lang})")
    return None


def get_default_model_path() -> Path:

    model_path = find_tts_model()
    if model_path is None:
        default_path = TTS_MODEL_DIR / TTS_DEFAULT_LANGUAGE / TTS_DEFAULT_MODEL_NAME
        raise FileNotFoundError(
            f"Default TTS model not found: {default_path}\n"
            f"Searched in: {TTS_MODEL_SEARCH_PATHS}"
        )
    return model_path


def load_tts_synthesizer(
    model_path: Optional[Union[str, Path]] = None,
    use_cuda: Optional[bool] = None,
    voice: Optional[str] = None,
    language: Optional[str] = None,
    quality: Optional[str] = None,
    use_cache: bool = True,
) -> "TTSSynthesizer":
    """
    Load TTS synthesizer with caching support.
    """

    try:
        from .synthesizer import TTSSynthesizer
    except ImportError:
        raise ImportError(
            "piper-tts not installed. Install with: pip install piper-tts"
        )
    
    # Determine model path
    if model_path is None:
        model_path = find_tts_model(voice=voice, language=language, quality=quality)
        if model_path is None:
            model_path = get_default_model_path()
    
    model_path = Path(model_path)
    
    # Determine CUDA usage
    if use_cuda is None:
        use_cuda = TTS_USE_CUDA_DEFAULT
    
    # Create cache key
    cache_key = f"{model_path}_{use_cuda}"
    
    # Check cache
    if use_cache and cache_key in _tts_model_cache:
        logger.debug(f"Using cached TTS synthesizer: {cache_key}")
        return _tts_model_cache[cache_key]
    
    # Load synthesizer
    logger.info(f"Loading TTS synthesizer: {model_path} (CUDA: {use_cuda})")
    
    try:
        synthesizer = TTSSynthesizer(
            model_path=str(model_path),
            use_cuda=use_cuda,
        )
        
        # Cache the instance
        if use_cache:
            _tts_model_cache[cache_key] = synthesizer
        
        logger.info("TTS synthesizer loaded successfully")
        return synthesizer
        
    except Exception as e:
        # Try fallback to CPU if CUDA fails
        if use_cuda and TTS_CUDA_FALLBACK_TO_CPU:
            logger.warning(f"CUDA load failed: {e}. Falling back to CPU...")
            return load_tts_synthesizer(
                model_path=model_path,
                use_cuda=False,
                use_cache=use_cache,
            )
        raise RuntimeError(f"Failed to load TTS synthesizer: {e}") from e


def preload_tts_model(use_cuda: Optional[bool] = None) -> "TTSSynthesizer":
    """
    Preload TTS model upfront to avoid first-use delay.
    """
    import time
    
    logger.info("=" * 60)
    logger.info("Preloading TTS model...")
    logger.info("=" * 60)
    
    start_time = time.perf_counter()
    
    try:
        synthesizer = load_tts_synthesizer(use_cuda=use_cuda, use_cache=True)
        load_time = time.perf_counter() - start_time
        
        logger.info("=" * 60)
        logger.info("TTS model preloaded successfully in %.2f seconds", load_time)
        logger.info("=" * 60)
        
        return synthesizer
    except Exception as e:
        logger.error("Failed to preload TTS model: %s", e, exc_info=True)
        raise RuntimeError(f"TTS model preload failed: {e}") from e


def clear_cache() -> None:
    """Clear TTS model cache."""
    global _tts_model_cache
    _tts_model_cache.clear()
    logger.debug("TTS model cache cleared")
