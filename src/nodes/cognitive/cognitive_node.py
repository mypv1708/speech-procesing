"""
Cognitive Node - Processes transcribed text through intent classification.
"""
import logging
import re
from typing import Optional, Dict, Any

from .classify_intent.intent_classifier import classify_intent

logger = logging.getLogger(__name__)


def normalize_stt_text(text: str) -> str:
    """
    Normalize STT output text by removing punctuation and normalizing whitespace.
    
    Examples:
        "Turn left, 48." -> "turn left 48"
        "Go forward, 2.5 meters." -> "go forward 2.5 meters"
        "Hello, how are you?" -> "hello how are you"
    
    Args:
        text: Raw text from STT (may contain punctuation).
        
    Returns:
        Normalized text (lowercase, no punctuation, normalized whitespace).
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation but keep numbers and spaces
    # Keep: letters, numbers, spaces
    # Remove: punctuation marks (.,!?;:()[]{}\"'`-)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class CognitiveNode:
    """
    Cognitive node that processes text input through intent classification.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        verbose: bool = True,
        use_tts: bool = True,
        preload_models: bool = True,
    ):
        """
        Initialize CognitiveNode with all models preloaded.
        
        Args:
            use_gpu: Whether to use GPU for models.
            verbose: Whether to print progress.
            use_tts: Whether to enable TTS for responses.
            preload_models: Whether to preload all models upfront (default: True).
        """
        logger.info("Initializing CognitiveNode...")
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.use_tts = use_tts
        
        # Preload all cognitive models upfront
        if preload_models:
            try:
                from .utils.model_loader import preload_all_cognitive_models
                preload_all_cognitive_models(use_gpu=use_gpu, verbose=verbose)
            except Exception as e:
                logger.warning(f"Failed to preload cognitive models: {e}. Models will be loaded lazily.")
        
        logger.info("CognitiveNode initialized successfully")
    
    def process_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process transcribed text through intent classification.
        
        Args:
            text: Transcribed text from STT.
            
        Returns:
            Intent classification result dictionary or None if processing fails.
        """
        if not text or not text.strip():
            logger.debug("Empty text received, skipping processing")
            return None
        
        try:
            # Normalize text: remove punctuation, lowercase, normalize whitespace
            original_text = text
            normalized_text = normalize_stt_text(text)
            
            if normalized_text != original_text:
                logger.debug("Text normalized: '%s' -> '%s'", original_text, normalized_text)
            
            logger.info("Processing text: '%s'", normalized_text)
            
            result = classify_intent(
                text_input=normalized_text,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
                use_tts=self.use_tts
            )
            
            intent = result.get('intent', 'unknown')
            logger.info("Intent classified: %s", intent)
            
            return result
            
        except Exception as e:
            logger.error("Failed to process text: %s", e, exc_info=True)
            return None
