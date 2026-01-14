import logging
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch
    from eff_word_net.engine import HotwordDetector
    from libdf import DF

from .mic_driver.model_loader import load_all_models
from .mic_driver.recording import run_recording_loop
from .mic_driver.wake_word import wait_for_wake_word
from .speech_recognition_node import SpeechRecognitionNode

logger = logging.getLogger(__name__)


class MicDriverNode:
    def __init__(self, use_gpu: bool = True, use_tts: bool = True):
        """
        Initialize MicDriverNode with full pipeline.
        
        Args:
            use_gpu: Whether to use GPU for models.
            use_tts: Whether to enable TTS for responses.
        """
        logger.info("Initializing MicDriverNode...")
        try:
            # Load perception models (wake word, audio enhancement)
            self.model, self.df_state, self.target_sr, self.device, self.wake_word_detector = load_all_models()
            
            # Initialize speech recognition (STT) - preloads models
            self.speech_recognition = SpeechRecognitionNode()
            
            # Preload TTS model if enabled
            if use_tts:
                from nodes.actuator.tts.model_loader import preload_tts_model
                try:
                    preload_tts_model(use_cuda=use_gpu)
                except Exception as e:
                    logger.warning(f"TTS preload failed (will use lazy loading): {e}")
            
            # Initialize cognitive processing (intent classification)
            from nodes.cognitive.cognitive_node import CognitiveNode
            self.cognitive = CognitiveNode(
                use_gpu=use_gpu,
                verbose=True,
                use_tts=use_tts
            )
            
            logger.info("MicDriverNode initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize MicDriverNode: %s", e)
            raise

    def run(self) -> None:
        logger.info("Starting mic driver main loop...")
        try:
            while True:
                logger.debug("Waiting for wake word...")
                detected = wait_for_wake_word(self.wake_word_detector)
                
                if not detected:
                    logger.debug("Wake word detection cancelled or failed")
                    continue
                
                logger.info("Wake word detected! Starting recording loop...")
                result = run_recording_loop(
                    model=self.model,
                    df_state=self.df_state,
                    target_sr=self.target_sr,
                    device=self.device,
                    on_utterance=self._on_utterance
                )
                
                if result is not None:
                    audio, sr = result
                    logger.info("Recording completed: %d samples at %d Hz", len(audio), sr)
                    
        except KeyboardInterrupt:
            logger.info("MicDriverNode interrupted by user")
        except Exception as e:
            logger.exception("MicDriverNode error: %s", e)
            raise

    def _on_utterance(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Process audio utterance through the full pipeline:
        1. Speech-to-Text (STT)
        2. Intent Classification (Cognitive)
        3. Response (TTS if enabled)
        
        Args:
            audio: Audio data.
            sample_rate: Sample rate of audio.
            
        Returns:
            False to continue recording loop.
        """
        # Step 1: Speech-to-Text
        text = self.speech_recognition.process_audio(audio, sample_rate)
        
        if not text:
            logger.debug("No transcription received, skipping cognitive processing")
            return False
        
        # Step 2: Intent Classification and Response
        result = self.cognitive.process_text(text)
        
        if result:
            intent = result.get('intent', 'unknown')
            logger.info("Intent processing completed: %s", intent)
        else:
            logger.warning("Cognitive processing returned no result")
        
        return False

def main():
    from config.logging_config import setup_logging
    
    setup_logging()
    
    try:
        node = MicDriverNode()
        node.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

