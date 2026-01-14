"""
Simple TTS synthesizer wrapper for Piper TTS.
"""
from __future__ import annotations

import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional, Union

from .config import TTS_CUDA_FALLBACK_TO_CPU

logger = logging.getLogger(__name__)


class TTSSynthesizer:
    """Text-to-Speech synthesizer using Piper (English)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        self._voice = None
        self._model_path = model_path
        self._use_cuda = use_cuda

    def _ensure_loaded(self) -> None:
        """Lazy load the voice model."""
        if self._voice is not None:
            return

        try:
            from piper.voice import PiperVoice
        except ImportError:
            raise ImportError(
                "piper-tts not installed. Install with: pip install piper-tts"
            )

        if self._model_path is None:
            from .model_loader import get_default_model_path
            self._model_path = get_default_model_path()

        logger.info(f"Loading TTS model: {self._model_path}")
        try:
            # Try to load with CUDA if requested
            if self._use_cuda:
                try:
                    # Suppress ONNX Runtime CUDA warnings temporarily
                    old_env = os.environ.get('ORT_LOGGING_LEVEL', None)
                    os.environ['ORT_LOGGING_LEVEL'] = '3'  # Only show errors
                    
                    try:
                        self._voice = PiperVoice.load(
                            model_path=str(self._model_path),
                            config_path=None,
                            use_cuda=True,
                        )
                        logger.info("TTS model loaded successfully with CUDA")
                        return
                    finally:
                        # Restore original logging level
                        if old_env is not None:
                            os.environ['ORT_LOGGING_LEVEL'] = old_env
                        elif 'ORT_LOGGING_LEVEL' in os.environ:
                            del os.environ['ORT_LOGGING_LEVEL']
                            
                except Exception as cuda_error:
                    if TTS_CUDA_FALLBACK_TO_CPU:
                        # Suppress ONNX Runtime error messages when falling back
                        logger.warning(
                            "CUDA not available for TTS (missing libcublasLt.so.12). "
                            "Falling back to CPU. This is normal if CUDA libraries are not installed."
                        )
                        self._use_cuda = False
                    else:
                        raise
            
            # Load with CPU
            self._voice = PiperVoice.load(
                model_path=str(self._model_path),
                config_path=None,
                use_cuda=False,
            )
            logger.info("TTS model loaded successfully with CPU")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise

    def synthesize_to_bytes(self, text: str) -> bytes:

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self._ensure_loaded()

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self._voice.synthesize_wav(text, wav_file)

            wav_buffer.seek(0)
            return wav_buffer.read()
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise RuntimeError(f"Failed to synthesize audio: {e}") from e

    # def synthesize_to_file(self, text: str, output_path: Union[str, Path]) -> Path:
    #     """Synthesize text to audio and save to file (currently unused)."""
    #     if not text or not text.strip():
    #         raise ValueError("Text cannot be empty")
    #
    #     self._ensure_loaded()
    #
    #     output_path = Path(output_path)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     try:
    #         with wave.open(str(output_path), "wb") as wav_file:
    #             self._voice.synthesize_wav(text, wav_file)
    #         return output_path
    #     except Exception as e:
    #         logger.error(f"TTS synthesis to file failed: {e}")
    #         raise RuntimeError(f"Failed to synthesize audio to file: {e}") from e

