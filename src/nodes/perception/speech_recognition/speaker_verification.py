import logging
import time
from typing import Optional

import numpy as np
import torch
import torchaudio

from .config import ENROLL_FILE, SPEAKER_THRESHOLD, STT_SAMPLE_RATE

logger = logging.getLogger(__name__)


class SpeakerVerifier:
    def __init__(self, speaker_model, enroll_wavs: Optional[torch.Tensor] = None):
        self.speaker_model = speaker_model
        self.enroll_wavs = enroll_wavs
        self.threshold = SPEAKER_THRESHOLD

    @classmethod
    def load_enrollment(cls, enroll_file: str = ENROLL_FILE) -> Optional[torch.Tensor]:
        try:
            wavs, sr = torchaudio.load(enroll_file)
            if wavs.shape[0] > 1:
                wavs = wavs.mean(dim=0, keepdim=True)
            if sr != STT_SAMPLE_RATE:
                wavs = torchaudio.functional.resample(wavs, sr, STT_SAMPLE_RATE)
            return wavs
        except Exception as e:
            logger.exception("Failed to load enrollment voice from %s", enroll_file)
            return None

    def verify(self, audio: np.ndarray, sample_rate: int) -> bool:
        if self.speaker_model is None or self.enroll_wavs is None:
            return False

        try:
            audio_tensor = torch.from_numpy(audio.astype(np.float32, copy=False))
            if sample_rate != STT_SAMPLE_RATE:
                audio_tensor = audio_tensor.unsqueeze(0)
                audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, STT_SAMPLE_RATE)
                audio_tensor = audio_tensor.squeeze(0)
            
            test_wavs = audio_tensor.unsqueeze(0)
            
            inference_start = time.perf_counter()
            score, prediction = self.speaker_model.verify_batch(
                self.enroll_wavs, 
                test_wavs, 
                threshold=self.threshold
            )
            inference_time = time.perf_counter() - inference_start
            
            score_val = float(score[0].item()) if len(score) > 0 else float(score)
            is_match = bool(prediction[0].item()) if len(prediction) > 0 else bool(prediction)
            
            audio_duration = len(audio) / sample_rate
            speed_ratio = audio_duration / inference_time if inference_time > 0 else 0.0
            
            if is_match:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "Speaker verification PASSED: score=%.4f (threshold=%.4f, inference=%.3fs, audio=%.2fs, speed=%.2fx)",
                        score_val,
                        self.threshold,
                        inference_time,
                        audio_duration,
                        speed_ratio
                    )
            else:
                logger.warning(
                    "Speaker verification FAILED: score=%.4f (threshold=%.4f, inference=%.3fs, audio=%.2fs, speed=%.2fx)",
                    score_val,
                    self.threshold,
                    inference_time,
                    audio_duration,
                    speed_ratio
                )
            
            return is_match
            
        except Exception:
            logger.exception("Speaker verification error")
            return False
