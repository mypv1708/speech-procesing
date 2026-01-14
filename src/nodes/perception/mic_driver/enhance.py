import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
from df.enhance import enhance

from .config import FRAME_DURATION_MS, INT16_MAX, MILLISECONDS_PER_SECOND

if TYPE_CHECKING:
    from libdf import DF

FRAME_DURATION_SEC = FRAME_DURATION_MS / MILLISECONDS_PER_SECOND

logger = logging.getLogger(__name__)

_INT16_MAX_INV = 1.0 / INT16_MAX  # Pre-computed for faster normalization


def convert_frames_to_tensor(recorded_frames: List[bytes]) -> torch.Tensor:
    """Convert PCM frames to normalized float32 tensor [1, T]."""
    if not recorded_frames:
        raise ValueError("recorded_frames cannot be empty")
    pcm = b"".join(recorded_frames)
    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32, copy=False) * _INT16_MAX_INV
    return torch.from_numpy(audio_np).unsqueeze(0)


def enhance_utterance(
    recorded_frames: List[bytes],
    model: torch.nn.Module,
    df_state: "DF",
    target_sr: int,
    raw_filepath: Optional[str] = None,
    enhanced_filepath: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[np.ndarray, int, Optional[str], Optional[str]]:
    if not recorded_frames:
        raise ValueError("recorded_frames cannot be empty")

    # Save raw audio if requested (non-blocking - failures are logged but don't stop processing)
    # DISABLED for performance - I/O blocking in hot path
    # if raw_filepath:
    #     try:
    #         save_wave(recorded_frames, raw_filepath)
    #         logger.info("Saved raw audio: %s", raw_filepath)
    #     except Exception:
    #         logger.exception("Failed to save raw audio")

    # Convert frames to tensor
    audio_tensor = convert_frames_to_tensor(recorded_frames)
    
    # Calculate actual audio duration from tensor (more accurate than frame count)
    num_samples = audio_tensor.shape[-1] if audio_tensor.ndim > 0 else 0
    audio_duration = num_samples / target_sr if num_samples > 0 and target_sr > 0 else len(recorded_frames) * FRAME_DURATION_SEC
    
    # Determine device (model's device if not specified)
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    inference_start = time.perf_counter()
    try:
        enhanced = enhance(model, df_state, audio_tensor)
    except Exception as e:
        logger.exception("Audio enhancement failed: %s", e)
        raise RuntimeError(f"Audio enhancement failed: {e}") from e

    inference_time = time.perf_counter() - inference_start

    # Log performance metrics (only if INFO level enabled)
    if logger.isEnabledFor(logging.INFO):
        speed_ratio = audio_duration / inference_time if inference_time > 0 else 0.0
        logger.info(
            "Inference completed: %.3fs (audio: %.2fs, speed: %.2fx real-time)",
            inference_time,
            audio_duration,
            speed_ratio,
        )

    # Convert enhanced output to numpy array
    if isinstance(enhanced, torch.Tensor):
        enhanced_np = enhanced.cpu().numpy()
        if enhanced_np.ndim == 2 and enhanced_np.shape[0] == 1:
            enhanced_np = enhanced_np.squeeze(0)  # [1, T] -> [T]
        if enhanced_np.dtype != np.float32:
            enhanced_np = enhanced_np.astype(np.float32, copy=False)
    else:
        enhanced_np = np.asarray(enhanced, dtype=np.float32)

    # Validate output
    if enhanced_np.size == 0:
        raise RuntimeError("Enhanced audio is empty")
    if not np.isfinite(enhanced_np).all():
        logger.warning("Enhanced audio contains non-finite values, clipping")
        enhanced_np = np.clip(enhanced_np, -1.0, 1.0)

    # Free GPU memory if using CUDA
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.empty_cache()

    # Save enhanced audio if requested (non-blocking - failures are logged but don't stop processing)
    # DISABLED for performance - I/O blocking in hot path
    # if enhanced_filepath:
    #     try:
    #         sf.write(enhanced_filepath, enhanced_np, target_sr)
    #         logger.info("Saved enhanced audio: %s", enhanced_filepath)
    #     except Exception:
    #         logger.exception("Failed to save enhanced audio")

    return enhanced_np, target_sr, raw_filepath, enhanced_filepath
