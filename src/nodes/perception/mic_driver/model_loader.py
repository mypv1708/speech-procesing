import logging
import os
from typing import Tuple

import torch
from df.enhance import init_df
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net.engine import HotwordDetector
from libdf import DF

from .config import (
    DF_LOG_LEVEL,
    DF_POST_FILTER,
    HOTWORD_NAME,
    HOTWORD_REFERENCE_FILE,
    HOTWORD_RELAXATION_TIME,
    HOTWORD_THRESHOLD,
)

logger = logging.getLogger(__name__)


def load_deepfilternet() -> Tuple[torch.nn.Module, DF, int, str]:
    """Load DeepFilterNet model for noise reduction."""
    logger.info("Loading DeepFilterNet...")
    try:
        model, df_state, _ = init_df(
            model_base_dir=None,
            post_filter=DF_POST_FILTER,
            log_level=DF_LOG_LEVEL
        )
        
        target_sr = df_state.sr()
        device = next(model.parameters()).device
        device_str = str(device)
        
        logger.info(
            "DeepFilterNet loaded successfully: %d Hz, device=%s",
            target_sr,
            device_str
        )
        
        return model, df_state, target_sr, device_str
    except Exception as e:
        logger.exception("Failed to load DeepFilterNet: %s", e)
        raise RuntimeError(f"DeepFilterNet initialization failed: {e}") from e


def load_wake_word_detector() -> HotwordDetector:
    """Load EfficientWord-Net detector for wake word 'hello_robot'."""
    logger.info("Loading EfficientWord-Net detector...")
    try:
        base_model = Resnet50_Arc_loss()
        ref_path = os.path.abspath(HOTWORD_REFERENCE_FILE)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Wake word reference file not found: {ref_path}")
        
        detector = HotwordDetector(
            hotword=HOTWORD_NAME,
            model=base_model,
            reference_file=ref_path,
            threshold=HOTWORD_THRESHOLD,
            relaxation_time=HOTWORD_RELAXATION_TIME,
        )
        
        logger.info("Wake word detector loaded successfully")
        return detector
    except Exception as e:
        logger.exception("Failed to load wake word detector: %s", e)
        raise RuntimeError(f"Wake word detector initialization failed: {e}") from e


def load_all_models() -> Tuple[torch.nn.Module, DF, int, str, HotwordDetector]:
    logger.info("Loading all models...")
    model, df_state, target_sr, device = load_deepfilternet()
    wake_word_detector = load_wake_word_detector()
    logger.info("All models loaded successfully")
    return model, df_state, target_sr, device, wake_word_detector