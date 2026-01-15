import os
import time
from typing import Optional, Tuple, Any, Dict

from ..config import (
    MODEL_REPO,
    EMOTIONAL_MODEL_REPO,
    DEFAULT_CONTEXT_SIZE,
)

_intent_model_cache: Dict[str, Any] = {}
_intent_tokenizer_cache: Dict[str, Any] = {}
_emotional_model_cache: Dict[str, Any] = {}
_emotional_tokenizer_cache: Dict[str, Any] = {}


def preload_intent_model(use_gpu: bool = True, verbose: bool = False) -> None:
    """
    Preload FunctionGemma model for intent classification.
    
    Args:
        use_gpu: Whether to use GPU.
        verbose: Whether to print progress.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Preloading Intent Classification model (FunctionGemma)...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        load_intent_model(use_gpu=use_gpu, verbose=verbose)
        load_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Intent Classification model preloaded successfully in %.2f seconds", load_time)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Failed to preload intent model: %s", e, exc_info=True)
        raise RuntimeError(f"Intent model preload failed: {e}") from e


def preload_emotional_model(use_gpu: bool = True, verbose: bool = False) -> None:
    """
    Preload LFM2-350M emotional model.
    
    Args:
        use_gpu: Whether to use GPU.
        verbose: Whether to print progress.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Preloading Emotional Chat model (LFM2-350M)...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        load_emotional_model(use_gpu=use_gpu, verbose=verbose)
        load_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Emotional Chat model preloaded successfully in %.2f seconds", load_time)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Failed to preload emotional model: %s", e, exc_info=True)
        raise RuntimeError(f"Emotional model preload failed: {e}") from e


def preload_all_cognitive_models(use_gpu: bool = True, verbose: bool = False) -> None:
    """
    Preload all cognitive models (FunctionGemma and LFM2-350M).
    
    Args:
        use_gpu: Whether to use GPU.
        verbose: Whether to print progress.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Preloading all Cognitive models...")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    try:
        preload_intent_model(use_gpu=use_gpu, verbose=verbose)
        preload_emotional_model(use_gpu=use_gpu, verbose=verbose)
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("All Cognitive models preloaded in %.2f seconds", total_time)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Failed to preload cognitive models: %s", e, exc_info=True)
        raise RuntimeError(f"Cognitive models preload failed: {e}") from e


def load_intent_model(use_gpu: bool = True, verbose: bool = False) -> Tuple[Any, Any]:
    """
    Load FunctionGemma-270M-IT model from HuggingFace Transformers.
    """
    cache_key = f"intent_{MODEL_REPO}_{use_gpu}"
    
    if cache_key in _intent_model_cache and cache_key in _intent_tokenizer_cache:
        if verbose:
            try:
                device = str(next(_intent_model_cache[cache_key].parameters()).device)
                print(f"✓ Using cached intent model (device: {device})")
            except Exception:
                print("✓ Using cached intent model")
        return _intent_model_cache[cache_key], _intent_tokenizer_cache[cache_key]
    
    start_time = time.time()
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import HfFolder
    except ImportError as e:
        raise RuntimeError("transformers and torch are required. Install with: pip install transformers torch") from e
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = "cuda" if use_gpu and cuda_available else "cpu"
    
    if verbose:
        print(f"use_gpu={use_gpu}, CUDA available={cuda_available}")
        print(f"Loading {MODEL_REPO} from HuggingFace...")
    
    # Get HuggingFace token if needed
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = HfFolder.get_token()
        except Exception:
            token = None
    
    if not token and verbose:
        print("⚠ No HuggingFace token found. Model may require authentication.")
        print(f"   Accept license at: https://huggingface.co/{MODEL_REPO}")
        print("   Then run: huggingface-cli login")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=token,
        use_fast=True
    )
    
    # Load model
    if verbose:
        print(f"Loading model on {device}...")
    
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    if token:
        model_kwargs["token"] = token
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            device_map="auto",
            **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            device_map=None,
            **model_kwargs
        )
        model = model.to(device)
    
    load_time = time.time() - start_time
    
    if verbose:
        try:
            actual_device = str(next(model.parameters()).device)
            time_str = f" - {load_time:.2f}s"
            if "cuda" in actual_device:
                print(f"✓ Intent model loaded on GPU ({actual_device}){time_str}")
            else:
                print(f"✓ Intent model loaded on {actual_device.upper()}{time_str}")
        except Exception:
            print(f"✓ Intent model loaded on {device.upper()} - {load_time:.2f}s")
    
    # Cache
    _intent_model_cache[cache_key] = model
    _intent_tokenizer_cache[cache_key] = tokenizer
    
    return model, tokenizer


def load_emotional_model(use_gpu: bool = True, verbose: bool = False) -> Tuple[Any, Any]:
    """
    Load LFM2-350M emotional model from HuggingFace Transformers.
    """
    cache_key = f"emotional_{EMOTIONAL_MODEL_REPO}_{use_gpu}"
    
    if cache_key in _emotional_model_cache and cache_key in _emotional_tokenizer_cache:
        if verbose:
            try:
                cached_model = _emotional_model_cache[cache_key]
                cached_device = str(next(cached_model.parameters()).device)
                print(f"✓ Using cached emotional model (device: {cached_device})")
            except Exception:
                print(f"✓ Using cached emotional model")
        return _emotional_model_cache[cache_key], _emotional_tokenizer_cache[cache_key]
    
    start_time = time.time()
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers and torch are required for emotional model. Install with: pip install transformers torch") from e
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = "cuda" if use_gpu and cuda_available else "cpu"
    
    if verbose:
        print(f"use_gpu={use_gpu}, CUDA available={cuda_available}")
        print(f"Loading {EMOTIONAL_MODEL_REPO} from HuggingFace...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMOTIONAL_MODEL_REPO)
    
    # Load model
    if verbose:
        print(f"Loading model on {device}...")
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            EMOTIONAL_MODEL_REPO,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            EMOTIONAL_MODEL_REPO,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    
    load_time = time.time() - start_time
    
    if verbose:
        try:
            actual_device = str(next(model.parameters()).device)
            time_str = f" - {load_time:.2f}s"
            if "cuda" in actual_device:
                print(f"✓ Emotional model loaded on GPU ({actual_device}){time_str}")
            else:
                print(f"✓ Emotional model loaded on {actual_device.upper()}{time_str}")
        except Exception:
            print(f"✓ Emotional model loaded on {device.upper()} - {load_time:.2f}s")
    
    # Cache
    _emotional_model_cache[cache_key] = model
    _emotional_tokenizer_cache[cache_key] = tokenizer
    
    return model, tokenizer
