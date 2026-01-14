import os
import time
from typing import Optional, Tuple, Any, Dict

from llama_cpp import Llama
from huggingface_hub import hf_hub_download, list_repo_files

from ..config import (
    MODEL_REPO,
    DEFAULT_QUANTIZATION,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_N_THREADS,
    MAX_GPU_LAYERS,
    MODEL_SEARCH_PATHS,
    EMOTIONAL_MODEL_REPO,
    _model_cache
)

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
        # Download model if needed
        model_path = download_model(verbose=verbose)
        
        # Configure GPU layers
        n_gpu_layers, gpu_info = _configure_gpu_layers(use_gpu, None)
        
        if verbose:
            logger.info(f"Loading model{gpu_info}...")
        
        # Load model (will be cached)
        _load_model(model_path, n_gpu_layers, use_cache=True)
        
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
        # Load model (will be cached)
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
        # Preload intent classification model
        preload_intent_model(use_gpu=use_gpu, verbose=verbose)
        
        # Preload emotional chat model
        preload_emotional_model(use_gpu=use_gpu, verbose=verbose)
        
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("All Cognitive models preloaded in %.2f seconds", total_time)
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("Failed to preload cognitive models: %s", e, exc_info=True)
        raise RuntimeError(f"Cognitive models preload failed: {e}") from e


def _find_gguf_file(quantization: str, gguf_files: list) -> Optional[str]:
    target_quant = quantization.upper()

    for f in gguf_files:
        if target_quant in f.upper():
            return f

    full_name = f"functiongemma-270m-it-{quantization}.gguf"
    if full_name in gguf_files:
        return full_name

    short_name = f"{quantization}.gguf"
    if short_name in gguf_files:
        return short_name

    return None


def _find_local_model(quantization: str = DEFAULT_QUANTIZATION) -> Optional[str]:
    target_quant = quantization.upper()

    for search_path in MODEL_SEARCH_PATHS:
        if not os.path.exists(search_path):
            continue

        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.gguf') and target_quant in file.upper():
                    return os.path.join(root, file)

    return None


def download_model(quantization: str = DEFAULT_QUANTIZATION, verbose: bool = True) -> str:
    local_model = _find_local_model(quantization)
    if local_model and os.path.exists(local_model):
        if verbose:
            print(f"✓ Found local model: {local_model}")
        return local_model

    if verbose:
        print(f"Checking {MODEL_REPO}...")

    preferred_file = f"functiongemma-270m-it-{quantization}.gguf"

    try:
        repo_files = list_repo_files(repo_id=MODEL_REPO, repo_type="model")
        gguf_files = [f for f in repo_files if f.endswith('.gguf')]

        if not gguf_files:
            raise ValueError(f"No GGUF files found in {MODEL_REPO}")

        if verbose:
            print(f"Found {len(gguf_files)} GGUF files")

        if preferred_file in gguf_files:
            selected_file = preferred_file
        else:
            selected_file = _find_gguf_file(quantization, gguf_files)

        if selected_file is None:
            selected_file = gguf_files[0]
            if verbose:
                print(f"Using {selected_file} instead of {quantization}")
        elif verbose:
            print(f"✓ Using: {selected_file}")

    except Exception as e:
        if verbose:
            print(f"Error listing files: {e}, trying {preferred_file}")
        selected_file = preferred_file

    if verbose:
        print(f"Downloading {selected_file}...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=selected_file,
        local_dir="./models"
    )

    if verbose:
        print(f"✓ Downloaded: {model_path}")

    return model_path


def _configure_gpu_layers(use_gpu: bool, n_gpu_layers: Optional[int]) -> Tuple[int, str]:
    if not use_gpu:
        return 0, " (CPU only)"

    if n_gpu_layers is None:
        return MAX_GPU_LAYERS, " (GPU - all layers)"
    elif n_gpu_layers > 0:
        return n_gpu_layers, f" (GPU - {n_gpu_layers} layers)"
    else:
        return 0, " (CPU only)"


def _load_model(model_path: str, n_gpu_layers: int, use_cache: bool = True) -> Llama:
    cache_key = f"{model_path}_{n_gpu_layers}"

    if use_cache and cache_key in _model_cache:
        return _model_cache[cache_key]

    start_time = time.time()
    try:
        try:
            import llama_cpp
            has_cuda = hasattr(llama_cpp, 'llama_supports_gpu_offload') and llama_cpp.llama_supports_gpu_offload()
        except Exception:
            has_cuda = False
        
        if n_gpu_layers > 0 and not has_cuda:
            print("⚠ Warning: llama_cpp not built with CUDA support, falling back to CPU")
            n_gpu_layers = 0
        
        llm = Llama(
            model_path=model_path,
            n_ctx=DEFAULT_CONTEXT_SIZE,
            n_threads=DEFAULT_N_THREADS,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        load_time = time.time() - start_time

        device_str = "CPU" if n_gpu_layers == 0 else f"GPU ({n_gpu_layers} layers)"
        print(f"✓ Model loaded on {device_str} - {load_time:.2f}s")

        if use_cache:
            _model_cache[cache_key] = llm

        return llm
    except Exception as e:
        if n_gpu_layers > 0:
            print(f"GPU error: {e}, fallback to CPU...")
            return _load_model(model_path, 0, use_cache)
        else:
            raise RuntimeError(f"Failed to load model: {e}") from e


def load_emotional_model(use_gpu: bool = True, verbose: bool = False) -> Tuple[Any, Any]:
    """
    Load LFM2-350M emotional model from HuggingFace Transformers.
    """
    cache_key = f"emotional_{EMOTIONAL_MODEL_REPO}_{use_gpu}"
    
    if cache_key in _emotional_model_cache and cache_key in _emotional_tokenizer_cache:
        if verbose:
            # Check device of cached model
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
    
    if verbose:
        print(f"use_gpu={use_gpu}, torch.cuda.is_available()={cuda_available}")
        if not cuda_available:
            print(f"  PyTorch version: {torch.__version__}")
            try:
                import subprocess
                nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                           capture_output=True, text=True, timeout=2)
                if nvidia_smi.returncode == 0:
                    gpu_name = nvidia_smi.stdout.strip().split('\n')[0]
                    print(f"  GPU detected: {gpu_name}")
                    print(f"  ⚠ PyTorch cannot access GPU - possible CUDA version mismatch")
                    print(f"  Solution: Reinstall PyTorch with matching CUDA version")
            except Exception:
                pass
    
    device = "cuda" if use_gpu and cuda_available else "cpu"
    
    if use_gpu and not cuda_available:
        print(f"⚠ Warning: GPU requested but CUDA not available. Using CPU instead.")
        print(f"  This may be due to CUDA version mismatch between PyTorch and system.")
        print(f"  Check: nvidia-smi, torch.cuda.is_available(), CUDA installation")
    
    if verbose:
        print(f"Loading {EMOTIONAL_MODEL_REPO} from HuggingFace...")
        print(f"Loading tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTIONAL_MODEL_REPO)
        
        if verbose:
            print(f"Loading model on {device}...")
        
        if device == "cuda":
            # Use device_map="auto" for automatic GPU allocation
            model = AutoModelForCausalLM.from_pretrained(
                EMOTIONAL_MODEL_REPO,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Load on CPU
            model = AutoModelForCausalLM.from_pretrained(
                EMOTIONAL_MODEL_REPO,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        load_time = time.time() - start_time
        time_str = f" - {load_time:.2f}s" if verbose else ""
        
        # Check actual device of model
        if device == "cuda":
            try:
                # Get first parameter's device to verify GPU usage
                first_param_device = next(model.parameters()).device
                actual_device = str(first_param_device)
                if "cuda" in actual_device:
                    print(f"✓ Emotional model loaded on GPU ({actual_device}){time_str}")
                else:
                    print(f"⚠ Warning: Emotional model loaded on {actual_device.upper()} instead of GPU{time_str}")
                    print(f"  Attempting to move model to GPU...")
                    try:
                        model = model.to("cuda")
                        print(f"✓ Model moved to GPU successfully")
                    except Exception as move_error:
                        print(f"✗ Failed to move model to GPU: {move_error}")
            except Exception:
                print(f"✓ Emotional model loaded on {device.upper()}{time_str}")
        else:
            print(f"✓ Emotional model loaded on {device.upper()}{time_str}")
        
        _emotional_model_cache[cache_key] = model
        _emotional_tokenizer_cache[cache_key] = tokenizer
        
        return model, tokenizer
    
    except Exception as e:
        if use_gpu and torch.cuda.is_available():
            if verbose:
                print(f"GPU error: {e}, fallback to CPU...")
            return load_emotional_model(use_gpu=False, verbose=verbose)
        raise RuntimeError(f"Failed to load emotional model: {e}") from e