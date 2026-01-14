"""
Emotional chat handler using LFM2-350M model from HuggingFace Transformers
"""
import time
from typing import Dict

import torch

from ..utils.model_loader import load_emotional_model
from ..prompts.prompt_formatter import format_emotional_advice_prompt
from ..config import (
    EMOTIONAL_MAX_NEW_TOKENS,
    EMOTIONAL_MIN_NEW_TOKENS,
    EMOTIONAL_TEMPERATURE,
    EMOTIONAL_DO_SAMPLE,
    EMOTIONAL_REPETITION_PENALTY,
    EMOTIONAL_ASSISTANT_MARKER,
    EMOTIONAL_ASSISTANT_MARKER_SHORT,
    EMOTIONAL_SENTENCE_END_THRESHOLD,
)
from nodes.actuator.tts.tts_helper import speak_text


def get_emotional_advice(
    text_input: str,
    use_gpu: bool = True,
    verbose: bool = True,
    use_tts: bool = True
) -> Dict:
    """
    Get advice for emotional/physical state using LFM2-350M model.
    
    Args:
        text_input: User's emotional statement
        use_gpu: Whether to use GPU
        verbose: Whether to print progress
        
    Returns:
        Dictionary with intent, source_text, and advice response
    """
    total_start = time.time()
    timing = {}
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if verbose:
        print(f"Loading emotional model...")
        print(f"use_gpu={use_gpu}, CUDA available={cuda_available}")
    elif use_gpu:
        print(f"Loading emotional model on GPU...")
        if not cuda_available:
            print(f"⚠ Warning: GPU requested but CUDA not available")
    
    load_start = time.time()
    model, tokenizer = load_emotional_model(use_gpu=use_gpu, verbose=verbose)
    timing['model_load'] = time.time() - load_start
    
    # Determine device - model should already be on correct device from cache/load
    # Only check device, don't move (model is already cached with correct device)
    try:
        first_param_device = next(model.parameters()).device
        device = str(first_param_device)
        if verbose and use_gpu and "cuda" not in device:
            print(f"⚠ Warning: Model is on {device}, expected GPU")
    except Exception:
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    prompt = format_emotional_advice_prompt(text_input)
    
    if verbose:
        print(f"\nInput: {text_input}")
        print(f"Generating advice on {device}...")
    
    inference_start = time.time()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EMOTIONAL_MAX_NEW_TOKENS,
                min_new_tokens=EMOTIONAL_MIN_NEW_TOKENS,
                temperature=EMOTIONAL_TEMPERATURE,
                do_sample=EMOTIONAL_DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=EMOTIONAL_REPETITION_PENALTY,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if EMOTIONAL_ASSISTANT_MARKER in generated_text:
            result = generated_text.split(EMOTIONAL_ASSISTANT_MARKER)[-1].strip()
        elif EMOTIONAL_ASSISTANT_MARKER_SHORT in generated_text:
            result = generated_text.split(EMOTIONAL_ASSISTANT_MARKER_SHORT)[-1].strip()
        else:
            result = generated_text.replace(prompt, "").strip()
        
        if result and not result.endswith(('.', '!', '?')):
            last_sentence_end = max(
                result.rfind('.'),
                result.rfind('!'),
                result.rfind('?')
            )
            if last_sentence_end > len(result) * EMOTIONAL_SENTENCE_END_THRESHOLD:
                result = result[:last_sentence_end + 1].strip()
        
        inference_time = time.time() - inference_start
        timing['inference'] = inference_time
        
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        if output_tokens > 0:
            timing['tokens_per_second'] = output_tokens / inference_time
            timing['tokens_generated'] = output_tokens
        
    except Exception as e:
        inference_time = time.time() - inference_start
        timing['inference'] = inference_time
        raise RuntimeError(f"Failed to generate advice: {e}") from e
    
    timing['total'] = time.time() - total_start
    
    if verbose:
        print(f"\nAdvice generated in {timing['total']:.3f}s")
        print(f"Response: {result}\n")
    
    if use_tts and result:
        speak_text(result, use_cuda=use_gpu, verbose=verbose)
    
    return {
        'intent': 'chat',
        'source_text': text_input,
        'response': result,
        'is_emotional': True,
        'timing': timing
    }
