"""
Logging and timing utilities
"""
from typing import Dict, Optional, Any, Callable


def print_timing_info(
    timing: Dict[str, float],
    verbose: bool = True,
    additional_output: Optional[Any] = None,
    result_formatter: Optional[Callable[[Any], None]] = None
) -> None:
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print("TIMING INFO:")
    print(f"{'='*60}")
    
    if 'model_download' in timing:
        print(f"Model download: {timing['model_download']:.3f}s")
    
    if 'model_load' in timing:
        print(f"Model load:     {timing.get('model_load', 0):.3f}s")
    
    if 'prompt_format' in timing:
        print(f"Prompt format:   {timing.get('prompt_format', 0):.3f}s")
    
    if 'inference' in timing:
        print(f"Inference:       {timing['inference']:.3f}s")
    
    if 'tokens_per_second' in timing:
        print(f"Tokens/sec:      {timing['tokens_per_second']:.2f}")
        print(f"Tokens:          {timing.get('tokens_generated', 0)}")
    
    if 'parse' in timing:
        print(f"Parse:           {timing.get('parse', 0):.3f}s")
    
    if 'total' in timing:
        print(f"Total time:      {timing['total']:.3f}s")
    
    print(f"{'='*60}")
    
    if result_formatter:
        result_formatter(additional_output)
    elif additional_output is not None:
        print(f"\nOutput:\n{additional_output}\n")

