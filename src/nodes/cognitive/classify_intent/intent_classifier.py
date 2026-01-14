import json
import logging
import os
import re
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Compile regex patterns at module level for performance
_JSON_CLEAN_PATTERN = re.compile(r'```json\s*')
_CODE_BLOCK_PATTERN = re.compile(r'```\s*')
_JSON_EXTRACT_PATTERN = re.compile(r'\{[^{}]*\}', re.DOTALL)

from ..utils.model_loader import download_model, _configure_gpu_layers, _load_model
from ..prompts.prompt_formatter import format_intent_classification_prompt
from ..parsers.navigate_parser import parse_navigate_command
from ..parsers.navigate_formatter import NavigateFormatter
from ..utils.logger import print_timing_info
from ..handlers.response_templates import get_greeting_intent_data, get_chat_intent_data
from ..handlers.emotional_detector import is_emotional_statement
from ..handlers.emotional_chat import get_emotional_advice
from ..config import INTENT_MAX_TOKENS, INTENT_TEMPERATURE, INTENT_STOP_SEQUENCES


def classify_intent(
    text_input: str,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    verbose: bool = True,
    use_tts: bool = True
) -> Dict:

    total_start = time.time()
    timing = {}

    # Early detection for greeting - no need to load model
    if _is_greeting(text_input):
        if verbose:
            print(f"\nInput: {text_input}")
            print("Detected greeting intent (using template response)")
        
        intent_data = get_greeting_intent_data(
            text_input,
            use_tts=use_tts,
            use_cuda=use_gpu,
            verbose=verbose
        )
        timing['total'] = time.time() - total_start
        
        if verbose:
            print(f"\nClassified intent: {intent_data.get('intent', 'unknown')}")
            print(f"Response: {intent_data.get('response', '')}")
            print(f"Result: {intent_data}\n")
        
        return intent_data

    if model_path is None or not os.path.exists(model_path):
        if verbose and model_path:
            print(f"File {model_path} not found, downloading model...")
        model_start = time.time()
        model_path = download_model(verbose=verbose)
        timing['model_download'] = time.time() - model_start
    elif verbose:
        print(f"✓ Using model: {model_path}")

    n_gpu_layers, gpu_info = _configure_gpu_layers(use_gpu, None)

    if verbose:
        print(f"Loading model{gpu_info}...")
    load_start = time.time()
    llm = _load_model(model_path, n_gpu_layers, use_cache=True)
    timing['model_load'] = time.time() - load_start

    prompt_start = time.time()
    prompt = format_intent_classification_prompt(text_input)
    timing['prompt_format'] = time.time() - prompt_start

    if verbose:
        print(f"\nInput: {text_input}")
        print("Classifying intent...")

    inference_start = time.time()
    output = llm(
        prompt,
        max_tokens=INTENT_MAX_TOKENS,
        temperature=INTENT_TEMPERATURE,
        stop=INTENT_STOP_SEQUENCES,
        echo=False
    )
    inference_time = time.time() - inference_start
    timing['inference'] = inference_time

    result = output['choices'][0]['text']

    if 'usage' in output:
        tokens_generated = output['usage'].get('completion_tokens', 0)
        if tokens_generated > 0:
            timing['tokens_per_second'] = tokens_generated / inference_time
            timing['tokens_generated'] = tokens_generated

    parse_start = time.time()
    intent_data = _parse_intent_json(result, text_input)
    
    if intent_data.get('intent') == 'navigate':
        navigate_data = parse_navigate_command(text_input)
        # Format navigate JSON to command string
        try:
            formatted_command = NavigateFormatter.format_command(navigate_data)
            navigate_data['formatted_command'] = formatted_command
        except Exception as e:
            logger.warning(f"Failed to format navigate command: {e}")
        intent_data = navigate_data
    elif intent_data.get('intent') == 'greeting':
        intent_data = get_greeting_intent_data(
            text_input,
            use_tts=use_tts,
            use_cuda=use_gpu,
            verbose=verbose
        )
    elif intent_data.get('intent') == 'follow':
        intent_data = {
            'intent': 'follow',
            'source_text': text_input
        }
    elif intent_data.get('intent') == 'stop_follow':
        intent_data = {
            'intent': 'stop_follow',
            'source_text': text_input
        }
    elif intent_data.get('intent') == 'chat':
        # Check if it's an emotional statement
        if is_emotional_statement(text_input):
            if verbose:
                print("Detected emotional statement, using LFM2-350M for advice")
            emotional_data = get_emotional_advice(
                text_input,
                use_gpu=use_gpu,
                verbose=verbose,
                use_tts=use_tts
            )
            intent_data = emotional_data
        else:
            # Regular chat - always provide response with TTS
            intent_data = get_chat_intent_data(
                text_input,
                use_tts=use_tts,
                use_cuda=use_gpu,
                verbose=verbose
            )
    else:
        intent_data = {
            'intent': intent_data.get('intent', 'chat'),
            'source_text': text_input
        }
    
    timing['parse'] = time.time() - parse_start

    timing['total'] = time.time() - total_start

    def format_intent_result(data: Dict) -> None:
        """Format and print intent classification result"""
        if data.get('intent') == 'navigate':
            print(f"\nClassified intent: {data.get('intent', 'unknown')}")
            print(f"JSON Result:\n{json.dumps(data, indent=2)}\n")
            # Print formatted command if available
            if 'formatted_command' in data:
                print("=" * 80)
                print("FORMATTED COMMAND:")
                print("=" * 80)
                print(repr(data['formatted_command']))  # Show with \n
                print("\nCommand (raw):")
                print(data['formatted_command'], end="")
                print("=" * 80 + "\n")
        elif data.get('intent') == 'greeting':
            print(f"\nClassified intent: {data.get('intent', 'unknown')}")
            print(f"Response: {data.get('response', '')}")
            print(f"Result: {data}\n")
        elif data.get('intent') == 'chat' and data.get('is_emotional'):
            print(f"\nClassified intent: {data.get('intent', 'unknown')} (emotional)")
            print(f"Advice: {data.get('response', '')}")
            print(f"Result: {data}\n")
        else:
            print(f"\nRaw output: {result}")
            print(f"Classified intent: {data.get('intent', 'unknown')}")
            print(f"Result: {data}\n")

    print_timing_info(timing, verbose, intent_data, format_intent_result)

    return intent_data


def _parse_intent_json(text: str, original_text: str) -> Dict[str, str]:
    """Parse JSON from model output"""
    text = _JSON_CLEAN_PATTERN.sub('', text)
    text = _CODE_BLOCK_PATTERN.sub('', text)
    text = text.strip()

    json_match = _JSON_EXTRACT_PATTERN.search(text)

    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)

            intent = data.get('intent', '').lower().strip()
            if intent not in ['greeting', 'chat', 'navigate', 'follow', 'stop_follow']:
                intent = _infer_intent(original_text)

            return {
                'intent': intent,
                'text': original_text
            }
        except json.JSONDecodeError:
            pass

    return {
        'intent': _infer_intent(original_text),
        'text': original_text
    }


def _is_greeting(text: str) -> bool:
    """Check if text is a greeting (early detection, no LLM needed)
    
    Args:
        text: Input text (can be already lowercased or original)
    """
    # Handle both lowercased and original text
    if text.islower():
        text_lower = text.strip()
    else:
        text_lower = text.lower().strip()
    words = text_lower.split()

    # Use tuple for immutable and slightly faster iteration
    greeting_keywords = (
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
        'good evening', 'good night', 'howdy', 'what\'s up', 'sup'
    )
    if any(text_lower.startswith(kw) or text_lower == kw for kw in greeting_keywords):
        return True
    if any(keyword in words[:2] for keyword in greeting_keywords):
        return True
    return False


def _infer_intent(text: str) -> str:
    """Infer intent from text if JSON parsing fails (English only)"""
    text_lower = text.lower().strip()
    words = text_lower.split()

    if _is_greeting(text_lower):
        return 'greeting'

    navigate_action_words = (
        'move', 'go', 'turn', 'walk', 'drive', 'step', 'navigate', 'stop'
    )
    navigate_direction_words = (
        'forward', 'backward', 'back', 'left', 'right', 'straight', 'ahead'
    )
    navigate_distance_words = ('meters', 'meter', 'm', 'feet', 'ft')

    stop_follow_keywords = ('stop following', 'stop follow', "don't follow", 'cancel follow', 'stop tracking', 'quit following', 'end follow')
    if any(keyword in text_lower for keyword in stop_follow_keywords):
        return 'stop_follow'
    if text_lower.startswith('stop') and ('follow' in text_lower or 'track' in text_lower):
        return 'stop_follow'

    follow_keywords = ('follow', 'track', 'following')
    if any(keyword in text_lower for keyword in follow_keywords):
        if 'me' in words or 'us' in words or text_lower.endswith('follow') or 'follow me' in text_lower:
            return 'follow'

    if text_lower in ['stop', 'halt', 'pause']:
        return 'navigate'

    has_navigate_action = any(
        word in text_lower for word in navigate_action_words
    )
    has_navigate_direction = any(
        word in text_lower for word in navigate_direction_words
    )
    has_navigate_distance = any(
        word in text_lower for word in navigate_distance_words
    )

    if has_navigate_action or (
        has_navigate_direction and has_navigate_distance
    ) or (has_navigate_direction and len(words) <= 3):
        return 'navigate'

    return 'chat'