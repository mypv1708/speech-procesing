from typing import Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

# FunctionGemma Model Configuration (Transformers)
MODEL_REPO = "google/functiongemma-270m-it"
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.1

# Intent Classification Configuration
INTENT_MAX_TOKENS = 64
INTENT_TEMPERATURE = 0.0
INTENT_STOP_SEQUENCES = ["<end_of_turn>", "\n\n"]

# Emotional Model Configuration (Transformers)
EMOTIONAL_MODEL_REPO = "LiquidAI/LFM2-350M"
EMOTIONAL_MAX_NEW_TOKENS = 128
EMOTIONAL_MIN_NEW_TOKENS = 20
EMOTIONAL_TEMPERATURE = 0.7
EMOTIONAL_DO_SAMPLE = True
EMOTIONAL_REPETITION_PENALTY = 1.1
EMOTIONAL_ASSISTANT_MARKER = "Assistant: Here's what you should do:"
EMOTIONAL_ASSISTANT_MARKER_SHORT = "Assistant:"
EMOTIONAL_SENTENCE_END_THRESHOLD = 0.5

# Navigation Validation Limits
MAX_DISTANCE = 1000.0  # Maximum distance in meters
MAX_ANGLE = 360.0      # Maximum angle in degrees

# Internal Caches
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}
