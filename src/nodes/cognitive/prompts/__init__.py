"""
Prompt templates and formatters for LLM interactions.
"""

from .prompt_formatter import (
    format_intent_classification_prompt,
    format_emotional_advice_prompt
)
from .prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    EMOTIONAL_ADVICE_PROMPT
)

__all__ = [
    'format_intent_classification_prompt',
    'format_emotional_advice_prompt',
    'INTENT_CLASSIFICATION_PROMPT',
    'EMOTIONAL_ADVICE_PROMPT',
]

