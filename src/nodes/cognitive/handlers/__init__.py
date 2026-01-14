"""
Handlers for different intent types and responses.
"""

from .emotional_chat import get_emotional_advice
from .emotional_detector import is_emotional_statement
from .response_templates import (
    get_greeting_response,
    get_greeting_intent_data
)

__all__ = [
    'get_emotional_advice',
    'is_emotional_statement',
    'get_greeting_response',
    'get_greeting_intent_data',
]

