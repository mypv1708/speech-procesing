"""
Emotional statement detector for chat intent
"""
import re
from typing import Optional


EMOTIONAL_PATTERNS = [
    # Feel patterns
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(tired|exhausted|sleepy|drowsy|fatigued)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(a\s+)?(headache|pain|ache|hurt)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(sad|depressed|down|upset|unhappy)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(anxious|worried|stressed|nervous)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(sick|ill|unwell|nauseous)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(hungry|thirsty)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(cold|hot|warm|cool)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+feel\s+(dizzy|lightheaded)', re.IGNORECASE),
    
    # Have patterns
    re.compile(r'\b(i|i\'m|im)\s+have\s+(a\s+)?(headache|pain|ache|hurt)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+have\s+(a\s+)?(fever|temperature)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+have\s+(a\s+)?(cold|cough|flu)', re.IGNORECASE),
    
    # Am patterns
    re.compile(r'\b(i|i\'m|im)\s+(am\s+)?(tired|exhausted|sleepy)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+(am\s+)?(sad|depressed|down|upset)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+(am\s+)?(anxious|worried|stressed)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+(am\s+)?(sick|ill|unwell)', re.IGNORECASE),
    re.compile(r'\b(i|i\'m|im)\s+(am\s+)?(hungry|thirsty)', re.IGNORECASE),
    
    # Direct statements
    re.compile(r'\b(tired|exhausted|sleepy)\b', re.IGNORECASE),
    re.compile(r'\b(headache|pain|ache)\b', re.IGNORECASE),
    re.compile(r'\b(sad|depressed|upset)\b', re.IGNORECASE),
    re.compile(r'\b(anxious|worried|stressed)\b', re.IGNORECASE),
]

# Use set for O(1) lookup performance
EMOTIONAL_KEYWORDS = {
    'feel', 'feeling', 'tired', 'exhausted', 'sleepy', 'headache', 'pain',
    'ache', 'sad', 'depressed', 'anxious', 'worried', 'stressed', 'sick',
    'ill', 'hungry', 'thirsty', 'dizzy', 'cold', 'hot', 'hurt', 'unwell'
}


def is_emotional_statement(text: str) -> bool:
    """
    Check if text contains emotional or physical state statements.
    
    Args:
        text: Input text to check
        
    Returns:
        True if text contains emotional statement, False otherwise
    """
    text_lower = text.lower().strip()
    
    for pattern in EMOTIONAL_PATTERNS:
        if pattern.search(text_lower):
            return True
    
    words = text_lower.split()
    emotional_word_count = sum(1 for word in words if word in EMOTIONAL_KEYWORDS)
    
    if 'feel' in text_lower or 'feeling' in text_lower:
        if emotional_word_count > 0:
            return True
    
    return False

