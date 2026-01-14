from .prompts import INTENT_CLASSIFICATION_PROMPT, EMOTIONAL_ADVICE_PROMPT


def format_intent_classification_prompt(user_text: str) -> str:
    """Format prompt for intent classification"""
    return "\n".join([
        f"<start_of_turn>developer\n{INTENT_CLASSIFICATION_PROMPT}<end_of_turn>",
        f"<start_of_turn>user\n{user_text}<end_of_turn>",
        "<start_of_turn>model\n"
    ])


def format_emotional_advice_prompt(user_text: str) -> str:
    """Format prompt for emotional advice generation"""
    return f"{EMOTIONAL_ADVICE_PROMPT}\n\nUser: {user_text}\nAssistant: Here's what you should do:"
