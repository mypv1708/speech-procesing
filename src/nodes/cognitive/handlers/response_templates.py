"""
Response templates for different intents
"""
import logging
import random
from datetime import datetime
from typing import Dict, List, Any

from nodes.actuator.tts.tts_helper import speak_text

logger = logging.getLogger(__name__)


GREETING_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What would you like to do?",
    "Hey! Nice to meet you. How can I assist you?",
    "Greetings! I'm here to help. What do you need?",
    "Hello! I'm ready to help. What can I do for you?",
    "Hi! Good to see you. How may I assist you today?",
    "Hey there! What would you like me to do?",
    "Hello! I'm here and ready to help. What's on your mind?",
    "Hi! How can I be of service today?",
    "Greetings! What would you like to do?",
    "Hello! How are you doing today?",
    "Hi! Nice to see you. What can I help with?",
    "Hey! I'm here to help. What do you need?",
    "Hello there! What would you like to do?",
    "Hi! I'm ready when you are. What can I do for you?",
]

MORNING_RESPONSES = [
    "Good morning! How can I help you today?",
    "Good morning! What would you like to do?",
    "Morning! I'm here to help. What can I do for you?",
    "Good morning! How may I assist you?",
]

AFTERNOON_RESPONSES = [
    "Good afternoon! How can I help you today?",
    "Good afternoon! What would you like to do?",
    "Afternoon! I'm here to help. What can I do for you?",
    "Good afternoon! How may I assist you?",
]

EVENING_RESPONSES = [
    "Good evening! How can I help you today?",
    "Good evening! What would you like to do?",
    "Evening! I'm here to help. What can I do for you?",
    "Good evening! How may I assist you?",
]

NIGHT_RESPONSES = [
    "Good night! How can I help you?",
    "Good night! What would you like to do?",
    "Night! I'm here to help. What can I do for you?",
]

CASUAL_GREETING_RESPONSES = [
    "Hey! What's up? How can I help you?",
    "Hey there! What can I do for you?",
    "Hi! What's going on? How can I assist?",
    "Hey! I'm here to help. What do you need?",
]


def _get_time_of_day() -> str:
    """
    Get time of day based on current system time.
    
    Returns:
        'morning' (5:00 - 11:59)
        'afternoon' (12:00 - 16:59)
        'evening' (17:00 - 21:59)
        'night' (22:00 - 4:59)
    """
    now = datetime.now()
    current_hour = now.hour
    current_time_str = now.strftime("%H:%M:%S")
    
    if 5 <= current_hour < 12:
        time_of_day = 'morning'
    elif 12 <= current_hour < 17:
        time_of_day = 'afternoon'
    elif 17 <= current_hour < 22:
        time_of_day = 'evening'
    else:
        time_of_day = 'night'
    
    logger.debug(f"Current time: {current_time_str} (hour: {current_hour}) -> {time_of_day}")
    return time_of_day


def get_greeting_response(text_input: str) -> str:
    """
    Get a natural greeting response based on input and current time.
    """
    text_lower = text_input.lower().strip()
    
    # Check for casual greetings first (highest priority)
    casual_keywords = ("what's up", "whats up", "sup", "howdy")
    if any(word in text_lower for word in casual_keywords):
        return random.choice(CASUAL_GREETING_RESPONSES)
    
    # Check if input contains time-specific keywords (priority over system time)
    if 'morning' in text_lower:
        return random.choice(MORNING_RESPONSES)
    
    if 'afternoon' in text_lower:
        return random.choice(AFTERNOON_RESPONSES)
    
    if 'evening' in text_lower:
        return random.choice(EVENING_RESPONSES)
    
    if 'night' in text_lower:
        return random.choice(NIGHT_RESPONSES)
    
    # If no time keyword in input, use current system time
    time_of_day = _get_time_of_day()
    
    logger.info(f"Selecting greeting based on time of day: {time_of_day}")
    
    if time_of_day == 'morning':
        return random.choice(MORNING_RESPONSES)
    elif time_of_day == 'afternoon':
        return random.choice(AFTERNOON_RESPONSES)
    elif time_of_day == 'evening':
        return random.choice(EVENING_RESPONSES)
    else:
        return random.choice(NIGHT_RESPONSES)


def get_greeting_intent_data(
    text_input: str,
    use_tts: bool = True,
    use_cuda: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Get greeting intent data with template response and optional TTS.
    """
    response = get_greeting_response(text_input)
    
    if use_tts:
        speak_text(response, use_cuda=use_cuda, verbose=verbose)
    
    return {
        'intent': 'greeting',
        'source_text': text_input,
        'response': response
    }


CHAT_RESPONSES = [
    "I understand. How else can I help you?",
    "Got it. What would you like to do next?",
    "I see. Is there anything else I can assist with?",
    "Understood. How can I help you further?",
    "Okay. What would you like me to do?",
    "I hear you. What can I do for you?",
    "Noted. How may I assist you?",
    "Alright. What's next?",
]


def get_chat_response(text_input: str) -> str:
    """
    Get a natural chat response for general conversation.
    """
    return random.choice(CHAT_RESPONSES)


def get_chat_intent_data(
    text_input: str,
    use_tts: bool = True,
    use_cuda: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Get chat intent data with template response and optional TTS.

    """
    response = get_chat_response(text_input)
    
    if use_tts:
        speak_text(response, use_cuda=use_cuda, verbose=verbose)
    
    return {
        'intent': 'chat',
        'source_text': text_input,
        'response': response
    }