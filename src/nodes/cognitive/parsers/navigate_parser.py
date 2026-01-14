import re
from typing import Dict, List, Optional

from ..utils.validator import validate_navigate_json, ValidationError, normalize_angle

# Word to number mapping for parsing text numbers
_WORD_TO_NUMBER = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100
}

_TURN_MOVE_PATTERN = re.compile(
    r'turn\s+(left|right)\s+for\s+(\d+\.?\d*|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:meters?|meter|m|feet|ft)',
    re.IGNORECASE
)
_SPLIT_PATTERN = re.compile(r'\s*(?:then|and|,)\s+')
_TURN_FOR_CHECK_PATTERN = re.compile(r'turn\s+(left|right)\s+for\s+\d+', re.IGNORECASE)

# Pattern for matching numbers (both digits and words)
_NUMBER_PATTERN = r'(\d+\.?\d*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'

_MOVE_PATTERNS = [
    (re.compile(rf'(?:move|go|walk|drive|step|navigate)\s+straight\s+ahead\s+{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)', re.IGNORECASE), ('forward', 0)),
    (re.compile(rf'(?:move|go|walk|drive|step|navigate)\s+(?:straight|forward|ahead)\s+for\s+{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)', re.IGNORECASE), ('forward', 0)),
    (re.compile(rf'(?:move|go|walk|drive|step|navigate)\s+(?:forward|straight|ahead)\s+{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)', re.IGNORECASE), ('forward', 0)),
    (re.compile(rf'(?:move|go|walk|drive|step)\s+backward\s+(?:for\s+)?{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)', re.IGNORECASE), ('backward', 0)),
    (re.compile(rf'(?:move|go|walk|drive)\s+(left|right)\s+{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)', re.IGNORECASE), (None, 0)),
    (re.compile(rf'{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)\s+(?:forward|straight|ahead)', re.IGNORECASE), ('forward', 0)),
    (re.compile(rf'^(?:go|move|walk|drive|step|navigate)\s+{_NUMBER_PATTERN}\s*(?:meters?|meter|m|feet|ft)(?:\s+(?:forward|straight|ahead))?$', re.IGNORECASE), ('forward', 0)),
    (re.compile(rf'^(?:move|go|walk|drive|step|navigate)\s+(?:forward|straight|ahead)(?:\s+for)?(?:\s+{_NUMBER_PATTERN})?\s*(?:meters?|meter|m|feet|ft)?$', re.IGNORECASE), ('forward', None)),
    (re.compile(rf'^(?:move|go|walk|drive)\s+(?:backward|back)(?:\s+for)?(?:\s+{_NUMBER_PATTERN})?\s*(?:meters?|meter|m|feet|ft)?$', re.IGNORECASE), ('backward', None)),
]

_TURN_PATTERNS = [
    (re.compile(rf'turn\s+(left|right)\s+{_NUMBER_PATTERN}\s*(?:degrees?|degree|deg)', re.IGNORECASE), 'dir_angle'),
    (re.compile(rf'turn\s+{_NUMBER_PATTERN}\s*(?:degrees?|degree|deg)\s+(left|right)', re.IGNORECASE), 'angle_dir'),
    (re.compile(rf'turn\s+(left|right)\s+{_NUMBER_PATTERN}(?!\s*(?:meters?|meter|m|feet|ft|for))$', re.IGNORECASE), 'dir_number'),
    (re.compile(r'^turn\s+(left|right)(?:\s+(?:for|by))?(?:\s+\d+\.?\d*)?\s*(?:meters?|meter|m|feet|ft|degrees?|degree|deg)?$', re.IGNORECASE), 'dir_only'),
]


def _word_to_number(word: str) -> Optional[float]:
    """
    Convert word number to numeric value.
    
    Args:
        word: Word number (e.g., "one", "two", "twenty")
        
    Returns:
        Numeric value or None if not recognized
    """
    word_lower = word.lower().strip()
    return _WORD_TO_NUMBER.get(word_lower)


def _parse_number(value: str) -> Optional[float]:
    """
    Parse number from string (supports both digits and words).
    
    Args:
        value: Number string (e.g., "1", "1.5", "one", "two")
        
    Returns:
        Numeric value or None if parsing fails
    """
    if not value:
        return None
    
    value = value.strip().lower()
    
    # Try direct float conversion first
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try word to number conversion
    number = _word_to_number(value)
    if number is not None:
        return float(number)
    
    return None


def parse_navigate_command(text: str) -> Dict:
    """
    Parse navigation command text into structured JSON format.
    
    Args:
        text: Navigation command text (e.g., "Go straight for 2 meters")
        
    Returns:
        Dictionary with intent, confidence, source_text, and actions array
        
    Raises:
        ValidationError: If parsed JSON fails validation
    """
    source_text = text.strip()
    actions = _extract_actions(source_text)
    confidence = _calculate_confidence(source_text, actions)
    
    result = {
        "intent": "navigate",
        "confidence": confidence,
        "source_text": source_text,
        "actions": actions
    }
    
    try:
        validate_navigate_json(result)
    except ValidationError as e:
        raise ValidationError(f"Navigation command validation failed: {e}") from e
    
    return result


def _create_turn_move_actions(direction: str, distance: float) -> List[Dict]:
    """Create turn and move actions for 'turn X for Y meters' pattern"""
    return [
        {
            "type": "turn",
            "direction": direction,
            "angle": 90.0,
            "unit": "degree",
            "is_default": True
        },
        {
            "type": "move",
            "direction": "forward",
            "distance": distance,
            "unit": "meter",
            "is_default": False
        }
    ]


def _extract_actions(text: str) -> List[Dict]:
    """Extract navigation actions from text"""
    text_lower = text.lower()
    actions = []
    
    parts = _SPLIT_PATTERN.split(text_lower)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        turn_move_match = _TURN_MOVE_PATTERN.search(part)
        if turn_move_match:
            direction = turn_move_match.group(1)
            distance_str = turn_move_match.group(2)
            distance = _parse_number(distance_str)
            if distance is not None:
                actions.extend(_create_turn_move_actions(direction, distance))
            continue
        
        turn_action = _parse_turn_action(part)
        if turn_action:
            actions.append(turn_action)
            continue
            
        move_action = _parse_move_action(part)
        if move_action:
            actions.append(move_action)
            continue
    
    if not actions:
        turn_move_match = _TURN_MOVE_PATTERN.search(text_lower)
        if turn_move_match:
            direction = turn_move_match.group(1)
            distance_str = turn_move_match.group(2)
            distance = _parse_number(distance_str)
            if distance is not None:
                actions.extend(_create_turn_move_actions(direction, distance))
        else:
            turn_action = _parse_turn_action(text_lower)
            if turn_action:
                actions.append(turn_action)
            else:
                move_action = _parse_move_action(text_lower)
                if move_action:
                    actions.append(move_action)
    
    return actions


def _parse_move_action(text: str) -> Optional[Dict]:
    """Parse move action from text"""
    if _TURN_FOR_CHECK_PATTERN.search(text):
        return None
    
    for pattern, (default_dir, group_idx) in _MOVE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            
            if default_dir:
                direction = default_dir
                if group_idx is not None and group_idx < len(groups):
                    distance_str = groups[group_idx]
                else:
                    distance_str = groups[0] if groups else None
            else:
                direction = groups[0] if groups[0] in ['left', 'right'] else 'forward'
                distance_str = groups[1] if len(groups) > 1 else None
            
            # Parse number (supports both digits and words)
            if distance_str:
                distance = _parse_number(distance_str)
                if distance is None:
                    distance = 1.0
                    is_default = True
                else:
                    is_default = False
            else:
                distance = 1.0
                is_default = True
            
            return {
                "type": "move",
                "direction": direction,
                "distance": distance,
                "unit": "meter",
                "is_default": is_default
            }
    
    return None


def _parse_turn_action(text: str) -> Optional[Dict]:
    """Parse turn action from text"""
    if _TURN_FOR_CHECK_PATTERN.search(text):
        return None
    
    for pattern, pattern_type in _TURN_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            
            if pattern_type == 'dir_angle':
                direction = groups[0]
                angle_str = groups[1]
            elif pattern_type == 'angle_dir':
                angle_str = groups[0]
                direction = groups[1]
            elif pattern_type == 'dir_number':
                # Pattern: "turn left/right <number>" (without "degrees")
                direction = groups[0]
                angle_str = groups[1]
            else:  # dir_only
                direction = groups[0]
                angle_str = None
            
            if angle_str:
                # Parse number (supports both digits and words)
                angle = _parse_number(angle_str)
                if angle is not None:
                    angle = abs(angle)  # Always use positive angle
                    angle = normalize_angle(angle)
                    is_default = False
                else:
                    # If parsing fails, use default
                    angle = 90.0
                    is_default = True
            else:
                angle = 90.0  # Default angle for both left and right
                is_default = True
            
            return {
                "type": "turn",
                "direction": direction,
                "angle": angle,
                "unit": "degree",
                "is_default": is_default
            }
    
    return None


def _calculate_confidence(text: str, actions: List[Dict]) -> float:
    """Calculate confidence score based on parsing success"""
    if not actions:
        return 0.5
    
    base_confidence = 0.85
    
    has_specific_values = any(
        not action.get('is_default', False) for action in actions
    )
    if has_specific_values:
        base_confidence += 0.1
    
    if len(actions) > 1:
        base_confidence += 0.02
    
    return min(base_confidence, 0.99)