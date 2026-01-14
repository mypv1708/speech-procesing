"""
JSON Schema and Semantic validation for navigation commands
"""
from typing import Dict, List, Any, Optional

from ..config import MAX_DISTANCE, MAX_ANGLE


class ValidationError(Exception):
    """Raised when JSON validation fails"""
    pass


def validate_navigate_json(data: Dict[str, Any]) -> None:
    required_fields = ['intent', 'confidence', 'source_text', 'actions']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    if not isinstance(data['intent'], str):
        raise ValidationError(
            f"Invalid type for 'intent': expected str, got {type(data['intent']).__name__}"
        )
    if not isinstance(data['confidence'], (int, float)):
        raise ValidationError(
            f"Invalid type for 'confidence': expected number, got {type(data['confidence']).__name__}"
        )
    if not isinstance(data['source_text'], str):
        raise ValidationError(
            f"Invalid type for 'source_text': expected str, got {type(data['source_text']).__name__}"
        )
    if not isinstance(data['actions'], list):
        raise ValidationError(
            f"Invalid type for 'actions': expected list, got {type(data['actions']).__name__}"
        )
    
    if data['intent'] != 'navigate':
        raise ValidationError(
            f"Invalid 'intent' value: expected 'navigate', got '{data['intent']}'"
        )
    if not (0.0 <= data['confidence'] <= 1.0):
        raise ValidationError(
            f"Invalid 'confidence' value: expected 0.0-1.0, got {data['confidence']}"
        )
    if not data['source_text'].strip():
        raise ValidationError("Invalid 'source_text': cannot be empty")
    if len(data['actions']) == 0:
        raise ValidationError("Invalid 'actions': list cannot be empty")
    
    for i, action in enumerate(data['actions']):
        validate_action(action, index=i)


def validate_action(action: Dict[str, Any], index: Optional[int] = None) -> None:
    prefix = f"actions[{index}]" if index is not None else "action"
    
    if 'type' not in action:
        raise ValidationError(f"{prefix}: Missing required field 'type'")
    
    action_type = action['type']
    if action_type not in ['move', 'turn']:
        raise ValidationError(
            f"{prefix}: Invalid 'type' value: expected 'move' or 'turn', got '{action_type}'"
        )
    
    if not isinstance(action_type, str):
        raise ValidationError(
            f"{prefix}: Invalid type for 'type': expected str, got {type(action_type).__name__}"
        )
    
    if action_type == 'move':
        validate_move_action(action, prefix)
    elif action_type == 'turn':
        validate_turn_action(action, prefix)


def validate_move_action(action: Dict[str, Any], prefix: str) -> None:
    """Validate a move action with JSON schema and semantic checks"""
    required_fields = ['direction', 'distance', 'unit', 'is_default']
    missing_fields = [field for field in required_fields if field not in action]
    if missing_fields:
        raise ValidationError(
            f"{prefix}: Missing required fields: {', '.join(missing_fields)}"
        )
    
    if not isinstance(action['direction'], str):
        raise ValidationError(
            f"{prefix}: Invalid type for 'direction': expected str, "
            f"got {type(action['direction']).__name__}"
        )
    if not isinstance(action['distance'], (int, float)):
        raise ValidationError(
            f"{prefix}: Invalid type for 'distance': expected number, "
            f"got {type(action['distance']).__name__}"
        )
    if not isinstance(action['unit'], str):
        raise ValidationError(
            f"{prefix}: Invalid type for 'unit': expected str, "
            f"got {type(action['unit']).__name__}"
        )
    if not isinstance(action['is_default'], bool):
        raise ValidationError(
            f"{prefix}: Invalid type for 'is_default': expected bool, "
            f"got {type(action['is_default']).__name__}"
        )
    
    valid_directions = ['forward', 'backward', 'left', 'right']
    if action['direction'] not in valid_directions:
        raise ValidationError(
            f"{prefix}: Invalid 'direction' value: expected one of {valid_directions}, "
            f"got '{action['direction']}'"
        )
    if action['unit'] != 'meter':
        raise ValidationError(
            f"{prefix}: Invalid 'unit' value: expected 'meter', got '{action['unit']}'"
        )
    
    if 'angle' in action:
        raise ValidationError(
            f"{prefix}: Semantic error: 'move' action cannot have 'angle' field"
        )
    
    if action['distance'] == 0:
        raise ValidationError(
            f"{prefix}: Semantic error: 'distance' cannot be 0"
        )
    
    if action['distance'] > MAX_DISTANCE:
        raise ValidationError(
            f"{prefix}: Semantic error: 'distance' ({action['distance']}) exceeds "
            f"MAX_DISTANCE ({MAX_DISTANCE})"
        )
    
    if action['distance'] < 0:
        raise ValidationError(
            f"{prefix}: Semantic error: 'distance' must be positive, got {action['distance']}"
        )


def validate_turn_action(action: Dict[str, Any], prefix: str) -> None:
    """Validate a turn action with JSON schema and semantic checks"""
    required_fields = ['direction', 'angle', 'unit', 'is_default']
    missing_fields = [field for field in required_fields if field not in action]
    if missing_fields:
        raise ValidationError(
            f"{prefix}: Missing required fields: {', '.join(missing_fields)}"
        )
    
    if not isinstance(action['direction'], str):
        raise ValidationError(
            f"{prefix}: Invalid type for 'direction': expected str, "
            f"got {type(action['direction']).__name__}"
        )
    if not isinstance(action['angle'], (int, float)):
        raise ValidationError(
            f"{prefix}: Invalid type for 'angle': expected number, "
            f"got {type(action['angle']).__name__}"
        )
    if not isinstance(action['unit'], str):
        raise ValidationError(
            f"{prefix}: Invalid type for 'unit': expected str, "
            f"got {type(action['unit']).__name__}"
        )
    if not isinstance(action['is_default'], bool):
        raise ValidationError(
            f"{prefix}: Invalid type for 'is_default': expected bool, "
            f"got {type(action['is_default']).__name__}"
        )
    
    valid_directions = ['left', 'right']
    if action['direction'] not in valid_directions:
        raise ValidationError(
            f"{prefix}: Invalid 'direction' value: expected 'left' or 'right', "
            f"got '{action['direction']}'"
        )
    if action['unit'] != 'degree':
        raise ValidationError(
            f"{prefix}: Invalid 'unit' value: expected 'degree', got '{action['unit']}'"
        )
    
    if 'distance' in action:
        raise ValidationError(
            f"{prefix}: Semantic error: 'turn' action cannot have 'distance' field"
        )
    
    if action['angle'] == 0:
        raise ValidationError(
            f"{prefix}: Semantic error: 'angle' cannot be 0"
        )
    
    angle = action['angle']
    if abs(angle) > MAX_ANGLE:
        normalized_angle = angle % (360 if angle > 0 else -360)
        if normalized_angle == 0:
            normalized_angle = 360 if angle > 0 else -360
        raise ValidationError(
            f"{prefix}: Semantic warning: 'angle' ({angle}) exceeds ±360, "
            f"should be normalized to {normalized_angle}"
        )


def normalize_angle(angle: float) -> float:
    if abs(angle) <= MAX_ANGLE:
        return angle
    
    normalized = angle % 360
    
    if normalized == 0:
        normalized = 360 if angle > 0 else -360
    elif angle < 0 and normalized > 0:
        normalized = normalized - 360
    
    return normalized

