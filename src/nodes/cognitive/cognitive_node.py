"""
Cognitive Node - Processes transcribed text through intent classification.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .classify_intent.intent_classifier import classify_intent

logger = logging.getLogger(__name__)

# Import client để gửi lệnh đến robot
try:
    # Thêm project root vào path để import client
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from client import send_command
    CLIENT_AVAILABLE = True
    logger.info("Robot client loaded successfully - commands will be sent to server")
except ImportError as e:
    logger.warning(f"Failed to import client: {e}. Robot commands will not be sent.")
    CLIENT_AVAILABLE = False
    send_command = None


def normalize_stt_text(text: str) -> str:
    """
    Normalize STT output text by removing punctuation and normalizing whitespace.
    
    Examples:
        "Turn left, 48." -> "turn left 48"
        "Go forward, 2.5 meters." -> "go forward 2.5 meters"
        "Hello, how are you?" -> "hello how are you"
    
    Args:
        text: Raw text from STT (may contain punctuation).
        
    Returns:
        Normalized text (lowercase, no punctuation, normalized whitespace).
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation but keep numbers and spaces
    # Keep: letters, numbers, spaces
    # Remove: punctuation marks (.,!?;:()[]{}\"'`-)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class CognitiveNode:
    """
    Cognitive node that processes text input through intent classification.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        verbose: bool = True,
        use_tts: bool = True,
        preload_models: bool = True,
    ):
        """
        Initialize CognitiveNode with all models preloaded.
        
        Args:
            use_gpu: Whether to use GPU for models.
            verbose: Whether to print progress.
            use_tts: Whether to enable TTS for responses.
            preload_models: Whether to preload all models upfront (default: True).
        """
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.use_tts = use_tts
        
        # Preload all cognitive models upfront
        if preload_models:
            try:
                from .utils.model_loader import preload_all_cognitive_models
                preload_all_cognitive_models(use_gpu=use_gpu, verbose=verbose)
            except Exception:
                pass
    
    def process_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process transcribed text through intent classification.
        
        Args:
            text: Transcribed text from STT.
            
        Returns:
            Intent classification result dictionary or None if processing fails.
        """
        if not text or not text.strip():
            return None
        
        try:
            # Normalize text: remove punctuation, lowercase, normalize whitespace
            original_text = text
            normalized_text = normalize_stt_text(text)
            
            if normalized_text != original_text:
                logger.debug("Text normalized: '%s' -> '%s'", original_text, normalized_text)
            
            logger.info("Processing text: '%s'", normalized_text)
            
            result = classify_intent(
                text_input=normalized_text,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
                use_tts=self.use_tts
            )
            
            intent = result.get('intent', 'unknown')
            logger.info("Intent classified: %s", intent)
            
            # Debug: Kiểm tra client availability
            if intent == 'navigate':
                logger.info(f"Navigate intent detected. CLIENT_AVAILABLE={CLIENT_AVAILABLE}, send_command={send_command is not None}")
            
            # Nếu là intent navigate và có formatted_command hợp lệ, gửi lệnh đến robot
            if intent == 'navigate' and CLIENT_AVAILABLE and send_command:
                formatted_command = result.get('formatted_command', '').strip()
                actions = result.get('actions', [])
                
                # Debug: Log thông tin navigate
                logger.info(f"Navigate intent - formatted_command: {repr(formatted_command)}, actions count: {len(actions) if actions else 0}")
                
                # Chỉ gửi nếu có formatted_command và có actions (không phải navigate empty)
                if formatted_command and actions and len(actions) > 0:
                    # Kiểm tra xem có action thực sự không (không phải chỉ có STOP)
                    has_real_action = False
                    for action in actions:
                        action_type = action.get('type', '')
                        if action_type in ['move', 'turn']:
                            has_real_action = True
                            break
                    
                    logger.info(f"Navigate intent - has_real_action: {has_real_action}")
                    
                    if has_real_action:
                        # Kiểm tra lệnh có hợp lệ không (phải bắt đầu bằng $SEQ)
                        if formatted_command.startswith('$SEQ'):
                            logger.info(f"✓ Sending command to robot: {repr(formatted_command)}")
                            try:
                                success = send_command(formatted_command)
                                if success:
                                    logger.info("✓✓ Command sent to robot successfully!")
                                else:
                                    logger.warning("✗ Failed to send command to robot")
                            except Exception as e:
                                logger.error(f"✗ Error sending command to robot: {e}", exc_info=True)
                        else:
                            logger.warning(f"✗ Invalid command format (must start with $SEQ): {repr(formatted_command)}")
                    else:
                        logger.info("Navigate intent has no real actions (only STOP), skipping robot command")
                else:
                    if not formatted_command:
                        logger.info(f"Navigate intent missing formatted_command, skipping robot command")
                    elif not actions or len(actions) == 0:
                        logger.info(f"Navigate intent has empty actions, skipping robot command")
            elif intent == 'navigate':
                # Navigate nhưng không có client
                if not CLIENT_AVAILABLE:
                    logger.warning("Navigate intent detected but CLIENT_AVAILABLE=False - command will not be sent")
                elif not send_command:
                    logger.warning("Navigate intent detected but send_command is None - command will not be sent")
            
            return result
            
        except Exception as e:
            logger.error("Failed to process text: %s", e, exc_info=True)
            return None
