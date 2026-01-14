from typing import Dict


class NavigateFormatter:
    """
    Formatter to convert navigate JSON to standard command format.
    
    Example:
        Input JSON:
        {
            "intent": "navigate",
            "actions": [
                {"type": "move", "direction": "forward", "distance": 10.0},
                {"type": "turn", "direction": "right", "angle": 90.0}
            ]
        }
        
        Output: "$SEQ,FWD,10;TR,90;STOP\n"
    """
    
    # Direction mappings
    MOVE_FORWARD = "FWD"
    MOVE_BACKWARD = "BWD"
    TURN_LEFT = "TL"
    TURN_RIGHT = "TR"
    
    @staticmethod
    def format_command(navigate_json: Dict) -> str:
        """
        Convert navigate JSON to standard command format.
        
        """
        if not isinstance(navigate_json, dict):
            raise ValueError("navigate_json must be a dictionary")
        
        if navigate_json.get("intent") != "navigate":
            raise ValueError(f"Expected intent 'navigate', got '{navigate_json.get('intent')}'")
        
        actions = navigate_json.get("actions", [])
        if not actions:
            raise ValueError("Actions list cannot be empty")
        
        command_parts = ["$SEQ"]
        
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "move":
                direction = action.get("direction", "").lower()
                distance = action.get("distance", 0.0)
                
                if direction == "forward":
                    cmd = f"{NavigateFormatter.MOVE_FORWARD},{int(distance)}"
                elif direction == "backward":
                    cmd = f"{NavigateFormatter.MOVE_BACKWARD},{int(distance)}"
                else:
                    # Default to forward if direction not recognized
                    cmd = f"{NavigateFormatter.MOVE_FORWARD},{int(distance)}"
                
                command_parts.append(cmd)
                
            elif action_type == "turn":
                direction = action.get("direction", "").lower()
                angle = action.get("angle", 0.0)
                
                if direction == "left":
                    cmd = f"{NavigateFormatter.TURN_LEFT},{int(angle)}"
                elif direction == "right":
                    cmd = f"{NavigateFormatter.TURN_RIGHT},{int(angle)}"
                else:
                    # Default to right if direction not recognized
                    cmd = f"{NavigateFormatter.TURN_RIGHT},{int(angle)}"
                
                command_parts.append(cmd)
        
        # Add STOP at the end
        command_parts.append("STOP")
        
        # Join with semicolon and add newline
        return ";".join(command_parts) + "\n"