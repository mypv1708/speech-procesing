"""
Test script for NavigateFormatter - demonstrates JSON to command format conversion.
"""
import sys
from pathlib import Path

# Add src directory to path
current_file = Path(__file__).resolve()
src_path = current_file.parents[3]  # Go up: parsers(0) -> cognitive(1) -> nodes(2) -> src(3)
sys.path.insert(0, str(src_path))

from nodes.cognitive.parsers.navigate_formatter import NavigateFormatter


def main():
    """Test NavigateFormatter with example JSON."""
    
    # Example JSON from terminal selection
    example_json = {
        "intent": "navigate",
        "confidence": 0.97,
        "source_text": "Move forward 10 meters and turn right 90 degrees",
        "actions": [
            {
                "type": "move",
                "direction": "forward",
                "distance": 10.0,
                "unit": "meter",
                "is_default": False
            },
            {
                "type": "turn",
                "direction": "right",
                "angle": 90.0,
                "unit": "degree",
                "is_default": False
            }
        ]
    }
    
    print("Testing NavigateFormatter...")
    print("\n" + "=" * 80)
    
    # Format and print
    formatted = NavigateFormatter.format_and_print(example_json, print_json=True)
    
    # Additional test cases
    test_cases = [
        {
            "intent": "navigate",
            "actions": [
                {"type": "move", "direction": "forward", "distance": 5.0},
            ]
        },
        {
            "intent": "navigate",
            "actions": [
                {"type": "turn", "direction": "left", "angle": 45.0},
                {"type": "move", "direction": "forward", "distance": 20.0},
            ]
        },
        {
            "intent": "navigate",
            "actions": [
                {"type": "move", "direction": "backward", "distance": 3.0},
                {"type": "turn", "direction": "left", "angle": 90.0},
                {"type": "move", "direction": "forward", "distance": 15.0},
            ]
        },
    ]
    
    print("\n" + "=" * 80)
    print("ADDITIONAL TEST CASES:")
    print("=" * 80)
    
    for i, test_json in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        formatted_cmd = NavigateFormatter.format_and_print(test_json, print_json=True)


if __name__ == "__main__":
    main()

