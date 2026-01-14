"""
Example of intent classification
"""
import json
import sys
import os
from pathlib import Path

# Add src directory to path to allow imports
# File is at: src/nodes/cognitive/tests/example_intent.py
# Need to go up 3 levels to reach src/
current_file = Path(__file__).resolve()
src_path = current_file.parents[3]  # Go up: tests(0) -> cognitive(1) -> nodes(2) -> src(3)
sys.path.insert(0, str(src_path))

from nodes.cognitive.classify_intent.intent_classifier import classify_intent


if __name__ == "__main__":
    test_cases = [
        # Greeting cases (basic)
        # "Hello, how are you?",
        # "Hi there!",
        
        # # Chat cases (basic)
        # "What is the weather like today?",
        # "How does this work?",
        
        
        # Emotional chat cases
        # "I feel tired",
        # "I feel a headache",
        # "I feel sad",
        # "I feel anxious",
        # "I have a headache",
        # "I am tired",
        # "I feel sick",
        
        # # Follow cases
        # "Follow me",
        # "Let's follow me",
        # "Follow",
        # "Please follow me",
        # "Can you follow me?",
        # "Start following me",
        # "Follow us",
        
        # # Stop follow cases
        # "Stop following",
        # "Stop follow me",
        # "Stop follow",
        # "Don't follow me",
        # "Cancel follow",
        # "Stop tracking",
        # "Quit following",
        
        # # Navigate cases - Main test cases
        # # Basic movements with defaults
        "Go straight",
        "Turn right",
        # "Turn left",
        # "Move forward",
        # "Move backward",
        
        # # Basic movements with specific values
        # "Move forward 5 meters",
        "Go straight for 2 meters",
        "Turn right 90 degrees",
        # "Turn left 45 degrees",
        # "Move backward 3 meters",
        
        # # Complex commands with "for" pattern
        "Go straight for 2 meters, then turn right for 3 meters.",
        "Turn right for 5 meters",
        # "Turn left for 10 meters",
        
        # # Multi-action commands
        "Move forward 10 meters then turn right",
        # "Go straight 5 meters and turn left 90 degrees",
        # "Turn left and move forward 20 meters",
        # "Move forward 15 meters, turn left, go 10 meters",
        # "Turn right 45 degrees and move forward 5 meters",
        
        # # Distance variations
        # "Move 5 meters forward",
        # "Go 10 meters straight",
        # "Walk 20 meters ahead",
        
        # # Combined commands
        "Move forward 10 meters and turn right 90 degrees",
        # "Turn left 45 degrees and go forward 15 meters",
        # "Move backward 3 meters, turn left, go 20 meters",
        # "Go straight ahead 10 meters, turn right 90 degrees, move 5 meters",
        
        # # Edge cases
        # "Move forward 0.5 meters",
        # "Turn left 180 degrees",
        # "Go forward 1 meter",
        # "Move 100 meters forward",
    ]

    print("=" * 80)
    print("INTENT CLASSIFICATION EXAMPLES")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print("=" * 80)

    results = {'greeting': 0, 'chat': 0, 'navigate': 0, 'follow': 0, 'stop_follow': 0}
    navigate_cases = []
    navigate_results = []

    for i, text in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Input: {text}")
        result = classify_intent(text, verbose=False)
        intent = result['intent']
        results[intent] = results.get(intent, 0) + 1
        
        if intent == 'navigate':
            navigate_cases.append(text)
            navigate_results.append(result)
            print(f"✓ Intent: {intent.upper()}")
            print(f"  Actions: {len(result.get('actions', []))} action(s)")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            for idx, action in enumerate(result.get('actions', []), 1):
                action_type = action.get('type', 'unknown')
                if action_type == 'move':
                    direction = action.get('direction', 'unknown')
                    distance = action.get('distance', 0)
                    is_default = action.get('is_default', False)
                    default_str = " (default)" if is_default else ""
                    print(f"    {idx}. {action_type}: {direction} {distance}m{default_str}")
                elif action_type == 'turn':
                    direction = action.get('direction', 'unknown')
                    angle = action.get('angle', 0)
                    is_default = action.get('is_default', False)
                    default_str = " (default)" if is_default else ""
                    print(f"    {idx}. {action_type}: {direction} {angle}°{default_str}")
        elif intent == 'greeting':
            print(f"✓ Intent: {intent.upper()}")
            print(f"  Response: {result.get('response', '')}")
        elif intent == 'chat':
            if result.get('is_emotional'):
                print(f"✓ Intent: {intent.upper()} (emotional)")
                response = result.get('response', '')
                print(f"  Advice: {response}")
            else:
                print(f"  Intent: {intent}")
        else:
            print(f"  Intent: {intent}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Greeting:   {results.get('greeting', 0)} cases")
    print(f"Chat:       {results.get('chat', 0)} cases")
    print(f"Follow:     {results.get('follow', 0)} cases")
    print(f"Stop follow: {results.get('stop_follow', 0)} cases")
    print(f"Navigate:   {results.get('navigate', 0)} cases")
    print("=" * 80)
    
    if navigate_cases:
        print(f"\nNavigate Cases ({len(navigate_cases)}):")
        print("=" * 80)
        for i, (case, result) in enumerate(zip(navigate_cases, navigate_results), 1):
            print(f"\n{i}. {case}")
            print(json.dumps(result, indent=2))
        print("=" * 80)

