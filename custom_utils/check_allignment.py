#!/usr/bin/env python3
"""
Check if saved attacks align with validation dataset.

Usage:
    python check_data_alignment.py \
        --attacks_path saved_adv_prompts/csv_attacks/test_dataset/attacks_gen0.json \
        --validation_path data/InjecAgent/dataset/test.json
"""

import json
import argparse
from collections import Counter


def check_alignment(attacks_path, validation_path):
    """Check if attacks and validation data align."""
    
    # Load attacks
    with open(attacks_path, 'r') as f:
        attacks = json.load(f)
    
    # Load validation data
    with open(validation_path, 'r') as f:
        validation_data = json.load(f)
    
    print(f"Attacks: {len(attacks)} entries")
    print(f"Validation: {len(validation_data)} entries")
    print()
    
    # Check count match
    if len(attacks) != len(validation_data):
        print(f"⚠️  WARNING: Count mismatch!")
        print(f"   Attacks: {len(attacks)}, Validation: {len(validation_data)}")
        print(f"   The evaluation script matches by index, so counts must match.")
        print()
    
    # Check if adv_goals match Attacker Instructions
    matches = 0
    mismatches = []
    
    min_len = min(len(attacks), len(validation_data))
    for i in range(min_len):
        attack_goal = attacks[i].get('adv_goal', '').strip()
        validation_goal = validation_data[i].get('Attacker Instruction', '').strip()
        
        if attack_goal == validation_goal:
            matches += 1
        else:
            mismatches.append({
                'index': i,
                'attack_goal': attack_goal[:100],
                'validation_goal': validation_goal[:100]
            })
    
    print(f"Goal matching: {matches}/{min_len} match exactly")
    
    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches:")
        print("   First 5 mismatches:")
        for mm in mismatches[:5]:
            print(f"   Index {mm['index']}:")
            print(f"     Attack goal: {mm['attack_goal']}")
            print(f"     Validation goal: {mm['validation_goal']}")
            print()
    
    # Check required fields in validation data
    print("Checking validation data structure...")
    required_fields = [
        "Attacker Instruction",
        "User Instruction",
        "User Tool",
        "Attacker Tools",
        "Tool Parameters",
        "Tool Response Template",
        "Thought"
    ]
    
    missing_fields = []
    for i, entry in enumerate(validation_data[:min_len]):
        for field in required_fields:
            if field not in entry:
                missing_fields.append((i, field))
    
    if missing_fields:
        print(f"⚠️  Found {len(missing_fields)} missing fields:")
        for idx, field in missing_fields[:10]:
            print(f"   Entry {idx}: missing '{field}'")
    else:
        print("✓ All required fields present")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Attacks count: {len(attacks)}")
    print(f"  Validation count: {len(validation_data)}")
    print(f"  Counts match: {'✓' if len(attacks) == len(validation_data) else '✗'}")
    print(f"  Goals match: {matches}/{min_len} ({matches*100/min_len:.1f}%)")
    print(f"  Structure valid: {'✓' if not missing_fields else '✗'}")
    
    if len(attacks) == len(validation_data) and matches == min_len and not missing_fields:
        print("\n✓ Data is properly aligned! Ready for evaluation.")
        return True
    else:
        print("\n⚠️  Data alignment issues found. Review the warnings above.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Check alignment between saved attacks and validation dataset'
    )
    parser.add_argument(
        '--attacks_path',
        type=str,
        required=True,
        help='Path to saved attacks JSON file'
    )
    parser.add_argument(
        '--validation_path',
        type=str,
        required=True,
        help='Path to validation JSON file'
    )
    
    args = parser.parse_args()
    
    check_alignment(args.attacks_path, args.validation_path)


if __name__ == '__main__':
    main()

