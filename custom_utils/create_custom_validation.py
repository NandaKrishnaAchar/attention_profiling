#!/usr/bin/env python3
"""
Create validation dataset from InjecAgent test_cases_dh_base.json
that matches your saved attacks.

This script:
1. Loads your saved attacks (to get attack goals)
2. Loads the original InjecAgent test_cases_dh_base.json
3. Matches attacks by "Attacker Instruction" (adv_goal)
4. Creates a validation JSON with matching entries

Usage:
    python create_validation_from_attacks.py \
        --attacks_path saved_adv_prompts/csv_attacks/test_dataset/attacks_gen0.json \
        --source_data_path data/InjecAgent/raw/test_cases_dh_base.json \
        --output_path data/InjecAgent/dataset/custom_test.json
"""

import json
import argparse
from collections import defaultdict


def load_attacks(attacks_path):
    """Load saved attacks and extract attack goals."""
    with open(attacks_path, 'r') as f:
        attacks = json.load(f)
    
    # Extract unique attack goals (in order)
    attack_goals = []
    seen_goals = set()
    
    for attack in attacks:
        goal = attack.get('adv_goal', '').strip()
        if goal and goal not in seen_goals:
            attack_goals.append(goal)
            seen_goals.add(goal)
    
    return attacks, attack_goals


def load_source_data(source_path):
    """Load the original InjecAgent test cases."""
    with open(source_path, 'r') as f:
        source_data = json.load(f)
    
    # Create a mapping by Attacker Instruction
    # Handle multiple entries with same instruction (store all matches)
    goal_to_entries = defaultdict(list)
    for entry in source_data:
        attacker_instruction = entry.get('Attacker Instruction', '').strip()
        if attacker_instruction:
            goal_to_entries[attacker_instruction].append(entry)
    
    return source_data, goal_to_entries


def create_validation_dataset(attacks, attack_goals, goal_to_entries, match_strategy="first"):
    """
    Create validation dataset matching attacks.
    
    Args:
        attacks: List of attack dictionaries
        attack_goals: List of unique attack goals
        goal_to_entries: Dict mapping goal to list of matching entries
        match_strategy: "first" (use first match), "all" (create entry for each match), "rotate" (cycle through matches)
    """
    validation_data = []
    matched_count = 0
    unmatched_goals = []
    duplicate_warnings = []
    
    # Track which entry index to use for each goal (for rotate strategy)
    goal_index_tracker = defaultdict(int)
    
    # Match each attack goal with source data
    for i, attack in enumerate(attacks):
        goal = attack.get('adv_goal', '').strip()
        
        if goal in goal_to_entries:
            matching_entries = goal_to_entries[goal]
            
            # Handle multiple matches
            if len(matching_entries) > 1:
                duplicate_warnings.append((i, goal, len(matching_entries)))
                
                if match_strategy == "first":
                    # Use first match (default behavior)
                    source_entry = matching_entries[0]
                elif match_strategy == "rotate":
                    # Cycle through matches
                    idx = goal_index_tracker[goal] % len(matching_entries)
                    source_entry = matching_entries[idx]
                    goal_index_tracker[goal] += 1
                elif match_strategy == "all":
                    # Create entry for each match (will create multiple entries per attack)
                    for source_entry in matching_entries:
                        validation_entry = create_validation_entry(source_entry, goal)
                        validation_data.append(validation_entry)
                    matched_count += len(matching_entries)
                    continue
                else:
                    # Default to first
                    source_entry = matching_entries[0]
            else:
                # Single match
                source_entry = matching_entries[0]
            
            # Create validation entry
            validation_entry = create_validation_entry(source_entry, goal)
            validation_data.append(validation_entry)
            matched_count += 1
        else:
            # No match found
            unmatched_goals.append((i, goal))
            print(f"⚠️  Warning: No match found for attack {i}: {goal[:80]}...")
    
    return validation_data, matched_count, unmatched_goals, duplicate_warnings


def create_validation_entry(source_entry, goal):
    """Create a validation entry from source entry."""
    validation_entry = {
        "Attacker Instruction": source_entry.get("Attacker Instruction", goal),
        "User Instruction": source_entry.get("User Instruction", ""),
        "User Tool": source_entry.get("User Tool", ""),
        "Attacker Tools": source_entry.get("Attacker Tools", []),
        "Tool Parameters": source_entry.get("Tool Parameters", "{}"),
        "Tool Response Template": source_entry.get("Tool Response Template", ""),
        "Thought": source_entry.get("Thought", ""),
    }
    
    # Copy any additional fields that might be useful
    if "Expected Achievements" in source_entry:
        validation_entry["Expected Achievements"] = source_entry["Expected Achievements"]
    if "Attack Type" in source_entry:
        validation_entry["Attack Type"] = source_entry["Attack Type"]
    
    return validation_entry


def main():
    parser = argparse.ArgumentParser(
        description='Create validation dataset from InjecAgent source data matching your attacks'
    )
    parser.add_argument(
        '--attacks_path',
        type=str,
        required=True,
        help='Path to saved attacks JSON file (e.g., attacks_gen0.json)'
    )
    parser.add_argument(
        '--source_data_path',
        type=str,
        default='data/InjecAgent/raw/test_cases_dh_base.json',
        help='Path to InjecAgent test_cases_dh_base.json'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for validation JSON file'
    )
    parser.add_argument(
        '--match_strategy',
        type=str,
        default='first',
        choices=['first', 'rotate', 'all'],
        help='How to handle multiple matches: "first" (use first match), "rotate" (cycle through), "all" (create entry for each)'
    )
    
    args = parser.parse_args()
    
    # Load attacks
    print(f"Loading attacks from: {args.attacks_path}")
    attacks, attack_goals = load_attacks(args.attacks_path)
    print(f"Found {len(attacks)} attacks with {len(attack_goals)} unique goals")
    
    # Load source data
    print(f"\nLoading source data from: {args.source_data_path}")
    source_data, goal_to_entries = load_source_data(args.source_data_path)
    print(f"Loaded {len(source_data)} source entries")
    print(f"Found {len(goal_to_entries)} unique attack goals in source data")
    
    # Check for duplicates
    duplicates = {goal: entries for goal, entries in goal_to_entries.items() if len(entries) > 1}
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} attack goals with multiple scenarios:")
        for goal, entries in list(duplicates.items())[:5]:
            print(f"  '{goal[:60]}...' has {len(entries)} scenarios")
            for j, entry in enumerate(entries):
                print(f"    [{j}] User Tool: {entry.get('User Tool', 'N/A')}, "
                      f"User Instruction: {entry.get('User Instruction', 'N/A')[:50]}...")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")
        print(f"\nUsing match strategy: '{args.match_strategy}'")
    
    # Create validation dataset
    print("\nMatching attacks with source data...")
    validation_data, matched_count, unmatched_goals, duplicate_warnings = create_validation_dataset(
        attacks, attack_goals, goal_to_entries, match_strategy=args.match_strategy
    )
    
    # Save validation data
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Total attacks: {len(attacks)}")
    print(f"Matched entries: {matched_count}")
    print(f"Unmatched entries: {len(unmatched_goals)}")
    print(f"Validation dataset size: {len(validation_data)}")
    print(f"\nSaved to: {args.output_path}")
    
    if duplicate_warnings:
        print("\n⚠️  WARNING: Some attacks have multiple matching scenarios!")
        print("Duplicates found:")
        for idx, goal, count in duplicate_warnings[:5]:
            print(f"  Attack {idx}: '{goal[:60]}...' has {count} matching scenarios")
            matching_entries = goal_to_entries[goal]
            for j, entry in enumerate(matching_entries):
                print(f"    [{j}] User Tool: {entry.get('User Tool', 'N/A')}")
                print(f"        User Instruction: {entry.get('User Instruction', 'N/A')[:70]}...")
        if len(duplicate_warnings) > 5:
            print(f"  ... and {len(duplicate_warnings) - 5} more")
        print(f"\nStrategy '{args.match_strategy}' was used to select scenarios.")
        if args.match_strategy == "first":
            print("  (Using first match for each duplicate)")
        elif args.match_strategy == "rotate":
            print("  (Rotating through matches)")
        elif args.match_strategy == "all":
            print("  (Created entry for each match - validation dataset may be larger than attacks)")
    
    if unmatched_goals:
        print("\n⚠️  WARNING: Some attacks couldn't be matched!")
        print("Unmatched goals:")
        for idx, goal in unmatched_goals[:5]:  # Show first 5
            print(f"  [{idx}] {goal[:100]}...")
        if len(unmatched_goals) > 5:
            print(f"  ... and {len(unmatched_goals) - 5} more")
        print("\nYou may need to manually add these entries or check for typos.")
    else:
        print("\n✓ All attacks matched successfully!")
    
    # Verify alignment
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    if len(validation_data) == len(attacks):
        print("✓ Counts match!")
    else:
        print(f"⚠️  Count mismatch: {len(validation_data)} validation entries vs {len(attacks)} attacks")
    
    # Check required fields
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
    for i, entry in enumerate(validation_data):
        for field in required_fields:
            if field not in entry or not entry[field]:
                missing_fields.append((i, field))
    
    if missing_fields:
        print(f"⚠️  Found {len(missing_fields)} missing fields:")
        for idx, field in missing_fields[:5]:
            print(f"  Entry {idx}: missing '{field}'")
    else:
        print("✓ All required fields present")
    
    # Show sample entry
    if validation_data:
        print("\n" + "="*60)
        print("SAMPLE ENTRY:")
        print("="*60)
        sample = validation_data[0]
        print(json.dumps(sample, indent=2))


if __name__ == '__main__':
    main()

