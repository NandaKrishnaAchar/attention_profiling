#!/usr/bin/env python3
"""
Convert CSV file with generated attacks to saved adversarial prompts format
for ALL generations (0-3) at once.

Usage:
    python convert_all_generations.py \
        --csv_path "example_attacks - Sheet1.csv" \
        --output_dir saved_adv_prompts/csv_attacks/test_dataset
"""

import csv
import json
import os
import argparse
from pathlib import Path


def parse_csv_with_multiline_fields(csv_path):
    """Parse CSV file handling multiline fields properly."""
    attacks = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_idx, row in enumerate(reader):
            original_prompt = row.get('Original Prompt', '').strip()
            
            # Get all generation columns
            generations = []
            for gen_idx in range(4):  # Generation 0-3
                gen_key = f'Generation {gen_idx}'
                if gen_key in row:
                    gen_text = row[gen_key].strip()
                    if gen_text:
                        generations.append(gen_text)
            
            if original_prompt:
                attacks.append({
                    'original_prompt': original_prompt,
                    'generations': generations
                })
    
    return attacks


def convert_to_saved_format(attacks, generation_idx=0):
    """Convert attacks to the format expected by injecagent_eval.py"""
    saved_format = []
    
    for attack in attacks:
        original_prompt = attack['original_prompt']
        generations = attack['generations']
        
        # Use the specified generation index, or fallback to first available
        if generation_idx < len(generations):
            adv_prompt = generations[generation_idx]
        elif len(generations) > 0:
            adv_prompt = generations[0]
        else:
            adv_prompt = original_prompt
        
        saved_format.append({
            "adv_goal": original_prompt,
            "attacker_output": "",
            "attacker_adv_prompt": adv_prompt
        })
    
    return saved_format


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV attacks to saved adversarial prompts format for all generations'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to the CSV file with attacks'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory (e.g., saved_adv_prompts/csv_attacks/test_dataset)'
    )
    parser.add_argument(
        '--base_name',
        type=str,
        default='attacks',
        help='Base name for output files (will create attacks_gen0.json, attacks_gen1.json, etc.)'
    )
    
    args = parser.parse_args()
    
    # Parse CSV
    print(f"Reading CSV from: {args.csv_path}")
    attacks = parse_csv_with_multiline_fields(args.csv_path)
    print(f"Found {len(attacks)} attacks in CSV")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert and save for each generation
    for gen_idx in range(4):
        print(f"\nProcessing Generation {gen_idx}...")
        saved_format = convert_to_saved_format(attacks, generation_idx=gen_idx)
        
        output_path = os.path.join(args.output_dir, f"{args.base_name}_gen{gen_idx}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(saved_format, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved {len(saved_format)} attacks to: {output_path}")
        
        # Count how many actually have this generation
        has_generation = sum(1 for a in attacks if gen_idx < len(a['generations']))
        print(f"  ✓ {has_generation} attacks have Generation {gen_idx}")
    
    print(f"\n✓ Successfully converted all 4 generations!")
    print(f"\nOutput files:")
    for gen_idx in range(4):
        output_path = os.path.join(args.output_dir, f"{args.base_name}_gen{gen_idx}.json")
        print(f"  - {output_path}")


if __name__ == '__main__':
    main()

