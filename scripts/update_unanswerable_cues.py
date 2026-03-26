import os
import json
import time
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add current directory to sys.path to ensure build_visual_cues can be imported
sys.path.append(str(Path(__file__).parent))

try:
    from build_visual_cues import (
        build_gemini_client,
        get_visual_cues,
        assign_category
    )
except ImportError:
    print("Error: Could not import functions from build_visual_cues.py. Ensure it is in the same directory.")
    sys.exit(1)

def main(args):
    client = build_gemini_client()
    input_file = args.input_jsonl
    output_file = args.output_jsonl
    
    print(f"Reading manifest from {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Total records in manifest: {len(records)}")
    
    updated_count = 0
    errors = 0
    
    # Process only UNANSWERABLE entries
    for record in tqdm(records, desc="Updating cues for UNANSWERABLE entries"):
        gemini_tag = record.get("gemini_tag", {})
        answerability = gemini_tag.get("answerability", "").upper()
        
        if answerability == "UNANSWERABLE":
            # In perturbed_manifest, the image is at 'perturbed_image_path'
            image_path = record.get("perturbed_image_path")
            if not image_path:
                print(f"Warning: No perturbed_image_path for record {record.get('id')}")
                continue
                
            question = record.get("question", "")
            # For unanswerable cues, we explicitly tell the model it might be unanswerable or provide the reason.
            # Providing "unanswerable" as the gold answer helps the cue generator focus on why it can't be answered.
            gold_answer = "UNANSWERABLE" 
            category = record.get("category") or assign_category(question)

            try:
                # Generate new cues based on the PERTURBED image
                new_cues = get_visual_cues(
                    client=client,
                    image_path=image_path,
                    question=question,
                    gold_answer=gold_answer,
                    category=category
                )
                
                record["visual_cues"] = new_cues["visual_cues"]
                record["cue_short_reason"] = new_cues["short_reason"]
                
                # Update answer to "I don't know" for RL training
                record["answer"] = "I don't know"
                
                updated_count += 1
                
                # Immediate save/flush isn't possible as we are modifying the list, 
                # but we'll print progress.
                if (updated_count) % 10 == 0:
                    print(f"  Successfully updated {updated_count} records so far.")
                
                # Sleep to avoid rate limits
                if args.sleep_interval > 0:
                    time.sleep(args.sleep_interval)
                    
            except Exception as e:
                print(f"\nError updating record {record.get('id')} at {image_path}: {e}")
                errors += 1
                if errors > 5:
                    print("Too many consecutive errors, stopping.")
                    break

    print(f"Updated cues for {updated_count} unanswerable records.")
    
    print(f"Saving updated manifest to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update visual cues for UNANSWERABLE perturbed entries.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to perturbed_manifest.jsonl")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save updated manifest")
    parser.add_argument("--sleep_interval", type=float, default=1.0, help="Seconds between API calls")
    
    args = parser.parse_args()
    main(args)
