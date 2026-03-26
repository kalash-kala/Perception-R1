import json
import random
import argparse
from pathlib import Path

def main(args):
    clean_file = args.clean_jsonl
    perturbed_file = args.perturbed_jsonl
    output_file = args.output_jsonl
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    combined_records = []
    
    # Process Clean dataset
    print(f"Loading clean dataset from {clean_file}")
    with open(clean_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            # Map specific fields for clean dataset
            # It already has 'image_path' which is what we want for the final merged set
            original_id = record.get("id", "")
            record["source_id"] = str(original_id)
            record["id"] = f"{original_id}_clean"
            
            record["variant"] = "clean"
            record["perturbation_type"] = "none"
            
            # For consistency, keep an original_image_path
            if "image_path" in record:
                record["original_image_path"] = record["image_path"]
                
            combined_records.append(record)
            
    # Process Perturbed dataset
    print(f"Loading perturbed dataset from {perturbed_file}")
    with open(perturbed_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            # Change 'perturbed_image_path' to just 'image_path'
            if "perturbed_image_path" in record:
                record["image_path"] = record.pop("perturbed_image_path")
                
            combined_records.append(record)
            
    print(f"Total merged records: {len(combined_records)}")
    
    # Optional: shuffle the combined dataset
    if args.shuffle:
        print("Shuffling combined dataset...")
        random.seed(args.seed)
        random.shuffle(combined_records)
        
    print(f"Saving merged dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in combined_records:
            f.write(json.dumps(record) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Clean and Perturbed VQA JSONL datasets.")
    parser.add_argument("--clean_jsonl", type=str, required=True, help="Path to the tagged clean JSONL.")
    parser.add_argument("--perturbed_jsonl", type=str, required=True, help="Path to the perturbed JSONL.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the merged JSONL.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the merged dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    
    args = parser.parse_args()
    main(args)
