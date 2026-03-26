import json
import argparse
from pathlib import Path

def main(args):
    input_file = args.input_jsonl
    output_file = args.output_jsonl
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            
            # Add the default ANSWERABLE gemini_tag referencing the perturbed structure
            record["gemini_tag"] = {
                "answerability": "ANSWERABLE",
                "failure_type": "none",
                "short_reason": "The question is answerable as the original image is clear and clearly shows visual information to answer the question without any perturbation."
            }
            
            # Write the updated record to the output JSONL file
            outfile.write(json.dumps(record) + "\n")
            processed_count += 1
            
    print(f"Successfully processed {processed_count} records.")
    print(f"Tagged output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add default gemini tags (ANSWERABLE) to clean VQA dataset.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the clean VQA JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the tagged JSONL file.")
    
    args = parser.parse_args()
    main(args)
