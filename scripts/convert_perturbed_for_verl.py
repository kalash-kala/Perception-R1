#!/usr/bin/env python3
"""
Convert Perturbed JSONL → VERL-compatible training Parquet.

This version aligns the prompt/output format with Perception-R1:
<think> ... </think>
<answer> ... </answer>
"""

import os
import argparse
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
    "The reasoning process MUST BE enclosed within <think></think> tags.\n"
    "The answer MUST BE enclosed within <answer></answer> tags.\n"
    "Do not output anything outside these tags.\n"
    "Base your reasoning only on the visible image evidence.\n"
    "If the image does not support a confident answer, output I don't know inside <answer></answer>.\n"
)

# This mirrors the style used in Perception-R1 / related multimodal RLVR prompting:
# the image/question is followed by a direct formatting instruction.
USER_FORMAT_SUFFIX = (
    "\nOutput the thinking process in <think> </think> and the final answer in <answer> </answer> tags."
)


def build_user_prompt(question: str) -> str:
    question = (question or "").strip()
    return f"<image>\n{question}{USER_FORMAT_SUFFIX}"


def build_verl_row(row, row_index):
    """
    Transform a single perturbed JSONL row into the VERL-expected schema.
    """
    question = row.get("question", "")
    image_path = row.get("perturbed_image_path", "")

    answer = row.get("answer", "")
    acceptable_answers = [answer] if answer else []

    gemini_tag = row.get("gemini_tag", {}) or {}
    answerability = gemini_tag.get("answerability", "unknown")

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ],
        "images": [{"image": f"file://{image_path}"}],
        "ability": "visual_question_answering",
        "reward_model": {
            "acceptable_answers": acceptable_answers,
            "multiple_choice_answer": "",
            "style": "vqa_llm_judge",
            "response_format": "perception_r1",
            "answerability": answerability,
            "visual_cues": row.get("visual_cues", []),
            "cue_source": row.get("cue_short_reason", ""),
            "variant": row.get("variant", ""),
            "perturbation_type": row.get("perturbation_type", ""),
        },
        "extra_info": {
            "index": row_index,
            "source_id": str(row.get("source_id", "")),
            "category": row.get("category", ""),
            "split": "train",
            "prompt_style": "perception_r1",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Perturbed JSONL to VERL-compatible training format"
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True,
        help="Path to the source perturbed JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output parquet"
    )
    parser.add_argument(
        "--output_name", type=str, default="train_perturbed_vqa.parquet",
        help="Filename for the output parquet"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Limit rows to process (0 = all)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input_jsonl} ...")
    df = pd.read_json(args.input_jsonl, lines=True)
    total = len(df)
    print(f"  Total rows in source: {total}")

    if args.max_samples > 0:
        df = df.head(args.max_samples)
        print(f"  Limiting to first {args.max_samples} rows")

    formatted_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        verl_row = build_verl_row(row, idx)
        formatted_data.append(verl_row)

    if len(formatted_data) > 0:
        print("\n── Sanity Check (first row) ──")
        sample = formatted_data[0]
        print(f"  Prompt roles:       {[m['role'] for m in sample['prompt']]}")
        print(f"  System prompt:      {sample['prompt'][0]['content']}")
        print(f"  User prompt:        {sample['prompt'][1]['content']}")
        print(f"  Image path:         {sample['images'][0]['image']}")
        print(f"  Ability:            {sample['ability']}")
        print(f"  Acceptable answers: {sample['reward_model']['acceptable_answers']}")
        print(f"  Answerability:      {sample['reward_model']['answerability']}")
        print(f"  Visual cues:        {sample['reward_model']['visual_cues']}")
        print(f"  Cue source:         {sample['reward_model']['cue_source']}")
        print(f"  Variant:            {sample['reward_model']['variant']}")
        print(f"  Perturbation:       {sample['reward_model']['perturbation_type']}")
        print(f"  Response format:    {sample['reward_model']['response_format']}")
        print(f"  Prompt style:       {sample['extra_info']['prompt_style']}")
        print(f"  Category:           {sample['extra_info']['category']}")
        print(f"  Source ID:          {sample['extra_info']['source_id']}")

    print("\nSaving VERL-compatible Parquet ...")
    dataset = Dataset.from_list(formatted_data)
    output_path = os.path.join(args.output_dir, args.output_name)
    dataset.to_parquet(output_path)
    print(f"✅ Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()