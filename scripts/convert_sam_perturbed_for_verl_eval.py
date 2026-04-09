#!/usr/bin/env python3
"""
Convert SAM Perturbed JSONL → VERL-compatible training Parquet.

This version aligns the prompt/output format with Perception-R1:
<think> ... </think>
<answer> ... </answer>

It extracts rich metadata unique to the SAM targeted perturbation pipeline.
"""

import os
import argparse
import pandas as pd
import json
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


def is_missing(val):
    if val is None: return True
    if isinstance(val, float) and pd.isna(val): return True
    return False

def get_val(row, key, default=None):
    val = row.get(key)
    if is_missing(val):
        return default
    return val

def build_verl_row(row, row_index, split_name="train"):
    """
    Transform a single perturbed JSONL row into the VERL-expected schema.
    """
    question = get_val(row, "question", "")
    image_path = get_val(row, "image_path", "")

    answer = get_val(row, "answers", "")
    acceptable_answers = answer

    gemini_tag = get_val(row, "gemini_tag", {}) or {}
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
            "ground_truth": acceptable_answers[0],
            "multiple_choice_answer": "",
            "style": "vqa_llm_judge",
            "response_format": "perception_r1",
            "answerability": answerability,
            "visual_cues": get_val(row, "visual_cues", []),
            "cue_source": get_val(row, "cue_short_reason", ""),
            "uncertainty_reason": get_val(row, "uncertainty_reason", ""),
            "variant": get_val(row, "variant", ""),
            "perturbation_type": get_val(row, "perturbation_type", ""),
            "failure_type": gemini_tag.get("failure_type", ""),
            # Serialize dynamically sized dicts as JSON strings to avoid PyArrow schema errors
            "masking_details_json": json.dumps(get_val(row, "masking_details", {})),
        },
        "extra_info": {
            "index": str(get_val(row, "id", str(row_index))),
            "source_id": str(get_val(row, "source_id", "")),
            "original_image_path": get_val(row, "original_image_path", ""),
            "category": get_val(row, "category", ""),
            "split": split_name,
            "prompt_style": "perception_r1",
            "gemini_confidence": gemini_tag.get("confidence", ""),
            "gemini_short_reason": gemini_tag.get("short_reason", ""),
            "gemini_attempted_answer": gemini_tag.get("attempted_answer", ""),
        },
    }

def stratified_split(df, stratify_col, test_size=0.20, random_state=42):
    """
    Pandas-based stratified split to avoid requiring scikit-learn.
    """
    train_dfs = []
    val_dfs = []
    for _, group in df.groupby(stratify_col):
        train_group = group.sample(frac=1 - test_size, random_state=random_state)
        val_group = group.drop(train_group.index)
        train_dfs.append(train_group)
        val_dfs.append(val_group)
        
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, val_df


def extract_answerability(row):
    tag = row.get("gemini_tag", {})
    if isinstance(tag, dict):
        return tag.get("answerability", "unknown")
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM Perturbed JSONL to VERL-compatible training format"
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True,
        help="Path to the source SAM perturbed JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output parquet"
    )
    parser.add_argument(
        "--train_output_name", type=str, default="train_sam_perturbed_vqa.parquet",
        help="Filename for the output train parquet"
    )
    parser.add_argument(
        "--val_output_name", type=str, default="val_sam_perturbed_vqa.parquet",
        help="Filename for the output test/val parquet"
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

    # Extract answerability string for stratified splitting
    print("Extracting stratification labels (answerability)...")
    df['stratify_label'] = df.apply(extract_answerability, axis=1)

    print("Splitting into Train (80%) and Val (20%) sets based on answerability...")
    train_df, val_df = stratified_split(df, 'stratify_label', test_size=0.25, random_state=42)

    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    
    # Optional: print distribution to verify
    print("\nTrain Answerability Distribution:")
    print(train_df['stratify_label'].value_counts(normalize=True))
    print("\nVal Answerability Distribution:")
    print(val_df['stratify_label'].value_counts(normalize=True))

    def process_and_save(dataframe, split_name, output_name):
        formatted_data = []
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Converting {split_name}"):
            verl_row = build_verl_row(row, idx, split_name=split_name)
            formatted_data.append(verl_row)

        if len(formatted_data) > 0 and split_name == "train":
            print(f"\n── Sanity Check (first row of {split_name}) ──")
            sample = formatted_data[0]
            print(f"  System prompt:      {sample['prompt'][0]['content']}")
            print(f"  User prompt:        {sample['prompt'][1]['content']}")
            print(f"  Image path:         {sample['images'][0]['image']}")
            print(f"  Ability:            {sample['ability']}")
            print(f"  Acceptable answers: {sample['reward_model']['acceptable_answers']}")
            print(f"  Answerability:      {sample['reward_model']['answerability']}")
            print(f"  Uncertainty Reason: {sample['reward_model'].get('uncertainty_reason', '')}")
            print(f"  Original Image:     {sample['extra_info'].get('original_image_path', '')}")
            print(f"  ID Index:           {sample['extra_info']['index']}")
            print(f"  Split:              {sample['extra_info']['split']}")

        if len(formatted_data) > 0:
            print(f"\nSaving {split_name} VERL-compatible Parquet ...")
            dataset = Dataset.from_list(formatted_data)
            output_path = os.path.join(args.output_dir, output_name)
            dataset.to_parquet(output_path)
            print(f"✅ Saved {len(dataset)} examples to {output_path}")

    process_and_save(train_df, "train", args.train_output_name)
    process_and_save(val_df, "val", args.val_output_name)


if __name__ == "__main__":
    main()
