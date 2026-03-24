#!/usr/bin/env python3
"""
Sample reproducible stratified split from the VQAv2 validation parquet
based on specific question type prefixes.

Steps:
  1. Load the full VQAv2 validation parquet.
  2. Map each question to a category based on the longest matching prefix.
  3. Filter out questions that don't match any specified category.
  4. De-duplicate by image_id to ensure one question per image globally.
  5. Sample exactly `n` rows per category (default: 100).
  6. Save the resulting stratified dataset as a parquet file.

Usage:
  python3 vqa_stratified_setup.py                         # uses defaults
  python3 vqa_stratified_setup.py --samples_per_type 100  # explicit count
"""

import argparse
import pandas as pd

# The provided taxonomy of question types mapping to starting phrases
QUESTION_TYPES = {
    "object_recognition": [
        "what",
        "what is",
        "what is this",
        "what is the",
        "what are",
        "what are the",
        "what kind of",
        "what type of",
        "which",
        "what animal is"
    ],
    "attribute_recognition": [
        "what color is",
        "what color is the",
        "what color are the",
        "what is the color of the",
        "what color",
        "what number is"
    ],
    "counting": [
        "how many",
        "how many people are",
        "how many people are in"
    ],
    "existence_presence": [
        "is there",
        "is there a",
        "are there",
        "are there any",
        "is this",
        "is this a",
        "is this an",
        "is that a",
        "is it",
        "are",
        "are these",
        "are the",
        "is the",
        "is",
        "are they",
        "is he",
        "is the person",
        "is the man",
        "is the woman"
    ],
    "spatial_relational": [
        "where is the",
        "where are the",
        "what is on the",
        "what is in the"
    ],
    "activity_interaction": [
        "what is the man",
        "what is the woman",
        "what is the person",
        "what does the",
        "who is",
        "what sport is"
    ],
    "scene_context": [
        "what room is"
    ]
}


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible stratified sample based on question categories"
    )
    parser.add_argument(
        "--input_parquet", type=str,
        default="/home/kalashkala/Datasets/VQAv2/lmms-lab_VQAv2_default_validation.parquet",
        help="Path to the full VQAv2 validation parquet",
    )
    parser.add_argument(
        "--output_parquet", type=str,
        default="/home/kalashkala/Datasets/VQAv2/vqa_stratified_100.parquet",
        help="Output path for the stratified split",
    )
    parser.add_argument("--samples_per_type", type=int, default=100, help="Number of samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"Loading {args.input_parquet} ...")
    df = pd.read_parquet(args.input_parquet)
    print(f"  Total rows: {len(df)}")

    # ── 2. Create Prefix-to-Category Map ──────────────────────────────────
    # Map all prefixes to category, and sort them by string length DESCENDING
    # This ensures that "what color is" matches before "what is" or "what"
    prefix_map = {}
    for category, prefixes in QUESTION_TYPES.items():
        for prefix in prefixes:
            # lowercased just to be safe
            prefix_map[prefix.lower()] = category
            
    sorted_prefixes = sorted(prefix_map.keys(), key=len, reverse=True)

    def get_category(q):
        for prefix in sorted_prefixes:
            # match precisely the prefix with an optional space 
            # so 'what' doesn't match 'whatever'
            if q.startswith(prefix + " ") or q == prefix:
                return prefix_map[prefix]
        return "other"

    df["q_lower"] = df["question"].str.lower().str.strip()
    df["category"] = df["q_lower"].apply(get_category)

    # ── 3. Filter Valid Categories ────────────────────────────────────────
    df_filtered = df[df["category"] != "other"]
    print(f"  Rows matching a requested category: {len(df_filtered)}")

    # ── 4. De-duplicate by Image ID Globally ──────────────────────────────
    # Shuffle first so the image_id deduplication is random regarding which question is kept
    df_unique = df_filtered.sample(frac=1, random_state=args.seed).drop_duplicates(subset="image_id")
    print(f"  Unique images matching categories: {len(df_unique)}")

    print("\nAvailable breakdown by category after deduplication:")
    print(df_unique["category"].value_counts())

    # ── 5. Sample Stratified Rows ─────────────────────────────────────────
    sampled_dfs = []
    
    for category in QUESTION_TYPES.keys():
        cat_df = df_unique[df_unique["category"] == category]
        available = len(cat_df)
        n_sample = min(args.samples_per_type, available)
        
        if n_sample < args.samples_per_type:
            print(f"  [Warning] Only found {n_sample}/{args.samples_per_type} for '{category}'")
            
        if n_sample > 0:
            sampled_dfs.append(cat_df.sample(n=n_sample, random_state=args.seed))

    final_df = pd.concat(sampled_dfs)
    
    # Shuffle final dataset
    final_df = final_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Drop intermediate columns
    final_df = final_df.drop(columns=["q_lower"])
    
    print("\nFinal sampled dataset distribution (by 'category'):")
    print(final_df["category"].value_counts())

    # ── 6. Save Parquet ───────────────────────────────────────────────────
    final_df.to_parquet(args.output_parquet)
    print(f"\n  ✅ Saved stratified split ({len(final_df)} rows) → {args.output_parquet}")


if __name__ == "__main__":
    main()
