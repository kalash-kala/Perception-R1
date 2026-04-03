#!/usr/bin/env python3
"""
Step 1 – Build Visual Cues (no images saved to disk)

Reads a VQAv2 Parquet file with embedded image bytes, loads each image
in memory, calls Gemini for visual cues, and writes a lightweight JSONL.
The Parquet row_index is stored so Step 2 can look up image bytes later.

Usage:
    python step1_build_visual_cues.py \
        --input_parquet /path/to/new_vqa.parquet \
        --output_jsonl  /path/to/output/clean_cues.jsonl \
        --sleep_interval 1.0
"""

import os
import io
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/debarpanb1/kalashkala/Perception-R1/.env')


# =========================================================
# CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"


# =========================================================
# CATEGORY MAP
# =========================================================

CATEGORY_PREFIX_MAP = {
    "object_recognition": [
        "what", "what is", "what is this", "what is the",
        "what are", "what are the", "what kind of", "what type of",
        "which", "what animal is",
    ],
    "attribute_recognition": [
        "what color is", "what color is the", "what color are the",
        "what is the color of the", "what color", "what number is",
    ],
    "counting": [
        "how many", "how many people are", "how many people are in",
    ],
    "existence_presence": [
        "is there", "is there a", "are there", "are there any",
        "is this", "is this a", "is this an", "is that a", "is it",
        "are", "are these", "are the", "is the", "is",
        "are they", "is he", "is the person", "is the man", "is the woman",
    ],
    "spatial_relational": [
        "where is the", "where are the", "what is on the", "what is in the",
    ],
    "activity_interaction": [
        "what is the man", "what is the woman", "what is the person",
        "what does the", "who is", "what sport is",
    ],
    "scene_context": ["what room is"],
}

SPATIAL_KEYWORDS = [
    "left", "right", "behind", "in front of", "front of", "next to",
    "under", "below", "above", "on top of", "near", "between",
    "inside", "outside", "beside",
]
ATTRIBUTE_KEYWORDS = [
    "color", "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "gray", "grey", "striped", "number",
]
COUNTING_KEYWORDS = ["how many", "number of", "count"]
ACTIVITY_KEYWORDS = [
    "doing", "playing", "holding", "eating", "riding", "running",
    "walking", "standing", "sitting", "wearing", "drinking", "looking",
]


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def match_prefix(question: str) -> Optional[str]:
    q = normalize_question(question)
    pairs = [(p, cat) for cat, pxs in CATEGORY_PREFIX_MAP.items() for p in pxs]
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    for prefix, category in pairs:
        if q.startswith(prefix):
            return category
    return None


def refine_category(question: str, coarse: Optional[str]) -> str:
    q = normalize_question(question)
    if any(k in q for k in SPATIAL_KEYWORDS):
        return "spatial_relational"
    if any(k in q for k in COUNTING_KEYWORDS):
        return "counting"
    if any(k in q for k in ATTRIBUTE_KEYWORDS):
        return "attribute_recognition"
    if any(k in q for k in ACTIVITY_KEYWORDS):
        return "activity_interaction"
    return coarse or "object_recognition"


def assign_category(question: str) -> str:
    return refine_category(question, match_prefix(question))


# =========================================================
# HELPERS
# =========================================================

def image_bytes_to_jpeg(raw_bytes: bytes) -> bytes:
    """Load raw image bytes (any format) and re-encode as JPEG."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def extract_answer(row) -> str:
    """Best-effort gold answer from the VQAv2 `answers` array."""
    answers_array = row.get("answers", [])
    if isinstance(answers_array, np.ndarray):
        answers_array = answers_array.tolist()
    if not answers_array:
        return row.get("multiple_choice_answer", "")
    confident = [a.get("answer", "") for a in answers_array
                 if a.get("answer_confidence") == "yes"]
    if confident:
        return confident[0]
    return answers_array[0].get("answer", "")


def get_image_name(row) -> str:
    """Get the image filename from the Parquet row."""
    image_info = row.get("image", {})
    return image_info.get("path", f"img_{row.get('image_id', 'unknown')}.jpg")


# =========================================================
# GEMINI HELPERS
# =========================================================

def build_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def build_visual_cue_prompt(question: str, gold_answer: str, category: str) -> str:
    return f"""
You are creating atomic visual evidence annotations for a multimodal reasoning dataset.

Task:
Given the image and the question, extract only the visual evidence relevant to answering the question.

Instructions:
- Return 2 to 5 short atomic visual cues.
- Each cue must be directly visible in the image.
- Do not use background knowledge.
- Do not speculate.
- Do not provide chain-of-thought.
- Do not mention information irrelevant to the question.
- Prefer literal, compact observations.

Return JSON only in exactly this format:
{{
  "visual_cues": ["cue 1", "cue 2", "cue 3"],
  "short_reason": "one short sentence"
}}

Metadata:
- Question category: {category}

Question: {question}
Gold answer: {gold_answer}
""".strip()


def extract_json_from_text(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Could not parse JSON from Gemini response.")


def normalize_cue_output(obj: Dict) -> Dict:
    cues = obj.get("visual_cues", [])
    if not isinstance(cues, list):
        cues = []
    cues = [str(x).strip() for x in cues if str(x).strip()][:5]
    return {
        "visual_cues": cues,
        "short_reason": str(obj.get("short_reason", "")).strip(),
    }


def get_visual_cues(
    client: genai.Client,
    jpeg_bytes: bytes,
    question: str,
    gold_answer: str,
    category: str,
    max_retries: int = 3,
) -> Dict:
    """Send in-memory JPEG bytes to Gemini and get visual cues."""
    prompt = build_visual_cue_prompt(question, gold_answer, category)

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(response.text)
            return normalize_cue_output(parsed)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Gemini visual cue extraction failed: {last_err}")


# =========================================================
# MAIN
# =========================================================

def main(args) -> None:
    client = build_gemini_client()

    print(f"Loading parquet from: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    all_records = df.to_dict("records")
    total = len(all_records)

    start = args.start_row if args.start_row is not None else 0
    end   = args.end_row   if args.end_row   is not None else total
    start = max(0, min(start, total))
    end   = max(start, min(end, total))
    print(f"Processing rows {start} – {end - 1} ({end - start} rows out of {total} total)",
          flush=True)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for idx in range(start, end):
            rec = all_records[idx]
            local_idx = idx - start

            question   = rec.get("question", "")
            category   = rec.get("category") or assign_category(question)
            answer     = extract_answer(rec)
            image_name = get_image_name(rec)

            # Load image from Parquet binary → JPEG bytes (in memory only)
            raw_bytes = rec.get("image", {}).get("bytes", b"")
            if not raw_bytes:
                print(f"  [SKIP] row {idx}: no image bytes")
                continue

            try:
                jpeg_bytes = image_bytes_to_jpeg(raw_bytes)
            except Exception as e:
                print(f"  [SKIP] row {idx}: failed to decode image: {e}")
                continue

            try:
                cues = get_visual_cues(
                    client=client,
                    jpeg_bytes=jpeg_bytes,
                    question=question,
                    gold_answer=answer,
                    category=category,
                )
            except Exception as e:
                print(f"  [ERROR] row {idx}: {e}")
                continue

            record = {
                "row_index": idx,
                "image_name": image_name,
                "question": question,
                "answer": answer,
                "category": category,
                "visual_cues": cues["visual_cues"],
                "cue_short_reason": cues["short_reason"],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (local_idx + 1) % 5 == 0 or (local_idx + 1) == (end - start):
                print(f"Processed {local_idx + 1}/{end - start} (global row {idx})",
                      flush=True)

            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)

    print(f"Done. Output: {args.output_jsonl}")
    print("NOTE: No images were saved to disk. Step 2 will re-read the Parquet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Build visual cues from Parquet (no images saved to disk)."
    )
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Path to the input VQAv2 Parquet file.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to save the output JSONL with visual cues.")
    parser.add_argument("--sleep_interval", type=float, default=1.0,
                        help="Seconds to sleep between Gemini API calls.")
    parser.add_argument("--start_row", type=int, default=None,
                        help="0-based first row (inclusive). Omit = 0.")
    parser.add_argument("--end_row", type=int, default=None,
                        help="0-based last row (exclusive). Omit = end.")
    args = parser.parse_args()
    main(args)
