import os
import io
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from PIL import Image
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/debarpanb1/kalashkala/Perception-R1/.env')


# =========================================================
# CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"
# Default paths (can be overridden via argparse)
DEFAULT_OUTPUT_JSONL = "clean_vqa_with_visual_cues.jsonl"
DEFAULT_IMAGE_DIR = "/home/debarpanb1/kalashkala/visual-question-answering/processed_for_verl/images"


# =========================================================
# CATEGORY MAP
# =========================================================

CATEGORY_PREFIX_MAP = {
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
        "what animal is",
    ],
    "attribute_recognition": [
        "what color is",
        "what color is the",
        "what color are the",
        "what is the color of the",
        "what color",
        "what number is",
    ],
    "counting": [
        "how many",
        "how many people are",
        "how many people are in",
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
        "is the woman",
    ],
    "spatial_relational": [
        "where is the",
        "where are the",
        "what is on the",
        "what is in the",
    ],
    "activity_interaction": [
        "what is the man",
        "what is the woman",
        "what is the person",
        "what does the",
        "who is",
        "what sport is",
    ],
    "scene_context": [
        "what room is",
    ],
}

SPATIAL_KEYWORDS = [
    "left", "right", "behind", "in front of", "front of", "next to",
    "under", "below", "above", "on top of", "near", "between",
    "inside", "outside", "beside"
]
ATTRIBUTE_KEYWORDS = [
    "color", "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "gray", "grey", "striped", "number"
]
COUNTING_KEYWORDS = ["how many", "number of", "count"]
ACTIVITY_KEYWORDS = [
    "doing", "playing", "holding", "eating", "riding", "running",
    "walking", "standing", "sitting", "wearing", "drinking", "looking"
]


def normalize_question(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def match_prefix(question: str) -> Optional[str]:
    q = normalize_question(question)
    pairs = []
    for category, prefixes in CATEGORY_PREFIX_MAP.items():
        for p in prefixes:
            pairs.append((p, category))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for prefix, category in pairs:
        if q.startswith(prefix):
            return category
    return None


def refine_category(question: str, coarse_category: Optional[str]) -> str:
    q = normalize_question(question)

    if any(k in q for k in SPATIAL_KEYWORDS):
        return "spatial_relational"
    if any(k in q for k in COUNTING_KEYWORDS):
        return "counting"
    if any(k in q for k in ATTRIBUTE_KEYWORDS):
        return "attribute_recognition"
    if any(k in q for k in ACTIVITY_KEYWORDS):
        return "activity_interaction"

    return coarse_category or "object_recognition"


def assign_category(question: str) -> str:
    return refine_category(question, match_prefix(question))


# =========================================================
# GEMINI HELPERS
# =========================================================

def build_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


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


def extract_json_from_text(text: Optional[str]) -> Dict:
    if text is None:
        return {"visual_cues": [], "short_reason": "Gemini response was empty (possibly blocked by safety filters)"}
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

    raise ValueError(f"Could not parse JSON from Gemini response. Response: {text}")


def normalize_cue_output(obj: Dict) -> Dict:
    cues = obj.get("visual_cues", [])
    if not isinstance(cues, list):
        cues = []
    cues = [str(x).strip() for x in cues if str(x).strip()]
    cues = cues[:5]

    short_reason = str(obj.get("short_reason", "")).strip()

    return {
        "visual_cues": cues,
        "short_reason": short_reason,
    }


def get_visual_cues(
    client: genai.Client,
    image_path: str,
    question: str,
    gold_answer: str,
    category: str,
    max_retries: int = 3,
) -> Dict:
    img = load_image(image_path)
    image_bytes = pil_to_jpeg_bytes(img)
    prompt = build_visual_cue_prompt(question, gold_answer, category)

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(response.text)
            return normalize_cue_output(parsed)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to get visual cues for {image_path}: {last_err}")


# =========================================================
# MAIN
# =========================================================

def main(args) -> None:
    client = build_gemini_client()

    print(f"Loading parquet from: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    all_samples = df.to_dict('records')
    total = len(all_samples)

    # ── Row range slicing for resuming ───────────────────────────────────────
    start = args.start_row if args.start_row is not None else 0
    end   = args.end_row   if args.end_row   is not None else total
    start = max(0, min(start, total))
    end   = max(start, min(end, total))
    samples = all_samples[start:end]
    print(f"Processing rows {start} – {end-1} ({len(samples)} rows out of {total} total)", flush=True)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    
    # Open in append mode so we can resume or save incrementally
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for local_idx, sample in enumerate(samples):
            idx = start + local_idx  # preserve original global index
            question = sample.get("question", "")
            category = sample.get("category", "")
            if not category:
                category = assign_category(question)

            answers_array = sample.get("answers", [])
            answer = ""
            if len(answers_array) > 0:
                if isinstance(answers_array, np.ndarray):
                    answers_array = answers_array.tolist()
                
                valid_answers = [a.get('answer', '') for a in answers_array if a.get('answer_confidence') in ['yes', 'maybe']]
                if valid_answers:
                    answer = valid_answers[0]
                else:
                    answer = answers_array[0].get('answer', '')

            image_info = sample.get("image", {})
            image_name = image_info.get("path", "")
            image_path = os.path.join(args.image_dir, image_name)

            cues = get_visual_cues(
                client=client,
                image_path=image_path,
                question=question,
                gold_answer=answer,
                category=category,
            )
            
            record = {
                "id": str(idx),
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "category": category,
                "visual_cues": cues["visual_cues"],
                "cue_short_reason": cues["short_reason"],
            }
            
            # Write record immediately to file as a JSON line
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (local_idx + 1) % 5 == 0 or (local_idx + 1) == len(samples):
                print(f"Processed {local_idx + 1}/{len(samples)} (global row {idx})", flush=True)

            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)

    print(f"Saved to {args.output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual cues from VQA samples using Gemini.")
    parser.add_argument("--input_parquet", type=str, required=True, help="Path to the input parquet file.")
    parser.add_argument("--output_jsonl", type=str, default=DEFAULT_OUTPUT_JSONL, help="Path to save the output JSONL.")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Directory containing the images.")
    parser.add_argument("--sleep_interval", type=float, default=2.0, help="Seconds to sleep between generations (rate limiting).")
    parser.add_argument("--start_row", type=int, default=None, help="0-based index of the first row to process (inclusive). Omit to start from the beginning.")
    parser.add_argument("--end_row", type=int, default=None, help="0-based index of the last row to process (exclusive). Omit to process until the end.")
    
    args = parser.parse_args()
    main(args)

# Example Usage with nohup (Run in background):
# Full run:
# nohup python scripts/build_visual_cues.py --input_parquet /home/debarpanb1/kalashkala/visual-question-answering/vqa_stratified_100.parquet --output_jsonl /home/debarpanb1/kalashkala/visual-question-answering/clean_vqa_with_visual_cues.jsonl > build_visual_cues.log 2>&1 &
#
# Resume from row 303 to end:
# nohup python scripts/build_visual_cues.py --input_parquet /home/debarpanb1/kalashkala/visual-question-answering/vqa_stratified_100.parquet --output_jsonl /home/debarpanb1/kalashkala/visual-question-answering/clean_vqa_with_visual_cues.jsonl --start_row 303 > build_visual_cues_resume.log 2>&1 &
#
# Process a specific window (e.g. rows 100 to 199 inclusive):
# nohup python scripts/build_visual_cues.py --input_parquet /home/debarpanb1/kalashkala/visual-question-answering/vqa_stratified_100.parquet --output_jsonl /home/debarpanb1/kalashkala/visual-question-answering/clean_vqa_with_visual_cues.jsonl --start_row 100 --end_row 200 > build_visual_cues_partial.log 2>&1 &