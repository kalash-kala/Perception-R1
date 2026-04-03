#!/usr/bin/env python3
"""
Step 2 – Perturb, tag, and save ONLY confirmed unanswerable pairs.

Reads the JSONL from Step 1 (visual cues + row_index) and the original
Parquet. For each record: re-loads image bytes from Parquet by row_index,
applies a STRONG perturbation in memory, calls Gemini to tag answerability.

If UNANSWERABLE:
  - Saves original image to disk
  - Saves perturbed image to disk
  - Regenerates visual cues on the perturbed image
  - Writes TWO records to the output JSONL:
      1. Original (ANSWERABLE) with clean visual cues
      2. Perturbed (UNANSWERABLE) with new cues, answer = "I don't know"

If ANSWERABLE:
  - Discards everything. Nothing hits disk.

Usage:
    python step2_perturb_and_tag.py \
        --input_jsonl    /path/to/step1_clean_cues.jsonl \
        --input_parquet  /path/to/source.parquet \
        --output_jsonl   /path/to/unanswerable_final.jsonl \
        --image_dir      /path/to/output_images \
        --sleep_interval 2.0
"""

import os
import io
import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/debarpanb1/kalashkala/Perception-R1/.env')


# =========================================================
# CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"


# =========================================================
# CATEGORY HELPERS (for fallback)
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


def assign_category(question: str) -> str:
    q = normalize_question(question)
    if any(k in q for k in SPATIAL_KEYWORDS):
        return "spatial_relational"
    if any(k in q for k in COUNTING_KEYWORDS):
        return "counting"
    if any(k in q for k in ATTRIBUTE_KEYWORDS):
        return "attribute_recognition"
    if any(k in q for k in ACTIVITY_KEYWORDS):
        return "activity_interaction"
    pairs = [(p, cat) for cat, pxs in CATEGORY_PREFIX_MAP.items() for p in pxs]
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    for prefix, category in pairs:
        if q.startswith(prefix):
            return category
    return "object_recognition"


# =========================================================
# IMAGE HELPERS
# =========================================================

def bytes_to_pil(raw_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def save_image(img: Image.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


# =========================================================
# PERTURBATION FUNCTIONS
# =========================================================

def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def downsample_restore(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR).resize(
        (w, h), Image.Resampling.BILINEAR)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def center_crop_restore(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize(
        (w, h), Image.Resampling.BILINEAR)


def random_occlusion(img: Image.Image, area_ratio: float,
                     fill_mode: str = "gray") -> Image.Image:
    img = img.copy()
    w, h = img.size
    pw = ph = max(1, min(int(np.sqrt(w * h * area_ratio)), min(w, h)))
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    fills = {"gray": (128, 128, 128), "black": (0, 0, 0), "white": (255, 255, 255)}
    fill = fills.get(fill_mode, tuple(np.random.randint(0, 256, 3).tolist()))
    ImageDraw.Draw(img).rectangle([x1, y1, x1 + pw, y1 + ph], fill=fill)
    return img


def darken_region(img: Image.Image, area_ratio: float,
                  darkness: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape
    pw = ph = max(1, min(int(np.sqrt(w * h * area_ratio)), min(w, h)))
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    arr[y1:y1 + ph, x1:x1 + pw] *= darkness
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def clutter_patch(img: Image.Image, area_ratio: float) -> Image.Image:
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    pw = ph = max(1, min(int(np.sqrt(w * h * area_ratio)), min(w, h)))
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    arr[y1:y1 + ph, x1:x1 + pw] = np.random.randint(
        0, 256, (ph, pw, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# =========================================================
# CATEGORY → STRONG PERTURBATIONS ONLY
# =========================================================

STRONG_OPS = {
    "spatial_relational": [
        ("blur_strong",     lambda x: gaussian_blur(x, 4.0)),
        ("crop_strong",     lambda x: center_crop_restore(x, 0.60)),
        ("occlusion_large", lambda x: random_occlusion(x, 0.18, "black")),
        ("clutter_patch",   lambda x: clutter_patch(x, 0.15)),
    ],
    "attribute_recognition": [
        ("brightness_strong", lambda x: adjust_brightness(x, 0.45)),
        ("contrast_strong",   lambda x: adjust_contrast(x, 0.45)),
        ("blur_strong",       lambda x: gaussian_blur(x, 3.5)),
        ("darken_region",     lambda x: darken_region(x, 0.18, 0.08)),
    ],
    "counting": [
        ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
        ("clutter_patch",     lambda x: clutter_patch(x, 0.18)),
        ("occlusion_large",   lambda x: random_occlusion(x, 0.18, "black")),
        ("crop_strong",       lambda x: center_crop_restore(x, 0.65)),
    ],
    "existence_presence": [
        ("occlusion_large",   lambda x: random_occlusion(x, 0.18, "black")),
        ("darken_region",     lambda x: darken_region(x, 0.20, 0.05)),
        ("crop_strong",       lambda x: center_crop_restore(x, 0.60)),
        ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
    ],
    "activity_interaction": [
        ("blur_strong",     lambda x: gaussian_blur(x, 4.0)),
        ("crop_strong",     lambda x: center_crop_restore(x, 0.60)),
        ("occlusion_large", lambda x: random_occlusion(x, 0.16, "black")),
        ("darken_region",   lambda x: darken_region(x, 0.18, 0.05)),
    ],
    "scene_context": [
        ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
        ("blur_strong",       lambda x: gaussian_blur(x, 4.0)),
        ("crop_strong",       lambda x: center_crop_restore(x, 0.60)),
    ],
}

DEFAULT_STRONG_OPS = [
    ("blur_strong",       lambda x: gaussian_blur(x, 4.0)),
    ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
    ("occlusion_large",   lambda x: random_occlusion(x, 0.16, "black")),
    ("crop_strong",       lambda x: center_crop_restore(x, 0.60)),
]


def apply_strong_perturbation(img: Image.Image,
                              category: str) -> Tuple[Image.Image, str]:
    ops = STRONG_OPS.get(category, DEFAULT_STRONG_OPS)
    name, fn = random.choice(ops)
    return fn(img), name


# =========================================================
# GEMINI HELPERS
# =========================================================

def build_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


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


# ── Answerability tagging ────────────────────────────────────────────────

def build_tag_prompt(
    question: str,
    gold_answer: str,
    category: str,
    perturbation_type: str,
    visual_cues: List[str],
) -> str:
    cues_text = "\n".join(f"- {c}" for c in visual_cues) if visual_cues \
        else "- None provided"
    return f"""
You are verifying whether a perturbed image still contains enough visual evidence to answer a question.

Instructions:
- Use the image and the question.
- Use the provided canonical visual cues from the original clean image as reference.
- Decide whether the perturbed image is still ANSWERABLE or UNANSWERABLE.
- If the key evidence is missing, too blurred, too dark, too cropped, too occluded, or too cluttered, mark UNANSWERABLE.
- Keep the output compact.
- Do not provide chain-of-thought.

Return JSON only in exactly this format:
{{
  "answerability": "ANSWERABLE or UNANSWERABLE",
  "failure_type": "none | blur | crop | occlusion | low_resolution | darkness | clutter | other",
  "short_reason": "one short sentence"
}}

Metadata:
- Category: {category}
- Perturbation type: {perturbation_type}

Question: {question}
Gold answer: {gold_answer}

Canonical visual cues from the original image:
{cues_text}
""".strip()


def normalize_tag(obj: Dict) -> Dict:
    ans = str(obj.get("answerability", "")).strip().upper()
    if ans not in {"ANSWERABLE", "UNANSWERABLE"}:
        ans = "ANSWERABLE"
    ft = str(obj.get("failure_type", "other")).strip().lower()
    allowed = {"none", "blur", "crop", "occlusion", "low_resolution",
               "darkness", "clutter", "other"}
    if ft not in allowed:
        ft = "other"
    return {
        "answerability": ans,
        "failure_type": ft,
        "short_reason": str(obj.get("short_reason", "")).strip(),
    }


def tag_perturbed(
    client: genai.Client,
    perturbed_jpeg: bytes,
    question: str,
    gold_answer: str,
    category: str,
    perturbation_type: str,
    visual_cues: List[str],
    max_retries: int = 3,
) -> Dict:
    prompt = build_tag_prompt(question, gold_answer, category,
                              perturbation_type, visual_cues)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=perturbed_jpeg,
                                          mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            return normalize_tag(extract_json_from_text(resp.text))
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini tagging failed: {last_err}")


# ── Visual cue regeneration ──────────────────────────────────────────────

def build_visual_cue_prompt(question: str, gold_answer: str,
                            category: str) -> str:
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
    prompt = build_visual_cue_prompt(question, gold_answer, category)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes,
                                          mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            return normalize_cue_output(extract_json_from_text(resp.text))
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini cue regeneration failed: {last_err}")


# =========================================================
# MAIN
# =========================================================

def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    client = build_gemini_client()

    # ── Load Step 1 JSONL ─────────────────────────────────────────────────
    print(f"Loading Step 1 JSONL: {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        cue_records = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(cue_records)} records from Step 1")

    # ── Load Parquet (for image bytes lookup) ─────────────────────────────
    print(f"Loading Parquet: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    parquet_rows = df.to_dict("records")
    print(f"  {len(parquet_rows)} rows in Parquet")

    # ── Apply row range slicing ───────────────────────────────────────────
    total = len(cue_records)
    start = args.start_row if args.start_row is not None else 0
    end   = args.end_row   if args.end_row   is not None else total
    start = max(0, min(start, total))
    end   = max(start, min(end, total))
    records_to_process = cue_records[start:end]
    print(f"Processing JSONL rows {start} – {end - 1} ({len(records_to_process)} rows)",
          flush=True)

    # ── Prepare output ────────────────────────────────────────────────────
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    unanswerable_count = 0
    answerable_skip_count = 0
    error_count = 0

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for local_idx, rec in enumerate(records_to_process):
            row_index  = rec["row_index"]
            image_name = rec["image_name"]
            question   = rec["question"]
            answer     = rec["answer"]
            category   = rec.get("category") or assign_category(question)
            visual_cues     = rec.get("visual_cues", [])
            cue_short_reason = rec.get("cue_short_reason", "")

            # ── Look up image bytes from Parquet ──────────────────────────
            if row_index < 0 or row_index >= len(parquet_rows):
                print(f"  [SKIP] row_index {row_index} out of range")
                continue

            raw_bytes = parquet_rows[row_index].get("image", {}).get("bytes", b"")
            if not raw_bytes:
                print(f"  [SKIP] row_index {row_index}: no image bytes")
                continue

            try:
                original_img = bytes_to_pil(raw_bytes)
            except Exception as e:
                print(f"  [SKIP] row_index {row_index}: decode failed: {e}")
                continue

            # ── Perturb in memory ─────────────────────────────────────────
            try:
                perturbed_img, perturbation_type = apply_strong_perturbation(
                    original_img, category)
                perturbed_jpeg = pil_to_jpeg_bytes(perturbed_img)
            except Exception as e:
                print(f"  [ERROR] row {row_index} perturbation: {e}")
                error_count += 1
                continue

            # ── Tag answerability via Gemini ───────────────────────────────
            try:
                tag = tag_perturbed(
                    client=client,
                    perturbed_jpeg=perturbed_jpeg,
                    question=question,
                    gold_answer=answer,
                    category=category,
                    perturbation_type=perturbation_type,
                    visual_cues=visual_cues,
                )
            except Exception as e:
                print(f"  [ERROR] row {row_index} Gemini tag: {e}")
                error_count += 1
                continue

            # ── Check answerability ───────────────────────────────────────
            if tag["answerability"] != "UNANSWERABLE":
                answerable_skip_count += 1
                if (local_idx + 1) % 10 == 0:
                    print(f"  [{local_idx + 1}/{len(records_to_process)}] "
                          f"ANSWERABLE (skipped). "
                          f"Unanswerable so far: {unanswerable_count}",
                          flush=True)
                if args.sleep_interval > 0:
                    time.sleep(args.sleep_interval)
                continue

            # ══════════════════════════════════════════════════════════════
            # CONFIRMED UNANSWERABLE — save images + write records
            # ══════════════════════════════════════════════════════════════

            # ── Save original image to disk ───────────────────────────────
            original_path = image_dir / image_name
            save_image(original_img, str(original_path))

            # ── Save perturbed image to disk ──────────────────────────────
            perturbed_name = f"PERTURBED_{image_name}"
            perturbed_path = image_dir / perturbed_name
            save_image(perturbed_img, str(perturbed_path))

            # ── Regenerate visual cues on the perturbed image ─────────────
            try:
                new_cues = get_visual_cues(
                    client=client,
                    jpeg_bytes=perturbed_jpeg,
                    question=question,
                    gold_answer="UNANSWERABLE",
                    category=category,
                )
            except Exception as e:
                print(f"  [WARN] row {row_index}: cue regen failed ({e}), "
                      "using empty cues")
                new_cues = {"visual_cues": [], "short_reason": ""}

            # ── Record 1: Original (ANSWERABLE) ──────────────────────────
            original_record = {
                "id": f"{row_index}_clean",
                "source_id": str(row_index),
                "image_path": str(original_path),
                "question": question,
                "answer": answer,
                "category": category,
                "variant": "clean",
                "perturbation_type": "none",
                "visual_cues": visual_cues,
                "cue_short_reason": cue_short_reason,
                "gemini_tag": {
                    "answerability": "ANSWERABLE",
                    "failure_type": "none",
                    "short_reason": "Original image is clear and unperturbed.",
                },
            }

            # ── Record 2: Perturbed (UNANSWERABLE) ───────────────────────
            perturbed_record = {
                "id": f"{row_index}_strong",
                "source_id": str(row_index),
                "original_image_path": str(original_path),
                "image_path": str(perturbed_path),
                "question": question,
                "answer": "I don't know",
                "category": category,
                "variant": "strong",
                "perturbation_type": perturbation_type,
                "visual_cues": new_cues["visual_cues"],
                "cue_short_reason": new_cues["short_reason"],
                "gemini_tag": tag,
            }

            out_f.write(json.dumps(original_record) + "\n")
            out_f.write(json.dumps(perturbed_record) + "\n")
            out_f.flush()

            unanswerable_count += 1
            print(f"  [{local_idx + 1}/{len(records_to_process)}] "
                  f"✅ UNANSWERABLE #{unanswerable_count} "
                  f"(row {row_index}, {perturbation_type})",
                  flush=True)

            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)

            # ── Stop early if target reached ──────────────────────────────
            if args.target_count > 0 and unanswerable_count >= args.target_count:
                print(f"\n🎯 Reached target of {args.target_count} "
                      "unanswerable pairs. Stopping.")
                break

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  UNANSWERABLE saved:  {unanswerable_count} "
          f"(× 2 records = {unanswerable_count * 2} JSONL lines)")
    print(f"  ANSWERABLE skipped:  {answerable_skip_count}")
    print(f"  Errors:              {error_count}")
    print(f"  Output JSONL:        {args.output_jsonl}")
    print(f"  Images saved:        {args.image_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Perturb, tag, save images + records for "
                    "UNANSWERABLE pairs only."
    )
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="JSONL from Step 1 (visual cues + row_index).")
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Original Parquet file (to re-read image bytes).")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path for the final output JSONL.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory to save original + perturbed images.")
    parser.add_argument("--target_count", type=int, default=0,
                        help="Stop after this many unanswerable pairs "
                             "(0 = process all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--sleep_interval", type=float, default=2.0,
                        help="Seconds to sleep between Gemini API calls.")
    parser.add_argument("--start_row", type=int, default=None,
                        help="0-based first JSONL row (inclusive).")
    parser.add_argument("--end_row", type=int, default=None,
                        help="0-based last JSONL row (exclusive).")
    args = parser.parse_args()
    main(args)
