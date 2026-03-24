import os
import io
import re
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/kalashkala/Perception-R1/.env')


# =========================================================
# CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"
OUT_DIR = "perturbed_vqa_training"
OUTPUT_JSON = f"{OUT_DIR}/perturbed_manifest.json"


# =========================================================
# IMAGE HELPERS
# =========================================================

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def downsample_restore(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    small = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def center_crop_restore(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = img.size
    cw = int(w * crop_ratio)
    ch = int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def random_occlusion(img: Image.Image, area_ratio: float, fill_mode: str = "gray") -> Image.Image:
    img = img.copy()
    w, h = img.size
    patch_area = int(w * h * area_ratio)
    pw = max(1, int(np.sqrt(patch_area)))
    ph = max(1, int(np.sqrt(patch_area)))
    pw = min(pw, w)
    ph = min(ph, h)

    x1 = random.randint(0, w - pw)
    y1 = random.randint(0, h - ph)
    x2, y2 = x1 + pw, y1 + ph

    if fill_mode == "gray":
        fill = (128, 128, 128)
    elif fill_mode == "black":
        fill = (0, 0, 0)
    elif fill_mode == "white":
        fill = (255, 255, 255)
    else:
        fill = tuple(np.random.randint(0, 256, size=3).tolist())

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=fill)
    return img


def darken_region(img: Image.Image, area_ratio: float, darkness: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape
    patch_area = int(w * h * area_ratio)
    pw = max(1, int(np.sqrt(patch_area)))
    ph = max(1, int(np.sqrt(patch_area)))
    pw = min(pw, w)
    ph = min(ph, h)

    x1 = random.randint(0, w - pw)
    y1 = random.randint(0, h - ph)
    x2, y2 = x1 + pw, y1 + ph

    arr[y1:y2, x1:x2] *= darkness
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def clutter_patch(img: Image.Image, area_ratio: float) -> Image.Image:
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    patch_area = int(w * h * area_ratio)
    pw = max(1, int(np.sqrt(patch_area)))
    ph = max(1, int(np.sqrt(patch_area)))
    pw = min(pw, w)
    ph = min(ph, h)

    x1 = random.randint(0, w - pw)
    y1 = random.randint(0, h - ph)
    x2, y2 = x1 + pw, y1 + ph

    noise = np.random.randint(0, 256, size=(ph, pw, 3), dtype=np.uint8)
    arr[y1:y2, x1:x2] = noise
    return Image.fromarray(arr)


# =========================================================
# CATEGORY-SPECIFIC PERTURBATIONS
# =========================================================

def get_ops(category: str, severity: str):
    assert severity in {"mild", "strong"}

    if category == "spatial_relational":
        mild = [
            ("blur_mild", lambda x: gaussian_blur(x, 1.5)),
            ("crop_mild", lambda x: center_crop_restore(x, 0.90)),
            ("occlusion_small", lambda x: random_occlusion(x, 0.05, "gray")),
        ]
        strong = [
            ("blur_strong", lambda x: gaussian_blur(x, 4.0)),
            ("crop_strong", lambda x: center_crop_restore(x, 0.60)),
            ("occlusion_large", lambda x: random_occlusion(x, 0.18, "black")),
            ("clutter_patch", lambda x: clutter_patch(x, 0.15)),
        ]
    elif category == "attribute_recognition":
        mild = [
            ("brightness_mild", lambda x: adjust_brightness(x, 0.85)),
            ("contrast_mild", lambda x: adjust_contrast(x, 0.85)),
            ("blur_mild", lambda x: gaussian_blur(x, 1.2)),
        ]
        strong = [
            ("brightness_strong", lambda x: adjust_brightness(x, 0.45)),
            ("contrast_strong", lambda x: adjust_contrast(x, 0.45)),
            ("blur_strong", lambda x: gaussian_blur(x, 3.5)),
            ("darken_region", lambda x: darken_region(x, 0.18, 0.08)),
        ]
    elif category == "counting":
        mild = [
            ("downsample_mild", lambda x: downsample_restore(x, 0.75)),
            ("occlusion_small", lambda x: random_occlusion(x, 0.05, "gray")),
            ("contrast_mild", lambda x: adjust_contrast(x, 0.85)),
        ]
        strong = [
            ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
            ("clutter_patch", lambda x: clutter_patch(x, 0.18)),
            ("occlusion_large", lambda x: random_occlusion(x, 0.18, "black")),
            ("crop_strong", lambda x: center_crop_restore(x, 0.65)),
        ]
    elif category == "existence_presence":
        mild = [
            ("blur_mild", lambda x: gaussian_blur(x, 1.2)),
            ("brightness_mild", lambda x: adjust_brightness(x, 0.85)),
            ("occlusion_small", lambda x: random_occlusion(x, 0.04, "gray")),
        ]
        strong = [
            ("occlusion_large", lambda x: random_occlusion(x, 0.18, "black")),
            ("darken_region", lambda x: darken_region(x, 0.20, 0.05)),
            ("crop_strong", lambda x: center_crop_restore(x, 0.60)),
            ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
        ]
    elif category == "activity_interaction":
        mild = [
            ("blur_mild", lambda x: gaussian_blur(x, 1.3)),
            ("crop_mild", lambda x: center_crop_restore(x, 0.90)),
            ("downsample_mild", lambda x: downsample_restore(x, 0.70)),
        ]
        strong = [
            ("blur_strong", lambda x: gaussian_blur(x, 4.0)),
            ("crop_strong", lambda x: center_crop_restore(x, 0.60)),
            ("occlusion_large", lambda x: random_occlusion(x, 0.16, "black")),
            ("darken_region", lambda x: darken_region(x, 0.18, 0.05)),
        ]
    elif category == "scene_context":
        mild = [
            ("brightness_mild", lambda x: adjust_brightness(x, 0.85)),
            ("downsample_mild", lambda x: downsample_restore(x, 0.70)),
            ("blur_mild", lambda x: gaussian_blur(x, 1.2)),
        ]
        strong = [
            ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
            ("blur_strong", lambda x: gaussian_blur(x, 4.0)),
            ("crop_strong", lambda x: center_crop_restore(x, 0.60)),
        ]
    else:
        mild = [
            ("blur_mild", lambda x: gaussian_blur(x, 1.2)),
            ("downsample_mild", lambda x: downsample_restore(x, 0.70)),
            ("crop_mild", lambda x: center_crop_restore(x, 0.92)),
        ]
        strong = [
            ("blur_strong", lambda x: gaussian_blur(x, 4.0)),
            ("downsample_strong", lambda x: downsample_restore(x, 0.25)),
            ("occlusion_large", lambda x: random_occlusion(x, 0.16, "black")),
            ("crop_strong", lambda x: center_crop_restore(x, 0.60)),
        ]

    return mild if severity == "mild" else strong


def apply_category_perturbation(img: Image.Image, category: str, severity: str) -> Tuple[Image.Image, str]:
    ops = get_ops(category, severity)
    op_name, op_fn = random.choice(ops)
    return op_fn(img), op_name


# =========================================================
# GEMINI HELPERS
# =========================================================

def build_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def build_tag_prompt(
    question: str,
    gold_answer: str,
    category: str,
    perturbation_type: str,
    visual_cues: List[str],
) -> str:
    cues_text = "\n".join([f"- {c}" for c in visual_cues]) if visual_cues else "- None provided"

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


def normalize_tag_output(obj: Dict) -> Dict:
    answerability = str(obj.get("answerability", "")).strip().upper()
    if answerability not in {"ANSWERABLE", "UNANSWERABLE"}:
        answerability = "ANSWERABLE"

    failure_type = str(obj.get("failure_type", "other")).strip().lower()
    allowed = {"none", "blur", "crop", "occlusion", "low_resolution", "darkness", "clutter", "other"}
    if failure_type not in allowed:
        failure_type = "other"

    short_reason = str(obj.get("short_reason", "")).strip()

    return {
        "answerability": answerability,
        "failure_type": failure_type,
        "short_reason": short_reason,
    }


def tag_perturbed_image(
    client: genai.Client,
    perturbed_img: Image.Image,
    question: str,
    gold_answer: str,
    category: str,
    perturbation_type: str,
    visual_cues: List[str],
    max_retries: int = 3,
) -> Dict:
    prompt = build_tag_prompt(
        question=question,
        gold_answer=gold_answer,
        category=category,
        perturbation_type=perturbation_type,
        visual_cues=visual_cues,
    )
    image_bytes = pil_to_jpeg_bytes(perturbed_img)

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
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(response.text)
            return normalize_tag_output(parsed)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to tag perturbed image: {last_err}")


# =========================================================
# MAIN
# =========================================================

def main(input_json: str, output_json: str, out_dir: str, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    client = build_gemini_client()

    with open(input_json, "r") as f:
        clean_records = json.load(f)

    image_out_dir = Path(out_dir) / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    half = len(clean_records) // 2

    for idx, rec in enumerate(clean_records):
        sample_id = str(rec["id"])
        image_path = rec["image_path"]
        question = rec["question"]
        answer = rec.get("answer", "")
        category = rec["category"]
        visual_cues = rec.get("visual_cues", [])

        severity = "mild" if idx < half else "strong"

        img = load_image(image_path)
        perturbed_img, perturbation_type = apply_category_perturbation(
            img=img,
            category=category,
            severity=severity,
        )

        out_name = f"{sample_id}_{category}_{severity}_{perturbation_type}.jpg"
        out_path = image_out_dir / out_name
        save_image(perturbed_img, str(out_path))

        tag = tag_perturbed_image(
            client=client,
            perturbed_img=perturbed_img,
            question=question,
            gold_answer=answer,
            category=category,
            perturbation_type=perturbation_type,
            visual_cues=visual_cues,
        )

        record = {
            "id": f"{sample_id}_{severity}",
            "source_id": sample_id,
            "image_path": str(out_path),
            "question": question,
            "answer": answer,
            "category": category,
            "variant": severity,
            "perturbation_type": perturbation_type,
            "visual_cues": visual_cues,
            "cue_short_reason": rec.get("cue_short_reason", ""),
            "gemini_tag": tag,
        }
        manifest.append(record)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(clean_records)}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved to {output_json}")


if __name__ == "__main__":
    input_json = "clean_vqa_with_visual_cues.json"
    main(input_json, OUTPUT_JSON, OUT_DIR)