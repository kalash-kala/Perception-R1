import os
import io
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/sriramg/kalashabhayk/Perception-R1/.env')


# =========================================================
# CONFIG
# =========================================================

GEMINI_MODEL = "gemini-2.5-flash"
OUTPUT_JSON = "clean_vqa_with_visual_cues.json"


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
                    max_output_tokens=256,
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

def main(input_json: str, output_json: str) -> None:
    client = build_gemini_client()

    with open(input_json, "r") as f:
        samples = json.load(f)

    output = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        category = assign_category(question)

        cues = get_visual_cues(
            client=client,
            image_path=sample["image_path"],
            question=question,
            gold_answer=sample.get("answer", ""),
            category=category,
        )

        record = {
            "id": str(sample["id"]),
            "image_path": sample["image_path"],
            "question": question,
            "answer": sample.get("answer", ""),
            "category": category,
            "visual_cues": cues["visual_cues"],
            "cue_short_reason": cues["short_reason"],
        }
        output.append(record)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(samples)}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_json}")


if __name__ == "__main__":
    input_json = "vqa_samples.json"
    main(input_json, OUTPUT_JSON)