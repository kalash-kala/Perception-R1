import re
import os
import json
from datetime import datetime
from collections import Counter

from openai import OpenAI, APIConnectionError, RateLimitError


# ---------------------------------------------------------------------------
# 1) ANSWER JUDGE PROMPT
# ---------------------------------------------------------------------------

VQA_JUDGE_INSTRUCTIONS = """You are an expert evaluator for Visual Question Answering tasks.

You will be given:
1. A visual question that was asked about an image.
2. A list of acceptable human-provided ground truth answers.
3. A model-predicted answer.

Your task is to judge whether the predicted answer is semantically equivalent to any of the acceptable answers.

Rules:
- The predicted answer does NOT need to be an exact string match.
- Minor spelling differences, synonyms, or paraphrases of the same concept should be treated as CORRECT.
- Answers that are more specific but still correct (e.g., "golden retriever" when acceptable answers include "dog") should be considered CORRECT.
- Answers that are too vague, wrong, or unrelated should be considered INCORRECT.
- If the answer is a reasonable response to the question but not covered by the acceptable answers, still mark it INCORRECT — we only reward answers consistent with human consensus.

Output JSON with exactly two fields:
{
  "verdict": "CORRECT or INCORRECT",
  "explanation": "brief explanation"
}
"""

VQA_JUDGE_IN_CONTEXT_EXAMPLES = """Here are examples to guide your judgment:

Example 1:
Question: "What color is the car?"
Acceptable answers: ["red", "red", "dark red", "red", "maroon", "red", "red", "dark red", "red", "red"]
Predicted answer: "red"
Output: {"verdict": "CORRECT", "explanation": "The prediction exactly matches the majority human answer."}

Example 2:
Question: "What are the people doing?"
Acceptable answers: ["surfing", "surfing", "surf", "surfing", "water surfing", "riding waves", "surfing", "surfing", "surfing", "surfing"]
Predicted answer: "they are surfing"
Output: {"verdict": "CORRECT", "explanation": "The core concept 'surfing' matches the consensus human answer."}

Example 3:
Question: "How many people are in the image?"
Acceptable answers: ["3", "3", "three", "3", "3", "3", "3", "three", "3", "3"]
Predicted answer: "5"
Output: {"verdict": "INCORRECT", "explanation": "The predicted count does not match the consensus answer."}
"""


# ---------------------------------------------------------------------------
# 2) VISUAL CUE JUDGE PROMPT
# ---------------------------------------------------------------------------

VISUAL_JUDGE_INSTRUCTIONS = """You are an expert evaluator for multimodal reasoning grounding.

You will be given:
1. A visual question.
2. A list of canonical visual cues extracted from the image.
3. The model's reasoning text.

Your job is to decide whether the reasoning is consistent with each visual cue.

Rules:
- Evaluate ONLY whether the reasoning reflects or is consistent with each cue.
- Do not check final answer correctness here.
- If a cue is clearly reflected, mark 1.
- If a cue is missing, contradicted, or unsupported, mark 0.
- Be conservative.
- Output JSON only.

Required JSON schema:
{
  "cue_scores": [0, 1, 1],
  "support_score": 0.67,
  "explanation": "brief explanation"
}
"""

VISUAL_JUDGE_IN_CONTEXT_EXAMPLES = """Example 1:
Question: "What color is the bus?"
Visual cues: ["The bus is yellow.", "The bus occupies the center of the image."]
Reasoning: "The bus in the image is clearly yellow, so the answer is yellow."
Output: {"cue_scores": [1, 1], "support_score": 1.0, "explanation": "The reasoning reflects both the bus color and the referenced object."}

Example 2:
Question: "Where is the cat?"
Visual cues: ["The cat is on the sofa.", "The cat is near the left side of the image."]
Reasoning: "I see a cat somewhere in the room, but I cannot tell where exactly."
Output: {"cue_scores": [0, 0], "support_score": 0.0, "explanation": "The reasoning does not reflect the specific visual cues."}
"""


# ---------------------------------------------------------------------------
# 3) OpenAI / vLLM client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "empty"),
)


# ---------------------------------------------------------------------------
# 4) Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """Lightweight text normalization for VQA answers."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.strip('"').strip("'")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def log_reward_detail(data):
    """Optional JSONL logging — activate with TRUTHRL_ENABLE_TRAIN_LOGS=1."""
    if os.environ.get("TRUTHRL_ENABLE_TRAIN_LOGS") != "1":
        return

    log_name = os.environ.get("TRUTHRL_LOG_NAME", "training_default")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
    log_dir = os.path.join(project_root, "outputs/reward_logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"vqa_reward_detail_{os.getpid()}.jsonl")
    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to write VQA reward log: {e}")


def attempt_api_call(messages, max_retries=3, max_tokens=256):
    """Call a judge LLM with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get(
                    "VQA_JUDGE_MODEL",
                    "/home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct",
                ),
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            print(f"[Judge] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                return None
        except Exception as e:
            print(f"[Judge] Unexpected error: {e}")
            return None


def extract_last_json_blob(response):
    """Extract the last JSON-like blob from model text output."""
    if response is None:
        return None
    matches = re.findall(r"\{.*?\}", response, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1]


def parse_answer_judge_response(response):
    if response is None:
        return None, None

    text = extract_last_json_blob(response)
    if text is None:
        return None, None

    try:
        obj = json.loads(text)
        verdict = str(obj.get("verdict", "")).upper().strip()
        explanation = str(obj.get("explanation", "")).strip()

        if verdict not in ("CORRECT", "INCORRECT"):
            print(f"[AnswerJudge] Invalid verdict '{verdict}' in response: {response}")
            return None, None

        return verdict, explanation
    except Exception as e:
        print(f"[AnswerJudge] Parsing error: {e}, response: {response}")
        return None, None


def parse_visual_judge_response(response, num_cues):
    if response is None:
        return None, None, None

    text = extract_last_json_blob(response)
    if text is None:
        return None, None, None

    try:
        obj = json.loads(text)
        cue_scores = obj.get("cue_scores", [])
        explanation = str(obj.get("explanation", "")).strip()

        if not isinstance(cue_scores, list):
            return None, None, None

        cleaned = []
        for x in cue_scores[:num_cues]:
            if str(x).strip() in ("1", "true", "True"):
                cleaned.append(1)
            else:
                cleaned.append(0)

        if len(cleaned) < num_cues:
            cleaned.extend([0] * (num_cues - len(cleaned)))

        support_score = sum(cleaned) / max(1, num_cues)
        return float(support_score), cleaned, explanation

    except Exception as e:
        print(f"[VisualJudge] Parsing error: {e}, response: {response}")
        return None, None, None


def get_vqa_judge_verdict(question, acceptable_answers, predicted_answer):
    system_message = VQA_JUDGE_INSTRUCTIONS + "\n\n" + VQA_JUDGE_IN_CONTEXT_EXAMPLES
    unique_answers = list(dict.fromkeys(acceptable_answers))

    user_content = (
        f'Question: "{question}"\n'
        f'Acceptable answers: {json.dumps(unique_answers, ensure_ascii=False)}\n'
        f'Predicted answer: "{predicted_answer}"\n'
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    llm_response = attempt_api_call(messages, max_tokens=128)
    if llm_response is None:
        return None, "API call failed"

    verdict, explanation = parse_answer_judge_response(llm_response)
    return verdict, explanation


def get_visual_cue_support(question, visual_cues, reasoning_text):
    if not visual_cues:
        return 0.0, [], "No visual cues provided."

    if not reasoning_text:
        return 0.0, [0] * len(visual_cues), "Reasoning text is empty."

    system_message = VISUAL_JUDGE_INSTRUCTIONS + "\n\n" + VISUAL_JUDGE_IN_CONTEXT_EXAMPLES

    user_content = (
        f'Question: "{question}"\n'
        f'Visual cues: {json.dumps(visual_cues, ensure_ascii=False)}\n'
        f'Reasoning: "{reasoning_text}"\n'
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    llm_response = attempt_api_call(messages, max_tokens=192)
    if llm_response is None:
        return None, None, "API call failed"

    return parse_visual_judge_response(llm_response, num_cues=len(visual_cues))


def compute_vqa_accuracy(predicted, acceptable_answers):
    """
    Standard VQAv2 soft accuracy metric as a fast fallback.
    accuracy = min(1.0, num_exact_matches / 3)
    """
    pred_norm = normalize_answer(predicted)
    match_count = sum(
        1 for ans in acceptable_answers
        if normalize_answer(ans) == pred_norm
    )
    return min(1.0, match_count / 3.0)


# ---------------------------------------------------------------------------
# Prompt / response parsing for new Perception-R1-style format
# ---------------------------------------------------------------------------

def extract_tag_content(text, tag):
    """
    Extract content inside an XML-style tag, e.g. <think>...</think>.
    Returns None if the tag is absent.
    """
    if not isinstance(text, str):
        return None

    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def extract_answer_text(solution_str):
    """
    Preferred new format:
      <answer> ... </answer>

    Backward compatibility:
      /box[...]
    """
    answer = extract_tag_content(solution_str, "answer")
    if answer is not None:
        return answer

    legacy = re.search(r"/box\[(.*?)\]/?", solution_str, flags=re.DOTALL)
    if legacy:
        return legacy.group(1).strip()

    return None


def extract_reasoning_text(solution_str, predicted_answer=None):
    """
    Preferred new format:
      <think> ... </think>

    Fallback:
      strip the answer block from the whole response.
    """
    reasoning = extract_tag_content(solution_str, "think")
    if reasoning:
        return reasoning

    cleaned = re.sub(
        r"<answer>\s*.*?\s*</answer>",
        "",
        solution_str,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"/box\[(.*?)\]/?", "", cleaned, flags=re.DOTALL)

    if predicted_answer:
        cleaned = cleaned.replace(predicted_answer, "")

    cleaned = cleaned.strip()
    return cleaned


def has_perception_r1_format(solution_str):
    """
    Format is considered valid if the answer tag exists.
    Optionally we also reward/expect a think block, but we do not hard-fail
    if <think> is absent and <answer> is present.
    """
    has_answer = re.search(r"<answer>\s*.*?\s*</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE) is not None
    has_legacy = re.search(r"/box\[(.*?)\]/?", solution_str, flags=re.DOTALL) is not None
    return has_answer or has_legacy


def clean_user_question_text(text):
    """
    Remove image token and formatting suffix from the user message.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<image>\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove common format instructions appended in your parquet builder
    text = re.sub(
        r"\n?\s*Output the thinking process in <think>\s*</think>\s*and the final answer in <answer>\s*</answer>\s*tags\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"\n?\s*The reasoning process MUST BE enclosed within <think>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    return text


def extract_question_from_inputs(kwargs=None, ground_truth=None):
    """
    Recover raw question from:
      1. kwargs['prompt']  (preferred, matches new parquet format)
      2. kwargs['extra_info']['question'] / ground_truth['question']
      3. legacy extra_info['prompt_text']
    """
    kwargs = kwargs or {}
    extra_info = kwargs.get("extra_info", {}) if isinstance(kwargs, dict) else {}
    prompt = kwargs.get("prompt", None) if isinstance(kwargs, dict) else None

    # Preferred: prompt is a list of chat messages from VERL parquet
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = clean_user_question_text(msg.get("content", ""))
                if content:
                    return content

    # Sometimes prompt may arrive as a dict-like serialized object
    if isinstance(prompt, dict):
        user_content = prompt.get("content", "")
        content = clean_user_question_text(user_content)
        if content:
            return content

    # Explicit question field if you decide to store it
    if isinstance(extra_info, dict):
        q = extra_info.get("question", "")
        if isinstance(q, str) and q.strip():
            return q.strip()

    if isinstance(ground_truth, dict):
        q = ground_truth.get("question", "")
        if isinstance(q, str) and q.strip():
            return q.strip()

    # Legacy fallback: old prompt_text serialization
    prompt_text = ""
    if isinstance(extra_info, dict):
        prompt_text = extra_info.get("prompt_text", "")

    if isinstance(prompt_text, str) and prompt_text.strip():
        match = re.search(r"(?:\n|^)user\n(.*?)(?:\nassistant\n?$|$)", prompt_text, re.DOTALL)
        if match:
            question = match.group(1).strip()
        else:
            question = prompt_text.strip()
            if question and "\n" in question:
                question = question.split("\n")[-1].strip()

        question = clean_user_question_text(question)
        if question:
            return question

    return "(question unavailable)"


def compute_repetition_penalty(text, n=3, coef=0.2):
    """
    Simple n-gram repetition penalty.
    Returns a non-positive value.

    rep_ratio = repeated_ngrams / total_ngrams
    penalty = -coef * rep_ratio
    """
    if not text or not isinstance(text, str):
        return 0.0

    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)

    if total == 0:
        return 0.0

    rep_ratio = repeated / total
    return -coef * rep_ratio


# ---------------------------------------------------------------------------
# 5) Main reward function — VERL entry point
# ---------------------------------------------------------------------------

def compute_score(solution_str, ground_truth, method="strict",
                  format_score=-1.0, score=1.0, **kwargs):
    """
    Extended VQA reward with:
      - format check for <think> ... </think> + <answer> ... </answer>
      - answer correctness judge
      - visual grounding reward
      - repetition penalty

    Expected ground_truth / reward_model format:
    {
      "acceptable_answers": [...],
      "multiple_choice_answer": "",
      "style": "vqa_llm_judge",
      "response_format": "perception_r1",
      "answerability": "...",
      "visual_cues": [...],
      "cue_source": "...",
      "variant": "...",
      "perturbation_type": "..."
    }
    """

    # ------------------------------------------------------------------
    # A. Read reward metadata
    # ------------------------------------------------------------------
    if isinstance(ground_truth, dict):
        acceptable_answers = ground_truth.get("acceptable_answers", [])
        mc_answer = ground_truth.get("multiple_choice_answer", "")
        visual_cues = ground_truth.get("visual_cues", [])
        answerability = ground_truth.get("answerability", "")
        variant = ground_truth.get("variant", "original")
        perturbation_type = ground_truth.get("perturbation_type", "none")
        response_format = ground_truth.get("response_format", "perception_r1")
        cue_source = ground_truth.get("cue_source", "")
    else:
        acceptable_answers = [str(ground_truth)]
        mc_answer = str(ground_truth)
        visual_cues = []
        answerability = ""
        variant = "original"
        perturbation_type = "none"
        response_format = "perception_r1"
        cue_source = ""

    if not isinstance(acceptable_answers, list):
        acceptable_answers = [str(acceptable_answers)]

    # Hyperparameters
    visual_gamma = float(os.environ.get("VQA_VISUAL_REWARD_GAMMA", "0.5"))
    repetition_coef = float(os.environ.get("VQA_REPETITION_COEF", "0.2"))
    repetition_ngram = int(os.environ.get("VQA_REPETITION_NGRAM", "3"))
    apply_rep_to_abstention = os.environ.get("VQA_APPLY_REP_TO_ABSTENTION", "0") == "1"

    # ------------------------------------------------------------------
    # B. Format check + answer extraction
    # ------------------------------------------------------------------
    if response_format == "perception_r1":
        format_ok = has_perception_r1_format(solution_str)
    else:
        format_ok = True  # fallback for other formats if you add them later

    predicted_answer = extract_answer_text(solution_str)

    if (not format_ok) or (predicted_answer is None):
        penalty = format_score if format_score < 0 else -1.0

        log_reward_detail({
            "event": "format_missing",
            "score": penalty,
            "response_format": response_format,
            "solution_preview": solution_str[:300],
        })

        return {
            "score": float(penalty),
            "accuracy": 0.0,
            "abstention": 0.0,
            "format_error": float(penalty),
            "base_answer_score": float(penalty),
            "visual_reward": 0.0,
            "repetition_penalty": 0.0,
        }

    predicted_answer = predicted_answer.strip()

    # ------------------------------------------------------------------
    # C. Question extraction
    # ------------------------------------------------------------------
    question_for_judge = extract_question_from_inputs(kwargs=kwargs, ground_truth=ground_truth)

    # ------------------------------------------------------------------
    # D. Reasoning text + repetition penalty
    # ------------------------------------------------------------------
    reasoning_text = extract_reasoning_text(solution_str, predicted_answer=predicted_answer)

    repetition_source = reasoning_text if reasoning_text else solution_str
    repetition_penalty = compute_repetition_penalty(
        repetition_source,
        n=repetition_ngram,
        coef=repetition_coef,
    )

    # ------------------------------------------------------------------
    # E. Abstention check
    # ------------------------------------------------------------------
    pred_normalized = normalize_answer(predicted_answer)
    unknown_triggers = [
        "i don't know", "i dont know", "unsure", "i do not know",
        "not sure", "cannot determine", "can't determine",
        "unable to determine", "cannot tell", "can't tell",
        "cannot be determined", "can't be determined",
        "insufficient information"
    ]
    is_abstention = any(trigger in pred_normalized for trigger in unknown_triggers)

    if is_abstention:
        final_score = 0.0 + (repetition_penalty if apply_rep_to_abstention else 0.0)

        log_reward_detail({
            "event": "abstention",
            "question": question_for_judge,
            "predicted_answer": predicted_answer,
            "reasoning_text_preview": reasoning_text[:200],
            "answerability": answerability,
            "variant": variant,
            "perturbation_type": perturbation_type,
            "visual_reward": 0.0,
            "repetition_penalty": repetition_penalty if apply_rep_to_abstention else 0.0,
            "score": final_score,
        })

        return {
            "score": float(final_score),
            "accuracy": 0.0,
            "abstention": 1.0,
            "format_error": 0.0,
            "base_answer_score": 0.0,
            "visual_reward": 0.0,
            "repetition_penalty": float(repetition_penalty if apply_rep_to_abstention else 0.0),
        }

    # ------------------------------------------------------------------
    # F. Base answer reward
    # ------------------------------------------------------------------
    soft_accuracy = compute_vqa_accuracy(predicted_answer, acceptable_answers)
    verdict, explanation = get_vqa_judge_verdict(
        question_for_judge, acceptable_answers, predicted_answer
    )

    if verdict == "CORRECT":
        base_answer_score = float(score)
        accuracy = 1.0
    elif verdict == "INCORRECT":
        base_answer_score = -1.0
        accuracy = 0.0
    else:
        base_answer_score = (soft_accuracy * 2.0) - 1.0
        accuracy = soft_accuracy
        explanation = f"Judge unavailable, fell back to soft accuracy: {soft_accuracy:.2f}"

    # ------------------------------------------------------------------
    # G. Visual grounding reward
    # ------------------------------------------------------------------
    visual_support_score, cue_scores, visual_explanation = get_visual_cue_support(
        question_for_judge, visual_cues, reasoning_text
    )

    if visual_support_score is None:
        visual_reward = 0.0
        cue_scores = []
        visual_explanation = "Visual judge unavailable."
    else:
        visual_reward = visual_gamma * visual_support_score

    # ------------------------------------------------------------------
    # H. Final score
    # ------------------------------------------------------------------
    final_score = base_answer_score + visual_reward + repetition_penalty

    # ------------------------------------------------------------------
    # I. Logging
    # ------------------------------------------------------------------
    log_reward_detail({
        "event": "scored",
        "question": question_for_judge,
        "predicted_answer": predicted_answer,
        "acceptable_answers": acceptable_answers[:5],
        "mc_answer": mc_answer,
        "soft_accuracy": soft_accuracy,
        "judge_verdict": verdict,
        "judge_explanation": explanation,
        "base_answer_score": base_answer_score,
        "visual_cues": visual_cues[:5],
        "cue_source": cue_source,
        "cue_scores": cue_scores,
        "visual_support_score": visual_support_score,
        "visual_reward": visual_reward,
        "visual_explanation": visual_explanation,
        "repetition_penalty": repetition_penalty,
        "final_score": final_score,
        "answerability": answerability,
        "variant": variant,
        "perturbation_type": perturbation_type,
        "response_format": response_format,
        "reasoning_text_preview": reasoning_text[:300],
    })

    return {
        "score": float(final_score),
        "accuracy": float(accuracy),
        "abstention": 0.0,
        "format_error": 0.0,
        "base_answer_score": float(base_answer_score),
        "visual_reward": float(visual_reward),
        "repetition_penalty": float(repetition_penalty),
    }