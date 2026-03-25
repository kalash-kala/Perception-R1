
import re
import os
import json
from datetime import datetime

from openai import OpenAI, APIConnectionError, RateLimitError


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: VQA Answer Equivalence Evaluation
# ---------------------------------------------------------------------------
# The judge compares the model's predicted answer (from <answer>...</answer>)
# against the list of acceptable human answers and determines whether the
# prediction is semantically equivalent.
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

### Output a JSON blob with exactly two fields:
- "verdict": either "CORRECT" or "INCORRECT"
- "explanation": a brief explanation of your judgment (1-2 sentences)

Example output:
{"verdict": "CORRECT", "explanation": "The predicted answer 'carrots' matches several human annotations including 'carrot' and 'carrots'."}
"""

VQA_JUDGE_IN_CONTEXT_EXAMPLES = """Here are examples to guide your judgment:

Example 1 (CORRECT — exact match):
Question: "What color is the car?"
Acceptable answers: ["red", "red", "dark red", "red", "maroon", "red", "red", "dark red", "red", "red"]
Predicted answer: "red"
Output: {"verdict": "CORRECT", "explanation": "The prediction exactly matches the majority human answer."}

Example 2 (CORRECT — synonym / minor variation):
Question: "What type of plant is shown here?"
Acceptable answers: ["carrots, bok choy", "vegetables", "carrot", "carrots", "carrots", "carrot", "vegetable", "vegetables", "carrot", "collard greens"]
Predicted answer: "carrots"
Output: {"verdict": "CORRECT", "explanation": "'carrots' directly matches multiple human annotations."}

Example 3 (CORRECT — semantically equivalent):
Question: "What are the people doing?"
Acceptable answers: ["surfing", "surfing", "surf", "surfing", "water surfing", "riding waves", "surfing", "surfing", "surfing", "surfing"]
Predicted answer: "they are surfing"
Output: {"verdict": "CORRECT", "explanation": "The core concept 'surfing' matches the consensus human answer despite the extra words."}

Example 4 (INCORRECT — wrong answer):
Question: "How many people are in the image?"
Acceptable answers: ["3", "3", "three", "3", "3", "3", "3", "three", "3", "3"]
Predicted answer: "5"
Output: {"verdict": "INCORRECT", "explanation": "The predicted count '5' does not match the human consensus of '3'."}

Example 5 (INCORRECT — related but wrong):
Question: "What animal is shown?"
Acceptable answers: ["cat", "cat", "kitten", "cat", "cat", "cat", "cat", "cat", "kitten", "cat"]
Predicted answer: "dog"
Output: {"verdict": "INCORRECT", "explanation": "The prediction 'dog' is a different animal from the consensus answer 'cat'."}

Example 6 (CORRECT — more specific but valid):
Question: "What is on the table?"
Acceptable answers: ["food", "food", "meal", "food", "dinner", "food", "food", "food", "pizza", "food"]
Predicted answer: "pizza"
Output: {"verdict": "CORRECT", "explanation": "'pizza' is a specific type of food and matches at least one human annotation directly."}
"""


# ---------------------------------------------------------------------------
# OpenAI / vLLM client (module-level singleton)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "empty"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """
    Lightweight text normalization for VQA answers.
    Lowercase, strip whitespace / quotes and trailing punctuation.
    """
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
    """Call the judge LLM with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get(
                    "VQA_JUDGE_MODEL",
                    "/home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct",
                ),
                messages=messages,
                temperature=0,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            print(f"[VQAJudge] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                return None
        except Exception as e:
            print(f"[VQAJudge] Unexpected error: {e}")
            return None


def extract_last_json_blob(response):
    """Extract the last JSON-like blob from model text output."""
    if response is None:
        return None
    matches = re.findall(r"\{.*?\}", response, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1]


def parse_judge_response(response):
    """
    Parse the judge LLM response to extract the verdict.

    Returns:
        (verdict, explanation) where verdict is "CORRECT" / "INCORRECT"
        or (None, None) on parse failure.
    """
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
            print(f"[VQAJudge] Invalid verdict '{verdict}' in response: {response}")
            return None, None

        return verdict, explanation
    except Exception as e:
        print(f"[VQAJudge] Parsing error: {e}, response: {response}")
        return None, None


def get_vqa_judge_verdict(question, acceptable_answers, predicted_answer):
    """
    Call the judge LLM to evaluate whether the predicted answer is
    semantically equivalent to any acceptable human answer.

    Args:
        question: The original visual question.
        acceptable_answers: List of human-annotated answers.
        predicted_answer: The model's extracted answer string.

    Returns:
        (verdict, explanation)
        - verdict: "CORRECT", "INCORRECT", or None on failure
        - explanation: The judge's reasoning string
    """
    system_message = VQA_JUDGE_INSTRUCTIONS + "\n" + VQA_JUDGE_IN_CONTEXT_EXAMPLES

    # De-duplicate answers for a cleaner judge prompt
    unique_answers = list(dict.fromkeys(acceptable_answers))

    user_content = (
        f'Question: "{question}"\n'
        f'Acceptable answers: {json.dumps(unique_answers, ensure_ascii=False)}\n'
        f'Predicted answer: "{predicted_answer}"\n'
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_content},
    ]

    llm_response = attempt_api_call(messages, max_tokens=256)
    if llm_response is None:
        return None, "API call failed"

    verdict, explanation = parse_judge_response(llm_response)
    return verdict, explanation


def compute_vqa_accuracy(predicted, acceptable_answers):
    """
    Standard VQAv2 soft accuracy metric as a fast fallback.
    accuracy = min(1.0, num_exact_matches / 3)

    Used when the LLM judge is unavailable (API failure).
    """
    pred_norm = normalize_answer(predicted)
    match_count = sum(
        1 for ans in acceptable_answers
        if normalize_answer(ans) == pred_norm
    )
    return min(1.0, match_count / 3.0)


# ---------------------------------------------------------------------------
# Prompt / response parsing — Perception-R1-style format
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


def has_perception_r1_format(solution_str):
    """
    Format is considered valid if the <answer> tag (or legacy /box[]) exists.
    <think> block is encouraged but not required for a valid format.
    """
    has_answer = re.search(
        r"<answer>\s*.*?\s*</answer>",
        solution_str,
        flags=re.DOTALL | re.IGNORECASE,
    ) is not None
    has_legacy = re.search(r"/box\[(.*?)\]/?", solution_str, flags=re.DOTALL) is not None
    return has_answer or has_legacy


def clean_user_question_text(text):
    """
    Remove <image> token and format-instruction suffixes from the user message.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<image>\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove common format instructions appended in the parquet builder
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
    Recover the raw question from multiple possible sources, in priority order:
      1. kwargs['prompt']  — list of chat messages (preferred VERL parquet format)
      2. kwargs['extra_info']['question'] / ground_truth['question']
      3. Legacy extra_info['prompt_text'] serialization
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

    # Explicit question field
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


# ---------------------------------------------------------------------------
# Main reward function — entry point called by VERL
# ---------------------------------------------------------------------------

def compute_score(solution_str, ground_truth, method="strict",
                  format_score=-1.0, score=1.0, **kwargs):
    """
    VQA LLM-as-Judge Reward Function (Perception-R1 data intake + TruthRL judge).

    Two rewards only:

        1. FORMAT REWARD
           Checks that the response follows the Perception-R1 format:
             <think>...</think>  (encouraged but not required)
             <answer>...</answer>
           Missing format → format_score penalty (default -1.0).

        2. TERNARY ANSWER REWARD  (LLM-judge based)
           - CORRECT verdict  →  +score   (default +1.0)
           - INCORRECT verdict →  -1.0
           - Judge unavailable  →  soft VQAv2 accuracy mapped to [-1, +1]
                                   (continuous fallback, partial credit)

        Abstention ("I don't know" etc.) → 0.0 (safe refusal).

    Args:
        solution_str:  The model's full generated output string.
        ground_truth:  Dict from the parquet reward_model column:
                       {
                         "acceptable_answers": [...],
                         "multiple_choice_answer": "...",
                         "style": "vqa_llm_judge",
                         "response_format": "perception_r1",
                         "answerability": "...",
                         "variant": "...",
                         "perturbation_type": "..."
                       }
        format_score:  Penalty for missing <answer> format (default -1.0).
        score:         Reward for a correct answer (default +1.0).

    Returns:
        dict with keys: score, accuracy, abstention, format_error
    """

    # ------------------------------------------------------------------
    # A. Read reward metadata
    # ------------------------------------------------------------------
    if isinstance(ground_truth, dict):
        acceptable_answers = ground_truth.get("acceptable_answers", [])
        mc_answer          = ground_truth.get("multiple_choice_answer", "")
        answerability      = ground_truth.get("answerability", "")
        variant            = ground_truth.get("variant", "original")
        perturbation_type  = ground_truth.get("perturbation_type", "none")
        response_format    = ground_truth.get("response_format", "perception_r1")
    else:
        # Shouldn't happen with properly formatted parquet, but be safe
        acceptable_answers = [str(ground_truth)]
        mc_answer          = str(ground_truth)
        answerability      = ""
        variant            = "original"
        perturbation_type  = "none"
        response_format    = "perception_r1"

    if not isinstance(acceptable_answers, list):
        acceptable_answers = [str(acceptable_answers)]

    # ------------------------------------------------------------------
    # B. FORMAT REWARD — check <answer> tag presence
    # ------------------------------------------------------------------
    if response_format == "perception_r1":
        format_ok = has_perception_r1_format(solution_str)
    else:
        format_ok = True  # future-proof for other formats

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
        }

    predicted_answer = predicted_answer.strip()

    # ------------------------------------------------------------------
    # C. Question extraction (for the LLM judge)
    # ------------------------------------------------------------------
    question_for_judge = extract_question_from_inputs(kwargs=kwargs, ground_truth=ground_truth)

    # ------------------------------------------------------------------
    # D. Abstention check (safe refusal — better than hallucinating)
    # ------------------------------------------------------------------
    pred_normalized = normalize_answer(predicted_answer)
    unknown_triggers = [
        "i don't know", "i dont know", "unsure", "i do not know",
        "not sure", "cannot determine", "can't determine",
        "unable to determine", "cannot tell", "can't tell",
        "cannot be determined", "can't be determined",
        "insufficient information",
    ]
    is_abstention = any(trigger in pred_normalized for trigger in unknown_triggers)

    if is_abstention:
        log_reward_detail({
            "event": "abstention",
            "question": question_for_judge,
            "predicted_answer": predicted_answer,
            "answerability": answerability,
            "variant": variant,
            "perturbation_type": perturbation_type,
            "score": 0.0,
        })

        return {
            "score": 0.0,
            "accuracy": 0.0,
            "abstention": 1.0,
            "format_error": 0.0,
        }

    # ------------------------------------------------------------------
    # E. Quick VQA soft accuracy (used as fallback & for logging)
    # ------------------------------------------------------------------
    soft_accuracy = compute_vqa_accuracy(predicted_answer, acceptable_answers)

    # ------------------------------------------------------------------
    # F. TERNARY ANSWER REWARD — call LLM judge
    # ------------------------------------------------------------------
    verdict, explanation = get_vqa_judge_verdict(
        question_for_judge, acceptable_answers, predicted_answer
    )

    if verdict == "CORRECT":
        final_score = float(score)   # default +1.0
        accuracy    = 1.0
    elif verdict == "INCORRECT":
        final_score = -1.0
        accuracy    = 0.0
    else:
        # Judge failed — fall back to soft accuracy mapped to [-1, +1]
        # This gives partial credit proportional to human agreement
        final_score = (soft_accuracy * 2.0) - 1.0
        accuracy    = soft_accuracy
        explanation = f"Judge unavailable, fell back to soft accuracy: {soft_accuracy:.2f}"

    # ------------------------------------------------------------------
    # G. Logging
    # ------------------------------------------------------------------
    log_reward_detail({
        "event": "scored",
        "question": question_for_judge,
        "predicted_answer": predicted_answer,
        "acceptable_answers": acceptable_answers[:5],  # truncate for log size
        "mc_answer": mc_answer,
        "soft_accuracy": soft_accuracy,
        "judge_verdict": verdict,
        "judge_explanation": explanation,
        "final_score": final_score,
        "answerability": answerability,
        "variant": variant,
        "perturbation_type": perturbation_type,
        "response_format": response_format,
    })

    return {
        "score": float(final_score),
        "accuracy": float(accuracy),
        "abstention": 0.0,
        "format_error": 0.0,
    }
