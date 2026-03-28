#!/usr/bin/env python3
"""
Evaluate Qwen2.5-VL on VSR open-text task using LLM-as-Judge verification.
(Perception-R1 format: <think>...</think> + <answer>...</answer>)

This script:
  1. Loads a VSR open-text parquet (with open-ended questions and caption ground truths).
  2. Runs inference with Qwen2.5-VL to generate answers.
  3. Extracts the answer from <answer>...</answer> tags.
  4. Extracts reasoning from <think>...</think> tags.
  5. Checks for abstention ("I don't know") → score 0.0.
  6. Uses an LLM judge (via vLLM/OpenAI-compatible API) to verify whether
     the model's answer is semantically equivalent to the ground truth caption.
  7. Logs detailed per-sample results and aggregated metrics.

Reward structure:
  +1.0  Correct (LLM judge says equivalent)
   0.0  Abstention ("I don't know")
  -1.0  Incorrect / Hallucination / Missing format

Usage:
  # 1. Start the judge LLM server (in a separate terminal):
  python3 -m vllm.entrypoints.openai.api_server \
    --model /home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct \
    --dtype auto --port 8000 --gpu-memory-utilization 0.85

  # 2. Run evaluation:
  python3 /home/kalashkala/Perception-R1/evaluation/evaluate_vsr_llm_verifier.py \
    --model_path /home/kalashkala/Models/Qwen2.5-VL-3B-Instruct \
    --data_path /home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train_open_text.parquet \
    --judge_api_base http://localhost:8000/v1 \
    --judge_model /home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct \
    --name vsr_open_text_eval
"""

import os
import re
import json
import copy
import ast
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI, APIConnectionError, RateLimitError


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: Semantic Equivalence Evaluation for Open-Text VSR
# ---------------------------------------------------------------------------

VQA_JUDGE_INSTRUCTIONS = """You are an expert evaluator for Visual Question Answering tasks.

You will be given:
1. A visual question that was asked about an image.
2. The ground truth answer.
3. A model-predicted answer.

Your task is to judge whether the predicted answer is semantically equivalent to the ground truth.

Rules:
- The predicted answer does NOT need to be an exact string match.
- Minor spelling differences, synonyms, or paraphrases of the same spatial relationship should be treated as CORRECT.
- Answers that capture the same meaning as the ground truth but use different wording are CORRECT (e.g., "to the left of" vs "on the left side of" or "The object is a table" vs "table" vs "The furniture is a table").
- Answers that mean different than the ground truth or are factually wrong should be INCORRECT.
- If the predicted answer is too vague to confirm the ground truth, mark it INCORRECT.

### Output a JSON blob with exactly two fields:
- "verdict": either "CORRECT" or "INCORRECT"
- "explanation": a brief explanation of your judgment (1-2 sentences)

Example output:
{"verdict": "CORRECT", "explanation": "The predicted answer correctly identifies the spatial relationship described in the ground truth."}
"""

VQA_JUDGE_IN_CONTEXT_EXAMPLES = """Here are examples to guide your judgment:

Example 1 (CORRECT — same relationship, different wording):
Question: "Where is the wine glass in relation to the dining table?"
Ground truth: "The wine glass is at the right side of the dining table."
Predicted answer: "The wine glass is on the right of the dining table."
Output: {"verdict": "CORRECT", "explanation": "Both describe the wine glass being to the right of the dining table."}

Example 2 (CORRECT — concise but accurate):
Question: "Where is the cup relative to the sandwich?"
Ground truth: "The cup is in front of the sandwich."
Predicted answer: "in front of the sandwich"
Output: {"verdict": "CORRECT", "explanation": "The prediction captures the same spatial relationship as the ground truth."}

Example 3 (INCORRECT — wrong spatial relationship):
Question: "Where is the cat relative to the dog?"
Ground truth: "The cat is behind the dog."
Predicted answer: "The cat is next to the dog on the left side."
Output: {"verdict": "INCORRECT", "explanation": "The prediction says 'next to on the left side' but the ground truth says 'behind'."}

Example 4 (INCORRECT — too vague):
Question: "Where is the lamp relative to the sofa?"
Ground truth: "The lamp is above the sofa."
Predicted answer: "The lamp is near the sofa."
Output: {"verdict": "INCORRECT", "explanation": "'Near' is too vague and does not specify the 'above' relationship from the ground truth."}

Example 5 (CORRECT — semantically identical):
Question: "Where is the book?"
Ground truth: "The book is on the table."
Predicted answer: "It is sitting on top of the table."
Output: {"verdict": "CORRECT", "explanation": "'Sitting on top of the table' is semantically identical to 'on the table'."}

Example 6 (INCORRECT — completely wrong):
Question: "Where is the dog relative to the boy?"
Ground truth: "The dog is to the left of the boy."
Predicted answer: "The dog is behind the boy."
Output: {"verdict": "INCORRECT", "explanation": "The prediction says 'behind' but the ground truth says 'to the left of'."}

Example 7 (CORRECT - semantically equivalent)
Question: "What is the piece of object is to the right of the bed near the window?"
Ground truth: "The piece of furniture is a Lamp."
Predicted answer: "Lamp"
Output: {"verdict": "CORRECT", "explanation": "The predicted answer 'Lamp' describes what is given in the ground truth."}

Example 8 (INCORRECT - semantically equivalent)
Question: "What is the piece of object is to the right of the bed near the window?"
Ground truth: "The piece of furniture is a Lamp."
Predicted answer: "chair"
Output: {"verdict": "INCORRECT", "explanation": "The predicted answer 'chair' does not describe what is given in the ground truth which is a lamp."}

Example 9 (INCORRECT - completely wrong)
Question: "What is the piece of object is to the right of the bed near the window?"
Ground truth: "The piece of furniture is a Lamp."
Predicted answer: "The furniture to the left of the bed is a chair."
Output: {"verdict": "INCORRECT", "explanation": "The predicted answer 'The furniture to the left of the bed is a chair' does not describe what is given in the ground truth which is a lamp on the right of the bed near the window."}

"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """Lowercase, strip whitespace and trailing punctuation."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.rstrip(".")
    return s


def log_reward_detail(data, run_dir):
    """Logs detailed reward information to a JSONL file."""
    log_file = os.path.join(run_dir, "judge_reward_detail.jsonl")
    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Failed to write reward log: {e}")


# ---------------------------------------------------------------------------
# Answer / Reasoning Extraction (Perception-R1 format)
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
    Extract the predicted answer from <answer>...</answer> tags.
    Returns None if no valid answer tag is found.
    """
    return extract_tag_content(solution_str, "answer")


def extract_reasoning_text(solution_str, predicted_answer=None):
    """
    Extract reasoning from <think>...</think> tags.
    Falls back to stripping the answer block from the whole response.
    """
    reasoning = extract_tag_content(solution_str, "think")
    if reasoning:
        return reasoning

    # Fallback: strip the answer block from the whole response
    cleaned = re.sub(
        r"<answer>\s*.*?\s*</answer>",
        "",
        solution_str,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if predicted_answer:
        cleaned = cleaned.replace(predicted_answer, "")

    cleaned = cleaned.strip()
    return cleaned


def has_valid_format(solution_str):
    """
    Format is considered valid if the <answer> tag exists.
    """
    return re.search(
        r"<answer>\s*.*?\s*</answer>",
        solution_str,
        flags=re.DOTALL | re.IGNORECASE,
    ) is not None


# ---------------------------------------------------------------------------
# LLM Judge Client
# ---------------------------------------------------------------------------

class LLMJudge:
    """Wraps the OpenAI-compatible vLLM client for judge calls."""

    def __init__(self, base_url, model_name, api_key="empty"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.system_message = VQA_JUDGE_INSTRUCTIONS + "\n" + VQA_JUDGE_IN_CONTEXT_EXAMPLES

    def attempt_api_call(self, messages, max_retries=3):
        """Call the judge LLM with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    top_p=0.9,
                    max_tokens=256,
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError) as e:
                print(f"[LLMJudge] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt + 2}/{max_retries})")
                else:
                    print(f"All {max_retries} attempts failed. Last error: {e}")
                    return None
            except Exception as e:
                print(f"[LLMJudge] Unexpected error: {e}")
                return None

    def parse_response(self, response):
        """Parse JSON verdict from judge response."""
        if response is None:
            return None, "No response from API"

        # Try to extract the JSON block
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)
        text = match.group(0) if match else response

        # Attempt 1: Parse mathematically/strictly
        import ast
        try:
            # Often LLMs output keys without quotes or with single quotes
            # replace true/false/null temporarily to avoid eval errors if needed
            # but usually it's just verdict and explanation
            # json.loads is safer, but ast.literal_eval handles single quotes
            import json
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # If json fails, try ast.literal_eval
                try:
                    data = ast.literal_eval(text)
                except Exception:
                    data = None

            if isinstance(data, dict):
                verdict = str(data.get("verdict", "")).upper().strip()
                explanation = str(data.get("explanation", ""))
                if verdict in ("CORRECT", "INCORRECT"):
                    return verdict, explanation
        except Exception:
            pass

        # Attempt 2: Flexible regex parsing
        try:
            verdict_pattern = r'["\']?verdict["\']?\s*[:=]\s*["\']?([^"\'\n\}\,]+)["\']?'
            verdict_match = re.search(verdict_pattern, text, flags=re.IGNORECASE)
            
            explanation_pattern = r'["\']?explanation["\']?\s*[:=]\s*["\'](.*?)["\']?(?:\n|\}|$)'
            explanation_match = re.search(explanation_pattern, text, flags=re.IGNORECASE | re.DOTALL)
            
            if verdict_match:
                verdict = verdict_match.group(1).upper().strip()
                # Remove punctuation
                verdict = re.sub(r'[^A-Z]', '', verdict)
                explanation = explanation_match.group(1).strip() if explanation_match else "Extracted via regex, no explanation found."
                
                if verdict in ("CORRECT", "INCORRECT"):
                    return verdict, explanation
        except Exception:
            pass

        # If we reach here, parsing failed. We return 'None' but include the raw response
        # so it gets logged instead of "Judge parse failure"
        print(f"[LLMJudge] Parsing error or invalid response format. Raw response: {response}")
        return None, f"Parse failure. Raw output: {response}"

    def judge(self, question, ground_truth, predicted_answer):
        """
        Call the LLM judge to evaluate semantic equivalence.

        Returns:
            (verdict, explanation) — verdict is "CORRECT" / "INCORRECT" / None
        """
        user_content = (
            f'Question: "{question}"\n'
            f'Ground truth: "{ground_truth}"\n'
            f'Predicted answer: "{predicted_answer}"\n'
        )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user",   "content": user_content},
        ]

        llm_response = self.attempt_api_call(messages)
        if llm_response is None:
            return None, "API call failed"

        return self.parse_response(llm_response)


# ---------------------------------------------------------------------------
# Scoring Logic
# ---------------------------------------------------------------------------

def score_prediction(prediction, ground_truth_str, question, judge):
    """
    Score a single prediction using the reward structure.

    Returns:
        (score, result_type, verdict, explanation, predicted_answer, reasoning)
    """
    # 1. Check format — extract answer from <answer>...</answer>
    predicted_answer = extract_answer_text(prediction)
    if predicted_answer is None:
        return -1.0, "no_format", None, "Missing <answer></answer> format", None, None

    # Extract reasoning from <think>...</think>
    reasoning = extract_reasoning_text(prediction, predicted_answer=predicted_answer)

    # 2. Check for abstention ("I don't know")
    pred_normalized = normalize_answer(predicted_answer)
    unknown_triggers = [
        "i don't know", "i dont know", "unsure", "i do not know",
        "not sure", "cannot determine", "can't determine",
        "unable to determine", "cannot tell", "can't tell"
    ]
    if any(trigger in pred_normalized for trigger in unknown_triggers):
        return 0.0, "abstention", "ABSTENTION", "Model chose to abstain", predicted_answer, reasoning

    # 3. Call LLM Judge
    verdict, explanation = judge.judge(question, ground_truth_str, predicted_answer)

    if verdict == "CORRECT":
        return 1.0, "correct", verdict, explanation, predicted_answer, reasoning
    elif verdict == "INCORRECT":
        return -1.0, "incorrect", verdict, explanation, predicted_answer, reasoning
    else:
        # Judge failure — mark as unknown
        return -1.0, "judge_failed", None, explanation or "Judge parse failure", predicted_answer, reasoning


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL on VSR open-text task with LLM-as-Judge (Perception-R1 format)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint or HuggingFace model"
    )
    parser.add_argument(
        "--processor_path", type=str, default=None,
        help="Path to processor/tokenizer (defaults to model_path). "
             "Use base model path when evaluating VeRL checkpoints."
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the open-text parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/vsr_llm_verifier_eval",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--name", type=str, default="eval_run",
        help="Name of the evaluation run"
    )
    parser.add_argument(
        "--no_timestamp", action="store_true",
        help="Disable timestamp in output directory name"
    )
    # LLM Judge arguments
    parser.add_argument(
        "--judge_api_base", type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL for the judge"
    )
    parser.add_argument(
        "--judge_model", type=str,
        default="/home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct",
        help="Model name for the LLM judge"
    )
    parser.add_argument(
        "--judge_api_key", type=str, default="empty",
        help="API key for the judge (vLLM typically doesn't need one)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    processor_path = args.processor_path if args.processor_path else args.model_path

    # ── Setup output directory ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        run_dir = os.path.join(args.output_dir, args.name)
    else:
        run_dir = os.path.join(args.output_dir, f"{args.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Starting VSR Open-Text Evaluation (LLM Verifier — Perception-R1 format)")
    print(f"  Model:      {args.model_path}")
    print(f"  Data:       {args.data_path}")
    print(f"  Judge:      {args.judge_model}")
    print(f"  Judge API:  {args.judge_api_base}")
    print(f"  Output:     {run_dir}")
    print(f"  Format:     <think>...</think> + <answer>...</answer>")
    print("=" * 60)

    # ── Initialize LLM Judge ──────────────────────────────────────────────
    judge = LLMJudge(
        base_url=args.judge_api_base,
        model_name=args.judge_model,
        api_key=args.judge_api_key,
    )

    # ── Load Model & Processor ────────────────────────────────────────────
    print("Loading model ...")
    try:
        processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ── Load Data ─────────────────────────────────────────────────────────
    print("Loading evaluation data ...")
    try:
        df = pd.read_parquet(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    print(f"  Total samples: {len(df)}")

    # ── Metrics Counters ──────────────────────────────────────────────────
    n_correct = 0
    n_incorrect = 0
    n_abstention = 0
    n_no_format = 0
    n_judge_failed = 0
    n_total = 0
    results = []

    # ── Inference + Evaluation Loop ──────────────────────────────────────
    print("Starting inference + evaluation ...")

    detail_log_path = os.path.join(run_dir, "evaluation_details.jsonl")

    with open(detail_log_path, "w") as detail_file:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # ── 1. Prepare inputs ─────────────────────────────────
                raw_prompt = row['prompt']
                if hasattr(raw_prompt, 'tolist'):
                    raw_prompt = raw_prompt.tolist()
                messages = copy.deepcopy(list(raw_prompt))

                # Extract image path
                image_path = "unknown"
                if 'images' in row and len(row['images']) > 0:
                    first_img = row['images'][0]
                    if isinstance(first_img, dict) and 'image' in first_img:
                        image_path = first_img['image']
                    elif hasattr(first_img, 'get'):
                        image_path = first_img.get('image', first_img)
                    elif isinstance(first_img, str):
                        image_path = first_img

                # Convert user message to Qwen2.5-VL multimodal format
                if image_path != "unknown":
                    for msg in messages:
                        if msg['role'] == 'user':
                            orig_content = msg['content']
                            clean_text = orig_content.replace("<image>\n", "").replace("<image>", "").strip()
                            msg['content'] = [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": clean_text}
                            ]

                # Extract ground truth
                ground_truth_str = "unknown"
                if 'reward_model' in row:
                    rm_data = row['reward_model']
                    if isinstance(rm_data, dict):
                        ground_truth_str = str(rm_data.get('ground_truth', 'unknown'))
                    elif isinstance(rm_data, str):
                        try:
                            rm_dict = ast.literal_eval(rm_data)
                            ground_truth_str = str(rm_dict.get('ground_truth', 'unknown'))
                        except Exception:
                            pass

                # Extract question from the user message
                question = "unknown"
                for msg in row['prompt']:
                    if msg['role'] == 'user':
                        q_text = msg['content']
                        q_text = q_text.replace("<image>\n", "").replace("<image>", "").strip()
                        question = q_text

                # ── 2. Preprocess for Qwen ────────────────────────────
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda:0")

                # ── 3. Generate ───────────────────────────────────────
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                prediction = output_text.strip()

                # ── 4. Score using LLM Judge ──────────────────────────
                score, result_type, verdict, explanation, predicted_answer, reasoning = score_prediction(
                    prediction, ground_truth_str, question, judge
                )

                # ── 5. Update Metrics ─────────────────────────────────
                n_total += 1
                if result_type == "correct":
                    n_correct += 1
                elif result_type == "abstention":
                    n_abstention += 1
                elif result_type == "no_format":
                    n_no_format += 1
                elif result_type == "judge_failed":
                    n_judge_failed += 1
                else:  # incorrect
                    n_incorrect += 1

                # ── 6. Log ────────────────────────────────────────────
                result_entry = {
                    "index": index,
                    "question": question,
                    "image_location": str(image_path),
                    "ground_truth": ground_truth_str,
                    "model_raw_output": prediction,
                    "extracted_answer": predicted_answer,
                    "extracted_reasoning": reasoning[:500] if reasoning else None,
                    "verdict": result_type,
                    "judge_verdict": verdict,
                    "judge_explanation": explanation,
                    "score": score,
                }
                results.append(result_entry)
                detail_file.write(json.dumps(result_entry) + "\n")
                detail_file.flush()

            except Exception as e:
                print(f"Error processing item {index}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ── Final Metrics & Summary ───────────────────────────────────────────
    accuracy = n_correct / n_total if n_total > 0 else 0
    abstention_rate = n_abstention / n_total if n_total > 0 else 0
    hallucination_rate = n_incorrect / n_total if n_total > 0 else 0
    no_format_rate = n_no_format / n_total if n_total > 0 else 0
    judge_fail_rate = n_judge_failed / n_total if n_total > 0 else 0

    # Truthfulness Score: (correct - incorrect - no_format) / total
    truthfulness_score = (n_correct - n_incorrect - n_no_format) / n_total if n_total > 0 else 0

    summary = {
        "model_path": args.model_path,
        "judge_model": args.judge_model,
        "data_path": args.data_path,
        "response_format": "perception_r1 (<think> + <answer>)",
        "n_samples": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_abstention": n_abstention,
        "n_no_format": n_no_format,
        "n_judge_failed": n_judge_failed,
        "accuracy": accuracy,
        "abstention_rate": abstention_rate,
        "hallucination_rate": hallucination_rate,
        "no_format_rate": no_format_rate,
        "judge_fail_rate": judge_fail_rate,
        "truthfulness_score": truthfulness_score,
        "timestamp": timestamp,
    }

    summary_path = os.path.join(run_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print(f"  Format:                <think>...</think> + <answer>...</answer>")
    print(f"  Accuracy:              {accuracy:.2%} ({n_correct}/{n_total})")
    print(f"  Truthfulness Score:    {truthfulness_score:.4f}")
    print(f"  Abstention Rate:       {abstention_rate:.2%} ({n_abstention}/{n_total})")
    print(f"  Hallucination Rate:    {hallucination_rate:.2%} ({n_incorrect}/{n_total})")
    print(f"  No Format Rate:        {no_format_rate:.2%} ({n_no_format}/{n_total})")
    print(f"  Judge Failure Rate:    {judge_fail_rate:.2%} ({n_judge_failed}/{n_total})")
    print(f"  Results saved to:      {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
