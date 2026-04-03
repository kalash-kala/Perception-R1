#!/bin/bash
# =========================================================
#   Unanswerable-Only VQA Pipeline (2-Step)
#
#   Generates unanswerable question pairs from a fresh
#   VQAv2 Parquet file. Two steps:
#     1. Build visual cues in memory (no images saved)
#     2. Perturb + tag + save ONLY confirmed unanswerable
#        (both original & perturbed images + 2 JSONL records)
# =========================================================
set -e

echo "=========================================================="
echo "    Unanswerable-Only VQA Pipeline (2-Step)"
echo "=========================================================="

# ---------------------------------------------------------------------------
# CONFIGURATION – update these for your setup
# ---------------------------------------------------------------------------
BASE_DIR="/home/debarpanb1/kalashkala"
SCRIPTS_DIR="${BASE_DIR}/Perception-R1/scripts/unanswerable"
DATA_DIR="${BASE_DIR}/visual-question-answering"

# ── INPUT: your new Parquet file with fresh data ──
INPUT_PARQUET="${DATA_DIR}/new_unanswerable_source.parquet"

# ── IMAGE DIRECTORY (only UNANSWERABLE images land here) ──
IMAGE_DIR="${DATA_DIR}/unanswerable_images"

# ── INTERMEDIATE + FINAL FILES ──
PIPELINE_DIR="${DATA_DIR}/unanswerable_pipeline"
STEP1_JSONL="${PIPELINE_DIR}/step1_clean_cues.jsonl"
FINAL_JSONL="${PIPELINE_DIR}/unanswerable_final.jsonl"

# ── TARGET: how many unanswerable pairs do you want? (0 = all) ──
TARGET_COUNT=700

# ── API RATE LIMITS ──
SLEEP_STEP1=1.0
SLEEP_STEP2=2.0

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

mkdir -p ${PIPELINE_DIR}

echo ""
echo "Step 1: Building visual cues from Parquet (no images saved) ..."
echo "  Input:  ${INPUT_PARQUET}"
echo "  Output: ${STEP1_JSONL}"
python ${SCRIPTS_DIR}/step1_build_visual_cues.py \
    --input_parquet  ${INPUT_PARQUET} \
    --output_jsonl   ${STEP1_JSONL} \
    --sleep_interval ${SLEEP_STEP1}

echo ""
echo "Step 2: Perturb + tag + save unanswerable pairs ..."
echo "  Input JSONL:   ${STEP1_JSONL}"
echo "  Input Parquet:  ${INPUT_PARQUET}"
echo "  Output JSONL:   ${FINAL_JSONL}"
echo "  Images dir:     ${IMAGE_DIR}"
echo "  Target count:   ${TARGET_COUNT}"
python ${SCRIPTS_DIR}/step2_perturb_and_tag.py \
    --input_jsonl    ${STEP1_JSONL} \
    --input_parquet  ${INPUT_PARQUET} \
    --output_jsonl   ${FINAL_JSONL} \
    --image_dir      ${IMAGE_DIR} \
    --target_count   ${TARGET_COUNT} \
    --sleep_interval ${SLEEP_STEP2}

echo ""
echo "=========================================================="
echo "  Pipeline complete!"
echo "  Final dataset:  ${FINAL_JSONL}"
echo "  Images:         ${IMAGE_DIR}"
echo ""
echo "  Each unanswerable pair = 2 JSONL records:"
echo "    1. Original (ANSWERABLE)    → image_path points to clean image"
echo "    2. Perturbed (UNANSWERABLE) → image_path points to degraded image"
echo "=========================================================="
echo ""
echo "Next: merge with your existing dataset and convert to VERL:"
echo "  cat existing_train.jsonl ${FINAL_JSONL} > merged.jsonl"
echo "  python scripts/convert_perturbed_for_verl.py --input_jsonl merged.jsonl ..."
echo "=========================================================="
