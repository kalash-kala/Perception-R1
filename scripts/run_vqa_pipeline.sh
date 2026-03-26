#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "          End-to-End VQA Pipeline for VERL                "
echo "=========================================================="

# ---------------------------------------------------------------------------
# CONFIGURATION
# Simply update these paths for the new server
# ---------------------------------------------------------------------------
# Base directory for the repository
BASE_DIR="/home/debarpanb1/kalashkala"
SCRIPTS_DIR="${BASE_DIR}/Perception-R1/scripts"
DATA_DIR="${BASE_DIR}/visual-question-answering"

# Inputs
INPUT_PARQUET="${DATA_DIR}/vqa_stratified_100.parquet"
IMAGE_DIR="${DATA_DIR}/processed_for_verl/images"

# Intermediate Files
CLEAN_JSONL="${DATA_DIR}/clean_vqa_with_visual_cues.jsonl"
CLEAN_TAGGED_JSONL="${DATA_DIR}/clean_vqa_with_visual_cues_tagged.jsonl"
PERTURBED_JSONL="${DATA_DIR}/perturbed_manifest.jsonl"
PERTURBED_UPDATED_JSONL="${DATA_DIR}/perturbed_manifest_updated.jsonl"
MERGED_JSONL="${DATA_DIR}/merged_vqa_manifest.jsonl"

# Outputs
OUTPUT_DIR="${DATA_DIR}/processed_for_verl"
TRAIN_PARQUET="train_perturbed_vqa.parquet"
VAL_PARQUET="val_perturbed_vqa.parquet"

# API Limits
SLEEP_INTERVAL=1.0

# ---------------------------------------------------------------------------
# PIPELINE EXECUTION
# ---------------------------------------------------------------------------

echo "Step 1: Building initial visual cues from VQA parquet..."
python ${SCRIPTS_DIR}/build_visual_cues.py \
    --input_parquet ${INPUT_PARQUET} \
    --output_jsonl ${CLEAN_JSONL} \
    --image_dir ${IMAGE_DIR} \
    --sleep_interval ${SLEEP_INTERVAL}

echo "Step 2: Applying perturbations and evaluating Answerability with Gemini..."
python ${SCRIPTS_DIR}/build_perturbations_and_tags.py \
    --input_jsonl ${CLEAN_JSONL} \
    --output_jsonl ${PERTURBED_JSONL} \
    --sleep_interval ${SLEEP_INTERVAL}

echo "Step 3: Attaching default ANSWERABLE tags to the clean dataset..."
python ${SCRIPTS_DIR}/add_tags_to_clean_vqa.py \
    --input_jsonl ${CLEAN_JSONL} \
    --output_jsonl ${CLEAN_TAGGED_JSONL}

echo "Step 4: Updating visual cues/answers for UNANSWERABLE perturbed entries..."
python ${SCRIPTS_DIR}/update_unanswerable_cues.py \
    --input_jsonl ${PERTURBED_JSONL} \
    --output_jsonl ${PERTURBED_UPDATED_JSONL} \
    --sleep_interval ${SLEEP_INTERVAL}

echo "Step 5: Merging the clean and perturbed datasets..."
python ${SCRIPTS_DIR}/merge_vqa_datasets.py \
    --clean_jsonl ${CLEAN_TAGGED_JSONL} \
    --perturbed_jsonl ${PERTURBED_UPDATED_JSONL} \
    --output_jsonl ${MERGED_JSONL} \
    --shuffle

echo "Step 6: Converting merged dataset to VERL Parquet (Train/Val Split)..."
python ${SCRIPTS_DIR}/convert_perturbed_for_verl.py \
    --input_jsonl ${MERGED_JSONL} \
    --output_dir ${OUTPUT_DIR} \
    --train_output_name ${TRAIN_PARQUET} \
    --val_output_name ${VAL_PARQUET}

echo "=========================================================="
echo "Pipeline completed successfully! Outputs are in:"
echo "Train: ${OUTPUT_DIR}/${TRAIN_PARQUET}"
echo "Val:   ${OUTPUT_DIR}/${VAL_PARQUET}"
echo "=========================================================="
