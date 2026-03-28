#!/bin/bash
# ============================================================================
# VQA + Qwen2.5-VL-3B FULL PARAMETER WITHOUT LoRA — 2× H200 Server (KIAC)
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=vqa_qwen2_5_vl_3b_full_bsz8_gs4_lr1e6_epochs3_h200_ternary_perturbed
#SBATCH --output=/home/sriramg/kalashabhayk/Perception-R1/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/Perception-R1/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/Perception-R1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "/home/sriramg/kalashabhayk/Perception-R1"

# Ensure log directories exist
mkdir -p slurm_logs/logs slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Clean up old Ray sessions
ray stop
rm -rf /tmp/ray/*

# Kill any existing vLLM server launched by us just in case
pkill -u $USER -f "vllm.entrypoints.openai.api_server" || true

# Wait for old processes to fully release GPU memory
sleep 10

# Pick a random port for the judge to avoid conflicts on shared nodes
export JUDGE_MODEL="/home/sriramg/kalashabhayk/models/gemma-3-27b-it"
export JUDGE_PORT=$(( 8000 + RANDOM % 1000 ))
export OPENAI_API_BASE="http://localhost:${JUDGE_PORT}/v1"

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

# Network config
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
export PYTHONUNBUFFERED=1

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.95

# 2× H200 GPUs
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# START LOCAL vLLM JUDGE SERVER (BACKGROUND)
# ============================================================================
echo "Starting local vLLM judge server on GPUs 0,1 in the background on port $JUDGE_PORT..."
# We use tensor-parallel-size 2 and low gpu-memory-utilization to leave room for VERL
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.24 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --language-model-only \
    --enforce-eager \
    --dtype bfloat16 \
    --port "${JUDGE_PORT}" \
    > "vllm_judge_server_gemma3_27b_textonly_${JUDGE_PORT}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID. Waiting for judge server to be ready..."

WAITED=0
INTERVAL=10
while true; do
    if curl -s http://localhost:${JUDGE_PORT}/v1/models > /dev/null 2>&1; then
        echo "Judge server is ready! (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Error: vLLM judge server process died. Check vllm_judge_server_vqa_gemma-3-27b-it_${JUDGE_PORT}.log"
        exit 1
    fi
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    echo "Still waiting for judge server... (${WAITED}s elapsed)"
    
done
echo "Proceeding with training..."

# Ensure we cleanup vLLM process on script exit
trap "echo 'Cleaning up vLLM server (PID $VLLM_PID)...'; kill $VLLM_PID; exit" INT TERM EXIT

# ============================================================================
# PATHS
# ============================================================================
# Local paths for KIAC
DATA_DIR=/home/sriramg/kalashabhayk/visual-question-answering/processed_for_verl
MODEL_PATH=/home/sriramg/kalashabhayk/models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/home/sriramg/kalashabhayk/Perception-R1/verl/utils/reward_score/vqa_llm_judge_ternary.py

# ============================================================================
# Hyperparameters
# ============================================================================
# Learning Rate: Lower LR for full fine-tuning (all weights are updated)
LR=1e-6

GROUP_SIZE=4
ROLLOUT_TP_SIZE=1
EPOCHS=3

micro_batch_size_per_device_for_update=8
micro_batch_size_per_device_for_experience=4
gradient_accumulation_steps=2
nnodes=1
n_gpus_per_node=2

global_batch_size=$((${n_gpus_per_node} * ${nnodes} * ${micro_batch_size_per_device_for_update} * ${gradient_accumulation_steps}))
rollout_batch_size=$((${global_batch_size} * 1))

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags.
 The final answer MUST BE put in \boxed{} in <answer> </answer> tags."""

# ============================================================================
# Launch Training
# ============================================================================
python3 -m verl.trainer.main \
    config=examples/default_config.yaml \
    data.train_files=$DATA_DIR/train_perturbed_vqa.parquet \
    data.val_files=$DATA_DIR/val_perturbed_vqa.parquet \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    data.rollout_batch_size=${rollout_batch_size} \
    data.min_pixels=3136 \
    data.max_pixels=401408 \
    algorithm.use_kl_loss=false \
    algorithm.disable_kl=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${global_batch_size} \
    worker.actor.micro_batch_size_per_device_for_update=${micro_batch_size_per_device_for_update} \
    worker.actor.micro_batch_size_per_device_for_experience=${micro_batch_size_per_device_for_experience} \
    worker.actor.optim.lr=${LR} \
    worker.actor.offload.offload_params=false \
    worker.ref.offload.offload_params=false \
    worker.rollout.n=${GROUP_SIZE} \
    worker.rollout.enforce_eager=true \
    worker.rollout.gpu_memory_utilization=0.20 \
    +worker.rollout.engine_kwargs.vllm.max_model_len=2048 \
    +worker.rollout.engine_kwargs.vllm.max_num_seqs=8 \
    worker.rollout.tensor_parallel_size=${ROLLOUT_TP_SIZE} \
    worker.reward.score_function=$REWARD_FN_PATH \
    trainer.experiment_name="vqa_qwen2_5_vl_3b_2gpu_h200_full_bsz8_lr1e6_gs4_epoch3_ternary_perturbed" \
    trainer.logger="['console','tensorboard']" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_episodes=${EPOCHS} \
    trainer.val_freq=50 \
    trainer.save_freq=375 \
    trainer.nnodes=${nnodes} "$@"

echo "Training complete."
