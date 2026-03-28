#!/bin/bash
# ============================================================================
# SLURM Wrapper for Merge and Open-Text Evaluation — H200 Server (KIAC)
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=open_text_eval_perception_r1
#SBATCH --output=/home/sriramg/kalashabhayk/Perception-R1/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/Perception-R1/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/Perception-R1
#SBATCH --time=04:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#sbatch ./merge_and_eval_open_text_slurm.sh -c /home/sriramg/kalashabhayk/models/Qwen2.5-VL-3B-Instruct -d /home/sriramg/kalashabhayk/GQA/val_spatial_for_verl_new_system_prompt.parquet -n eval_gqa_qwen2_5_vl_3b_vanilla_zeroshot -s

# Ensure log directories exist
mkdir -p slurm_logs/logs slurm_logs/errors

# Setup Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Environment Variables for Performance/Stability
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
export PYTHONUNBUFFERED=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

echo "=================================================="
echo "Starting SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Arguments: $@"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Run the merge and eval script
# Pass all arguments through to the original script
chmod +x ./merge_and_eval_open_text.sh
./merge_and_eval_open_text.sh "$@"

if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Job Completed Successfully"
    echo "=================================================="
else
    echo "=================================================="
    echo "Job Failed with exit code $?"
    echo "=================================================="
    exit 1
fi
