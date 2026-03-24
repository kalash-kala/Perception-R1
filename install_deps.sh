#!/bin/bash
# Configuration
REQUIREMENTS_FILE="requirements.txt"

echo "Step 1: Installing torch first (required by flash-attn)..."
pip install torch==2.6.0 --no-cache-dir

echo "Step 2: Installing FlashAttention from pre-built wheel..."
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" --no-cache-dir

echo "Step 3: Installing remaining dependencies from ${REQUIREMENTS_FILE}..."
# Now we can safely use --no-build-isolation because torch is present
pip install --no-cache-dir --no-build-isolation -r "${REQUIREMENTS_FILE}"

echo "Installation complete."
