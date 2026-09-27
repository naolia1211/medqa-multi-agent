#!/bin/bash
# Clone llm-attacks, cai editable, dat config GCG, va patch ModelWorker.
# Chay sau setup_env.sh. Cac file .py di kem phai nam cung thu muc script nay.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gcg

# clone (bo smudge LFS vi model tai rieng bang download_model.sh)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/llm-attacks/llm-attacks.git /workspace/llm-attacks
pip install -e /workspace/llm-attacks --no-deps
mkdir -p /workspace/results

# dat config GCG (universal, 1 model Vicuna)
cp "$HERE/transfer_vicuna_single.py" /workspace/llm-attacks/experiments/configs/transfer_vicuna_single.py

# patch ModelWorker: 1 ban model + requires_grad_(False)
python "$HERE/patch_attack_manager.py"

echo "clone + install + config + patch DONE"
