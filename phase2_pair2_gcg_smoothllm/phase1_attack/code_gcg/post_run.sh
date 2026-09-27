#!/bin/bash
# Chay SAU khi GCG xong (GPU ranh): build input SmoothLLM + verify jailbreak that.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gcg
echo '[post] build smoothllm inputs (CPU)'
python "$HERE/build_smoothllm_inputs.py"
echo '[post] verify jailbreak that (GPU, 25 cau held-out)'
python "$HERE/verify_jailbreak.py" 25
echo '[post] DONE'
