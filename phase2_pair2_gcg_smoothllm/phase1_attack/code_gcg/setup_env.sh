#!/bin/bash
# Tao conda env gcg (Python 3.10) + cai deps de chay GCG (llm-attacks) tren GPU.
# Chay tren instance Vast.ai (conda o /opt/miniforge3).
set -e
source /opt/miniforge3/etc/profile.d/conda.sh
echo "[env] creating conda env gcg (py3.10)"
conda create -y -n gcg python=3.10
conda activate gcg
python --version
echo "[env] installing torch cu121"
pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
echo "[env] installing gcg deps"
pip install --no-cache-dir "numpy<2" transformers==4.28.1 ml_collections fschat==0.2.20 accelerate sentencepiece protobuf pandas
echo "[env] verify import"
python -c "import torch,transformers,fastchat; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('transformers',transformers.__version__); print('fastchat',fastchat.__version__)"
echo "[env] DONE"
