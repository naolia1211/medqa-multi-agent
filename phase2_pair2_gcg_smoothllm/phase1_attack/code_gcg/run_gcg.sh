#!/bin/bash
# Chay GCG universal: 25 cau train (1 suffix chung, progressive) + 25 cau test held-out.
# 500 buoc, batch 256, tren 1x RTX 3090 24GB (~21 gio).
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gcg
cd /workspace/llm-attacks/experiments
export WANDB_MODE=disabled
python -u main.py \
  --config=configs/transfer_vicuna_single.py \
  --config.attack=gcg \
  --config.train_data=/workspace/llm-attacks/data/advbench/harmful_behaviors.csv \
  --config.result_prefix=/workspace/results/transfer_vicuna_universal \
  --config.progressive_goals=True \
  --config.stop_on_success=False \
  --config.num_train_models=1 \
  --config.allow_non_ascii=False \
  --config.n_train_data=25 --config.n_test_data=25 --config.data_offset=0 \
  --config.n_steps=500 --config.test_steps=50 \
  --config.batch_size=256
echo "GCG_EXIT=$?"
touch /workspace/results/GCG_DONE
