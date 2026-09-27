#!/bin/bash
# Tai Vicuna-7B-v1.3 fp16 (full precision .bin) tu HuggingFace ve /workspace/models.
# Tai truc tiep tung file (khong qua HF cache) de tiet kiem dung luong dia.
set -e
D=/workspace/models/vicuna-7b-v1.3
mkdir -p "$D"; cd "$D"
BASE=https://huggingface.co/lmsys/vicuna-7b-v1.3/resolve/main
for f in config.json generation_config.json pytorch_model.bin.index.json special_tokens_map.json tokenizer.model tokenizer_config.json pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin; do
  echo "[dl] $f"
  wget -q --show-progress -c -O "$f" "$BASE/$f"
done
echo "[dl] DONE"; ls -la "$D"
