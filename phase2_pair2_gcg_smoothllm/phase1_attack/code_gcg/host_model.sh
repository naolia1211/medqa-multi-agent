#!/bin/bash
# Host Vicuna-7B-v1.3 fp16 qua OpenAI-compatible API (FastChat), cong 8080.
# Host DUNG ban fp16 da bi tan cong -> khop suffix (khac Ollama/GGUF quantized).
# Neu tai dung may Vast: `vastai start instance 50898440` roi SSH vao truoc.
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate gcg

python -m fastchat.serve.controller --host 127.0.0.1 --port 21001 \
  > /workspace/serve_controller.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.model_worker \
  --model-path /workspace/models/vicuna-7b-v1.3 \
  --model-names vicuna-7b-v1.3 \
  --controller-address http://127.0.0.1:21001 \
  --host 127.0.0.1 --port 21002 \
  > /workspace/serve_worker.log 2>&1 &

python -m fastchat.serve.openai_api_server \
  --controller-address http://127.0.0.1:21001 \
  --host 0.0.0.0 --port 8080 \
  > /workspace/serve_api.log 2>&1 &

echo "FastChat API: http://localhost:8080/v1  (model: vicuna-7b-v1.3)"
echo "Test: curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\":\"vicuna-7b-v1.3\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
