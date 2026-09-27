# Source code phase 1 — GCG universal trên Vicuna-7B-v1.3

Toàn bộ code đã chạy trên GPU thuê (Vast.ai instance 50898440, 1× RTX 3090 24GB) để sinh suffix
universal GCG và pack input cho SmoothLLM. Đây là **code phase 1** — phase 2 không chạy lại GCG
(tốn ~21 giờ GPU) mà chỉ kế thừa sản phẩm của nó; xem `../../code/` cho mã phase 2.

Tất cả path trong script là path **trên máy remote** (`/workspace/...`, `/opt/miniforge3/...`),
giữ nguyên như lần chạy thật để tái lập được. Cần re-path nếu chạy trên máy khác.

## Thứ tự chạy
| Bước | File | Mô tả | Nơi chạy |
|---|---|---|---|
| 1 | `setup_env.sh` | Tạo conda env `gcg` (py3.10) + torch cu121 + transformers 4.28.1 + fschat 0.2.20 … | CPU |
| 2 | `download_model.sh` | Tải Vicuna-7B-v1.3 fp16 từ HuggingFace về `/workspace/models` | mạng |
| 3 | `clone_and_patch.sh` | Clone llm-attacks, `pip install -e`, đặt `transfer_vicuna_single.py`, chạy `patch_attack_manager.py` | CPU |
| 4 | `run_gcg.sh` | Chạy GCG universal 500 bước (nền: `nohup bash run_gcg.sh &`) | GPU (~21h) |
| 5 | `post_run.sh` | Sau khi GCG xong: build input SmoothLLM + verify jailbreak thật | CPU + GPU |
| — | `host_model.sh` | (Tuỳ chọn) host Vicuna fp16 qua FastChat OpenAI API cổng 8080 cho eval C1/C2 | GPU |

## Các file
- **`transfer_vicuna_single.py`** — config GCG (llm-attacks): `transfer=True`, `progressive_goals=True`, 1 model Vicuna, template `vicuna`, `cuda:0`. Đặt vào `llm-attacks/experiments/configs/`.
- **`patch_attack_manager.py`** — patch lớp `ModelWorker`: chạy đồng bộ 1 tiến trình (bỏ `spawn` nhân đôi model) + `requires_grad_(False)`. **Bắt buộc** để 7B fp16 vừa 24GB (nếu không sẽ OOM).
- **`build_smoothllm_inputs.py`** — đọc logfile GCG + advbench → xuất `universal_suffix.json`, `attack_prompts.jsonl/.csv` (100 câu held-out row 25:125), `refusal_prefixes.json`, `smoothllm_config.json` vào `/workspace/results/smoothllm`.
- **`verify_jailbreak.py [K]`** — sinh câu trả lời thật (goal+suffix vs goal-only) cho K câu → `jailbreak_verification.jsonl` + `jailbreak_summary.json` (ASR).
- **`gcg_universal_vicuna.ipynb`** — notebook mô tả toàn bộ pipeline đã chạy (mirror các script trên).

## Tham số lần chạy thật

Đọc từ `../vicuna-7b-v1.3/transfer_vicuna_universal_20260913-163343.json` (khoá `params`).

| Tham số | Giá trị |
|---|---|
| attack | GCG universal (llm-attacks, Zou et al. 2023) |
| model | `lmsys/vicuna-7b-v1.3`, fp16 HF (KHÔNG quantized — GCG cần gradient) |
| conv template | `vicuna` (fastchat → `vicuna_v1.1`, roles USER/ASSISTANT) |
| n_steps / test_steps | 500 / 50 |
| batch_size / topk | 256 / 256 |
| control_init | 20 token `!` |
| n_train_data / n_test_data / data_offset | 25 / 25 / 0 |
| progressive_goals / stop_on_success / allow_non_ascii | True / False / False |
| target_weight / control_weight | 1.0 / 0.0 |
| GPU / thời gian | 1× RTX 3090 24GB / ~21h |

Kết quả: 500/500 bước, loss cuối **0,263** (thấp nhất 0,252), verify ASR có suffix =
**1,00 (25/25)**, baseline sạch = **0,08 (2/25)** — xem `../smoothllm/jailbreak_summary.json`.

## Output đã kéo về repo

Output trên remote (`/workspace/results/...`) đã được đưa vào repo như sau:

| Trên remote | Trong repo |
|---|---|
| `/workspace/results/transfer_vicuna_universal*.json` | `../vicuna-7b-v1.3/transfer_vicuna_universal_20260913-163343.json` |
| `/workspace/results/smoothllm/universal_suffix.json` | `../vicuna-7b-v1.3/universal_suffix.json`, `../smoothllm/universal_suffix.json` |
| `/workspace/results/smoothllm/attack_prompts.jsonl` + `.csv` | `../smoothllm/` |
| `/workspace/results/smoothllm/refusal_prefixes.json` | `../smoothllm/refusal_prefixes.json` |
| `/workspace/results/smoothllm/smoothllm_config.json` | `../smoothllm/smoothllm_config.json` |
| `/workspace/results/smoothllm/jailbreak_verification.jsonl` + `jailbreak_summary.json` | `../smoothllm/` |

Bản copy mà **code phase 2 đọc trực tiếp** nằm ở `../../data/phase1_gcg/` (trùng byte với
`../smoothllm/`, chỉ đổi tên `gcg_run_log.json`). Đừng di chuyển thư mục đó —
`../../code/src/datasets_build.py` và `../../code/src/judges.py` phụ thuộc vào nó.

## Lưu ý
- Suffix tối ưu trên **fp16 HF**. Eval C1/C2 nên chạy trên đúng bản fp16 (dùng `host_model.sh`); nếu chạy Ollama/GGUF quantized phải đo lại ASR. Đây chính là lý do arm B (gemma, quantized) có ASR 0% — suffix không chuyển giao.
- `"Passed"` trong log GCG chỉ là heuristic 16-token — dùng `../smoothllm/jailbreak_verification.jsonl` để xác nhận jailbreak thật.
- MedQA / C0 / C3 là benign, không gắn suffix.
