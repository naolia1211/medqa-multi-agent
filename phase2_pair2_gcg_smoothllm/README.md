# Pair 2 — Tấn công GCG và phòng thủ SmoothLLM

Đồ án cuối kỳ môn An toàn hệ thống LLM. Đối tượng đánh giá là hệ thống
Medical Multi-Agent QA (MedQA-USMLE) của bài giữa kỳ.

Toàn bộ số liệu là kết quả đo thực tế, thực hiện 17–19/09/2026. Mọi giá trị
chưa đo được nêu rõ là chưa đo, không ước lượng.

## Nội dung thư mục

> Thư mục đã được phân loại lại 21/09/2026 — xem `MUCLUC.md` để biết sơ đồ đầy đủ.

| Đường dẫn | Nội dung |
|---|---|
| `de_bai/Final_Project_Attack_Defense.pdf` | Đề bài tổng (Attack & Defend) — không phải báo cáo |
| `code/` | Toàn bộ mã nguồn phase 2 |
| `data/eval_sets/` | Ba tập đánh giá đã dựng sẵn |
| `data/phase1_gcg/` | Sản phẩm phase 1 kế thừa lại (code đọc trực tiếp) |
| `phase1_attack/` | Gói bàn giao phase-1 (log GCG + suffix + input SmoothLLM); = nội dung `Pair2_ATTACK_results.zip` |
| `ket_qua/` | Bản ghi thô, bảng chỉ số, và `ghi_chu_phan_tich.md` (ghi chú phân tích gốc) |
| `demo/replay_console.html` | Demo offline, phát lại 4 ca đã đo (mở bằng trình duyệt) |
| `_giua_ky/` | Tài liệu giữa kỳ — KHÔNG thuộc Pair 2, để cách ly |
| `../_archive_zip/` | Hai zip gốc lưu trữ |

Kiểm thử: 13 test function gồm **96 phép kiểm tra có tên**, tất cả đạt
(38 smoke + 31 integration + 27 mô phỏng lỗi HTTP). Chạy bằng `pytest`, không
cần GPU và không cần model thật.

## Hai file phase 1 được sử dụng

Phase 2 **không chạy lại GCG** (tốn ~21 giờ GPU), mà kế thừa đúng hai file
trong `data/phase1_gcg/`:

- **`universal_suffix.json`** — chuỗi tấn công 20 token đã tối ưu xong.
  `src/datasets_build.py` đọc file này để ghép suffix vào 100 goal độc hại,
  tạo ra nhánh "đã bị tấn công" của tập safety. Đây là input của mọi thí nghiệm.
- **`refusal_prefixes.json`** — danh sách tiền tố câu từ chối, dùng làm thước đo
  trong `src/judges.py`. Phải dùng đúng file của phase 1, nếu không con số của
  hai phase không so sánh được với nhau.

Hai file còn lại (`gcg_run_log.json`, `jailbreak_verification.jsonl`) là bằng
chứng trích vào báo cáo: 500 bước, loss cuối 0,263, và 25 goal kiểm chứng cho
ASR 1,00 khi có suffix so với 0,08 khi không có.

## Kết quả chính

| | Arm A — Vicuna-7B-v1.3 | Arm B — gemma4:12b-it-qat |
|---|---|---|
| Jailbreak ASR, không phòng thủ | 97% | 0% |
| Jailbreak ASR, có SmoothLLM | 22% | 0% |
| Residual ASR (tuyệt đối) | **22%** | 0% |
| False-refusal benign, C0 → C3 | 6,67% → 12,0% | 0,0% → 0,0% |
| Chi phí | 10× lượt gọi | 4× lượt gọi |

Giá trị của phòng thủ dịch chuyển hoàn toàn giữa hai cột; chi phí thì không.

## Tái lập

```bash
cd code
pip install -r requirements.txt

python src/datasets_build.py          # dựng 3 tập đánh giá
python -m pytest tests/               # 13 test function / 96 phép kiểm tra, không cần GPU

# Arm A (cần GPU ~16GB, tự cài vLLM và phục vụ Vicuna-7B)
python arm_a_run.py --skip-sweep

# Arm B (cần Ollama chạy sẵn với gemma4:12b-it-qat)
python step1_check_transfer.py
python run_medqa_gemma.py

# Bảng chỉ số — LUÔN chạy rescore trước khi trích bất kỳ con số ASR nào
python make_report.py --logs "results/*.jsonl" --out reports/metrics
python rescore.py    --logs "results/*.jsonl" --out results/rescored.jsonl

# Demo 3 act dùng khi thuyết trình
python demo.py --backend vicuna --index 23
```

## Bản ghi thô

Bản ghi từng dòng của các lần chạy trên Kaggle đã tải về và nằm trong `ket_qua/`:

| File | Nội dung | Chứng minh |
|---|---|---|
| `armA_eval_raw.jsonl` | 700 bản ghi arm A, đủ C0–C3 safety + benign | ASR 97% → 22%, n=100/100/50 |
| `armA_medqa_raw.jsonl` | 400 bản ghi MedQA arm A | phép đo không hợp lệ: C3 đoán "A" ở 195/200 câu |
| `sweep_q10_N10.jsonl` | ô sweep N=10, q=10% | 6 ca hòa 5–5, phân giải 3–3 |
| `sweep_q10_N9.jsonl` | ô sweep N=9, q=10% | 0 ca hòa, margin nhỏ nhất 0,556 |
| `sweep_summary_goc.csv` | bảng sweep gốc do `sweep.py` sinh ra | 10 cấu hình q/N |
| `demo_transcript.txt` | transcript demo 4 ca trên Vicuna | act 1/2/3 và phiếu từng bản sao |

Các bảng `.csv` khác trong `ket_qua/` được trích từ chính những file này.

Còn lại trên Kaggle (notebook `st1rr1ng/arm-a-vicuna`) nếu cần đối chiếu thêm:
8 ô sweep khác, `rescored.jsonl`, và log đầy đủ của từng version.

## Ba lỗi trong harness đã phát hiện và sửa

1. `make_report.py` ghi file gộp `_merged.jsonl` vào ngay `results/`, khiến bước
   rescore đọc lại nó và **nhân đôi cột n**. Mọi tỉ lệ vẫn đúng vì bản sao là
   chính xác — chỉ cột `n` sai. Đã chuyển file gộp ra ngoài vùng glob.
2. `rescore.py` không khử trùng lặp. Nay khử theo khóa
   `(prompt_id, condition, suffix_id, backend)`.
3. `arm_a_run.py` thiếu khai báo cờ `--skip-sweep`.

Nếu đọc lại log cũ trên Kaggle, cột `n` ở bảng rescore bị nhân đôi — số đúng là
100 mỗi điều kiện safety, 100 Alpaca, 50 XSTest.

---

## Cập nhật cuối kỳ (bản nộp) — ánh xạ deliverable ↔ rubric

| Deliverable (đề bài §4) | File trong repo |
|---|---|
| **Code + README (35%)** — attack/defense/harness | `code/` (`src/smoothllm.py`, `src/judges.py`, `arm_a_run.py`, `run_medqa_gemma.py`, `step1_check_transfer.py`), `code/requirements.txt`, `code/tests/` |
| **Báo cáo** | *nộp riêng — không nằm trong repo* |
| **Live demo** | `demo/replay_console.html` (offline, phát lại 4 ca) |
| **Slide + thuyết trình** | *nộp riêng — không nằm trong repo* |
| **Bằng chứng thô** | `ket_qua/` + `ket_qua/armA_supplementary_20260921/` (đo 21/09: MedQA constrained + 2 semantic attack) |
| **Đề bài** | `de_bai/Final_Project_Attack_Defense.pdf` |

### Bốn chỉ số đề bài yêu cầu (§3) — đã báo cáo đủ
- **Attack metric (ASR):** GCG 97% (Vicuna) / 0% (gemma, không chuyển giao); semantic 96.7–100%.
- **Utility metric:** MedQA accuracy C0→C3 (arm A 33→30, arm B 69→63) + false-refusal benign.
- **Cost axes:** 10× truy vấn, ~6.6× token, 2.6× latency; false-refusal +5.33pp (XSTest +12pp).
- **Baseline:** ASR trên hệ chưa phòng thủ (C1) đối chiếu C0.

### "Cái gì vẫn lọt" (nguyên tắc cốt lõi của đề)
Residual ASR 22% trên GCG (18% tấn công thật + 4% model tự sẵn sàng); **residual 77–87% trên jailbreak ngữ nghĩa** — SmoothLLM mù trước tấn công không kém-bền-ký-tự. Chi tiết §7 báo cáo + `ket_qua/armA_supplementary_20260921/GHI_CHU_BO_SUNG.md`.
