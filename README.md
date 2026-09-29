# Offline Indexing Pipeline — MedQA-USMLE (Dense + Sparse Hybrid)
## AI Security Research Contribution
My contribution to this project focuses on the security evaluation and
adversarial robustness of the MedQA multi-agent LLM system.

### Research Scope
- Reproduced optimization-based GCG jailbreak attacks against aligned LLMs.
- Implemented and evaluated SmoothLLM as a perturbation-based jailbreak defense.
- Evaluated semantic jailbreak attacks and residual defense weaknesses.
- Built an attack-defense evaluation workflow covering jailbreak attack success
  rate (ASR), benign utility, query/latency overhead, and transferability.
- Analyzed the effectiveness and limitations of SmoothLLM across different
  adversarial attack strategies.

**Attack:** GCG (Greedy Coordinate Gradient)  
**Defense:** SmoothLLM  
**System:** MedQA Multi-Agent LLM  
**Evaluation:** ASR, benign utility, overhead, transferability, and residual failure modes
➡️ **Implementation and experiments:** [`phase2_pair2_gcg_smoothllm/`](./phase2_pair2_gcg_smoothllm/)
---
> ## 🎓 Final Project (Phase 2) — Attack & Defense
> Repo này là **hệ thống Phase 1** (Medical Multi-Agent QA). Đồ án cuối kỳ (Pair 2 — tấn công **GCG**, phòng thủ **SmoothLLM**) nằm ở **[`phase2_pair2_gcg_smoothllm/`](phase2_pair2_gcg_smoothllm/)** — code tấn công/phòng thủ + harness đánh giá + dữ liệu + kết quả đo. (Báo cáo, slide, kịch bản demo nộp riêng, không nằm trong repo.)
>
> **Kết quả cốt lõi (đo thật):** trên Vicuna-7B, SmoothLLM cắt ASR của GCG **97% → 22%**; nhưng với jailbreak ngữ nghĩa chỉ cắt **97% → 77–87%** (residual cao). Trên model nền gemma, suffix **không chuyển giao** (ASR 0%). Chi phí 10× truy vấn trả trên cả hai. → xem [`phase2_pair2_gcg_smoothllm/README.md`](phase2_pair2_gcg_smoothllm/README.md).


Nạp 18 textbook → chunk (tokenizer thật MedCPT) → 2 index **cùng tập chunk, cùng `chunk_id`**:
- **Dense**: Qdrant (vector 768-dim MedCPT-Article, distance Dot, HNSW tắt = exact)
- **Sparse**: Elasticsearch (BM25, analyzer y khoa `medical_en`)
- **Hybrid online**: RRF gộp thứ hạng 2 nhánh theo `chunk_id`.

Embedding đi **qua API** (`http://10.10.50.99:8000`) — **không** cài `torch`/`faiss`/`rank_bm25` local.

---

## 0. Yêu cầu máy
- **RAM ≥ 8GB** (Qdrant ~0.3GB + ES ~1.5GB + Python ~1–2GB). Máy này: 8GB — vừa đủ.
- Docker + Docker Compose, ~2GB đĩa cho index.
- Mạng tới `10.10.50.99:8000` (embedding) và HuggingFace (tải tokenizer 1 lần).

## 1. Cài đặt

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Dựng Qdrant + Elasticsearch

```bash
docker compose up -d
```

⚠️ **Lỗi thường gặp — ES chết khi khởi động** (`failed to obtain lock ... AccessDeniedException`):
thư mục bind-mount `es_storage/` phải cho user ES (UID 1000) ghi được. Chạy:

```bash
mkdir -p es_storage && sudo chown -R 1000:0 es_storage
docker compose restart elasticsearch
```

Health check (đợi ES ~30–60s):
```bash
curl -s localhost:6333/healthz                  # -> healthz check passed
curl -s localhost:9200/_cluster/health | jq .status   # -> green/yellow
```

## 3. Chạy pipeline (theo thứ tự)

```bash
source venv/bin/activate

python setup_indices.py            # Bước 2: tạo Qdrant collection + ES index
python build_corpus.py             # Bước 3: 18 sách -> index/chunks.json (~3 phút)
python build_index.py --only sparse   # ES: vài giây
python build_index.py --only dense    # Qdrant: ~6h qua API (checkpoint/resume)
#   hoặc gộp: python build_index.py   (dense rồi sparse)

python make_manifest.py            # Bước 6a: index/manifest.json (chạy SAU CÙNG)
python verify_index.py             # Bước 6b: 10 test, exit 0 = PASS hết
python smoke_test.py               # Bước 5: 1 query hybrid thật
python readiness_test.py           # Bước 7c: 20 câu dev, SẴN SÀNG CHO RAG?
```

**Dense build lâu (~6h)** vì server embedding là CPU/quantized (~1.8 chunk/s, concurrency
không giúp). An toàn khi crash: checkpoint mỗi 256 chunk ở `index/.checkpoint/`, chạy lại
`build_index.py --only dense` sẽ **resume**. Nạp lại từ đầu: thêm `--restart-dense`.

## 3b. ONLINE — Hybrid RAG (retrieve → rerank → LLM)

Luồng (**không HyDE**):
`question → MedCPT query-embed + BM25 → RRF → MedCPT rerank → top-5 chunk → LLM (OpenRouter)`

**Chuẩn bị key OpenRouter** (bắt buộc cho LLM) — mỗi dòng 1 key, file **gitignored**:
```bash
# openrouter_keys.txt  (đã có; KHÔNG commit)
sk-or-v1-....
```
Client xoay vòng (round-robin) qua tất cả key + failover khi gặp 429/402/5xx.

**Hỏi-đáp mở:**
```bash
python ask.py "What is the first-line treatment for myelofibrosis?"
# -> ANSWER + SOURCES (top-5 chunk_id/source, kèm rerank score)
```

**Chấm accuracy trắc nghiệm:**
```bash
python eval.py                    # eval.limit/eval.split trong config (mặc định 50 câu dev)
python eval.py --limit 200 --split MedQA-USMLE/questions/US/test.jsonl --concurrency 8
# -> ACCURACY x/n + index/eval_results.json
```

Tham số online ở `config.yaml`: `rerank` (candidates 20 → top_k 5), `llm` (model, temperature,
reasoning, keys_file), `eval` (split, limit, concurrency). Component online (dùng chung):
`retrieval.retrieve_rerank()`, `rag.answer(question, options=None)`.

## 4. Build sạch lại từ đầu

```bash
python setup_indices.py --recreate     # xóa + tạo lại cả 2
python build_index.py --restart-dense  # nạp lại từ chunk 0
```

## 5. File & vai trò

| File | Vai trò |
|---|---|
| `config.yaml` | **Mọi tham số** — không hardcode ở script |
| `common.py` | Config loader, tokenizer thật, EmbedClient (retry/backoff), client Qdrant/ES |
| `setup_indices.py` | Tạo Qdrant (Dot, HNSW off) + ES (analyzer `medical_en`, BM25) |
| `build_corpus.py` | Gom đoạn ngắn + chunk, đếm token **thật**, đảm bảo ≤512 |
| `build_index.py` | Nạp **cả** Qdrant + ES từ cùng `chunks.json`, checkpoint/resume dense |
| `retrieval.py` | `rrf_fuse()` + `hybrid_search()` + `retrieve_rerank()` (RRF→MedCPT rerank→top-5) |
| `smoke_test.py` | 1 query hybrid thật |
| `verify_index.py` | 10 test (số lượng, tập chunk_id, ≤512, 768-dim, text khớp 3 nơi...) |
| `make_manifest.py` | Sinh `manifest.json` — mọi script downstream verify `chunks_hash` |
| `readiness_test.py` | 20 câu dev → pipeline RAG có thông không |
| `rag.py` | **ONLINE** `answer(question, options=None)` — MC hoặc hỏi-đáp mở |
| `ask.py` | **ONLINE** CLI hỏi-đáp mở |
| `eval.py` | **ONLINE** chấm accuracy trắc nghiệm (song song + xoay key) |
| `openrouter_keys.txt` | Key OpenRouter (gitignored, không commit) |

## 6. Bất biến quan trọng (đừng phá)
- **Chunk cho dense đi qua `/embed/article`** (Article-Encoder). Query online đi qua
  `/embed/query` (Query-Encoder). Dùng nhầm **không báo lỗi** nhưng hỏng retrieval ngầm.
- **`chunk_id` là khóa gộp RRF** — Qdrant payload `chunk_id` == ES `_id`/`chunk_id`.
  ES map `chunk_id` là `keyword` (không analyze) để khớp chính xác.
- **HNSW tắt** (`m=0`) → exact search, tất định, không mất recall âm thầm.
- **Mọi người dùng chung một `chunks.json`** (verify qua `manifest.json` → `chunks_hash`).
  Nếu hash khác nhau giữa các máy → index không tương thích, ablation vô nghĩa.

## 7. Tham số đã xác nhận (Bước 0)
| | |
|---|---|
| API | `http://10.10.50.99:8000` (HTTP, không HTTPS) |
| Batch `/embed/article` | **8** (9 → 422) |
| Vector | 768-dim, `normalize=false` (dot-product) |
| Subword ratio thật | ~1.41–1.48 tok/từ (y khoa cao → phải đếm token thật) |
| Cắt cụt @512 | Server cắt **âm thầm** — chunk phải ≤512 token thật |
