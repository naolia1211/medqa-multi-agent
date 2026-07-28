# SPEC — Luồng chạy V3 (trừ Long-term memory) · Medical Multi-Agent MedQA

> Phạm vi: **pipeline RAG cố định → 3 agent → short-term memory → trích đáp án**.
> Long-term memory nằm ở spec riêng (`LONGTERM_MEMORY_SPEC.md`) — ở đây chỉ đánh dấu **điểm cắm**.
>
> Nhắc lại quyết định đã chốt: **KHÔNG HyDE · KHÔNG agent Retrieval (CRAG) · 3 agent · Aggregator
> CÓ bằng chứng gốc · dot product, không cosine.**

---

## 0. Sơ đồ luồng

```
Câu hỏi test (1 trong 1.273)
        │
        ▼
[A] PIPELINE RAG — cố định, không phải agent
        Dense (Qdrant) ─┐
                        ├─→ RRF ─→ Re-rank ─→ top 3–5 chunk = BẰNG CHỨNG
        Sparse (ES) ────┘
        │
        ▼
[B] AGENT 1 — Reasoning      ←─ đọc/ghi ─┐
        │                                 │
        ▼                                 │
[C] AGENT 2 — Verifier       ←─ đọc/ghi ─┤── SHORT-TERM (dict RAM)
        │                                 │   trace + bảng ứng viên
        ▼                                 │
[D] AGENT 3 — Aggregator     ←─ đọc/ghi ─┘
        │
        ▼
[E] TRÍCH ĐÁP ÁN → 1 ký tự A–E (lỗi → invalid)
```

---

## 2. [A] PIPELINE RAG — cố định

### 2.1. Nhánh DENSE

```python
# 1) embed câu hỏi
r = requests.post(f"{MEDCPT}/embed/query",
                  json={"queries": [question], "normalize": False})   # DOT PRODUCT
qvec = r.json()["embeddings"][0]

# 2) tìm trong Qdrant
hits = qdrant.search("medqa_dense", query_vector=qvec, limit=30)
dense_ids = [h.payload["chunk_id"] for h in hits]     # đã xếp hạng
```

> ⚠ **CẢNH BÁO QUAN TRỌNG — query bị cắt cụt.**
> `MedCPT-Query-Encoder` có `max_length=64` token (nó được train trên câu tìm kiếm PubMed rất ngắn).
> Nhưng câu hỏi MedQA **median ~114 từ** → khoảng 150–180 WordPiece token → **hơn một nửa câu hỏi
> bị cắt bỏ** trước khi embed. Phần bị mất thường là đoạn cuối — nơi hay chứa câu hỏi thật sự
> ("Which of the following is the most likely diagnosis?") và các chỉ số lab.
>
> **Bắt buộc xử lý một trong các cách sau, và ghi rõ đã chọn cách nào vào report:**
> - **(a) Rút gọn trước khi embed** — tách lấy các cụm lâm sàng cốt lõi (tuổi, giới, triệu chứng,
>   chỉ số bất thường, câu hỏi cuối) rồi mới đưa vào query encoder. Giữ dưới 64 token.
> - **(b) Chia câu hỏi thành 2–3 đoạn**, embed từng đoạn, hợp nhất kết quả bằng RRF.
> - **(c) Dựa nhiều hơn vào nhánh sparse** cho câu dài (BM25 không giới hạn độ dài) — tăng trọng số
>   sparse trong RRF cho những câu vượt ngưỡng token.
>
> Đo thử trên dev để chọn. **Không được bỏ qua** — đây là điểm hỏng âm thầm, dense sẽ trả về kết quả
> lệch mà không báo lỗi gì.

### 2.2. Nhánh SPARSE

```python
res = es.search(index="medqa_sparse", size=30, query={
    "match": {"text": {"query": query_text_for_bm25}}
})
sparse_ids = [h["_id"] for h in res["hits"]["hits"]]
```

`query_text_for_bm25`: dùng **metamap_phrases** nếu có sẵn (thư mục
`questions/US/metamap_extracted_phrases/` trong dataset), vì nó đã trích sẵn thuật ngữ y khoa và
bỏ từ thông dụng. Không có thì dùng nguyên câu hỏi — BM25 tự hạ trọng số từ phổ biến.

### 2.3. Hợp nhất bằng RRF

```python
def rrf(rank_lists, k=60, pool=30):
    scores = {}
    for lst in rank_lists:                       # [dense_ids, sparse_ids]
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])][:pool]
```

RRF chỉ dùng **thứ hạng**, không dùng điểm số — nên không cần chuẩn hoá thang điểm giữa dot-product
và BM25 (hai thang hoàn toàn khác nhau). **Điều kiện tiên quyết: hai nhánh phải cùng hệ `chunk_id`**
(đã đảm bảo ở pha offline).

### 2.4. Re-rank

```python
pairs = [{"query": question, "document": chunk_text[cid]} for cid in rrf_ids]
r = requests.post(f"{MEDCPT}/rerank", json={"query": question, "documents": [...]})
evidence = top_k(r.json(), k=4)     # 3–5 chunk
```
Cross-encoder chấm trực tiếp cặp (câu hỏi, chunk) nên chính xác hơn nhiều so với xếp hạng ban đầu.
**Đây là bước lọc chất lượng chính** — nhờ nó mà 3 agent không cần đi truy lại.

**Output của [A]:** `evidence = [{chunk_id, text, source, rerank_score}, ...]` — 3–5 phần tử.

---

## 3. Short-term memory — khởi tạo

```python
state = {
    "question": question,
    "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
    "evidence": evidence,                    # giữ nguyên, mọi agent đọc được
    "trace": [],                             # CHỈ APPEND, không ghi đè
    "candidates": {k: {"status": "chưa xét", "reason": "", "evidence_ids": []}
                   for k in "ABCDE"},
}
```

**Vòng đời:** tạo mới đầu mỗi câu, **xoá sạch** khi sang câu sau. Không ghi ra đĩa, không DB.

**Quy tắc ghi:**
- `trace.append({...})` — không bao giờ sửa entry cũ.
- `candidates[X]` — được cập nhật, nhưng mọi thay đổi phải kèm một entry trong `trace`.

---

## 4. [B] AGENT 1 — Reasoning

**Thấy:** câu hỏi + 5 lựa chọn + **bằng chứng** (+ lesson từ long-term — *điểm cắm, xem spec riêng*).

**Prompt:**
```
Bạn là bác sĩ đang giải một câu hỏi trắc nghiệm y khoa USMLE.

BẰNG CHỨNG (trích từ sách giáo khoa y khoa):
[1] {evidence[0].text}
[2] {evidence[1].text}
...

CÂU HỎI:
{question}

CÁC LỰA CHỌN:
A. {...}   B. {...}   C. {...}   D. {...}   E. {...}

Nhiệm vụ:
1. Phân tích ca bệnh dựa trên bằng chứng ở trên.
2. Đề xuất 1–2 phương án khả dĩ nhất, nêu rõ lý do.
3. Với mỗi phương án, ghi rõ dựa vào bằng chứng số mấy.
4. Nếu bằng chứng không đủ, nói rõ thiếu gì — KHÔNG bịa.

Trả về JSON thuần:
{"candidates": ["C","B"],
 "reasoning": "...",
 "evidence_used": [1,3]}
```

**Ghi vào state:**
```python
state["trace"].append({"step":1, "agent":"reasoning", "output": out})
for c in out["candidates"]:
    state["candidates"][c] = {"status":"đề xuất", "reason": out["reasoning"],
                              "evidence_ids": out["evidence_used"]}
```

---

## 5. [C] AGENT 2 — Verifier

**Thấy:** câu hỏi + lựa chọn + **bằng chứng** + **toàn bộ trace của Agent 1**.

Vai trò: **phản biện**, không phải trả lời lại. Nó soát từng phương án và loại những cái không đứng vững.

**Prompt:**
```
Bạn là bác sĩ phản biện. Một đồng nghiệp đã phân tích ca bệnh dưới đây.

BẰNG CHỨNG:
[1] ... [2] ... [3] ...

CÂU HỎI: {question}
CÁC LỰA CHỌN: A... B... C... D... E...

PHÂN TÍCH CỦA ĐỒNG NGHIỆP:
{trace của Agent 1}

Nhiệm vụ:
1. Kiểm tra từng khẳng định có ĐÚNG với bằng chứng không. Chỉ ra chỗ suy diễn quá xa
   hoặc không có bằng chứng đỡ.
2. Duyệt CẢ 5 phương án — kể cả phương án đồng nghiệp không nhắc tới.
3. Với mỗi phương án: GIỮ hoặc LOẠI, kèm lý do ngắn.
4. Không cần chốt đáp án cuối.

Trả về JSON thuần:
{"verdicts": {"A":{"decision":"loại","reason":"..."},
              "B":{"decision":"loại","reason":"..."},
              "C":{"decision":"giữ","reason":"..."},
              "D":{"decision":"loại","reason":"..."},
              "E":{"decision":"loại","reason":"..."}},
 "critique": "..."}
```

**Ghi vào state:** append trace step 2, cập nhật `candidates` theo `verdicts`.

> **Điểm cần chú ý:** bắt Agent 2 duyệt **cả 5** phương án (không chỉ cái A1 đề xuất) là cố ý —
> nếu A1 bỏ sót đáp án đúng ngay từ đầu, đây là cơ hội duy nhất để cứu.

---

## 6. [D] AGENT 3 — Aggregator

**Thấy:** câu hỏi + lựa chọn + **BẰNG CHỨNG GỐC** + toàn bộ trace (A1 + A2) + bảng ứng viên.

> Đây là điểm **khác V2**: ở V2, Aggregator KHÔNG thấy bằng chứng gốc (chỉ thấy cái A1/A2 chuyển
> tiếp). Ở V3 nhờ nháp chung nên thấy được. Khi viết report nhớ nêu rõ khoảng cách V3−V2 đến từ
> **cả memory lẫn yếu tố này**.

**Prompt:**
```
Bạn là bác sĩ trưởng, chốt đáp án cuối cùng.

BẰNG CHỨNG GỐC:
[1] ... [2] ... [3] ...

CÂU HỎI: {question}
CÁC LỰA CHỌN: A... B... C... D... E...

PHÂN TÍCH (Agent 1):  {trace step 1}
PHẢN BIỆN (Agent 2):  {trace step 2}
BẢNG ỨNG VIÊN:        {candidates}

Nhiệm vụ:
- Bằng chứng gốc ở trên dùng để KIỂM CHỨNG các khẳng định của hai đồng nghiệp.
- Nếu phản biện của Agent 2 mâu thuẫn với bằng chứng, nêu rõ và xử lý.
- Chốt ĐÚNG MỘT phương án trong A–E. Bắt buộc phải chọn, không được bỏ trống.

Trả về JSON thuần:
{"answer": "C", "explanation": "..." }
```

> Câu *"bằng chứng dùng để KIỂM CHỨNG"* là cố ý — tránh việc Aggregator tự suy luận lại từ đầu và
> bỏ qua phản biện của A2, biến A1/A2 thành thừa.

---

## 7. [E] Trích đáp án

```python
import re, json

def extract_answer(raw):
    # 1) thử parse JSON
    try:
        obj = json.loads(strip_code_fence(raw))
        a = str(obj.get("answer","")).strip().upper()
        if a in "ABCDE" and len(a) == 1:
            return a
    except Exception:
        pass
    # 2) fallback: tìm ký tự A–E đứng độc lập
    m = re.search(r'\b([A-E])\b', raw.upper())
    return m.group(1) if m else "INVALID"
```

- Kết quả hợp lệ: **đúng 1 ký tự A–E**.
- Không trích được → `"INVALID"`, tính là **sai** khi chấm accuracy, đồng thời đếm riêng vào
  **invalid response rate** (đề bài yêu cầu báo cáo chỉ số này).

---

## 8. Vòng lặp chính

```python
for q in test_questions:                       # 1.273 câu
    evidence = rag_pipeline(q["question"])     # [A]
    state    = init_state(q, evidence)         # §3
    run_agent1(state)                          # [B]
    run_agent2(state)                          # [C]
    run_agent3(state)                          # [D]
    ans = extract_answer(state["trace"][-1]["output"])   # [E]
    save_result(q["id"], ans, state["trace"])  # log để error analysis
    state = None                               # xoá short-term
```

**Log bắt buộc lưu:** `qid, predicted, gold, correct, trace, evidence_chunk_ids, latency, n_tokens`.
Đây là nguyên liệu cho: Error Analysis, Win/Loss/Tie, cost & latency — đều là mục đề bài chấm.

---

## 9. Điểm cắm Long-term memory

Ba chỗ, khi bật long-term (xem spec riêng):
1. Trước [B]: `lessons = get_lessons(question)` .
2. Nạp `lessons` vào prompt **Agent 1 và Agent 2** (không nạp cho Agent 3).
3. Log `lessons_used` vào kết quả.

Tắt long-term (để chạy V2, hoặc pha học) = bỏ qua cả 3 chỗ này, phần còn lại giữ nguyên.

---


## 11. Checklist trước khi chạy test

- [ ] `temperature = 0` cho mọi LLM call.
- [ ] Đã xử lý **query truncation 64 token** (§2.1) và ghi rõ cách chọn.
- [ ] Dense dùng **dot product**, `normalize=false` — khớp pha offline.
- [ ] RRF hoạt động trên **cùng hệ `chunk_id`** giữa Qdrant và ES.
- [ ] Aggregator **bắt buộc chọn 1 trong A–E**, không được bỏ trống.
- [ ] `extract_answer` có fallback + đếm `INVALID`.
- [ ] Log đủ trường cho error analysis.
- [ ] Mọi tinh chỉnh (top_k, prompt, ngưỡng) làm **trên dev**; test chạy **một lần**.
