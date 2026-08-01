# KẾT QUẢ ĐÁNH GIÁ — MedQA-USMLE (test split, 1273 câu)

**Model:** `gemma-4-12B-it-qat-q4_0` (Ollama, temperature 0) · **Mẫu số cố định:** 1273 · **Seed:** 42 (cùng thứ tự câu cho mọi variant → so sánh ghép cặp hợp lệ) · **Invalid = 0%** cho cả 4 variant.

Long-term memory học từ **dev** (1272 câu, chỉ lưu câu SAI + đáp án đúng → 304 lessons), đóng băng khi test (`save_wrong=false`, không rò test vào memory).

Số liệu máy-đọc đầy đủ: [`metrics_summary.json`](metrics_summary.json). Sinh lại: `python gen_summary.py` · Bảng so sánh: `python final_metrics.py`.

---

## 1. Bảng kết quả chính (4 variant)

| Hạng | Variant | Kiến trúc | Accuracy | Bootstrap 95% CI | Latency median |
|:---:|---|---|---:|:---:|---:|
| 🥇 | **Direct LLM** | không RAG, không agent | **81.70%** (1040/1273) | 79.6–83.8 | 31s |
| 🥈 | **RAG-only** | single-LLM + RAG | **79.50%** (1012/1273) | 77.3–81.7 | 46s |
| 🥉 | **V3** | 3 agent + **memory** (short-term + long-term) | **76.83%** (978/1273) | 74.5–79.1 | 178s |
| 4 | **V2** | 3 agent, **KHÔNG memory** | **76.75%** (977/1273) | 74.4–79.0 | 81s |

**Xếp hạng:** Direct > RAG-only > (V3 ≈ V2).

---

## 2. So sánh ghép cặp — thang bậc pipeline (McNemar exact + bootstrap 95% CI, paired)

| Bước nâng cấp | Gain | Win/Loss/Tie | McNemar p | CI của hiệu | Ý nghĩa |
|---|---:|:---:|:---:|:---:|---|
| Direct → **+RAG** | −2.20đ | 77/105/1091 | 0.045 | [−4.32, −0.16] | ⚠️ RAG hơi hại |
| RAG-only → **+multi-agent** (→V2) | −2.75đ | 71/106/1096 | 0.010 | [−4.79, −0.79] | 🔴 3 agent làm hại |
| **V2 → +memory** (→V3) | **+0.08đ** | 78/77/1118 | **1.0000** | [−1.89, +2.04] | ⚖️ **KHÔNG có tác dụng ròng** |

---

## 3. 🎯 Phát hiện then chốt: memory KHÔNG cải thiện hệ multi-agent

- **V2 (3 agent, ZERO memory) = 76.75%** vs **V3 (3 agent, đầy đủ memory) = 76.83%** → hiệu **+0.08đ, p=1.0000** — thống kê **không phân biệt được**.
- Nghĩa là **thêm memory vào hệ 3-agent KHÔNG mang lại cải thiện ròng** trên model đủ mạnh.

**Vì sao?** (phân tích bổ sung — dùng điểm chẩn đoán trung gian "chỉ short-term", tức Aggregator được thấy lại evidence gốc):
- **Short-term** (Aggregator dùng "nháp chung" thấy evidence): **−3.46đ, p=0.0003** → làm HẠI.
- **Long-term** (nạp lesson câu-sai vào Agent 1+2): **+3.53đ, p=0.0002** → GIÚP.
- → Hai thành phần **triệt tiêu nhau** ⇒ V3 (cả hai) ≈ V2 (không có).

Long-term memory **có tác dụng thật**, nhưng trong kiến trúc này nó chỉ **bù đúng phần thiệt hại do short-term gây ra**, không đẩy hệ vượt mức "3 agent không memory".

---

## 4. Chi phí & độ trễ

| Variant | Latency mean/median/p95 | Token tổng (in+out) |
|---|:---:|---:|
| Direct LLM | 34 / 31 / 53 s | 905,219 |
| RAG-only | 48 / 46 / 73 s | 4,105,143 |
| V2 (3 agent, no-mem) | 98 / 81 / 248 s | 9,625,679 |
| V3 (3 agent, +memory) | 183 / 178 / 246 s | 15,009,003 |

V3 tốn **~16× token** và **~5-6× thời gian** so với Direct nhưng accuracy **thấp hơn ~5 điểm**. V2 rẻ hơn V3 rõ (Aggregator không cần evidence → prompt ngắn) mà accuracy tương đương.

---

## 5. Ghi chú phương pháp — non-determinism ở temperature 0

RAG-only đo 2 lần cùng cấu hình: 79.81% (`eval.py`) vs 79.50% (`eval_rag.py`, có token) — chênh 0.31đ do **LLM temp=0 KHÔNG bit-exact** trên Ollama/GPU (batching/floating-point). Dùng lần 2 làm canonical (accuracy + token cùng run). Drift ~0.3đ < bootstrap CI (±2–4đ) nên không đổi kết luận; nhưng với so sánh sát ngưỡng (Direct→+RAG p=0.045) cần thận trọng.

---

## 6. Kết luận

1. **Đơn giản nhất thắng:** Direct LLM (81.70%) tốt nhất cả accuracy lẫn chi phí. Mỗi tầng thêm vào (RAG −2.2đ, multi-agent −2.75đ) đều **giảm** accuracy trên model đủ mạnh.
2. **Memory không cải thiện hệ multi-agent** (V2 ≈ V3, +0.08đ, p=1.0): tác dụng long-term (+3.53) bị short-term (−3.46) triệt tiêu.
3. **Long-term memory tự thân có ý nghĩa** (+3.53đ, p=0.0002, học từ lỗi quá khứ, không rò) — nhưng cần **bỏ short-term** (Aggregator không nên thấy lại evidence) thì mới thành cải thiện ròng.
4. **Khuyến nghị:** với model mạnh, cân nhắc bỏ tầng multi-agent; nếu giữ, dùng long-term memory + bỏ short-term để memory thực sự có ích.
