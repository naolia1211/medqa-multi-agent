# KẾT QUẢ ĐÁNH GIÁ — MedQA-USMLE (test split, 1273 câu)

**Model:** `gemma-4-12B-it-qat-q4_0` (Ollama, temperature 0) · **Mẫu số cố định:** 1273 · **Seed:** 42 (cùng thứ tự câu cho mọi variant → so sánh ghép cặp hợp lệ) · **Invalid = 0%** cho cả 4 variant.

Long-term memory học từ **dev** (1272 câu, chỉ lưu câu SAI + đáp án đúng → 304 lessons), đóng băng khi test (`save_wrong=false`, không rò test vào memory).

---

## 1. Bảng kết quả chính (4 variant)

| Hạng | Variant | Accuracy | Bootstrap 95% CI | Invalid | Latency median |
|:---:|---|---:|:---:|:---:|---:|
| 🥇 | **Direct LLM** (không RAG, không agent) | **81.70%** (1040/1273) | 79.6–83.8% | 0% | 31s |
| 🥈 | **RAG-only** (single-LLM + RAG) | **79.81%** (1016/1273) | 77.6–82.0% | 0% | 46s |
| 🥉 | **V3 multi-agent ON** (memory bật) | **76.83%** (978/1273) | 74.5–79.1% | 0% | 178s |
| 4 | **V3 multi-agent OFF** (memory tắt) | **73.29%** (933/1273) | 70.9–75.7% | 0% | 164s |

**Xếp hạng:** Direct LLM > RAG-only > V3-ON > V3-OFF.

---

## 2. So sánh ghép cặp (McNemar exact two-sided + bootstrap 95% CI của hiệu, paired)

| Nâng cấp pipeline | Accuracy gain | Win / Loss / Tie | McNemar p | CI của hiệu | Ý nghĩa |
|---|---:|:---:|:---:|:---:|---|
| Direct → **+RAG** | −1.89đ | 84 / 108 / 1081 | 0.097 | [−4.01, +0.24] | ❌ Không ý nghĩa (RAG không giúp) |
| RAG-only → **+multi-agent** | **−6.52đ** | 50 / 133 / 1090 | <0.0001 | [−8.56, −4.48] | 🔴 Có ý nghĩa — agent **làm hại** |
| V3-OFF → **+long-term memory** | **+3.53đ** | 94 / 49 / 1130 | 0.0002 | [+1.73, +5.34] | ✅ Có ý nghĩa — memory **giúp** |
| V3-ON → Direct LLM | +4.87đ | 142 / 80 / 1051 | <0.0001 | [+2.59, +7.15] | Direct thắng hệ đầy đủ nhất |

*Win = variant mới đúng & variant gốc sai. McNemar: b = gốc đúng/mới sai, c = mới đúng/gốc sai.*

---

## 3. Chi phí & độ trễ (cost/latency)

| Variant | Latency mean / median / p95 | Tổng latency* | Token tổng (in+out) |
|---|:---:|---:|---:|
| Direct LLM | 34 / 31 / 53 s | 11.9h | 905,219 |
| RAG-only | 48 / 46 / 73 s | 17.1h | (đang chạy lại qua `eval_rag.py` để bổ sung)** |
| V3-OFF | 168 / 164 / 216 s | 59.5h | 12,391,675 |
| V3-ON | 183 / 178 / 246 s | 64.7h | 15,009,003 |

\* Tổng latency = tổng thời-gian-mỗi-câu (chi phí tính toán); wall-clock ≈ tổng ÷ concurrency (Direct/V3 chạy conc 8, RAG-only conc 3).
\** `eval.py` (baseline) không log token — đang chạy lại RAG-only qua `eval_rag.py` (dùng `chat_usage()`) để bổ sung token, giữ nguyên accuracy 79.81%.

Số liệu máy-đọc đầy đủ (per-variant + pairwise): [`metrics_summary.json`](metrics_summary.json). Sinh lại: `python gen_summary.py`. Bảng so sánh: `python final_metrics.py`.

**Nhận xét chi phí:** V3 tốn **~5–6× thời gian** và **~15–16× token** so với Direct/RAG-only, nhưng accuracy THẤP hơn. Memory (V3-ON) thêm +21% token, +8% latency so với V3-OFF để đổi lấy +3.53đ.

---

## 4. Kết luận

1. **Long-term memory là đóng góp tích cực có ý nghĩa thống kê** (+3.53đ, p=0.0002, CI loại trừ 0) — học từ câu sai quá khứ giúp V3 trả lời đúng hơn, không rò rỉ (dev⊥test, store đóng băng).
2. **Kiến trúc multi-agent debate làm giảm accuracy mạnh** trên model đủ mạnh (−6.52đ so với RAG-only, p<0.0001): tranh luận giữa các agent đưa nhiễu/đảo đáp án đúng.
3. **RAG không cải thiện** so với dùng thẳng model (−1.89đ, không ý nghĩa): model đã có đủ kiến thức y khoa; retrieval thêm ngữ cảnh gây nhiễu nhẹ.
4. **Đơn giản nhất thắng:** Direct LLM (81.70%) tốt nhất về cả accuracy lẫn chi phí. Càng thêm tầng (RAG → agent) accuracy càng giảm và chi phí càng tăng.

**Hàm ý:** memory hữu ích nhưng chỉ bù được một phần thiệt hại của debate; với model mạnh nên cân nhắc bỏ tầng multi-agent, giữ memory như cơ chế học từ lỗi.
