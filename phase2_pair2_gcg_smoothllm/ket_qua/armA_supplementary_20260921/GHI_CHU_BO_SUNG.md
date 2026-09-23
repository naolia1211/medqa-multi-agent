# Kết quả bổ sung arm-A — đo trực tiếp trên Vicuna-7B-v1.3 fp16 (Tesla P40)

Đo ngày 2026-09-21 để đóng hai lỗ hổng của báo cáo gốc. Model tải mới từ HF
(`lmsys/vicuna-7b-v1.3`, fp16), host in-process qua transformers 4.36.2 /
torch 2.2.2+cu121. **Tái dùng nguyên vẹn** `judges.py` (refusal-prefix) và
perturbation `random_swap` của nhóm để số liệu so sánh được với báo cáo.

Script: `armA_supp.py` (subcommands medqa / semantic). Raw đính kèm thư mục này.

---

## #2 — MedQA arm-A với CONSTRAINED decoding (đóng §7.6)

100 câu MedQA-USMLE test (spread-sample offset đều), N=4, q=10%, decode ràng
buộc 1 token trong {A,B,C,D} (logit-restricted).

| | C0 (không phòng thủ) | C3 (SmoothLLM) |
|---|---|---|
| Accuracy | 33.0% | 30.0% |
| **INVALID** | **0** | **0** |
| Số câu đoán "A" | 71/100 | 62/100 |

- **Constrained decoding loại sạch INVALID** (bản free-form §7.6 cũ: 21.5% C0 / 0% C3
  — artifact bất đối xứng). Phép đo arm-A MedQA **giờ hợp lệ**.
- Chi phí phòng thủ: **−3.0 pp**, McNemar b=12 / c=9 → **không có ý nghĩa thống kê**.
  Nhất quán với arm B (−6 pp, p=0.21). Hai arm cùng chiều, cùng "không phân biệt được với 0".
- **Phát hiện mới:** kể cả không phòng thủ, Vicuna-7B **thiên lệch chọn "A" 71%**
  (gold-A cân bằng chỉ 28% — A/B/C/D = 28/27/25/20). Accuracy 33% chỉ nhỉnh trên
  mức ngẫu nhiên 25%. → §7.6 "sụp về A" **một phần là hạn chế năng lực của model**,
  không chỉ là artifact free-form. Constrained decoding vạch ra điều này thay vì
  che đi. SmoothLLM còn *giảm nhẹ* thiên lệch A (71→62) do nhiễu, nhưng không cải thiện accuracy.

Kết luận §7.6: giả thuyết "constrained decoding cứu được phép đo" — **ĐÚNG về tính
hợp lệ** (0 INVALID, hai arm so sánh được), và cho ra một chi phí −3pp không đáng kể.

---

## #1 — Semantic (persuasive) jailbreak vs SmoothLLM (chứng minh §7.4)

Template "AIM" (roleplay tuân lệnh tuyệt đối, mạch lạc ngữ nghĩa, KHÔNG chứa chuỗi
token kém bền như suffix GCG), áp lên 30 goal AdvBench held-out. N=10, q=10% swap.

| Attack | C0 bare | C1 (không phòng thủ) | C2 (SmoothLLM) | Giảm | Residual |
|---|---|---|---|---|---|
| **GCG** (báo cáo §6.2) | 7% | 97% | **22%** | 75 pp | 22% |
| **Semantic AIM** (đo mới) | 10% | 96.7% | **76.7%** | **20 pp** | **76.7%** |
| **Semantic persona** (đo mới) | 10% | 100% | **86.7%** | **13.3 pp** | **86.7%** |

- Cùng một SmoothLLM, cùng judge: chặn GCG **75pp** nhưng semantic **chỉ 20pp**.
  Residual **76.7% so với 22%** — SmoothLLM gần như **mù** trước jailbreak ngữ nghĩa.
- Đây là **bằng chứng định lượng** cho câu §7.4 (vốn chỉ nêu, chưa đo). Đúng như
  cơ chế: nhiễu ký tự phá suffix GCG kém bền, nhưng không phá được chỉ dẫn mạch lạc.
- **Tính hợp lệ đã kiểm:** C1 tạo compliance có hại THẬT (29/30), ví dụ fake news,
  blog cổ vũ chán ăn, hướng dẫn ransomware — không phải artifact "Sure, here is: I'm sorry".

**Caveat (giữ trung thực):** ASR đo bằng refusal-prefix judge = "affirmative-prefix
compliance", không phải "delivered harm" — cùng giới hạn §7.2, áp cho CẢ hai attack
nên phép so sánh 20pp-vs-75pp vẫn vững. n=30 (SE ~±8pp) nên residual 76.7% là ước
lượng điểm; khác biệt với 22% của GCG lớn hơn nhiều lần noise band.

## Hạ tầng
Tesla P40 24GB (Pascal SM 6.1 → không chạy vLLM; dùng transformers fp16). MedQA
constrained ~9 phút; mỗi template semantic ~26 phút (P40 decode latency-bound).
MedQA từng OOM ở batch 16 trên vignette dài → hạ batch + `expandable_segments`.
