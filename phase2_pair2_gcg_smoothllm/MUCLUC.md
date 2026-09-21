# Mục lục thư mục — Pair 2 (GCG vs SmoothLLM)

Đã phân loại lại ngày 2026-09-21. `README.md` là bản gốc của nhóm; file này chỉ
mô tả cách sắp xếp thư mục sau khi dọn.

| Thư mục | Nội dung | Ghi chú |
|---|---|---|
| `de_bai/` | `Final_Project_Attack_Defense.pdf` | Đề bài tổng (Attack & Defend), KHÔNG phải báo cáo. |
| `demo/` | `replay_console.html` — demo offline, phát lại 4 ca đã đo | Mở bằng trình duyệt, không cần GPU. |
| `code/` | Toàn bộ mã phase 2 (`src/`, `tests/`, runner…) | Đường dẫn trong code theo layout Kaggle gốc (`data/raw/gcg`, `results/…`), cần re-path khi chạy lại. |
| `data/eval_sets/` | 3 tập đánh giá (safety, benign, MedQA) | |
| `data/phase1_gcg/` | Sản phẩm phase-1 mà code đọc (suffix, refusal prefixes…) | Không di chuyển — code phụ thuộc. |
| `ket_qua/` | Raw records + bảng chỉ số + `ghi_chu_phan_tich.md` | 700+400+200 dòng raw, đã đối chiếu khớp headline. |
| `phase1_attack/` | Gói bàn giao phase-1 (GCG log + suffix + input SmoothLLM) | = nội dung `Pair2_ATTACK_results.zip`, trùng byte với `data/phase1_gcg`. |
| `_giua_ky/` | 2 PDF giữa kỳ | KHÔNG thuộc Pair 2 — cách ly để khỏi nhầm. |

Zip gốc lưu ở `../_archive_zip/` (`Pair2_ATTACK_results.zip`, `Pair2_GCG_SmoothLLM.zip`).
