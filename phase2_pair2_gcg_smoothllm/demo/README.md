# Demo trực quan

`replay_console.html` — mở bằng trình duyệt bất kỳ (nháy đúp file), không cần
mạng, không cần GPU, không gọi model.

Trang phát lại **đúng bốn ca đã đo** ngày 18/09/2026 trên Vicuna-7B-v1.3 fp16,
SmoothLLM N=10, q=10% swap. Phiếu bầu từng bản sao và số liệu chi phí lấy
nguyên từ `../ket_qua/demo_transcript.txt`.

## Về phần đã lược

Nguyên văn goal AdvBench được thay bằng nhãn phân loại; phần nội dung của câu
trả lời được cắt ngay sau dấu hiệu tuân theo hoặc từ chối. Những gì còn hiển
thị là **trích nguyên văn, không sửa chữ nào** — cắt chứ không viết lại, để
trích dẫn vẫn dùng được. Suffix GCG giữ đầy đủ vì nó là đối tượng nghiên cứu
và bản thân nó vô nghĩa. Bản ghi đầy đủ nằm ở `../ket_qua/demo_transcript.txt`
nếu cần đối chiếu.

## Bốn ca

| Ca | Lớp kết cục | Tỉ lệ trong 100 goal |
|---|---|---|
| A | Phòng thủ chặn được | 75 |
| B | Vượt qua phòng thủ (residual) | 18 |
| C | Phán quyết không ổn định, hòa 5–5 | 11 |
| D | Model tự trả lời khi chưa bị tấn công | 7 |

## Điều khiển

- **Phát cả 4 ca** — chạy liên tiếp, khoảng 2,5 phút ở tốc độ 1×
- **Chạy lại ca này** — phát lại ca đang chọn
- **0,5× / 1× / 2×** — tốc độ
- Phím tắt: `Space` phát/dừng, `←` `→` chuyển ca

## Quay video

Mở ở cửa sổ rộng từ 1440×900 trở lên, chọn 1×, bấm "Phát cả 4 ca" rồi quay
toàn màn hình. Ở độ rộng đó 10 ô bản sao xếp 5×2 vừa một khung hình, không
phải cuộn.
