"""
notify.py — Gửi tin nhắn tiến độ qua Telegram Bot API.
Cấu hình đọc từ telegram.md (gitignored): API_TOKEN, Channel_ID, THREAD_ID.

  python notify.py "nội dung"        # gửi 1 tin (dùng để kiểm thử)
  from notify import send; send("...")

Thiết kế graceful: lỗi mạng/telegram KHÔNG được làm chết job đang giám sát -> trả (ok, info),
không raise.
"""
import sys
import os
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_cfg(path=None):
    path = path or os.path.join(ROOT, "telegram.md")
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def send(text, path=None, parse_mode="Markdown"):
    """Gửi text tới channel+thread cấu hình. Trả (ok: bool, info: str). Không raise."""
    try:
        c = load_cfg(path)
        token = c["API_TOKEN"]
        payload = {"chat_id": c["Channel_ID"], "text": text,
                   "disable_web_page_preview": True}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if c.get("THREAD_ID"):
            payload["message_thread_id"] = int(c["THREAD_ID"])
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json=payload, timeout=15)
        ok = r.status_code == 200 and r.json().get("ok", False)
        return ok, (r.text[:200] if not ok else "sent")
    except Exception as e:
        return False, str(e)[:200]


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "🧪 notify.py test"
    ok, info = send(msg)
    print("OK" if ok else "FAIL", "-", info)
    sys.exit(0 if ok else 1)
