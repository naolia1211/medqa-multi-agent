"""
tool_probe.py — Kiểm tra model đang cấu hình có hỗ trợ FUNCTION/TOOL CALLING native không.

Gửi 1 request kèm định nghĩa tool + prompt kích hoạt, xem model có trả `tool_calls` đúng cấu trúc.
Đây là DIAGNOSTIC live (giống smoke_test.py) — cần model sống. Verdict:
  SUPPORTED    — model trả tool_calls đúng tên hàm + tham số.
  NOT_SUPPORTED— model bỏ qua tools, trả text thường.
  ERROR        — backend từ chối tools (HTTP != 200).

  python tool_probe.py
"""
import json
import requests

from common import load_config, setup_logging

TOOL = {
    "type": "function",
    "function": {
        "name": "search_textbooks",
        "description": "Tra cứu sách giáo khoa y khoa để lấy bằng chứng cho một truy vấn lâm sàng.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "cụm từ khoá y khoa cần tra cứu"},
            },
            "required": ["query"],
        },
    },
}
PROMPT = ("Bạn cần bằng chứng về first-line treatment của myelofibrosis nhưng CHƯA có dữ liệu. "
          "Hãy DÙNG công cụ search_textbooks để tra cứu (đừng tự trả lời).")


def _args_dict(fn):
    """Ollama/OpenAI: arguments có thể là dict hoặc chuỗi JSON."""
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a or {}


def probe_ollama(cfg):
    oc = cfg["llm"]["ollama"]
    body = {
        "model": oc["model"], "stream": False,
        "keep_alive": oc.get("keep_alive", "30m"),
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [TOOL],
        "options": {"temperature": 0},
    }
    r = requests.post(oc["base_url"].rstrip("/") + "/api/chat", json=body, timeout=180)
    ctype = r.headers.get("content-type", "")
    return r.status_code, (r.json() if ctype.startswith("application/json") else r.text), oc["model"]


def probe_openrouter(cfg):
    import os
    c = cfg["llm"]
    kp = c["keys_file"]
    kp = kp if os.path.isabs(kp) else os.path.join(os.path.dirname(os.path.abspath(__file__)), kp)
    key = open(kp).read().split()[0]
    body = {"model": c["model"], "messages": [{"role": "user", "content": PROMPT}],
            "tools": [TOOL], "temperature": 0}
    r = requests.post(c["base_url"], headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    return r.status_code, r.json(), c["model"]


def verdict_from_message(msg):
    """msg: dict message của assistant. Trả (verdict, chi tiết)."""
    tcs = msg.get("tool_calls")
    if tcs:
        fn = tcs[0].get("function", {})
        args = _args_dict(fn)
        ok = fn.get("name") == "search_textbooks" and "query" in args
        return ("SUPPORTED" if ok else "PARTIAL",
                f"tool_calls={json.dumps(tcs, ensure_ascii=False)[:300]}")
    return "NOT_SUPPORTED", f"content={(msg.get('content') or '')[:200]!r}"


def main():
    log = setup_logging("tool_probe")
    cfg = load_config()
    prov = cfg["llm"].get("provider", "openrouter")
    log.info("Provider=%s — probe tool-calling...", prov)
    try:
        if prov == "ollama":
            code, j, model = probe_ollama(cfg)
        elif prov == "openrouter":
            code, j, model = probe_openrouter(cfg)
        else:
            print(f"Provider '{prov}' chưa có nhánh probe. openrouter/google hỗ trợ tools native.")
            return
    except Exception as e:
        print(f"VERDICT: ERROR — probe ném lỗi: {str(e)[:200]}")
        return

    print(f"\nModel: {model}")
    print(f"HTTP: {code}")
    if code != 200:
        print(f"VERDICT: ERROR — backend từ chối tools: {str(j)[:250]}")
        return

    if prov == "ollama":
        msg = j.get("message", {})
    else:  # openrouter
        msg = (j.get("choices") or [{}])[0].get("message", {})

    v, detail = verdict_from_message(msg)
    print(detail)
    labels = {"SUPPORTED": "SUPPORTED ✅ — model trả tool_calls đúng chuẩn, DÙNG được function-calling.",
              "PARTIAL": "PARTIAL ⚠️ — có tool_calls nhưng cấu trúc lệch, cần chuẩn hoá.",
              "NOT_SUPPORTED": "NOT_SUPPORTED ❌ — model bỏ qua tools, chỉ trả text -> phải emulate qua prompt."}
    print("VERDICT:", labels[v])


if __name__ == "__main__":
    main()
