"""
demo/app.py — Live demo web: chọn câu MedQA (hoặc tự nhập) -> chạy từng version, STREAM token
real-time (thấy thinking) + metric mỗi câu (đáp án/đúng-sai/latency/token) + accuracy tổng hợp.
Chạy:  python demo/app.py   (mặc định http://0.0.0.0:7860)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, Response, request, send_from_directory
from common import load_config
from eval import load_questions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # để import pipeline.py cùng thư mục
import pipeline as P

CFG = load_config()
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

_ROWS = load_questions("MedQA-USMLE/questions/US/test.jsonl", 1273, CFG["seed"])   # cùng thứ tự eval

VARIANTS = [
    {"id": "direct", "name": "Direct LLM", "desc": "Không RAG, không agent"},
    {"id": "rag",    "name": "RAG-only",   "desc": "Single-LLM + RAG"},
    {"id": "v2",     "name": "V2",         "desc": "3 agent, không memory"},
    {"id": "v3on",   "name": "V3",         "desc": "3 agent + memory (short-term + long-term)"},
]


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/meta")
def meta():
    # danh sách 60 câu đầu cho dropdown + variants + accuracy tổng hợp
    qs = [{"idx": i, "q": r["question"][:90] + ("..." if len(r["question"]) > 90 else ""),
           "topic": r.get("meta_info", "")} for i, r in enumerate(_ROWS[:60])]
    return {"variants": VARIANTS, "questions": qs}


@app.route("/api/question/<int:idx>")
def question(idx):
    r = _ROWS[idx]
    return {"idx": idx, "question": r["question"], "options": r["options"], "gold": r["answer_idx"],
            "topic": r.get("meta_info", "")}


@app.route("/stream")
def stream():
    variant = request.args.get("variant", "direct")
    idx = request.args.get("idx")
    if idx is not None and idx != "":
        r = _ROWS[int(idx)]
        question, options, gold = r["question"], r["options"], r["answer_idx"]
    else:  # câu tự nhập: options + gold truyền qua query (JSON encode)
        question = request.args.get("question", "")
        options = json.loads(request.args.get("options", "{}"))
        gold = request.args.get("gold") or None

    def gen():
        try:
            for ev in P.run_variant(variant, question, options, gold):
                if ev.get("type") == "_result":
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)[:200]})}\n\n"
        yield "data: {\"type\":\"end\"}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", 7860))
    print(f"Live demo: http://0.0.0.0:{port}  (test.jsonl {len(_ROWS)} câu)", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
