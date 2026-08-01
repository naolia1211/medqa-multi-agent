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


def _mlflow_client():
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    return mlflow, MlflowClient()


@app.route("/api/experiments")
def experiments():
    # filter PHIÊN BẢN: liệt kê experiment trên MLflow (v1 = medqa-v3; v2 tương lai ở experiment khác). Server-side.
    try:
        _, c = _mlflow_client()
        exps = [{"id": e.experiment_id, "name": e.name}
                for e in c.search_experiments() if e.name != "Default"]
        return {"experiments": exps}
    except Exception as e:
        return {"experiments": [], "error": str(e)[:120]}


_VAR_CANDIDATES = ["direct", "rag", "v2", "v3on", "v3-mem-on", "v3-mem-off", "v3"]

@app.route("/api/variants")
def variants_in_exp():
    # variant có trong 1 experiment. Probe từng ứng viên bằng filtered search (mỗi cái ~0.2s -> nhanh,
    # bắt được cả variant ở trace CŨ mà sample-recent bỏ sót). Server-side.
    exp_id = request.args.get("experiment_id")
    try:
        mlflow, c = _mlflow_client()
        if not exp_id:
            e = c.get_experiment_by_name(CFG["mlflow"]["experiment"]); exp_id = e.experiment_id if e else None
        vs = []
        for v in _VAR_CANDIDATES:
            tr = mlflow.search_traces(experiment_ids=[exp_id], filter_string=f"tags.variant = '{v}'",
                                      max_results=1, return_type="list")
            if tr:
                vs.append(v)
        return {"variants": vs}
    except Exception as e:
        return {"variants": [], "error": str(e)[:120]}


def _span_io(sp):
    """Rút request (messages) + response từ 1 span, cắt gọn."""
    req = ""
    inp = sp.inputs or {}
    if isinstance(inp, dict) and inp.get("messages"):
        req = "\n".join(m.get("content", "") for m in inp["messages"] if isinstance(m, dict))
    else:
        req = str(inp)
    out = sp.outputs or {}
    resp = out.get("response") if isinstance(out, dict) and out.get("response") else str(out)
    return req[:6000], (resp or "")[:6000]


@app.route("/api/traces")
def traces():
    # DETAIL theo variant: lấy mẫu trace (request/response/pred/correct) từ MLflow. Server-side (client chỉ gọi BE).
    exp_id = request.args.get("experiment_id")
    variant = request.args.get("variant", "")
    limit = min(int(request.args.get("limit", 15)), 40)
    try:
        mlflow, c = _mlflow_client()
        if not exp_id:
            e = c.get_experiment_by_name(CFG["mlflow"]["experiment"]); exp_id = e.experiment_id if e else None
        fs = f"tags.variant = '{variant}'" if variant else None
        infos = mlflow.search_traces(experiment_ids=[exp_id], filter_string=fs,
                                     max_results=limit, return_type="list")
        out = []
        for ti in infos:
            tid = getattr(ti.info, "trace_id", None) or getattr(ti.info, "request_id", None)
            tg = getattr(ti.info, "tags", {}) or {}
            spans = []
            try:
                full = c.get_trace(tid)
                for sp in full.data.spans:
                    if sp.inputs and (("messages" in str(sp.inputs)) or ("query" in str(sp.inputs))):
                        req, resp = _span_io(sp)
                        spans.append({"agent": sp.name, "request": req, "response": resp})
            except Exception:
                pass
            out.append({"q_index": tg.get("q_index"), "gold": tg.get("gold"),
                        "pred": tg.get("pred"), "correct": tg.get("correct") == "True",
                        "spans": spans})
        # sắp theo q_index
        out.sort(key=lambda x: int(x["q_index"]) if str(x.get("q_index") or "").isdigit() else 0)
        return {"variant": variant, "n": len(out), "traces": out}
    except Exception as e:
        return {"traces": [], "error": str(e)[:160]}


@app.route("/api/metrics")
def metrics():
    # đọc kết quả benchmark tổng hợp (docs/metrics_summary.json — sinh bởi gen_summary.py)
    p = os.path.join(os.path.dirname(HERE), "docs", "metrics_summary.json")
    if not os.path.exists(p):
        return {"error": "chưa có metrics_summary.json"}, 404
    return json.load(open(p, encoding="utf-8"))


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
        # Padding 2KB đầu stream: ép proxy (nginx/NPM) FLUSH ngay, vượt qua buffer nhỏ mặc định.
        yield ": " + (" " * 2048) + "\n\n"
        yield "retry: 3000\n\n"
        try:
            for ev in P.run_variant(variant, question, options, gold):
                if ev.get("type") == "_result":
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)[:200]})}\n\n"
        yield "data: {\"type\":\"end\"}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", 7860))
    print(f"Live demo: http://0.0.0.0:{port}  (test.jsonl {len(_ROWS)} câu)", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
