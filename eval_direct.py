"""
eval_direct.py — Baseline DIRECT LLM (không RAG, không agent) cho MedQA test.
Chỉ đưa câu hỏi + 5 lựa chọn cho LLM, parse letter. CÙNG model/temperature/parse
như RAG-only để so sánh công bằng — điểm khác DUY NHẤT: KHÔNG có context truy hồi.
Output {i,gold,pred,ok,err,latency,n_tokens} tương thích metrics.py. Checkpoint/resume.
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
from common import load_config, make_llm
from eval import load_questions
from rag import parse_choice

DIRECT_SYS = ("You are a medical expert taking the USMLE. Answer the multiple-choice question "
              "using your medical knowledge.")

def build_messages(question, options):
    opt = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (f"Question: {question}\n\nOptions:\n{opt}\n\n"
            f"Think briefly, then end your reply with a line exactly in the form:\n"
            f"Answer: <letter>\nwhere <letter> is one of {', '.join(options.keys())}.")
    return [{"role": "system", "content": DIRECT_SYS}, {"role": "user", "content": user}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1273)
    ap.add_argument("--split", default="MedQA-USMLE/questions/US/test.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cp = os.path.join(cfg["paths"]["checkpoint"], f"direct_{os.path.basename(a.split).replace('.jsonl','')}_{a.limit}.jsonl")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    if a.fresh and os.path.exists(cp):
        os.remove(cp)
    done = {}
    if os.path.exists(cp):
        for l in open(cp):
            try:
                r = json.loads(l)
                if r.get("pred") is None or r.get("err"):  # retry lỗi/không parse
                    continue
                done[r["i"]] = r
            except: pass
    rows = load_questions(a.split, a.limit, cfg["seed"])   # CÙNG seed => cùng thứ tự => i khớp OFF/ON/RAG-only
    todo = [(i, q) for i, q in enumerate(rows) if i not in done]
    print(f"Direct LLM: {len(rows)} câu, đã xong {len(done)}, còn {len(todo)} | conc={a.concurrency}", flush=True)
    llm = make_llm(cfg)
    fh = open(cp, "a")
    import threading; lock = threading.Lock()
    def work(iq):
        i, q = iq
        opts = q["options"]; gold = q["answer_idx"]
        t0 = time.time()
        try:
            msgs = build_messages(q["question"], opts)
            if hasattr(llm, "chat_usage"):
                text, usage = llm.chat_usage(msgs)
            else:
                text, usage = llm.chat(msgs), None
            pred = parse_choice(text, set(opts.keys()))
            ntok = ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)) or None if usage else None
            r = {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "err": None,
                 "latency": round(time.time() - t0, 2), "n_tokens": ntok}
        except Exception as e:
            r = {"i": i, "gold": gold, "pred": None, "ok": False, "err": str(e)[:150],
                 "latency": round(time.time() - t0, 2), "n_tokens": None}
        with lock:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
        return r
    results = dict(done)
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for r in ex.map(work, todo):
            results[r["i"]] = r
    fh.close()
    allr = [results[i] for i in sorted(results)]
    corr = sum(1 for r in allr if r["ok"])
    inval = sum(1 for r in allr if r["pred"] is None or str(r["pred"]).strip().upper() not in set("ABCDE"))
    out = {"total": len(rows), "correct": corr, "accuracy": corr/len(rows),
           "invalid": inval, "invalid_rate": inval/len(rows), "results": allr}
    json.dump(out, open(cfg["paths"]["index_dir"] + "/eval_direct_results.json", "w"), ensure_ascii=False)
    print(f"Direct LLM XONG: {corr}/{len(rows)} = {corr/len(rows)*100:.2f}% | invalid {inval}", flush=True)

if __name__ == "__main__":
    main()
