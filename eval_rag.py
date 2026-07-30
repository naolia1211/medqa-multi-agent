"""
eval_rag.py — RAG-only baseline CÓ log token (bổ sung cho eval.py baseline không log token).
Dùng LẠI y hệt pipeline RAG của rag.py (retrieve_rerank + format_context + build_mc_messages
+ parse_choice) nên accuracy TRÙNG eval.py; khác DUY NHẤT: gọi llm.chat_usage() để bắt token
thật từ API. eval.py/rag.py GIỮ NGUYÊN (CLAUDE.md). Record {i,gold,pred,ok,err,latency,n_tokens}
tương thích metrics.py. Checkpoint/resume + concurrency.
"""
import os, sys, time, json, argparse, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
from common import load_config, make_llm, EmbedClient, get_qdrant, get_es
from eval import load_questions
from retrieval import retrieve_rerank
from rag import format_context, build_mc_messages, parse_choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1273)
    ap.add_argument("--split", default="MedQA-USMLE/questions/US/test.jsonl")
    ap.add_argument("--concurrency", type=int, default=3)   # retrieval-heavy -> conc thấp (MedCPT CPU)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cp = os.path.join(cfg["paths"]["checkpoint"],
                      f"rag_{os.path.basename(a.split).replace('.jsonl','')}_{a.limit}.jsonl")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    if a.fresh and os.path.exists(cp):
        os.remove(cp)
    done = {}
    if os.path.exists(cp):
        for l in open(cp):
            try:
                r = json.loads(l)
                if r.get("pred") is None or r.get("err"):
                    continue
                done[r["i"]] = r
            except: pass
    rows = load_questions(a.split, a.limit, cfg["seed"])   # cùng seed -> i khớp variant khác
    todo = [(i, q) for i, q in enumerate(rows) if i not in done]
    print(f"RAG-only(+token): {len(rows)} câu, đã xong {len(done)}, còn {len(todo)} | conc={a.concurrency}", flush=True)
    ec, qc, es, llm = EmbedClient(cfg), get_qdrant(cfg), get_es(cfg), make_llm(cfg)
    fh = open(cp, "a"); lock = threading.Lock()

    def work(iq):
        i, q = iq
        opts = q["options"]; gold = q["answer_idx"]
        t0 = time.time()
        try:
            chunks = retrieve_rerank(cfg, q["question"], embed_client=ec, qdrant=qc, es=es)
            msgs = build_mc_messages(q["question"], opts, format_context(chunks))
            text, usage = llm.chat_usage(msgs)
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
    out = {"total": len(rows), "correct": corr, "accuracy": corr / len(rows),
           "invalid": inval, "invalid_rate": inval / len(rows), "results": allr}
    json.dump(out, open(cfg["paths"]["index_dir"] + "/eval_rag_results.json", "w"), ensure_ascii=False)
    print(f"RAG-only(+token) XONG: {corr}/{len(rows)} = {corr/len(rows)*100:.2f}% | invalid {inval}", flush=True)


if __name__ == "__main__":
    main()
