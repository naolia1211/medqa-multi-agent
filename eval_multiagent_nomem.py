"""
eval_multiagent_nomem.py — Variant "3 agent, ZERO memory".
Không short-term (agent KHÔNG chia sẻ state) và không long-term (không lesson).
3 agent reasoning ĐỘC LẬP, mỗi agent 1 góc suy luận khác nhau (tạo đa dạng ở temp=0),
mỗi agent chỉ thấy: câu hỏi + lựa chọn + bằng chứng RAG. Gộp bằng MAJORITY VOTE.
=> cô lập tác dụng của kiến trúc multi-agent KHI không có bất kỳ memory nào,
   so với V3-OFF (có short-term/debate) và RAG-only (1 call).
Record {i,gold,pred,ok,err,latency,n_tokens,votes} tương thích metrics.py. Checkpoint/resume.
"""
import os, sys, time, json, argparse, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import load_config, make_llm, EmbedClient, get_qdrant, get_es
from eval import load_questions
from retrieval import retrieve_rerank
from rag import format_context, parse_choice

# 3 góc suy luận độc lập (persona khác nhau -> đa dạng dù temp=0). KHÔNG agent nào thấy output agent khác.
PERSONAS = [
    "You are a medical expert. Solve the question by step-by-step clinical reasoning from the presentation.",
    "You are a medical expert. Solve the question by DIFFERENTIAL DIAGNOSIS: rule out each incorrect option, keep the best.",
    "You are a medical expert. Solve the question by identifying the KEY pathophysiology/mechanism, then map it to the answer.",
]

def build_msgs(persona, question, options, context):
    opt = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (f"Reference context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{opt}\n\n"
            f"Think briefly, then end with a line exactly:\nAnswer: <letter>\n"
            f"where <letter> is one of {', '.join(options.keys())}.")
    return [{"role": "system", "content": persona}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1273)
    ap.add_argument("--split", default="MedQA-USMLE/questions/US/test.jsonl")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cp = os.path.join(cfg["paths"]["checkpoint"],
                      f"manomem_{os.path.basename(a.split).replace('.jsonl','')}_{a.limit}.jsonl")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    if a.fresh and os.path.exists(cp):
        os.remove(cp)
    done = {}
    if os.path.exists(cp):
        for l in open(cp):
            try:
                r = json.loads(l)
                if r.get("pred") is None or r.get("err"): continue
                done[r["i"]] = r
            except: pass
    rows = load_questions(a.split, a.limit, cfg["seed"])
    todo = [(i, q) for i, q in enumerate(rows) if i not in done]
    print(f"3-agent-nomem: {len(rows)} câu, đã xong {len(done)}, còn {len(todo)} | conc={a.concurrency}", flush=True)
    ec, qc, es, llm = EmbedClient(cfg), get_qdrant(cfg), get_es(cfg), make_llm(cfg)
    fh = open(cp, "a"); lock = threading.Lock()

    def vote(preds):
        valid = [p for p in preds if p in list("ABCDE")]
        if not valid: return None
        cnt = Counter(valid)
        top = cnt.most_common(1)[0][1]
        winners = [p for p in valid if cnt[p] == top]          # tie -> ưu tiên thứ tự agent (agent 1 trước)
        for p in valid:
            if p in winners: return p
        return valid[0]

    def work(iq):
        i, q = iq
        opts = q["options"]; gold = q["answer_idx"]
        t0 = time.time()
        try:
            chunks = retrieve_rerank(cfg, q["question"], embed_client=ec, qdrant=qc, es=es)
            ctx = format_context(chunks)
            preds, tok = [], 0
            for persona in PERSONAS:                            # 3 call ĐỘC LẬP, không chia sẻ gì
                text, usage = llm.chat_usage(build_msgs(persona, q["question"], opts, ctx))
                preds.append(parse_choice(text, set(opts.keys())))
                if usage: tok += (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
            pred = vote(preds)
            r = {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "err": None,
                 "latency": round(time.time() - t0, 2), "n_tokens": tok or None, "votes": preds}
        except Exception as e:
            r = {"i": i, "gold": gold, "pred": None, "ok": False, "err": str(e)[:150],
                 "latency": round(time.time() - t0, 2), "n_tokens": None, "votes": None}
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
    json.dump(out, open(cfg["paths"]["index_dir"] + "/eval_manomem_results.json", "w"), ensure_ascii=False)
    print(f"3-agent-nomem XONG: {corr}/{len(rows)} = {corr/len(rows)*100:.2f}% | invalid {inval}", flush=True)


if __name__ == "__main__":
    main()
