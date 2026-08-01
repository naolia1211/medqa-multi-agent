"""
eval_v2.py — Variant V2 (spec docs/V3_PIPELINE_SPEC.md §): RAG + 3 agent, KHÔNG memory.
Khác V3 DUY NHẤT ở Aggregator: V2 Aggregator MÙ bằng chứng — chỉ thấy ứng viên (A1) +
phản biện (A2) chuyển tiếp, KHÔNG thấy evidence gốc (không có "nháp chung"/short-term).
Không long-term (lessons=""). Reasoning + Verifier tái dùng NGUYÊN của agents.py (=V3).
=> cô lập "short-term memory" (Aggregator thấy evidence) = khoảng cách V3 − V2.
Record {i,gold,pred,ok,err,latency,n_tokens,trace} tương thích metrics.py. Checkpoint/resume.
"""
import os, sys, time, json, argparse, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
from common import load_config, make_llm, EmbedClient, get_qdrant, get_es
from eval import load_questions
from retrieval import retrieve_rerank
from agents import (init_state, run_reasoning, run_verifier, extract_answer,
                    format_options, _call, parse_json_loose, _dumps, make_token_counter, _provider_of)
from tracking import Tracker, tag_trace, span as trace_span, set_trace_tags

# Aggregator V2: KHÔNG có EVIDENCE. Chỉ nhận ứng viên A1 + phản biện A2 (đúng hình/spec §).
AGGREGATOR_V2_PROMPT = """You are an attending physician answering a USMLE multiple-choice question.
You do NOT have access to the source evidence. You must decide based ONLY on your two colleagues' analysis below.

QUESTION:
{question}

OPTIONS:
{options}

Agent 1 (reasoning) proposed these candidate answers ranked high->low, with reasoning:
Candidates: {candidates}
Reasoning: {reasoning}

Agent 2 (reviewer) raised per-option concerns and an overall critique:
Concerns: {concerns}
Critique: {critique}

Weigh the above and commit to EXACTLY ONE option in {letters}. You must choose; do not leave it blank.
Return pure JSON:
{{"answer": "C", "explanation": "..."}}"""


def v2_aggregate(state, llm, counter, tracker=None):
    a1 = state["trace"][0]["output"]
    a2 = state["trace"][1]["output"]
    a1 = a1 if isinstance(a1, dict) else {"candidates": str(a1), "reasoning": str(a1)}
    a2 = a2 if isinstance(a2, dict) else {"concerns": str(a2), "critique": str(a2)}
    prompt = AGGREGATOR_V2_PROMPT.format(
        question=state["question"], options=format_options(state["options"]),
        candidates=_dumps(a1.get("candidates", "")), reasoning=_dumps(a1.get("reasoning", "")),
        concerns=_dumps(a2.get("concerns", "")), critique=_dumps(a2.get("critique", "")),
        letters=", ".join(state["options"].keys()),
    )
    with trace_span(tracker, "aggregator", inputs={"messages": [{"role": "user", "content": prompt}]}) as sp:
        raw, tel = _call(llm, prompt, counter)
        out = parse_json_loose(raw, required_keys=["answer"])
        status = "ok" if out else "parse_fail"
        state["trace"].append({"step": 3, "agent": "aggregator_v2", "raw": raw,
                               "output": out if out is not None else raw, "status": status, **tel})
        sp.set_outputs({"response": raw, "parsed": out})
        sp.set_attributes({"agent": "aggregator_v2", "status": status,
                           "latency_s": tel["latency"], "tokens_src": tel.get("tokens_src")})
        sp.set_model(getattr(llm, "model", None), _provider_of(llm))
        sp.set_token_usage(tel["tokens_in"], tel["tokens_out"])
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1273)
    ap.add_argument("--split", default="MedQA-USMLE/questions/US/test.jsonl")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cp = os.path.join(cfg["paths"]["checkpoint"],
                      f"v2_{os.path.basename(a.split).replace('.jsonl','')}_{a.limit}.jsonl")
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
    print(f"V2 (3-agent, no-mem, aggregator mù evidence): {len(rows)} câu, xong {len(done)}, còn {len(todo)} | conc={a.concurrency}", flush=True)
    ec, qc, es, llm = EmbedClient(cfg), get_qdrant(cfg), get_es(cfg), make_llm(cfg)
    counter = make_token_counter(cfg) if cfg.get("embedding") else None
    tracker = Tracker(cfg)      # GenAI Tracing -> mỗi câu 1 MLflow trace (variant=v2), no-op nếu server chết
    split_name = os.path.basename(a.split)
    fh = open(cp, "a"); lock = threading.Lock()

    def work(iq):
        i, q = iq
        opts = q["options"]; gold = q["answer_idx"]
        t0 = time.time(); trace_id = None
        try:
            with trace_span(tracker, "multiagent_v2", span_type="CHAIN", root=True,
                            inputs={"question": q["question"], "options": opts},
                            attributes={"n_agents": 3, "variant": "v2"}) as root:
                with trace_span(tracker, "retrieve", span_type="RETRIEVER",
                                inputs={"query": q["question"]}) as rsp:
                    evidence = retrieve_rerank(cfg, q["question"], embed_client=ec, qdrant=qc, es=es)
                    rsp.set_outputs({"chunk_ids": [e["chunk_id"] for e in evidence], "n": len(evidence)})
                state = init_state(q["question"], opts, evidence)
                run_reasoning(state, llm, counter, tracker, lessons="")   # A1 = như V3 (không lessons)
                run_verifier(state, llm, counter, tracker, lessons="")     # A2 = như V3 (không lessons)
                raw = v2_aggregate(state, llm, counter, tracker)           # A3 = mù evidence (V2)
                pred = extract_answer(raw, valid=opts.keys())
                root.set_outputs({"choice": pred, "candidates": state["candidates"]})
                srcs = {t.get("tokens_src") for t in state["trace"]}
                set_trace_tags(tracker, {"variant": "v2", "model": getattr(llm, "model", None),
                                         "provider": (cfg.get("llm") or {}).get("provider"),
                                         "pred": pred, "memory": "off", "lessons_used": False,
                                         "tokens_src": srcs.pop() if len(srcs) == 1 else "mixed"})
                trace_id = root.trace_id
            ntok = sum((t.get("tokens_in") or 0) + (t.get("tokens_out") or 0) for t in state["trace"]) or None
            r = {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "err": None,
                 "latency": round(time.time() - t0, 2), "n_tokens": ntok, "trace": state["trace"]}
        except Exception as e:
            r = {"i": i, "gold": gold, "pred": None, "ok": False, "err": str(e)[:150],
                 "latency": round(time.time() - t0, 2), "n_tokens": None, "trace": None}
        if trace_id:      # tag sau khi trace kết thúc (giống eval_agents): split/gold/pred/correct/invalid
            inv = r["pred"] is None or str(r["pred"]).strip().upper() not in set("ABCDE")
            tag_trace(tracker, trace_id, {"split": split_name, "q_index": i, "gold": gold,
                                          "pred": str(r["pred"]), "correct": str(bool(r["ok"])),
                                          "invalid": str(inv)})
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
    json.dump(out, open(cfg["paths"]["index_dir"] + "/eval_v2_results.json", "w"), ensure_ascii=False)
    print(f"V2 XONG: {corr}/{len(rows)} = {corr/len(rows)*100:.2f}% | invalid {inval}", flush=True)


if __name__ == "__main__":
    main()
