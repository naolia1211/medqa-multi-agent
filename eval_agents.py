"""
eval_agents.py — Chấm accuracy MedQA bằng luồng MULTI-AGENT V3 (agents.answer_multiagent).
Song song với eval.py (baseline single-LLM); KHÔNG đụng eval.py.

  python eval_agents.py                     # eval.split / eval.limit trong config
  python eval_agents.py --limit 50 --split MedQA-USMLE/questions/US/dev.jsonl
  python eval_agents.py --fresh             # bỏ checkpoint, chạy lại từ đầu

Checkpoint/resume: mỗi câu xong ghi 1 dòng JSONL vào
index/.checkpoint/agents_<split>_<limit>.jsonl (fsync). Restart -> bỏ câu đã xong;
câu lỗi / pred INVALID coi như chưa xong -> retry.

Log đủ trường cho error analysis (§8): i, gold, pred, ok, invalid, latency, n_tokens_est,
evidence_chunk_ids, trace.
"""
import os
import json
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (load_config, setup_logging, resolve,
                    EmbedClient, make_llm, get_qdrant, get_es)
from agents import answer_multiagent
from tracking import Tracker, tag_trace
from eval import load_questions, load_checkpoint   # tái dùng, không sao chép

log = setup_logging("eval_agents")


def checkpoint_file(cfg, split, limit):
    cp = resolve(cfg, cfg["paths"]["checkpoint"])
    os.makedirs(cp, exist_ok=True)
    base = os.path.basename(split).replace("/", "_").replace(".jsonl", "")
    return os.path.join(cp, f"agents_{base}_{limit or 'all'}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="Bỏ checkpoint, chạy lại từ đầu.")
    args = ap.parse_args()

    cfg = load_config()
    ev = cfg["eval"]
    split = args.split or ev["split"]
    limit = args.limit if args.limit is not None else ev["limit"]
    conc = args.concurrency or ev["concurrency"]
    path = resolve(cfg, split)

    rows = load_questions(path, limit, cfg["seed"])
    cp_path = checkpoint_file(cfg, split, limit)
    if args.fresh and os.path.exists(cp_path):
        os.remove(cp_path)
        log.info("--fresh: đã xóa checkpoint cũ.")
    done = load_checkpoint(cp_path)
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]

    # Tạo clients TRƯỚC để lấy tên model THỰC TẾ theo provider (llm.model), tránh log
    # nhầm cfg['llm']['model'] — key này là model fallback OpenRouter, sai khi provider=ollama/google.
    llm = make_llm(cfg, log)
    active_model = getattr(llm, "model", cfg["llm"]["model"])
    # Adaptive concurrency: bọc LLM để tự hạ/tăng số call đồng thời theo tỉ lệ nghẽn Ollama.
    ac = (cfg["eval"].get("adaptive_concurrency") or {})
    if ac.get("enabled"):
        from throttle import AdaptiveGate, ThrottledLLM

        def _on_conc_change(old, new, rate):
            msg = f"Adaptive concurrency: {old} -> {new} (nghẽn {rate*100:.0f}%)"
            log.info(msg)
            try:
                from notify import send
                send(f"⚙️ {msg}")
            except Exception:
                pass

        gate = AdaptiveGate(max_conc=ac.get("max", conc), min_conc=ac.get("min", 1),
                            window=ac.get("window", 10), high=ac.get("high_congestion", 0.3),
                            low=ac.get("low_congestion", 0.1),
                            latency_threshold=ac.get("latency_threshold_s", 60),
                            on_change=_on_conc_change)
        llm = ThrottledLLM(llm, gate)
        log.info("Adaptive concurrency BẬT: max=%d min=%d (bắt đầu ở max)",
                 ac.get("max", conc), ac.get("min", 1))
    clients = (EmbedClient(cfg, log), get_qdrant(cfg), get_es(cfg), llm)

    # §9 Long-term memory (học từ câu sai) — no-op nếu memory.enabled=false
    from long_mem import LongMemory
    memory = LongMemory(cfg, clients[0], log)
    if memory.enabled:
        log.info("Long-term memory BẬT: %d lessons đã có · top_k=%d · save_wrong=%s",
                 len(memory._lessons), memory.top_k, memory.save_wrong_on)

    log.info("Eval V3 multi-agent %d câu từ %s (concurrency=%d, model=%s)",
             len(rows), split, conc, active_model)
    if done:
        log.info("RESUME: %d/%d câu đã có, còn %d câu.", len(done), len(rows), len(todo))

    tracker = Tracker(cfg)   # chỉ GenAI Tracing — KHÔNG tạo run/metrics kiểu model-training
    cp_lock = threading.Lock()
    cp_fh = open(cp_path, "a", encoding="utf-8")

    def persist(r):
        with cp_lock:
            cp_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            cp_fh.flush()
            os.fsync(cp_fh.fileno())

    def run_one(i, row):
        gold = row["answer_idx"]
        try:
            res = answer_multiagent(row["question"], row["options"], cfg=cfg,
                                    clients=clients, tracker=tracker, memory=memory)
            pred = res["choice"]
            r = {"i": i, "gold": gold, "pred": pred,
                 "ok": pred == gold, "invalid": pred == "INVALID", "err": None,
                 "latency": round(res["latency"], 2),
                 "n_tokens": res["n_tokens"], "tokens_real": res["tokens_real"],
                 "tokens_src": res.get("tokens_src"),
                 "lessons_used": res.get("lessons_used", False),
                 "evidence_chunk_ids": [e["chunk_id"] for e in res["evidence"]],
                 "trace": res["trace"]}
            # §9 pha HỌC: câu SAI -> lưu vào long-term memory (no-op nếu save_wrong=false)
            if not r["ok"]:
                memory.save_wrong(row["question"], row["options"], gold, pred,
                                  explanation=str(res.get("answer_text", ""))[:500],
                                  topic=row.get("meta_info", "unknown"))
            # Gắn kết quả câu này lên TRACE (thay cho metric của run) -> lọc/tính accuracy ở tab Traces.
            tag_trace(tracker, res.get("trace_id"),
                      {"split": os.path.basename(split), "q_index": i, "gold": gold,
                       "pred": pred, "correct": r["ok"], "invalid": r["invalid"],
                       "lessons_used": res.get("lessons_used", False)})
        except Exception as e:
            r = {"i": i, "gold": gold, "pred": None, "ok": False, "invalid": False,
                 "err": str(e)[:200]}
        persist(r)
        return r

    results = dict(done)
    t0 = time.time()
    processed = 0
    if todo:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(run_one, i, r): i for i, r in todo}
            for fut in as_completed(futs):
                r = fut.result()
                results[r["i"]] = r
                processed += 1
                if processed % 10 == 0 or processed == len(todo):
                    tot = len(results)
                    cur = sum(1 for x in results.values() if x["ok"])
                    log.info("  +%d/%d mới · tổng %d/%d · acc %.1f%%",
                             processed, len(todo), tot, len(rows), 100 * cur / tot)
    cp_fh.close()
    dt = time.time() - t0

    allr = [results[i] for i in range(len(rows)) if i in results]
    n = len(allr)
    correct = sum(1 for x in allr if x["ok"])
    errs = sum(1 for x in allr if x.get("err"))
    invalid = sum(1 for x in allr if x.get("invalid"))
    lat = [x["latency"] for x in allr if x.get("latency") is not None]
    toks = [x["n_tokens"] for x in allr if x.get("n_tokens") is not None]

    # Tổng hợp theo TỪNG agent (từ trace) — cho bản tóm tắt console + file JSON cục bộ.
    # (Không đẩy làm metric của MLflow run: telemetry token/agent đã nằm trong TRACE.)
    per_agent = {}
    for x in allr:
        for t in x.get("trace", []):
            a = per_agent.setdefault(t["agent"], {"calls": 0, "parse_fail": 0, "latency": 0.0,
                                                  "tokens": 0, "tokens_in": 0, "tokens_out": 0})
            a["calls"] += 1
            a["parse_fail"] += 1 if t.get("status") == "parse_fail" else 0
            a["latency"] += t.get("latency") or 0
            a["tokens_in"] += t.get("tokens_in") or 0
            a["tokens_out"] += t.get("tokens_out") or 0
            a["tokens"] += (t.get("tokens_in") or 0) + (t.get("tokens_out") or 0)

    total = len(rows)                       # OFFICIAL: mẫu số = TỔNG câu (câu thiếu/lỗi/invalid = SAI)
    missing = total - n                     # câu chưa có kết quả (run dở) -> tính là sai
    print("\n" + "=" * 56)
    print(f"ACCURACY (V3 multi-agent): {correct}/{total} = {100*correct/max(1,total):.2f}%  ({split})")
    if missing:
        print(f"  ⚠️ {missing} câu CHƯA có kết quả (tính là SAI theo official). Kết quả có: {n}/{total}")
    print(f"  lỗi LLM/mạng: {errs}   invalid response: {invalid} "
          f"({100*invalid/max(1,total):.1f}%)")
    print(f"  latency/câu TB: {(sum(lat)/len(lat) if lat else 0):.1f}s   "
          f"token/câu TB: {(sum(toks)/len(toks) if toks else 0):.0f}")
    for a, s in per_agent.items():
        print(f"    [{a}] calls={s['calls']} parse_fail={s['parse_fail']} "
              f"lat_TB={s['latency']/max(1,s['calls']):.1f}s tok_TB={s['tokens']/max(1,s['calls']):.0f}")
    print(f"  câu chạy phiên này: {processed}   thời gian phiên: {dt:.1f}s")
    print(f"  checkpoint: {cp_path}")
    print(f"  -> metric official đầy đủ (gain/W-L-T/bootstrap/McNemar): python metrics.py --v3 {out if False else 'index/eval_agents_results.json'}")
    print("=" * 56)

    out = resolve(cfg, cfg["paths"]["index_dir"]) + "/eval_agents_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": split, "n": n, "total": total, "correct": correct,
                   "accuracy": correct / max(1, total), "invalid": invalid,
                   "invalid_rate": invalid / max(1, total), "missing": missing,
                   "session_seconds": dt, "per_agent": per_agent, "results": allr},
                  f, ensure_ascii=False, indent=2)
    log.info("Ghi tổng hợp -> %s", out)
    # MỌI telemetry (token in/out native, status, latency, model, gold/pred/correct…) đã nằm
    # trong GenAI Traces (tab Traces). Không tạo run/metrics kiểu model-training nữa.
    if tracker.mlflow:
        log.info("Traces -> %s (experiment=%s). Lọc theo tag variant/split/correct để tính accuracy.",
                 cfg["mlflow"]["tracking_uri"], cfg["mlflow"]["experiment"])


if __name__ == "__main__":
    main()
