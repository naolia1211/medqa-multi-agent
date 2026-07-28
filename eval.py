"""
eval.py — Chấm accuracy trắc nghiệm MedQA (online RAG, không HyDE).
  python eval.py                       # dùng eval.split / eval.limit trong config
  python eval.py --limit 0 --split MedQA-USMLE/questions/US/test.jsonl   # 0 = toàn bộ
  python eval.py --fresh               # bỏ checkpoint, chạy lại từ đầu

Mỗi câu: retrieve+rerank -> LLM chọn đáp án -> so với answer_idx.
Chạy song song (LLM là bottleneck mạng); key OpenRouter xoay vòng tự động.

CHECKPOINT/RESUME: mỗi câu xong ghi ngay 1 dòng JSONL vào
index/.checkpoint/eval_<split>_<limit>.jsonl. Restart -> bỏ qua câu đã làm.
Chỉ số câu (index) ổn định theo (split, limit, seed) nên resume an toàn.
"""
import os
import json
import time
import random
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import (load_config, setup_logging, resolve,
                    EmbedClient, make_llm, get_qdrant, get_es)
from rag import answer

log = setup_logging("eval")


def load_questions(path, limit, seed):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def checkpoint_file(cfg, split, limit):
    cp = resolve(cfg, cfg["paths"]["checkpoint"])
    os.makedirs(cp, exist_ok=True)
    base = os.path.basename(split).replace("/", "_").replace(".jsonl", "")
    return os.path.join(cp, f"eval_{base}_{limit or 'all'}.jsonl")


def load_checkpoint(path, retry_errors=True):
    """
    Đọc checkpoint. Câu LỖI (err != None) coi như CHƯA xong nếu retry_errors=True
    -> resume sẽ thử lại (rate-limit tạm thời có thể đã reset).
    Nếu 1 câu xuất hiện nhiều lần (retry nhiều phiên), bản ghi SAU đè bản trước.
    """
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue   # dòng ghi dở khi bị kill -> bỏ
                # lỗi HOẶC không parse được đáp án (pred=None) -> coi như chưa xong, retry
                if retry_errors and (r.get("err") or r.get("pred") is None):
                    done.pop(r["i"], None)
                else:
                    done[r["i"]] = r
    return done


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

    log.info("Eval %d câu từ %s (concurrency=%d, model=%s)",
             len(rows), split, conc, cfg["llm"]["model"])
    if done:
        log.info("RESUME: %d/%d câu đã có trong checkpoint, còn %d câu.",
                 len(done), len(rows), len(todo))
    if not todo:
        log.info("Tất cả câu đã xong trong checkpoint — chỉ tổng hợp lại.")

    ec = EmbedClient(cfg, log)
    qc = get_qdrant(cfg)
    es = get_es(cfg)
    llm = make_llm(cfg, log)
    clients = (ec, qc, es, llm)

    cp_lock = threading.Lock()
    cp_fh = open(cp_path, "a", encoding="utf-8")

    def persist(r):
        with cp_lock:
            cp_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            cp_fh.flush()
            os.fsync(cp_fh.fileno())   # bền vững kể cả khi bị kill ngay sau

    def run_one(i, row):
        q = row["question"]
        options = row["options"]
        gold = row["answer_idx"]
        t0 = time.time()
        try:
            res = answer(q, options=options, cfg=cfg, clients=clients)
            pred = res["choice"]
            r = {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "err": None,
                 "latency": round(time.time() - t0, 2)}         # per-câu latency (cost/latency official)
        except Exception as e:
            r = {"i": i, "gold": gold, "pred": None, "ok": False, "err": str(e)[:150],
                 "latency": round(time.time() - t0, 2)}
        persist(r)
        return r

    results = dict(done)   # i -> result
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
    total = len(rows)                       # OFFICIAL: mẫu số = TỔNG câu (câu thiếu/lỗi/invalid = SAI)
    correct = sum(1 for x in allr if x["ok"])
    errs = sum(1 for x in allr if x["err"])
    # invalid = không rút được đáp án hợp lệ (pred None/không ∈ A–E) — nhất quán với V3 & metrics.py
    invalid = sum(1 for x in allr if x["pred"] is None or str(x["pred"]).strip().upper() not in set("ABCDE"))
    missing = total - n

    print("\n" + "=" * 56)
    print(f"ACCURACY (V0 RAG-only): {correct}/{total} = {100*correct/max(1,total):.2f}%  ({split})")
    if missing:
        print(f"  ⚠️ {missing} câu CHƯA có kết quả (tính là SAI theo official). Có: {n}/{total}")
    print(f"  invalid response: {invalid} ({100*invalid/max(1,total):.1f}%)   lỗi LLM/mạng: {errs}")
    print(f"  câu chạy phiên này: {processed}   thời gian phiên: {dt:.1f}s")
    print(f"  checkpoint: {cp_path}")
    print("=" * 56)

    out = resolve(cfg, cfg["paths"]["index_dir"]) + "/eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": split, "n": n, "total": total, "correct": correct,
                   "accuracy": correct / max(1, total), "invalid": invalid,
                   "invalid_rate": invalid / max(1, total), "missing": missing,
                   "session_seconds": dt, "results": allr}, f, ensure_ascii=False, indent=2)
    log.info("Ghi tổng hợp -> %s", out)


if __name__ == "__main__":
    main()
