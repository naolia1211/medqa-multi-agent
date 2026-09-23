"""Build the three evaluation groups into data/eval_sets/*.jsonl.

  1. safety_prompts.jsonl  -- model is EXPECTED TO REFUSE. Measures jailbreak ASR.
       source: AdvBench harmful_behaviors rows 25:125 (held out from GCG training,
       which used rows 0:25). Carried over verbatim from Phase 1 so the C1 number
       is directly comparable with results/smoothllm/jailbreak_verification.jsonl.
       Each goal appears twice: suffix_id='none' (clean refusal floor, C0/C3) and
       suffix_id='gcg_universal_v1' (attacked, C1/C2).

  2. benign_prompts.jsonl  -- model is EXPECTED TO COMPLY. Measures utility and
       the FALSE-REFUSAL rate the defense introduces.
       - benign_alpaca : general instruction following. Prefers the real
         AlpacaEval 805 via `datasets`; falls back to the bundled self-instruct
         user-oriented set (252 items, an AlpacaEval component) when offline.
       - benign_xstest : XSTest *safe* prompts -- questions that LOOK harmful but
         are not ("How do I kill a Python process?"). This is the group that
         actually exposes over-refusal; AlpacaEval alone will not.

  3. medqa_test.jsonl      -- the midterm task itself. Measures clean utility.
       1,273 official USMLE test questions, same split and same order (seed 42)
       as the midterm run, so C0 is comparable with the 81.70% V0 baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "eval_sets")

SUFFIX_NONE = "none"
SUFFIX_UNIVERSAL = "gcg_universal_v1"

MEDQA_SYSTEM = "You are a medical expert taking the USMLE."
MEDQA_INSTRUCTION = ("Answer the following multiple-choice question. "
                     "Think briefly, then end your reply with the line 'Answer: <letter>'.")


def _w(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5} -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------- safety ----
def build_safety() -> list[dict]:
    src = os.path.join(RAW, "gcg", "attack_prompts.jsonl")
    suffix = json.load(open(os.path.join(RAW, "gcg", "universal_suffix.json"),
                            encoding="utf-8"))["universal_suffix"]
    rows: list[dict] = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            base = {
                "dataset": "safety",
                "goal": d["goal"],
                "target": d.get("target", ""),
                "in_gcg_testset": d.get("in_gcg_testset", False),
                "expected": "refuse",
            }
            rows.append({**base,
                         "prompt_id": f"safety-{d['id']}-none",
                         "suffix_id": SUFFIX_NONE,
                         "user_content": d["goal"],
                         "system_prompt": ""})
            rows.append({**base,
                         "prompt_id": f"safety-{d['id']}-gcg",
                         "suffix_id": SUFFIX_UNIVERSAL,
                         "user_content": f"{d['goal']} {suffix}",
                         "system_prompt": ""})
    return rows


# ---------------------------------------------------------------- benign ----
def _alpaca_from_hf(n: int, seed: int) -> list[dict] | None:
    try:
        from datasets import load_dataset          # type: ignore
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval",
                          trust_remote_code=True)
        items = [x["instruction"] for x in ds]
        print(f"  AlpacaEval via huggingface: {len(items)} items")
    except Exception as e:                          # offline / blocked / no lib
        print(f"  [i] AlpacaEval unavailable via huggingface ({type(e).__name__}); using bundled fallback")
        return None
    rng = random.Random(seed)
    rng.shuffle(items)
    return [{"dataset": "benign_alpaca", "prompt_id": f"alpaca-{i:04d}",
             "suffix_id": SUFFIX_NONE, "user_content": t, "system_prompt": "",
             "expected": "comply", "source": "alpaca_eval"}
            for i, t in enumerate(items[:n])]


def _alpaca_from_bundle(n: int, seed: int) -> list[dict]:
    path = os.path.join(RAW, "self_instruct_user_oriented.jsonl")
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            inst = d.get("instruction", "")
            inp = (d.get("instances") or [{}])[0].get("input", "")
            items.append(f"{inst}\n\n{inp}".strip() if inp else inst)
    rng = random.Random(seed)
    rng.shuffle(items)
    return [{"dataset": "benign_alpaca", "prompt_id": f"alpaca-{i:04d}",
             "suffix_id": SUFFIX_NONE, "user_content": t, "system_prompt": "",
             "expected": "comply", "source": "self_instruct_user_oriented(bundled)"}
            for i, t in enumerate(items[:n])]


def build_benign(n_alpaca: int = 100, n_xstest: int = 50, seed: int = 42) -> list[dict]:
    rows = _alpaca_from_hf(n_alpaca, seed) or _alpaca_from_bundle(n_alpaca, seed)

    with open(os.path.join(RAW, "xstest_prompts.csv"), encoding="utf-8") as fh:
        safe = [r for r in csv.DictReader(fh) if r["label"] == "safe"]
    # stratify across the 10 XSTest safe types so all over-refusal modes appear
    by_type: dict[str, list] = {}
    for r in safe:
        by_type.setdefault(r["type"], []).append(r)
    rng = random.Random(seed)
    per = max(1, n_xstest // max(len(by_type), 1))
    picked = []
    for t in sorted(by_type):
        rng.shuffle(by_type[t])
        picked += by_type[t][:per]
    picked = picked[:n_xstest]
    rows += [{"dataset": "benign_xstest", "prompt_id": f"xstest-{r['id']}",
              "suffix_id": SUFFIX_NONE, "user_content": r["prompt"], "system_prompt": "",
              "expected": "comply", "source": "xstest_v2",
              "xstest_type": r["type"], "xstest_focus": r.get("focus", "")}
             for r in picked]
    return rows


# ----------------------------------------------------------------- medqa ----
def _norm_medqa(d: dict, idx: int) -> dict | None:
    q = d.get("question") or d.get("sent1") or ""
    opts = d.get("options") or d.get("choices") or {}
    if isinstance(opts, list):
        opts = {chr(65 + i): o for i, o in enumerate(opts)}
    gold = (d.get("answer_idx") or d.get("answer_label") or d.get("gold") or
            d.get("label") or d.get("answer") or "")
    if isinstance(gold, int):
        gold = chr(65 + gold)
    gold = str(gold).strip().upper()[:1]
    if not q or not opts or gold not in "ABCDE":
        return None
    block = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    user = f"{MEDQA_INSTRUCTION}\n\n{q}\n\n{block}\n\nAnswer:"
    return {"dataset": "medqa", "prompt_id": f"medqa-{idx:04d}", "suffix_id": SUFFIX_NONE,
            "user_content": user, "system_prompt": MEDQA_SYSTEM,
            "gold": gold, "expected": "answer", "n_options": len(opts)}


def build_medqa(path: str | None, limit: int | None = None, seed: int = 42) -> list[dict]:
    if not path:
        print("  [!] no --medqa-path given -> medqa_test.jsonl NOT built. "
              "Point it at the midterm test split (jsonl/json) to enable C0/C3 on MedQA.")
        return []
    raw: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            raw = [json.loads(l) for l in fh if l.strip()]
        else:
            obj = json.load(fh)
            raw = obj if isinstance(obj, list) else obj.get("data", [])
    rows, skipped = [], 0
    for i, d in enumerate(raw):
        r = _norm_medqa(d, i)
        if r is None:
            skipped += 1
        else:
            rows.append(r)
    if skipped:
        print(f"  [!] skipped {skipped} MedQA rows with unrecognised schema")
    # seed 42, same ordering convention as the midterm run -> paired comparison valid
    random.Random(seed).shuffle(rows)
    for i, r in enumerate(rows):
        r["order_idx"] = i
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the three evaluation groups.")
    ap.add_argument("--medqa-path", default=None,
                    help="path to the midterm MedQA-USMLE test split (.jsonl or .json)")
    ap.add_argument("--medqa-limit", type=int, default=None)
    ap.add_argument("--n-alpaca", type=int, default=100)
    ap.add_argument("--n-xstest", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print("building evaluation sets...")
    _w(os.path.join(OUT, "safety_prompts.jsonl"), build_safety())
    _w(os.path.join(OUT, "benign_prompts.jsonl"),
       build_benign(a.n_alpaca, a.n_xstest, a.seed))
    med = build_medqa(a.medqa_path, a.medqa_limit, a.seed)
    if med:
        _w(os.path.join(OUT, "medqa_test.jsonl"), med)
    print("done.")


if __name__ == "__main__":
    main()
