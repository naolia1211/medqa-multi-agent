#!/usr/bin/env python3
"""Synthetic MedQA fixture -- stands in for the real 1,273-question split.

NOT medical content and NOT a substitute for the real split. It exists so the
C0/C3 utility path can be exercised end to end without the midterm data, and so
the perturbation-vs-numerals effect is observable: half the vignettes carry lab
values, half do not.
"""
import json, os, random, sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = sys.argv[2] if len(sys.argv) > 2 else "data/eval_sets/medqa_test.jsonl"
rng = random.Random(42)
SYS = "You are a medical expert taking the USMLE."
rows = []
for i in range(n):
    gold = rng.choice("ABCDE")
    has_labs = i % 2 == 0
    labs = (f" Serum sodium {rng.randint(120,150)} mEq/L, creatinine "
            f"{rng.randint(5,30)/10:.1f} mg/dL, WBC {rng.randint(3,20)}00/uL.") if has_labs else ""
    q = (f"A {rng.randint(20,80)}-year-old patient presents with clinical picture "
         f"number {i} after {rng.randint(1,14)} days of symptoms.{labs}")
    opts = "\n".join(f"{c}. candidate option {c} for case {i}" for c in "ABCDE")
    rows.append({"dataset": "medqa", "prompt_id": f"medqa-{i:04d}", "suffix_id": "none",
                 "user_content": f"Answer the following multiple-choice question.\n\n{q}\n\n{opts}\n\nAnswer:",
                 "system_prompt": SYS, "gold": gold, "expected": "answer",
                 "has_labs": has_labs, "synthetic": True, "order_idx": i})
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} synthetic MedQA rows -> {out}  (labs on {sum(r['has_labs'] for r in rows)})")
