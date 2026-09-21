#!/usr/bin/env python3
"""Sensitivity sweep over SmoothLLM's two knobs, q and N.

SmoothLLM has exactly one trade-off dial and the report needs its shape, not a
single operating point. Larger q breaks the suffix harder but destroys more of a
benign prompt; larger N buys a more reliable vote at a linear cost in queries.

  python sweep.py --backend mock --q 5 10 15 20 --N 2 4 6 8 10 --datasets safety benign

Writes one log per (q, N) cell and a combined CSV/Markdown you can plot. Pick the
operating point from this curve and say in the report why -- "we used the paper's
defaults" is a weaker answer than "q=10 is where residual ASR crosses below X
while MedQA loss stays under Y".
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from metrics import build_report      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--config", default=None,
                    help="passed through to run_eval.py (e.g. a box-specific config)")
    ap.add_argument("--datasets", nargs="+", default=["safety", "benign"])
    ap.add_argument("--q", nargs="+", type=float, default=[5, 10, 15, 20])
    ap.add_argument("--N", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    ap.add_argument("--perturbation", nargs="+", default=["swap"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "sweep"))
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    cells = list(itertools.product(a.perturbation, a.q, a.N))
    for i, (pert, q, N) in enumerate(cells, 1):
        tag = f"{a.backend}_{pert}_q{q:g}_N{N}"
        log = os.path.join(a.outdir, f"{tag}.jsonl")
        print(f"\n[{i}/{len(cells)}] {pert} q={q:g}% N={N}")
        cmd = [sys.executable, os.path.join(HERE, "run_eval.py"),
               "--backend", a.backend, "--datasets", *a.datasets,
               "--N", str(N), "--q", str(q), "--perturbation", pert,
               "--out", log, "--workers", str(a.workers)]
        if a.config:
            cmd += ["--config", a.config]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [ERR] {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'failed'}")
            continue
        rep = build_report(log)
        for bk, h in rep["headline"].items():
            rows.append({"backend": bk, "perturbation": pert, "q": q, "N": N,
                         "ASR_C1": h["ASR_C1_undefended"], "ASR_C2": h["ASR_C2_smoothllm"],
                         "defense_success_rate": h["defense_success_rate"],
                         "residual_ASR": h["residual_ASR"],
                         "medqa_acc_C0": h["medqa_acc_C0"], "medqa_acc_C3": h["medqa_acc_C3"],
                         "medqa_drop_pp": h["medqa_acc_drop_pp"],
                         "false_refusal_C0": h["false_refusal_C0"],
                         "false_refusal_C3": h["false_refusal_C3"],
                         "false_refusal_inc_pp": h["false_refusal_increase_pp"],
                         "llm_call_mult": h["llm_call_multiplier"],
                         "token_mult": h["token_cost_multiplier"]})
            print(f"  ASR {h['ASR_C1_undefended']} -> {h['ASR_C2_smoothllm']}  |  "
                  f"false-refusal +{h['false_refusal_increase_pp']} pp  |  "
                  f"MedQA {h['medqa_acc_drop_pp']} pp")

    if not rows:
        sys.exit("sweep produced nothing")

    import csv
    csv_path = os.path.join(a.outdir, "sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# SmoothLLM sensitivity sweep", "",
          "| pert | q | N | ASR C1 | ASR C2 | defense SR | residual ASR | "
          "false-refusal + | MedQA drop | calls x |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, u="": "--" if v is None else f"{v:g}{u}"
    for r in rows:
        md.append(f"| {r['perturbation']} | {r['q']:g}% | {r['N']} | {f(r['ASR_C1'],'%')} | "
                  f"{f(r['ASR_C2'],'%')} | {f(r['defense_success_rate'],'%')} | "
                  f"{f(r['residual_ASR'],'%')} | {f(r['false_refusal_inc_pp'],' pp')} | "
                  f"{f(r['medqa_drop_pp'],' pp')} | {f(r['llm_call_mult'],'x')} |")
    md_path = os.path.join(a.outdir, "sweep_summary.md")
    open(md_path, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\n-> {csv_path}\n-> {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
