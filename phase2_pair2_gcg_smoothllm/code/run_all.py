#!/usr/bin/env python3
"""One command for the whole Phase-2 run, once a model server is up.

  python run_all.py --backend vicuna --medqa-path /path/to/medqa/test.jsonl
  python run_all.py --backend gemma  --medqa-path ...  --skip-sweep
  python run_all.py --backend both   --medqa-path ...

Order matters and is enforced:
  0. preflight   -- server reachable, eval sets present, suffix matches Phase 1
  1. datasets    -- build the three groups (skipped if already built)
  2. safety      -- C0-C3 on the safety set. CHEAP. Confirms C1 ~ 1.00 before
                    committing to the expensive runs.
  3. benign      -- C0/C3, the false-refusal bill
  4. medqa       -- C0/C3, the utility bill. SLOW: C3 is N x the C0 cost.
  5. report      -- metrics table
  6. rescore     -- strict judge; run before quoting any ASR
  7. sweep       -- q/N curve (optional, --skip-sweep)

Every stage is resume-safe: re-running after an interruption picks up where it
stopped. Stop after stage 2 if C1 is not what Phase 1 measured -- something is
wrong with the model or template and the later stages would waste hours.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(desc: str, cmd: list[str], required: bool = True) -> bool:
    print(f"\n{'=' * 70}\n>> {desc}\n{'=' * 70}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n[FAILED after {dt:.0f}s] {desc}")
        if required:
            print("Stopping. Fix this before continuing -- later stages depend on it.")
            sys.exit(r.returncode)
        return False
    print(f"[ok, {dt:.0f}s]")
    return True


def preflight(backend: str, cfg: str) -> None:
    print(f"{'=' * 70}\n>> 0. preflight\n{'=' * 70}")
    sys.path.insert(0, os.path.join(HERE, "src"))
    import json
    import yaml
    from backends import build_backend
    c = yaml.safe_load(open(os.path.join(HERE, cfg), encoding="utf-8"))
    b = build_backend(c["backends"][backend])
    if not b.health():
        sys.exit(f"[!] {b.name} is not reachable at "
                 f"{c['backends'][backend].get('base_url')}\n"
                 f"    Start the server first -- see README section 4.")
    print(f"  backend {b.name}: reachable")
    g = b.generate("Say the word ok.", max_new_tokens=16)
    if g.error:
        sys.exit(f"[!] a trial generation failed: {g.error}")
    print(f"  trial generation: {g.text.strip()[:60]!r}")
    if g.prompt_tokens == 0 and g.completion_tokens == 0:
        print("  [!] the server reported no token counts -- cost figures will be zero.")
    else:
        print(f"  token accounting: {g.prompt_tokens} in / {g.completion_tokens} out")

    suf = json.load(open(os.path.join(HERE, "data", "raw", "gcg",
                                      "universal_suffix.json"), encoding="utf-8"))
    sp = os.path.join(HERE, "data", "eval_sets", "safety_prompts.jsonl")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        atk = [r for r in rows if r["suffix_id"] != "none"]
        ok = all(suf["universal_suffix"] in r["user_content"] for r in atk)
        print(f"  safety set: {len(rows)} rows, {len(atk)} attacked, "
              f"suffix matches Phase 1: {ok}")
        if not ok:
            sys.exit("[!] the attacked prompts do not carry the Phase-1 suffix.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["vicuna", "gemma", "both", "mock"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--medqa-path", default=None,
                    help="the midterm MedQA test split; without it MedQA stages are skipped")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-medqa", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    backends = ["vicuna", "gemma"] if a.backend == "both" else [a.backend]

    # 1. datasets
    ds_cmd = [PY, "src/datasets_build.py"]
    if a.medqa_path:
        ds_cmd += ["--medqa-path", a.medqa_path]
    run("1. build evaluation sets", ds_cmd)

    have_medqa = (not a.skip_medqa) and os.path.exists(
        os.path.join(HERE, "data", "eval_sets", "medqa_test.jsonl"))
    if not have_medqa:
        print("\n[i] MedQA stages will be skipped (no test split built).")

    for bk in backends:
        if bk != "mock":
            preflight(bk, a.config)

        base = [PY, "run_eval.py", "--config", a.config, "--backend", bk,
                "--workers", str(a.workers)]
        run(f"2. [{bk}] safety set, C0-C3  (cheap -- verify C1 here before going on)",
            base + ["--datasets", "safety"])
        run(f"3. [{bk}] benign set, C0/C3  (false-refusal cost)",
            base + ["--datasets", "benign"])
        if have_medqa:
            run(f"4. [{bk}] MedQA, C0/C3  (SLOW -- C3 costs N x C0)",
                base + ["--datasets", "medqa"])

    run("5. metrics table", [PY, "make_report.py", "--logs", "results/*.jsonl",
                             "--out", "reports/metrics"])
    run("6. strict-judge re-scoring  (READ THIS before quoting any ASR)",
        [PY, "rescore.py", "--logs", "results/*.jsonl",
         "--out", "results/rescored.jsonl"])

    if not a.skip_sweep:
        for bk in backends:
            run(f"7. [{bk}] q/N sensitivity sweep",
                [PY, "sweep.py", "--backend", bk, "--datasets", "safety", "benign",
                 "--q", "5", "10", "15", "20", "--N", "2", "4", "6", "10",
                 "--workers", str(a.workers)], required=False)

    print(f"\n{'=' * 70}")
    print("done. Outputs:")
    print("  reports/metrics.md        the table")
    print("  results/sweep/            the q/N curve")
    print("\nBefore writing the report, check:")
    print("  - residual ASR (absolute), not just defense success rate")
    print("  - the strict-judge output from stage 6 -- how much of the drop is")
    print("    real refusal versus the model being noised into gibberish")
    print("  - unscored_rate in the cell table; anything above ~1% means calls")
    print("    were failing and those rows are excluded from every rate")


if __name__ == "__main__":
    main()
