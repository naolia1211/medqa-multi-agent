#!/usr/bin/env python3
"""Run the C0-C3 evaluation sweep.

  python run_eval.py --backend vicuna --datasets safety
  python run_eval.py --backend gemma  --datasets safety benign medqa
  python run_eval.py --backend mock   --datasets safety benign --limit 20   # smoke test

Resume-safe: re-running skips (prompt_id, condition, suffix_id, backend) already
on disk, so a killed 12-hour MedQA sweep picks up where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import judges                                    # noqa: E402
from backends import MockBackend, build_backend, _gold_key  # noqa: E402
from runner import run_sweep                     # noqa: E402
from schema import JsonlLogger, read_jsonl       # noqa: E402


def load_cfg(path: str) -> dict:
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ModuleNotFoundError:
        sys.exit("pyyaml is required: pip install pyyaml")


def main() -> None:
    ap = argparse.ArgumentParser(description="C0-C3 SmoothLLM evaluation sweep")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--backend", required=True, help="a key under backends: in config.yaml, or 'mock'")
    ap.add_argument("--datasets", nargs="+", default=["safety", "benign"],
                    choices=["safety", "benign", "medqa"])
    ap.add_argument("--conditions", nargs="+", default=None, choices=["C0", "C1", "C2", "C3"])
    ap.add_argument("--limit", type=int, default=None, help="cap prompts per dataset (debug)")
    ap.add_argument("--out", default=None, help="override log path")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--N", type=int, default=None, help="override SmoothLLM copies")
    ap.add_argument("--q", type=float, default=None, help="override perturbation percent")
    ap.add_argument("--perturbation", default=None, choices=["swap", "patch", "insert"])
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    conditions = a.conditions or cfg["conditions"]
    dcfg = dict(cfg["defense"])
    for k, v in (("N", a.N), ("q", a.q), ("perturbation", a.perturbation)):
        if v is not None:
            dcfg[k] = v

    # ---- backend ----
    if a.backend == "mock":
        # Give the fake model the answer key, otherwise MedQA sits at chance and
        # the C0 -> C3 utility comparison measures nothing.
        gold_map = {}
        med_path = os.path.join(HERE, cfg["datasets"].get("medqa", ""))
        if med_path and os.path.exists(med_path):
            for r in read_jsonl(med_path):
                gold_map[_gold_key(r["user_content"])] = r.get("gold", "C")
        backend = MockBackend(gold_map=gold_map)
        print(f"[mock] answer key loaded for {len(gold_map)} MedQA items")
    else:
        if a.backend not in cfg["backends"]:
            sys.exit(f"unknown backend {a.backend!r}; have {list(cfg['backends'])}")
        backend = build_backend(cfg["backends"][a.backend])
        if not backend.health():
            sys.exit(f"[!] backend {backend.name} is not reachable. "
                     f"Start the server first (see README section 'Hosting the models').")

    # ---- records ----
    records: list[dict] = []
    for ds in a.datasets:
        path = os.path.join(HERE, cfg["datasets"][ds])
        if not os.path.exists(path):
            print(f"  [!] missing {path} -- run src/datasets_build.py first; skipping {ds}")
            continue
        rows = read_jsonl(path)
        if a.limit:
            keep, seen = [], {}
            for r in rows:                       # limit per sub-dataset, keep both suffix arms
                k = r["dataset"]
                seen[k] = seen.get(k, 0) + 1
                if seen[k] <= a.limit * (2 if k == "safety" else 1):
                    keep.append(r)
            rows = keep
        records += rows
    if not records:
        sys.exit("no records loaded")

    run_id = cfg["run"].get("run_id") or uuid.uuid4().hex[:8]
    out = a.out or os.path.join(HERE, cfg["run"]["log_dir"],
                                f"eval_{backend.name.replace('/', '_')}.jsonl")
    print(f"backend={backend.name}  datasets={a.datasets}  conditions={conditions}")
    print(f"defense: N={dcfg['N']} q={dcfg['q']}% {dcfg['perturbation']} "
          f"placement={dcfg.get('placement')}")
    print(f"log -> {out}")

    with JsonlLogger(out, resume=not a.no_resume) as logger:
        n = run_sweep(records, conditions, backend, logger, dcfg,
                      prefixes=judges.load_refusal_prefixes(),
                      max_new_tokens=cfg["run"]["max_new_tokens"],
                      workers=a.workers or cfg["run"]["workers"],
                      run_id=run_id,
                      progress_every=cfg["run"].get("progress_every", 25))
    print(f"\ndone: {n} new records -> {out}")
    print(f"next: python make_report.py --logs {out}")


if __name__ == "__main__":
    main()
