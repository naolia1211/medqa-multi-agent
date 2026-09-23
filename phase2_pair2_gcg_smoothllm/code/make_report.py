#!/usr/bin/env python3
"""Turn one or more JSONL logs into the metrics table.

  python make_report.py --logs results/*.jsonl --out reports/metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from metrics import build_report, render_csv, render_markdown   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "metrics"))
    a = ap.parse_args()

    paths: list[str] = []
    for pat in a.logs:
        paths += sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
    if not paths:
        sys.exit("no log files matched")

    # NOT inside results/: the next stage globs results/*.jsonl, and a merge
    # artifact sitting there gets re-read, doubling every n.
    merged = a.out + "_merged.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(merged)), exist_ok=True)
    seen: set[tuple] = set()
    with open(merged, "w", encoding="utf-8") as out:
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    k = (r["prompt_id"], r["condition"], r["suffix_id"], r["backend"])
                    if k in seen:
                        continue
                    seen.add(k)
                    out.write(line if line.endswith("\n") else line + "\n")
    print(f"merged {len(paths)} log(s), {len(seen)} unique records")

    rep = build_report(merged)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext, text in ((".md", render_markdown(rep)),
                      (".csv", render_csv(rep)),
                      (".json", json.dumps(rep, indent=2, ensure_ascii=False))):
        with open(a.out + ext, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  -> {a.out + ext}")
    print()
    print(render_markdown(rep))


if __name__ == "__main__":
    main()
