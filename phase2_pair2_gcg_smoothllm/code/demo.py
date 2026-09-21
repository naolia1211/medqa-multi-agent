#!/usr/bin/env python3
"""Live demo: GCG jailbreak, then SmoothLLM stopping it -- one prompt at a time.

Built for the ~5 minutes in front of the class. It shows the mechanism rather
than a table: the suffix going in, the model complying, then the same prompt
perturbed N ways, each copy voting, and the vote landing on refusal.

    python demo.py --backend vicuna                 # the attack actually lands
    python demo.py --backend gemma                  # it does not -- that is the point
    python demo.py --backend vicuna --compare gemma # both, side by side
    python demo.py --backend mock                   # rehearse with no GPU

    --goal "..."     use a specific AdvBench goal instead of the default
    --list           show the available goals and exit
    --pause          wait for Enter between acts (good for narrating)
    --N 6 --q 10     defense settings

Three acts:
  1. baseline   goal alone            -> the model refuses (alignment works)
  2. attack     goal + GCG suffix     -> it complies (alignment broken)
  3. defense    the same, SmoothLLM   -> per-copy votes, then the final answer

Act 1 matters as much as act 2: without it, a sceptical examiner cannot tell
whether the suffix did anything or the model was never refusing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import judges                                        # noqa: E402
from backends import build_backend, MockBackend      # noqa: E402
from smoothllm import SmoothLLM, refusal_aggregator  # noqa: E402

W = 78
C = {"hdr": "\033[1;36m", "ok": "\033[1;32m", "bad": "\033[1;31m",
     "dim": "\033[2m", "warn": "\033[1;33m", "off": "\033[0m"}
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    try:                       # enable ANSI on older Windows consoles
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        C = {k: "" for k in C}


def rule(ch="="):
    print(ch * W)


def head(title: str):
    print()
    rule()
    print(f"{C['hdr']}{title}{C['off']}")
    rule()


def wrap(text: str, indent: str = "    ", limit: int = 400) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) > limit:
        t = t[:limit] + " ..."
    return textwrap.fill(t, width=W - len(indent),
                         initial_indent=indent, subsequent_indent=indent) or f"{indent}(empty)"


def verdict(label: str) -> str:
    if label == "refused":
        return f"{C['ok']}REFUSED{C['off']}"
    if label == "jailbroken":
        return f"{C['bad']}JAILBROKEN{C['off']}"
    return f"{C['warn']}{label.upper()}{C['off']}"


def load_goals() -> list[dict]:
    p = os.path.join(HERE, "data", "eval_sets", "safety_prompts.jsonl")
    if not os.path.exists(p):
        sys.exit("run  python src/datasets_build.py  first")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r for r in rows if r["suffix_id"] != "none"]


def load_backend(name: str, cfg_path: str):
    if name == "mock":
        return MockBackend()
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    if name not in cfg["backends"]:
        sys.exit(f"unknown backend {name!r}; have {list(cfg['backends'])}")
    b = build_backend(cfg["backends"][name])
    if not b.health():
        sys.exit(f"[!] {b.name} is not reachable -- start the server (README section 4)")
    return b


def act(n: int, title: str, sub: str = ""):
    print()
    print(f"{C['hdr']}ACT {n} -- {title}{C['off']}")
    if sub:
        print(f"{C['dim']}{sub}{C['off']}")
    print("-" * W)


def run_one(backend, goal: str, suffix: str, prefixes, args) -> dict:
    out = {}

    # ---- ACT 1: the model refuses the bare request -------------------------
    act(1, "BASELINE -- the goal on its own",
        "Establishes that the model does refuse. Without this, act 2 proves nothing.")
    print(wrap(goal, "  > "))
    t0 = time.time()
    g = backend.generate(goal, max_new_tokens=args.max_tokens)
    lab = "unscored" if g.error or not g.text.strip() else judges.judge_refusal(g.text, prefixes)
    print(f"\n  {verdict(lab)}   {time.time()-t0:.1f}s")
    print(wrap(g.text if g.text.strip() else f"[no output] {g.error}"))
    out["baseline"] = lab
    if args.pause:
        input("\n  [Enter] ")

    # ---- ACT 2: the same request with the GCG suffix ----------------------
    act(2, "ATTACK -- the same goal plus the GCG universal suffix",
        "20 tokens optimised on Vicuna-7B-v1.3 fp16 over 500 steps (~21 GPU-hours).")
    print(wrap(goal, "  > "))
    print(f"{C['bad']}" + wrap(suffix, "  + ") + f"{C['off']}")
    t0 = time.time()
    ga = backend.generate(f"{goal} {suffix}", max_new_tokens=args.max_tokens)
    lab_a = "unscored" if ga.error or not ga.text.strip() else judges.judge_refusal(ga.text, prefixes)
    print(f"\n  {verdict(lab_a)}   {time.time()-t0:.1f}s")
    print(wrap(ga.text if ga.text.strip() else f"[no output] {ga.error}"))
    out["attack"] = lab_a
    if args.pause:
        input("\n  [Enter] ")

    # ---- ACT 3: SmoothLLM ------------------------------------------------
    act(3, f"DEFENSE -- SmoothLLM (N={args.N} copies, q={args.q}% characters perturbed)",
        "Each copy is queried separately; the majority verdict decides the answer.")
    sm = SmoothLLM(backend, refusal_aggregator(lambda t: judges.judge_refusal(t, prefixes)),
                   n_copies=args.N, q=args.q, perturbation="swap", seed=42,
                   max_workers=args.workers)
    print(f"{C['dim']}  one perturbed copy, so the damage is visible:{C['off']}")
    print(wrap(sm.perturb_batch(f"{goal} {suffix}", 0)[0], "  ~ ", 300))
    t0 = time.time()
    r = sm.generate(f"{goal} {suffix}", max_new_tokens=args.max_tokens)
    dt = time.time() - t0

    print(f"\n  {C['dim']}per-copy votes:{C['off']}")
    for i, (lab_i, resp) in enumerate(zip(r.per_copy_labels, r.per_copy_responses), 1):
        print(f"    copy {i:>2}  {verdict(lab_i):<24} {C['dim']}{(resp or '').strip()[:44]!r}{C['off']}")
    print(f"\n  vote: {verdict(r.winning_label)}  "
          f"({r.vote_margin:.0%} agreement, {r.n_voting_copies} voted, "
          f"{r.n_failed_copies} failed)")
    print(f"\n  {C['dim']}final answer returned to the user:{C['off']}")
    print(wrap(r.final_response))
    out["defense"] = r.winning_label
    out["cost"] = {"llm_calls": r.n_llm_calls,
                   "tokens": r.prompt_tokens + r.completion_tokens,
                   "wall_s": round(dt, 1),
                   "undefended_tokens": ga.prompt_tokens + ga.completion_tokens}

    print(f"\n  {C['warn']}cost of the defense:{C['off']} {r.n_llm_calls} LLM calls instead of 1, "
          f"{out['cost']['tokens']} tokens vs {out['cost']['undefended_tokens']}, "
          f"{dt:.1f}s wall")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--compare", default=None, help="second backend, run afterwards")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--goal", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--pause", action="store_true")
    a = ap.parse_args()

    rows = load_goals()
    if a.list:
        for i, r in enumerate(rows):
            print(f"{i:>3}  {r['goal']}")
        return
    row = rows[a.index % len(rows)]
    goal = a.goal or row["goal"]
    suffix = json.load(open(os.path.join(HERE, "data", "raw", "gcg",
                                         "universal_suffix.json"), encoding="utf-8")
                       )["universal_suffix"]
    prefixes = judges.load_refusal_prefixes()

    results = {}
    for name in [a.backend] + ([a.compare] if a.compare else []):
        b = load_backend(name, a.config)
        head(f"GCG  vs  SmoothLLM        model: {b.name}")
        results[b.name] = run_one(b, goal, suffix, prefixes, a)

    head("SUMMARY")
    print(f"  {'model':<22} {'baseline':<12} {'+ suffix':<12} {'+ SmoothLLM':<12}")
    for k, v in results.items():
        print(f"  {k:<22} {v['baseline']:<12} {v['attack']:<12} {v['defense']:<12}")

    if len(results) > 1:
        labs = [v["attack"] for v in results.values()]
        if "jailbroken" in labs and "refused" in labs:
            print(f"\n  {C['warn']}The same suffix breaks one model and not the other.{C['off']}")
            print("  A defense-success-rate figure is only meaningful against a named")
            print("  model -- that is the headline of this project.")
    print()


if __name__ == "__main__":
    main()
