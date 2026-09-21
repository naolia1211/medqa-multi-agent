#!/usr/bin/env python3
"""Why is the model returning an empty string? Dump the raw response and find out.

step1 recorded 200 empty completions. Before spending any more GPU time we need
the actual response shape, not a guess. This makes a handful of single calls with
different settings and prints the FULL JSON each time.

Most likely cause: gemma4 is a reasoning model. Ollama puts the chain of thought
in message.thinking and the answer in message.content -- so a small num_predict
budget gets consumed entirely by thinking and content comes back "".

    python diagnose_ollama.py

Roughly 5-8 minutes: a few calls at this machine's ~67 s/call.
"""
from __future__ import annotations

import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), time.time() - t0, ""
    except Exception as e:
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def show(tag, d, dt, err):
    print(f"\n{'-'*70}\n### {tag}   ({dt:.1f}s)")
    if err:
        print(f"  ERROR: {err}")
        return None
    msg = d.get("message", {}) if isinstance(d, dict) else {}
    content = msg.get("content", d.get("response", "")) if isinstance(d, dict) else ""
    thinking = msg.get("thinking", "")
    print(f"  top-level keys : {sorted(d.keys())}")
    if msg:
        print(f"  message keys   : {sorted(msg.keys())}")
    print(f"  content  len={len(content):<5} {content[:220]!r}")
    if thinking:
        print(f"  THINKING len={len(thinking):<5} {thinking[:220]!r}")
        print("  >>> This is a REASONING model. The answer lives in message.content,")
        print("      and the token budget is being spent on message.thinking first.")
    print(f"  done_reason    : {d.get('done_reason')}")
    print(f"  prompt_eval    : {d.get('prompt_eval_count')}   eval: {d.get('eval_count')}")
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    model = a.model
    if not model:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=15) as r:
            names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
        if not names:
            sys.exit("[!] no models installed")
        model = names[0]
    print(f"model: {model}\nbase : {base}")

    Q = "What is the capital of France? Answer in one short sentence."
    results = {}

    d, dt, e = post(f"{base}/api/chat", {
        "model": model, "messages": [{"role": "user", "content": Q}],
        "stream": False, "options": {"temperature": 0, "num_predict": 128, "seed": 42}})
    results["A chat, num_predict=128 (what step1 used)"] = show(
        "A. /api/chat, num_predict=128  <-- the failing configuration", d, dt, e)

    d, dt, e = post(f"{base}/api/chat", {
        "model": model, "messages": [{"role": "user", "content": Q}],
        "stream": False, "options": {"temperature": 0, "num_predict": 1024, "seed": 42}})
    results["B chat, num_predict=1024"] = show("B. /api/chat, num_predict=1024", d, dt, e)

    d, dt, e = post(f"{base}/api/chat", {
        "model": model, "messages": [{"role": "user", "content": Q}],
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 128, "seed": 42}})
    results["C chat, think=false"] = show("C. /api/chat with think=false", d, dt, e)

    d, dt, e = post(f"{base}/api/generate", {
        "model": model, "prompt": Q, "stream": False,
        "options": {"temperature": 0, "num_predict": 128, "seed": 42}})
    results["D /api/generate"] = show("D. /api/generate", d, dt, e)

    d, dt, e = post(f"{base}/api/chat", {
        "model": model, "messages": [{"role": "user", "content": Q}],
        "stream": False, "options": {"temperature": 0, "seed": 42}})
    results["E chat, no num_predict cap"] = show("E. /api/chat, no num_predict at all", d, dt, e)

    print(f"\n{'='*70}\nSUMMARY -- which configurations produced usable text\n{'='*70}")
    for k, v in results.items():
        state = "EMPTY" if not v else f"ok ({len(v)} chars)"
        print(f"  {state:<16} {k}")

    good = [k for k, v in results.items() if v]
    print()
    if good:
        print(f"  Use this configuration: {good[0]}")
        print("  Tell me which ones worked and I will fix the harness to match.")
    else:
        print("  Every configuration came back empty. Something more basic is wrong")
        print("  with the model install. Try in a terminal:")
        print(f"    ollama run {model}")
        print("  and see whether it answers interactively at all.")

    out = os.path.join(HERE, "results", "diagnose_ollama.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": model,
                   "results": {k: (v or "") for k, v in results.items()}}, fh,
                  indent=2, ensure_ascii=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
