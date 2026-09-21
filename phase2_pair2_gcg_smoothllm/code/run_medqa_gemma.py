#!/usr/bin/env python3
"""MedQA utility cost of SmoothLLM, on the system's own base model.

This is arm B's §6.3: what the defense costs on the task the system actually
does. It fetches MedQA-USMLE straight from the HuggingFace datasets-server (no
`datasets` package, no login, standard library only) and runs two conditions on
the same questions:

    C0   question as-is,                 1 call     -> baseline accuracy
    C3   N perturbed copies + majority vote on the extracted letter

    python run_medqa_gemma.py                    # 100 questions, N=4, q=10%
    python run_medqa_gemma.py --n 200 --N 6
    python run_medqa_gemma.py --local-file med.jsonl     # skip the download

Resume-safe: re-running continues from results/medqa_raw.jsonl.

Two deliberate settings, both of which must be stated in the report:

* `think=false`. gemma4 is a reasoning model; left on, a single call costs ~100 s
  and a small generation budget returns EMPTY content. The midterm system
  assumed no separate think channel, so this is also the faithful setting.
* Direct-answer prompting with a small budget, not the midterm's chain of
  thought. Absolute accuracy is therefore NOT comparable with the midterm's
  81.70% V0 figure. The C0-vs-C3 *paired difference* is still valid, because
  both arms use the identical setting -- and that difference is what measures
  the defense's cost.
"""
from __future__ import annotations

import argparse, json, os, random, re, string, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
HF = "https://datasets-server.huggingface.co/rows"
DATASET = "GBaker/MedQA-USMLE-4-options"
SYSTEM = "You are a medical expert taking the USMLE."
ALPHABET = string.printable[:94]


# ------------------------------------------------------------------ data ----
def fetch_medqa(n: int, seed: int) -> list[dict]:
    """Spread sample across the 1,273-row official test split."""
    got, offsets = [], []
    per = min(100, max(1, n // 4))
    step = max(1, 1273 // max(1, (n // per)))
    off = 0
    while len(offsets) * per < n and off < 1273:
        offsets.append(off)
        off += step
    print(f"  fetching {n} questions from {DATASET} (test split, 1273 rows)")
    for o in offsets:
        q = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
                                    "split": "test", "offset": o,
                                    "length": min(per, n - len(got))})
        try:
            with urllib.request.urlopen(f"{HF}?{q}", timeout=60) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            sys.exit(f"[!] could not reach the HuggingFace datasets-server: {e}\n"
                     f"    Use --local-file with a jsonl of "
                     f"{{question, options, answer_idx}} instead.")
        for it in d.get("rows", []):
            row = it["row"]
            got.append({"idx": it["row_idx"], "question": row["question"],
                        "options": row["options"], "gold": row["answer_idx"]})
        if len(got) >= n:
            break
    return got[:n]


def load_local(path: str, n: int) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            d = json.loads(line)
            opts = d.get("options") or d.get("choices") or {}
            if isinstance(opts, list):
                opts = {chr(65 + j): o for j, o in enumerate(opts)}
            gold = str(d.get("answer_idx") or d.get("gold") or d.get("answer") or "")[:1].upper()
            if d.get("question") and opts and gold in "ABCDE":
                out.append({"idx": i, "question": d["question"], "options": opts, "gold": gold})
    return out[:n]


def build_prompt(r: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(r["options"].items()))
    return (f"Answer the following multiple-choice question.\n\n{r['question']}\n\n"
            f"{opts}\n\nRespond with only the single letter of the correct option.")


def answer_schema(options: dict) -> dict:
    """Constrain the model to emit exactly {"answer": "<letter>"}.

    Measured the hard way: with free-form decoding gemma4 sometimes answers "B"
    (2 tokens) and sometimes starts explaining ("To solve this clinical vignette,
    let's break down the findings:") and hits the generation cap before reaching
    a letter. That produced a 39% INVALID rate on the undefended arm -- while the
    defended arm, voting over N copies, recovered from it and showed 0%. The
    result looked like SmoothLLM IMPROVING accuracy by 25 points. It was entirely
    a truncation artefact. A JSON schema removes the failure mode at the source.
    """
    return {"type": "object",
            "properties": {"answer": {"type": "string",
                                      "enum": sorted(options.keys())}},
            "required": ["answer"]}


# ------------------------------------------------------------- extraction ----
_ANS = re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*[*_`(\[]*([A-E])\b", re.IGNORECASE)
_BOLD = re.compile(r"[*_`]{1,2}\s*([A-E])\s*[*_`]{1,2}")
_LEAD = re.compile(r"^\s*[*_`(\[]*([A-E])\b")
_ANY = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def extract(text: str) -> str:
    """Return 'A'..'E' or 'INVALID'.

    With the JSON schema in force the first branch always fires; the regex tiers
    remain as a fallback for --no-schema runs. A high INVALID rate is a
    measurement problem, not a model result -- it drags accuracy down for a
    reason unrelated to the defense, so the summary reports it separately rather
    than letting it hide inside the accuracy figure.
    """
    if not text or not text.strip():
        return "INVALID"
    try:
        v = json.loads(text).get("answer", "")
        if isinstance(v, str) and len(v) == 1 and v.upper() in "ABCDE":
            return v.upper()
    except Exception:
        pass
    for rx in (_ANS, _LEAD, _BOLD):
        m = rx.search(text)
        if m:
            return m.group(1).upper()
    hits = _ANY.findall(text)
    return hits[-1].upper() if hits else "INVALID"


# ------------------------------------------------------------------ model ----
def chat(base: str, model: str, user: str, np_: int, timeout: int, think: bool,
         schema: dict | None = None) -> dict:
    body = {"model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": np_, "seed": 42}}
    if not think:
        body["think"] = False
    if schema:
        body["format"] = schema
    t0 = time.time()
    try:
        req = urllib.request.Request(f"{base}/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        msg = d.get("message", {}) or {}
        return {"text": msg.get("content", "") or "", "secs": time.time() - t0,
                "tokens": d.get("eval_count", 0), "done": d.get("done_reason", ""),
                "thinking": len(msg.get("thinking", "") or ""), "err": ""}
    except Exception as e:
        return {"text": "", "secs": time.time() - t0, "tokens": 0, "done": "",
                "thinking": 0, "err": f"{type(e).__name__}: {e}"}


def perturb(text: str, q: float, rng: random.Random) -> str:
    a = list(text)
    for i in rng.sample(range(len(a)), max(1, int(len(a) * q / 100))):
        a[i] = rng.choice(ALPHABET)
    return "".join(a)


# ------------------------------------------------------------------- main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--model", default="gemma4:12b-it-qat")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--q", type=float, default=10.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--no-schema", action="store_true",
                    help="free-form decoding instead of the JSON schema; expect "
                         "truncation-driven INVALIDs on the undefended arm")
    ap.add_argument("--think", action="store_true", help="leave reasoning ON (slow)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-file", default=None)
    a = ap.parse_args()

    print("=" * 70)
    print("MedQA -- what SmoothLLM costs on the system's own task")
    print("=" * 70)

    rows = load_local(a.local_file, a.n) if a.local_file else fetch_medqa(a.n, a.seed)
    if not rows:
        sys.exit("[!] no questions loaded")
    print(f"  {len(rows)} questions, N={a.N}, q={a.q}%, "
          f"think={'on' if a.think else 'off'}, num_predict={a.max_tokens}")

    os.makedirs(RESULTS, exist_ok=True)
    raw_path = os.path.join(RESULTS, "medqa_raw.jsonl")
    done = {}
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r
        print(f"  resuming: {len(done)} already on disk")

    schema = None if a.no_schema else answer_schema(rows[0]["options"])
    warm = chat(a.base, a.model, "Reply with the single letter A.", a.max_tokens,
                a.timeout, a.think, schema)
    if warm["err"]:
        sys.exit(f"[!] cannot reach the model: {warm['err']}")
    if not warm["text"].strip():
        sys.exit(f"[!] the model returned EMPTY content "
                 f"(done_reason={warm['done']!r}, thinking={warm['thinking']} chars).\n"
                 f"    Raise --max-tokens, or drop --think. Run diagnose_ollama.py.")
    print(f"  warm-up ok in {warm['secs']:.1f}s -> {warm['text'].strip()[:40]!r}\n")

    t0, fh = time.time(), open(raw_path, "a", encoding="utf-8")
    for i, r in enumerate(rows):
        if r["idx"] in done:
            continue
        prompt = build_prompt(r)
        sch = None if a.no_schema else answer_schema(r["options"])
        c0 = chat(a.base, a.model, prompt, a.max_tokens, a.timeout, a.think, sch)
        copies = []
        for j in range(a.N):
            rng = random.Random(f"{a.seed}|{r['idx']}|{j}")
            copies.append(chat(a.base, a.model, perturb(prompt, a.q, rng),
                               a.max_tokens, a.timeout, a.think, sch))
        preds = [extract(c["text"]) for c in copies]
        valid = [p for p in preds if p != "INVALID"]
        c3 = max(set(valid), key=valid.count) if valid else "INVALID"
        rec = {"idx": r["idx"], "gold": r["gold"],
               "c0_pred": extract(c0["text"]), "c0_text": c0["text"][:200],
               "c0_secs": round(c0["secs"], 2), "c0_tokens": c0["tokens"],
               "c0_err": c0["err"],
               "c3_pred": c3, "c3_preds": preds,
               "c3_texts": [c["text"][:80] for c in copies],
               "c3_secs": round(sum(c["secs"] for c in copies), 2),
               "c3_tokens": sum(c["tokens"] for c in copies),
               "c3_errors": sum(1 for c in copies if c["err"])}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        done[r["idx"]] = rec
        n = len(done)
        if n % 5 == 0 or n == len(rows):
            el = time.time() - t0
            print(f"    {n}/{len(rows)}  {el/60:.1f} min  "
                  f"eta {(el/max(i+1,1))*(len(rows)-n)/60:.1f} min")
    fh.close()

    d = list(done.values())
    n = len(d)
    acc = lambda k: round(100.0 * sum(x[k] == x["gold"] for x in d) / n, 1)
    inv = lambda k: round(100.0 * sum(x[k] == "INVALID" for x in d) / n, 1)
    mean = lambda k: round(sum(x[k] for x in d) / n, 1)
    b = sum(1 for x in d if x["c0_pred"] == x["gold"] and x["c3_pred"] != x["gold"])
    c = sum(1 for x in d if x["c0_pred"] != x["gold"] and x["c3_pred"] == x["gold"])

    import math
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(b + c, i) for i in range(k + 1)) * 0.5 ** (b + c)) \
        if b + c else 1.0

    summary = {
        "arm": "B -- gemma4, the midterm system's own base model",
        "model": a.model, "n": n, "N": a.N, "q": a.q,
        "think": a.think, "num_predict": a.max_tokens,
        "constrained_decoding": not a.no_schema,
        "dataset": f"{DATASET} test split" if not a.local_file else a.local_file,
        "C0": {"accuracy": acc("c0_pred"), "invalid_rate": inv("c0_pred"),
               "secs_mean": mean("c0_secs"), "tokens_mean": mean("c0_tokens")},
        "C3": {"accuracy": acc("c3_pred"), "invalid_rate": inv("c3_pred"),
               "secs_mean": mean("c3_secs"), "tokens_mean": mean("c3_tokens")},
        "accuracy_drop_pp": round(acc("c3_pred") - acc("c0_pred"), 1),
        "paired": {"c0_only_correct": b, "c3_only_correct": c,
                   "tie": n - b - c, "mcnemar_exact_p": round(p, 6)},
        "cost": {"llm_calls_multiplier": a.N,
                 "wall_multiplier": round(mean("c3_secs") / max(mean("c0_secs"), 1e-9), 2),
                 "token_multiplier": round(mean("c3_tokens") / max(mean("c0_tokens"), 1e-9), 2)},
        "caveat": ("Direct-answer prompting with a small generation budget, not the "
                   "midterm's chain of thought. Absolute accuracy is NOT comparable "
                   "with the midterm's 81.70% V0 figure; the C0-vs-C3 paired "
                   "difference is, because both arms use the identical setting."),
    }
    out = os.path.join(RESULTS, "medqa_summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  C0 accuracy   {summary['C0']['accuracy']}%   "
          f"(invalid {summary['C0']['invalid_rate']}%)")
    print(f"  C3 accuracy   {summary['C3']['accuracy']}%   "
          f"(invalid {summary['C3']['invalid_rate']}%)")
    print(f"  drop          {summary['accuracy_drop_pp']:+} pp   "
          f"McNemar p={summary['paired']['mcnemar_exact_p']}  "
          f"(C0-only {b}, C3-only {c}, tie {n-b-c})")
    print(f"  cost          {a.N}x calls, "
          f"{summary['cost']['wall_multiplier']}x wall, "
          f"{summary['cost']['token_multiplier']}x tokens")
    if summary["C0"]["invalid_rate"] > 5:
        print(f"\n  [!] C0 invalid rate is {summary['C0']['invalid_rate']}% -- that is a")
        print("      PARSING problem, not a model result, and it drags accuracy down")
        print("      for a reason unrelated to the defense. Inspect c0_text in")
        print("      results/medqa_raw.jsonl and fix extract() before quoting accuracy.")
    print(f"\n  wrote {out} and {raw_path}")


if __name__ == "__main__":
    main()
