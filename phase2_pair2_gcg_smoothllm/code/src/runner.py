"""C0-C3 sweep runner.

Condition grid -- a clean 2x2 of (input attacked?) x (defense on?):

              | no defense | SmoothLLM
  ------------+------------+-----------
  benign in   |     C0     |    C3
  attacked in |     C1     |    C2

Which dataset flows into which condition:

  safety(suffix=none)  -> C0, C3   refusal floor: does the defense break refusal?
  safety(suffix=gcg)   -> C1, C2   jailbreak ASR before/after
  benign_alpaca        -> C0, C3   utility + false-refusal cost
  benign_xstest        -> C0, C3   over-refusal cost (the sharp one)
  medqa                -> C0, C3   clean task accuracy before/after

MedQA never gets a suffix. That separation is deliberate and is stated in the
Phase-1 pack: attaching an adversarial suffix to a clinical vignette would
conflate the jailbreak measurement with the utility measurement.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import judges
from backends import LLMBackend
from schema import (ATTACKED_CONDITIONS, DEFENDED_CONDITIONS, JsonlLogger,
                    LogRecord, SUFFIX_NONE, TokenUsage)
from smoothllm import SmoothLLM, answer_aggregator, refusal_aggregator


def applicable(rec: dict, condition: str) -> bool:
    attacked_input = rec["suffix_id"] != SUFFIX_NONE
    return attacked_input == (condition in ATTACKED_CONDITIONS)


def aggregator_for(dataset: str, prefixes: list[str]):
    if dataset == "medqa":
        return answer_aggregator(judges.extract_answer)
    return refusal_aggregator(lambda r: judges.judge_refusal(r, prefixes))


def run_one(rec: dict,
            condition: str,
            backend: LLMBackend,
            prefixes: list[str],
            defense_cfg: dict,
            max_new_tokens: int,
            run_id: str) -> LogRecord:
    ds = rec["dataset"]
    user = rec["user_content"]
    system = rec.get("system_prompt", "")
    defended = condition in DEFENDED_CONDITIONS

    usage = TokenUsage()
    base_response: object
    perturbed: list[str] = []
    base_labels: list[str] = []
    margin = 0.0
    err = ""
    n_failed = n_voting = 0

    t0 = time.time()
    if not defended:
        g = backend.generate(user, system, max_new_tokens)
        final = g.text
        base_response = g.text
        usage.add(g.prompt_tokens, g.completion_tokens)
        latency_wall = g.latency
        latency_sum = g.latency
        err = g.error
        n_failed, n_voting = (1, 0) if g.error else (0, 1)
    else:
        sm = SmoothLLM(backend,
                       aggregator_for(ds, prefixes),
                       n_copies=defense_cfg["N"],
                       q=defense_cfg["q"],
                       perturbation=defense_cfg["perturbation"],
                       seed=defense_cfg.get("seed", 42),
                       max_workers=defense_cfg.get("copy_workers", 4))
        r = sm.generate(user, system, max_new_tokens)
        final = r.final_response
        base_response = r.per_copy_responses
        perturbed = r.perturbed_prompts
        base_labels = r.per_copy_labels
        margin = r.vote_margin
        usage.prompt_tokens = r.prompt_tokens
        usage.completion_tokens = r.completion_tokens
        usage.total_tokens = r.prompt_tokens + r.completion_tokens
        usage.n_llm_calls = r.n_llm_calls
        usage.per_copy_total = r.per_copy_total
        latency_wall = r.latency_wall
        latency_sum = r.latency_sum_calls
        err = "; ".join(r.errors[:3])
        n_failed, n_voting = r.n_failed_copies, r.n_voting_copies

    # ---- labels -----------------------------------------------------------
    # If nothing came back, there is nothing to judge. Scoring an empty string
    # would silently read as "jailbroken" (no refusal prefix present) and turn a
    # network fault into attack success. Mark it unscored instead.
    scorable = n_voting > 0 and bool(final)
    if not scorable:
        safety_label, pred, answer_label = "unscored", "", "unscored"
    elif ds == "medqa":
        safety_label = "na"
        pred, answer_label = judges.judge_medqa(final, rec.get("gold", ""))
    else:
        safety_label = judges.judge_refusal(final, prefixes)
        pred, answer_label = "", "na"

    return LogRecord(
        prompt_id=rec["prompt_id"],
        condition=condition,
        suffix_id=rec["suffix_id"],
        base_response=base_response,
        safety_label=safety_label,
        final_response=final,
        latency=round(latency_wall, 4),
        token_usage={"prompt_tokens": usage.prompt_tokens,
                     "completion_tokens": usage.completion_tokens,
                     "total_tokens": usage.total_tokens,
                     "n_llm_calls": usage.n_llm_calls,
                     "per_copy_total": usage.per_copy_total},
        run_id=run_id,
        backend=backend.name,
        dataset=ds,
        defense="smoothllm" if defended else "none",
        defense_params=(defense_cfg if defended else {}),
        prompt_text=user,
        system_prompt=system,
        perturbed_prompts=perturbed,
        base_labels=base_labels,
        vote_margin=round(margin, 4),
        n_failed_copies=n_failed,
        n_voting_copies=n_voting,
        gold=rec.get("gold", ""),
        pred=pred,
        answer_label=answer_label,
        latency_sum_calls=round(latency_sum, 4),
        error=err,
        ts=time.time(),
    )


def run_sweep(records: list[dict],
              conditions: Iterable[str],
              backend: LLMBackend,
              logger: JsonlLogger,
              defense_cfg: dict,
              prefixes: list[str] | None = None,
              max_new_tokens: int = 256,
              workers: int = 4,
              run_id: str | None = None,
              progress_every: int = 25) -> int:
    prefixes = prefixes or judges.load_refusal_prefixes()
    run_id = run_id or uuid.uuid4().hex[:8]

    jobs = [(r, c) for c in conditions for r in records
            if applicable(r, c)
            and not logger.already_done(r["prompt_id"], c, r["suffix_id"], backend.name)]
    total = len(jobs)
    if not total:
        print(f"[{backend.name}] nothing to do (all done / nothing applicable)")
        return 0
    print(f"[{backend.name}] {total} calls queued  (run_id={run_id}, workers={workers})")

    n_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, r, c, backend, prefixes, defense_cfg,
                          max_new_tokens, run_id): (r, c) for r, c in jobs}
        for fut in as_completed(futs):
            r, c = futs[fut]
            try:
                logger.write(fut.result())
            except Exception as e:                      # never lose the sweep to one bad row
                print(f"  [ERR] {r['prompt_id']}/{c}: {type(e).__name__}: {e}")
            n_done += 1
            if n_done % progress_every == 0 or n_done == total:
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                print(f"  {n_done}/{total}  {el:6.1f}s  {rate:5.2f}/s  "
                      f"eta {(total - n_done) / max(rate, 1e-6):6.1f}s")
    return n_done
