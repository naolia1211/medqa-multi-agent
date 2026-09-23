"""Log schema for the C0-C3 evaluation harness.

The eight fields agreed with the team are the CONTRACT -- every writer must emit
them and every reader may rely on them:

    prompt_id, condition, suffix_id, base_response, safety_label,
    final_response, latency, token_usage

Everything else below is additive context (backend, dataset, gold/pred, ...).
Additive fields may be extended; the eight core fields may not be renamed.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Optional

SCHEMA_VERSION = "1.0"

# --- the contract -----------------------------------------------------------
CORE_FIELDS = (
    "prompt_id",
    "condition",
    "suffix_id",
    "base_response",
    "safety_label",
    "final_response",
    "latency",
    "token_usage",
)

# --- controlled vocabularies ------------------------------------------------
CONDITIONS = {
    "C0": "benign input, no defense        -> clean baseline (utility + refusal floor)",
    "C1": "attacked input, no defense      -> jailbreak ASR baseline",
    "C2": "attacked input, SmoothLLM       -> defense effectiveness",
    "C3": "benign input, SmoothLLM         -> defense cost (utility drop + false refusal)",
}
ATTACKED_CONDITIONS = ("C1", "C2")
DEFENDED_CONDITIONS = ("C2", "C3")

DATASETS = ("safety", "benign_alpaca", "benign_xstest", "medqa")

# safety_label is the refusal-prefix judge verdict.
# "unscored" = the call failed, so there is no output to judge. It is NOT a
# refusal and NOT a jailbreak; metrics exclude it from every denominator.
SAFETY_LABELS = ("refused", "jailbroken", "na", "unscored")
# answer_label is the MedQA correctness verdict (additive, MedQA only).
ANSWER_LABELS = ("correct", "incorrect", "invalid", "na", "unscored")

SUFFIX_NONE = "none"
SUFFIX_UNIVERSAL = "gcg_universal_v1"


@dataclass
class TokenUsage:
    """Token accounting. For defended conditions these are SUMS over the N copies."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_llm_calls: int = 0            # 1 undefended; N (x n_agents) defended
    per_copy_total: list[int] = field(default_factory=list)

    def add(self, prompt_t: int, completion_t: int) -> None:
        self.prompt_tokens += prompt_t
        self.completion_tokens += completion_t
        self.total_tokens += prompt_t + completion_t
        self.n_llm_calls += 1
        self.per_copy_total.append(prompt_t + completion_t)


@dataclass
class LogRecord:
    # ---- the eight core fields ----------------------------------------
    prompt_id: str
    condition: str
    suffix_id: str
    base_response: Any            # str (undefended) | list[str] (defended, N copies)
    safety_label: str
    final_response: str
    latency: float                # wall-clock seconds for the whole defended call
    token_usage: dict

    # ---- additive context ---------------------------------------------
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    backend: str = ""             # "vicuna-7b-v1.3" | "gemma-4-12b" | "mock"
    dataset: str = ""             # see DATASETS
    defense: str = "none"         # "none" | "smoothllm"
    defense_params: dict = field(default_factory=dict)   # {q, N, perturbation}

    prompt_text: str = ""         # user content actually sent (unperturbed)
    system_prompt: str = ""

    # per-copy detail for defended conditions (empty when undefended)
    perturbed_prompts: list[str] = field(default_factory=list)
    base_labels: list[str] = field(default_factory=list)  # judge verdict per copy
    vote_margin: float = 0.0      # fraction of VOTING copies agreeing with the winner
    n_failed_copies: int = 0      # copies whose call failed, excluded from the vote
    n_voting_copies: int = 0      # copies that actually voted

    # MedQA-only
    gold: str = ""
    pred: str = ""
    answer_label: str = "na"

    # cost detail
    latency_sum_calls: float = 0.0   # sum of per-call latencies (serial-equivalent cost)
    error: str = ""
    ts: float = 0.0

    def validate(self) -> None:
        assert self.condition in CONDITIONS, f"bad condition {self.condition!r}"
        assert self.safety_label in SAFETY_LABELS, f"bad safety_label {self.safety_label!r}"
        assert self.answer_label in ANSWER_LABELS, f"bad answer_label {self.answer_label!r}"
        assert self.dataset in DATASETS or self.dataset == "", f"bad dataset {self.dataset!r}"
        # a defended condition must carry per-copy evidence
        if self.condition in DEFENDED_CONDITIONS and not self.error:
            assert isinstance(self.base_response, list), \
                "defended conditions must log base_response as the list of N copy responses"
        if self.condition in ATTACKED_CONDITIONS:
            assert self.suffix_id != SUFFIX_NONE, "attacked conditions need a real suffix_id"
        else:
            assert self.suffix_id == SUFFIX_NONE, "unattacked conditions must use suffix_id='none'"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class JsonlLogger:
    """Append-only, thread-safe, resume-safe JSONL writer.

    Resume key is (prompt_id, condition, suffix_id, backend) -- re-running a
    partially finished sweep skips what is already on disk.
    """

    def __init__(self, path: str, resume: bool = True):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self.done: set[tuple] = set()
        if resume and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # tolerate a truncated final line
                    self.done.add(self.key_of(r))
        self._fh = open(path, "a", encoding="utf-8")

    @staticmethod
    def key_of(r: dict | LogRecord) -> tuple:
        g = (lambda k: getattr(r, k) if isinstance(r, LogRecord) else r.get(k, ""))
        return (g("prompt_id"), g("condition"), g("suffix_id"), g("backend"))

    def already_done(self, prompt_id: str, condition: str, suffix_id: str, backend: str) -> bool:
        return (prompt_id, condition, suffix_id, backend) in self.done

    def write(self, rec: LogRecord) -> None:
        rec.validate()
        with self._lock:
            self._fh.write(rec.to_json() + "\n")
            self._fh.flush()
            self.done.add(self.key_of(rec))

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
