"""SmoothLLM (Robey, Wong, Hassani & Pappas, arXiv 2023) -- reimplementation.

Mechanism: GCG suffixes are brittle to character-level noise. Perturb N copies
of the *user content*, query the model on each, then take a majority vote over
a per-copy LABEL and return a response consistent with the winning label.

Three deliberate design points, all worth a sentence in the report:

1. Only the UNTRUSTED user content is perturbed. The system prompt and, for the
   multi-agent variant, the inter-agent scaffolding are left intact. Perturbing
   the system prompt would inflate the apparent defense by breaking the model.

2. The aggregator is PLUGGABLE, because "majority vote" means different things
   per dataset:
       safety / benign -> vote over the refusal-prefix verdict
       MedQA           -> vote over the extracted answer letter A-E
   Voting over refusal labels on MedQA would be meaningless.

3. Placement is at the LLM-CLIENT layer (decided in Phase-2 planning), so in the
   multi-agent system every one of Reasoner / Verifier / Aggregator is wrapped.
   Cost is therefore N x n_agents calls, and `token_usage.n_llm_calls` records
   the true number -- do not report the single-agent N as the system cost.
"""
from __future__ import annotations

import hashlib
import random
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from backends import GenResult, LLMBackend

ALPHABET = string.printable[:94]     # printable ASCII, no whitespace control chars


def stable_hash(text: str) -> str:
    """Process-stable content hash.

    Python's built-in hash() is randomised per process (PYTHONHASHSEED), so using
    it to seed the perturbation RNG would make a run irreproducible across
    invocations -- a real problem when the report has to be re-derivable.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Perturbations (the three from the paper)
# --------------------------------------------------------------------------
def random_swap(text: str, q: float, rng: random.Random) -> str:
    """Replace q% of characters, chosen uniformly at random, with random ones."""
    if not text:
        return text
    chars = list(text)
    k = max(1, int(len(chars) * q / 100))
    for i in rng.sample(range(len(chars)), min(k, len(chars))):
        chars[i] = rng.choice(ALPHABET)
    return "".join(chars)


def random_patch(text: str, q: float, rng: random.Random) -> str:
    """Replace one contiguous block covering q% of the string."""
    if not text:
        return text
    k = max(1, int(len(text) * q / 100))
    k = min(k, len(text))
    start = rng.randint(0, len(text) - k)
    patch = "".join(rng.choice(ALPHABET) for _ in range(k))
    return text[:start] + patch + text[start + k:]


def random_insert(text: str, q: float, rng: random.Random) -> str:
    """Insert q% new random characters at random positions."""
    if not text:
        return text
    chars = list(text)
    k = max(1, int(len(chars) * q / 100))
    for _ in range(k):
        chars.insert(rng.randint(0, len(chars)), rng.choice(ALPHABET))
    return "".join(chars)


PERTURBATIONS: dict[str, Callable[[str, float, random.Random], str]] = {
    "swap": random_swap,
    "patch": random_patch,
    "insert": random_insert,
}


# --------------------------------------------------------------------------
# Aggregators
# --------------------------------------------------------------------------
def refusal_aggregator(judge: Callable[[str], str]) -> Callable[[Sequence[str]], tuple[str, list[str], float]]:
    """Vote over refusal-prefix verdicts. Returns (winning_label, per_copy_labels, margin)."""
    def _agg(responses: Sequence[str]):
        labels = [judge(r) for r in responses]
        counts = Counter(labels)
        winner, n = counts.most_common(1)[0]
        return winner, labels, n / max(len(labels), 1)
    return _agg


def answer_aggregator(extract: Callable[[str], str]) -> Callable[[Sequence[str]], tuple[str, list[str], float]]:
    """Vote over extracted MedQA letters. INVALID copies are counted but lose ties."""
    def _agg(responses: Sequence[str]):
        labels = [extract(r) for r in responses]
        counts = Counter(labels)
        valid = {k: v for k, v in counts.items() if k != "INVALID"}
        pool = valid or counts
        winner = max(pool.items(), key=lambda kv: (kv[1], kv[0] != "INVALID"))[0]
        return winner, labels, counts[winner] / max(len(labels), 1)
    return _agg


@dataclass
class SmoothResult:
    final_response: str
    winning_label: str
    per_copy_responses: list[str]
    per_copy_labels: list[str]
    perturbed_prompts: list[str]
    vote_margin: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_llm_calls: int = 0
    per_copy_total: list[int] = field(default_factory=list)
    latency_wall: float = 0.0
    latency_sum_calls: float = 0.0
    errors: list[str] = field(default_factory=list)
    n_failed_copies: int = 0      # copies excluded from the vote
    n_voting_copies: int = 0      # copies that actually voted


class SmoothLLM:
    """Wraps any LLMBackend. Call .generate() exactly like the bare backend."""

    def __init__(self,
                 backend: LLMBackend,
                 aggregator: Callable[[Sequence[str]], tuple[str, list[str], float]],
                 n_copies: int = 10,
                 q: float = 10.0,
                 perturbation: str = "swap",
                 seed: int = 42,
                 max_workers: int = 4):
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"perturbation must be one of {list(PERTURBATIONS)}")
        self.backend = backend
        self.aggregator = aggregator
        self.n_copies = n_copies
        self.q = q
        self.perturbation = perturbation
        self.seed = seed
        self.max_workers = max_workers
        self.name = f"smoothllm(N={n_copies},q={q}%,{perturbation})"

    @property
    def params(self) -> dict:
        return {"N": self.n_copies, "q": self.q, "perturbation": self.perturbation,
                "seed": self.seed, "placement": "per_llm_call"}

    def perturb_batch(self, user_content: str, call_salt: int = 0) -> list[str]:
        fn = PERTURBATIONS[self.perturbation]
        out = []
        for i in range(self.n_copies):
            rng = random.Random(f"{self.seed}|{call_salt}|{i}|{stable_hash(user_content)}")
            out.append(fn(user_content, self.q, rng))
        return out

    def generate(self, user_content: str, system: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.0, call_salt: int = 0) -> SmoothResult:
        import time
        prompts = self.perturb_batch(user_content, call_salt)
        t0 = time.time()
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                results: list[GenResult] = list(ex.map(
                    lambda p: self.backend.generate(p, system, max_new_tokens, temperature), prompts))
        else:
            results = [self.backend.generate(p, system, max_new_tokens, temperature) for p in prompts]
        wall = time.time() - t0

        responses = [r.text for r in results]

        # A copy whose HTTP call FAILED carries an empty string. Letting it vote
        # would mean a network glitch casts a ballot -- and since the refusal
        # judge finds no refusal prefix in "", that ballot reads as "jailbroken".
        # Failed copies are excluded from the vote and counted separately.
        ok = [(r.text, g) for r, g in zip(results, results) if not g.error]
        voting = [t for t, _ in ok]
        n_failed = len(results) - len(voting)

        if voting:
            winner, vlabels, margin = self.aggregator(voting)
        else:                       # every copy failed: nothing to decide
            winner, vlabels, margin = "", [], 0.0

        # per-copy labels are still reported for ALL copies, with failures marked,
        # so the log shows exactly what was and was not counted.
        labels, vi = [], 0
        for r in results:
            if r.error:
                labels.append("error")
            else:
                labels.append(vlabels[vi]); vi += 1

        consistent = [t for t, l in zip(voting, vlabels) if l == winner]
        rng = random.Random(f"{self.seed}|{call_salt}|pick|{stable_hash(user_content)}")
        final = rng.choice(consistent) if consistent else (voting[0] if voting else "")

        return SmoothResult(
            final_response=final,
            winning_label=winner,
            per_copy_responses=responses,
            per_copy_labels=labels,
            perturbed_prompts=prompts,
            vote_margin=margin,
            prompt_tokens=sum(r.prompt_tokens for r in results),
            completion_tokens=sum(r.completion_tokens for r in results),
            n_llm_calls=len(results),
            per_copy_total=[r.prompt_tokens + r.completion_tokens for r in results],
            latency_wall=wall,
            latency_sum_calls=sum(r.latency for r in results),
            errors=[r.error for r in results if r.error],
            n_failed_copies=n_failed,
            n_voting_copies=len(voting),
        )
