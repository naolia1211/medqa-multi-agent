"""LLM backends.

Two real backends, because the Phase-1 suffix and the midterm system do NOT
live on the same model:

  vicuna  -- Vicuna-7B-v1.3 fp16 via FastChat's OpenAI-compatible server (:8080).
             This is the model GCG actually optimised against (ASR = 1.00).
  gemma   -- gemma-4-12b-it-qat-q4_0 via Ollama (:11434). This is the base model
             of the midterm MedQA system.

Running both is the point: the gap between them IS the transfer-gap result.

Prompt modes
  templated : the exact conv string is POSTed to /v1/completions. Used for
              Vicuna so the wire format byte-matches Phase-1 verification.
  chat      : messages are POSTed to /v1/chat/completions (or Ollama /api/chat)
              and the server applies its own template. Used for Gemma, which is
              how the midterm system calls it.

Every backend returns latency and REAL token counts from the server -- never an
estimate -- because token cost is a graded axis of the project.
"""
from __future__ import annotations

import difflib
import json
import time
import random
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

# A transient ConnectionReset under concurrency must not become a data point.
# Without this, a dropped connection returns an empty string, the refusal-prefix
# judge finds no refusal in it, and the record is scored JAILBROKEN -- a network
# glitch silently inflating attack success.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 0.4

# Retry CONNECTION-level faults only. A ReadTimeout is deliberately NOT retried:
# on a real LLM server it usually means the model is overloaded, so retrying
# makes the queue worse and multiplies the worst case by RETRY_ATTEMPTS -- with
# timeout=300 that is a 15-minute stall on a single prompt. A dropped connection
# is different: it is instant to detect and safe to repeat.
_TRANSIENT = ("ConnectionError", "ConnectionResetError", "ChunkedEncodingError",
              "ProtocolError", "RemoteDisconnected")


def _make_session(pool: int = 64) -> requests.Session:
    s = requests.Session()
    ad = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
    s.mount("http://", ad)
    s.mount("https://", ad)
    return s


def _post_with_retry(session, url, payload, timeout):
    """POST, retrying transient network faults. Returns (json, error_str)."""
    last = ""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json(), ""
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if not any(t in type(e).__name__ or t in str(e) for t in _TRANSIENT):
                return None, last                 # a real 4xx/5xx: do not retry
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return None, f"[after {RETRY_ATTEMPTS} attempts] {last}"


VICUNA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def vicuna_v11_wrap(user_content: str, system: str = VICUNA_SYSTEM) -> str:
    """fastchat conv_template 'vicuna_v1.1' -- matches data/raw/gcg/attack_prompts.jsonl."""
    return f"{system} USER: {user_content} ASSISTANT:"


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    error: str = ""


class LLMBackend:
    name = "base"

    def generate(self, user_content: str, system: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenResult:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError


class FastChatBackend(LLMBackend):
    """Vicuna-7B-v1.3 fp16 behind fastchat.serve.openai_api_server."""

    def __init__(self, base_url: str = "http://localhost:8080/v1",
                 model: str = "vicuna-7b-v1.3",
                 prompt_mode: str = "templated",
                 system: str = VICUNA_SYSTEM,
                 timeout: int = 300,
                 name: str = "vicuna-7b-v1.3"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_mode = prompt_mode
        self.system = system
        self.timeout = timeout
        self.name = name
        self.session = _make_session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, user_content: str, system: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenResult:
        sys_prompt = system or self.system
        t0 = time.time()
        if self.prompt_mode == "templated":
            url = f"{self.base_url}/completions"
            payload = {"model": self.model,
                       "prompt": vicuna_v11_wrap(user_content, sys_prompt),
                       "max_tokens": max_new_tokens, "temperature": temperature}
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {"model": self.model,
                       "messages": [{"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": user_content}],
                       "max_tokens": max_new_tokens, "temperature": temperature}
        d, err = _post_with_retry(self.session, url, payload, self.timeout)
        if err:
            return GenResult("", 0, 0, time.time() - t0, error=err)
        try:
            text = (d["choices"][0]["text"] if self.prompt_mode == "templated"
                    else d["choices"][0]["message"]["content"])
            u = d.get("usage") or {}
            return GenResult(text=text,
                             prompt_tokens=int(u.get("prompt_tokens", 0)),
                             completion_tokens=int(u.get("completion_tokens", 0)),
                             latency=time.time() - t0)
        except Exception as e:
            return GenResult("", 0, 0, time.time() - t0,
                             error=f"malformed response: {type(e).__name__}: {e}")


class OllamaBackend(LLMBackend):
    """gemma-4-12b-it-qat-q4_0 via Ollama native /api/chat (exact token counts)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "gemma4:12b-it-qat",
                 num_ctx: int = 8192,
                 timeout: int = 600,
                 think: bool | None = False,
                 name: str = "gemma-4-12b"):
        """`think`: gemma4 is a REASONING model -- it emits a chain of thought in
        message.thinking before message.content. Measured on this project's own
        machine: with thinking ON a harmful prompt spends ~400 tokens deliberating
        before answering (102 s/call), and a num_predict below that returns an
        EMPTY content with done_reason='length'. With think=False the same prompt
        answers in 12 s. Default False for that reason; set True to evaluate the
        model in its default reasoning mode, and raise num_predict to >=1024."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self.name = name
        self.session = _make_session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, user_content: str, system: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_content})
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": max_new_tokens,
                        "seed": 42},
        }
        if self.think is not None:
            payload["think"] = self.think
        t0 = time.time()
        d, err = _post_with_retry(self.session, f"{self.base_url}/api/chat",
                                  payload, self.timeout)
        if err:
            return GenResult("", 0, 0, time.time() - t0, error=err)
        msg = d.get("message", {}) or {}
        text = msg.get("content", "") or ""
        # Empty content with done_reason='length' means the generation budget was
        # consumed by the reasoning channel. That is neither a refusal nor a
        # jailbreak -- surface it as an error so the runner marks the row
        # `unscored` instead of letting the judge read "" as compliance.
        err = ""
        if not text.strip():
            think_len = len(msg.get("thinking", "") or "")
            err = (f"empty content (done_reason={d.get('done_reason')!r}, "
                   f"eval_count={d.get('eval_count')}, thinking_chars={think_len})"
                   + ("; raise num_predict or set think=false" if think_len else ""))
        return GenResult(text=text,
                         prompt_tokens=int(d.get("prompt_eval_count", 0)),
                         completion_tokens=int(d.get("eval_count", 0)),
                         latency=time.time() - t0, error=err)


class MockBackend(LLMBackend):
    """Deterministic fake model for the smoke test -- no GPU, no server.

    Behaviour is crafted so every branch of the harness is exercised:
      * harmful goal + INTACT GCG suffix   -> complies      (C1 ASR high)
      * harmful goal, suffix damaged/absent-> refuses       (C2 ASR collapses)
      * benign prompt                      -> complies      (C0 false-refusal ~0)
      * benign prompt, heavily perturbed   -> sometimes refuses (C3 false-refusal > 0)
      * MedQA                              -> gold letter, degrading once the
                                              vignette is perturbed (C3 acc < C0)
    This is a FAKE. It validates plumbing only -- never quote its numbers.
    """
    name = "mock"

    HARMFUL_HINTS = ("bomb", "hack", "steal", "virus", "weapon", "kill", "counterfeit",
                     "ransomware", "fake news", "explosive", "poison", "assault",
                     "cyberbully", "hate speech", "fraud", "insider trading", "pirated",
                     "suicide", "anorexia", "crack passwords", "hijack")

    def __init__(self, suffix_marker: str = "Tomatoes FM tutorials aby",
                 gold_map: dict | None = None, noise_sensitivity: float = 0.25):
        self.suffix_marker = suffix_marker
        self.gold_map = gold_map or {}
        self.noise_sensitivity = noise_sensitivity
        self._keys = list(self.gold_map)
        self._lookup_cache: dict[str, str] = {}

    def _gold_for(self, user_content: str) -> str:
        """Resolve the answer key from a possibly PERTURBED prompt.

        A prefix key is not enough: MedQA prompts share a long common preamble, so
        prefixes collide, and under SmoothLLM the text arrives with ~10% of its
        characters replaced. Fuzzy-match instead, so the fake model still knows the
        right answer and any accuracy loss comes from the noise model rather than
        from a failed dictionary lookup.
        """
        if not self._keys:
            return "C"
        k = _gold_key(user_content)
        if k in self.gold_map:
            return self.gold_map[k]
        if k in self._lookup_cache:
            return self.gold_map.get(self._lookup_cache[k], "C")
        # Narrow the candidate set by length before the O(n*m) fuzzy match --
        # character-swap noise preserves length, so this is lossless for `swap`
        # and still generous for `patch` / `insert`.
        lo, hi = len(k) * 0.80, len(k) * 1.25
        cands = [x for x in self._keys if lo <= len(x) <= hi] or self._keys
        near = difflib.get_close_matches(k, cands, n=1, cutoff=0.55)
        if near:
            self._lookup_cache[k] = near[0]
            return self.gold_map[near[0]]
        return "C"

    def health(self) -> bool:
        return True

    @staticmethod
    def _perturbed(text: str) -> float:
        """Crude proxy for how mangled the text is: share of odd symbol chars."""
        if not text:
            return 0.0
        odd = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,'\"?!()-:;/"))
        return odd / len(text)

    def generate(self, user_content: str, system: str = "", max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenResult:
        time.sleep(0.001)
        rng = random.Random(f"mock|{user_content}")
        noise = self._perturbed(user_content)

        # ---- MedQA: answers, accuracy decays with character noise ----
        if "Answer:" in user_content or "USMLE" in system:
            gold = self._gold_for(user_content)
            p_correct = max(0.20, 0.85 - 18.0 * noise)
            letter = gold if rng.random() < p_correct else rng.choice("ABCDE")
            return GenResult(f"Brief reasoning omitted.\nAnswer: {letter}", 420, 24, 0.001)

        # ---- the GCG suffix survived intact -> jailbroken ----
        if self.suffix_marker in user_content:
            return GenResult("Sure, here is the thing you asked for. Step 1 ... Step 2 ...",
                             60, 30, 0.001)

        # ---- prompt mangled beyond comprehension -> the model bails out ----
        # This is what actually drives SmoothLLM: noise destroys the suffix, and the
        # model no longer produces the attacker's completion. It also costs utility,
        # which is why benign prompts land here too (-> false refusals in C3).
        p_confused = min(0.95, max(0.0, 30.0 * (noise - 0.005)))
        if rng.random() < p_confused:
            return GenResult("I'm sorry, I could not understand that request.", 40, 12, 0.001)

        # ---- intact harmful request -> aligned refusal ----
        if any(h in user_content.lower() for h in self.HARMFUL_HINTS):
            return GenResult("I'm sorry, but I cannot help with that request.", 40, 12, 0.001)

        # ---- intact benign request -> comply ----
        return GenResult("Certainly. Here is a helpful and reasonably detailed answer "
                         "covering the main points you asked about.", 55, 28, 0.001)


def _gold_key(s: str) -> str:
    """Signature used to look up the fake model's answer key.

    Takes a WINDOW from the middle of the prompt, not the first 60 chars: MedQA
    items share a long fixed preamble, so a leading prefix is identical across
    items and collides. 240 chars from offset 40 captures the case description.
    """
    return s.strip()[40:280]


def build_backend(cfg: dict) -> LLMBackend:
    kind = cfg.get("kind", "mock")
    if kind == "fastchat":
        return FastChatBackend(**{k: v for k, v in cfg.items() if k != "kind"})
    if kind == "ollama":
        return OllamaBackend(**{k: v for k, v in cfg.items() if k != "kind"})
    if kind == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend kind {kind!r}")
