#!/usr/bin/env python3
"""A fake model server that speaks the REAL FastChat and Ollama wire protocols.

Why: `FastChatBackend` and `OllamaBackend` talk HTTP. Until a GPU is available
their success paths never execute, so a mistake in the request payload or in
parsing `usage` / `prompt_eval_count` would only surface the night the real run
starts. This server closes that gap -- same endpoints, same JSON shapes, same
token accounting, over a real socket.

It also doubles as a rehearsal target: point `config.yaml` at it, run the whole
pipeline, confirm the plumbing, then swap in the real server.

  python tests/fake_model_server.py --port 8080 --api openai
  python tests/fake_model_server.py --port 11434 --api ollama

Endpoints
  openai : GET /v1/models, POST /v1/completions, POST /v1/chat/completions
  ollama : GET /api/tags,  POST /api/chat

The simulated model is "aligned and suffix-brittle": it complies when the GCG
suffix arrives intact, refuses an intact harmful request, bails out when the
prompt is mangled, and answers MedQA with accuracy that decays under noise.
It is a SIMULATOR. Its outputs are not results.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from backends import MockBackend, _gold_key   # noqa: E402  (reuse one behaviour model)

MODEL_NAME = "fake-model"
_ODD = re.compile(r"[^A-Za-z0-9\s.,'\"?!()\-:;/]")


def approx_tokens(text: str) -> int:
    """Rough token count. Real servers report exact counts; the harness must not
    care which, so this deliberately does NOT match any real tokenizer."""
    return max(1, len(text) // 4)


class Sim:
    def __init__(self, gold_map: dict | None = None, latency_ms: int = 0):
        self.mock = MockBackend(gold_map=gold_map or {})
        self.latency_ms = latency_ms
        self.calls = 0

    def reply(self, user_content: str, system: str = "") -> str:
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        return self.mock.generate(user_content, system).text


# NOTE: SIM/API used to be module globals, which silently broke the moment two
# servers ran in one process -- the second serve() overwrote the first one's API
# kind, so the OpenAI server started answering with Ollama routing. They now live
# on the server instance and the handler reads them from self.server.

# vicuna_v1.1 template, so /v1/completions input can be unwrapped back to the
# user turn -- exactly what a real FastChat server does internally.
_VICUNA = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*$", re.DOTALL)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # keep the test output readable
        pass

    @property
    def sim(self):
        return self.server.sim

    @property
    def api(self):
        return self.server.api

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        if self.api == "openai" and self.path.rstrip("/") == "/v1/models":
            return self._send({"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "created": int(time.time()),
                 "owned_by": "fastchat", "permission": []}]})
        if self.api == "ollama" and self.path.rstrip("/") == "/api/tags":
            return self._send({"models": [
                {"name": MODEL_NAME, "model": MODEL_NAME, "size": 1,
                 "digest": "0" * 64, "modified_at": "2026-01-01T00:00:00Z"}]})
        self._send({"error": "not found"}, 404)

    # --------------------------------------------------------------- POST
    def do_POST(self):
        p = self.path.rstrip("/")
        try:
            body = self._read()
        except Exception as e:
            return self._send({"error": {"message": str(e)}}, 400)

        # ---- FastChat: /v1/completions (templated mode) ----
        if p == "/v1/completions":
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            m = _VICUNA.search(prompt)
            user = m.group(1) if m else prompt
            system = prompt[:m.start()].strip() if m else ""
            text = self.sim.reply(user, system)
            return self._send({
                "id": f"cmpl-{random.randrange(1 << 40):x}", "object": "text_completion",
                "created": int(time.time()), "model": body.get("model", MODEL_NAME),
                "choices": [{"index": 0, "text": text, "logprobs": None,
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": approx_tokens(prompt),
                          "completion_tokens": approx_tokens(text),
                          "total_tokens": approx_tokens(prompt) + approx_tokens(text)}})

        # ---- FastChat / OpenAI: /v1/chat/completions ----
        if p == "/v1/chat/completions":
            msgs = body.get("messages", [])
            system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
            user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            text = self.sim.reply(user, system)
            return self._send({
                "id": f"chatcmpl-{random.randrange(1 << 40):x}", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model", MODEL_NAME),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": approx_tokens(system + user),
                          "completion_tokens": approx_tokens(text),
                          "total_tokens": approx_tokens(system + user) + approx_tokens(text)}})

        # ---- Ollama: /api/chat ----
        if p == "/api/chat":
            msgs = body.get("messages", [])
            system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
            user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            text = self.sim.reply(user, system)
            return self._send({
                "model": body.get("model", MODEL_NAME),
                "created_at": "2026-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": text},
                "done": True, "done_reason": "stop",
                "total_duration": 1_000_000, "load_duration": 1000,
                "prompt_eval_count": approx_tokens(system + user),
                "prompt_eval_duration": 1000,
                "eval_count": approx_tokens(text), "eval_duration": 1000})

        self._send({"error": "not found"}, 404)


def serve(port: int, api: str, gold_map: dict, latency_ms: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    srv.sim = Sim(gold_map, latency_ms)   # per-instance, so two servers can coexist
    srv.api = api
    return srv


def load_gold_map() -> dict:
    path = os.path.join(os.path.dirname(HERE), "data", "eval_sets", "medqa_test.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[_gold_key(r["user_content"])] = r.get("gold", "C")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--latency-ms", type=int, default=0)
    a = ap.parse_args()
    srv = serve(a.port, a.api, load_gold_map(), a.latency_ms)
    print(f"fake {a.api} server on http://127.0.0.1:{a.port}  (simulator -- not a result source)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
