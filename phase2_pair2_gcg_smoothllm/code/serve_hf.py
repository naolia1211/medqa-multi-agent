#!/usr/bin/env python3
"""A minimal OpenAI-compatible server for Vicuna, on plain transformers.

The fallback for when vLLM will not install or will not run — notably on a
Kaggle P100, which is compute capability 6.0 while vLLM needs 7.0+. It uses
whatever torch/transformers the box already has, so there is nothing heavy to
install and nothing to conflict with.

    python serve_hf.py --model lmsys/vicuna-7b-v1.3 --port 8080

Serves exactly what the harness uses: GET /v1/models and POST /v1/completions
(prompt in, text out, with a real token count). The vicuna_v1.1 template is
applied client-side by the harness, so this server deliberately knows nothing
about chat templates — the bytes on the wire match Phase-1 verification.

It batches. Requests that arrive close together are padded and generated
together, which is what makes ~4,000 short generations finish in tens of minutes
instead of hours. Left padding, because generation continues from the right.
"""
from __future__ import annotations

import argparse, json, queue, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"model": None, "tok": None, "name": "", "device": "cuda"}
JOBS: "queue.Queue" = queue.Queue()


class Job:
    __slots__ = ("prompt", "max_tokens", "temperature", "done", "text", "ptoks", "ctoks", "err")

    def __init__(self, prompt, max_tokens, temperature):
        self.prompt, self.max_tokens, self.temperature = prompt, max_tokens, temperature
        self.done = threading.Event()
        self.text = ""
        self.ptoks = self.ctoks = 0
        self.err = ""


def worker(batch_size: int, wait_ms: int) -> None:
    import torch
    tok, model = STATE["tok"], STATE["model"]
    while True:
        first = JOBS.get()
        batch = [first]
        deadline = time.time() + wait_ms / 1000.0
        # Group only jobs that share generation settings: a batch generates with
        # one config, so mixing max_tokens would truncate or overrun some of them.
        while len(batch) < batch_size and time.time() < deadline:
            try:
                nxt = JOBS.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if (nxt.max_tokens, nxt.temperature) == (first.max_tokens, first.temperature):
                batch.append(nxt)
            else:
                JOBS.put(nxt)
                break
        try:
            enc = tok([j.prompt for j in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=2048).to(STATE["device"])
            gen_kw = dict(max_new_tokens=first.max_tokens,
                          pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id)
            if first.temperature and first.temperature > 0:
                gen_kw.update(do_sample=True, temperature=first.temperature)
            else:
                gen_kw.update(do_sample=False)
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kw)
            plen = enc["input_ids"].shape[1]
            for i, (j, row) in enumerate(zip(batch, out)):
                new = row[plen:]
                j.text = tok.decode(new, skip_special_tokens=True)
                # index by position, not batch.index(j): two identical prompts
                # would otherwise both read the first one's mask
                j.ptoks = int(enc["attention_mask"][i].sum())
                # pad_token often IS eos_token, so counting non-pad tokens would
                # drop the real eos; count up to the first eos instead
                ids = new.tolist()
                j.ctoks = (ids.index(tok.eos_token_id) + 1
                           if tok.eos_token_id in ids else len(ids))
        except Exception as e:
            for j in batch:
                j.err = f"{type(e).__name__}: {e}"
        finally:
            for j in batch:
                j.done.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send({"object": "list",
                               "data": [{"id": STATE["name"], "object": "model",
                                         "created": int(time.time()),
                                         "owned_by": "local"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/completions":
            return self._send({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception as e:
            return self._send({"error": {"message": str(e)}}, 400)
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        job = Job(prompt, int(body.get("max_tokens", 256)), float(body.get("temperature", 0.0)))
        JOBS.put(job)
        job.done.wait()
        if job.err:
            return self._send({"error": {"message": job.err}}, 500)
        self._send({"id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion",
                    "created": int(time.time()), "model": body.get("model", STATE["name"]),
                    "choices": [{"index": 0, "text": job.text, "logprobs": None,
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": job.ptoks,
                              "completion_tokens": job.ctoks,
                              "total_tokens": job.ptoks + job.ctoks}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lmsys/vicuna-7b-v1.3")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--wait-ms", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model} ({a.dtype}) — first run downloads ~13 GB", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # decoder-only: generation continues at the right edge
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=getattr(torch, a.dtype),
        low_cpu_mem_usage=True,
        device_map="auto" if a.device == "cuda" else None)
    model.eval()
    STATE.update(model=model, tok=tok, name=a.model,
                 device=next(model.parameters()).device)
    print(f"loaded on {STATE['device']}", flush=True)

    threading.Thread(target=worker, args=(a.batch_size, a.wait_ms), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"serving http://127.0.0.1:{a.port}/v1  (batch {a.batch_size})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
