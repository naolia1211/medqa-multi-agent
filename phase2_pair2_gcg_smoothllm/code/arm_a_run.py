#!/usr/bin/env python3
"""Arm A, end to end on a GPU box: serve Vicuna, run the grid, write the reports.

Works anywhere there is an NVIDIA GPU with >=16 GB and this folder present —
Vast.ai over SSH, or a Kaggle / Colab notebook cell. It installs vLLM if needed,
downloads the fp16 weights, starts an OpenAI-compatible server, waits for it,
runs the sanity check, then the full sweep, and shuts the server down.

    python arm_a_run.py                      # safety + benign  (~1 GPU-hour all in)
    python arm_a_run.py --sanity-only        # ~100 calls, just confirm ASR ~ 1.00
    python arm_a_run.py --with-medqa         # adds MedQA (see the note below)
    python arm_a_run.py --no-serve           # a server is already up on --port

Why vLLM rather than the Phase-1 FastChat stack: the harness applies the
`vicuna_v1.1` template CLIENT-side and POSTs the finished string to
/v1/completions, so the server never needs to know the template. That removes the
dependency on fschat 0.2.20 / transformers 4.28.1, which no longer install
cleanly next to a current torch. The bytes on the wire are identical to Phase 1.

fp16, NOT a quantized build: the suffix was optimised against fp16 logits, and a
quantized copy changes them enough that the ASR stops being comparable.

MedQA on Vicuna is off by default on purpose. Vicuna-7B is a general 7B model on
a medical licensing exam — it scores near chance, so a utility-drop measurement
there is noise, and it is not this system's base model anyway. MedQA belongs to
arm B, where it has already been measured on gemma4.
"""
from __future__ import annotations

import argparse, json, os, shutil, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "lmsys/vicuna-7b-v1.3"


def sh(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw).returncode


def have(mod: str) -> bool:
    try:
        __import__(mod); return True
    except ImportError:
        return False


def gpu_report() -> None:
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"])
    else:
        print("  [!] nvidia-smi not found -- is this box actually GPU-backed?")


def wait_health(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"  server up after {time.time()-t0:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
        if int(time.time() - t0) % 60 < 5:
            print(f"    still waiting... {time.time()-t0:.0f}s "
                  f"(first run downloads ~13 GB)", flush=True)
    return False


def write_config(port: int) -> str:
    """A config pointing the `vicuna` backend at this box's server."""
    import yaml
    src = os.path.join(HERE, "config.yaml")
    cfg = yaml.safe_load(open(src, encoding="utf-8"))
    cfg["backends"]["vicuna"].update({
        "base_url": f"http://127.0.0.1:{port}/v1",
        "model": MODEL,                 # vLLM reports the full repo id
        "prompt_mode": "templated",     # byte-matches Phase-1 verification
        "name": "vicuna-7b-v1.3",
        "timeout": 300,
    })
    out = os.path.join(HERE, "config_arma.yaml")
    yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"), sort_keys=False,
                   allow_unicode=True)
    print(f"  wrote {out}")
    return out


def stage(title: str, cmd: list[str]) -> bool:
    print(f"\n{'='*70}\n>> {title}\n{'='*70}", flush=True)
    t0 = time.time()
    rc = sh(cmd, cwd=HERE)
    print(f"[{'ok' if rc == 0 else 'FAILED'}, {time.time()-t0:.0f}s]")
    return rc == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--with-medqa", action="store_true")
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip stage 7 (16 configs x ~1300 calls -- hours)")
    ap.add_argument("--boot-timeout", type=int, default=2400)
    a = ap.parse_args()

    print("=" * 70)
    print("ARM A -- Vicuna-7B-v1.3 fp16: the model the GCG suffix was built on")
    print("=" * 70)
    gpu_report()

    for need, pkg in (("yaml", "pyyaml"), ("requests", "requests")):
        if not have(need) and not a.no_install:
            sh([sys.executable, "-m", "pip", "install", "-q", pkg])

    proc = None
    if not a.no_serve:
        if not have("vllm") and not a.no_install:
            print("\n  installing vLLM (several minutes, once per box)...")
            if sh([sys.executable, "-m", "pip", "install", "-q", "vllm"]) != 0:
                sys.exit("[!] vLLM install failed. Install it yourself, then re-run "
                         "with --no-install, or serve the model another way and use "
                         "--no-serve.")
        print(f"\n  starting the server on :{a.port} (first run downloads ~13 GB)")
        log = open(os.path.join(HERE, "vllm_server.log"), "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MODEL, "--dtype", "float16",
             "--served-model-name", MODEL,
             "--gpu-memory-utilization", str(a.gpu_mem),
             "--max-model-len", str(a.max_len),
             "--host", "127.0.0.1", "--port", str(a.port)],
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None)
        if not wait_health(a.port, a.boot_timeout):
            print("\n[!] the server never became healthy. Last lines of "
                  "vllm_server.log:")
            os.system(f"tail -30 {os.path.join(HERE, 'vllm_server.log')}")
            if proc:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            sys.exit(1)
    elif not wait_health(a.port, 60):
        sys.exit(f"[!] nothing answering on :{a.port}")

    cfg = write_config(a.port)
    py = sys.executable
    ok = True
    try:
        if not os.path.exists(os.path.join(HERE, "data", "eval_sets",
                                           "safety_prompts.jsonl")):
            stage("0. build evaluation sets", [py, "src/datasets_build.py"])

        # --- the gate: C1 must reproduce Phase 1 before anything expensive ---
        ok = stage("1. sanity check -- C1 only, ~100 calls",
                   [py, "run_eval.py", "--config", cfg, "--backend", "vicuna",
                    "--datasets", "safety", "--conditions", "C1",
                    "--workers", str(a.workers)])
        if ok:
            stage("   sanity report", [py, "make_report.py", "--logs",
                                       "results/eval_vicuna-7b-v1.3.jsonl"])
            print("\n" + "!" * 70)
            print("  CHECK THE ASR ABOVE BEFORE READING ON.")
            print("  Phase 1 measured 1.00 on the held-out set. If this is near 0,")
            print("  something is wrong with the weights or the template -- stop and")
            print("  see RUNBOOK_VICUNA.md section 3. Continuing would waste the box.")
            print("!" * 70, flush=True)

        if a.sanity_only or not ok:
            return

        ok = stage("2. safety set, all of C0-C3 -- the defense curve",
                   [py, "run_eval.py", "--config", cfg, "--backend", "vicuna",
                    "--datasets", "safety", "--workers", str(a.workers)]) and ok
        ok = stage("3. benign set, C0/C3 -- the false-refusal cost",
                   [py, "run_eval.py", "--config", cfg, "--backend", "vicuna",
                    "--datasets", "benign", "--workers", str(a.workers)]) and ok
        if a.with_medqa:
            stage("4. MedQA, C0/C3 (see the note at the top of this file)",
                  [py, "run_eval.py", "--config", cfg, "--backend", "vicuna",
                   "--datasets", "medqa", "--workers", str(a.workers)])

        stage("5. metrics table", [py, "make_report.py", "--logs",
                                   "results/*.jsonl", "--out", "reports/metrics"])
        stage("6. strict-judge re-score -- READ THIS before quoting any ASR",
              [py, "rescore.py", "--logs", "results/*.jsonl",
               "--out", "results/rescored.jsonl"])
        if not a.skip_sweep:
            stage("7. q/N sensitivity sweep",
                  [py, "sweep.py", "--backend", "vicuna", "--config", cfg,
                   "--datasets", "safety", "--q", "5", "10", "15", "20",
                   "--N", "2", "4", "6", "10", "--workers", str(a.workers)])
        else:
            print("\n  [skip] stage 7 q/N sweep (--skip-sweep)", flush=True)
    finally:
        if proc:
            print("\n  stopping the server")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                pass

    print(f"\n{'='*70}")
    print("done. Copy these home BEFORE destroying the instance:")
    print("    results/   reports/")
    print("\nThen merge both arms on your laptop:")
    print('    python make_report.py --logs "results/*.jsonl" "results_vicuna/*.jsonl" \\')
    print('                          --out reports/metrics')
    print("\nThat merged table is figure 1 of the report: same suffix, same defense,")
    print("two models, two completely different security stories.")


if __name__ == "__main__":
    main()
