# Arm A on Kaggle Notebooks — step by step

Free, no card, ~30 GPU-hours a week. This run needs about one of them.

Kaggle's own numbers, worth knowing before you start: **P100** (16 GB) or
**T4 ×2** (2 × 16 GB), 32 GB system RAM, **~30 h/week**, **~12 h per session**,
`/kaggle/working` persistent at **~20 GB**, `/kaggle/temp` ephemeral and larger.
Accelerators and internet both need a **phone-verified** account.

---

## 0. Pick the right accelerator — this one matters

| | vLLM (fast) | `serve_hf.py` (fallback) |
|---|---|---|
| **T4 ×2** | ✅ works | ✅ works |
| **P100** | ❌ **will not run** | ✅ works |

vLLM needs CUDA compute capability ≥ 7.0. P100 is Pascal, 6.0. If you pick P100
you must use the fallback server — it is included and it works, just slower.

**Choose `GPU T4 x2`.** Same quota cost, both paths available.

## 1. Get the folder onto Kaggle

The clean way is as a Dataset, so it survives session restarts:

1. Zip `phase2_defense` on your machine.
2. kaggle.com → **Datasets** → **New Dataset** → upload the zip → name it
   `phase2-defense` → Create.
3. New Notebook → right panel **Input** → **Add Input** → your dataset.

It mounts read-only at `/kaggle/input/phase2-defense/`, so the first cell copies
it somewhere writable.

## 2. Notebook settings — three switches

Right panel → **Settings**:

- **Accelerator**: `GPU T4 x2`
- **Internet**: **On** — off by default, and without it the weight download
  fails with a confusing connection error rather than a clear one
- **Persistence**: Files only (optional; keeps `/kaggle/working` between runs)

## 3. Cell 1 — set up

```python
import os, shutil, subprocess, sys

# Weights go to /kaggle/temp: they are ~13 GB and /kaggle/working is only ~20 GB
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SRC = "/kaggle/input/phase2-defense"
DST = "/kaggle/working/phase2_defense"
if not os.path.exists(DST):
    inner = os.path.join(SRC, "phase2_defense")
    shutil.copytree(inner if os.path.exists(inner) else SRC, DST)
os.chdir(DST)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "requests", "hf_transfer"])
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
```

Expect two `Tesla T4, 15360 MiB` lines.

## 4. Cell 2 — the gate, before spending the session

```python
!cd /kaggle/working/phase2_defense && python arm_a_run.py --sanity-only --workers 8
```

First run downloads ~13 GB and installs vLLM, so give it 15–25 minutes.

**Then read the ASR it prints. It must be near 1.00.** Phase 1 measured 25/25 on
this held-out set. If it comes out near 0, stop — `RUNBOOK_VICUNA.md` §3 has the
symptom table (quantized weights, wrong `prompt_mode`, empty generations).
Continuing past a broken gate wastes the whole session.

### If vLLM fails to install or start

Kaggle preinstalls its own torch, and vLLM sometimes fights it. Use the fallback
server instead — it uses whatever torch is already there, so there is nothing to
conflict with:

```python
import subprocess, sys, time, urllib.request
srv = subprocess.Popen([sys.executable, "serve_hf.py",
                        "--model", "lmsys/vicuna-7b-v1.3",
                        "--port", "8080", "--batch-size", "16"],
                       cwd="/kaggle/working/phase2_defense")
for _ in range(180):                       # wait for the weights to load
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5); break
    except Exception: time.sleep(10)
print("server up")
```

then run every later cell with `--no-serve --port 8080`.

## 5. Cell 3 — the full run

```python
!cd /kaggle/working/phase2_defense && python arm_a_run.py --workers 8
```

Safety C0–C3, benign C0/C3, the metrics table, the strict-judge re-score and the
q/N sweep. 30–45 minutes on T4 ×2 with vLLM; 1.5–2.5 h on the fallback server.

Tight on time? Drop the sweep and run the two that matter:

```python
!cd /kaggle/working/phase2_defense && python run_eval.py --config config_arma.yaml \
    --backend vicuna --datasets safety --workers 8 --no-serve
```

Everything is resume-safe: if the session dies, re-run the same cell and it
continues from `results/*.jsonl`.

## 6. Cell 4 — record the demo while the model is loaded

This is 20% of the grade and it needs a model that actually gets jailbroken.

```python
!cd /kaggle/working/phase2_defense && python demo.py --backend vicuna \
    --config config_arma.yaml --N 6
```

Copy the whole cell output into a text file. It is your insurance against the
GPU being unavailable on presentation day.

## 7. Cell 5 — package the results before the session ends

`/kaggle/working` survives, but do not rely on it:

```python
!cd /kaggle/working/phase2_defense && zip -qr /kaggle/working/arm_a_results.zip results reports
!ls -la /kaggle/working/arm_a_results.zip
```

Download it from the right panel → **Output**.

## 8. Back on your machine

```
mkdir results_vicuna && unzip arm_a_results.zip -d results_vicuna
python make_report.py --logs "results/*.jsonl" "results_vicuna/results/*.jsonl" --out reports/metrics
python rescore.py     --logs "results/*.jsonl" "results_vicuna/results/*.jsonl"
```

The merged table has one column per model — **that is figure 1 of the report**.
Then fill the four `[__]` cells on slide 8 of the deck and §6.2 of
`REPORT_SKELETON.md`.

---

## Things that go wrong, and what they mean

| Symptom | Cause |
|---|---|
| `ConnectionError` while downloading | Internet is off in Settings |
| `No space left on device` | `HF_HOME` not set — weights landed in `/kaggle/working` |
| vLLM: `not supported`, `compute capability` | P100 selected; switch to T4 ×2 or use `serve_hf.py` |
| CUDA OOM on startup | `--gpu-mem 0.85 --max-len 1024`, or the fallback server |
| Session killed at 12 h | Per-session cap; re-run, it resumes |
| Accelerator greyed out | Account not phone-verified |
| Sanity ASR ≈ 0.08 | The suffix is not reaching the model — check `prompt_mode: templated` |
| Sanity ASR near 0 with refusals | Wrong weights: a quantized build, not fp16 |

## Quota

A full run costs ~1 h of the 30. Even with two or three false starts you are
nowhere near the limit — so run the gate on its own first rather than trying to
do everything in one shot.
