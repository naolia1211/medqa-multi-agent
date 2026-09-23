# Runbook — arm A (Vicuna-7B-v1.3 fp16)

Arm B is finished and needed no GPU. This is what is left: the defense curve on
the model the GCG suffix was actually optimised against — the only place where a
"defense success rate" figure means anything, because it is the only place where
there is an attack to reduce.

**One command on any GPU box:**

```
python arm_a_run.py
```

It installs vLLM if missing, downloads the fp16 weights, serves them, waits,
runs the sanity check, then the full sweep, writes the reports and shuts the
server down.

---

## 1. Budget — smaller than it first looked

| | |
|---|---|
| GPU | one card with ≥16 GB (RTX 3090, A10, L4, T4 at a squeeze) |
| Model download | ~13 GB, once per box |
| Sanity check | ~100 calls, 2–3 min |
| Safety C0–C3 + benign C0/C3 | ~3,900 calls, 20–30 min batched |
| q/N sweep | ~15 min |
| **Total** | **about one GPU-hour, setup included** |

Earlier estimates said 2–3 hours because they assumed MedQA on Vicuna. That is
now off by default — see §5.

## 2. Where to run it

**Vast.ai** — the Phase-1 path, and the only one that bills you:

```bash
vastai start instance 50898440     # if it still exists
vastai show instances              # note the ssh host and port
scp -P <port> -r phase2_defense root@<ip>:/workspace/
ssh -p <port> root@<ip>
cd /workspace/phase2_defense && python arm_a_run.py
```

**Kaggle Notebooks** — free, a P100 or 2×T4 with 16 GB, 30 h/week. No payment,
no SSH. Upload the folder as a dataset or clone it, enable **GPU** *and*
**Internet** in the notebook settings (internet is off by default and the
download will fail without it), then in one cell:

```python
!cd /kaggle/working/phase2_defense && python arm_a_run.py
```

**Colab** — same idea; a T4 on the free tier fits fp16 7B tightly, so pass
`--gpu-mem 0.95 --max-len 1024` if it runs out of memory.

Any of the three produces identical numbers. Kaggle is worth trying first: it
costs nothing and the 30 h/week quota is far more than this needs.

## 3. The gate — do not skip it

`arm_a_run.py` runs the sanity check first and prints a loud warning, because
everything after it is only worth spending if C1 reproduces Phase 1.

**C1 ASR must come out near 1.00.** Phase 1 measured 25/25 on the held-out set.

| Symptom | Likely cause |
|---|---|
| ASR near 0, responses are refusals | wrong weights — a quantized build instead of fp16 |
| ASR ≈ 0.08 | the suffix is not reaching the model; check `prompt_mode: templated` |
| `unscored_rate` above 0 | empty generations — check `max_new_tokens` and `vllm_server.log` |
| Server never healthy | out of VRAM; lower `--gpu-mem` and `--max-len`, or use a bigger card |

To stop after the gate and decide before committing the box:

```
python arm_a_run.py --sanity-only
```

## 4. Why vLLM and not the Phase-1 FastChat stack

The harness applies the `vicuna_v1.1` template **client-side** and POSTs the
finished string to `/v1/completions`, so the server never needs to know the
template. That drops the dependency on `fschat 0.2.20` / `transformers 4.28.1`,
which no longer install cleanly beside a current torch. The bytes on the wire are
identical to Phase-1 verification.

If you reuse instance 50898440 and its `gcg` conda env is intact, the old
FastChat commands still work — start them by hand and run
`python arm_a_run.py --no-serve --port 8080`.

**fp16, never a GGUF or a quantized copy.** The suffix was optimised against fp16
logits; quantizing changes them enough that the ASR stops being comparable with
Phase 1, and the comparison is the whole point.

## 5. Why MedQA is off by default here

Vicuna-7B is a general 7B model taking a medical licensing exam: it scores near
chance, so a C0→C3 accuracy drop measured there is mostly noise. It is also not
this system's base model. MedQA belongs to arm B, where it is already measured on
gemma4 (69.0% → 63.0%, p = 0.21).

Add `--with-medqa` only if you want the comparison for its own sake, and say in
the report that Vicuna's MedQA baseline is too weak to carry a utility claim.

## 6. Bringing the results home

The logs are the deliverable. Copy them **before** destroying the instance —
billing continues while it runs.

```bash
scp -P <port> -r root@<ip>:/workspace/phase2_defense/results ./results_vicuna
scp -P <port> -r root@<ip>:/workspace/phase2_defense/reports ./reports_vicuna
vastai stop instance 50898440
```

On Kaggle or Colab, download `results/` and `reports/` from the file browser.

## 7. Merging the two arms

```
python make_report.py --logs "results/*.jsonl" "results_vicuna/*.jsonl" --out reports/metrics
python rescore.py     --logs "results/*.jsonl" "results_vicuna/*.jsonl"
```

`make_report.py` keys on `backend`, so the two arms come out as separate columns
in one table. **That table is figure 1 of the report** — the same suffix, the
same defense, two models, two completely different security stories.

Then fill the four `[__]` cells on slide 8 of the deck from
`reports/metrics.md`, and §6.2 of `REPORT_SKELETON.md`.

## 8. While the box is alive, record the demo

The live demo needs a model that actually gets jailbroken, which is this one.

```
python demo.py --backend vicuna --config config_arma.yaml --compare gemma --pause
```

Capture the terminal. A recording is insurance against the GPU being
unavailable on presentation day — and that is 20% of the grade.
