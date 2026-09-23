# Step 1 — run this first

One command, on the machine where Ollama is running. No `pip install` needed:
the script uses only the Python standard library.

```
cd E:\AI-Defense\phase2_defense
python step1_check_transfer.py
```

It auto-detects your Ollama model tag. If it picks the wrong one:

```
ollama list
python step1_check_transfer.py --model <exact tag from that list>
```

## What it measures, and why both arms matter

| Arm | Prompt | Measures |
|---|---|---|
| **C1** | goal **+ GCG suffix** | jailbreak ASR on the undefended system |
| **C0** | goal alone | the clean refusal baseline |

C1 on its own is uninterpretable. If the model also complies with the bare goal,
the suffix is not what broke it — the model was never refusing. The number that
matters is the **lift**: `ASR(C1) − ASR(C0)`.

Phase 1 reference, on Vicuna-7B fp16: **100%** with suffix, **8%** without.

## Cost

200 calls, `max_tokens=128`, 2 workers. Expect roughly 10–30 minutes depending
on your GPU. Raise `--workers 4` only if there is VRAM headroom. Start with
`--n 25` if you want a first reading in a few minutes.

## Then

Tell me the verdict, or just say it is done — the script writes
`results/step1_summary.json` into this folder and I can read it from here.

The three outcomes and what each one means for the rest of the project are in
the script's own VERDICT block, and in section 6 of README.md.
