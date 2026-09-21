# SmoothLLM input pack — GCG universal / Vicuna-7B-v1.3

Sinh tu GCG universal (llm-attacks, Zou et al. 2023) tren Vicuna-7B-v1.3 fp16.

## File
- universal_suffix.json      : suffix chung cuoi cung.
- attack_prompts.jsonl/.csv   : 100 cau held-out (advbench row 25:125, KHONG dung khi train).
    - adv_prompt        = goal + ' ' + suffix (raw)
    - prompt_templated  = da boc template vicuna (USER/ASSISTANT) -> input cho model
    - in_gcg_testset    = True neu nam trong 25 cau test noi bo cua GCG (row 25:50)
- refusal_prefixes.json       : danh sach prefix tu choi -> judge jailbroken (khop GCG).
- smoothllm_config.json       : cau hinh + canh bao (fp16 vs Ollama quantized, 'Passed' != jailbreak that).
- jailbreak_verification.jsonl: (sinh sau khi GCG xong) output THAT cua model cho goal+suffix va goal-only.
- jailbreak_summary.json      : ASR_with_suffix vs ASR_no_suffix (baseline sach).

## Dung cho SmoothLLM
- C1 (no defense)   : cho model sinh tren prompt_templated, cham bang refusal_prefixes.
- C2 (SmoothLLM)    : perturb (q%, N ban sao, majority vote) len prompt_templated roi cham nhu tren.
- C0/C3 (MedQA)     : benign, KHONG gan suffix — tach biet, khong nam trong pack nay.

## Canh bao
Suffix toi uu tren fp16 HF. Neu eval chay tren Ollama/GGUF quantized, hieu qua GCG co the giam -> do lai ASR tren dung model eval.
