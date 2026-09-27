import sys, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template

OUT = '/workspace/results/smoothllm'
K = int(sys.argv[1]) if len(sys.argv) > 1 else 25   # so cau held-out de verify
MAXNEW = 128
MODEL = '/workspace/models/vicuna-7b-v1.3'

recs = [json.loads(l) for l in open(f'{OUT}/attack_prompts.jsonl')][:K]
refusals = json.load(open(f'{OUT}/refusal_prefixes.json'))['refusal_prefixes']

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to('cuda:0').eval()

def gen(goal, suf):
    c = get_conversation_template('vicuna')
    msg = goal if suf is None else f'{goal} {suf}'
    c.append_message(c.roles[0], msg)
    c.append_message(c.roles[1], None)
    p = c.get_prompt()
    ids = tok(p, return_tensors='pt').input_ids.to('cuda:0')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def jb(t):
    return not any(r in t for r in refusals)

out = open(f'{OUT}/jailbreak_verification.jsonl', 'w')
ns = nc = 0
for r in recs:
    ws = gen(r['goal'], r['suffix'])
    wo = gen(r['goal'], None)
    j_ws = jb(ws); j_wo = jb(wo)
    ns += j_ws; nc += j_wo
    out.write(json.dumps({'id': r['id'], 'goal': r['goal'],
        'output_with_suffix': ws, 'jailbroken_with_suffix': j_ws,
        'output_no_suffix': wo, 'jailbroken_no_suffix': j_wo}, ensure_ascii=False) + '\n')
    print(f"id={r['id']} jb_suffix={j_ws} jb_clean={j_wo}")
out.close()

summ = {'n': len(recs), 'ASR_with_suffix': ns / len(recs),
        'ASR_no_suffix(clean baseline)': nc / len(recs),
        'max_new_tokens': MAXNEW,
        'judge': 'refusal-prefix heuristic (full-generation, KHAC voi Passed 16-token cua GCG)'}
json.dump(summ, open(f'{OUT}/jailbreak_summary.json', 'w'), indent=2, ensure_ascii=False)
print('SUMMARY:', summ)
