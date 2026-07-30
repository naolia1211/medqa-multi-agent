"""Sinh docs/metrics_summary.json — đầy đủ giá trị theo bộ eval metric."""
import sys, json
sys.path.insert(0, ".")
import metrics as M

N = 1273
FILES = {
    "direct-llm": "index/eval_direct_results.DIRECT.json",
    "rag-only": "index/eval_rag_results.json",
    "v3-mem-off": "index/eval_agents_results.OFF.json",
    "v3-mem-on": "index/eval_agents_results.ON.json",
}
data = {k: M.load_results(v) for k, v in FILES.items()}

summary = {"dataset": "MedQA-USMLE test", "n_total": N, "model": "gemma-4-12B-it-qat-q4_0",
           "seed": 42, "per_variant": {}, "pairwise": []}

for k, r in data.items():
    acc, corr, _ = M.accuracy(r, N)
    ir, inv, _ = M.invalid_rate(r, N)
    lo, hi = M.bootstrap_ci(M.correct_flags(r))
    cl = M.cost_latency(r)
    summary["per_variant"][k] = {
        "accuracy": round(acc, 4), "correct": corr, "total": N,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "invalid": inv, "invalid_rate": round(ir, 4),
        "latency_mean_s": round(cl.get("latency_mean_s", 0), 1),
        "latency_median_s": round(cl.get("latency_median_s", 0), 1),
        "latency_p95_s": round(cl.get("latency_p95_s", 0), 1),
        "tokens_total": cl.get("tokens_total", 0),
    }

PAIRS = [("direct-llm", "rag-only"), ("rag-only", "v3-mem-off"),
         ("v3-mem-off", "v3-mem-on"), ("v3-mem-on", "direct-llm")]
for base, new in PAIRS:
    al = M.align(data[base], data[new])
    wlt = M.win_loss_tie(al); mc = M.mcnemar(al)
    v3f = [1 if a["v3_ok"] else 0 for a in al]; v0f = [1 if a["v0_ok"] else 0 for a in al]
    lo, hi = M.bootstrap_ci_diff(v3f, v0f)
    summary["pairwise"].append({
        "base": base, "new": new,
        "accuracy_gain": round((sum(v3f) - sum(v0f)) / len(al), 4),
        "win": wlt["win"], "loss": wlt["loss"], "tie": wlt["tie"],
        "mcnemar_b": mc["b"], "mcnemar_c": mc["c"], "mcnemar_p": round(mc["p_value"], 5),
        "diff_ci95": [round(lo, 4), round(hi, 4)],
        "significant": mc["p_value"] < 0.05,
    })

json.dump(summary, open("docs/metrics_summary.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
