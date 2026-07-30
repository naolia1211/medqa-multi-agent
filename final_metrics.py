"""#8 — Metric FINAL: so sánh 4 variant trên test 1273 (mẫu số cố định)."""
import sys, json
sys.path.insert(0, ".")
import metrics as M

N = 1273
VARIANTS = [
    ("Direct LLM (không RAG)", "index/eval_direct_results.DIRECT.json"),
    ("RAG-only (single-LLM+RAG)", "index/eval_results.RAGONLY.json"),
    ("V3 multi-agent OFF (mem tắt)", "index/eval_agents_results.OFF.json"),
    ("V3 multi-agent ON (mem bật)", "index/eval_agents_results.ON.json"),
]
data = {name: M.load_results(path) for name, path in VARIANTS}

print("=" * 78)
print("METRIC FINAL — MedQA-USMLE test (mẫu số cố định = 1273) — 4 variant, cùng model gemma-4-12B")
print("=" * 78)
print(f"\n{'Variant':<32}{'Accuracy':>16}{'95% CI':>16}{'Invalid':>9}{'Lat med':>9}")
print("-" * 82)
rank = []
for name, _ in VARIANTS:
    r = data[name]
    acc, corr, _ = M.accuracy(r, N)
    ir, inv, _ = M.invalid_rate(r, N)
    lo, hi = M.bootstrap_ci(M.correct_flags(r))
    cl = M.cost_latency(r)
    lat = cl.get("latency_median_s", 0)
    rank.append((name, acc, corr))
    print(f"{name:<32}{corr:>5}/{N} = {acc*100:>5.2f}%{f'{lo*100:.1f}-{hi*100:.1f}%':>16}{f'{inv}':>9}{f'{lat:.0f}s':>9}")

print("\n" + "-" * 82)
print("XẾP HẠNG:", " > ".join(f"{n.split('(')[0].strip()} {a*100:.1f}%" for n, a, _ in sorted(rank, key=lambda x: -x[1])))

# so sánh từng cặp có ý nghĩa (theo trình tự nâng cấp pipeline)
PAIRS = [
    ("RAG-only", "Direct LLM (không RAG)", "RAG-only (single-LLM+RAG)", "RAG có giúp so với Direct?"),
    ("V3-OFF vs RAG-only", "RAG-only (single-LLM+RAG)", "V3 multi-agent OFF (mem tắt)", "Multi-agent có giúp so với RAG-only?"),
    ("V3-ON vs V3-OFF", "V3 multi-agent OFF (mem tắt)", "V3 multi-agent ON (mem bật)", "Memory có giúp trong V3?"),
    ("Direct vs V3-ON", "V3 multi-agent ON (mem bật)", "Direct LLM (không RAG)", "Direct so với hệ đầy đủ nhất?"),
]
print("\n" + "=" * 78)
print("SO SÁNH GHÉP CẶP (paired McNemar exact + bootstrap CI của hiệu)")
print("=" * 78)
for label, base_name, new_name, q in PAIRS:
    al = M.align(data[base_name], data[new_name])   # v0=base, v3=new
    wlt = M.win_loss_tie(al); mc = M.mcnemar(al)
    v3f = [1 if a["v3_ok"] else 0 for a in al]; v0f = [1 if a["v0_ok"] else 0 for a in al]
    lo, hi = M.bootstrap_ci_diff(v3f, v0f)
    gain = (sum(v3f) - sum(v0f)) / len(al) * 100
    sig = "CÓ ý nghĩa" if mc["p_value"] < 0.05 else "chưa ý nghĩa"
    print(f"\n▸ {label}: {q}")
    print(f"   {base_name.split('(')[0].strip()} → {new_name.split('(')[0].strip()}")
    print(f"   GAIN = {gain:+.2f}đ | Win/Loss/Tie = {wlt['win']}/{wlt['loss']}/{wlt['tie']}")
    print(f"   McNemar p = {mc['p_value']:.4f} ({sig}) | bootstrap 95% CI hiệu = [{lo*100:+.2f}, {hi*100:+.2f}]đ")
