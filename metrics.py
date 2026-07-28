"""
metrics.py — Tính TOÀN BỘ metric CHÍNH THỨC MedQA từ file kết quả eval (V0 baseline + V3).

Sửa đúng công thức official:
  - Accuracy = # Correct / TỔNG số câu (mẫu số CỐ ĐỊNH; câu thiếu/lỗi/invalid = SAI).
  - Invalid response rate = # Invalid / Tổng (invalid = pred None/INVALID/không ∈ A–E).
  - Accuracy gain = Acc(V3) − Acc(V0).
  - Win / Loss / Tie (V3 vs V0, khớp theo index câu — assert gold trùng).
  - Bootstrap 95% CI (mỗi config + cho HIỆU, paired) + McNemar test (exact binomial).
  - Cost (token) + Latency (mean/median/p95).

Hàm lõi là PURE (test offline được); đọc file + CLI là lớp mỏng.

  python metrics.py --v3 index/eval_agents_results.json --n 1273
  python metrics.py --v0 index/eval_results.json --v3 index/eval_agents_results.json --n 1273
"""
import json
import math
import argparse

import numpy as np

VALID = set("ABCDE")


# ── phân loại 1 câu ─────────────────────────────────────────────────────────
def is_correct(r):
    return bool(r.get("ok"))


def is_invalid(r):
    """Invalid = không rút được đáp án hợp lệ (lỗi/parse-fail/ngoài A–E)."""
    p = r.get("pred")
    return (p is None) or (str(p).strip().upper() not in VALID)


# ── metric đơn config ───────────────────────────────────────────────────────
def accuracy(results, total=None):
    """(acc, correct, total). total mặc định = len(results); official: truyền total=1273.
    Câu THIẾU (total>len) hoặc lỗi/invalid -> KHÔNG cộng correct -> tính là SAI."""
    total = len(results) if total is None else total
    correct = sum(1 for r in results if is_correct(r))
    return (correct / total if total else 0.0), correct, total


def invalid_rate(results, total=None):
    total = len(results) if total is None else total
    inv = sum(1 for r in results if is_invalid(r))
    return (inv / total if total else 0.0), inv, total


def correct_flags(results):
    """list 0/1 theo index câu (đã sort) — dùng cho bootstrap."""
    return [1 if is_correct(r) else 0 for r in sorted(results, key=lambda x: x["i"])]


# ── so sánh V0 vs V3 (khớp câu) ──────────────────────────────────────────────
def align(v0, v3):
    """Khớp theo index i; assert gold trùng (đảm bảo CÙNG câu). -> list {i,gold,v0_ok,v3_ok}."""
    d0 = {r["i"]: r for r in v0}
    d3 = {r["i"]: r for r in v3}
    out = []
    for i in sorted(set(d0) & set(d3)):
        g0, g3 = d0[i].get("gold"), d3[i].get("gold")
        if g0 != g3:
            raise ValueError(f"Gold lệch ở i={i}: V0={g0} V3={g3} — KHÔNG cùng câu (sai seed/split/limit?)")
        out.append({"i": i, "gold": g0, "v0_ok": is_correct(d0[i]), "v3_ok": is_correct(d3[i])})
    return out


def win_loss_tie(aligned):
    win = sum(1 for a in aligned if a["v3_ok"] and not a["v0_ok"])
    loss = sum(1 for a in aligned if not a["v3_ok"] and a["v0_ok"])
    tie = sum(1 for a in aligned if a["v3_ok"] == a["v0_ok"])
    return {"win": win, "loss": loss, "tie": tie, "n": len(aligned)}


# ── thống kê ────────────────────────────────────────────────────────────────
def bootstrap_ci(flags, n_boot=10000, alpha=0.05, seed=42):
    """CI cho tỉ lệ (mean mảng 0/1). Trả (lo, hi)."""
    a = np.asarray(flags, dtype=np.float64)
    if a.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def bootstrap_ci_diff(v3_flags, v0_flags, n_boot=10000, alpha=0.05, seed=42):
    """CI cho HIỆU accuracy (V3−V0), bootstrap THEO CẶP (paired)."""
    d = np.asarray(v3_flags, dtype=np.float64) - np.asarray(v0_flags, dtype=np.float64)
    if d.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def mcnemar(aligned):
    """McNemar EXACT (two-sided binomial, p=0.5) trên cặp bất đồng.
    b = V0 đúng & V3 sai (loss); c = V3 đúng & V0 sai (win)."""
    b = sum(1 for a in aligned if a["v0_ok"] and not a["v3_ok"])
    c = sum(1 for a in aligned if a["v3_ok"] and not a["v0_ok"])
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "p_value": 1.0, "chi2_cc": 0.0}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) * (0.5 ** nd))
    chi2_cc = (abs(b - c) - 1) ** 2 / nd     # chi-square hiệu chỉnh liên tục (tham khảo)
    return {"b": b, "c": c, "p_value": p, "chi2_cc": chi2_cc}


def cost_latency(results):
    lat = [r["latency"] for r in results if r.get("latency") is not None]
    toks = [r["n_tokens"] for r in results if r.get("n_tokens") is not None]
    tin = tout = 0
    for r in results:
        for t in (r.get("trace") or []):
            tin += t.get("tokens_in") or 0
            tout += t.get("tokens_out") or 0
    out = {"tokens_in_total": tin, "tokens_out_total": tout,
           "tokens_total": (tin + tout) if (tin or tout) else sum(toks)}
    if lat:
        s = sorted(lat)
        out.update({"latency_mean_s": sum(lat) / len(lat), "latency_median_s": s[len(s) // 2],
                    "latency_p95_s": s[min(len(s) - 1, int(len(s) * 0.95))], "latency_total_s": sum(lat)})
    return out


# ── đọc file + báo cáo ───────────────────────────────────────────────────────
def load_results(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("results", d) if isinstance(d, dict) else d


def _report_one(name, results, total):
    acc, corr, tot = accuracy(results, total)
    ir, inv, _ = invalid_rate(results, total)
    lo, hi = bootstrap_ci(correct_flags(results))
    cl = cost_latency(results)
    print(f"\n[{name}]  (n_kết_quả={len(results)}, mẫu_số={tot})")
    print(f"  Accuracy = {corr}/{tot} = {100*acc:.2f}%   (bootstrap 95% CI: {100*lo:.1f}–{100*hi:.1f}%)")
    print(f"  Invalid response rate = {inv}/{tot} = {100*ir:.2f}%")
    if "latency_mean_s" in cl:
        print(f"  Latency: mean {cl['latency_mean_s']:.1f}s · median {cl['latency_median_s']:.1f}s · "
              f"p95 {cl['latency_p95_s']:.1f}s")
    if cl["tokens_total"]:
        print(f"  Cost (token): in={cl['tokens_in_total']} out={cl['tokens_out_total']} "
              f"tổng={cl['tokens_total']}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", default=None, help="file kết quả baseline (eval_results.json)")
    ap.add_argument("--v3", required=True, help="file kết quả V3 (eval_agents_results.json)")
    ap.add_argument("--n", type=int, default=None, help="TỔNG số câu official (mẫu số cố định, vd 1273)")
    args = ap.parse_args()

    v3 = load_results(args.v3)
    print("=" * 60)
    print("METRIC CHÍNH THỨC MedQA")
    acc3 = _report_one("V3 multi-agent", v3, args.n)

    if args.v0:
        v0 = load_results(args.v0)
        acc0 = _report_one("V0 baseline (RAG-only)", v0, args.n)
        try:
            al = align(v0, v3)
        except ValueError as e:
            print(f"\n⚠️ Không so sánh được: {e}")
            return
        wlt = win_loss_tie(al)
        v3f = [1 if a["v3_ok"] else 0 for a in al]
        v0f = [1 if a["v0_ok"] else 0 for a in al]
        dlo, dhi = bootstrap_ci_diff(v3f, v0f)
        mc = mcnemar(al)
        print("\n" + "-" * 60)
        print(f"SO SÁNH V3 vs V0 (trên {wlt['n']} câu khớp)")
        print(f"  Accuracy gain = {100*(acc3-acc0):+.2f} điểm   "
              f"(bootstrap 95% CI của HIỆU: {100*dlo:+.1f} … {100*dhi:+.1f})")
        print(f"  Win/Loss/Tie = {wlt['win']} / {wlt['loss']} / {wlt['tie']}")
        print(f"  McNemar (b=loss={mc['b']}, c=win={mc['c']}): p-value = {mc['p_value']:.4f}"
              f"  {'(khác biệt CÓ ý nghĩa, p<0.05)' if mc['p_value']<0.05 else '(CHƯA có ý nghĩa thống kê)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
