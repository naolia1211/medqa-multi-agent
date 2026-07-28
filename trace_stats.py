"""
trace_stats.py — Tổng hợp accuracy + metric TỪNG AGENT lấy TỪ GenAI Traces (không dùng run/metric).

  python trace_stats.py                          # mọi trace trong experiment medqa-v3
  python trace_stats.py --variant v3 --split dev.jsonl

Đọc token native (mlflow.chat.tokenUsage), status, latency của từng span agent trong mỗi trace,
rồi gộp lại — đây là bản "per-agent metric" theo hướng GenAI (thay cho metric của run cũ).
"""
import argparse
from collections import defaultdict

from common import load_config


def dedup_latest(traces):
    """
    Chạy lại eval tích lũy trace trùng (cùng variant/split/q_index) -> giữ trace MỚI NHẤT
    (request_time lớn nhất) cho mỗi (variant, split, q_index); trace thiếu q_index giữ nguyên.
    Thuần -> test được. Trả (deduped, n_dropped).
    """
    latest = {}          # key -> (request_time, index_giữ)
    keep = list(range(len(traces)))
    drop = set()
    for idx, t in enumerate(traces):
        tags = t["tags"]
        q = tags.get("q_index")
        if q is None:
            continue
        key = (tags.get("variant"), tags.get("split"), q)
        rt = t.get("request_time") or 0
        if key in latest:
            prev_rt, prev_idx = latest[key]
            drop.add(prev_idx if prev_rt <= rt else idx)
            if prev_rt <= rt:
                latest[key] = (rt, idx)
        else:
            latest[key] = (rt, idx)
    deduped = [t for i, t in enumerate(traces) if i not in drop]
    return deduped, len(drop)


def aggregate_agent_stats(traces):
    """
    Thuần (không phụ thuộc MLflow) -> test được.
    traces: list {"tags": {...}, "spans": [{"agent","status","latency_s","tokens_in","tokens_out"}]}
    -> (overall, per_agent).
    """
    n = len(traces)
    correct = sum(1 for t in traces if str(t["tags"].get("correct")) == "True")
    invalid = sum(1 for t in traces if str(t["tags"].get("invalid")) == "True")
    per = defaultdict(lambda: {"calls": 0, "parse_fail": 0, "latency": 0.0,
                               "tokens_in": 0, "tokens_out": 0})
    for t in traces:
        for sp in t["spans"]:
            a = sp.get("agent")
            if not a:
                continue
            p = per[a]
            p["calls"] += 1
            p["parse_fail"] += 1 if sp.get("status") == "parse_fail" else 0
            p["latency"] += sp.get("latency_s") or 0
            p["tokens_in"] += sp.get("tokens_in") or 0
            p["tokens_out"] += sp.get("tokens_out") or 0
    overall = {"n": n, "correct": correct, "accuracy": correct / max(1, n),
               "invalid": invalid, "invalid_rate": invalid / max(1, n)}
    return overall, dict(per)


def load_traces_from_mlflow(cfg, variant=None, split=None):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.tracing.constant import SpanAttributeKey
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment"])
    if exp is None:
        return []
    df = mlflow.search_traces(locations=[exp.experiment_id], max_results=10000)
    rt_by_id = dict(zip(df["trace_id"], df["request_time"])) if len(df) else {}
    c = MlflowClient()
    out = []
    for tid in list(df["trace_id"]):
        tr = c.get_trace(tid)
        tags = dict(tr.info.tags)
        if variant and tags.get("variant") != variant:
            continue
        if split and tags.get("split") != split:
            continue
        rt = rt_by_id.get(tid)
        rt = rt.value if hasattr(rt, "value") else rt   # pandas Timestamp -> ns int
        spans = []
        for s in tr.data.spans:
            u = s.attributes.get(SpanAttributeKey.CHAT_USAGE) or {}
            spans.append({"agent": s.attributes.get("agent"),
                          "status": s.attributes.get("status"),
                          "latency_s": s.attributes.get("latency_s"),
                          "tokens_in": u.get("input_tokens"),
                          "tokens_out": u.get("output_tokens")})
        out.append({"tags": tags, "spans": spans, "request_time": rt})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    cfg = load_config()
    traces = load_traces_from_mlflow(cfg, args.variant, args.split)
    traces, dropped = dedup_latest(traces)   # bỏ trace trùng (lần chạy lại) -> giữ mới nhất
    if dropped:
        print(f"(bỏ {dropped} trace trùng q_index — giữ bản mới nhất)")
    overall, per = aggregate_agent_stats(traces)

    print("\n" + "=" * 60)
    flt = f"variant={args.variant or '*'} split={args.split or '*'}"
    print(f"TRACE STATS ({flt})")
    print(f"  traces: {overall['n']}   accuracy: {overall['correct']}/{overall['n']} "
          f"= {100*overall['accuracy']:.1f}%   invalid: {overall['invalid']} "
          f"({100*overall['invalid_rate']:.1f}%)")
    print("  per-agent (gộp từ span trong trace):")
    for a, s in per.items():
        calls = max(1, s["calls"])
        print(f"    [{a}] calls={s['calls']} parse_fail={s['parse_fail']} "
              f"lat_TB={s['latency']/calls:.1f}s "
              f"tok_in_TB={s['tokens_in']/calls:.0f} tok_out_TB={s['tokens_out']/calls:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
