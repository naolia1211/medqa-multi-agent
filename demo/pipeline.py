"""
demo/pipeline.py — Sinh (generator) SỰ KIỆN STREAMING cho từng version, dùng ĐÚNG prompt thật
của pipeline (rag.py / agents.py / eval_v2.py / eval_direct.py). Mỗi variant yield các event dict:
  {"type":"phase","name":..,"detail":..}     — mốc (retrieve, lessons...)
  {"type":"agent","name":..}                  — bắt đầu 1 agent stream
  {"type":"token","agent":..,"text":..}       — token thinking real-time
  {"type":"agent_done","name":..,"tokens":..,"latency":..}
  {"type":"answer","pred":..,"gold":..,"correct":..}
  {"type":"metrics","latency":..,"tokens_total":..,"n_calls":..}
GPU/LLM đi qua Ollama /api/chat stream=true. Retrieval tái dùng retrieve_rerank.
"""
import os, sys, time, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, EmbedClient, get_qdrant, get_es
from retrieval import retrieve_rerank
from rag import format_context, build_mc_messages, parse_choice, MC_SYS
from agents import (REASONING_PROMPT, VERIFIER_PROMPT, AGGREGATOR_PROMPT,
                    format_evidence, format_options, parse_json_loose, _dumps, extract_answer)
from eval_v2 import AGGREGATOR_V2_PROMPT
from eval_direct import DIRECT_SYS, build_messages as direct_msgs
from long_mem import LongMemory, lessons_block

_CFG = load_config()
_OLL = _CFG["llm"]["ollama"]
_clients = {}

def _init():
    if not _clients:
        _clients["ec"] = EmbedClient(_CFG)
        _clients["qc"] = get_qdrant(_CFG)
        _clients["es"] = get_es(_CFG)
        m = dict(_CFG.get("memory", {})); m["enabled"] = True   # demo: luôn bật để V3-ON dùng lessons
        _cfg2 = dict(_CFG); _cfg2["memory"] = m
        _clients["mem"] = LongMemory(_cfg2, _clients["ec"])
    return _clients


def stream_chat(messages, num_ctx=None):
    """Gọi Ollama stream=true. Yield ('token', text) từng token, cuối cùng yield ('usage', {...})."""
    body = json.dumps({
        "model": _OLL["model"], "messages": messages, "stream": True, "think": False,
        "keep_alive": _OLL.get("keep_alive", "30m"),
        "options": {"temperature": 0.0, "num_ctx": num_ctx or _OLL.get("num_ctx", 8192)},
    }).encode()
    req = urllib.request.Request(_OLL["base_url"] + "/api/chat", body, {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=_OLL.get("timeout", 180))
    full, usage = "", {}
    for line in r:
        if not line.strip():
            continue
        d = json.loads(line)
        tok = (d.get("message") or {}).get("content", "")
        if tok:
            full += tok
            yield ("token", tok)
        if d.get("done"):
            usage = {"prompt_tokens": d.get("prompt_eval_count") or 0,
                     "completion_tokens": d.get("eval_count") or 0}
    yield ("full", full)
    yield ("usage", usage)


def _run_agent(name, messages, num_ctx=None):
    """Stream 1 agent; yield event; trả (full_text, tokens, latency) qua event agent_done."""
    yield {"type": "agent", "name": name}
    t0 = time.time(); full = ""; usage = {}
    for kind, val in stream_chat(messages, num_ctx):
        if kind == "token":
            yield {"type": "token", "agent": name, "text": val}
        elif kind == "full":
            full = val
        elif kind == "usage":
            usage = val
    tok = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    yield {"type": "agent_done", "name": name, "tokens": tok, "latency": round(time.time() - t0, 1)}
    yield {"type": "_result", "full": full, "tokens": tok}   # nội bộ, app không gửi ra


# ── retrieval helper (không stream) ──────────────────────────────────────────
def _retrieve(question):
    c = _init()
    ev = retrieve_rerank(_CFG, question, embed_client=c["ec"], qdrant=c["qc"], es=c["es"])
    return ev


# ── các generator theo variant ───────────────────────────────────────────────
def run_variant(variant, question, options, gold=None):
    """Yield mọi event cho 1 variant. options: {A..E}. gold: đáp án đúng (có thể None)."""
    t_all = time.time(); tok_total = 0; n_calls = 0
    letters = ", ".join(options.keys())

    def finish(pred):
        yield {"type": "answer", "pred": pred, "gold": gold,
               "correct": (pred == gold) if gold else None}
        yield {"type": "metrics", "latency": round(time.time() - t_all, 1),
               "tokens_total": tok_total, "n_calls": n_calls}

    # --- Direct: 0 retrieve, 1 call ---
    if variant == "direct":
        msgs = direct_msgs(question, options)
        res = {}
        for e in _run_agent("Direct LLM", msgs):
            if e["type"] == "_result": res = e
            else: yield e
        tok_total += res.get("tokens", 0); n_calls = 1
        yield from finish(parse_choice(res.get("full", ""), set(options.keys())))
        return

    # --- các variant có RAG: retrieve trước ---
    if variant in ("rag", "v2", "v3off", "v3on"):
        yield {"type": "phase", "name": "retrieve", "detail": "Hybrid dense+sparse+RRF+MedCPT rerank..."}
        ev = _retrieve(question)
        yield {"type": "phase", "name": "retrieve_done",
               "detail": f"{len(ev)} chunk: " + ", ".join(e["source"] for e in ev[:5])}

    if variant == "rag":
        msgs = build_mc_messages(question, options, format_context(ev))
        res = {}
        for e in _run_agent("RAG-only", msgs):
            if e["type"] == "_result": res = e
            else: yield e
        tok_total += res.get("tokens", 0); n_calls = 1
        yield from finish(parse_choice(res.get("full", ""), set(options.keys())))
        return

    # --- V2/V3: 3 agent ---
    lessons = ""
    if variant == "v3on":
        yield {"type": "phase", "name": "lessons", "detail": "Truy hồi lỗi quá khứ tương tự (long-term)..."}
        lessons = _init()["mem"].get_lessons(question)
        yield {"type": "phase", "name": "lessons_done",
               "detail": ("Nạp lessons vào Agent 1+2" if lessons else "Không tìm thấy lesson tương tự")}

    ev_fmt = format_evidence(ev); opt_fmt = format_options(options)
    # Agent 1 — Reasoning
    p1 = lessons_block(lessons) + REASONING_PROMPT.format(evidence=ev_fmt, question=question, options=opt_fmt)
    r1 = {}
    for e in _run_agent("Agent 1 · Reasoning", [{"role": "user", "content": p1}]):
        if e["type"] == "_result": r1 = e
        else: yield e
    tok_total += r1.get("tokens", 0); n_calls += 1
    a1 = parse_json_loose(r1.get("full", ""), ["candidates"]) or {"candidates": [], "reasoning": r1.get("full", "")[:800]}

    # Agent 2 — Verifier
    p2 = lessons_block(lessons) + VERIFIER_PROMPT.format(evidence=ev_fmt, question=question,
                                                         options=opt_fmt, analysis=_dumps(a1), n=len(options))
    r2 = {}
    for e in _run_agent("Agent 2 · Verifier", [{"role": "user", "content": p2}]):
        if e["type"] == "_result": r2 = e
        else: yield e
    tok_total += r2.get("tokens", 0); n_calls += 1
    a2 = parse_json_loose(r2.get("full", ""), ["concerns"]) or {"concerns": {}, "critique": r2.get("full", "")[:600]}

    # Agent 3 — Aggregator (V3: thấy evidence | V2: mù evidence)
    if variant == "v2":
        p3 = AGGREGATOR_V2_PROMPT.format(question=question, options=opt_fmt,
                                         candidates=_dumps(a1.get("candidates", "")), reasoning=_dumps(a1.get("reasoning", "")),
                                         concerns=_dumps(a2.get("concerns", "")), critique=_dumps(a2.get("critique", "")),
                                         letters=letters)
        agg_name = "Agent 3 · Aggregator (V2 — MÙ evidence)"
    else:  # v3off / v3on
        p3 = AGGREGATOR_PROMPT.format(evidence=ev_fmt, question=question, options=opt_fmt,
                                      critique=_dumps(a2.get("critique", a2)), letters=letters, n=len(options))
        agg_name = "Agent 3 · Aggregator (thấy evidence)"
    r3 = {}
    for e in _run_agent(agg_name, [{"role": "user", "content": p3}]):
        if e["type"] == "_result": r3 = e
        else: yield e
    tok_total += r3.get("tokens", 0); n_calls += 1
    yield from finish(extract_answer(r3.get("full", ""), valid=options.keys()))
