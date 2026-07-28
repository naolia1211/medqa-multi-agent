"""
agents.py — Multi-agent V3 (spec docs/V3_PIPELINE_SPEC.md, TRỪ long-term memory).

Luồng: [A] evidence (retrieve_rerank) -> short-term state -> 3 agent đọc/ghi chung:
  [B] Agent 1 Reasoning  -> [C] Agent 2 Verifier -> [D] Agent 3 Aggregator
  -> [E] extract_answer -> 1 ký tự A–E (không trích được -> "INVALID").

- Cả 3 agent CÙNG đọc `evidence` gốc (Aggregator cũng thấy evidence gốc — điểm V3).
- `state` là dict RAM: tạo mới mỗi câu, xoá khi sang câu sau. `trace` chỉ append.
- Mọi thay đổi `candidates[X]` đi kèm đúng 1 entry trace (agent nào ghi thì append trace đó).
- Long-term memory: chỉ có ĐIỂM CẮM (§9), không cài — `lessons` luôn rỗng ở đây.
- Mọi LLM call qua make_llm(cfg) (temperature 0, cấu hình ở config.yaml).
"""
import re
import json
import time

from common import load_config, EmbedClient, make_llm, get_qdrant, get_es
from retrieval import retrieve_rerank
from tracking import span as trace_span, set_trace_tags   # GenAI Tracing (no-op nếu tracker None)
from long_mem import lessons_block                          # §9 long-term memory (học từ câu sai)


# ── Parse JSON "lỏng" — LLM hay bọc code-fence / kèm reasoning trước JSON ─────
def _strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _balanced_objects(text):
    """Trả list substring là object {...} cân bằng ngoặc (bỏ qua ngoặc trong chuỗi)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
                start = None
    return objs


def parse_json_loose(raw, required_keys=None):
    """
    Rút object JSON đầu tiên hợp lệ (ưu tiên object CUỐI — thường JSON nằm sau phần reasoning).
    required_keys: nếu có, object phải chứa ít nhất một khoá trong đó.
    -> dict hoặc None.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    txt = _strip_fence(str(raw))
    for cand in [txt] + list(reversed(_balanced_objects(txt))):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if required_keys and not any(k in obj for k in required_keys):
            continue
        return obj
    return None


# ── [E] Trích đáp án ─────────────────────────────────────────────────────────
def extract_answer(raw, valid=None):
    """Ưu tiên JSON {"answer": "X"}; fallback ký tự A–E đứng độc lập. Không có -> 'INVALID'."""
    valid = set(valid) if valid else set("ABCDE")
    obj = parse_json_loose(raw, required_keys=["answer"])
    if obj is not None:
        a = str(obj.get("answer", "")).strip().upper()
        if len(a) == 1 and a in valid:
            return a
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    for c in re.findall(r"\b([A-E])\b", text.upper()):
        if c in valid:
            return c
    return "INVALID"


# ── Short-term memory ────────────────────────────────────────────────────────
def init_state(question, options, evidence):
    return {
        "question": question,
        "options": options,
        "evidence": evidence,                       # giữ nguyên, mọi agent đọc được
        "trace": [],                                # CHỈ APPEND
        "candidates": {k: {"status": "unreviewed", "reason": "", "evidence_ids": []}
                       for k in options},
    }


def format_evidence(evidence):
    return "\n".join(f"[{i}] {e['text']}" for i, e in enumerate(evidence, 1))


def format_options(options):
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def _dumps(x):
    return json.dumps(x, ensure_ascii=False)


# ── Đếm token THẬT (MedCPT WordPiece, dùng offline, cache sẵn) ────────────────
def make_token_counter(cfg):
    """Trả hàm đếm token nhất quán để theo dõi (không cần usage từ provider)."""
    try:
        from common import get_tokenizer, count_tokens
        tok = get_tokenizer(cfg["embedding"]["model"])
        return lambda text: count_tokens(tok, str(text))
    except Exception:
        return None


def _provider_of(llm):
    """Suy provider từ loại client (chỉ để gắn nhãn span mlflow.llm.provider)."""
    p = getattr(llm, "provider", None)     # ThrottledLLM expose sẵn provider của client gốc
    if p:
        return p
    n = type(llm).__name__.lower()
    if "ollama" in n:
        return "ollama"
    if "google" in n:
        return "google"
    if "llmclient" in n:
        return "openrouter"
    return None


def _call(llm, prompt, counter):
    """
    Gọi 1 LLM call, đo latency + token in/out.
    Ưu tiên token THẬT do provider trả về (chat_usage — đúng cả model :free / ollama);
    provider không kèm usage -> fallback tokenizer MedCPT (counter) -> luôn có token.
    Trả (raw, telemetry): latency, tokens_in, tokens_out, tokens_src ('provider'|'tokenizer'|None).
    """
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    if hasattr(llm, "chat_usage"):
        raw, usage = llm.chat_usage(msgs)
    else:
        raw, usage = llm.chat(msgs), None
    dt = time.time() - t0
    if usage and usage.get("prompt_tokens") is not None:
        tin, tout, src = usage.get("prompt_tokens"), usage.get("completion_tokens"), "provider"
    elif counter:
        tin, tout, src = counter(prompt), counter(raw), "tokenizer"
    else:
        tin, tout, src = None, None, None
    return raw, {"latency": round(dt, 3), "tokens_in": tin, "tokens_out": tout, "tokens_src": src}


# ── Prompt (theo spec §4/§5/§6) ──────────────────────────────────────────────
REASONING_PROMPT = """You are a physician solving a USMLE medical multiple-choice question.

EVIDENCE (excerpts from medical textbooks):
{evidence}

QUESTION:
{question}

OPTIONS:
{options}

Tasks:
1. Analyze the case using the evidence above.
2. List ALL still-plausible options, ranked from most to least likely. Do NOT narrow early:
   only drop an option when the evidence clearly REFUTES it. If unsure, KEEP it for Agent 2 to review
   (usually 2-4 options; do not pick only 1 unless it is truly obvious).
3. For each option, state which evidence number(s) support it.
4. If the evidence is insufficient, say what is missing - do NOT fabricate.

Return pure JSON (candidates ranked high -> low priority):
{{"candidates": ["C","B","A"], "reasoning": "...", "evidence_used": [1,3]}}"""

VERIFIER_PROMPT = """You are a critical-appraisal physician reviewing a colleague's analysis.

EVIDENCE:
{evidence}

QUESTION: {question}
OPTIONS:
{options}

COLLEAGUE'S ANALYSIS:
{analysis}

Task: For EACH of the {n} options A-E, note any evidence-based CONCERN (a specific fact from the
evidence that argues against that option), or "none" if you see no concern. You are ONLY flagging
concerns for the attending physician to weigh — do NOT eliminate, reject, or rank the options, and do
NOT pick an answer. Also give a short overall critique of the colleague's reasoning (point out any
over-reach or claims not supported by the evidence).

Return pure JSON:
{{"concerns": {{"A": "...", "B": "none", "C": "..."}}, "critique": "..."}}"""

AGGREGATOR_PROMPT = """You are an attending physician answering a USMLE multiple-choice question.

EVIDENCE:
{evidence}

QUESTION:
{question}

OPTIONS:
{options}

Reason from the EVIDENCE across ALL {n} options A-E and pick the single best-supported answer. Weigh
every option on its own merits — none has been eliminated.

A reviewer raised the following notes. They are ADVISORY ONLY and may be wrong — rely on your OWN
reading of the evidence and do NOT defer to them blindly:
{critique}

Commit to EXACTLY ONE option in {letters}. You must choose; do not leave it blank.

Return pure JSON:
{{"answer": "C", "explanation": "..."}}"""


# ── [B] Agent 1 — Reasoning ──────────────────────────────────────────────────
def run_reasoning(state, llm, counter=None, tracker=None, lessons=""):
    prompt = lessons_block(lessons) + REASONING_PROMPT.format(   # §9: nạp lessons cho Agent 1
        evidence=format_evidence(state["evidence"]),
        question=state["question"],
        options=format_options(state["options"]),
    )
    with trace_span(tracker, "reasoning",
                    inputs={"messages": [{"role": "user", "content": prompt}]}) as sp:
        raw, tel = _call(llm, prompt, counter)
        parsed = parse_json_loose(raw, required_keys=["candidates"])
        out = parsed or {"candidates": [], "reasoning": str(raw)[:800], "evidence_used": []}
        status = "ok" if parsed else "parse_fail"
        state["trace"].append({"step": 1, "agent": "reasoning", "raw": raw, "output": out,
                               "status": status, **tel})
        sp.set_outputs({"response": raw, "parsed": out})   # giữ CẢ raw lẫn parsed, không bỏ sót
        sp.set_attributes({"agent": "reasoning", "status": status,
                           "latency_s": tel["latency"], "tokens_src": tel.get("tokens_src")})
        sp.set_model(getattr(llm, "model", None), _provider_of(llm))   # mlflow.llm.model/provider
        sp.set_token_usage(tel["tokens_in"], tel["tokens_out"])        # mlflow.chat.tokenUsage
    for c in out.get("candidates", []):
        c = str(c).strip().upper()
        if c in state["candidates"]:
            state["candidates"][c] = {"status": "proposed",
                                      "reason": out.get("reasoning", ""),
                                      "evidence_ids": out.get("evidence_used", [])}
    return out


# ── [C] Agent 2 — Verifier ───────────────────────────────────────────────────
def run_verifier(state, llm, counter=None, tracker=None, lessons=""):
    prompt = lessons_block(lessons) + VERIFIER_PROMPT.format(   # §9: nạp lessons cho Agent 2
        evidence=format_evidence(state["evidence"]),
        question=state["question"],
        options=format_options(state["options"]),
        analysis=_dumps(state["trace"][0]["output"]),   # toàn bộ trace Agent 1
        n=len(state["options"]),
    )
    with trace_span(tracker, "verifier",
                    inputs={"messages": [{"role": "user", "content": prompt}]}) as sp:
        raw, tel = _call(llm, prompt, counter)
        parsed = parse_json_loose(raw, required_keys=["concerns", "critique"])
        out = parsed or {"concerns": {}, "critique": str(raw)[:800]}
        status = "ok" if parsed else "parse_fail"
        state["trace"].append({"step": 2, "agent": "verifier", "raw": raw, "output": out,
                               "status": status, **tel})
        sp.set_outputs({"response": raw, "parsed": out})
        sp.set_attributes({"agent": "verifier", "status": status,
                           "latency_s": tel["latency"], "tokens_src": tel.get("tokens_src")})
        sp.set_model(getattr(llm, "model", None), _provider_of(llm))
        sp.set_token_usage(tel["tokens_in"], tel["tokens_out"])
    # Fix 1: KHÔNG eliminate. Chỉ ghi quan ngại vào candidate — mọi option vẫn còn cho Aggregator.
    for k, concern in (out.get("concerns") or {}).items():
        k = str(k).strip().upper()
        if k in state["candidates"] and concern and str(concern).strip().lower() != "none":
            state["candidates"][k] = {**state["candidates"][k], "status": "reviewed",
                                      "concern": str(concern)}
    return out


# ── [D] Agent 3 — Aggregator (V3: THẤY evidence gốc) ─────────────────────────
def run_aggregator(state, llm, counter=None, tracker=None):
    # Fix 2: prompt SẠCH như RAG-only (evidence + options). Critique = phụ chú advisory,
    # KHÔNG nhồi analysis/candidate-table (thứ làm nhiễu suy luận sạch của model).
    prompt = AGGREGATOR_PROMPT.format(
        evidence=format_evidence(state["evidence"]),
        question=state["question"],
        options=format_options(state["options"]),
        critique=_dumps(state["trace"][1]["output"].get("critique", state["trace"][1]["output"])),
        letters=", ".join(state["options"].keys()),
        n=len(state["options"]),
    )
    with trace_span(tracker, "aggregator",
                    inputs={"messages": [{"role": "user", "content": prompt}]}) as sp:
        raw, tel = _call(llm, prompt, counter)
        out = parse_json_loose(raw, required_keys=["answer"])
        status = "ok" if out else "parse_fail"
        state["trace"].append({"step": 3, "agent": "aggregator", "raw": raw,
                               "output": out if out is not None else raw,
                               "status": status, **tel})
        sp.set_outputs({"response": raw, "parsed": out})
        sp.set_attributes({"agent": "aggregator", "status": status,
                           "latency_s": tel["latency"], "tokens_src": tel.get("tokens_src")})
        sp.set_model(getattr(llm, "model", None), _provider_of(llm))
        sp.set_token_usage(tel["tokens_in"], tel["tokens_out"])
    return raw


# ── Vòng chạy 1 câu (§8) ─────────────────────────────────────────────────────
def answer_multiagent(question, options, cfg=None, clients=None, evidence=None, tracker=None,
                      memory=None):
    """
    Trả dict: {choice, answer_text, trace, evidence, candidates, latency, n_tokens_est}.
    - evidence=None -> tự chạy [A] retrieve_rerank (cần embed/qdrant/es).
    - evidence truyền sẵn -> bỏ qua [A] (dùng cho test offline); vẫn cần llm trong clients.
    - clients: (embed_client, qdrant, es, llm) tái dùng khi chạy hàng loạt.
    - tracker: Tracker (tracking.py) -> mỗi câu thành 1 MLflow trace GenAI (no-op nếu None).
    """
    cfg = cfg or load_config()
    # Giải quyết llm/clients trước; chỉ tạo ec/qc/es khi thật sự cần retrieve.
    ec = qc = es = None
    if clients:
        ec, qc, es, llm = clients
    else:
        llm = make_llm(cfg)

    counter = make_token_counter(cfg) if cfg.get("embedding") else None
    t0 = time.time()
    trace_id = None
    with trace_span(tracker, "multiagent", span_type="CHAIN", root=True,
                    inputs={"question": question, "options": options},
                    attributes={"n_agents": 3}) as root:
        if evidence is None:
            if not clients:
                ec, qc, es = EmbedClient(cfg), get_qdrant(cfg), get_es(cfg)
            with trace_span(tracker, "retrieve", span_type="RETRIEVER",
                            inputs={"query": question}) as rsp:
                evidence = retrieve_rerank(cfg, question, embed_client=ec, qdrant=qc, es=es)
                rsp.set_outputs({"chunk_ids": [e["chunk_id"] for e in evidence],
                                 "n": len(evidence)})
        state = init_state(question, options, evidence)
        lessons = memory.get_lessons(question) if memory else ""   # §9: truy hồi lỗi quá khứ
        run_reasoning(state, llm, counter, tracker, lessons)          # [B] (có lessons)
        run_verifier(state, llm, counter, tracker, lessons)          # [C] (có lessons)
        agg_raw = run_aggregator(state, llm, counter, tracker)       # [D] (KHÔNG lessons — §9)
        choice = extract_answer(agg_raw, valid=options.keys())   # [E]
        # output root = kết quả + toàn bộ short-term state (bảng ứng viên) -> không bỏ sót
        root.set_outputs({"choice": choice, "candidates": state["candidates"]})

        # nguồn token của trace này (provider / tokenizer / mixed)
        srcs = {t.get("tokens_src") for t in state["trace"]}
        tokens_src = srcs.pop() if len(srcs) == 1 else "mixed"
        # biến & kết quả -> TAG của trace (lọc/nhóm/tính accuracy ở tab Traces, thay cho metric run)
        set_trace_tags(tracker, {"variant": "v3", "model": getattr(llm, "model", None),
                                 "provider": (cfg.get("llm") or {}).get("provider"),
                                 "pred": choice, "tokens_src": tokens_src,
                                 "memory": "on" if (memory and getattr(memory, "enabled", False)) else "off",
                                 "lessons_used": bool(lessons)})   # §9: có dùng long-term memory?
        trace_id = root.trace_id
    dt = time.time() - t0

    tr = state["trace"]
    tokens_real = all(t.get("tokens_in") is not None for t in tr)
    if tokens_real:
        n_tokens = sum((t["tokens_in"] or 0) + (t["tokens_out"] or 0) for t in tr)
    else:
        n_tokens = sum(len(str(t.get("raw", ""))) for t in tr) // 4   # ước lượng khi không có counter
    return {"choice": choice, "answer_text": agg_raw, "trace": tr,
            "evidence": evidence, "candidates": state["candidates"],
            "latency": dt, "n_tokens": n_tokens, "tokens_real": tokens_real,
            "tokens_src": tokens_src, "trace_id": trace_id,
            "lessons_used": bool(lessons)}   # §9: câu này có nạp lesson không


if __name__ == "__main__":
    # Smoke 1 câu thật (cần Qdrant/ES + service embedding + LLM sống).
    import sys
    from common import setup_logging
    log = setup_logging("agents")
    q = " ".join(sys.argv[1:]) or "A 25-year-old man presents with fever and neck stiffness. Which is the most likely diagnosis?"
    opts = {"A": "Migraine", "B": "Bacterial meningitis", "C": "Tension headache",
            "D": "Cluster headache", "E": "Sinusitis"}
    res = answer_multiagent(q, opts)
    print("\n=== CANDIDATES ===")
    print(_dumps(res["candidates"]))
    print("\n=== CHOICE ===", res["choice"], f"(latency {res['latency']:.1f}s)")
