"""
test_agents.py — Test cho luồng multi-agent V3 (agents.py). QD4: viết test + chạy pass.
  python test_agents.py            # phần tất định (offline, không cần service) — BẮT BUỘC pass
  python test_agents.py --smoke    # thêm 1 câu thật (cần Qdrant/ES/embed/LLM sống)

Không dùng framework (khớp verify_index.py/smoke_test.py): assert + exit 0 = PASS.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import module ở root

import agents
from agents import (parse_json_loose, extract_answer, init_state,
                    run_reasoning, run_verifier, run_aggregator, answer_multiagent)


class FakeLLM:
    """Trả lần lượt các response đã kịch bản hoá (mỗi .chat = 1 agent)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


# ── Fake MLflow để test GenAI Tracing offline (không cần server) ─────────────
class FakeSpan:
    def __init__(self, name):
        self.name, self.inputs, self.outputs, self.attributes = name, None, None, {}
        self.trace_id = "tr-fake"

    def set_inputs(self, x):
        self.inputs = x

    def set_outputs(self, x):
        self.outputs = x

    def set_attributes(self, d):
        self.attributes.update(d)

    def set_attribute(self, k, v):     # dùng cho token usage native (mlflow.chat.tokenUsage)
        self.attributes[k] = v


class _FakeSpanCM:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, *exc):
        return False


class FakeMLflow:
    """Ghi lại mọi start_span + tag trace; raise_on=tên span sẽ ném lỗi lúc tạo (test graceful)."""
    def __init__(self, raise_on=None):
        self.spans, self.raise_on, self.trace_tags = [], raise_on, {}

    def start_span(self, name="span", span_type=None, run_id=None):
        if self.raise_on and name == self.raise_on:
            raise RuntimeError("boom")
        sp = FakeSpan(name)
        self.spans.append(sp)
        return _FakeSpanCM(sp)

    def update_current_trace(self, tags=None):
        if tags:
            self.trace_tags.update(tags)


class FakeTracker:
    def __init__(self, mlflow, run_id="run-123"):
        self.mlflow, self.run_id = mlflow, run_id


class FakeLLMUsage(FakeLLM):
    """Như FakeLLM nhưng có chat_usage -> trả kèm usage giống provider (OpenRouter/Ollama)."""
    def __init__(self, responses, usages):
        super().__init__(responses)
        self.usages = list(usages)

    def chat_usage(self, messages):
        return self.chat(messages), self.usages.pop(0)


class FakeEmbed:
    """embed_article: vector 3 chiều theo keyword -> điều khiển cosine cho test memory."""
    def embed_article(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append([1.0 if "myosin" in t else 0.0,
                        1.0 if "kidney" in t else 0.0,
                        0.1])   # bias nhỏ tránh zero-vector
        return out


CFG_OFFLINE = {"offline": True}   # truthy nhưng KHÔNG có 'embedding' -> answer_multiagent bỏ token counter
EVID = [{"chunk_id": "c1", "source": "Book_A", "text": "Ceftriaxone là kháng sinh phổ rộng."},
        {"chunk_id": "c2", "source": "Book_B", "text": "Viêm màng não do vi khuẩn cần điều trị sớm."}]
OPTS = {"A": "Chloramphenicol", "B": "Gentamicin", "C": "Ciprofloxacin",
        "D": "Ceftriaxone", "E": "Trimethoprim"}


def check(name, cond):
    if not cond:
        print(f"  ✗ FAIL: {name}")
        raise AssertionError(name)
    print(f"  ✓ {name}")


def test_parse_json_loose():
    print("[parse_json_loose]")
    check("json thuần", parse_json_loose('{"answer": "C"}')["answer"] == "C")
    check("code fence", parse_json_loose('```json\n{"answer": "D"}\n```')["answer"] == "D")
    check("kèm reasoning trước JSON",
          parse_json_loose('Tôi nghĩ...\n{"candidates": ["A"]}')["candidates"] == ["A"])
    check("lấy object CUỐI khi có 2",
          parse_json_loose('{"answer": "A"} rồi sửa {"answer": "B"}')["answer"] == "B")
    check("required_keys lọc đúng object",
          parse_json_loose('{"foo": 1}\n{"answer": "E"}', required_keys=["answer"])["answer"] == "E")
    check("rác -> None", parse_json_loose("không có json ở đây") is None)
    check("dict truyền thẳng", parse_json_loose({"answer": "C"})["answer"] == "C")


def test_extract_answer():
    print("[extract_answer]")
    check("json answer", extract_answer('{"answer": "C"}') == "C")
    check("fence json", extract_answer('```json\n{"answer":"D"}\n```') == "D")
    check("fallback letter", extract_answer("Sau khi cân nhắc, chọn B.") == "B")
    check("rác -> INVALID", extract_answer("hoàn toàn không có đáp án") == "INVALID")
    check("ngoài A-E -> INVALID", extract_answer('{"answer": "Z"}') == "INVALID")
    check("valid hạn chế ABCD: E -> INVALID",
          extract_answer('{"answer": "E"}', valid="ABCD") == "INVALID")
    check("dict input", extract_answer({"answer": "A"}) == "A")


def test_reasoning_prompt_no_early_narrow():
    """Prompt Agent 1 không được ép '1–2 phương án' (gây rớt gold); phải yêu cầu liệt kê rộng."""
    print("[reasoning prompt — không thu hẹp sớm]")
    from agents import REASONING_PROMPT
    low = REASONING_PROMPT.lower()
    check("không ép chọn 1-2 phương án", "1-2 options" not in low)
    check("có chỉ thị KHÔNG thu hẹp sớm (list ALL / do not narrow)",
          "do not narrow" in low and "all still-plausible" in low)
    check("vẫn giữ format candidates JSON", '"candidates"' in REASONING_PROMPT)


def test_verifier_prompt_no_eliminate():
    """Fix 1: Verifier chỉ nêu QUAN NGẠI, KHÔNG được eliminate/reject/rank option."""
    print("[verifier prompt — không eliminate]")
    from agents import VERIFIER_PROMPT
    low = VERIFIER_PROMPT.lower()
    check("yêu cầu nêu concern", "concern" in low)
    check("cấm eliminate/reject", "do not eliminate" in low or "not eliminate" in low)
    check("format concerns JSON", '"concerns"' in VERIFIER_PROMPT)


def test_aggregator_prompt_clean():
    """Fix 2: Aggregator prompt SẠCH (evidence+options), critique là ADVISORY, không nhồi candidate-table."""
    print("[aggregator prompt — sạch + advisory]")
    from agents import AGGREGATOR_PROMPT
    low = AGGREGATOR_PROMPT.lower()
    check("critique chỉ advisory", "advisory" in low)
    check("KHÔNG nhồi candidate table", "candidate table" not in low)
    check("KHÔNG nhồi analysis (agent 1)", "analysis (agent 1)" not in low)
    check("prompt còn placeholder {n}", "{n}" in AGGREGATOR_PROMPT)


def test_init_state():
    print("[init_state]")
    s = init_state("Q?", OPTS, EVID)
    check("candidates đủ 5 khoá", set(s["candidates"]) == set("ABCDE"))
    check("trace rỗng ban đầu", s["trace"] == [])
    check("mọi candidate 'unreviewed'",
          all(v["status"] == "unreviewed" for v in s["candidates"].values()))
    check("evidence giữ nguyên", s["evidence"] is EVID)


def test_agent_writes_and_trace_invariant():
    """Mỗi agent ghi state phải kèm đúng 1 entry trace; candidates đổi đúng."""
    print("[agent writes + bất biến trace]")
    s = init_state("Q?", OPTS, EVID)

    llm1 = FakeLLM(['{"candidates": ["D","B"], "reasoning": "Ceftriaxone hợp lý", "evidence_used": [1]}'])
    run_reasoning(s, llm1)
    check("reasoning append trace step1", len(s["trace"]) == 1 and s["trace"][0]["step"] == 1)
    check("candidate D -> proposed", s["candidates"]["D"]["status"] == "proposed")
    check("candidate D giữ evidence_ids", s["candidates"]["D"]["evidence_ids"] == [1])
    check("candidate A vẫn unreviewed", s["candidates"]["A"]["status"] == "unreviewed")

    llm2 = FakeLLM(['{"concerns": {"D": "none", "B": "narrow spectrum"}, "critique": "..."}'])
    run_verifier(s, llm2)
    check("verifier append trace step2", len(s["trace"]) == 2 and s["trace"][1]["step"] == 2)
    # Fix 1: verifier KHÔNG eliminate — mọi option vẫn còn (không có status 'remove')
    check("không option nào bị remove",
          all(v["status"] != "remove" for v in s["candidates"].values()))
    check("B có concern ghi lại", s["candidates"]["B"].get("concern") == "narrow spectrum")
    check("D không concern (none) -> vẫn 'proposed'", s["candidates"]["D"]["status"] == "proposed")

    llm3 = FakeLLM(['{"answer": "D", "explanation": "Ceftriaxone"}'])
    raw = run_aggregator(s, llm3)
    check("aggregator append trace step3", len(s["trace"]) == 3 and s["trace"][2]["step"] == 3)
    check("aggregator thấy evidence gốc trong prompt",
          "Ceftriaxone là kháng sinh" in llm3.calls[0][0]["content"])
    check("extract từ aggregator = D", extract_answer(raw, valid=OPTS.keys()) == "D")


def test_answer_multiagent_offline():
    """Luồng đủ 3 agent, evidence truyền sẵn (không gọi [A]/service)."""
    print("[answer_multiagent offline]")
    fake = FakeLLM([
        '{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
        '{"concerns": {"D": "none"}, "critique": "c"}',
        '{"answer": "D", "explanation": "e"}',
    ])
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE, clients=(None, None, None, fake), evidence=EVID)
    check("choice = D", res["choice"] == "D")
    check("trace đủ 3 bước", len(res["trace"]) == 3)
    check("giữ evidence_chunk_ids", [e["chunk_id"] for e in res["evidence"]] == ["c1", "c2"])
    check("gọi LLM đúng 3 lần", len(fake.responses) == 0)
    # telemetry: mỗi bước có status + latency; không có counter -> tokens None
    check("mỗi bước có status", all("status" in t for t in res["trace"]))
    check("status ok khi JSON parse được",
          [t["status"] for t in res["trace"]] == ["ok", "ok", "ok"])
    check("mỗi bước có latency", all(isinstance(t["latency"], float) for t in res["trace"]))
    check("không counter -> tokens_real False", res["tokens_real"] is False)
    check("n_tokens ước lượng >= 0", res["n_tokens"] >= 0)

    # Aggregator trả rác -> INVALID (không crash)
    fake2 = FakeLLM([
        '{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
        '{"concerns": {}, "critique": "c"}',
        'tôi không chắc lắm',
    ])
    res2 = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE, clients=(None, None, None, fake2), evidence=EVID)
    check("aggregator rác -> INVALID", res2["choice"] == "INVALID")

    # LLM trả rác ở Agent 1/2 -> pipeline vẫn chạy, không ném lỗi
    fake3 = FakeLLM(["reasoning hỏng", "verifier hỏng", '{"answer": "A"}'])
    res3 = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE, clients=(None, None, None, fake3), evidence=EVID)
    check("agent 1/2 hỏng vẫn ra choice hợp lệ", res3["choice"] == "A")
    check("JSON hỏng -> status parse_fail",
          [t["status"] for t in res3["trace"][:2]] == ["parse_fail", "parse_fail"])


def test_token_counter():
    """Không có usage provider -> fallback tokenizer (counter); tokens_src='tokenizer'."""
    print("[token counter — fallback tokenizer]")
    s = init_state("Q?", OPTS, EVID)
    counter = lambda text: len(str(text).split())   # đếm từ, đủ để test đường ghi telemetry
    llm = FakeLLM(['{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}'])
    run_reasoning(s, llm, counter)
    t = s["trace"][0]
    check("tokens_in > 0", t["tokens_in"] > 0)
    check("tokens_out > 0", t["tokens_out"] > 0)
    check("tokens_src=tokenizer", t["tokens_src"] == "tokenizer")


def test_token_from_provider():
    """Đếm token THẬT từ provider usage (chat_usage) — đúng cả model free / ollama."""
    print("[token từ provider usage]")
    fake = FakeLLMUsage(
        ['{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
         '{"concerns": {"D": "none"}, "critique": "c"}',
         '{"answer": "D", "explanation": "e"}'],
        [{"prompt_tokens": 100, "completion_tokens": 20},
         {"prompt_tokens": 200, "completion_tokens": 30},
         {"prompt_tokens": 300, "completion_tokens": 40}])
    # CFG_OFFLINE: KHÔNG có counter tokenizer -> token chỉ có thể đến từ provider usage.
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE,
                            clients=(None, None, None, fake), evidence=EVID)
    tr = res["trace"]
    check("tokens_in lấy đúng từ provider (reasoning=100)", tr[0]["tokens_in"] == 100)
    check("tokens_out lấy đúng từ provider (reasoning=20)", tr[0]["tokens_out"] == 20)
    check("mọi bước tokens_src=provider", all(t["tokens_src"] == "provider" for t in tr))
    check("tokens_real=True dù không có tokenizer", res["tokens_real"] is True)
    check("tokens_src tổng = provider", res["tokens_src"] == "provider")
    check("n_tokens = tổng usong provider (690)", res["n_tokens"] == 100+20+200+30+300+40)


def _scripted_ok():
    return FakeLLM([
        '{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
        '{"concerns": {"D": "none"}, "critique": "c"}',
        '{"answer": "D", "explanation": "e"}',
    ])


def test_tracing_spans():
    """GenAI Tracing: mỗi câu tạo root span + span con từng agent, có input/output/attr."""
    print("[GenAI tracing spans]")
    mlf = FakeMLflow()
    tracker = FakeTracker(mlf)
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE,
                            clients=(None, None, None, _scripted_ok()),
                            evidence=EVID, tracker=tracker)
    names = [s.name for s in mlf.spans]
    check("kết quả KHÔNG đổi khi bật tracker", res["choice"] == "D")
    # evidence truyền sẵn -> KHÔNG có span 'retrieve'
    check("root multiagent + 3 agent span, đúng thứ tự",
          names == ["multiagent", "reasoning", "verifier", "aggregator"])
    root, reasoning, agg = mlf.spans[0], mlf.spans[1], mlf.spans[3]
    check("root input có question", root.inputs["question"] == "Q?")
    check("root output choice=D", root.outputs["choice"] == "D")
    check("root attr n_agents=3", root.attributes["n_agents"] == 3)
    check("reasoning input là messages (role=user)",
          reasoning.inputs["messages"][0]["role"] == "user")
    check("reasoning output giữ CẢ response lẫn parsed",
          reasoning.outputs["parsed"]["candidates"] == ["D"] and "response" in reasoning.outputs)
    check("reasoning attr agent+status", reasoning.attributes["agent"] == "reasoning"
          and reasoning.attributes["status"] == "ok")
    check("root output kèm candidates (shared state)", "candidates" in root.outputs)
    check("aggregator attr agent", agg.attributes["agent"] == "aggregator")
    # model/provider/pred -> TAG của trace (không phải metric của run)
    check("trace tag variant=v3", mlf.trace_tags.get("variant") == "v3")
    check("trace tag pred=D", mlf.trace_tags.get("pred") == "D")


def test_native_token_usage_on_span():
    """Token đi theo chuẩn native mlflow.chat.tokenUsage (MLflow tự tổng lên trace)."""
    print("[token usage native trên span]")
    from mlflow.tracing.constant import SpanAttributeKey
    mlf = FakeMLflow()
    tracker = FakeTracker(mlf)
    fake = FakeLLMUsage(
        ['{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
         '{"concerns": {"D": "none"}, "critique": "c"}',
         '{"answer": "D", "explanation": "e"}'],
        [{"prompt_tokens": 100, "completion_tokens": 20},
         {"prompt_tokens": 200, "completion_tokens": 30},
         {"prompt_tokens": 300, "completion_tokens": 40}])
    fake.model = "gemma-test"
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE,
                            clients=(None, None, None, fake), evidence=EVID, tracker=tracker)
    reasoning = [s for s in mlf.spans if s.name == "reasoning"][0]
    usage = reasoning.attributes.get(SpanAttributeKey.CHAT_USAGE)
    check("span reasoning có mlflow.chat.tokenUsage", usage is not None)
    check("input_tokens=100", usage["input_tokens"] == 100)
    check("output_tokens=20", usage["output_tokens"] == 20)
    check("total_tokens=120 (tự cộng)", usage["total_tokens"] == 120)
    check("SỐ token phẳng trên span: tokens_in=100", reasoning.attributes.get("tokens_in") == 100)
    check("SỐ token phẳng trên span: tokens_out=20", reasoning.attributes.get("tokens_out") == 20)
    check("SỐ token phẳng trên span: tokens_total=120", reasoning.attributes.get("tokens_total") == 120)
    check("span có mlflow.llm.model", reasoning.attributes.get(SpanAttributeKey.MODEL) == "gemma-test")
    check("answer_multiagent trả trace_id", res.get("trace_id") == "tr-fake")


def test_tracing_retrieve_span_uses_root():
    """Nội bộ: khi tự retrieve, có span 'retrieve' dưới root (đường evidence=None)."""
    print("[GenAI tracing: span retrieve]")
    from tracking import span as _span
    mlf = FakeMLflow()
    tracker = FakeTracker(mlf)
    with _span(tracker, "multiagent", span_type="CHAIN") as root:
        with _span(tracker, "retrieve", span_type="RETRIEVER") as r:
            r.set_outputs({"n": 2})
    check("tạo đúng 2 span multiagent+retrieve",
          [s.name for s in mlf.spans] == ["multiagent", "retrieve"])
    check("retrieve output ghi được", mlf.spans[1].outputs["n"] == 2)


def test_tracing_graceful():
    """Graceful: tracker None / start_span lỗi -> pipeline vẫn chạy, không ném."""
    print("[GenAI tracing graceful/no-op]")
    # tracker None đã cover ở offline; ở đây: start_span ném lỗi giữa chừng.
    mlf = FakeMLflow(raise_on="reasoning")
    tracker = FakeTracker(mlf)
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE,
                            clients=(None, None, None, _scripted_ok()),
                            evidence=EVID, tracker=tracker)
    check("start_span lỗi vẫn ra choice=D", res["choice"] == "D")
    check("trace nội bộ vẫn đủ 3 bước", len(res["trace"]) == 3)


def test_span_helper_contract():
    """span(): không nuốt exception thân lệnh; tracker None -> mọi set_* no-op."""
    print("[span() contract]")
    from tracking import span as _span
    raised = False
    try:
        with _span(FakeTracker(FakeMLflow()), "x") as sp:
            sp.set_inputs({"a": 1})
            raise ValueError("body")
    except ValueError:
        raised = True
    check("exception thân lệnh được propagate", raised)
    with _span(None, "y") as sp:      # tracker None -> _Span(None), mọi set_* no-op
        sp.set_inputs({"a": 1}); sp.set_outputs({"b": 2}); sp.set_attributes({"c": 3})
    check("tracker None -> no-op không lỗi", True)


def test_aggregate_agent_stats():
    """Per-agent metric gộp TỪ traces (thay metric của run) — accuracy + calls/parse_fail/latency/token."""
    print("[trace_stats.aggregate_agent_stats]")
    from trace_stats import aggregate_agent_stats
    traces = [
        {"tags": {"variant": "v3", "correct": "True", "invalid": "False"},
         "spans": [{"agent": "reasoning", "status": "ok", "latency_s": 1.0, "tokens_in": 100, "tokens_out": 20},
                   {"agent": "verifier", "status": "parse_fail", "latency_s": 2.0, "tokens_in": 200, "tokens_out": 30}]},
        {"tags": {"variant": "v3", "correct": "False", "invalid": "True"},
         "spans": [{"agent": "reasoning", "status": "ok", "latency_s": 3.0, "tokens_in": 300, "tokens_out": 40}]},
    ]
    ov, per = aggregate_agent_stats(traces)
    check("accuracy 1/2", ov["correct"] == 1 and ov["accuracy"] == 0.5)
    check("invalid=1", ov["invalid"] == 1)
    check("reasoning calls=2", per["reasoning"]["calls"] == 2)
    check("verifier parse_fail=1", per["verifier"]["parse_fail"] == 1)
    check("reasoning tokens_in tổng=400", per["reasoning"]["tokens_in"] == 400)
    check("reasoning latency tổng=4.0", per["reasoning"]["latency"] == 4.0)
    check("agent không tên bị bỏ qua", "None" not in per and None not in per)


def test_dedup_latest():
    """Chạy lại eval -> trace trùng (variant,split,q_index): giữ bản request_time mới nhất."""
    print("[trace_stats.dedup_latest]")
    from trace_stats import dedup_latest
    traces = [
        {"tags": {"variant": "v3", "split": "dev.jsonl", "q_index": "0"}, "spans": [], "request_time": 100},
        {"tags": {"variant": "v3", "split": "dev.jsonl", "q_index": "0"}, "spans": [], "request_time": 200},  # mới hơn
        {"tags": {"variant": "v3", "split": "dev.jsonl", "q_index": "1"}, "spans": [], "request_time": 150},
        {"tags": {"variant": "v3", "split": "dev.jsonl"}, "spans": [], "request_time": 50},  # thiếu q_index -> giữ
    ]
    dedup, dropped = dedup_latest(traces)
    check("bỏ đúng 1 trace trùng", dropped == 1)
    check("còn 3 trace", len(dedup) == 3)
    q0 = [t for t in dedup if t["tags"].get("q_index") == "0"]
    check("giữ bản q_index=0 MỚI NHẤT (rt=200)", len(q0) == 1 and q0[0]["request_time"] == 200)
    check("trace thiếu q_index vẫn giữ", any("q_index" not in t["tags"] for t in dedup))


def test_adaptive_gate():
    """AdaptiveGate: nghẽn cao -> hạ về min; nghẽn thấp -> tăng dần tới max."""
    print("[adaptive gate — điều tiết concurrency]")
    from throttle import AdaptiveGate
    changes = []
    g = AdaptiveGate(max_conc=3, min_conc=1, window=6, high=0.5, low=0.2,
                     latency_threshold=60, on_change=lambda o, n, r: changes.append((o, n)))
    check("khởi đầu ở max=3", g.limit == 3)
    for _ in range(6):
        g.acquire(); g.release(congested=True)
    check("nghẽn cao -> limit=1", g.limit == 1)
    for _ in range(6):
        g.acquire(); g.release(congested=False)
    check("nghẽn thấp -> limit tăng lại (>1)", g.limit > 1)
    check("on_change gọi khi đổi mức (>=2 lần)", len(changes) >= 2)


def test_throttled_llm():
    """ThrottledLLM trong suốt: trả (text,usage) như client gốc; fail -> đánh dấu nghẽn."""
    print("[throttled llm — trong suốt + phát tín hiệu nghẽn]")
    from throttle import AdaptiveGate, ThrottledLLM
    g = AdaptiveGate(max_conc=3, min_conc=1, window=4, high=0.5, low=0.2, latency_threshold=60)
    fake = FakeLLMUsage(['{"answer":"D"}'], [{"prompt_tokens": 5, "completion_tokens": 1}])
    fake.model = "m1"
    tl = ThrottledLLM(fake, g)
    check("expose .model client gốc", tl.model == "m1")
    txt, usage = tl.chat_usage([{"role": "user", "content": "q"}])
    check("chat_usage trả đúng text", "answer" in txt)
    check("chat_usage trả usage", usage["prompt_tokens"] == 5)

    class Boom:
        model = "m2"
        def chat_usage(self, m):
            raise RuntimeError("timeout")
    raised = False
    try:
        ThrottledLLM(Boom(), g).chat_usage([{"role": "user", "content": "q"}])
    except RuntimeError:
        raised = True
    check("lỗi client được propagate", raised)
    check("gate ghi nhận nghẽn", g.status()["congestion"] > 0)


def test_long_memory():
    """long_mem: frozen (tái lập) + mean-centering (BUG-4) + self-exclude (BUG-3) + dedup (BUG-5)."""
    print("[long-term memory — frozen + centering + chống rò + dedup]")
    import tempfile
    from long_mem import LongMemory, lessons_block
    store = os.path.join(tempfile.mkdtemp(), "lessons.jsonl")
    cfg = {"memory": {"enabled": True, "save_wrong": True,
                      "store": store, "top_k": 3, "similarity_threshold": 0.2}}
    QM1, QM2, QK = "First myosin power stroke q", "Second myosin cross-bridge q", "A kidney nephron q"
    mem = LongMemory(cfg, FakeEmbed())
    check("ban đầu không lesson", mem.get_lessons(QM1) == "")
    check("save myosin1", mem.save_wrong(QM1, {"A": "x"}, "A", "B", topic="physio") is True)
    check("save myosin2", mem.save_wrong(QM2, {"C": "y"}, "C", "B", topic="physio") is True)
    check("save kidney", mem.save_wrong(QK, {"B": "z"}, "B", "A", topic="renal") is True)
    # BUG-2: frozen -> cùng instance chưa dùng lesson vừa lưu (tái lập)
    check("frozen: cùng instance chưa retrieve", mem.get_lessons("new myosin q") == "")
    # BUG-5: dedup
    check("dedup: lưu lại myosin1 -> False", mem.save_wrong(QM1, {}, "A", "B") is False)

    # RELOAD -> mean-centering (3 lessons) khử anisotropy, phân biệt được chủ đề
    mem2 = LongMemory(cfg, FakeEmbed())
    les = mem2.get_lessons("brand new myosin cross-bridge case")
    check("retrieve lesson myosin (câu khác)", "PAST MISTAKE" in les and "physio" in les)
    check("centering PHÂN BIỆT: không lẫn kidney", "renal" not in les)
    lesk = mem2.get_lessons("brand new kidney nephron case")
    check("chiều ngược: retrieve kidney, không myosin", "renal" in lesk and "physio" not in lesk)
    # BUG-3: self-exclude — hỏi CHÍNH câu kidney đã lưu -> loại nó, myosin không liên quan -> ""
    check("self-exclude: chính câu kidney -> không rò lesson của nó", mem2.get_lessons(QK) == "")
    check("lessons_block rỗng/nạp đúng", lessons_block("") == "" and "PAST MISTAKES" in lessons_block("x"))
    # tắt memory -> no-op
    mem3 = LongMemory({"memory": {"enabled": False}}, FakeEmbed())
    check("tắt -> get rỗng", mem3.get_lessons("x") == "")
    check("tắt -> save no-op", mem3.save_wrong("q", {}, "A", "B") is False)


def test_long_memory_threadsafe():
    """BUG-1: nhiều thread save_wrong + get_lessons đồng thời -> không crash, file JSONL không hỏng."""
    print("[long-term memory — thread-safe (BUG-1)]")
    import tempfile
    import threading as _th
    from long_mem import LongMemory
    store = os.path.join(tempfile.mkdtemp(), "lessons.jsonl")
    cfg = {"memory": {"enabled": True, "save_wrong": True, "store": store,
                      "top_k": 3, "similarity_threshold": 0.0}}
    mem = LongMemory(cfg, FakeEmbed())
    errors = []

    def worker(w):
        try:
            for j in range(15):
                mem.save_wrong(f"myosin question {w}-{j}", {"A": "x"}, "A", "B")
                mem.get_lessons(f"myosin query {w}-{j}")
        except Exception as e:
            errors.append(str(e))

    ts = [_th.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("không exception khi chạy đồng thời", errors == [])
    bad = 0
    for line in open(store, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                json.loads(line)
            except Exception:
                bad += 1
    check("file JSONL không hỏng (mọi dòng parse được)", bad == 0)
    check("reload sau stress OK (>0 lessons)", len(LongMemory(cfg, FakeEmbed())._lessons) > 0)


def test_multiagent_memory_injection():
    """§9: lessons chèn vào Agent 1 + Agent 2, KHÔNG Agent 3; trả lessons_used."""
    print("[§9 memory chèn Agent 1+2, không Agent 3]")

    class FakeMem:
        def get_lessons(self, q):
            return "LESSON_MARKER_XYZ"

    fake = FakeLLM(['{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
                    '{"concerns": {"D": "none"}, "critique": "c"}',
                    '{"answer": "D", "explanation": "e"}'])
    res = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE, clients=(None, None, None, fake),
                            evidence=EVID, memory=FakeMem())
    r_prompt = fake.calls[0][0]["content"]
    v_prompt = fake.calls[1][0]["content"]
    a_prompt = fake.calls[2][0]["content"]
    check("lesson chèn vào Agent 1 (reasoning)", "LESSON_MARKER_XYZ" in r_prompt)
    check("lesson chèn vào Agent 2 (verifier)", "LESSON_MARKER_XYZ" in v_prompt)
    check("lesson KHÔNG chèn Agent 3 (aggregator)", "LESSON_MARKER_XYZ" not in a_prompt)
    check("trả lessons_used=True", res["lessons_used"] is True)
    # không memory -> không chèn, lessons_used=False
    fake2 = FakeLLM(['{"candidates": ["D"], "reasoning": "r", "evidence_used": [1]}',
                     '{"concerns": {"D": "none"}, "critique": "c"}',
                     '{"answer": "D", "explanation": "e"}'])
    res2 = answer_multiagent("Q?", OPTS, cfg=CFG_OFFLINE, clients=(None, None, None, fake2), evidence=EVID)
    check("không memory -> lessons_used=False", res2["lessons_used"] is False)


def test_metrics():
    """metrics.py: mẫu số cố định, invalid, align+gold-check, W/L/T, McNemar exact, bootstrap, cost/latency."""
    print("[metrics official]")
    import metrics as M
    res = [{"i": 0, "gold": "A", "pred": "A", "ok": True},
           {"i": 1, "gold": "B", "pred": "C", "ok": False},
           {"i": 2, "gold": "C", "pred": "C", "ok": True}]
    acc, corr, tot = M.accuracy(res, total=5)          # G1: mẫu số cố định, 3 câu thiếu = sai
    check("accuracy mẫu số cố định 2/5=40%", corr == 2 and tot == 5 and abs(acc - 0.4) < 1e-9)
    acc2, _, tot2 = M.accuracy(res)
    check("accuracy mặc định 2/3", tot2 == 3 and abs(acc2 - 2/3) < 1e-9)
    inv = [{"i": 0, "pred": None}, {"i": 1, "pred": "INVALID"}, {"i": 2, "pred": "Z"}, {"i": 3, "pred": "A"}]
    _, ninv, _ = M.invalid_rate(inv, total=4)
    check("invalid = None/INVALID/ngoài A-E = 3/4", ninv == 3)
    v0 = [{"i": 0, "gold": "A", "ok": True}, {"i": 1, "gold": "B", "ok": False}]
    v3 = [{"i": 0, "gold": "A", "ok": False}, {"i": 1, "gold": "B", "ok": True}]
    al = M.align(v0, v3)
    check("align 2 câu", len(al) == 2)
    mism = False
    try:
        M.align(v0, [{"i": 0, "gold": "X", "ok": True}])
    except ValueError:
        mism = True
    check("gold lệch -> raise (chống so sai câu)", mism)
    wlt = M.win_loss_tie(al)
    check("W/L/T = 1/1/0", wlt["win"] == 1 and wlt["loss"] == 1 and wlt["tie"] == 0)
    mc = M.mcnemar(al)
    check("McNemar b=1 c=1 -> p=1.0", mc["b"] == 1 and mc["c"] == 1 and mc["p_value"] == 1.0)
    al2 = [{"v0_ok": False, "v3_ok": True} for _ in range(6)]
    mc2 = M.mcnemar(al2)
    check("McNemar 6 win 0 loss -> p<0.05", mc2["c"] == 6 and mc2["b"] == 0 and mc2["p_value"] < 0.05)
    lo, hi = M.bootstrap_ci([1, 1, 1, 1, 1], n_boot=1000)
    check("bootstrap toàn đúng -> CI [1,1]", lo == 1.0 and hi == 1.0)
    lo2, hi2 = M.bootstrap_ci([1, 0] * 10, n_boot=2000)
    check("bootstrap 50% -> CI bao quanh 0.5", lo2 <= 0.5 <= hi2)
    cl = M.cost_latency([{"latency": 10, "n_tokens": 100, "trace": [{"tokens_in": 30, "tokens_out": 5}]},
                         {"latency": 20, "n_tokens": 200, "trace": [{"tokens_in": 40, "tokens_out": 6}]}])
    check("cost tokens_in tổng=70", cl["tokens_in_total"] == 70)
    check("latency mean=15", cl["latency_mean_s"] == 15)


def smoke_real():
    print("[SMOKE 1 câu thật] — cần Qdrant/ES/embed/LLM sống")
    q = "A 25-year-old man presents with fever, headache and neck stiffness. Most likely diagnosis?"
    opts = {"A": "Migraine", "B": "Bacterial meningitis", "C": "Tension headache",
            "D": "Cluster headache", "E": "Sinusitis"}
    res = answer_multiagent(q, opts)
    print(f"  choice={res['choice']} latency={res['latency']:.1f}s trace={len(res['trace'])} bước")
    check("choice hợp lệ (A-E hoặc INVALID)", res["choice"] in set("ABCDE") | {"INVALID"})
    check("trace đủ 3 bước", len(res["trace"]) == 3)


def main():
    test_parse_json_loose()
    test_extract_answer()
    test_reasoning_prompt_no_early_narrow()
    test_verifier_prompt_no_eliminate()
    test_aggregator_prompt_clean()
    test_init_state()
    test_agent_writes_and_trace_invariant()
    test_answer_multiagent_offline()
    test_token_counter()
    test_token_from_provider()
    test_tracing_spans()
    test_native_token_usage_on_span()
    test_tracing_retrieve_span_uses_root()
    test_tracing_graceful()
    test_span_helper_contract()
    test_aggregate_agent_stats()
    test_dedup_latest()
    test_adaptive_gate()
    test_throttled_llm()
    test_long_memory()
    test_long_memory_threadsafe()
    test_multiagent_memory_injection()
    test_metrics()
    if "--smoke" in sys.argv:
        smoke_real()
    print("\n=== TẤT CẢ TEST PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
