"""
rag.py — ONLINE RAG (module dùng chung)
Luồng (KHÔNG HyDE):
  question -> MedCPT query-embed + BM25 -> RRF -> MedCPT rerank -> top-5 chunk
           -> ghép context -> LLM (OpenRouter) -> câu trả lời.

answer(question, options=None):
  - options=None            -> hỏi-đáp mở: trả văn xuôi + trích dẫn chunk_id.
  - options={"A":..,"B":..} -> trắc nghiệm: LLM chọn 1 đáp án, parse ra letter.
"""
import re

from common import load_config, EmbedClient, make_llm, get_qdrant, get_es
from retrieval import retrieve_rerank


def format_context(chunks):
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] ({c['source']} · {c['chunk_id']})\n{c['text']}")
    return "\n\n".join(lines)


# ── Prompt trắc nghiệm ───────────────────────────────────────────────────────
MC_SYS = ("You are a medical expert taking the USMLE. Use the reference context when helpful, "
          "but rely on your medical knowledge. Answer the multiple-choice question.")

def build_mc_messages(question, options, context):
    opt_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (
        f"Reference context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opt_lines}\n\n"
        f"Think briefly, then end your reply with a line exactly in the form:\n"
        f"Answer: <letter>\n"
        f"where <letter> is one of {', '.join(options.keys())}."
    )
    return [{"role": "system", "content": MC_SYS}, {"role": "user", "content": user}]


def parse_choice(text, valid_letters):
    """Rút letter đáp án. Ưu tiên dòng 'Answer: X', fallback tìm letter cuối cùng hợp lệ."""
    if not text:
        return None
    m = re.findall(r"Answer\s*:?\s*\(?([A-Ea-e])\)?", text)
    if m:
        c = m[-1].upper()
        if c in valid_letters:
            return c
    # fallback: letter đứng riêng (vd "The answer is C" / "(D)")
    m = re.findall(r"\b([A-E])\b", text)
    for c in reversed(m):
        if c in valid_letters:
            return c
    return None


# ── Prompt hỏi-đáp mở ────────────────────────────────────────────────────────
OPEN_SYS = ("You are a medical expert. Answer the question using the reference context. "
            "Cite supporting passages inline as [n]. If the context is insufficient, say so "
            "and answer from general medical knowledge.")

def build_open_messages(question, context):
    user = f"Reference context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with citations [n]."
    return [{"role": "system", "content": OPEN_SYS}, {"role": "user", "content": user}]


# ── API chính ────────────────────────────────────────────────────────────────
def answer(question, options=None, cfg=None, clients=None):
    """
    Trả dict: {answer_text, chunks, [choice nếu MC]}.
    clients: (embed_client, qdrant, es, llm) tái sử dụng khi chạy hàng loạt.
    """
    cfg = cfg or load_config()
    if clients:
        ec, qc, es, llm = clients
    else:
        ec, qc, es, llm = EmbedClient(cfg), get_qdrant(cfg), get_es(cfg), make_llm(cfg)

    chunks = retrieve_rerank(cfg, question, embed_client=ec, qdrant=qc, es=es)
    context = format_context(chunks)

    if options:
        messages = build_mc_messages(question, options, context)
        text = llm.chat(messages)
        choice = parse_choice(text, set(options.keys()))
        return {"answer_text": text, "choice": choice, "chunks": chunks}
    else:
        messages = build_open_messages(question, context)
        text = llm.chat(messages)
        return {"answer_text": text, "chunks": chunks}
