"""
ask.py — CLI hỏi-đáp mở (online RAG, không HyDE).
  python ask.py "What is the first-line treatment for myelofibrosis?"
"""
import sys
import argparse

from common import load_config, setup_logging, EmbedClient, make_llm, get_qdrant, get_es
from rag import answer

log = setup_logging("ask")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="Câu hỏi y khoa")
    args = ap.parse_args()
    question = " ".join(args.question)

    cfg = load_config()
    clients = (EmbedClient(cfg, log), get_qdrant(cfg), get_es(cfg), make_llm(cfg, log))
    log.info("Query: %r", question)
    res = answer(question, options=None, cfg=cfg, clients=clients)

    print("\n=== ANSWER ===")
    print(res["answer_text"].strip())
    print("\n=== SOURCES (top-%d sau rerank) ===" % len(res["chunks"]))
    for i, c in enumerate(res["chunks"], 1):
        sc = f"{c['rerank_score']:.2f}" if c["rerank_score"] is not None else "-"
        print(f"[{i}] {c['chunk_id']} ({c['source']}) rerank={sc}")
        print(f"    {c['text'][:160].strip()}...")


if __name__ == "__main__":
    main()
