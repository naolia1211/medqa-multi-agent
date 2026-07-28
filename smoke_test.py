"""
smoke_test.py — BƯỚC 5b
Chạy 1 query hybrid THẬT để kiểm tra pipeline thông:
  dense (Query-Encoder -> Qdrant) + sparse (ES) + RRF -> top-5.
"""
import argparse
from common import load_config, setup_logging, EmbedClient, get_qdrant, get_es
from retrieval import hybrid_search

log = setup_logging("smoke")

DEFAULT_QUERY = "JAK2 mutation myelofibrosis treatment ruxolitinib"


def text_of(cfg, es, cid):
    try:
        return es.get(index=cfg["elasticsearch"]["index"], id=cid)["_source"]["text"]
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    args = ap.parse_args()

    cfg = load_config()
    ec = EmbedClient(cfg, log)
    qc = get_qdrant(cfg)
    es = get_es(cfg)

    log.info("=== SMOKE TEST ===")
    log.info("Query: %r", args.query)
    r = hybrid_search(cfg, args.query, embed_client=ec, qdrant=qc, es=es)

    log.info("Dense trả %d, Sparse trả %d.", len(r["dense_ids"]), len(r["sparse_ids"]))
    print("\n--- RRF TOP-5 ---")
    for rank, (cid, score) in enumerate(r["fused"], 1):
        # nguồn từ dense payload nếu có, else từ sparse
        payload = next((p for c, p, _ in r["dense"] if c == cid), None)
        source = payload["source"] if payload else next(
            (s for c, s, _ in r["sparse"] if c == cid), "?")
        snippet = (payload["text"] if payload else text_of(cfg, es, cid))[:150].replace("\n", " ")
        d_rank = r["dense_ids"].index(cid) if cid in r["dense_ids"] else "-"
        s_rank = r["sparse_ids"].index(cid) if cid in r["sparse_ids"] else "-"
        print(f"{rank}. {cid}  [{source}]  RRF={score:.4f}  (dense#{d_rank}, sparse#{s_rank})")
        print(f"   {snippet}")

    only_dense = set(r["dense_ids"]) - set(r["sparse_ids"])
    only_sparse = set(r["sparse_ids"]) - set(r["dense_ids"])
    print(f"\nBổ khuyết nhau: chỉ-dense-có = {len(only_dense)}, chỉ-sparse-có = {len(only_sparse)} "
          f"(trên prefetch {cfg['rrf']['prefetch_limit']} mỗi nhánh)")
    log.info("Smoke test xong. Nếu top-5 KHÔNG liên quan JAK/myelofibrosis/ruxolitinib -> nghi ngờ encoder/analyzer.")


if __name__ == "__main__":
    main()
