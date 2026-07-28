"""
retrieval.py — BƯỚC 5
Hàm search hybrid dùng chung: dense (Qdrant) + sparse (ES) + RRF.

RRF chỉ dùng THỨ HẠNG, không dùng score gốc:
  - BM25 score (không giới hạn, có thể 250+) và dot-product MedCPT ở thang khác hẳn,
    không so sánh trực tiếp được. RRF né vấn đề bằng cách chỉ nhìn rank.
"""
from common import get_qdrant, get_es, EmbedClient


def rrf_fuse(ranked_lists, k=60, top_n=5):
    """
    ranked_lists: list các danh sách chunk_id đã xếp hạng (rank 0 = tốt nhất).
    RRF score(cid) = SUM over lists 1/(k + rank + 1).   (+1 vì rank bắt đầu từ 0)
    Trả về: list[(chunk_id, score)] top_n.
    """
    scores = {}
    for hits in ranked_lists:
        for rank, cid in enumerate(hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))  # tie-break theo cid: tất định
    return ranked[:top_n]


def query_windows(cfg, query_text):
    """
    Cắt query dài thành các cửa sổ <=64 token (overlap) cho Query-Encoder.
    MedCPT Query-Encoder cắt cứng 64 token; vignette MedQA median 155 tok với câu
    hỏi + lab ở cuối -> nếu gửi nguyên, server chỉ thấy 64 token đầu. Trả list[str].
    """
    dq = cfg["dense_query"]
    from common import get_query_tokenizer
    tok = get_query_tokenizer(dq["query_tokenizer"])
    ids = tok.encode(query_text, add_special_tokens=False)
    win = dq["window_tokens"]
    stride = max(1, win - dq["overlap_tokens"])
    max_win = dq["max_windows"]
    if len(ids) <= win:
        return [query_text]
    windows, i = [], 0
    while i < len(ids) and len(windows) < max_win:
        windows.append(tok.decode(ids[i:i + win], clean_up_tokenization_spaces=True))
        if i + win >= len(ids):
            break
        i += stride
    return windows


def dense_search(cfg, embed_client, query_text, limit, qdrant=None):
    """
    Embed query bằng QUERY-Encoder (KHÔNG article) -> search Qdrant.
    Nếu dense_query.multi_window: cắt query dài thành cửa sổ <=64 tok, embed từng
    cửa sổ, search mỗi vector, GỘP bằng RRF -> toàn bộ vignette tham gia dense.
    -> list[(cid, payload, score)].
    """
    qc = qdrant or get_qdrant(cfg)
    name = cfg["qdrant"]["collection"]
    if cfg.get("dense_query", {}).get("multi_window"):
        windows = query_windows(cfg, query_text)
    else:
        windows = [query_text]

    vecs = embed_client.embed_query(windows)      # <=16/request; max_windows<=8 -> 1 request
    if len(vecs) == 1:
        hits = qc.query_points(collection_name=name, query=vecs[0],
                               limit=limit, with_payload=True).points
        return [(h.payload["chunk_id"], h.payload, h.score) for h in hits]

    # nhiều cửa sổ -> mỗi cửa sổ 1 ranked list, gộp RRF theo chunk_id
    ranked_lists, payloads, best_score = [], {}, {}
    for v in vecs:
        hits = qc.query_points(collection_name=name, query=v,
                               limit=limit, with_payload=True).points
        ranked_lists.append([h.payload["chunk_id"] for h in hits])
        for h in hits:
            cid = h.payload["chunk_id"]
            payloads.setdefault(cid, h.payload)
            best_score[cid] = max(best_score.get(cid, -1e30), h.score)  # score dense tốt nhất
    fused = rrf_fuse(ranked_lists, k=cfg["rrf"]["k"], top_n=limit)
    return [(cid, payloads[cid], best_score[cid]) for cid, _ in fused]


def sparse_search(cfg, query_text, limit, es=None):
    """ES match trên field text. -> list[(cid, source, score)]."""
    esc = es or get_es(cfg)
    resp = esc.search(
        index=cfg["elasticsearch"]["index"],
        query={"match": {"text": query_text}},
        size=limit, _source=["chunk_id", "source"],
    )
    return [(h["_source"]["chunk_id"], h["_source"]["source"], h["_score"])
            for h in resp["hits"]["hits"]]


def hybrid_search(cfg, query_text, embed_client=None, qdrant=None, es=None, top_n=None):
    """
    Chạy cả 2 nhánh + RRF. Trả dict gồm dense_ids, sparse_ids, fused.
    top_n: số kết quả fused (mặc định rrf.top_n; online rerank truyền rrf.candidates).
    """
    ec = embed_client or EmbedClient(cfg)
    limit = cfg["rrf"]["prefetch_limit"]
    top_n = top_n or cfg["rrf"]["top_n"]
    dense = dense_search(cfg, ec, query_text, limit, qdrant)
    sparse = sparse_search(cfg, query_text, limit, es)
    dense_ids = [cid for cid, _, _ in dense]
    sparse_ids = [cid for cid, _, _ in sparse]
    fused = rrf_fuse([dense_ids, sparse_ids], k=cfg["rrf"]["k"], top_n=top_n)
    return {
        "dense": dense, "sparse": sparse,
        "dense_ids": dense_ids, "sparse_ids": sparse_ids,
        "fused": fused,
    }


def _fetch_texts(cfg, chunk_ids, dense_hits, es):
    """
    Lấy text cho các chunk_id fused. Ưu tiên payload dense (đã có sẵn),
    còn lại (sparse-only) lấy 1 lần qua ES mget. -> dict[cid] = {"text","source"}.
    """
    out = {}
    dense_map = {cid: p for cid, p, _ in dense_hits}
    missing = []
    for cid in chunk_ids:
        if cid in dense_map:
            out[cid] = {"text": dense_map[cid]["text"], "source": dense_map[cid]["source"]}
        else:
            missing.append(cid)
    if missing:
        docs = es.mget(index=cfg["elasticsearch"]["index"], ids=missing,
                       _source=["text", "source"])["docs"]
        for d in docs:
            s = d.get("_source", {})
            out[d["_id"]] = {"text": s.get("text", ""), "source": s.get("source", "")}
    return out


def retrieve_rerank(cfg, question, embed_client=None, qdrant=None, es=None):
    """
    Pipeline retrieval ONLINE (không HyDE):
      dense(20) + sparse(20) -> RRF -> top-`candidates` -> MedCPT rerank -> top_k.
    Trả list[{chunk_id, source, text, rrf_rank, rerank_score}] (đã xếp theo rerank).
    Nếu rerank.enabled=false: trả thẳng top_k theo RRF.
    """
    ec = embed_client or EmbedClient(cfg)
    esc = es or get_es(cfg)
    rr = cfg["rerank"]
    cand_n = rr["candidates"]
    top_k = rr["top_k"]

    h = hybrid_search(cfg, question, embed_client=ec, qdrant=qdrant, es=esc, top_n=cand_n)
    cand_ids = [cid for cid, _ in h["fused"]]
    texts = _fetch_texts(cfg, cand_ids, h["dense"], esc)

    if not rr.get("enabled", True) or not cand_ids:
        return [{"chunk_id": cid, "source": texts[cid]["source"], "text": texts[cid]["text"],
                 "rrf_rank": i, "rerank_score": None}
                for i, cid in enumerate(cand_ids[:top_k])]

    # rerank (<= server_max mỗi request; candidates<=20 nên 1 request là đủ)
    articles = [{"id": cid, "title": texts[cid]["text"], "abstract": ""} for cid in cand_ids]
    results = ec.rerank(question, articles[:rr["server_max"]], top_k=top_k)
    out = []
    for item in results:
        cid = item["id"] if item.get("id") else cand_ids[item["index"]]
        out.append({
            "chunk_id": cid, "source": texts[cid]["source"], "text": texts[cid]["text"],
            "rrf_rank": cand_ids.index(cid) if cid in cand_ids else None,
            "rerank_score": item["score"],
        })
    return out
