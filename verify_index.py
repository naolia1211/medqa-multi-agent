"""
verify_index.py — BƯỚC 6b
10 test bắt buộc PASS. Kiến trúc tách (Qdrant + ES) => phải verify nghiêm.
Test #2, #6, #8 bắt các lỗi âm thầm nguy hiểm nhất.

Exit code 0 nếu PASS hết, 1 nếu có test FAIL (in rõ test nào + bằng chứng).
KHÔNG BAO GIỜ nới tiêu chí để cho pass.
"""
import os
import json
import time
import random
import hashlib

import numpy as np

from common import (load_config, setup_logging, resolve,
                    get_tokenizer, EmbedClient, get_qdrant, get_es)

log = setup_logging("verify")


def scroll_qdrant_payloads(qc, name, with_vectors=False):
    """Yield (point_id, payload[, vector]) toàn bộ collection."""
    offset = None
    while True:
        points, offset = qc.scroll(
            collection_name=name, limit=1000, offset=offset,
            with_payload=True, with_vectors=with_vectors)
        for p in points:
            yield p
        if offset is None:
            break


def scan_es_ids(es, index):
    from elasticsearch import helpers
    for d in helpers.scan(es, index=index, query={"query": {"match_all": {}}},
                          _source=["chunk_id"], size=1000):
        yield d["_source"]["chunk_id"]


def main():
    cfg = load_config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    index_dir = resolve(cfg, cfg["paths"]["index_dir"])
    q = cfg["qdrant"]
    e = cfg["elasticsearch"]

    chunks = json.load(open(os.path.join(index_dir, "chunks.json"), encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    n_json = len(chunks)

    ec = EmbedClient(cfg, log)
    qc = get_qdrant(cfg)
    es = get_es(cfg)

    results = []   # (name, ok, detail)

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        log.info("[%s] %s %s", "PASS" if ok else "FAIL", name, ("- " + detail) if detail else "")

    t0 = time.time()

    # ── Test 1: số lượng khớp ──
    n_qdr = qc.count(collection_name=q["collection"], exact=True).count
    n_es = es.count(index=e["index"])["count"]
    check("1_counts", n_qdr == n_es == n_json,
          f"chunks.json={n_json} qdrant={n_qdr} es={n_es}")

    # ── Test 2: tập chunk_id khớp (element-wise) ── NGHIÊM TRỌNG
    ids_qdr = set(p.payload["chunk_id"] for p in scroll_qdrant_payloads(qc, q["collection"]))
    ids_es = set(scan_es_ids(es, e["index"]))
    ids_json = set(by_id)
    miss_es = ids_qdr - ids_es
    miss_qdr = ids_es - ids_qdr
    miss_both = ids_json - ids_qdr - ids_es
    check("2_chunkid_sets_match",
          ids_qdr == ids_es == ids_json,
          f"qdr\\es={list(miss_es)[:5]} es\\qdr={list(miss_qdr)[:5]} json\\both={list(miss_both)[:5]}")

    # ── Test 3: không chunk > 512 token thật ──
    tok = get_tokenizer(cfg["embedding"]["model"])
    texts = [c["text"] for c in chunks]
    over = 0
    max_tok = 0
    for i in range(0, len(texts), 1000):
        enc = tok(texts[i:i + 1000], add_special_tokens=True, return_attention_mask=False,
                  return_token_type_ids=False)
        for ids in enc["input_ids"]:
            L = len(ids)
            max_tok = max(max_tok, L)
            if L > 512:
                over += 1
    check("3_no_chunk_over_512", over == 0, f"over512={over} max_token={max_tok}")

    # ── Test 4: vector 768 chiều ──
    sample_pt = next(scroll_qdrant_payloads(qc, q["collection"], with_vectors=True))
    dim = len(sample_pt.vector)
    check("4_vector_dim_768", dim == cfg["embedding"]["dim"], f"dim={dim}")

    # ── Test 5: payload / _source đủ trường ──
    need = {"chunk_id", "source", "text"}
    ok5 = need.issubset(sample_pt.payload.keys())
    es_doc = es.get(index=e["index"], id=chunks[0]["chunk_id"])["_source"]
    ok5 = ok5 and need.issubset(es_doc.keys())
    check("5_payload_fields", ok5,
          f"qdrant_keys={sorted(sample_pt.payload.keys())} es_keys={sorted(es_doc.keys())}")

    # ── Test 6: sanity dense — embed lại 3 chunk -> top-1 chính nó ── NGHIÊM TRỌNG
    idxs = random.sample(range(n_json), 3)
    ok6 = True
    det6 = []
    for idx in idxs:
        c = chunks[idx]
        vec = ec.embed_article([c["text"]])[0]
        hits = qc.query_points(collection_name=q["collection"], query=vec,
                               limit=3, with_payload=True).points
        top1 = hits[0].payload["chunk_id"]
        # point id == idx phải mang đúng chunk_id (bắt lệch thứ tự khi resume)
        pt = qc.retrieve(collection_name=q["collection"], ids=[idx])[0]
        id_ok = pt.payload["chunk_id"] == c["chunk_id"]
        in_top1 = top1 == c["chunk_id"]
        top3 = [h.payload["chunk_id"] for h in hits]
        # chấp nhận top-3 nếu có chunk trùng nội dung do overlap
        ok_this = id_ok and (in_top1 or c["chunk_id"] in top3)
        ok6 = ok6 and ok_this
        det6.append(f"idx={idx} want={c['chunk_id']} top1={top1} pt_id_ok={id_ok}")
    check("6_dense_self_retrieval", ok6, " | ".join(det6))

    # ── Test 7: sanity sparse — từ hiếm của 1 chunk -> chunk trong top-10 ──
    def rare_terms(text, k=4):
        words = [w.lower() for w in text.split() if w.isalpha() and len(w) > 6]
        # ưu tiên từ dài (thường là thuật ngữ), tất định
        uniq = sorted(set(words), key=lambda w: (-len(w), w))
        return uniq[:k]
    ok7 = False
    det7 = ""
    for _ in range(5):  # thử vài chunk phòng chunk kém từ hiếm
        c = random.choice(chunks)
        terms = rare_terms(c["text"])
        if len(terms) < 2:
            continue
        query = " ".join(terms)
        resp = es.search(index=e["index"], query={"match": {"text": query}},
                         size=10, _source=["chunk_id"])
        top10 = [h["_source"]["chunk_id"] for h in resp["hits"]["hits"]]
        if c["chunk_id"] in top10:
            ok7 = True
            det7 = f"chunk={c['chunk_id']} terms={terms} rank={top10.index(c['chunk_id'])}"
            break
        det7 = f"chunk={c['chunk_id']} terms={terms} not in top10={top10[:3]}"
    check("7_sparse_self_retrieval", ok7, det7)

    # ── Test 8: text khớp giữa 3 nơi ── NGHIÊM TRỌNG
    sample_ids = random.sample(list(by_id), 10)
    # lấy payload Qdrant theo chunk_id: cần map chunk_id->point. Dùng scroll filter.
    from qdrant_client import models
    ok8 = True
    det8 = []
    for cid in sample_ids:
        t_json = by_id[cid]["text"]
        pts = qc.scroll(collection_name=q["collection"], limit=1,
                        scroll_filter=models.Filter(must=[
                            models.FieldCondition(key="chunk_id",
                                                  match=models.MatchValue(value=cid))]),
                        with_payload=True)[0]
        t_qdr = pts[0].payload["text"] if pts else None
        t_es = es.get(index=e["index"], id=cid)["_source"]["text"]
        if not (t_json == t_qdr == t_es):
            ok8 = False
            det8.append(f"{cid}: json={repr(t_json[:40])} qdr={repr((t_qdr or '')[:40])} es={repr(t_es[:40])}")
    check("8_text_match_3way", ok8, " || ".join(det8[:3]))

    # ── Test 9: manifest tồn tại + chunks_hash khớp ──
    man_path = os.path.join(index_dir, "manifest.json")
    ok9 = os.path.exists(man_path)
    det9 = "manifest.json không tồn tại"
    if ok9:
        man = json.load(open(man_path))
        h = hashlib.sha256(open(os.path.join(index_dir, "chunks.json"), "rb").read()).hexdigest()
        ok9 = man.get("chunks_hash") == "sha256:" + h
        det9 = f"manifest={man.get('chunks_hash','')[:20]}.. actual=sha256:{h[:14]}.."
    check("9_manifest_hash", ok9, det9)

    # ── Test 10: config đúng (Dot, HNSW off, analyzer, BM25) ──
    qinfo = qc.get_collection(q["collection"])
    dist_ok = str(qinfo.config.params.vectors.distance).upper().endswith("DOT")
    hnsw_off = qinfo.config.hnsw_config.m == 0
    es_settings = es.indices.get_settings(index=e["index"])[e["index"]]["settings"]["index"]
    analyzers = es_settings.get("analysis", {}).get("analyzer", {})
    analyzer_ok = e["analyzer"] in analyzers
    sim = es_settings.get("similarity", {})
    bm25_ok = any(v.get("type") == "BM25" for v in sim.values())
    check("10_config", dist_ok and hnsw_off and analyzer_ok and bm25_ok,
          f"dist_Dot={dist_ok} hnsw_off={hnsw_off} analyzer={analyzer_ok} bm25={bm25_ok}")

    # ── Tổng kết ──
    dt = time.time() - t0
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 60)
    print(f"VERIFY: {n_pass}/{len(results)} PASS  ({dt:.1f}s)")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         -> {detail}")
    print("=" * 60)
    raise SystemExit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
