"""
make_manifest.py — BƯỚC 6a
Sinh index/manifest.json SAU CÙNG (sau khi chunk + nạp + mọi thứ ổn định).
Mọi script downstream (kể cả online) phải đọc manifest và verify chunks_hash.
"""
import os
import json
import hashlib
import datetime

from common import load_config, setup_logging, resolve, get_qdrant, get_es

log = setup_logging("manifest")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = load_config()
    index_dir = resolve(cfg, cfg["paths"]["index_dir"])
    chunks_path = os.path.join(index_dir, "chunks.json")
    stats = json.load(open(os.path.join(index_dir, "corpus_stats.json")))
    imeta = {}
    ip = os.path.join(index_dir, "index_meta.json")
    if os.path.exists(ip):
        imeta = json.load(open(ip))

    chunks_hash = sha256_file(chunks_path)
    with open(chunks_path) as f:
        n_chunks = len(json.load(f))

    qc = get_qdrant(cfg)
    es = get_es(cfg)
    q = cfg["qdrant"]
    e = cfg["elasticsearch"]
    qinfo = qc.get_collection(q["collection"])
    points_count = qc.count(collection_name=q["collection"], exact=True).count
    doc_count = es.count(index=e["index"])["count"]

    # build_id tất định-ish theo thời điểm chạy (được phép — chỉ là nhãn)
    now = datetime.datetime.now()
    manifest = {
        "build_id": now.strftime("%Y%m%d_%H%M%S"),
        "built_at": now.isoformat(timespec="seconds"),
        "chunks_hash": "sha256:" + chunks_hash,
        "n_chunks": n_chunks,
        "chunking": {
            "target_tokens": stats["target_tokens"],
            "hard_max_tokens": stats["hard_max_tokens"],
            "overlap_sentences": stats["overlap_sentences"],
            "min_tokens": stats["min_tokens"],
            "subword_ratio_actual": stats["subword_ratio_actual"],
            "max_token_actual": stats["token_max"],
        },
        "dense": {
            "backend": "qdrant", "collection": q["collection"],
            "model": cfg["embedding"]["model"], "dim": cfg["embedding"]["dim"],
            "distance": q["distance"], "normalize": cfg["embedding"]["normalize"],
            "exact_search": bool(qinfo.config.hnsw_config.m == 0),
            "points_count": points_count,
            "api_base": cfg["embed_api"]["base_url"],
            "seconds": imeta.get("dense_seconds"),
        },
        "sparse": {
            "backend": "elasticsearch", "index": e["index"],
            "analyzer": e["analyzer"], "similarity": e["similarity"]["type"],
            "k1": e["similarity"]["k1"], "b": e["similarity"]["b"],
            "doc_count": doc_count,
            "seconds": imeta.get("sparse_seconds"),
        },
        "rrf": {"k": cfg["rrf"]["k"], "prefetch_limit": cfg["rrf"]["prefetch_limit"]},
        "files": {
            "chunks.json": {
                "sha256": chunks_hash,
                "size_mb": round(os.path.getsize(chunks_path) / 1e6, 1),
            }
        },
        "corpus_stats": stats,
    }
    out = os.path.join(index_dir, "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("Ghi %s (n_chunks=%d, qdrant=%d, es=%d).",
             out, n_chunks, points_count, doc_count)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
