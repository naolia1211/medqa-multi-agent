"""
setup_indices.py — BƯỚC 2
Tạo Qdrant collection (HNSW TẮT, distance Dot) + ES index (analyzer y khoa, BM25).

An toàn chạy lại: mặc định KHÔNG xóa nếu đã tồn tại (tránh mất data vô ý).
Dùng --recreate để xóa và tạo lại từ đầu (khi cần build sạch).
"""
import argparse
import sys

from common import load_config, setup_logging, get_qdrant, get_es

log = setup_logging("setup")


def setup_qdrant(cfg, recreate):
    from qdrant_client import models
    q = cfg["qdrant"]
    client = get_qdrant(cfg)
    name = q["collection"]

    exists = client.collection_exists(name)
    if exists and not recreate:
        log.info("Qdrant collection '%s' đã tồn tại — bỏ qua (dùng --recreate để tạo lại).", name)
        return
    if exists and recreate:
        log.warning("Xóa Qdrant collection '%s' (--recreate).", name)
        client.delete_collection(name)

    distance = getattr(models.Distance, q["distance"].upper())  # DOT
    # TẮT HNSW: m=0 -> exact (brute-force) search. 40K vector rất nhỏ, exact chỉ vài ms.
    hnsw = models.HnswConfigDiff(m=0) if q.get("exact_search", True) else None

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=q["vector_size"], distance=distance),
        hnsw_config=hnsw,
    )
    info = client.get_collection(name)
    log.info(
        "Qdrant '%s' đã tạo: size=%d distance=%s hnsw.m=%s",
        name, q["vector_size"], q["distance"],
        getattr(getattr(info.config.hnsw_config, "m", "?"), "__int__", lambda: info.config.hnsw_config.m)(),
    )


def setup_es(cfg, recreate):
    es = get_es(cfg)
    e = cfg["elasticsearch"]
    name = e["index"]
    sim = e["similarity"]

    exists = es.indices.exists(index=name)
    if exists and not recreate:
        log.info("ES index '%s' đã tồn tại — bỏ qua (dùng --recreate để tạo lại).", name)
        return
    if exists and recreate:
        log.warning("Xóa ES index '%s' (--recreate).", name)
        es.indices.delete(index=name)

    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    # Analyzer y khoa: standard tokenizer + lowercase + stopword + stemmer NHẸ
                    e["analyzer"]: {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_stop", "english_stemmer"],
                    }
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    # minimal_english: CHỈ xử lý số nhiều — không phá thuật ngữ Latin/tên thuốc
                    "english_stemmer": {"type": "stemmer", "language": "minimal_english"},
                },
            },
            "similarity": {
                "medical_bm25": {"type": sim["type"], "k1": sim["k1"], "b": sim["b"]},
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                # keyword: KHÔNG analyze — phải khớp CHÍNH XÁC chunk_id trong Qdrant (khóa RRF)
                "chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": e["analyzer"],
                    "similarity": "medical_bm25",
                },
            }
        },
    }
    es.indices.create(index=name, body=body)
    log.info("ES '%s' đã tạo: analyzer=%s similarity=BM25(k1=%s,b=%s)",
             name, e["analyzer"], sim["k1"], sim["b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="Xóa collection/index cũ rồi tạo lại (build sạch).")
    args = ap.parse_args()

    cfg = load_config()
    log.info("=== BƯỚC 2: Tạo Qdrant collection + ES index (recreate=%s) ===", args.recreate)
    setup_qdrant(cfg, args.recreate)
    setup_es(cfg, args.recreate)
    log.info("Xong Bước 2.")


if __name__ == "__main__":
    main()
