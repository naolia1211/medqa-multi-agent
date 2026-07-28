"""
build_index.py — BƯỚC 4
Nạp CẢ Qdrant (dense) + ES (sparse) từ CÙNG một chunks.json, cùng một lần chạy.

Dense:  embed qua /embed/article (Article-Encoder) -> upsert Qdrant, point id = index chunk.
Sparse: bulk index ES, _id = chunk_id.

- Checkpoint/resume dense theo index (mỗi ~ upsert_batch chunk).
- Retry + backoff nằm trong EmbedClient.
- Point id = vị trí trong chunks.json => resume upsert đúng thứ tự (test #6 bắt lỗi này).

Chạy lại an toàn: dense resume từ checkpoint; sparse idempotent (_id=chunk_id, ghi đè).
Dùng --restart-dense để bỏ checkpoint và nạp dense lại từ đầu.
"""
import os
import json
import time
import argparse

from tqdm import tqdm

from common import (load_config, setup_logging, resolve,
                    EmbedClient, get_qdrant, get_es)

log = setup_logging("build")


def load_chunks(index_dir):
    with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# ── Dense (Qdrant) ───────────────────────────────────────────────────────────
def checkpoint_path(cfg):
    cp = resolve(cfg, cfg["paths"]["checkpoint"])
    os.makedirs(cp, exist_ok=True)
    return os.path.join(cp, "dense_progress.json")


def read_checkpoint(cfg):
    p = checkpoint_path(cfg)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("next_index", 0)
    return 0


def write_checkpoint(cfg, next_index):
    p = checkpoint_path(cfg)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"next_index": next_index}, f)
    os.replace(tmp, p)   # atomic


def build_dense(cfg, chunks, restart):
    from qdrant_client import models
    q = cfg["qdrant"]
    client = get_qdrant(cfg)
    ec = EmbedClient(cfg, log)
    name = q["collection"]
    api_batch = cfg["embed_api"]["batch_size"]      # 8
    upsert_batch = q["upsert_batch"]

    if restart:
        write_checkpoint(cfg, 0)
        log.info("--restart-dense: bỏ checkpoint, nạp dense từ đầu.")
    start = read_checkpoint(cfg)
    total = len(chunks)
    if start >= total:
        log.info("Dense đã nạp đủ %d/%d — bỏ qua.", start, total)
        return

    log.info("Dense: nạp từ index %d/%d (còn %d chunk). API batch=%d, upsert batch=%d",
             start, total, total - start, api_batch, upsert_batch)

    # kiểm tra dim ở batch đầu
    dim_checked = False
    t0 = time.time()
    pending = []   # list[PointStruct]
    i = start
    pbar = tqdm(total=total, initial=start, desc="embed+upsert", unit="chunk")
    while i < total:
        batch = chunks[i:i + api_batch]
        vecs = ec.embed_article([c["text"] for c in batch])
        if not dim_checked:
            assert len(vecs[0]) == cfg["embedding"]["dim"], \
                f"dim {len(vecs[0])} != {cfg['embedding']['dim']}"
            log.info("Batch đầu OK: vector %d chiều.", len(vecs[0]))
            dim_checked = True
        for j, (c, v) in enumerate(zip(batch, vecs)):
            pending.append(models.PointStruct(
                id=i + j,                       # point id = vị trí trong chunks.json
                vector=v,
                payload={"chunk_id": c["chunk_id"], "source": c["source"], "text": c["text"]},
            ))
        i += len(batch)
        pbar.update(len(batch))

        if len(pending) >= upsert_batch or i >= total:
            client.upsert(collection_name=name, points=pending, wait=True)
            pending = []
            write_checkpoint(cfg, i)
            done = i - start
            rate = done / max(1e-9, time.time() - t0)
            eta = (total - i) / max(1e-9, rate)
            pbar.set_postfix_str(f"{rate:.1f} chunk/s ETA {eta/60:.1f}m")
    pbar.close()
    dt = time.time() - t0
    cnt = client.count(collection_name=name, exact=True).count
    log.info("Dense xong: %d point trong Qdrant, %.1fs (%.1f chunk/s).",
             cnt, dt, (total - start) / max(1e-9, dt))
    return dt


# ── Sparse (Elasticsearch) ───────────────────────────────────────────────────
def build_sparse(cfg, chunks):
    from elasticsearch import helpers
    es = get_es(cfg)
    e = cfg["elasticsearch"]
    name = e["index"]
    t0 = time.time()

    def actions():
        for c in chunks:
            yield {
                "_index": name,
                "_id": c["chunk_id"],       # _id = chunk_id (khóa RRF, khớp Qdrant payload)
                "_source": {"chunk_id": c["chunk_id"], "source": c["source"], "text": c["text"]},
            }

    ok, errors = 0, []
    for success, info in helpers.streaming_bulk(
        es, actions(), chunk_size=e["bulk_chunk_size"], raise_on_error=False, max_retries=3):
        if success:
            ok += 1
        else:
            errors.append(info)
    es.indices.refresh(index=name)          # searchable ngay
    dt = time.time() - t0
    cnt = es.count(index=name)["count"]
    log.info("Sparse xong: %d doc index OK, %d lỗi, doc_count=%d, %.1fs.",
             ok, len(errors), cnt, dt)
    if errors:
        log.warning("Bulk errors (tối đa 5): %s", errors[:5])
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dense", "sparse"], help="Chỉ nạp một nhánh.")
    ap.add_argument("--restart-dense", action="store_true", help="Bỏ checkpoint, nạp dense từ đầu.")
    args = ap.parse_args()

    cfg = load_config()
    index_dir = resolve(cfg, cfg["paths"]["index_dir"])
    chunks = load_chunks(index_dir)
    log.info("=== BƯỚC 4: Nạp index từ chunks.json (%d chunk) ===", len(chunks))

    meta_path = os.path.join(index_dir, "index_meta.json")
    meta = {}
    if os.path.exists(meta_path):          # merge — giữ timing của nhánh chạy trước
        meta = json.load(open(meta_path))
    meta["n_chunks"] = len(chunks)
    if args.only != "sparse":
        meta["dense_seconds"] = build_dense(cfg, chunks, args.restart_dense)
    if args.only != "dense":
        meta["sparse_seconds"] = build_sparse(cfg, chunks)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Xong Bước 4. index_meta.json ghi xong.")


if __name__ == "__main__":
    main()
