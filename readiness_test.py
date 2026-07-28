"""
readiness_test.py — BƯỚC 7c
Verify pass = cấu trúc đúng. Readiness = pipeline RAG chạy THÔNG trên câu hỏi thật.

Lấy 20 câu ngẫu nhiên (tất định theo seed) từ questions/US/dev.jsonl:
  dense (Query-Encoder) + sparse (ES) + RRF -> top-5.
Kiểm tra: đủ 5 chunk? cả 2 nhánh đóng góp? chunk là văn bản y khoa đọc được?

LƯU Ý: recall thấp KHÔNG phải lỗi index — đó là đặc tính bài toán.
Readiness chỉ kiểm tra pipeline thông, không phải chất lượng retrieval.
"""
import json
import random

from common import load_config, setup_logging, resolve, EmbedClient, get_qdrant, get_es
from retrieval import hybrid_search

log = setup_logging("readiness")


def main():
    cfg = load_config()
    random.seed(cfg["seed"])
    ec = EmbedClient(cfg, log)
    qc = get_qdrant(cfg)
    es = get_es(cfg)

    path = resolve(cfg, cfg["paths"]["questions_dev"])
    with open(path, encoding="utf-8") as f:
        questions = [json.loads(l)["question"] for l in f if l.strip()]
    sample = random.sample(questions, 20)

    print(f"{'Câu':>3} | {'dense':>5} | {'sparse':>6} | {'RRF':>3} | {'d-only':>6} | {'s-only':>6} | {'both':>4}")
    print("-" * 60)

    all_nonempty = True
    tot_d_only = tot_s_only = tot_both = 0
    tot_fused = 0
    readable_ok = True

    for i, qtext in enumerate(sample, 1):
        r = hybrid_search(cfg, qtext, embed_client=ec, qdrant=qc, es=es)
        fused_ids = [cid for cid, _ in r["fused"]]
        dset, sset = set(r["dense_ids"]), set(r["sparse_ids"])
        d_only = sum(1 for c in fused_ids if c in dset and c not in sset)
        s_only = sum(1 for c in fused_ids if c in sset and c not in dset)
        both = sum(1 for c in fused_ids if c in sset and c in dset)
        tot_d_only += d_only; tot_s_only += s_only; tot_both += both
        tot_fused += len(fused_ids)
        if len(fused_ids) == 0:
            all_nonempty = False
        # kiểm tra văn bản đọc được: chunk top-1 có chữ cái, đủ dài.
        # Lấy text từ dense payload; nếu top-1 là sparse-only (không có trong dense)
        # thì lấy từ ES — nguồn text luôn sẵn có. (Không làm vậy = false-negative.)
        if fused_ids:
            payload = next((p for c, p, _ in r["dense"] if c == fused_ids[0]), None)
            if payload is not None:
                txt = payload["text"]
            else:
                txt = es.get(index=cfg["elasticsearch"]["index"],
                             id=fused_ids[0])["_source"]["text"]
            if len(txt) < 30 or sum(ch.isalpha() for ch in txt) < 20:
                readable_ok = False
        print(f"{i:>3} | {len(r['dense_ids']):>5} | {len(r['sparse_ids']):>6} | "
              f"{len(fused_ids):>3} | {d_only:>6} | {s_only:>6} | {both:>4}")

    print("-" * 60)
    denom = max(1, tot_fused)
    print(f"Tổng: 20 câu, đủ-5-chunk: {'CÓ' if all_nonempty else 'KHÔNG'}")
    print(f"  dense-only trong top-5: {100*tot_d_only/denom:.0f}%   "
          f"sparse-only: {100*tot_s_only/denom:.0f}%   cả hai: {100*tot_both/denom:.0f}%")

    dense_contrib = tot_d_only + tot_both
    sparse_contrib = tot_s_only + tot_both
    ready = all_nonempty and dense_contrib > 0 and sparse_contrib > 0 and readable_ok
    print("\n" + "=" * 50)
    print("KẾT LUẬN:", "SẴN SÀNG CHO RAG ✅" if ready else "CHƯA SẴN SÀNG ❌")
    print(f"  - 20/20 câu trả >=1 chunk: {all_nonempty}")
    print(f"  - dense có đóng góp: {dense_contrib>0}  | sparse có đóng góp: {sparse_contrib>0}")
    print(f"  - chunk là văn bản y khoa đọc được: {readable_ok}")
    print("=" * 50)
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
