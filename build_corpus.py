"""
build_corpus.py — BƯỚC 3
Gom 18 textbook -> chunk. Đếm token THẬT bằng tokenizer MedCPT (không xấp xỉ).

Chiến lược:
  1. Tách đoạn theo '\n\n'
  2. GOM đoạn ngắn: pack câu liên tiếp tới target_tokens rồi mới cắt
     (bắt buộc — Pathoma median 9 từ/đoạn, First_Aid ~15 -> chunk thẳng = rác)
  3. Recursive split theo câu (?<=[.!?])\s+
  4. Câu đơn > target -> cắt cứng theo TOKEN
  5. Overlap 1 câu giữa chunk liền kề
  6. Bỏ chunk < min_tokens
  7. VERIFY lại mọi chunk bằng tokenizer: > hard_max -> cắt tiếp (không lọt)

Tất định: file sort theo tên, câu theo thứ tự, chunk_id = "{source}__{seq}" tuần tự.
"""
import os
import re
import glob
import json
import time
import statistics

import numpy as np
from tqdm import tqdm

from common import load_config, setup_logging, resolve, get_tokenizer

log = setup_logging("corpus")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(paragraph):
    return [s.strip() for s in SENT_SPLIT.split(paragraph) if s.strip()]


def batch_token_counts(tok, texts):
    """Đếm token nội dung (KHÔNG special) cho nhiều text 1 lần — nhanh (Rust)."""
    if not texts:
        return []
    enc = tok(texts, add_special_tokens=False, return_attention_mask=False,
              return_token_type_ids=False)
    return [len(ids) for ids in enc["input_ids"]]


def hard_split_by_tokens(tok, text, budget_content):
    """Cắt cứng 1 câu quá dài thành các mảnh <= budget_content token nội dung."""
    ids = tok.encode(text, add_special_tokens=False)
    pieces = []
    for i in range(0, len(ids), budget_content):
        window = ids[i:i + budget_content]
        piece = tok.decode(window, skip_special_tokens=True,
                           clean_up_tokenization_spaces=True).strip()
        if piece:
            pieces.append(piece)
    return pieces


def build_sentence_stream(tok, file_text, budget_content):
    """
    File -> list[(sentence, n_content_tokens)].
    Câu dài hơn budget_content bị cắt cứng thành nhiều mảnh (mỗi mảnh <= budget).
    Việc pack câu liên tiếp (bất kể ranh giới đoạn) chính là 'gom đoạn ngắn'.
    """
    paras = [p.strip() for p in file_text.split("\n\n") if p.strip()]
    raw = []
    for p in paras:
        raw.extend(split_sentences(p))
    if not raw:
        return []
    counts = batch_token_counts(tok, raw)
    stream = []
    for s, n in zip(raw, counts):
        if n <= budget_content:
            stream.append((s, n))
        else:
            for piece in hard_split_by_tokens(tok, s, budget_content):
                pc = batch_token_counts(tok, [piece])[0]
                stream.append((piece, pc))
    return stream


def pack_chunks(stream, budget_content, overlap_sentences):
    """
    Greedy pack câu tới budget_content, overlap `overlap_sentences` câu.
    An toàn: mọi câu <= budget_content (đã pre-split) nên luôn tiến (không loop vô hạn).
    """
    chunks = []
    cur = []          # list[(sentence, n)]
    cur_tok = 0
    idx = 0
    n = len(stream)
    while idx < n:
        sent, ntok = stream[idx]
        if cur and cur_tok + ntok > budget_content:
            chunks.append([s for s, _ in cur])
            # overlap: giữ lại `overlap_sentences` câu cuối
            keep = cur[-overlap_sentences:] if overlap_sentences > 0 else []
            cur = list(keep)
            cur_tok = sum(x[1] for x in cur)
            # nếu overlap khiến câu mới không vừa -> bỏ overlap để đảm bảo tiến
            if cur_tok + ntok > budget_content:
                cur = []
                cur_tok = 0
        cur.append((sent, ntok))
        cur_tok += ntok
        idx += 1
    if cur:
        chunks.append([s for s, _ in cur])
    return chunks


def enforce_hard_max(tok, text, hard_max):
    """
    VERIFY cuối cùng bằng tokenizer thật (gồm CLS/SEP).
    Trả về list[text] mỗi phần <= hard_max token thật. Thường trả 1 phần.
    """
    n = len(tok.encode(text, add_special_tokens=True))
    if n <= hard_max:
        return [text]
    # Hiếm khi xảy ra — cắt cứng theo token nội dung (chừa 2 cho CLS/SEP)
    budget = hard_max - 2
    ids = tok.encode(text, add_special_tokens=False)
    out = []
    for i in range(0, len(ids), budget):
        piece = tok.decode(ids[i:i + budget], skip_special_tokens=True,
                           clean_up_tokenization_spaces=True).strip()
        if piece:
            # đệ quy phòng trường hợp decode làm token nở ra
            out.extend(enforce_hard_max(tok, piece, hard_max))
    return out


def main():
    cfg = load_config()
    ch = cfg["chunking"]
    target = ch["target_tokens"]
    hard_max = ch["hard_max_tokens"]
    overlap = ch["overlap_sentences"]
    min_tokens = ch["min_tokens"]
    budget_content = target - 2   # chừa [CLS], [SEP]

    corpus_dir = resolve(cfg, cfg["paths"]["corpus_dir"])
    index_dir = resolve(cfg, cfg["paths"]["index_dir"])
    os.makedirs(index_dir, exist_ok=True)
    out_path = os.path.join(index_dir, "chunks.json")

    log.info("=== BƯỚC 3: Chunk (target=%d, hard_max=%d, overlap=%d câu, min=%d) ===",
             target, hard_max, overlap, min_tokens)
    log.info("Tải tokenizer thật: %s", cfg["embedding"]["model"])
    tok = get_tokenizer(cfg["embedding"]["model"])

    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not files:
        raise SystemExit(f"Không thấy file .txt trong {corpus_dir}")
    log.info("Tìm thấy %d file.", len(files))

    all_chunks = []
    per_source_count = {}
    all_token_counts = []
    total_words = 0
    total_tokens_content = 0   # để tính subword ratio thật toàn corpus
    t0 = time.time()

    for fp in files:
        source = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        stream = build_sentence_stream(tok, text, budget_content)
        packed = pack_chunks(stream, budget_content, overlap)

        seq = 0
        kept = 0
        for chunk_sents in tqdm(packed, desc=source, leave=False):
            chunk_text = " ".join(chunk_sents)
            for piece in enforce_hard_max(tok, chunk_text, hard_max):
                n_tok = len(tok.encode(piece, add_special_tokens=True))
                if n_tok < min_tokens:
                    continue
                cid = f"{source}__{seq}"
                seq += 1
                kept += 1
                all_chunks.append({"chunk_id": cid, "source": source, "text": piece})
                all_token_counts.append(n_tok)
                # thống kê subword ratio (dùng token nội dung / số từ)
                w = len(piece.split())
                total_words += w
                total_tokens_content += (n_tok - 2)
        per_source_count[source] = kept
        log.info("%-24s -> %6d chunk", source, kept)

    # Ghi output (tất định: thứ tự theo file sort + seq)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    dt = time.time() - t0

    # ── Thống kê ──
    arr = np.array(all_token_counts)
    subword_ratio = total_tokens_content / total_words if total_words else 0.0
    max_tok = int(arr.max())

    log.info("=" * 64)
    log.info("Tổng chunk: %d  (ghi %s, %.1f MB)",
             len(all_chunks), out_path, os.path.getsize(out_path) / 1e6)
    log.info("Token thật/chunk: min=%d median=%d p90=%d p99=%d max=%d",
             int(arr.min()), int(np.median(arr)),
             int(np.percentile(arr, 90)), int(np.percentile(arr, 99)), max_tok)
    log.info("SUBWORD RATIO thật toàn corpus (token nội dung / từ) = %.4f", subword_ratio)
    log.info("Xác nhận max token = %d  %s 512", max_tok, "<=" if max_tok <= 512 else ">")
    if max_tok > hard_max:
        raise SystemExit(f"LỖI: có chunk {max_tok} > hard_max {hard_max} — không được để lọt!")
    log.info("Thời gian chunk: %.1fs", dt)
    log.info("=" * 64)

    # Lưu stats phụ để verify/manifest dùng lại
    stats = {
        "n_chunks": len(all_chunks),
        "per_source": per_source_count,
        "token_min": int(arr.min()),
        "token_median": int(np.median(arr)),
        "token_p90": int(np.percentile(arr, 90)),
        "token_max": max_tok,
        "subword_ratio_actual": round(subword_ratio, 4),
        "chunk_seconds": round(dt, 1),
        "target_tokens": target, "hard_max_tokens": hard_max,
        "overlap_sentences": overlap, "min_tokens": min_tokens,
    }
    with open(os.path.join(index_dir, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
