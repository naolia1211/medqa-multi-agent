"""
long_mem.py — Long-term memory cho V3 (spec §9): HỌC TỪ CÂU SAI.

Tham chiếu thiết kế /opt/medqa-long-term-memory (learn-from-mistakes), nhưng adapt vào
hạ tầng /opt/code: store JSONL local + embedding MedCPT (EmbedClient) + cosine numpy.
KHÔNG dùng langchain/chroma (giữ nhất quán codebase, không thêm dep nặng).

Cơ chế:
  - Chỉ lưu câu model trả SAI (kèm đáp án đúng) — không tốn LLM call để quyết "có đáng nhớ".
  - Câu mới -> truy hồi top-K lỗi tương tự (cosine >= ngưỡng) -> trả 'lessons' (text).
  - lessons được chèn vào prompt Agent 1 + Agent 2 (KHÔNG Agent 3) — đúng §9.

Vòng đời: pha HỌC (save_wrong bật) tích luỹ lessons; pha DÙNG (get_lessons) nạp vào agent.
Tất định: cùng store + cùng câu -> cùng lessons.
"""
import os
import json
import time
import threading

import numpy as np

from common import ROOT


class LongMemory:
    def __init__(self, cfg, embed_client, logger=None):
        m = (cfg or {}).get("memory", {}) or {}
        self.enabled = bool(m.get("enabled", False))
        self.save_wrong_on = bool(m.get("save_wrong", False))
        self.path = m.get("store", "index/lessons.jsonl")
        if not os.path.isabs(self.path):
            self.path = os.path.join(ROOT, self.path)
        self.top_k = int(m.get("top_k", 3))
        self.threshold = float(m.get("similarity_threshold", 0.35))  # cosine >= ngưỡng -> relevant
        self.ec = embed_client
        self.log = logger
        self._lock = threading.Lock()
        # BUG-2: tập lessons dùng cho retrieval được ĐÓNG BĂNG lúc init -> get_lessons luôn cho
        # cùng kết quả trong 1 run (không phụ thuộc thứ tự thread, TÁI LẬP). Câu sai lưu ở run này
        # chỉ ghi ra FILE (cho run SAU), KHÔNG thêm vào tập frozen đang dùng.
        self._lessons = self._load()
        self._seen_q = {l.get("question") for l in self._lessons}   # BUG-5: dedup theo câu hỏi
        # BUG-4: MedCPT embedding câu y khoa bị ANISOTROPIC (cosine câu-khác-nhau ~0.87-0.97) ->
        # absolute threshold vô dụng. Khử bằng MEAN-CENTERING: trừ vector trung bình rồi mới cosine
        # (cần >=2 lessons để ước lượng hướng anisotropy; ít hơn -> không retrieve).
        self._mean, self._mat = self._build_centered()

    def _build_centered(self):
        if len(self._lessons) < 2:
            return None, None
        M = np.asarray([l["emb"] for l in self._lessons], dtype=np.float32)   # đã chuẩn hoá lúc lưu
        mean = M.mean(axis=0)
        C = M - mean
        nrm = np.linalg.norm(C, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return mean, C / nrm

    def _load(self):
        out = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out

    def _embed(self, text):
        """MedCPT Article-Encoder (full text, không giới hạn 64 tok) -> vector chuẩn hoá (cosine=dot)."""
        v = np.asarray(self.ec.embed_article([text])[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def get_lessons(self, question):
        """Trả text các lỗi quá khứ tương tự (top-K, cosine-đã-centered >= ngưỡng), '' nếu tắt/không đủ."""
        if not self.enabled or self._mat is None:     # <2 lessons -> không đủ khử anisotropy
            return ""
        try:
            q = self._embed(question) - self._mean    # BUG-4: mean-centering
        except Exception as e:
            if self.log:
                self.log.warning("long_mem embed lỗi -> bỏ qua: %s", str(e)[:120])
            return ""
        nq = np.linalg.norm(q)
        if nq == 0:
            return ""
        sims = self._mat @ (q / nq)                    # cosine sau centering (frozen -> tất định)
        rel = []
        for idx in np.argsort(-sims):                  # từ tương tự nhất
            if len(rel) >= self.top_k:
                break
            if self._lessons[idx].get("question") == question:   # BUG-3: loại chính câu này (chống rò)
                continue
            if sims[idx] < self.threshold:             # đã sắp giảm dần -> phần còn lại đều dưới ngưỡng
                break
            rel.append(self._lessons[idx]["text"])
        return "\n\n".join(rel)

    def save_wrong(self, question, options, gold, pred, explanation="", topic="unknown"):
        """Lưu 1 lỗi (chỉ khi save_wrong bật). Không raise nếu lỗi embed/ghi. Thread-safe.
        Ghi ra FILE cho run SAU; KHÔNG thêm vào tập frozen đang dùng (giữ tái lập — BUG-2)."""
        if not (self.enabled and self.save_wrong_on):
            return False
        with self._lock:                          # BUG-5: dedup theo câu (đã có -> bỏ, tránh phình/trùng)
            if question in self._seen_q:
                return False
        try:
            gold_text = (options or {}).get(gold, "")
            text = (f"[PAST MISTAKE - topic: {topic}]\n"
                    f"Question: {question}\n"
                    f"Correct answer: {gold}. {gold_text}\n"
                    f"Model wrongly chose: {pred}\n"
                    f"Model's explanation: {str(explanation)[:500]}")
            rec = {"text": text, "emb": self._embed(question).tolist(),   # embed NGOÀI lock (chậm, gọi mạng)
                   "question": question, "gold": gold, "pred": pred, "topic": topic,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        except Exception as e:
            if self.log:
                self.log.warning("long_mem embed lỗi: %s", str(e)[:120])
            return False
        with self._lock:                          # BUG-1: ghi file + cập nhật _seen_q dưới lock (thread-safe)
            if question in self._seen_q:           # double-check sau khi embed
                return False
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._seen_q.add(question)
                return True
            except Exception as e:
                if self.log:
                    self.log.warning("long_mem save lỗi: %s", str(e)[:120])
                return False


def lessons_block(lessons):
    """Khối text chèn vào prompt Agent 1/2. Rỗng nếu không có lesson (không để header cụt)."""
    if not lessons:
        return ""
    return ("LESSONS FROM PAST MISTAKES (similar cases the model previously got WRONG — "
            "use them to avoid repeating the same error, but still reason from the evidence):\n"
            f"{lessons}\n\n")
