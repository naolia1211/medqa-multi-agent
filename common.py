"""
common.py — tiện ích dùng chung cho toàn pipeline.
Load config, tokenizer thật, client embedding API, client Qdrant/ES.
Không hardcode: mọi tham số đọc từ config.yaml.
"""
import os
import sys
import json
import time
import logging
import functools
import threading

import yaml
import requests

# ── Đường dẫn gốc = thư mục chứa file này ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(cfg, rel):
    """Đường dẫn tương đối trong config -> tuyệt đối theo ROOT."""
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(ROOT, rel))


def setup_logging(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


# ── Tokenizer thật (MedCPT-Article-Encoder, BERT WordPiece) ──────────────────
@functools.lru_cache(maxsize=1)
def get_tokenizer(model_name):
    """
    Tải tokenizer THẬT. Nếu không tải được -> raise (STOP condition #3).
    KHÔNG fallback về xấp xỉ.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            f"Không import được transformers: {e}. "
            "Cần transformers cho tokenizer thật — KHÔNG fallback xấp xỉ."
        )
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"STOP: không tải được tokenizer '{model_name}': {e}. "
            "Mạng nội bộ có thể chặn HuggingFace. KHÔNG fallback xấp xỉ — báo người dùng."
        )
    return tok


def count_tokens(tok, text):
    """Đếm token THẬT gồm [CLS], [SEP]."""
    return len(tok.encode(text, add_special_tokens=True))


_query_tok_lock = threading.Lock()
_query_tok_cache = {}

def get_query_tokenizer(model_name):
    """Tokenizer của Query-Encoder — để cắt query dài thành cửa sổ <=64 token.
    Thread-safe (double-checked locking): serialize lần nạp ĐẦU để tránh race lazy-import
    transformers khi nhiều luồng cùng gọi lần đầu (fast-path không lock sau khi đã cache)."""
    tok = _query_tok_cache.get(model_name)
    if tok is None:
        with _query_tok_lock:
            tok = _query_tok_cache.get(model_name)
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_name)
                _query_tok_cache[model_name] = tok
    return tok


# ── Client embedding API ─────────────────────────────────────────────────────
class EmbedClient:
    """
    Gọi server MedCPT. Retry + exponential backoff.
    - embed_article: cho CHUNK (Article-Encoder). Bắt buộc cho index dense.
    - embed_query:   cho QUERY (Query-Encoder). CHỈ dùng ở phần online/smoke/readiness.
    """

    def __init__(self, cfg, logger=None):
        e = cfg["embed_api"]
        self.base = e["base_url"].rstrip("/")
        self.batch_size = e["batch_size"]
        self.timeout = e["timeout"]
        self.retry = e["retry"]
        self.backoff_base = e.get("backoff_base", 2.0)
        self.normalize = cfg["embedding"]["normalize"]
        self.dim = cfg["embedding"]["dim"]
        self.log = logger or logging.getLogger("embed")
        self.session = requests.Session()

    def _post(self, path, payload):
        last = None
        for attempt in range(self.retry):
            try:
                r = self.session.post(
                    self.base + path, json=payload, timeout=self.timeout
                )
                if r.status_code == 501:
                    raise RuntimeError(
                        f"STOP: {path} trả 501 — server chưa bật LOAD_ARTICLE_ENCODER. "
                        "KHÔNG dùng /embed/query thay thế. Báo người dùng."
                    )
                if r.status_code == 200:
                    return r.json()
                # 4xx (trừ 501) là lỗi client — retry vô ích
                if 400 <= r.status_code < 500:
                    raise RuntimeError(
                        f"{path} trả {r.status_code}: {r.text[:300]}"
                    )
                last = RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
            except requests.RequestException as ex:
                last = ex
            wait = self.backoff_base ** attempt
            self.log.warning(
                "retry %d/%d %s sau %.1fs (%s)",
                attempt + 1, self.retry, path, wait, str(last)[:120],
            )
            time.sleep(wait)
        raise RuntimeError(f"Hết {self.retry} retry cho {path}: {last}")

    def embed_article(self, texts):
        """texts: list[str] (<= batch_size). title=text, abstract=''. -> list[vec]."""
        if len(texts) > self.batch_size:
            raise ValueError(
                f"batch {len(texts)} > giới hạn {self.batch_size} của /embed/article"
            )
        payload = {
            "articles": [{"id": str(i), "title": t, "abstract": ""} for i, t in enumerate(texts)],
            "normalize": self.normalize,
        }
        j = self._post("/embed/article", payload)
        embs = j["embeddings"]
        if j.get("dim") != self.dim:
            raise RuntimeError(f"dim {j.get('dim')} != {self.dim} kỳ vọng")
        return embs

    def embed_query(self, texts):
        """CHỈ dùng cho query online. Query-Encoder. -> list[vec]."""
        payload = {"queries": list(texts), "normalize": self.normalize}
        j = self._post("/embed/query", payload)
        return j["embeddings"]

    def rerank(self, query, articles, top_k=None):
        """
        MedCPT Cross-Encoder rerank. articles: list[{"id","title","abstract"}] (<=20).
        -> list[{"index","score","id","title"}] đã xếp theo score giảm dần.
        """
        if len(articles) > 20:
            raise ValueError(f"rerank batch {len(articles)} > 20 (giới hạn server)")
        payload = {"query": query, "articles": articles}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._post("/rerank", payload)["results"]

    def health(self):
        r = self.session.get(self.base + "/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ── Clients Qdrant / ES ──────────────────────────────────────────────────────
# ── LLM client (OpenRouter) — xoay vòng nhiều key + failover ─────────────────
class LLMClient:
    """
    Gọi OpenRouter. Xoay vòng (round-robin) qua nhiều API key + failover:
    gặp 429 (rate limit) / 402 (hết quota) / 5xx -> đổi key, thử lại.
    Thread-safe để eval song song.
    """

    def __init__(self, cfg, logger=None):
        import threading
        c = cfg["llm"]
        self.base = c["base_url"]
        self.model = c["model"]
        self.temperature = c["temperature"]
        self.max_tokens = c["max_tokens"]
        self.timeout = c["timeout"]
        self.retry = c["retry"]
        self.reasoning = c.get("reasoning", False)
        self.log = logger or logging.getLogger("llm")
        keys_path = os.path.join(ROOT, c["keys_file"]) if not os.path.isabs(c["keys_file"]) else c["keys_file"]
        with open(keys_path) as f:
            self.keys = [ln.strip() for ln in f if ln.strip()]
        if not self.keys:
            raise RuntimeError(f"Không có key trong {keys_path}")
        self._lock = threading.Lock()
        self._rr = 0
        self._bad = set()          # key trả 401/403 -> loại vĩnh viễn
        self.session = requests.Session()

    def _next_key(self):
        with self._lock:
            k = self.keys[self._rr % len(self.keys)]
            self._rr += 1
            return k

    def chat(self, messages):
        """messages: list[{'role','content'}] -> nội dung trả lời (str)."""
        return self.chat_usage(messages)[0]

    def chat_usage(self, messages):
        """Như chat() nhưng trả (content, usage). usage={'prompt_tokens','completion_tokens'}
        lấy TỪ API (đúng cả model :free), hoặc None nếu response không kèm usage."""
        body = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if self.reasoning:
            body["reasoning"] = {"enabled": True}
        last = None
        n = len(self.keys)
        # retry theo VÒNG: mỗi vòng xoay hết key còn tốt (không sleep giữa key).
        # Chỉ sleep giữa các vòng -> key chết (401/429) bị bỏ qua nhanh, không làm chậm.
        for round_i in range(self.retry):
            tried = 0
            for _ in range(n):
                key = self._next_key()
                if key in self._bad:
                    continue                      # key hỏng vĩnh viễn -> bỏ nhanh
                tried += 1
                try:
                    r = self.session.post(
                        self.base,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        data=json.dumps(body), timeout=self.timeout,
                    )
                    if r.status_code == 200:
                        j = r.json()
                        choices = j.get("choices")
                        if choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content") or ""
                            if not content and msg.get("reasoning"):
                                content = msg["reasoning"]
                            if content:
                                u = j.get("usage") or {}
                                return content, {"prompt_tokens": u.get("prompt_tokens"),
                                                 "completion_tokens": u.get("completion_tokens")}
                        last = f"200-empty: {str(j)[:120]}"   # 200 nhưng rỗng -> xoay key
                        continue
                    if r.status_code in (401, 403):
                        with self._lock:
                            self._bad.add(key)                # key sai -> loại vĩnh viễn
                        last = f"{r.status_code}: {r.text[:80]}"
                        continue
                    # 429/402/5xx hoặc khác -> xoay key (không sleep)
                    last = f"{r.status_code}: {r.text[:120]}"
                    continue
                except requests.RequestException as ex:
                    last = str(ex)[:150]
                    continue
            if tried == 0:
                break                              # tất cả key đều hỏng
            time.sleep(min(3.0, 0.5 * (round_i + 1)))   # hết 1 vòng -> nghỉ ngắn rồi thử lại
        raise RuntimeError(f"LLM hết retry ({self.retry} vòng): {last}")


# ── LLM client (Google AI Studio / Gemini API) — Gemma qua generateContent ───
class GoogleLLMClient:
    """
    Gọi Google Generative Language API (AI Studio). Model Gemma tự "thinking"
    (parts có thought=true) -> lấy phần answer (không thought).
    Xoay vòng nhiều key nếu có + retry/backoff. Thread-safe cho eval song song.
    Interface .chat(messages) giống LLMClient (nhận [{'role','content'}]).
    """

    def __init__(self, cfg, logger=None):
        import threading
        c = cfg["llm"]["google"]
        self.base = c["base_url"].rstrip("/")
        self.model = c["model"].split("/")[-1]
        self.temperature = c.get("temperature", 0.0)
        self.max_out = c.get("max_output_tokens", 2048)
        self.timeout = c.get("timeout", 120)
        self.retry = c.get("retry", 5)
        self.log = logger or logging.getLogger("gllm")
        kp = c["keys_file"]
        kp = os.path.join(ROOT, kp) if not os.path.isabs(kp) else kp
        with open(kp) as f:
            self.keys = [ln.strip() for ln in f if ln.strip()]
        if not self.keys:
            raise RuntimeError(f"Không có key trong {kp}")
        self._lock = threading.Lock()
        self._rr = 0
        self.session = requests.Session()

    def _next_key(self):
        with self._lock:
            k = self.keys[self._rr % len(self.keys)]
            self._rr += 1
            return k

    @staticmethod
    def _to_contents(messages):
        # Gemma không nhận systemInstruction -> gộp system vào user, nối thành 1 lượt user.
        sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        usr_txt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        text = (sys_txt + "\n\n" + usr_txt).strip() if sys_txt else usr_txt
        return [{"role": "user", "parts": [{"text": text}]}]

    @staticmethod
    def _extract(j):
        cands = j.get("candidates") or []
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", []) or []
        ans = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not ans:                       # fallback: gộp cả thought
            ans = "".join(p.get("text", "") for p in parts)
        return ans

    def chat(self, messages):
        return self.chat_usage(messages)[0]

    def chat_usage(self, messages):
        """Như chat() nhưng trả (content, usage) từ usageMetadata của API (hoặc None)."""
        body = {
            "contents": self._to_contents(messages),
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": self.max_out},
        }
        url = f"{self.base}/models/{self.model}:generateContent"
        last = None
        n = len(self.keys)
        for round_i in range(self.retry):
            for _ in range(n):
                key = self._next_key()
                try:
                    r = self.session.post(url, params={"key": key}, json=body,
                                          timeout=self.timeout)
                    if r.status_code == 200:
                        j = r.json()
                        txt = self._extract(j)
                        if txt:
                            u = j.get("usageMetadata") or {}
                            return txt, {"prompt_tokens": u.get("promptTokenCount"),
                                         "completion_tokens": u.get("candidatesTokenCount")}
                        last = "200-empty"
                        continue
                    # 429 rate limit / 5xx -> xoay key / vòng
                    last = f"{r.status_code}: {r.text[:120]}"
                    if r.status_code in (400, 403, 404):   # key/model sai -> vẫn thử key khác
                        continue
                    continue
                except requests.RequestException as ex:
                    last = str(ex)[:150]
                    continue
            time.sleep(min(5.0, 1.0 * (round_i + 1)))
        raise RuntimeError(f"Google LLM hết retry ({self.retry} vòng): {last}")


# ── LLM client (Ollama local) — không key, không rate-limit ──────────────────
class OllamaLLMClient:
    """
    Gọi Ollama /api/chat (local). Model có thinking -> thinking nằm ở
    message.thinking, ĐÁP ÁN sạch ở message.content (không cắt cụt, parse dễ).
    """

    def __init__(self, cfg, logger=None):
        c = cfg["llm"]["ollama"]
        self.base = c["base_url"].rstrip("/")
        self.model = c["model"]
        self.think = c.get("think", True)
        self.temperature = c.get("temperature", 0.0)
        self.timeout = c.get("timeout", 120)
        self.retry = c.get("retry", 3)
        self.keep_alive = c.get("keep_alive", "30m")   # giữ model nạp -> tránh 404 model-not-found
        # num_ctx phải đủ chứa prompt (5 chunk RAG ~3200 tok) + output; mặc định
        # Ollama 4096 -> tràn context, model bị cắt trước khi ra đáp án (content rỗng).
        self.num_ctx = c.get("num_ctx", 8192)
        self.num_predict = c.get("num_predict", 1024)
        self.log = logger or logging.getLogger("ollama")
        self.session = requests.Session()

    def chat(self, messages):
        return self.chat_usage(messages)[0]

    def chat_usage(self, messages):
        """Như chat() nhưng trả (content, usage). Ollama trả token thật ở
        prompt_eval_count (input) / eval_count (output) ngay trong response."""
        # Thử think=self.think trước; nếu trả rỗng -> ESCALATE think=False
        # (think=False buộc model trả đáp án trực tiếp ở content, luôn parse được).
        think_modes = [self.think, False] if self.think else [False]
        last = None
        for think in think_modes:
            body = {
                "model": self.model, "messages": messages, "stream": False,
                "think": think, "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature,
                            "num_ctx": self.num_ctx, "num_predict": self.num_predict},
            }
            for attempt in range(self.retry):
                try:
                    r = self.session.post(self.base + "/api/chat", json=body, timeout=self.timeout)
                    if r.status_code == 200:
                        j = r.json()
                        msg = j.get("message", {})
                        content = (msg.get("content") or "").strip()
                        thinking = (msg.get("thinking") or "").strip()
                        # đáp án có thể nằm trong thinking (content rỗng). Ghép thinking TRƯỚC,
                        # content SAU -> parse_choice lấy "Answer:" cuối (ưu tiên content).
                        combined = (thinking + "\n\n" + content).strip() if thinking else content
                        if combined.strip():
                            return combined, {"prompt_tokens": j.get("prompt_eval_count"),
                                              "completion_tokens": j.get("eval_count")}
                        last = "content+thinking rỗng"
                    else:
                        last = f"{r.status_code}: {r.text[:150]}"
                except requests.RequestException as ex:
                    last = str(ex)[:150]
                # backoff dài hơn: 404 model-not-found do evict/reload cần thời gian nạp lại
                time.sleep(min(12.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"Ollama hết retry: {last}")


def make_llm(cfg, logger=None):
    """Factory chọn backend LLM theo cfg['llm']['provider'] (ollama | google | openrouter)."""
    provider = cfg["llm"].get("provider", "openrouter")
    if provider == "ollama":
        return OllamaLLMClient(cfg, logger)
    if provider == "google":
        return GoogleLLMClient(cfg, logger)
    return LLMClient(cfg, logger)


def get_qdrant(cfg):
    from qdrant_client import QdrantClient
    q = cfg["qdrant"]
    return QdrantClient(host=q["host"], port=q["port"], timeout=120)


def get_es(cfg):
    from elasticsearch import Elasticsearch
    return Elasticsearch(cfg["elasticsearch"]["host"], request_timeout=120)
