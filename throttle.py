"""
throttle.py — Điều tiết concurrency LLM theo tỉ lệ NGHẼN gần đây (adaptive).

Vấn đề: Ollama shared dễ nghẽn (504/timeout) khi nhiều call đồng thời. Cơ chế:
  - Theo dõi outcome mỗi call: NGHẼN = call fail (hết retry) HOẶC latency > ngưỡng.
  - Tỉ lệ nghẽn cao -> hạ concurrency về `min` (thường 1, gửi 1 câu 1 lúc).
  - Tỉ lệ nghẽn thấp -> tăng dần tới `max` (thường 3).

`AdaptiveGate` = cổng đếm concurrency động (thread-safe). `ThrottledLLM` = bọc LLM client,
acquire trước call / release sau call — TRONG SUỐT với agents.py (không đổi logic agent).
"""
import time
import threading
from collections import deque


def _provider_name(llm):
    n = type(llm).__name__.lower()
    if "ollama" in n:
        return "ollama"
    if "google" in n:
        return "google"
    if "llmclient" in n:
        return "openrouter"
    return None


class AdaptiveGate:
    """Giới hạn số call ĐỒNG THỜI = `limit`, tự điều chỉnh theo tỉ lệ nghẽn cửa sổ gần nhất."""

    def __init__(self, max_conc=3, min_conc=1, window=10, high=0.3, low=0.1,
                 latency_threshold=60.0, on_change=None):
        self.max = max_conc
        self.min = min_conc
        self.limit = max_conc
        self.active = 0
        self.window = window
        self.high = high                       # tỉ lệ nghẽn -> hạ về min
        self.low = low                         # tỉ lệ nghẽn -> tăng dần
        self.latency_threshold = latency_threshold
        self.on_change = on_change
        self._recent = deque(maxlen=window)    # 1=nghẽn, 0=ok
        self._cv = threading.Condition()

    def acquire(self):
        with self._cv:
            while self.active >= self.limit:
                self._cv.wait()
            self.active += 1

    def release(self, congested):
        with self._cv:
            self.active = max(0, self.active - 1)
            self._recent.append(1 if congested else 0)
            self._maybe_adjust()
            self._cv.notify_all()

    def _maybe_adjust(self):
        # cần đủ mẫu mới đánh giá (tránh dao động vì 1-2 call)
        if len(self._recent) < max(3, self.window // 2):
            return
        rate = sum(self._recent) / len(self._recent)
        old = self.limit
        if rate >= self.high and self.limit > self.min:
            self.limit -= 1                    # nghẽn cao -> tụt TỪNG BẬC (đỡ dao động, tránh nhảy về 1 rồi lại lên)
        elif rate <= self.low and self.limit < self.max:
            self.limit += 1                    # nghẽn thấp -> tăng 1 bậc
        if self.limit != old:
            self._recent.clear()               # reset để đánh giá lại ở mức mới
            if self.on_change:
                try:
                    self.on_change(old, self.limit, rate)
                except Exception:
                    pass

    def status(self):
        with self._cv:
            rate = (sum(self._recent) / len(self._recent)) if self._recent else 0.0
            return {"limit": self.limit, "active": self.active, "congestion": rate}


class ThrottledLLM:
    """Bọc LLM client, điều tiết concurrency qua AdaptiveGate. Trong suốt: giữ .model/.provider,
    expose .chat/.chat_usage như client gốc để agents.py dùng y nguyên."""

    def __init__(self, llm, gate):
        self._llm = llm
        self._gate = gate
        self.model = getattr(llm, "model", None)
        self.provider = _provider_name(llm)

    def chat(self, messages):
        return self.chat_usage(messages)[0]

    def chat_usage(self, messages):
        self._gate.acquire()
        t0 = time.time()
        congested = False
        try:
            if hasattr(self._llm, "chat_usage"):
                return self._llm.chat_usage(messages)
            return self._llm.chat(messages), None
        except Exception:
            congested = True                    # fail sau khi hết retry = nghẽn nặng
            raise
        finally:
            slow = (time.time() - t0) > self._gate.latency_threshold
            self._gate.release(congested or slow)
