"""
tracking.py — GenAI Tracing (MLflow 3.x) cho luồng multi-agent. Bật/tắt qua config `mlflow.enabled`.

KHÔNG dùng "run" kiểu ML-training (params/metrics/artifact) nữa — MỌI telemetry nằm trong
trace: token qua chuẩn native `mlflow.chat.tokenUsage` (MLflow tự tổng lên cấp trace), còn
biến/kết quả (model, provider, gold, pred, correct, split…) là TAG của trace -> lọc/nhóm/tính
accuracy ngay trong tab Traces.

Nguyên tắc: KHÔNG bao giờ làm eval chết vì tracking. Nếu import lỗi / server không sống,
mọi thao tác thành no-op và eval chạy tiếp (chỉ log warning).

Dùng:
    tracker = Tracker(cfg)                      # chỉ nối MLflow + set experiment, KHÔNG tạo run
    with span(tracker, "multiagent", root=True) as root:
        set_trace_tags(tracker, {"variant": "v3", "model": ...})
        with span(tracker, "reasoning") as sp:
            sp.set_token_usage(tokens_in, tokens_out)   # -> mlflow.chat.tokenUsage (tự tổng)
    tag_trace(tracker, root.trace_id, {"gold": "D", "pred": "D", "correct": True})   # sau khi xong
"""
import sys
import logging
from contextlib import contextmanager

log = logging.getLogger("tracking")


class Tracker:
    """Chỉ giữ handle MLflow + experiment cho GenAI Tracing. KHÔNG tạo run."""

    def __init__(self, cfg):
        self.mlflow = None
        mc = (cfg or {}).get("mlflow", {}) or {}
        if not mc.get("enabled"):
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(mc["tracking_uri"])
            # set_experiment gọi REST -> nếu server không sống sẽ raise ở đây -> tắt tracking.
            mlflow.set_experiment(mc.get("experiment", "medqa-v3"))
            self.mlflow = mlflow
            log.info("MLflow Tracing ON: %s (experiment=%s)", mc["tracking_uri"],
                     mc.get("experiment", "medqa-v3"))
        except Exception as e:
            log.warning("MLflow TẮT (không kết nối được %s): %s",
                        mc.get("tracking_uri"), str(e)[:150])


# ── GenAI Tracing (MLflow 3.x) ──────────────────────────────────────────────
# Mỗi câu = 1 trace: root span (chain) -> con: retrieve / reasoning / verifier / aggregator.

class _Span:
    """Proxy an toàn cho LiveSpan — mọi set_* là no-op nếu span gốc None hoặc lỗi."""
    __slots__ = ("_sp",)

    def __init__(self, sp):
        self._sp = sp

    @property
    def trace_id(self):
        return getattr(self._sp, "trace_id", None) if self._sp is not None else None

    def _safe(self, fn):
        if self._sp is None:
            return
        try:
            fn(self._sp)
        except Exception:
            pass

    def set_inputs(self, x):
        self._safe(lambda s: s.set_inputs(x))

    def set_outputs(self, x):
        self._safe(lambda s: s.set_outputs(x))

    def set_attributes(self, d):
        d = {k: v for k, v in d.items() if v is not None}   # MLflow không nhận None
        if d:
            self._safe(lambda s: s.set_attributes(d))

    def set_token_usage(self, tokens_in, tokens_out):
        """Ghi token 2 dạng, không bỏ sót:
        - chuẩn native `mlflow.chat.tokenUsage` -> MLflow TỰ TỔNG lên trace + cột Tokens ở tab Traces;
        - attribute PHẲNG `tokens_in`/`tokens_out`/`tokens_total` -> SỐ token hiện ngay trong bảng
          attribute của từng agent span (cạnh `tokens_src`).
        Bỏ qua nếu không có tokens_in."""
        if self._sp is None or tokens_in is None:
            return
        try:
            from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
            ti, to = int(tokens_in), int(tokens_out or 0)
            self._sp.set_attribute(SpanAttributeKey.CHAT_USAGE, {
                TokenUsageKey.INPUT_TOKENS: ti,
                TokenUsageKey.OUTPUT_TOKENS: to,
                TokenUsageKey.TOTAL_TOKENS: ti + to,
            })
            self._sp.set_attributes({"tokens_in": ti, "tokens_out": to, "tokens_total": ti + to})
        except Exception:
            pass

    def set_model(self, model, provider=None):
        """Ghi model/provider theo chuẩn native (`mlflow.llm.model` / `mlflow.llm.provider`)."""
        if self._sp is None:
            return
        try:
            from mlflow.tracing.constant import SpanAttributeKey
            if model is not None:
                self._sp.set_attribute(SpanAttributeKey.MODEL, model)
            if provider is not None:
                self._sp.set_attribute(SpanAttributeKey.MODEL_PROVIDER, provider)
        except Exception:
            pass


@contextmanager
def span(tracker, name, span_type="LLM", inputs=None, attributes=None, root=False):
    """
    Tạo 1 MLflow span (GenAI Tracing). No-op nếu tracker None / MLflow tắt / start_span lỗi.
    KHÔNG nuốt exception của thân lệnh — span vẫn được đóng qua finally.
    (`root` giữ để đánh dấu span gốc; không còn link run vì đã bỏ run.)
    """
    mlf = getattr(tracker, "mlflow", None) if tracker is not None else None
    if mlf is None:
        yield _Span(None)
        return
    cm = None
    try:
        cm = mlf.start_span(name=name, span_type=span_type)
        raw = cm.__enter__()
    except Exception as e:
        log.warning("MLflow start_span lỗi -> bỏ qua trace: %s", str(e)[:150])
        if cm is not None:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:
                pass
        yield _Span(None)
        return
    sp = _Span(raw)
    if inputs is not None:
        sp.set_inputs(inputs)
    if attributes:
        sp.set_attributes(attributes)
    exc = (None, None, None)
    try:
        yield sp
    except BaseException:
        exc = sys.exc_info()
        raise
    finally:
        try:
            cm.__exit__(*exc)
        except Exception:
            pass


def set_trace_tags(tracker, tags):
    """Gắn tag cho trace ĐANG chạy (gọi bên trong span). Tag = biến để lọc/nhóm ở tab Traces."""
    mlf = getattr(tracker, "mlflow", None) if tracker is not None else None
    if mlf is None:
        return
    try:
        mlf.update_current_trace(tags={k: str(v) for k, v in tags.items() if v is not None})
    except Exception as e:
        log.warning("MLflow update_current_trace lỗi: %s", str(e)[:150])


def tag_trace(tracker, trace_id, tags):
    """Gắn tag cho 1 trace ĐÃ kết thúc (dùng ở eval để ghi gold/pred/correct sau khi biết kết quả)."""
    mlf = getattr(tracker, "mlflow", None) if tracker is not None else None
    if mlf is None or not trace_id:
        return
    try:
        from mlflow import MlflowClient
        c = MlflowClient()
        for k, v in tags.items():
            if v is not None:
                c.set_trace_tag(trace_id, k, str(v))
    except Exception as e:
        log.warning("MLflow set_trace_tag lỗi: %s", str(e)[:150])
