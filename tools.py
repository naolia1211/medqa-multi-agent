"""
tools.py — Định nghĩa TOOL cho agent (function-calling) + bộ thực thi.

Biến retrieval từ BƯỚC CỐ ĐỊNH ([A] gọi cứng trong code) thành CÔNG CỤ mà agent TỰ QUYẾT ĐỊNH
gọi khi cần — nền tảng cho luồng agentic thật (không fixed). Tham số ở config,
không hardcode; token/ngưỡng lấy từ cfg.
"""
from retrieval import retrieve_rerank

# Spec tool theo chuẩn OpenAI/Ollama function-calling (dùng chung cho mọi backend).
TOOL_SPECS = [{
    "type": "function",
    "function": {
        "name": "search_textbooks",
        "description": ("Tra cứu 18 sách giáo khoa y khoa (hybrid dense+sparse+MedCPT rerank) để lấy "
                        "bằng chứng cho một truy vấn. Gọi khi bằng chứng hiện có CHƯA ĐỦ hoặc cần khía "
                        "cạnh khác (cơ chế bệnh, tiêu chuẩn chẩn đoán, thuốc first-line...)."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Truy vấn y khoa cô đọng (triệu chứng, xét nghiệm, chẩn đoán, thuốc)."},
            },
            "required": ["query"],
        },
    },
}]


def execute_tool(name, args, cfg, clients):
    """
    Chạy 1 tool call. clients = (embed, qdrant, es, llm).
    Trả (result_obj_cho_LLM, evidence_list_mới). evidence_list để gộp vào state/trace.
    """
    if name == "search_textbooks":
        q = str((args or {}).get("query", "")).strip()
        if not q:
            return {"error": "thiếu tham số 'query'"}, []
        ec, qc, es = clients[0], clients[1], clients[2]
        ev = retrieve_rerank(cfg, q, embed_client=ec, qdrant=qc, es=es)
        # tóm tắt để đưa lại cho model (cắt text tránh phình context)
        brief = [{"chunk_id": e["chunk_id"], "source": e["source"], "text": e["text"][:400]}
                 for e in ev]
        return {"query": q, "n": len(ev), "chunks": brief}, ev
    return {"error": f"tool không tồn tại: {name}"}, []
