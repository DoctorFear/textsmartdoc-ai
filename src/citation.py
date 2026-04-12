# Source tracking, page number, highlight → 8.2.5

# src/citation.py
# ── Citation / Source tracking (8.2.5) ───────────────────────────────────────
# Tách toàn bộ logic build + render citations ra khỏi chat_panel.
import os


def build_citations(retrieved_docs: list) -> list[dict]:
    """
    Tạo danh sách citation từ các docs đã retrieve.
    Mỗi citation gồm: index, page, file, snippet.

    Args:
        retrieved_docs: list Document từ retriever.invoke()
    Returns:
        list[dict] với keys: index, page, file, snippet
    """
    citations = []
    for i, doc in enumerate(retrieved_docs):
        page_num = doc.metadata.get("page", -1) + 1
        raw_source = doc.metadata.get("source", "Unknown")
        file_name = (
            os.path.basename(raw_source)
            if raw_source != "Unknown"
            else "Unknown document"
        )
        snippet = doc.page_content[:250].replace("\n", " ") + "..."

        citations.append({
            "index":   i + 1,
            "page":    page_num if page_num > 0 else "N/A",
            "file":    file_name,
            "snippet": snippet,
        })

    return citations