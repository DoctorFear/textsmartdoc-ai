# Chunk params, search type UI → 8.2.4, 8.2.7
# ui/settings_panel.py
# ── Panel "Thiết lập & Tùy chọn" trong sidebar (8.2.4) ───────────────────────
# Tất cả giá trị mặc định lấy từ src/config.py, không hardcode ở đây.
from datetime import datetime
import streamlit as st
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K, FETCH_K, LAMBDA_MULT,
    SEARCH_TYPE, RETRIEVAL_MODE,
    USE_RERANKER, SELF_RAG_ENABLED,
    EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
)


def render_model_info():
    """Expander 'Cấu hình mô hình' — chỉ hiển thị, không widget."""
    with st.expander("Cấu hình mô hình"):
        st.markdown(f"""
        • **Embedding**: {EMBEDDING_MODEL}  
        • **LLM**: {LLM_MODEL}  
        • **Temperature**: {LLM_TEMPERATURE}  
        • **Ngày chạy**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """)


def render_settings_panel() -> dict:
    """
    Render expander 'Thiết lập & Tùy chọn'.
    Mọi default đều lấy từ src/config.py.

    Returns:
        dict settings với keys:
            chunk_size, chunk_overlap,
            top_k, fetch_k, search_type, lambda_mult,
            retrieval_mode, use_reranker
        (self_rag_enabled được ghi thẳng vào session_state)
    """
    settings = {}

    with st.expander("Thiết lập & Tùy chọn"):
        settings["chunk_size"]    = st.slider("Chunk Size",    200,  2000, CHUNK_SIZE,   100)
        settings["chunk_overlap"] = st.slider("Chunk Overlap",   0,   500, CHUNK_OVERLAP, 50)
        settings["top_k"]         = st.slider("Top-k",           1,    10, TOP_K,          1)
        settings["fetch_k"]       = st.slider(
            "Fetch-k",
            min_value=settings["top_k"],
            max_value=100,
            value=FETCH_K,
            step=5
        )
        settings["search_type"] = st.selectbox(
            "Search Type",
            ["similarity", "mmr"],
            index=0 if SEARCH_TYPE == "similarity" else 1
        )

        # Chỉ hiện lambda khi chọn MMR
        if settings["search_type"] == "mmr":
            settings["lambda_mult"] = st.slider(
                "Lambda Mult (Diversity)", 0.0, 1.0, LAMBDA_MULT, 0.05,
                help="0.7 là giá trị cân bằng tốt nhất cho tiếng Việt"
            )
        else:
            settings["lambda_mult"] = LAMBDA_MULT  # giá trị mặc định, không dùng

        settings["retrieval_mode"] = st.selectbox(
            "Retrieval Mode",
            ["faiss", "bm25", "hybrid"],
            index=["faiss", "bm25", "hybrid"].index(RETRIEVAL_MODE)
        )
        settings["use_reranker"] = st.checkbox("Use Cross-Encoder Reranking", value=USE_RERANKER)

        # Toggle Self-RAG (8.2.10)
        st.divider()
        st.session_state.self_rag_enabled = st.checkbox(
            "Bật Self-RAG",
            value=st.session_state.self_rag_enabled,
            help="Self-RAG cho phép tự đánh giá và cải thiện câu trả lời. Chậm hơn nhưng chính xác hơn."
        )

    return settings