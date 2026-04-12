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
    settings = {}

    with st.expander("Thiết lập & Tùy chọn"):
        settings["chunk_size"]    = st.slider("Chunk Size",    200,  2000, CHUNK_SIZE,   100)
        settings["chunk_overlap"] = st.slider("Chunk Overlap",   0,   500, CHUNK_OVERLAP, 50)
        settings["top_k"]         = st.slider("Top-k",           1,    10, TOP_K,          1)
        settings["fetch_k"]       = st.slider("Fetch-k", min_value=settings["top_k"], max_value=100, value=FETCH_K, step=5)
        
        settings["search_type"] = st.selectbox(
            "Search Type",
            ["similarity", "mmr"],
            index=0 if SEARCH_TYPE == "similarity" else 1
        )

        if settings["search_type"] == "mmr":
            settings["lambda_mult"] = st.slider(
                "Lambda Mult (Diversity)", 0.0, 1.0, LAMBDA_MULT, 0.05,
                help="0.7 là giá trị cân bằng tốt nhất cho tiếng Việt"
            )
        else:
            settings["lambda_mult"] = LAMBDA_MULT

        # Retrieval Mode - Sửa label cho dễ hiểu
        retrieval_options = ["faiss", "hybrid", "bm25"]
        retrieval_labels = [
            "FAISS (Vector Search)", 
            "Hybrid (FAISS + BM25)", 
            "BM25 (Keyword Search)"
        ]
        
        selected_label = st.selectbox(
            "Retrieval Mode",
            retrieval_labels,
            index=retrieval_options.index(RETRIEVAL_MODE)
        )
        settings["retrieval_mode"] = retrieval_options[retrieval_labels.index(selected_label)]

        # Reranking - Selectbox như bạn muốn
        settings["reranking_method"] = st.selectbox(
            "Reranking",
            ["Bi-encoder (Nhanh)", "Cross-Encoder (Chính xác hơn)"],
            index=0,
            help="Bi-encoder: Nhanh, phù hợp sử dụng thông thường.\nCross-Encoder: Độ chính xác cao hơn nhưng chậm hơn."
        )
        
        # Chuyển thành boolean để code cũ dễ dùng
        settings["use_reranker"] = (settings["reranking_method"] == "Cross-Encoder (Chính xác hơn)")

        # Self-RAG
        settings["self_rag_method"] = st.selectbox(
            "Chế độ Self-RAG",
            ["Tắt (Normal RAG)", "Bật Self-RAG (Tự đánh giá)"],
            index=0,   # Mặc định là Tắt
            help="Self-RAG sẽ tự viết lại câu hỏi và đánh giá chất lượng câu trả lời → chính xác hơn nhưng chậm hơn."
        )
        
        settings["self_rag_enabled"] = (settings["self_rag_method"] == "Bật Self-RAG (Tự đánh giá)")

        st.divider()

    return settings