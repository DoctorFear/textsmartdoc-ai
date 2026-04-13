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
    BM25_WEIGHT, FAISS_WEIGHT,
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

        # ================== SELF-RAG (Đặt trước để kiểm soát Retrieval Mode) ==================
        settings["self_rag_method"] = st.selectbox(
            "Chế độ Self-RAG",
            ["Tắt (Normal RAG)", "Bật Self-RAG (Tự đánh giá)"],
            index=0,
            help="Self-RAG sẽ tự viết lại câu hỏi và đánh giá chất lượng → chính xác hơn nhưng chậm hơn."
        )
        
        settings["self_rag_enabled"] = (settings["self_rag_method"] == "Bật Self-RAG (Tự đánh giá)")

        # ================== RETRIEVAL MODE ==================
        if settings["self_rag_enabled"]:
            # Tự động chuyển sang Hybrid khi bật Self-RAG, nhưng vẫn cho chỉnh trọng số
            settings["retrieval_mode"] = "hybrid"
        else:
            # Bình thường cho phép chọn chế độ
            retrieval_options = ["faiss", "hybrid", "bm25"]
            retrieval_labels = [
                "FAISS (Vector Search)", 
                "Hybrid (FAISS + BM25)", 
                "BM25 (Keyword Search)"
            ]
            
            selected_label = st.selectbox(
                "Retrieval Mode",
                retrieval_labels,
                index=retrieval_options.index(RETRIEVAL_MODE),
                key="retrieval_mode_select"
            )
            settings["retrieval_mode"] = retrieval_options[retrieval_labels.index(selected_label)]

        # ================== SLIDER ĐIỀU CHỈNH TỶ LỆ HYBRID ==================
        # Luôn hiển thị slider khi là Hybrid (dù Self-RAG bật hay tắt)
        if settings["retrieval_mode"] == "hybrid":
            st.markdown("**Điều chỉnh tỷ lệ Hybrid Search:**")
            
            hybrid_ratio = st.slider(
                "Tỷ lệ FAISS (Semantic) / BM25",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="hybrid_weight_slider",
                help="0.0 = Ưu tiên BM25 (từ khóa) | 1.0 = Ưu tiên FAISS (ý nghĩa)"
            )
            
            faiss_weight = round(hybrid_ratio, 2)
            bm25_weight = round(1.0 - hybrid_ratio, 2)
            
            settings["bm25_weight"] = bm25_weight
            settings["faiss_weight"] = faiss_weight

            st.markdown(
                f"""
                <div style="
                background-color: rgb(255 255 255);
                border: 1px solid rgb(199, 210, 254);
                border-radius: 8px;
                padding: 10px 0 10px 80px;
                font-size: 0.95rem;
                font-weight: 500;
                color: #3F51B5;
                line-height: 1.6;
                align-items: normal;
                margin-bottom: 20px;
                ">
                    <b>FAISS:</b> {faiss_weight} &nbsp;&nbsp; | &nbsp;&nbsp; <b>BM25:</b> {bm25_weight}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Không phải Hybrid thì dùng giá trị mặc định
            settings["bm25_weight"] = BM25_WEIGHT
            settings["faiss_weight"] = FAISS_WEIGHT

        # ================== RERANKING ==================
        settings["reranking_method"] = st.selectbox(
            "Reranking",
            ["Bi-encoder (Nhanh)", "Cross-Encoder (Chính xác hơn)"],
            index=0,
            help="Bi-encoder: Nhanh, phù hợp sử dụng thông thường.\nCross-Encoder: Độ chính xác cao hơn nhưng chậm hơn."
        )
        
        settings["use_reranker"] = (settings["reranking_method"] == "Cross-Encoder (Chính xác hơn)")

        # Rerun để cập nhật giao diện khi bật/tắt Self-RAG
        if settings["self_rag_enabled"] and st.session_state.get("prev_self_rag_state") != True:
            st.session_state.prev_self_rag_state = True
            st.rerun()
        elif not settings["self_rag_enabled"]:
            st.session_state.prev_self_rag_state = False

        st.divider()

    return settings