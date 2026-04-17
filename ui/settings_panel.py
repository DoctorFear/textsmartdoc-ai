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

from src.logger import setup_logger
logger = setup_logger()



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

        settings["ocr_enabled"] = st.toggle(
            "OCR Mode", 
            value=False
        )

        if settings["ocr_enabled"]:
            settings["ocr_mode"] = st.selectbox(
                "OCR Quality",
                ["fast", "balanced", "accurate"],
                index=1
            )

            # mapping preset
            if settings["ocr_mode"] == "fast":
                settings["dpi"] = 150
                settings["batch_size"] = 4
                settings["confidence_threshold"] = 0.2

            elif settings["ocr_mode"] == "balanced":
                settings["dpi"] = 220
                settings["batch_size"] = 8
                settings["confidence_threshold"] = 0.3

            else:
                settings["dpi"] = 300
                settings["batch_size"] = 8
                settings["confidence_threshold"] = 0.4

            # 🔥 HIỆN LUÔN
            settings["dpi"] = st.slider("DPI", 100, 400, settings["dpi"], 10)
            settings["batch_size"] = st.slider("Batch Size", 1, 32, settings["batch_size"])
            settings["confidence_threshold"] = st.slider(
                "Confidence Threshold", 0.0, 1.0, settings["confidence_threshold"], 0.05
            )
        
        settings["search_type"] = st.selectbox(
            "Search Type",
            ["similarity", "mmr"],
            index=0 if SEARCH_TYPE == "similarity" else 1,
        )

        # ================== FETCH-K & LAMBDA MULT (Chỉ hiển thị khi MMR) ==================
        if settings["search_type"] == "mmr":
            settings["fetch_k"] = st.slider(
                "Fetch-k",
                min_value=settings["top_k"],
                max_value=100,
                value=FETCH_K,
                step=5,
            )
            
            settings["lambda_mult"] = st.slider(
                "Lambda Mult",
                0.0, 1.0, LAMBDA_MULT, 0.05,
            )
        else:
            # Similarity không dùng Fetch-k và Lambda Mult
            settings["fetch_k"] = FETCH_K
            settings["lambda_mult"] = LAMBDA_MULT

        # ================== SELF-RAG (Đặt trước để kiểm soát Retrieval Mode) ==================
        settings["self_rag_method"] = st.selectbox(
            "Self-RAG Mode",
            ["Off | Nomal RAG", "On | Self-RAG"],
            index=0,
        )

        settings["self_rag_enabled"] = (settings["self_rag_method"] == "On | Self-RAG")

        # ================== COMBINED MODE ==================
        if settings["self_rag_enabled"]:
            # Tự động chuyển về Chỉ RAG khi bật Self-RAG
            settings["combined_mode"] = "rag"
            

        else:
            settings["combined_mode"] = st.selectbox(
                "Combined Mode",
                ["rag", "corag", "rag+corag"],
                index=0,
                format_func=lambda x: {
                    "rag": "RAG",
                    "corag": "CoRAG",
                    "rag+corag": "RAG & CoRAG"
                }[x]
            )

        # ================== RETRIEVAL MODE ==================
        if settings["self_rag_enabled"]:
            settings["retrieval_mode"] = "hybrid"
        else:
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

        # ================== HYBRID WEIGHT SLIDER ==================
        if settings["retrieval_mode"] == "hybrid":
            st.markdown("**Điều chỉnh tỷ lệ Hybrid Search:**")
            
            hybrid_ratio = st.slider(
                "Tỷ lệ FAISS (Semantic) / BM25",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="hybrid_weight_slider",
            )
            
            faiss_weight = round(hybrid_ratio, 2)
            bm25_weight = round(1.0 - hybrid_ratio, 2)
            
            settings["bm25_weight"] = bm25_weight
            settings["faiss_weight"] = faiss_weight

            st.markdown(
                f"""
                <div style="background-color: rgb(255 255 255); border: 1px solid rgb(199, 210, 254); 
                border-radius: 8px; padding: 10px 0 10px 80px; font-size: 0.95rem; font-weight: 500; 
                color: #3F51B5; line-height: 1.6; margin-bottom: 20px;">
                    <b>FAISS:</b> {faiss_weight} &nbsp;&nbsp; | &nbsp;&nbsp; <b>BM25:</b> {bm25_weight}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            settings["bm25_weight"] = BM25_WEIGHT
            settings["faiss_weight"] = FAISS_WEIGHT

        # ================== RERANKING ==================
        if settings["self_rag_enabled"]:
            settings["use_reranker"] = False
            settings["reranking_method"] = "Off (Self-RAG)"
            
            # Chỉ ghi log khi Self-RAG vừa được bật (tránh duplicate)
            if not st.session_state.get("prev_self_rag_state", False):
                logger.info("[Settings] Self-RAG ON → Cross-Encoder Reranker tự động tắt")
        else:
            settings["reranking_method"] = st.selectbox(
                "Reranking",
                ["Off | Bi-encoder", "On | Cross-Encoder"],
                index=0,
            )
            
            settings["use_reranker"] = (settings["reranking_method"] == "On | Cross-Encoder")

        # ================== Rerun khi thay đổi Self-RAG ==================
        if "prev_self_rag_state" not in st.session_state:
            st.session_state.prev_self_rag_state = False

        if settings["self_rag_enabled"] != st.session_state.prev_self_rag_state:
            st.session_state.prev_self_rag_state = settings["self_rag_enabled"]
            st.rerun()

        st.divider()

    return settings