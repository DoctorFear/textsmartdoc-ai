# ui/chat_panel.py
import streamlit as st
import time

from src.rag_chain import get_language_instruction, format_docs, rag_chain, self_rag_query
from src.conversational import rewrite_with_history
from src.reranker import rerank
from src.hybrid_search import get_retriever
from src.citation import build_citations
from src.logger import setup_logger
from src.config import BM25_WEIGHT, FAISS_WEIGHT
from src.corag_pipeline import corag_pipeline, self_corag_query

logger = setup_logger()


# ── Helper render Self-RAG metadata ──────────────────────────────────────────
def render_self_rag_meta(meta: dict):
    if not meta:
        return

    attempts = meta.get("attempts", 1)
    confidence = meta.get("confidence", "?")
    query_used = meta.get("query_used", "")
    reason = meta.get("evaluation", {}).get("reason", "")
    steps = meta.get("multi_hop_steps")

    blocks = [
        f"<div><strong>Số lần thử:</strong> {attempts} | <strong>Độ tin cậy:</strong> {confidence}/10</div>"
    ]

    if steps and isinstance(steps, list) and len(steps) > 0:
        steps_html = "<br>".join([f"• {step}" for step in steps])
        blocks.append(f"<div><strong>Multi-hop reasoning steps:</strong><br>{steps_html}</div>")
    elif attempts > 1 and query_used:
        blocks.append(f"<div><strong>Câu hỏi được viết lại:</strong> <em>{query_used}</em></div>")

    if reason:
        blocks.append(f"<div><strong>Lý do đánh giá:</strong> {reason}</div>")

    st.markdown(f"<div class='self-rag-meta'>{''.join(blocks)}</div>", unsafe_allow_html=True)


# ── Render lịch sử chat ───────────────────────────────────────────────────────
def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            mode = message.get("mode")

            # --- TRƯỜNG HỢP 1: CHẾ ĐỘ SO SÁNH RAG + CORAG ---
            if mode == "rag+corag" and message.get("compare_data"):
                st.caption("Chế độ: RAG + CoRAG")
                data = message["compare_data"]
                
                # 1. Hiển thị nội dung RAG
                st.subheader("RAG")
                st.markdown(data["rag"]["response"])
                if data["rag"].get("citations"):
                    # Đưa expander ra sát lề, ngang hàng với subheader
                    with st.expander("Xem nguồn trích dẫn (RAG)"):
                        for cite in data["rag"]["citations"]:
                            st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                            st.info(f'"{cite["snippet"]}"')

                st.divider() # Đường kẻ phân cách để không bị rối khi nhìn dọc

                # 2. Hiển thị nội dung CoRAG
                st.subheader("CoRAG")
                st.markdown(data["corag"]["response"])
                if data["corag"].get("citations"):
                    # Đưa expander ra sát lề, ngang hàng với subheader
                    with st.expander("Xem nguồn trích dẫn (CoRAG)"):
                        for cite in data["corag"]["citations"]:
                            st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                            st.info(f'"{cite["snippet"]}"')

            # --- TRƯỜNG HỢP 2: CHẾ ĐỘ ĐƠN (RAG hoặc CoRAG bình thường) ---
            else:
                if mode:
                    mode_label = {"rag": "RAG", "corag": "CoRAG"}.get(mode, mode)
                    st.caption(f"Chế độ: {mode_label}")
                
                st.markdown(message.get("content", ""))
                data = message.get("compare_data")

                # Self-RAG Meta (nếu có)
                if message.get("self_rag_meta"):
                    if mode == "corag":
                        with st.expander("Thông tin Self-CORAG"):
                            render_self_rag_meta(message["self_rag_meta"])
                    else:
                        with st.expander("Thông tin Self-RAG"):
                            render_self_rag_meta(message["self_rag_meta"])

                # Citations cho chế độ đơn
                if message.get("citations"):
                    with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                        st.markdown("Hệ thống đã dựa vào các đoạn sau để trả lời:")
                        for cite in message["citations"]:
                            st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                            st.info(f'"{cite["snippet"]}"')


# ── Xử lý câu hỏi & sinh câu trả lời ────────────────────────────────────────
def handle_query(prompt, settings, save_current_session_fn, create_new_chat_session_fn):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
        st.stop()

    # Tạo session mới nếu cần
    if st.session_state.current_chat_id is None:
        create_new_chat_session_fn(
            prompt[:40] + ("..." if len(prompt) > 40 else ""),
            keep_current_context=True
        )
        save_current_session_fn()
        st.session_state.pending_prompt = prompt
        st.session_state.should_scroll = True
        st.rerun()

    # Append user message
    is_first_message = len(st.session_state.messages) == 0
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Cập nhật title
    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id and is_first_message:
            s["title"] = prompt[:48] + ("..." if len(prompt) > 48 else "")
            break

    # Unpack settings
    top_k = settings["top_k"]
    fetch_k = settings["fetch_k"]
    search_type = settings["search_type"]
    lambda_mult = settings["lambda_mult"]
    retrieval_mode = settings["retrieval_mode"]
    use_reranker = settings.get("use_reranker", False)
    self_rag_enabled = settings.get("self_rag_enabled", False)
    self_corag_enabled = settings.get("self_corag_enabled", False)
    pipeline_mode = settings.get("pipeline_mode", "rag")

    response = ""
    citations = []
    self_rag_meta = None
    mode = pipeline_mode

    spinner_text = "Đang tìm kiếm và suy nghĩ..."
    if self_rag_enabled:
        spinner_text = "Đang chạy Self-RAG..."
    elif use_reranker:
        spinner_text = "Đang rerank bằng Cross-Encoder..."
    elif pipeline_mode == "corag":
        spinner_text = "Đang chạy CoRAG..."
    elif pipeline_mode == "rag+corag":
        spinner_text = "Đang chạy RAG + CoRAG..."

    with st.spinner(spinner_text):
        try:
            logger.info(f"Mode: '{pipeline_mode}' | Query: '{prompt[:80]}...'")

            # Filter theo tài liệu
            source_filter = None
            if "source_filter_select" in st.session_state:
                selected = st.session_state.source_filter_select
                if selected != "Tất cả tài liệu":
                    source_filter = selected
            logger.info(f"Tìm kiếm trong tài liệu:{source_filter if source_filter else "Tất cả tài liệu"}")
            retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
            if source_filter:
                retriever_kwargs["filter"] = {"source": source_filter}
            if search_type == "mmr":
                retriever_kwargs["lambda_mult"] = lambda_mult

            all_docs = list(st.session_state.vectorstore.docstore._dict.values())

            # ================== RAG + CoRAG ==================
            if pipeline_mode == "rag+corag":
                bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                retriever, st.session_state = _update_bm25_cache(
                    st.session_state, all_docs, top_k, retrieval_mode, search_type,
                    retriever_kwargs, bm25_weight, faiss_weight
                )
                query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

                # RAG
                logger.info(f"[RAG+CoRAG] Đang chạy Bi-encoder")
                rag_docs = retriever.invoke(query_for_retrieval)
                if use_reranker:
                    logger.info(f"[RAG+CoRAG] Đang chạy Cross-Encoder Reranker")
                    rag_docs = rerank(query_for_retrieval, rag_docs, top_k=top_k)

                rag_response = "Không tìm thấy thông tin (RAG)." if not rag_docs else rag_chain.invoke({
                    "context": format_docs(rag_docs),
                    "question": prompt,
                    "language_instruction": get_language_instruction(prompt)
                }).strip()

                rag_citations = build_citations(rag_docs)

                # CoRAG
                corag_result = corag_pipeline(prompt, retriever, top_k=top_k)
                corag_response = corag_result["answer"]
                corag_citations = build_citations(corag_result["docs"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "mode": "rag+corag",
                    "compare_data": {
                        "rag": {"response": rag_response, "citations": rag_citations},
                        "corag": {"response": corag_response, "citations": corag_citations}
                    },
                    "content": None,
                    "citations": None,
                    "self_rag_meta": None
                })
                st.session_state.should_scroll = True
                save_current_session_fn()
                
                st.rerun()

            # ================== NORMAL MODES ==================
            else:
                if pipeline_mode == "rag":
                    if self_rag_enabled:
                        query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)
                        bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                        faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state, all_docs, top_k, "hybrid", search_type,
                            retriever_kwargs, bm25_weight, faiss_weight
                        )

                        result = self_rag_query(prompt, retriever, max_retries=2)
                        logger.info("[Self-RAG] Bắt đầu tiến trình tự đánh giá và sinh câu trả lời")
                        response = result["answer"]

                        self_rag_meta = {
                            "attempts": result["attempts"],
                            "confidence": result["confidence"],
                            "query_used": result["query_used"],
                            "evaluation": result["evaluation"],
                            "multi_hop_steps": result.get("multi_hop_steps")
                        }
                        retrieved_docs = retriever.invoke(query_for_retrieval)
                        response = "Không tìm thấy thông tin liên quan trong tài liệu." if not retrieved_docs else rag_chain.invoke({
                            "context": format_docs(retrieved_docs),
                            "question": prompt,
                            "language_instruction": get_language_instruction(prompt)
                        }).strip()

                        citations = build_citations(retrieved_docs)

                    else:
                        query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

                        bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                        faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state, all_docs, top_k, retrieval_mode, search_type,
                            retriever_kwargs, bm25_weight, faiss_weight
                        )
                        logger.info(f"[Normal RAG] Đang chạy Bi-encoder (Stage 1)")
                        retrieved_docs = retriever.invoke(query_for_retrieval)

                        if settings.get("use_reranker", False):
                            logger.info(f"[Normal RAG] On | Cross-Encoder Reranking")
                            retrieved_docs = rerank(query_for_retrieval, retrieved_docs, top_k=top_k)
                        else:
                            logger.info(f"[Normal RAG] Off | Bi-encoder only")

                        response = "Không tìm thấy thông tin liên quan trong tài liệu." if not retrieved_docs else rag_chain.invoke({
                            "context": format_docs(retrieved_docs),
                            "question": prompt,
                            "language_instruction": get_language_instruction(prompt)
                        }).strip()

                        citations = build_citations(retrieved_docs)
                        self_rag_meta = None

                elif pipeline_mode == "corag":
                    if self_corag_enabled:
                        query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)
                        bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                        faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state, all_docs, top_k, retrieval_mode, search_type,
                            retriever_kwargs, bm25_weight, faiss_weight
                        )
                        logger.info("[Self-CORAG] Bắt đầu tiến trình tự đánh giá và sinh câu trả lời")
                        result = self_corag_query(prompt, retriever, max_retries=2)
                        response = result["answer"]

                        self_rag_meta = {
                            "attempts": result["attempts"],
                            "confidence": result["confidence"],
                            "query_used": result["query_used"],
                            "evaluation": result["evaluation"],
                            "multi_hop_steps": result.get("multi_hop_steps")
                        }
                        retrieved_docs = retriever.invoke(query_for_retrieval)
                        response = "Không tìm thấy thông tin liên quan trong tài liệu." if not retrieved_docs else rag_chain.invoke({
                            "context": format_docs(retrieved_docs),
                            "question": prompt,
                            "language_instruction": get_language_instruction(prompt)
                        }).strip()

                        citations = build_citations(retrieved_docs)

                    else:
                        bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                        faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state, all_docs, top_k, retrieval_mode, search_type,
                            retriever_kwargs, bm25_weight, faiss_weight
                        )
                        
                        logger.info(f"[Normal CORAG] On | Cross-Encoder Reranking")
                        result = corag_pipeline(prompt, retriever, top_k=top_k)
                        response = result["answer"]
                        citations = build_citations(result["docs"])
                        self_rag_meta = None

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "citations": citations,
                    "self_rag_meta": self_rag_meta,
                    "mode": pipeline_mode
                })
                st.session_state.should_scroll = True
                save_current_session_fn()
                
                st.rerun()

        except Exception as e:
            error_str = str(e).lower()

            if any(keyword in error_str for keyword in ["connection", "refused", "ollama", "timeout"]):
                error_msg = "❌ Không thể kết nối đến Ollama. Vui lòng kiểm tra Ollama đang chạy chưa."
                logger.error(f"Ollama connection error: {e}")
            elif "model" in error_str and "not found" in error_str:
                error_msg = "❌ Model không tồn tại. Vui lòng kiểm tra tên model trong config."
            elif any(keyword in error_str for keyword in ["faiss", "vectorstore", "retriever"]):
                error_msg = "⚠️ Lỗi khi tìm kiếm tài liệu. Có thể vectorstore bị hỏng hoặc chưa có tài liệu."
            else:
                error_msg = f"❌ Lỗi không xác định: {str(e)}"
                logger.error(f"Unexpected error in handle_query: {e}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "citations": [],
                "self_rag_meta": None,
                "mode": pipeline_mode
            })
            st.session_state.should_scroll = True
            save_current_session_fn()
            
            st.rerun()

        # Lưu và scroll
        

# ── Internal helper (giữ nguyên) ─────────────────────────────────────────────
def _update_bm25_cache(session_state, all_docs, top_k, retrieval_mode,
                       search_type, retriever_kwargs,
                       bm25_weight=None, faiss_weight=None):
    bm25_cache = {
        "bm25_retriever": session_state.get("bm25_retriever"),
        "bm25_doc_count": session_state.get("bm25_doc_count"),
    }

    source_filter = retriever_kwargs.get("filter", {}).get("source")

    retriever, bm25_cache = get_retriever(
        vectorstore=session_state.vectorstore,
        all_docs=all_docs,
        retrieval_mode=retrieval_mode,
        search_type=search_type,
        retriever_kwargs=retriever_kwargs,
        top_k=top_k,
        bm25_cache=bm25_cache,
    )

    if retrieval_mode == "hybrid" and bm25_weight is not None and faiss_weight is not None:
        faiss_retriever = session_state.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=retriever_kwargs
        )
        from src.hybrid_search import build_ensemble_retriever
        retriever = build_ensemble_retriever(
            bm25_cache["bm25_retriever"], faiss_retriever, bm25_weight, faiss_weight
        )
        logger.info(f"Hybrid weights applied: BM25={bm25_weight:.2f}, FAISS={faiss_weight:.2f}")

    session_state["bm25_retriever"] = bm25_cache["bm25_retriever"]
    session_state["bm25_doc_count"] = bm25_cache["bm25_doc_count"]

    return retriever, session_state