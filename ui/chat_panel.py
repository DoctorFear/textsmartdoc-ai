# Khu vực Q&A, hiển thị citation → 8.2.5, 8.2.6

# ui/chat_panel.py
import streamlit as st
import time
from streamlit_js_eval import streamlit_js_eval

from src.rag_chain import get_language_instruction, format_docs, rag_chain, self_rag_query
from src.conversational import rewrite_with_history
from src.reranker import rerank
from src.hybrid_search import get_retriever
from src.citation import build_citations
from src.logger import setup_logger

logger = setup_logger()


# ── Helper render Self-RAG metadata ──────────────────────────────────────────

def render_self_rag_meta(meta: dict):
    """Render khối thông tin Self-RAG gọn, đều dòng và dễ đọc hơn."""
    if not meta:
        return

    attempts = meta.get("attempts", 1)
    confidence = meta.get("confidence", "?")
    query_used = meta.get("query_used", "")
    reason = meta.get("evaluation", {}).get("reason", "")

    blocks = [
        f"<div><strong>Số lần thử:</strong> {attempts} | <strong>Độ tin cậy:</strong> {confidence}/10</div>"
    ]
    if attempts > 1 and query_used:
        blocks.append(
            f"<div><strong>Câu hỏi được viết lại:</strong> <em>{query_used}</em></div>"
        )
    if reason:
        blocks.append(f"<div><strong>Lý do đánh giá:</strong> {reason}</div>")

    st.markdown(
        f"<div class='self-rag-meta'>{''.join(blocks)}</div>",
        unsafe_allow_html=True
    )


# ── Render lịch sử chat ───────────────────────────────────────────────────────

def render_chat_history():
    """Hiển thị toàn bộ lịch sử messages của session hiện tại."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        if "self_rag_meta" in message and message["self_rag_meta"]:
            with st.expander("🔁 Thông tin Self-RAG"):
                render_self_rag_meta(message["self_rag_meta"])

        if "citations" in message and message["citations"]:
            with st.expander(" Xem nguồn trích dẫn (Citations) & Highlight"):
                for cite in message["citations"]:
                    st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                    st.info(f'"{cite["snippet"]}"')


# ── Xử lý câu hỏi & sinh câu trả lời ────────────────────────────────────────

def handle_query(prompt, settings, save_current_session_fn, create_new_chat_session_fn):
    """
    Nhận prompt từ user, chạy pipeline RAG, render kết quả.
    """
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
        st.stop()

    # Tạo session tự động nếu user hỏi mà chưa nhấn New Chat
    if st.session_state.current_chat_id is None:
        create_new_chat_session_fn(
            prompt[:40] + ("..." if len(prompt) > 40 else ""),
            keep_current_context=True
        )
        save_current_session_fn()
        st.session_state.pending_prompt = prompt
        st.rerun()

    # Append user message
    is_first_message = len(st.session_state.messages) == 0
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Cập nhật title nếu cần
    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id:
            if is_first_message:
                new_title = prompt[:48] + ("..." if len(prompt) > 48 else "")
                old_title = s["title"]
                s["title"] = new_title
                logger.info(f"Đổi title session {s['id']}: '{old_title}' → '{new_title}' (dựa trên câu hỏi đầu tiên)")
            break

    with st.chat_message("user"):
        st.markdown(prompt)

    # ================== UNPACK SETTINGS ==================
    top_k            = settings["top_k"]
    fetch_k          = settings["fetch_k"]
    search_type      = settings["search_type"]
    lambda_mult      = settings["lambda_mult"]
    retrieval_mode   = settings["retrieval_mode"]
    use_reranker     = settings.get("use_reranker", False)
    self_rag_enabled = settings.get("self_rag_enabled", False)   # ← Lấy từ selectbox

    retrieved_docs = []
    self_rag_meta  = None

    with st.chat_message("assistant"):
        spinner_text = "Đang tìm kiếm và suy nghĩ..."
        if self_rag_enabled:
            spinner_text = "Đang chạy Self-RAG (tự đánh giá)..."
        elif use_reranker:
            spinner_text = "Đang tìm kiếm và rerank bằng Cross-Encoder..."

        with st.spinner(spinner_text):
            if st.session_state.vectorstore is None:
                response  = "Vui lòng upload tài liệu trước khi hỏi."
                citations = []
            else:
                try:
                    logger.info(f"Query: '{prompt[:80]}...'" if len(prompt) > 80 else f"Query: '{prompt}'")

                    # Áp dụng filter tài liệu
                    source_filter = None
                    if "source_filter_select" in st.session_state:
                        selected = st.session_state.source_filter_select
                        if selected != "Tất cả tài liệu":
                            source_filter = selected

                    retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
                    if source_filter:
                        retriever_kwargs["filter"] = {"source": source_filter}
                        logger.info(f"Áp dụng filter source: {source_filter}")
                    if search_type == "mmr":
                        retriever_kwargs["lambda_mult"] = lambda_mult

                    all_docs = list(st.session_state.vectorstore.docstore._dict.values())

                    # ================== SELF-RAG ==================
                    if self_rag_enabled:                    # ← Sửa ở đây
                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state,
                            all_docs, top_k,
                            retrieval_mode="hybrid",        # Self-RAG thường dùng hybrid
                            search_type=search_type,
                            retriever_kwargs=retriever_kwargs,
                        )
                        result = self_rag_query(prompt, retriever, max_retries=2)
                        response       = result["answer"]
                        retrieved_docs = result["docs"]
                        self_rag_meta  = {
                            "attempts":   result["attempts"],
                            "confidence": result["confidence"],
                            "query_used": result["query_used"],
                            "evaluation": result["evaluation"],
                        }
                        logger.info(f"Self-RAG | attempts={result['attempts']} | confidence={result['confidence']}")

                    else:
                        # Normal RAG
                        query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state,
                            all_docs, top_k,
                            retrieval_mode=retrieval_mode,
                            search_type=search_type,
                            retriever_kwargs=retriever_kwargs,
                        )

                        retrieved_docs = retriever.invoke(query_for_retrieval)

                        # Re-ranking
                        if use_reranker:
                            retrieved_docs = rerank(query_for_retrieval, retrieved_docs, top_k=top_k)

                        logger.info(
                            f"Retrieved {len(retrieved_docs)} docs | "
                            f"retrieval_mode={retrieval_mode} | search_type={search_type}"
                        )

                        if not retrieved_docs:
                            response = "Không tìm thấy thông tin liên quan trong tài liệu."
                        else:
                            context  = format_docs(retrieved_docs)
                            response = rag_chain.invoke({
                                "context": context,
                                "question": prompt,
                                "language_instruction": get_language_instruction(prompt)
                            }).strip()

                            if "không tìm thấy" in response.lower() or len(response.strip()) < 15:
                                response = "Không tìm thấy thông tin phù hợp trong tài liệu."

                    citations = build_citations(retrieved_docs)

                except Exception as e:
                    err = str(e).lower()
                    if "connection" in err or "refused" in err:
                        response = "🔌 Mất kết nối đến Ollama. Kiểm tra `ollama serve` và thử lại."
                    elif "timeout" in err:
                        response = "⏱️ Ollama phản hồi quá chậm."
                    else:
                        response = f"❌ Lỗi xử lý câu hỏi: {str(e)}"
                    citations = []
                    self_rag_meta = None

        st.markdown(response)

    # Hiển thị Self-RAG meta nếu có
    if self_rag_meta:
        with st.expander("🔁 Thông tin Self-RAG"):
            render_self_rag_meta(self_rag_meta)

    # Hiển thị citations
    if citations:
        with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
            st.markdown("Hệ thống đã dựa các đoạn văn bản sau để tạo câu trả lời:")
            for cite in citations:
                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                st.info(f'"{cite["snippet"]}"')

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       response,
        "citations":     citations,
        "self_rag_meta": self_rag_meta,
    })

    save_current_session_fn()

    streamlit_js_eval(
        js_expressions="parent.document.querySelectorAll('*').forEach(el => el.scrollTop = el.scrollHeight);",
        key=f"scroll_{int(time.time() * 100000)}"
    )
# ── Internal helper ───────────────────────────────────────────────────────────

def _update_bm25_cache(session_state, all_docs, top_k, retrieval_mode, search_type, retriever_kwargs):
    """
    Cập nhật retriever với filter source nếu có.
    """
    bm25_cache = {
        "bm25_retriever": session_state.get("bm25_retriever"),
        "bm25_doc_count": session_state.get("bm25_doc_count"),
    }

    # Lấy filter từ retriever_kwargs (nếu có)
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

    # === SỬA QUAN TRỌNG: Áp dụng filter thủ công cho BM25 và Hybrid ===
    if source_filter:
        # Lọc docs trước khi đưa vào BM25 hoặc Ensemble
        filtered_docs = [doc for doc in all_docs 
                        if doc.metadata.get("source") == source_filter]
        
        if filtered_docs:
            # Tạo lại BM25 chỉ trên docs đã lọc
            from src.hybrid_search import build_bm25_retriever
            bm25_retriever = build_bm25_retriever(filtered_docs, top_k)
            bm25_cache["bm25_retriever"] = bm25_retriever
            bm25_cache["bm25_doc_count"] = len(filtered_docs)

            # Nếu là hybrid → rebuild ensemble với BM25 đã lọc
            if retrieval_mode == "hybrid":
                faiss_retriever = session_state.vectorstore.as_retriever(
                    search_type=search_type,
                    search_kwargs={**retriever_kwargs, "filter": {"source": source_filter}}
                )
                from src.hybrid_search import build_ensemble_retriever
                retriever = build_ensemble_retriever(
                    bm25_retriever, 
                    faiss_retriever,
                    bm25_weight=0.3,   # hoặc lấy từ config
                    faiss_weight=0.7
                )
            else:
                # Nếu chỉ BM25 thì dùng retriever đã lọc
                retriever = bm25_retriever

    session_state["bm25_retriever"] = bm25_cache["bm25_retriever"]
    session_state["bm25_doc_count"] = bm25_cache["bm25_doc_count"]

    return retriever, session_state