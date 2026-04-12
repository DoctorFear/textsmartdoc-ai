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
    settings: dict từ sidebar (top_k, fetch_k, search_type, lambda_mult, retrieval_mode, use_reranker)
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
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Cập nhật title nếu cần
    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id:
            if s["title"] == "Cuộc trò chuyện mới":
                s["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
            break

    with st.chat_message("user"):
        st.markdown(prompt)

    # Unpack settings
    top_k          = settings["top_k"]
    fetch_k        = settings["fetch_k"]
    search_type    = settings["search_type"]
    lambda_mult    = settings["lambda_mult"]
    retrieval_mode = settings["retrieval_mode"]
    use_reranker   = settings["use_reranker"]

    retrieved_docs = []
    self_rag_meta  = None

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và suy nghĩ..."):
            if st.session_state.vectorstore is None:
                response  = "Vui lòng upload tài liệu trước khi hỏi."
                citations = []
            else:
                try:
                    logger.info(f"Query: '{prompt[:80]}...'" if len(prompt) > 80 else f"Query: '{prompt}'")  # ← LOG 4

                    # Áp dụng filter tài liệu nếu người dùng chọn (8.2.8)
                    source_filter = None
                    if "source_filter_select" in st.session_state:
                        selected = st.session_state.source_filter_select
                        if selected != "Tất cả tài liệu":
                            source_filter = selected

                    # Cấu hình retriever kwargs
                    retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
                    if source_filter:
                        retriever_kwargs["filter"] = {"source": source_filter}
                    if search_type == "mmr":
                        retriever_kwargs["lambda_mult"] = lambda_mult

                    all_docs = list(st.session_state.vectorstore.docstore._dict.values())

                    if st.session_state.self_rag_enabled:
                        # ── Chế độ Self-RAG (8.2.10) ─────────────────────────
                        # Self-RAG dùng hybrid bên trong get_retriever
                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state,
                            all_docs, top_k,
                            retrieval_mode="hybrid",
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
                        logger.info(
                            f"Self-RAG | attempts={result['attempts']} | "
                            f"confidence={result['confidence']} | "
                            f"query_used='{result['query_used'][:60]}'"
                        )
                    else:
                        # ── Chế độ thường: Conversational RAG (8.2.6) ────────
                        query_for_retrieval = rewrite_with_history(
                            prompt,
                            st.session_state.messages
                        )

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state,
                            all_docs, top_k,
                            retrieval_mode=retrieval_mode,
                            search_type=search_type,
                            retriever_kwargs=retriever_kwargs,
                        )

                        retrieved_docs = retriever.invoke(query_for_retrieval)

                        # Re-ranking (8.2.9)
                        if use_reranker:
                            retrieved_docs = rerank(query_for_retrieval, retrieved_docs, top_k=top_k)

                        logger.info(
                            f"Retrieved {len(retrieved_docs)} docs | "
                            f"retrieval_mode={retrieval_mode} | search_type={search_type} | "
                            f"sources: {[doc.metadata.get('source', 'unknown') for doc in retrieved_docs]}"
                        )

                        if not retrieved_docs:
                            response = "Không tìm thấy thông tin liên quan trong tài liệu."
                        else:
                            context  = format_docs(retrieved_docs)
                            response = rag_chain.invoke({
                                "context":              context,
                                "question":             prompt,
                                "language_instruction": get_language_instruction(prompt)
                            }).strip()

                            if "không tìm thấy" in response.lower() or len(response.strip()) < 15:
                                response = "Không tìm thấy thông tin phù hợp trong tài liệu."

                    # Tạo citations (8.2.5) — dùng src/citation.py
                    citations = build_citations(retrieved_docs)

                    logger.info(f"Response length: {len(response)} chars")  # ← LOG 6

                except Exception as e:
                    err = str(e).lower()
                    if "connection" in err or "refused" in err:
                        response = "🔌 Mất kết nối đến Ollama. Kiểm tra `ollama serve` và thử lại."
                    elif "timeout" in err:
                        response = "⏱️ Ollama phản hồi quá chậm."
                    else:
                        response = f"❌ Lỗi xử lý câu hỏi: {str(e)}"
                    citations     = []
                    self_rag_meta = None

        st.markdown(response)

    # Hiển thị thông tin Self-RAG nếu có (8.2.10)
    if self_rag_meta:
        with st.expander("🔁 Thông tin Self-RAG"):
            render_self_rag_meta(self_rag_meta)

    # Hiển thị citations (8.2.5)
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

    # Lưu sau mỗi lượt trả lời
    save_current_session_fn()

    streamlit_js_eval(
        js_expressions="""
            parent.document.querySelectorAll('*').forEach(function(el) {
                el.scrollTop = el.scrollHeight;
            });
        """,
        key=f"scroll_{int(time.time() * 100000)}"
    )


# ── Internal helper ───────────────────────────────────────────────────────────

def _update_bm25_cache(session_state, all_docs, top_k, retrieval_mode, search_type, retriever_kwargs):
    """
    Gọi src/hybrid_search.get_retriever(), truyền session_state làm bm25_cache,
    rồi cập nhật lại session_state với cache mới trả về.
    """
    bm25_cache = {
        "bm25_retriever": session_state.get("bm25_retriever"),
        "bm25_doc_count": session_state.get("bm25_doc_count"),
    }

    retriever, bm25_cache = get_retriever(
        vectorstore=session_state.vectorstore,
        all_docs=all_docs,
        retrieval_mode=retrieval_mode,
        search_type=search_type,
        retriever_kwargs=retriever_kwargs,
        top_k=top_k,
        bm25_cache=bm25_cache,
    )

    session_state["bm25_retriever"] = bm25_cache["bm25_retriever"]
    session_state["bm25_doc_count"] = bm25_cache["bm25_doc_count"]

    return retriever, session_state