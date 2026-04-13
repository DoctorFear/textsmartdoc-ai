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
from src.config import BM25_WEIGHT, FAISS_WEIGHT   # ← THÊM DÒNG NÀY
from src.corag_pipeline import corag_pipeline

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

    # Hiển thị Multi-hop steps nếu có
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
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    #     if "self_rag_meta" in message and message["self_rag_meta"]:
    #         with st.expander("🔁 Thông tin Self-RAG"):
    #             render_self_rag_meta(message["self_rag_meta"])

    #     if "citations" in message and message["citations"]:
    #         with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
    #             for cite in message["citations"]:
    #                 st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
    #                 st.info(f'"{cite["snippet"]}"')
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "compare_data" in message:
                st.markdown("### Câu trả lời")
                st.subheader("**RAG Truyền thống**")
                st.markdown(message["compare_data"]["rag"]["content"])
                if message["compare_data"]["rag"].get("citations"):
                    with st.expander("📚 Nguồn RAG"):
                        for cite in message["compare_data"]["rag"]["citations"]:
                            st.write(f"[{cite['index']}] {cite['file']} (Tr. {cite['page']})")
            
                st.subheader("**CoRAG (Corrective)**")
                st.markdown(message["compare_data"]["corag"]["content"])
                if message["compare_data"]["corag"].get("citations"):
                    with st.expander("🔍 Nguồn CoRAG"):
                        for cite in message["compare_data"]["corag"]["citations"]:
                            st.write(f"[{cite['index']}] {cite['file']} (Tr. {cite['page']})")
            else:
                # Hiển thị tin nhắn assistant bình thường
                st.markdown(message["content"])
                if "self_rag_meta" in message and message["self_rag_meta"]:
                    with st.expander("🔁 Thông tin Self-RAG"):
                        render_self_rag_meta(message["self_rag_meta"])
                if "citations" in message and message["citations"]:
                    with st.expander("Xem nguồn trích dẫn"):
                        for cite in message["citations"]:
                            st.markdown(f"**Nguồn {cite['index']}:** `{cite['file']}` - P.{cite['page']}")
    streamlit_js_eval(
        js_expressions="window.scrollTo(0, document.body.scrollHeight)",
        key=f"scroll_{len(st.session_state.messages)}"
    )

# ── Xử lý câu hỏi & sinh câu trả lời ────────────────────────────────────────
# def handle_query(prompt, settings, save_current_session_fn, create_new_chat_session_fn):
#     if st.session_state.vectorstore is None:
#         st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
#         st.stop()

#     # Tạo session tự động nếu cần
#     if st.session_state.current_chat_id is None:
#         create_new_chat_session_fn(
#             prompt[:40] + ("..." if len(prompt) > 40 else ""),
#             keep_current_context=True
#         )
#         save_current_session_fn()
#         st.session_state.pending_prompt = prompt
#         st.rerun()

#     # Append user message
#     is_first_message = len(st.session_state.messages) == 0
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Cập nhật title nếu là tin nhắn đầu tiên
#     for s in st.session_state.chat_sessions:
#         if s["id"] == st.session_state.current_chat_id and is_first_message:
#             new_title = prompt[:48] + ("..." if len(prompt) > 48 else "")
#             s["title"] = new_title
#             break

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # ================== UNPACK SETTINGS ==================
#     top_k            = settings["top_k"]
#     fetch_k          = settings["fetch_k"]
#     search_type      = settings["search_type"]
#     lambda_mult      = settings["lambda_mult"]
#     retrieval_mode   = settings["retrieval_mode"]
#     use_reranker     = settings.get("use_reranker", False)
#     self_rag_enabled = settings.get("self_rag_enabled", False)

#     retrieved_docs = []
#     self_rag_meta  = None

#     with st.chat_message("assistant"):
#         spinner_text = "Đang tìm kiếm và suy nghĩ..."
#         if self_rag_enabled:
#             spinner_text = "Đang chạy Self-RAG..."
#         elif use_reranker:
#             spinner_text = "Đang rerank bằng Cross-Encoder..."

#         with st.spinner(spinner_text):
#             try:
#                 logger.info(f"Query: '{prompt[:80]}...'")

#                 # Filter theo tài liệu
#                 source_filter = None
#                 if "source_filter_select" in st.session_state:
#                     selected = st.session_state.source_filter_select
#                     if selected != "Tất cả tài liệu":
#                         source_filter = selected

#                 retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
#                 if source_filter:
#                     retriever_kwargs["filter"] = {"source": source_filter}
#                 if search_type == "mmr":
#                     retriever_kwargs["lambda_mult"] = lambda_mult

#                 all_docs = list(st.session_state.vectorstore.docstore._dict.values())

#                 # ================== SELF-RAG ==================
#                 if self_rag_enabled:

#                     bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
#                     faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)
                
#                     retriever, st.session_state = _update_bm25_cache(
#                         st.session_state,
#                         all_docs, top_k,
#                         retrieval_mode="hybrid",
#                         search_type=search_type,
#                         retriever_kwargs=retriever_kwargs,
#                         bm25_weight=bm25_weight,
#                         faiss_weight=faiss_weight
#                     )
#                     result = self_rag_query(prompt, retriever, max_retries=2)
#                     response = result["answer"]
#                     retrieved_docs = result["docs"]
#                     self_rag_meta = {
#                         "attempts": result["attempts"],
#                         "confidence": result["confidence"],
#                         "query_used": result["query_used"],
#                         "evaluation": result["evaluation"],
#                         "multi_hop_steps": result.get("multi_hop_steps")
#                     }
#                     logger.info(f"Self-RAG | attempts={result['attempts']} | "
#                                 f"confidence={result['confidence']} | "
#                                 f"bm25_weight={bm25_weight}, faiss_weight={faiss_weight}")

#                 else:
#                     # ================== NORMAL RAG ==================
#                     query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

#                     # Lấy trọng số từ settings
#                     bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
#                     faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

#                     retriever, st.session_state = _update_bm25_cache(
#                         st.session_state,
#                         all_docs, 
#                         top_k,
#                         retrieval_mode=retrieval_mode,
#                         search_type=search_type,
#                         retriever_kwargs=retriever_kwargs,
#                         bm25_weight=bm25_weight,
#                         faiss_weight=faiss_weight
#                     )

#                     retrieved_docs = retriever.invoke(query_for_retrieval)

#                     if use_reranker:
#                         retrieved_docs = rerank(query_for_retrieval, retrieved_docs, top_k=top_k)

#                     if not retrieved_docs:
#                         response = "Không tìm thấy thông tin liên quan trong tài liệu."
#                     else:
#                         context = format_docs(retrieved_docs)
#                         response = rag_chain.invoke({
#                             "context": context,
#                             "question": prompt,
#                             "language_instruction": get_language_instruction(prompt)
#                         }).strip()

#                 citations = build_citations(retrieved_docs)

#             except Exception as e:
#                 response = f"❌ Lỗi xử lý: {str(e)}"
#                 citations = []
#                 self_rag_meta = None

#         st.markdown(response)

#     # Hiển thị meta
#     if self_rag_meta:
#         with st.expander("🔁 Thông tin Self-RAG"):
#             render_self_rag_meta(self_rag_meta)

#     if citations:
#         with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
#             st.markdown("Hệ thống đã dựa vào các đoạn sau để trả lời:")
#             for cite in citations:
#                 st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
#                 st.info(f'"{cite["snippet"]}"')

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response,
#         "citations": citations,
#         "self_rag_meta": self_rag_meta,
#     })

#     save_current_session_fn()

#     streamlit_js_eval(
#         js_expressions="parent.document.querySelectorAll('*').forEach(el => el.scrollTop = el.scrollHeight);",
#         key=f"scroll_{int(time.time() * 100000)}"
#     )

def handle_query(prompt, settings, save_current_session_fn, create_new_chat_session_fn):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
        st.stop()

    # Create session if needed
    if st.session_state.current_chat_id is None:
        create_new_chat_session_fn(prompt[:40] + "...", keep_current_context=True)
        save_current_session_fn()
        st.session_state.pending_prompt = prompt
        st.rerun()

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ===== SETTINGS =====
    top_k            = settings["top_k"]
    fetch_k          = settings["fetch_k"]
    search_type      = settings["search_type"]
    lambda_mult      = settings["lambda_mult"]
    retrieval_mode   = settings["retrieval_mode"]
    use_reranker     = settings.get("use_reranker", False)
    self_rag_enabled = settings.get("self_rag_enabled", False)

    compare_mode = st.session_state.get("compare_mode", False)
    all_docs = list(st.session_state.vectorstore.docstore._dict.values())

    # ===== BUILD RETRIEVER =====
    retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
    if search_type == "mmr":
        retriever_kwargs["lambda_mult"] = lambda_mult

    retriever, _ = _update_bm25_cache(
        st.session_state,
        all_docs,
        top_k,
        retrieval_mode,
        search_type,
        retriever_kwargs,
        bm25_weight=settings.get("bm25_weight", BM25_WEIGHT),
        faiss_weight=settings.get("faiss_weight", FAISS_WEIGHT)
    )

    # ===== EXECUTION =====
    with st.chat_message("assistant"):

        # ================== COMPARE MODE ==================
        if compare_mode:
            # --- RAG Processing ---
            docs_rag = retriever.invoke(prompt)
            if use_reranker:
                docs_rag = rerank(prompt, docs_rag, top_k=top_k)

            ans_rag = rag_chain.invoke({
                "context": format_docs(docs_rag),
                "question": prompt,
                "language_instruction": get_language_instruction(prompt)
            }).strip()
            cite_rag = build_citations(docs_rag)

            # --- CoRAG Processing ---
            result_corag = corag_pipeline(prompt, retriever)
            ans_corag = result_corag["answer"]
            docs_corag = result_corag["docs"]
            cite_corag = build_citations(docs_corag)

            # ONLY SAVE DATA, DO NOT RENDER UI HERE
            st.session_state.messages.append({
                "role": "assistant",
                "compare_data": {
                    "rag": {"content": ans_rag, "citations": cite_rag},
                    "corag": {"content": ans_corag, "citations": cite_corag}
                }
            })
            # Force a rerun to let render_chat_history() pick up the new message
            st.rerun()

        # ================== NORMAL / SELF-RAG ==================
        else:
            with st.spinner("Đang xử lý..."):
                try:
                    if self_rag_enabled:
                        result = self_rag_query(prompt, retriever, max_retries=2)

                        response = result["answer"]
                        retrieved_docs = result["docs"]

                        self_rag_meta = {
                            "attempts": result["attempts"],
                            "confidence": result["confidence"],
                            "query_used": result["query_used"],
                            "evaluation": result["evaluation"],
                            "multi_hop_steps": result.get("multi_hop_steps")
                        }
                    else:
                        query = rewrite_with_history(prompt, st.session_state.messages)

                        retrieved_docs = retriever.invoke(query)

                        if use_reranker:
                            retrieved_docs = rerank(query, retrieved_docs, top_k=top_k)

                        if not retrieved_docs:
                            response = "Không tìm thấy thông tin liên quan trong tài liệu."
                        else:
                            response = rag_chain.invoke({
                                "context": format_docs(retrieved_docs),
                                "question": prompt,
                                "language_instruction": get_language_instruction(prompt)
                            }).strip()

                        self_rag_meta = None

                    citations = build_citations(retrieved_docs)

                except Exception as e:
                    response = f"❌ Lỗi: {str(e)}"
                    citations = []
                    self_rag_meta = None

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "citations": citations,
                "self_rag_meta": self_rag_meta
            })

    save_current_session_fn()

    # No rerun needed


# ── Internal helper ───────────────────────────────────────────────────────────
def _update_bm25_cache(session_state, all_docs, top_k, retrieval_mode, 
                       search_type, retriever_kwargs, 
                       bm25_weight=None, faiss_weight=None):
    """Cập nhật retriever với hỗ trợ trọng số động cho Hybrid"""
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

    # Rebuild EnsembleRetriever nếu là hybrid và có trọng số tùy chỉnh
    if retrieval_mode == "hybrid" and bm25_weight is not None and faiss_weight is not None:
        faiss_retriever = session_state.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=retriever_kwargs
        )
        from src.hybrid_search import build_ensemble_retriever
        
        retriever = build_ensemble_retriever(
            bm25_cache["bm25_retriever"],
            faiss_retriever,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight
        )
        logger.info(f"Hybrid weights applied: BM25={bm25_weight:.2f}, FAISS={faiss_weight:.2f}")

    session_state["bm25_retriever"] = bm25_cache["bm25_retriever"]
    session_state["bm25_doc_count"] = bm25_cache["bm25_doc_count"]

    return retriever, session_state