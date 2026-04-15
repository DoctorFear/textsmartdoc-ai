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
from src.config import BM25_WEIGHT, FAISS_WEIGHT 
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
# def render_chat_history():
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             if "mode" in message:
#                 st.caption(f"Chế độ: {message['mode']}")
#             st.markdown(message["content"])

#         if "self_rag_meta" in message and message["self_rag_meta"]:
#             with st.expander("🔁 Thông tin Self-RAG"):
#                 render_self_rag_meta(message["self_rag_meta"])

#         if "citations" in message and message["citations"]:
#             with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
#                 for cite in message["citations"]:
                    
#                     st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
#                     st.info(f'"{cite["snippet"]}"')
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
            mode = message.get("mode")
            
            # ================== COMPARE MODE ==================
            if mode == "rag+corag" and message.get("compare_data"):

                data = message["compare_data"]
                
                # ----- RAG -----
                st.subheader("RAG")
                st.markdown(f'<div class="rag-container">{data["rag"]["response"]}</div>', unsafe_allow_html=True)

                if data["rag"]["citations"]:
                    with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                        for cite in data["rag"]["citations"]:
                            st.markdown(
                                f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**"
                            )
                            st.info(f'"{cite["snippet"]}"')

                st.markdown("---")

                # ----- CoRAG -----
                st.subheader("CoRAG")
                st.markdown(f'<div class="rag-container">{data["corag"]["response"]}</div>', unsafe_allow_html=True)

                if data["corag"]["citations"]:
                    with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                        for cite in data["corag"]["citations"]:
                            st.markdown(
                                f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**"
                            )
                            st.info(f'"{cite["snippet"]}"')

                
                st.caption("Chế độ: RAG + CoRAG")


            # ================== NORMAL MODES ==================
            else:
                st.markdown(message.get("content", ""))

                if mode:
                    mode_label = {
                        "rag": "RAG",
                        "corag": "CoRAG"
                    }.get(mode, mode)

                    st.caption(f"Chế độ: {mode_label}")

        # ----- SELF-RAG META -----
        if message.get("self_rag_meta"):
            with st.expander("🔁 Thông tin Self-RAG"):
                render_self_rag_meta(message["self_rag_meta"]) 

        # ----- NORMAL CITATIONS ONLY -----
        if (
            message.get("citations")
            and message.get("mode") != "rag+corag"
        ):
            with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                for cite in message["citations"]:
                    st.markdown(
                        f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}"
                    )
                    st.info(f'"{cite["snippet"]}"')


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
    combined_mode    = settings.get("combined_mode", "rag")

    retrieved_docs = []
    self_rag_meta  = None

    with st.chat_message("assistant"):
        spinner_text = "Đang tìm kiếm và suy nghĩ..."
        if self_rag_enabled:
            spinner_text = "Đang chạy Self-RAG..."
        elif use_reranker:
            spinner_text = "Đang rerank bằng Cross-Encoder..."
        elif combined_mode == "corag":
            spinner_text = "Đang chạy CoRAG..."
        elif combined_mode == "rag+corag":
            spinner_text = "Đang chạy RAG + CoRAG..."
        elif self_rag_enabled:
            spinner_text = "Đang chạy Self-RAG..."
        elif use_reranker:
            spinner_text = "Đang rerank bằng Cross-Encoder..."

        with st.spinner(spinner_text):
            try:
                logger.info(f"Query: '{prompt[:80]}...'")

                # Filter theo tài liệu
                source_filter = None
                if "source_filter_select" in st.session_state:
                    selected = st.session_state.source_filter_select
                    if selected != "Tất cả tài liệu":
                        source_filter = selected

                retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
                if source_filter:
                    retriever_kwargs["filter"] = {"source": source_filter}
                if search_type == "mmr":
                    retriever_kwargs["lambda_mult"] = lambda_mult

                all_docs = list(st.session_state.vectorstore.docstore._dict.values())
                
    
                if combined_mode == "rag+corag":
                    # Build retriever FIRST
                    bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                    faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                    retriever, st.session_state = _update_bm25_cache(
                        st.session_state,
                        all_docs,
                        top_k,
                        retrieval_mode=retrieval_mode,
                        search_type=search_type,
                        retriever_kwargs=retriever_kwargs,
                        bm25_weight=bm25_weight,
                        faiss_weight=faiss_weight
                    )

                    query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

                    # ================== RAG ==================
                    rag_docs = retriever.invoke(query_for_retrieval)

                    if use_reranker:
                        rag_docs = rerank(query_for_retrieval, rag_docs, top_k=top_k)

                    if not rag_docs:
                        rag_response = "Không tìm thấy thông tin (RAG)."
                    else:
                        context = format_docs(rag_docs)
                        rag_response = rag_chain.invoke({
                            "context": context,
                            "question": prompt,
                            "language_instruction": get_language_instruction(prompt)
                        }).strip()

                    rag_citations = build_citations(rag_docs)

                    # ================== CoRAG ==================
                    corag_result = corag_pipeline(prompt, retriever, top_k=top_k)

                    corag_response = corag_result["answer"]
                    corag_docs = corag_result["docs"]
                    corag_citations = build_citations(corag_docs)

                    logger.info("Mode=RAG+CoRAG (Compare)")

                    # ================== UI ==================
                    st.markdown("### RAG Result")
                    st.markdown(rag_response)

                    if rag_citations:
                        with st.expander("RAG - Xem nguồn trích dẫn (Citations) & Highlight"):
                            for cite in rag_citations:
                                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                                st.info(f'"{cite["snippet"]}"')

                    st.markdown("---")

                    st.markdown("### CoRAG Result")
                    st.markdown(corag_response)

                    if corag_citations:
                        with st.expander("CoRAG - Xem nguồn trích dẫn (Citations) & Highlight"):
                            for cite in corag_citations:
                                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                                st.info(f'"{cite["snippet"]}"')
                    
                    st.caption("Chế độ: RAG + CoRAG")

                    # ================== SAVE MESSAGE ==================
                    combined_response = (
                        "### RAG Result\n" + rag_response +
                        "\n\n---\n\n### CoRAG Result\n" + corag_response
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "mode": "rag+corag",
                        "compare_data": {
                            "rag": {
                                "response": rag_response,
                                "citations": rag_citations
                            },
                            "corag": {
                                "response": corag_response,
                                "citations": corag_citations
                            }
                        },
                        "content": None,
                        "citations": None,
                        "self_rag_meta": None
                    })

                else:
                    if combined_mode == "rag":
                        # ================== SELF-RAG ==================
                        if self_rag_enabled:

                            bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                            faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)
                        
                            retriever, st.session_state = _update_bm25_cache(
                                st.session_state,
                                all_docs, top_k,
                                retrieval_mode="hybrid",
                                search_type=search_type,
                                retriever_kwargs=retriever_kwargs,
                                bm25_weight=bm25_weight,
                                faiss_weight=faiss_weight
                            )
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
                            logger.info(f"Self-RAG | attempts={result['attempts']} | "
                                        f"confidence={result['confidence']} | "
                                        f"bm25_weight={bm25_weight}, faiss_weight={faiss_weight}")

                        else:
                            # ================== NORMAL RAG ==================
                            query_for_retrieval = rewrite_with_history(prompt, st.session_state.messages)

                            # Lấy trọng số từ settings
                            bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                            faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                            retriever, st.session_state = _update_bm25_cache(
                                st.session_state,
                                all_docs, 
                                top_k,
                                retrieval_mode=retrieval_mode,
                                search_type=search_type,
                                retriever_kwargs=retriever_kwargs,
                                bm25_weight=bm25_weight,
                                faiss_weight=faiss_weight
                            )

                            retrieved_docs = retriever.invoke(query_for_retrieval)

                            if use_reranker:
                                retrieved_docs = rerank(query_for_retrieval, retrieved_docs, top_k=top_k)

                            if not retrieved_docs:
                                response = "Không tìm thấy thông tin liên quan trong tài liệu."
                            else:
                                context = format_docs(retrieved_docs)
                                response = rag_chain.invoke({
                                    "context": context,
                                    "question": prompt,
                                    "language_instruction": get_language_instruction(prompt)
                                }).strip()
                        logger.info("Mode=RAG")
                        citations = build_citations(retrieved_docs)

                    elif combined_mode == "corag":
                        bm25_weight = settings.get("bm25_weight", BM25_WEIGHT)
                        faiss_weight = settings.get("faiss_weight", FAISS_WEIGHT)

                        retriever, st.session_state = _update_bm25_cache(
                            st.session_state,
                            all_docs,
                            top_k,
                            retrieval_mode=retrieval_mode,
                            search_type=search_type,
                            retriever_kwargs=retriever_kwargs,
                            bm25_weight=bm25_weight,
                            faiss_weight=faiss_weight
                        )

                        result = corag_pipeline(prompt, retriever)

                        response = result["answer"]
                        retrieved_docs = result["docs"]
                        self_rag_meta = None

                        logger.info("Mode=CoRAG")
                        citations = build_citations(retrieved_docs)
                    
                    st.markdown(response)

                    # Hiển thị meta
                    if self_rag_meta:
                        with st.expander("🔁 Thông tin Self-RAG"):
                            render_self_rag_meta(self_rag_meta)

                    if citations:
                        with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
                            st.markdown("Hệ thống đã dựa vào các đoạn sau để trả lời:")
                            for cite in citations:
                                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                                st.info(f'"{cite["snippet"]}"')

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "citations": citations,
                        "self_rag_meta": self_rag_meta,
                        "mode": combined_mode 
                    })
                
                if combined_mode == "rag":
                    st.caption(f"Chế độ: RAG")
                elif combined_mode == "corag":
                    st.caption(f"Chế độ: CoRAG")
                elif combined_mode == "rag+corag":
                    st.caption(f"Chế độ: RAG & CoRAG")

            except Exception as e:
                response = f"❌ Lỗi xử lý: {str(e)}"
                citations = []
                self_rag_meta = None


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