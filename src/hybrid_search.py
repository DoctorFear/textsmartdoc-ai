# BM25 + vector ensemble retriever → 8.2.7# src/hybrid_search.py
# ── Hybrid Search: kết hợp BM25 (keyword) + FAISS (semantic) (8.2.7) ─────────
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from src.config import BM25_WEIGHT, FAISS_WEIGHT
from src.logger import setup_logger

logger = setup_logger()


def build_bm25_retriever(all_docs: list, k: int) -> BM25Retriever:
    """
    Tạo BM25Retriever từ toàn bộ documents trong vectorstore.
    Được cache ở tầng UI (session_state) để tránh tạo lại mỗi lần query.

    Args:
        all_docs: list Document lấy từ vectorstore.docstore._dict.values()
        k:        số kết quả trả về
    Returns:
        BM25Retriever đã cấu hình
    """
    logger.info(f"Đang tạo index BM25 từ {len(all_docs)} docs (k={k})...")
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = k
    return retriever

def build_ensemble_retriever(
    bm25_retriever: BM25Retriever,
    faiss_retriever,
    bm25_weight: float = BM25_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT,
) -> EnsembleRetriever:
    """
    Kết hợp BM25 và FAISS thành EnsembleRetriever với trọng số động.
    """
    # Đảm bảo tổng trọng số = 1.0
    total = bm25_weight + faiss_weight
    if total == 0:
        bm25_weight = 0.3
        faiss_weight = 0.7
    else:
        bm25_weight = bm25_weight / total
        faiss_weight = faiss_weight / total

    logger.info(f"Xây dựng EnsembleRetriever | BM25={bm25_weight:.2f} | FAISS={faiss_weight:.2f}")

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, faiss_weight]
    )


def get_retriever(
    vectorstore: FAISS,
    all_docs: list,
    retrieval_mode: str,
    search_type: str,
    retriever_kwargs: dict,
    top_k: int,
    bm25_cache: dict,
) -> tuple:
    """
    Hàm chính: trả về retriever phù hợp theo retrieval_mode.
    Tự động cache BM25 index vào bm25_cache để tránh build lại.

    Args:
        vectorstore:     FAISS vectorstore hiện tại
        all_docs:        list Document từ docstore
        retrieval_mode:  "faiss" | "bm25" | "hybrid"
        search_type:     "similarity" | "mmr"
        retriever_kwargs: dict search_kwargs truyền vào FAISS retriever
        top_k:           số kết quả
        bm25_cache:      dict dùng làm cache, thường là st.session_state
                         (phải có key "bm25_retriever" và "bm25_doc_count")
    Returns:
        (retriever, bm25_cache) — trả lại cache đã cập nhật
    """
    faiss_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=retriever_kwargs
    )

    current_doc_count = len(all_docs)

    # Tạo / cập nhật BM25 nếu cần
    if (
        "bm25_retriever" not in bm25_cache
        or bm25_cache.get("bm25_doc_count") != current_doc_count
    ):
        bm25_cache["bm25_retriever"] = build_bm25_retriever(all_docs, top_k)
        bm25_cache["bm25_doc_count"] = current_doc_count
    else:
        bm25_cache["bm25_retriever"].k = top_k

    bm25_retriever = bm25_cache["bm25_retriever"]

    lambda_val = retriever_kwargs.get("lambda_mult", "N/A") if "retriever_kwargs" in locals() else "Unknown"
    mmr_info = f" | lambda_mult={lambda_val}" if search_type == "mmr" else ""
    if retrieval_mode == "faiss":
        retriever = faiss_retriever

        logger.info(f"Retrieval mode: FAISS | search_type={search_type} | top_k={top_k}{mmr_info} ")
    elif retrieval_mode == "bm25":
        retriever = bm25_retriever
        logger.info(f"Retrieval mode: BM25 |search_type={search_type} | top_k={top_k}{mmr_info} ")
    else:  # hybrid (default)
        retriever = build_ensemble_retriever(bm25_retriever, faiss_retriever)
        logger.info(f"Retrieval mode: Hybrid | search_type={search_type} | top_k={top_k}{mmr_info} |  (weights will be applied dynamically)")

    return retriever, bm25_cache