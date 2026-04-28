# src/hybrid_search.py
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from src.config import BM25_WEIGHT, FAISS_WEIGHT
from src.logger import setup_logger

logger = setup_logger()


def build_bm25_retriever(all_docs: list, k: int) -> BM25Retriever:
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
    bm25_weight: float = None,      # ← Thêm tham số này
    faiss_weight: float = None,
) -> tuple:
    """
    Trả về retriever hỗ trợ filter theo source (file chỉ định)
    """
    metadata_filter = retriever_kwargs.get("filter")   # {"source": "tên_file.pdf"}

    # === 1. FAISS Retriever (hỗ trợ filter tốt) ===
    faiss_search_kwargs = dict(retriever_kwargs)
    if metadata_filter:
        faiss_search_kwargs["filter"] = metadata_filter

    faiss_retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=faiss_search_kwargs
    )

    # === 2. BM25 Retriever - BUỘC LỌC THEO FILE KHI CÓ FILTER ===
    if metadata_filter and "source" in metadata_filter:
        source_name = metadata_filter["source"]
        filtered_docs = [doc for doc in all_docs if doc.metadata.get("source") == source_name]
        
        logger.info(f"BM25: Lọc theo file '{source_name}' | Số docs: {len(filtered_docs)}")
        
        if filtered_docs:
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = top_k
        else:
            logger.warning(f"Không tìm thấy docs cho file {source_name}, fallback toàn bộ")
            bm25_retriever = build_bm25_retriever(all_docs, top_k)
    else:
        # Không có filter → dùng cache
        current_doc_count = len(all_docs)
        if ("bm25_retriever" not in bm25_cache or 
            bm25_cache.get("bm25_doc_count") != current_doc_count):
            bm25_retriever = build_bm25_retriever(all_docs, top_k)
            bm25_cache["bm25_retriever"] = bm25_retriever
            bm25_cache["bm25_doc_count"] = current_doc_count
        else:
            bm25_retriever = bm25_cache["bm25_retriever"]
            bm25_retriever.k = top_k

    # === 3. Chọn retriever theo mode ===
    if retrieval_mode == "faiss":
        retriever = faiss_retriever
        logger.info(f"Retrieval: FAISS | Filter: {metadata_filter}")

    elif retrieval_mode == "bm25":
        retriever = bm25_retriever
        logger.info(f"Retrieval: BM25 | Filter: {metadata_filter}")

    else:  # hybrid
        retriever = build_ensemble_retriever(
            bm25_retriever, 
            faiss_retriever,
            bm25_weight=bm25_weight or BM25_WEIGHT,   # Sửa ở đây
            faiss_weight=faiss_weight or FAISS_WEIGHT
        )
        logger.info(f"Retrieval: HYBRID | Filter theo file: {metadata_filter}")

    return retriever, bm25_cache