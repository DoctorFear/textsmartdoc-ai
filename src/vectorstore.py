# src/vectorstore.py FAISS CRUD + metadata filter → 8.2.8
from langchain_community.vectorstores import FAISS
from src.loader import load_and_split
from src.embedder import embedder

def create_vectorstore(chunks, embedder):
    # Tạo FAISS vector store mới từ danh sách chunks (lần đầu upload)
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore

def add_to_vectorstore(exisiting_vectorstore: FAISS, new_chunks, embedder) -> FAISS:
    # Thêm tài liệu mới vào vector store đã có sẵn (8.2.8 multi-document upload)
    if exisiting_vectorstore is None:
        return create_vectorstore(new_chunks, embedder)

    # Thêm chunks mới vào index hiện tại của vector store
    exisiting_vectorstore.add_documents(new_chunks)
    return exisiting_vectorstore

def get_retriever(vectorstore: FAISS, search_type="similarity", k=3, fetch_k=30, lambda_mult=0.7, source_filter: str = None):
    """Tạo retriever từ vector store với các tùy chọn tìm kiếm.
 
    Args:
        vectorstore:   FAISS vector store đã được tạo.
        search_type:   'similarity' hoặc 'mmr'.
        k:             Số chunks trả về (top-k).
        fetch_k:       Số chunks lấy trước khi lọc (dùng cho MMR).
        lambda_mult:   Hệ số đa dạng cho MMR (0=đa dạng, 1=chính xác).
        source_filter: Tên file nguồn để lọc kết quả (dùng cho multi-doc).
    """

    search_kwargs = {"k": k, "fetch_k": fetch_k}

    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    if search_type == "mmr":
        search_kwargs["lambda_mult"] = lambda_mult
    
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def get_uploaded_sources(vectorstore: FAISS):
    # Lấy danh sách các nguồn tài liệu đã được upload vào vector store
    if vectorstore is None:
        return []
    
    try:
        # Duyệt qua toàn bộ docstore để thu thập metadata 'source'
        sources = set()
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            src = doc.metadata.get("source", "")
            if src:
                sources.add(src)
        return sorted(sources)
    except Exception:
        return []