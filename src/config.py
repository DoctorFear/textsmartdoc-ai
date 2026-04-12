# Tập trung chunk_size, k, model name,... → 8.2.4# src/config.py
# ── Tập trung toàn bộ hyperparameter mặc định của hệ thống (8.2.4) ───────────
# Khi người dùng chỉnh ở UI → giá trị UI ghi đè lên default này.
# Khi cần thay đổi default toàn hệ thống → chỉ sửa file này.

# ── Document processing ───────────────────────────────────────────────────────
CHUNK_SIZE        = 1200   # Số ký tự tối đa mỗi chunk
CHUNK_OVERLAP     = 200    # Số ký tự overlap giữa 2 chunk liền kề
MAX_FILE_SIZE_MB  = 50     # Giới hạn kích thước file upload (MB)

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = 3      # Số chunks trả về sau retrieval
FETCH_K           = 30     # Số chunks lấy trước khi lọc (dùng cho MMR)
LAMBDA_MULT       = 0.7    # Hệ số đa dạng MMR (0=đa dạng, 1=chính xác)
SEARCH_TYPE       = "similarity"   # "similarity" hoặc "mmr"
RETRIEVAL_MODE    = "hybrid"       # "faiss", "bm25", hoặc "hybrid"

# ── Hybrid search weights (8.2.7) ─────────────────────────────────────────────
BM25_WEIGHT       = 0.3    # Trọng số BM25 trong ensemble
FAISS_WEIGHT      = 0.7    # Trọng số FAISS trong ensemble

# ── Reranker (8.2.9) ──────────────────────────────────────────────────────────
USE_RERANKER      = True   # Bật/tắt cross-encoder reranking mặc định

# ── Self-RAG (8.2.10) ─────────────────────────────────────────────────────────
SELF_RAG_ENABLED  = False  # Mặc định tắt để tiết kiệm tài nguyên
SELF_RAG_MAX_RETRIES = 2   # Số lần retry tối đa

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL         = "qwen2.5:7b"
LLM_TEMPERATURE   = 0.7
LLM_TOP_P         = 0.9
LLM_REPEAT_PENALTY = 1.1

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DEVICE  = "cpu"   # "cuda" nếu có GPU

# ── UI ────────────────────────────────────────────────────────────────────────
SIDEBAR_TEXT_COLOR = "#FFFFFF"
MAX_HISTORY_TURNS  = 4     # Số lượt hội thoại tối đa đưa vào context (8.2.6)