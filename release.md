smartdoc-ai/
│
├── app.py # Entry point - chỉ gọi UI, không chứa logic
│
├── requirements.txt
├── README.md
├── .env # Config (model name, paths,...)
│
├── src/ # Toàn bộ business logic
│ │
│ ├── loader.py # (đã có) Load PDF, DOCX → 8.2.1
│ ├── embedder.py # (đã có) HuggingFace embeddings
│ ├── vectorstore.py # (đã có) FAISS CRUD + metadata filter → 8.2.8
│ ├── rag*chain.py # (đã có) LLM + prompt + Self-RAG → 8.2.10
│ ├── reranker.py # (đã có) Cross-encoder rerank → 8.2.9
│ ├── conversational.py # (đã có) Rewrite question with history → 8.2.6
│ ├── logger.py # (đã có) Logging
│ ├── persistence.py # (đã có) Save/load FAISS + chat JSON → 8.2.2
│ │
│ ├── hybrid_search.py # BM25 + vector ensemble retriever → 8.2.7
│ ├── citation.py # Source tracking, page number, highlight → 8.2.5
│ └── config.py # Tập trung chunk_size, k, model name,... → 8.2.4
│
├── ui/ # Toàn bộ Streamlit UI tách theo feature
│ ├── **init**.py
│ ├── sidebar.py # Sidebar: lịch sử, settings, clear buttons → 8.2.2, 8.2.3
│ ├── upload_panel.py # Khu vực upload file (PDF/DOCX) → 8.2.1, 8.2.8
│ ├── chat_panel.py # Khu vực Q&A, hiển thị citation → 8.2.5, 8.2.6
│ └── settings_panel.py # Chunk params, search type UI → 8.2.4, 8.2.7
│
├── data/ # File mẫu
│ └── sample.pdf
│
├── faiss_store/ # Auto-generated, lưu FAISS index theo session
│ └── chat*<id>/
│
├── logs/ # Auto-generated
│ └── smartdoc_YYYYMMDD.log
│
└── tests/ # → 8.3 Hướng dẫn đánh giá (Testing 10%)
├── test_loader.py
├── test_vectorstore.py
├── test_rag_chain.py
└── test_reranker.py
