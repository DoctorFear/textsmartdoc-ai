# SmartDoc-AI

Ứng dụng RAG (Retrieval-Augmented Generation) dùng **LangChain**, **FAISS**, **SentenceTransformers** và **Streamlit** để đọc, chia nhỏ, embed và truy vấn tài liệu PDF.

## 📂 Cấu trúc dự án

```
smartdoc-ai/
├── app.py              # Streamlit chính
├── requirements.txt    # danh sách thư viện Python
├── data/               # đặt PDF test ở đây
├── src/                # code modules
│   ├── loader.py       # Load + split PDF
│   ├── splitter.py     # Hiện tại đang gộp chung với loader.py
│   ├── embedder.py     # Tạo embedding
│   ├── vectorstore.py  # FAISS vectorstore
│   └── rag_chain.py    # RAG pipeline + test
├── .gitignore          # ignore venv/, __pycache__/, .env, etc.
└── README.md           # hướng dẫn
```

## 🚀 Cài đặt

### 1. Tạo và kích hoạt môi trường ảo

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

### 2. Cài đặt thư viện Python

```bash
pip install -r requirements.txt
```

### 3. Cài đặt Ollama và model

- Tải Ollama: [https://ollama.com](https://ollama.com)
- Chạy model Qwen2.5 7B:

```bash
ollama run qwen2.5:7b
```

## ▶️ Chạy ứng dụng

Trong môi trường `.venv`, chạy:

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trên trình duyệt tại `http://localhost:8501`.

1. **Loader (`loader.py`)**
   - Đọc file PDF bằng `PDFPlumberLoader`.
   - Tách văn bản thành các chunk bằng `RecursiveCharacterTextSplitter`.

2. **Embedder (`embedder.py`)**
   - Tạo vector embedding cho từng chunk bằng mô hình SentenceTransformer.

3. **Vectorstore (`vectorstore.py`)**
   - Lưu trữ embedding bằng FAISS.
   - Cho phép tìm kiếm ngữ nghĩa để lấy các đoạn văn bản liên quan.

4. **RAG Chain (`rag_chain.py`)**
   - Ghép context từ vectorstore vào prompt.
   - Gửi prompt cho LLM (Ollama model `qwen2.5:7b`).
   - Trả về câu trả lời dựa trên tài liệu.
