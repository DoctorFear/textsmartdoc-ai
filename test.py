# test_vectorstore.py
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore

# Load PDF thành chunks
chunks = load_and_split("data/VoNhat.pdf")
print(f"Loaded {len(chunks)} chunks")

# Tạo vectorstore từ chunks + embedder
vs = create_vectorstore(chunks, embedder)

# Lưu index để tái sử dụng
vs.save_local("faiss_index")
print("FAISS index saved to faiss_index/")

# Thử truy vấn
query = "Nhân vật Tràng gặp ai ở chợ?"
docs = vs.similarity_search(query, k=2)
for i, doc in enumerate(docs, 1):
    print(f"\nResult {i}:")
    print(doc.page_content[:200])
