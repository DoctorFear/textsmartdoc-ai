# src/vectorstore.py
from langchain_community.vectorstores import FAISS
from src.loader import load_and_split
from src.embedder import embedder

def create_vectorstore(chunks, embedder):
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore

# Test kết hợp với loader + embedder
chunks = load_and_split("data/VoNhat.pdf")
vs = create_vectorstore(chunks, embedder)
vs.save_local("faiss_index")  # lưu để reuse