# src/vectorstore.py
from langchain_community.vectorstores import FAISS
from src.loader import load_and_split
from src.embedder import embedder

def create_vectorstore(chunks, embedder):
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore

