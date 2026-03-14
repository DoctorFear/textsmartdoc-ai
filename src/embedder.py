# src/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'gpu'},          # 'cuda' nếu có GPU
    encode_kwargs={'normalize_embeddings': True}
)

