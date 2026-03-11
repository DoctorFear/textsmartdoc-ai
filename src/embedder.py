# src/embedder.py
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},          # 'cuda' nếu có GPU
    encode_kwargs={'normalize_embeddings': True}
)

# Test
if __name__ == "__main__":
    vec = embedder.embed_query("Thử embedding tiếng Việt và English")
    print(len(vec))       # số chiều
    print(vec[:10])       # in 10 giá trị đầu để xem vector
