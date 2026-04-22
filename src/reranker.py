#Cross-encoder rerank → 8.2.9
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

def rerank(query, docs, top_k=3):
    docs = docs[:15]
    pairs = [(query, doc.page_content) for doc in docs]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]