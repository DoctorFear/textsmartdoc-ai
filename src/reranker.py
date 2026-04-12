#Cross-encoder rerank → 8.2.9
from sentence_transformers import CrossEncoder

# good balance model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]