def corag_select(query, docs, k=5, lambda_mult=0.5):
    """
    Simple MMR-style selection.
    
    """
    selected = []
    candidates = docs.copy()

    while len(selected) < k and candidates:
        best_doc = None
        best_score = -1e9

        for doc in candidates:
            relevance = doc.metadata.get("score", 0)

            diversity = 0
            for s in selected:
                diversity += similarity(doc, s)  # you implement

            score = relevance - lambda_mult * diversity

            if score > best_score:
                best_score = score
                best_doc = doc

        selected.append(best_doc)
        candidates.remove(best_doc)

    return selected