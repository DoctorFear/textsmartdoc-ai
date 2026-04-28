import json
from src.rag_chain import (
    rewrite_query,
    evaluate_answer,
    format_docs,
    rag_chain,
    get_language_instruction,
    decompose_question,
    llm,
    detect_language,           # ← Thêm dòng này
    get_self_rag_language_lock
)
from langchain.prompts import PromptTemplate

# Prompt to grade document relevance
GRADE_PROMPT = """Bạn là một giám khảo chấm điểm mức độ liên quan.
Đánh giá xem tài liệu dưới đây có chứa thông tin để trả lời câu hỏi không.
Chỉ trả về 'yes' hoặc 'no'.

Câu hỏi: {question}
Tài liệu: {context}
Kết quả (yes/no):"""

def grade_documents(question, documents):
    scored = []

    for doc in documents:
        prompt = GRADE_PROMPT.format(question=question, context=doc.page_content)
        score = llm.invoke(prompt).strip().lower()

        val = 1 if score.startswith("yes") else 0
        scored.append((val, doc))

    # sort descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored]

def corag_pipeline(question, retriever, top_k):
    """
    Corrective RAG: 
    1. Retrieve 
    2. Grade 
    3. If poor quality, signal for Web Search (or rewrite)
    """
    initial_docs = retriever.invoke(question)
    filtered_docs = grade_documents(question, initial_docs)[:top_k]
    
    # Nếu không có tài liệu nào đạt yêu cầu, chúng ta "khắc phục" bằng cách mở rộng tìm kiếm
    if not filtered_docs:
        # Trong môi trường Local, ta thử rewrite câu hỏi để tìm lại (tương tự Self-RAG)
        # Hoặc dùng k cao hơn
        filtered_docs = retriever.invoke(f"Thông tin chi tiết về {question}")
    
    context = format_docs(filtered_docs)
    answer = rag_chain.invoke({
        "context": context, 
        "question": question, 
        "language_instruction": get_language_instruction(question)
    })
    
    return {
        "answer": answer,
        "docs": filtered_docs,
        "mode": "CoRAG"
    }

def self_corag_query(question, retriever, max_retries: int=2):
    """
    Self + CoRAG:
    - Retrieve
    - Grade docs (CoRAG)
    - Generate answer
    - Self-evaluate
    - Retry with rewrite + multi-hop if needed
    """

    last_result = {}
    multi_hop_steps = None

    for attempt in range(max_retries + 1):
        query = question if attempt == 0 else rewrite_query(question)

        # 🔁 Multi-hop when retry
        if attempt >= 1:
            steps = decompose_question(question)
            multi_hop_steps = steps

            all_docs = []
            for step in steps:
                docs = retriever.invoke(step)
                all_docs.extend(docs)

        else:
            all_docs = retriever.invoke(query)

        # 🧠 CoRAG grading step
        graded_docs = grade_documents(question, all_docs)

        # fallback if nothing useful
        if not graded_docs:
            graded_docs = retriever.invoke(f"Thông tin chi tiết về {question}")

        selected_docs = graded_docs[:5]

        if not selected_docs:
            last_result = {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "confidence": 0,
                "query_used": query,
                "attempts": attempt + 1,
                "evaluation": {"score": 0, "reason": "Không có docs", "is_sufficient": False},
                "docs": [],
                "multi_hop_steps": multi_hop_steps
            }
            continue

        context = format_docs(selected_docs)

        # ✨ Generate answer
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "language_instruction": get_language_instruction(question)
        }).strip()

        # 🔍 Self-evaluate
        lang = detect_language(question)                    # ← thêm dòng này
        language_lock = get_self_rag_language_lock(lang)    # ← thêm dòng này

        evaluation = evaluate_answer(question, context, answer, language_lock)

        last_result = {
            "answer": answer,
            "confidence": evaluation.get("score", 5),
            "query_used": query,
            "attempts": attempt + 1,
            "evaluation": evaluation,
            "docs": selected_docs,
            "multi_hop_steps": multi_hop_steps
        }

        # ✅ Stop if good enough
        if evaluation.get("is_sufficient", False):
            break

    return last_result