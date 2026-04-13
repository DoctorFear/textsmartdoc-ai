import json
from src.rag_chain import llm, rag_chain, get_language_instruction, format_docs
from langchain.prompts import PromptTemplate

# Prompt to grade document relevance
GRADE_PROMPT = """Bạn là một giám khảo chấm điểm mức độ liên quan.
Đánh giá xem tài liệu dưới đây có chứa thông tin để trả lời câu hỏi không.
Chỉ trả về 'yes' hoặc 'no'.

Câu hỏi: {question}
Tài liệu: {context}
Kết quả (yes/no):"""

def grade_documents(question, documents):
    """Lọc bỏ các documents không liên quan."""
    relevant_docs = []
    for doc in documents:
        prompt = GRADE_PROMPT.format(question=question, context=doc.page_content)
        score = llm.invoke(prompt).strip().lower()
        if 'yes' in score:
            relevant_docs.append(doc)
    return relevant_docs

def corag_pipeline(question, retriever):
    """
    Corrective RAG: 
    1. Retrieve 
    2. Grade 
    3. If poor quality, signal for Web Search (or rewrite)
    """
    initial_docs = retriever.invoke(question)
    filtered_docs = grade_documents(question, initial_docs)
    
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