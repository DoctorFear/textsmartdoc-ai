# src/rag_chain.py
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore

llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

prompt_template = """
Bạn là trợ lý thông minh, trả lời CHÍNH XÁC dựa trên tài liệu.

QUY TẮC BẮT BUỘC:
1. CHỈ dùng thông tin trong CONTEXT. Không bịa thêm, không suy luận ngoài.
2. Nếu CONTEXT có thông tin liên quan (dù chỉ một phần nhỏ), BẮT BUỘC phải trả lời dựa trên đó.
3. Chỉ được nói "Không tìm thấy thông tin" khi CONTEXT hoàn toàn KHÔNG có bất kỳ thông tin nào liên quan đến câu hỏi.
4. Trả lời ngắn gọn, rõ ràng, tự nhiên (3-6 câu).

{language_instruction}

CONTEXT:
{context}

Câu hỏi: {question}

Trả lời:
"""

def get_language_instruction(question: str) -> str:
    viet_chars = 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    is_viet = any(c in question.lower() for c in viet_chars)
    
    if is_viet:
        return """BẮT BUỘC: Trả lời BẰNG TIẾNG VIỆT, tự nhiên, không lẫn tiếng Anh/Trung.
    Ưu tiên trả lời nếu có bất kỳ thông tin liên quan nào trong CONTEXT."""
    else:
        return """Answer in ENGLISH, concise and natural.
    Only say you don't know if there is truly no relevant information."""

PROMPT = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    # docs là list Document
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

rag_chain = (
    PROMPT
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    chunks = load_and_split("data/VoNhat.pdf")
    vs = create_vectorstore(chunks, embedder)

    query = "Tóm tắt nội dung chính của tài liệu?"
    docs = vs.similarity_search(query, k=3)

    # Chuyển docs thành string trước khi đưa vào chain
    context_text = format_docs(docs)

    answer = rag_chain.invoke({"context": context_text, "question": query})
    print("Q:", query)
    print("A:", answer)