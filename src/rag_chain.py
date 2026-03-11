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
Bạn là trợ lý thông minh trả lời dựa trên tài liệu sau. Chỉ dùng thông tin trong CONTEXT, không bịa thêm.
Nếu không biết thì nói "Không tìm thấy thông tin trong tài liệu".

CONTEXT: {context}

Câu hỏi: {question}

Trả lời bằng tiếng Việt nếu câu hỏi bằng tiếng Việt, tiếng Anh nếu bằng tiếng Anh.
"""

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
