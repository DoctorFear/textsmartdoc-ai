# src/rag_chain.py LLM + prompt + Self-RAG → 8.2.10
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
import json

# --- Khởi tạo LLM ---
llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)
# --- Thiết kế Prompt Template chính cho RAG ---
prompt_template = """
Bạn là trợ lý thông minh, trả lời CHÍNH XÁC dựa trên tài liệu.

QUY TẮC BẮT BUỘC:
1. CHỈ dùng thông tin trong CONTEXT. Không bịa thêm, không suy luận ngoài.
2. Nếu CONTEXT có thông tin liên quan (dù chỉ một phần nhỏ), BẮT BUỘC phải trả lời dựa trên đó.
3. Chỉ được nói "Không tìm thấy thông tin" khi CONTEXT hoàn toàn KHÔNG có bất kỳ thông tin nào liên quan đến câu hỏi.
4. Trả lời ngắn gọn, rõ ràng, tự nhiên (3-6 câu).
5. Nếu câu hỏi là tiếng Việt, TUYỆT ĐỐI trả lời hoàn toàn bằng tiếng Việt.
6. TUYỆT ĐỐI không dùng tiếng Trung hoặc ký tự Hán trong câu trả lời.

{language_instruction}

CONTEXT:
{context}

Câu hỏi: {question}

Trả lời:
"""
# --- Thiết kế Prompt Template để viết lại câu hỏi (Self-RAG: Query Rewriting) ---
REWRITE_PROMPT_TEMPLATE = """Bạn là trợ lý thông minh, hãy viết lại câu hỏi sau để tối ưu cho việc tìm kiếm ngữ nghĩa trong tài liệu.
Chỉ trả về câu hỏi đã được viết lại, không giải thích bất cứ điều gì thêm.

YÊU CẦU NGÔN NGỮ:
{language_lock}

Câu hỏi gốc: {question}
Câu hỏi đã được viết lại:"""

# --- Thiết kế Prompt Template dùng để tự đánh giá câu trả lười (Self-RAG: Self-Evaluation) ---
EVAL_PROMPT_TEMPLATE = """Bạn là trợ lý thông minh, hãy đánh giá câu trả lời sau dựa trên câu hỏi và mức độ chính xác liên quan đến ngữ cảnh tài liệu cung cấp.
Chỉ trả về JSON hợp lệ, KHÔNG giải thích, KHÔNG thêm bất kỳ text nào khác.
Format bắt buộc: {{"score": <số từ 1 đến 10>, "reason": "<lý do ngắn gọn>", "is_sufficient": <true hoặc false>}}
 
YÊU CẦU NGÔN NGỮ CHO FIELD reason:
{language_lock}

Câu hỏi: {question}
Ngữ cảnh tài liệu: {context}
Câu trả lời: {answer}

JSON:"""

# --- Hàm kiểm tra xem câu hỏi có thể là tiếng Việt hay không ---
def is_probably_vietnamese(text: str) -> bool:
    text_lower = f" {text.lower().strip()} "
    viet_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    viet_hints = [
        " là ", " và ", " của ", " trong ", " cho ", " với ",
        " tóm tắt ", " câu ", " truyện ", " tài liệu ",
        " ai ", " gì ", " nào ", " vì sao ", " như thế nào "
    ]
    return any(ch in text_lower for ch in viet_chars) or any(word in text_lower for word in viet_hints)


# --- Hàm tạo chỉ thị ngôn ngữ dựa trên độ ngữ tiếng Việt của câu hỏi ---
def get_language_instruction(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "BẮT BUỘC: Trả lời HOÀN TOÀN bằng tiếng Việt tự nhiên. "
            "TUYỆT ĐỐI không dùng tiếng Trung, không xen tiếng Trung, "
            "không dùng ký tự Hán trong câu trả lời."
        )
    return (
        "Answer ONLY in natural ENGLISH. "
        "Do not use Chinese characters or Chinese words."
    )


# --- Hàm tạo khóa ngôn ngữ cho bước self-rag (rewrite/evaluate) ---
def get_self_rag_language_lock(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Trả về HOÀN TOÀN bằng tiếng Việt. "
            "Tuyệt đối không dùng tiếng Trung hay ký tự Hán."
        )
    return "Return ONLY in English. Do not use Chinese."

# --- Hàm định dạng danh sách tài liệu thành một chuỗi context ---
def format_docs(docs):
    # docs là list Document
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

PROMPT = PromptTemplate.from_template(prompt_template)

rag_chain = (
    PROMPT
    | llm
    | StrOutputParser()
)

# -----------------------------------------------------
# SELF-RAG (8.2.10)
# -----------------------------------------------------
# --- Hàm viết lại câu hỏi (query rewrite) cho self-rag ---
def rewrite_query(question: str) -> str:
    """Viết lại câu hỏi để cải thiện chất lượng tìm kiếm ngữ nghĩa.
    Được gọi từ lần thử thứ 2 trở đi trong self_rag_query.
    """
    rewrite_prompt = PromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = rewrite_chain.invoke({
        "question": question,
        "language_lock": get_self_rag_language_lock(question)
    }).strip()

 
    # Đảm bảo không trả về chuỗi rỗng
    rewritten = rewritten.replace("```", "").replace("json", "").strip().strip('"')
    return rewritten if rewritten else question
 
 
# --- Hàm đánh giá chất lượng câu trả lời do LLM sinh ra ---
def evaluate_answer(question: str, context: str, answer: str) -> dict:
    """LLM tự đánh giá chất lượng câu trả lời dựa trên ngữ cảnh.

    Returns:
        dict với các key: score (1-10), reason (str), is_sufficient (bool)
    """
    eval_prompt = PromptTemplate.from_template(EVAL_PROMPT_TEMPLATE)
    eval_chain = eval_prompt | llm | StrOutputParser()
    raw = eval_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
        "language_lock": get_self_rag_language_lock(question)
    }).strip()
 
    try:
        # Loại bỏ markdown code block nếu LLM trả về có backtick
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        # Fallback nếu LLM không trả về JSON hợp lệ
        return {"score": 3, "reason": "LLM trả về JSON không hợp lệ", "is_sufficient": False}
 
 
# --- Hàm chính Self-RAG: retrieve, generate, evaluate, retry ---
def self_rag_query(question: str, retriever, max_retries: int = 2) -> dict:
    """Pipeline Self-RAG hoàn chỉnh: retrieve → generate → evaluate → retry nếu cần.
 
    Args:
        question:    Câu hỏi gốc của người dùng.
        retriever:   FAISS retriever đã được cấu hình (từ app.py truyền vào).
        max_retries: Số lần thử tối đa (mặc định 2).
 
    Returns:
        dict gồm: answer, confidence (score), query_used, attempts, evaluation, docs
    """
    last_result = {}
 
    for attempt in range(max_retries+1):
        # Lần thử đầu dùng câu hỏi gốc, từ lần 2 trở đi thì rewrite
        query = question if attempt == 0 else rewrite_query(question)
 
        # Truy xuất các đoạn văn bản liên quan
        retrieved_docs = retriever.invoke(query)
 
        if not retrieved_docs:
            last_result = {
                "answer": "Không tìm thấy thông tin liên quan trong tài liệu.",
                "confidence": 0,
                "query_used": query,
                "attempts": attempt + 1,
                "evaluation": {"score": 0, "reason": "Không có docs", "is_sufficient": False},
                "docs": []
            }
            continue
 
        # Tạo context và sinh câu trả lời
        context = format_docs(retrieved_docs)
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "language_instruction": get_language_instruction(question)
        }).strip()
 
        # Tự đánh giá chất lượng câu trả lời
        evaluation = evaluate_answer(question, context, answer)
 
        last_result = {
            "answer": answer,
            "confidence": evaluation.get("score", 5),
            "query_used": query,
            "attempts": attempt + 1,
            "evaluation": evaluation,
            "docs": retrieved_docs
        }
 
        # Nếu câu trả lời đã đủ tốt thì dừng sớm, không retry
        if evaluation.get("is_sufficient", False):
            break
 
    return last_result

# --- Chạy trực tiếp ---
if __name__ == "__main__":
    chunks = load_and_split("data/VoNhat.pdf")
    vs = create_vectorstore(chunks, embedder)

    query = "Tóm tắt nội dung chính của tài liệu?"
    docs = vs.similarity_search(query, k=3)

    # Chuyển docs thành string trước khi đưa vào chain
    context_text = format_docs(docs)

    answer = rag_chain.invoke({"context": context_text, "question": query, "language_instruction": get_language_instruction(query)})
    print("Q:", query)
    print("A:", answer)