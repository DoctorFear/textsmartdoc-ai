# src/rag_chain.py LLM + prompt + Self-RAG → 8.2.10
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
import json
from src.logger import setup_logger

logger = setup_logger()

# --- Khởi tạo LLM ---
llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

# --- Prompt Template chính cho RAG (giữ nguyên) ---
prompt_template = """
Bạn là trợ lý thông minh, trả lời CHÍNH XÁC dựa trên tài liệu.

QUY TẮC BẮT BUỘC:
1. CHỈ dùng thông tin trong TÀI LIỆU (CONTEXT). Không bịa thêm, không suy luận ngoài.
2. Nếu TÀI LIỆU có thông tin liên quan (dù chỉ một phần nhỏ), BẮT BUỘC phải trả lời dựa trên đó.
3. Chỉ được nói "Không tìm thấy thông tin trong tài liệu" khi TÀI LIỆU hoàn toàn KHÔNG có bất kỳ thông tin nào liên quan đến câu hỏi.
4. Trả lời ngắn gọn, rõ ràng, tự nhiên (3-6 câu).
5. Nếu câu hỏi là tiếng Việt, TUYỆT ĐỐI trả lời hoàn toàn bằng tiếng Việt.
6. TUYỆT ĐỐI không dùng tiếng Trung hoặc ký tự Hán trong câu trả lời.

{language_instruction}

CONTEXT:
{context}

Câu hỏi: {question}

Trả lời:
"""

# --- Prompt Rewrite (giữ nguyên) ---
REWRITE_PROMPT_TEMPLATE = """Bạn là trợ lý thông minh chuyên tối ưu hóa câu hỏi tìm kiếm.

NHẬN XÉT NGÔN NGỮ:
{language_lock}

YÊU CẦU NGHIÊM NGẶT:
- Chỉ trả về câu hỏi đã được viết lại, KHÔNG giải thích thêm bất kỳ điều gì.
- TUYỆT ĐỐI không dùng tiếng Trung, không dùng ký tự Hán.
- Giữ nguyên ý nghĩa của câu hỏi gốc.
- Làm cho câu hỏi rõ ràng, tự nhiên và dễ tìm kiếm hơn.

Câu hỏi gốc: {question}

Câu hỏi đã được viết lại:"""

# --- Prompt Evaluate (giữ nguyên) ---
EVAL_PROMPT_TEMPLATE = """Bạn là trợ lý thông minh, hãy đánh giá câu trả lời sau dựa trên câu hỏi và mức độ chính xác liên quan đến ngữ cảnh tài liệu cung cấp.
Chỉ trả về JSON hợp lệ, KHÔNG giải thích, KHÔNG thêm bất kỳ text nào khác.
Format bắt buộc: {{"score": <số từ 1 đến 10>, "reason": "<lý do ngắn gọn>", "is_sufficient": <true hoặc false>}}
 
YÊU CẦU NGÔN NGỮ CHO FIELD reason:
{language_lock}

Câu hỏi: {question}
Ngữ cảnh tài liệu: {context}
Câu trả lời: {answer}

JSON:"""

# --- Các hàm hỗ trợ ngôn ngữ (giữ nguyên) ---
def is_probably_vietnamese(text: str) -> bool:
    text_lower = f" {text.lower().strip()} "
    viet_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    viet_hints = [
        " là ", " và ", " của ", " trong ", " cho ", " với ",
        " tóm tắt ", " câu ", " truyện ", " tài liệu ",
        " ai ", " gì ", " nào ", " vì sao ", " như thế nào "
    ]
    return any(ch in text_lower for ch in viet_chars) or any(word in text_lower for word in viet_hints)


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


def get_self_rag_language_lock(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Câu hỏi gốc là tiếng Việt. "
            "BẮT BUỘC phải viết lại câu hỏi HOÀN TOÀN bằng tiếng Việt tự nhiên. "
            "TUYỆT ĐỐI không được dùng tiếng Trung, không dùng bất kỳ ký tự Hán nào."
        )
    return (
        "Return ONLY the rewritten question in natural English. "
        "Do not use any Chinese characters."
    )


def format_docs(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


PROMPT = PromptTemplate.from_template(prompt_template)
rag_chain = PROMPT | llm | StrOutputParser()


# -----------------------------------------------------
# SIMPLE MULTI-HOP REASONING (Thêm mới)
# -----------------------------------------------------
def decompose_question(question: str) -> list:
    """Phân tích câu hỏi thành tối đa 3 bước tìm kiếm logic (Simple Multi-hop)"""
    prompt = PromptTemplate.from_template("""
Bạn là trợ lý thông minh chuyên phân tích câu hỏi.

Hãy phân tích câu hỏi sau thành **tối đa 3 bước tìm kiếm rõ ràng, logic và cụ thể**.
Mỗi bước phải là một câu hỏi con có thể dùng để tìm kiếm thông tin trong tài liệu.

YÊU CẦU:
- Trả về dạng danh sách đánh số (1., 2., 3.)
- Không giải thích thêm, không thêm chữ gì ngoài các bước.
- Nếu câu hỏi đơn giản thì có thể chỉ cần 1 hoặc 2 bước.
- Giữ nguyên ngôn ngữ tiếng Việt nếu câu hỏi gốc là tiếng Việt.

Câu hỏi: {question}

Các bước cần tìm:""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip()

    # Xử lý output để lấy danh sách các bước
    lines = [line.strip() for line in result.split('\n') if line.strip()]
    steps = []
    
    for line in lines:
        # Loại bỏ số thứ tự và dấu đầu dòng
        cleaned = line.split(' ', 1)[-1].strip() if line[0].isdigit() or line.startswith(('-', '•', '1.', '2.', '3.')) else line.strip()
        if cleaned and len(cleaned) > 5:   # Lọc bỏ các bước quá ngắn
            steps.append(cleaned)

    # Fallback nếu không phân tích được
    if not steps or len(steps) == 0:
        steps = [question]

    logger.info(f"[Multi-hop] Decomposed steps: {steps}")
    return steps[:3]


# -----------------------------------------------------
# SELF-RAG (Đã cập nhật với Simple Multi-hop)
# -----------------------------------------------------
def rewrite_query(question: str) -> str:
    """Viết lại câu hỏi để cải thiện chất lượng tìm kiếm ngữ nghĩa."""
    rewrite_prompt = PromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    rewritten = rewrite_chain.invoke({
        "question": question,
        "language_lock": get_self_rag_language_lock(question)
    }).strip()

    cleaned = rewritten.replace("```", "").replace("json", "").replace("Câu hỏi đã được viết lại:", "").strip()
    
    if not cleaned or any(ord(c) > 0x4E00 for c in cleaned):
        return question

    return cleaned


def evaluate_answer(question: str, context: str, answer: str) -> dict:
    """LLM tự đánh giá chất lượng câu trả lời dựa trên ngữ cảnh."""
    eval_prompt = PromptTemplate.from_template(EVAL_PROMPT_TEMPLATE)
    eval_chain = eval_prompt | llm | StrOutputParser()
    raw = eval_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer,
        "language_lock": get_self_rag_language_lock(question)
    }).strip()
 
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return {"score": 3, "reason": "LLM trả về JSON không hợp lệ", "is_sufficient": False}


def self_rag_query(question: str, retriever, max_retries: int = 2) -> dict:
    last_result = {}
    multi_hop_steps = None  # Khởi tạo biến để lưu trữ xuyên suốt các vòng lặp

    for attempt in range(max_retries + 1):
        query = question if attempt == 0 else rewrite_query(question)

        # Nếu là retry hoặc force multi-hop
        if attempt >= 1:
            steps = decompose_question(question)
            multi_hop_steps = steps # Lưu vào biến ngoài scope vòng lặp
            
            all_docs = []
            for step in steps:
                docs = retriever.invoke(step)
                all_docs.extend(docs[:3])
            
            seen = set()
            retrieved_docs = []
            for doc in all_docs:
                key = doc.page_content[:150]
                if key not in seen:
                    seen.add(key)
                    retrieved_docs.append(doc)
        else:
            retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
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

        context = format_docs(retrieved_docs)
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "language_instruction": get_language_instruction(question)
        }).strip()

        evaluation = evaluate_answer(question, context, answer)

        last_result = {
            "answer": answer,
            "confidence": evaluation.get("score", 5),
            "query_used": query,
            "attempts": attempt + 1,
            "evaluation": evaluation,
            "docs": retrieved_docs,
            "multi_hop_steps": multi_hop_steps          # ← Đảm bảo luôn có
        }

        if evaluation.get("is_sufficient", False):
            break

    return last_result

# --- Chạy trực tiếp ---
if __name__ == "__main__":
    chunks = load_and_split("data/VoNhat.pdf")
    vs = create_vectorstore(chunks, embedder)

    query = "Tóm tắt nội dung chính của tài liệu?"
    docs = vs.similarity_search(query, k=3)

    context_text = format_docs(docs)
    answer = rag_chain.invoke({"context": context_text, "question": query, "language_instruction": get_language_instruction(query)})
    print("Q:", query)
    print("A:", answer)