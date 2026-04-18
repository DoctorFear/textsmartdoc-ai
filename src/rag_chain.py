# src/rag_chain.py
# LLM + prompt + Self-RAG → 8.2.10

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
import json, re
from src.logger import setup_logger

logger = setup_logger()

# --- Khởi tạo LLM ---
llm = OllamaLLM(model="qwen2.5:7b", temperature=0.7, top_p=0.9, repeat_penalty=1.1)

# =====================================================
# --- Language Detection ---
# =====================================================
_VIET_CHARS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
_VIET_HINTS = [
    " là ", " và ", " của ", " trong ", " cho ", " với ",
    " tóm tắt ", " câu ", " truyện ", " tài liệu ",
    " ai ", " gì ", " nào ", " vì sao ", " như thế nào ",
    " bạn ", " tôi ", " này ", " đó ", " được ", " không ",
]
_HAN_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
_LANGS_NO_CHINESE = {"vi", "en", "ko", "other"}

def detect_language(text: str) -> str:
    """Detect ngôn ngữ: ưu tiên regex Việt, fallback LLM."""
    text_lower = f" {text.lower().strip()} "
    if any(ch in text_lower for ch in _VIET_CHARS) or any(h in text_lower for h in _VIET_HINTS):
        return "vi"
    prompt = (
        "Detect the language of the following text.\n"
        "Return ONLY one of these codes: vi, en, ko, ja, zh, other\n\n"
        f"Text: {text}\n\nAnswer:"
    )
    result = llm.invoke(prompt).strip().lower().replace(".", "").replace("\n", "")
    return result if result in ("vi", "en", "ko", "ja", "zh", "other") else "other"

def _contains_chinese(text: str) -> bool:
    return bool(_HAN_PATTERN.search(text))

def _strip_chinese(text: str) -> str:
    return re.sub(r" {2,}", " ", _HAN_PATTERN.sub("", text)).strip()

# =====================================================
# --- Language Instructions ---
# =====================================================
# Mỗi key → (answer_instruction, self_rag_lock)
_LANG_INSTRUCTIONS = {
    "vi": (
        "[LANGUAGE=VI]\n"
        "BẮT BUỘC PHẢI TRẢ LỜI 100% BẰNG TIẾNG VIỆT.\n"
        "TUYỆT ĐỐI KHÔNG được dùng bất kỳ tiếng Trung nào.\n"
        "TUYỆT ĐỐI KHÔNG được dùng ký tự Hán (Chinese characters).\n"
        "Chỉ sử dụng bảng chữ cái Latinh + dấu tiếng Việt (àáảãạăâêôơưđ).\n"
        "Nếu vi phạm ngôn ngữ → toàn bộ câu trả lời bị coi là SAI.",
        "CHỈ dùng tiếng Việt. TUYỆT ĐỐI không dùng tiếng Trung, không dùng ký tự Hán, không trộn ngôn ngữ."
    ),
    "en": (
        "[LANGUAGE=EN] Answer ONLY in English. Do not translate.",
        "Use English only. No Chinese characters allowed."
    ),
    "ko": (
        "[LANGUAGE=KO] 한국어로만 답변하세요. 한자(Chinese characters)를 절대 사용하지 마세요.",
        "한국어만 사용하세요. 중국어 문자 금지."
    ),
    "ja": (
        "[LANGUAGE=JA] 日本語のみで答えてください。中国語の文章は禁止。",
        "日本語のみ使用してください。中国語の文章禁止。"
    ),
    "zh": (
        "[LANGUAGE=ZH] 只能用中文回答。",
        "只能使用中文。"
    ),
}
_DEFAULT_INSTRUCTIONS = (
    "[LANGUAGE=ORIGINAL] Answer in the same language as the question.",
    "Use the same language as the question."
)

def get_language_instruction(question: str) -> str:
    """Nhận câu hỏi (string), detect ngôn ngữ rồi trả về instruction."""
    lang = detect_language(question)
    return _LANG_INSTRUCTIONS.get(lang, _DEFAULT_INSTRUCTIONS)[0]

def get_self_rag_language_lock(lang: str) -> str:
    """Nhận lang code, trả về language lock cho Self-RAG prompts."""
    return _LANG_INSTRUCTIONS.get(lang, _DEFAULT_INSTRUCTIONS)[1]

# =====================================================
# --- Prompt Templates ---
# =====================================================
PROMPT = PromptTemplate.from_template("""
Bạn là trợ lý thông minh, trả lời CHÍNH XÁC dựa trên tài liệu.

QUY TẮC:
1. CHỈ dùng thông tin trong TÀI LIỆU (CONTEXT). Không bịa thêm, không suy luận ngoài.
2. Nếu TÀI LIỆU có thông tin liên quan (dù chỉ một phần nhỏ), BẮT BUỘC phải trả lời dựa trên đó.
3. Chỉ được nói "Không tìm thấy thông tin trong tài liệu" khi TÀI LIỆU hoàn toàn KHÔNG có bất kỳ thông tin nào liên quan đến câu hỏi.
4. Trả lời ngắn gọn, rõ ràng, tự nhiên (3-6 câu).

{language_instruction}

CONTEXT:
{context}

Câu hỏi: {question}

Trả lời:""")

# Dùng {language_lock} trực tiếp trong mỗi prompt để model luôn thấy ngôn ngữ cần dùng
REWRITE_PROMPT = PromptTemplate.from_template(
    "Bạn là trợ lý thông minh chuyên tối ưu hóa câu hỏi tìm kiếm.\n\n"
    "YÊU CẦU NGÔN NGỮ (BẮT BUỘC): {language_lock}\n\n"
    "YÊU CẦU:\n"
    "- Chỉ trả về câu hỏi đã được viết lại, KHÔNG giải thích thêm.\n"
    "- TUYỆT ĐỐI không dùng tiếng Trung, không dùng ký tự Hán.\n"
    "- Giữ nguyên ý nghĩa, làm rõ ràng và dễ tìm kiếm hơn.\n\n"
    "Câu hỏi gốc: {question}\n\nCâu hỏi đã được viết lại:"
)

EVAL_PROMPT = PromptTemplate.from_template(
    "Bạn là trợ lý đánh giá câu trả lời dựa trên câu hỏi và ngữ cảnh tài liệu.\n"
    "Chỉ trả về JSON hợp lệ, KHÔNG giải thích, KHÔNG thêm text nào khác.\n"
    'Format bắt buộc: {{"score": <1-10>, "reason": "<lý do ngắn gọn>", "is_sufficient": <true/false>}}\n\n'
    "YÊU CẦU NGÔN NGỮ CHO FIELD reason (BẮT BUỘC): {language_lock}\n\n"
    "Câu hỏi: {question}\n"
    "Ngữ cảnh tài liệu: {context}\n"
    "Câu trả lời: {answer}\n\nJSON:"
)

DECOMPOSE_PROMPT = PromptTemplate.from_template(
    "Bạn là trợ lý phân tích câu hỏi.\n\n"
    "YÊU CẦU NGÔN NGỮ (BẮT BUỘC): {language_lock}\n"
    "TUYỆT ĐỐI không dùng tiếng Trung, không dùng ký tự Hán trong câu trả lời.\n\n"
    "Phân tích câu hỏi thành tối đa 3 bước tìm kiếm rõ ràng, logic, cụ thể.\n"
    "Mỗi bước là một câu hỏi con dùng để tìm thông tin trong tài liệu.\n\n"
    "YÊU CẦU:\n"
    "- Trả về danh sách đánh số (1., 2., 3.)\n"
    "- Không giải thích thêm, không thêm chữ gì ngoài các bước.\n"
    "- Câu hỏi đơn giản → 1 hoặc 2 bước là đủ.\n\n"
    "Câu hỏi: {question}\n\nCác bước cần tìm:"
)

rag_chain = PROMPT | llm | StrOutputParser()

# =====================================================
# --- Các hàm xử lý RAG ---
# =====================================================
def format_docs(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def decompose_question(question: str, language_lock: str) -> list:
    """Phân tích câu hỏi thành tối đa 3 bước tìm kiếm (Simple Multi-hop)."""
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question, "language_lock": language_lock}).strip()

    steps = []
    for line in result.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Bỏ số thứ tự đầu dòng
        cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
        if cleaned and len(cleaned) > 5 and not _contains_chinese(cleaned):
            steps.append(cleaned)

    logger.info(f"[Multi-hop] Decomposed steps: {steps}")
    return steps[:3] if steps else [question]

def rewrite_query(question: str, language_lock: str) -> str:
    """Viết lại câu hỏi để cải thiện chất lượng tìm kiếm ngữ nghĩa."""
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    
    rewritten = chain.invoke({
        "question": question, 
        "language_lock": language_lock
    }).strip()

    cleaned = re.sub(r'```.*?```', '', rewritten, flags=re.DOTALL).replace("Câu hỏi đã được viết lại:", "").strip()
    return question if not cleaned or _contains_chinese(cleaned) else cleaned

def evaluate_answer(question: str, context: str, answer: str, language_lock: str) -> dict:
    """LLM tự đánh giá chất lượng câu trả lời."""
    chain = EVAL_PROMPT | llm | StrOutputParser()
    
    raw = chain.invoke({
        "question": question, 
        "context": context, 
        "answer": answer, 
        "language_lock": language_lock
    }).strip()

    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except (json.JSONDecodeError, ValueError):
        return {"score": 3, "reason": "LLM trả về JSON không hợp lệ", "is_sufficient": False}

def _regenerate_without_chinese(context: str, question: str, language_instruction: str) -> str:
    """Regenerate câu trả lời khi phát hiện chữ Hán."""
    strict_instruction = (
        language_instruction
        + "\n\n[CẢNH BÁO]: Câu trả lời trước chứa ký tự Hán — BỊ TỪ CHỐI."
        "\nLần này BẮT BUỘC chỉ dùng chữ Latin + dấu tiếng Việt."
    )

    answer = rag_chain.invoke({
        "context": context, 
        "question": question, 
        "language_instruction": strict_instruction
    }).strip()
    
    return _strip_chinese(answer) if _contains_chinese(answer) else answer

# =====================================================
# --- Self-RAG (với Simple Multi-hop) ---
# =====================================================
def self_rag_query(question: str, retriever, max_retries: int = 2) -> dict:
    last_result = {}
    multi_hop_steps = None
    lang = detect_language(question)  # detect 1 lần dùng xuyên suốt
    language_lock = get_self_rag_language_lock(lang)
    language_instruction = get_language_instruction(lang)

    for attempt in range(max_retries + 1):
        query = question if attempt == 0 else rewrite_query(question, language_lock)

        # Multi-hop từ attempt >= 1
        if attempt >= 1:
            steps = decompose_question(question, language_lock)
            multi_hop_steps = steps
            all_docs = []
            for step in steps:
                all_docs.extend(retriever.invoke(step)[:3])
            # Dedup docs
            seen, retrieved_docs = set(), []
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
        answer = rag_chain.invoke({"context": context, "question": question, "language_instruction": language_instruction}).strip()

        # Post-processing: chống chữ Hán
        if lang in _LANGS_NO_CHINESE and _contains_chinese(answer):
            logger.warning(f"[Language Fix] Phát hiện chữ Hán → regenerate (attempt {attempt+1})")
            answer = _regenerate_without_chinese(context, question, language_instruction)

        evaluation = evaluate_answer(question, context, answer, language_lock)
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
    lang = detect_language(query)
    docs = vs.similarity_search(query, k=3)
    answer = rag_chain.invoke({"context": format_docs(docs), "question": query, "language_instruction": get_language_instruction(lang)})
    print("Q:", query)
    print("A:", answer)