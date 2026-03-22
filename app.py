# app.py
import streamlit as st
import os
import tempfile
from datetime import datetime
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
from src.rag_chain import get_language_instruction, llm, format_docs, rag_chain
from streamlit_js_eval import streamlit_js_eval
import time
from src.logger import setup_logger
# ── Import persistence (lưu/load lịch sử + FAISS ra disk) ───────────────────
from src.persistence import (
    save_history, load_history,
    save_vectorstore, load_vectorstore,
    delete_all_vectorstores
)
logger = setup_logger()



# ── Load CSS từ file riêng ───────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Theme & Color Palette ────────────────────────────────────────────────────
TEXT_SIDEBAR = "#FFFFFF"     # Màu chữ trắng cho sidebar

st.set_page_config( # Cấu hình giao diện
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (load từ file styles.css cùng thư mục)
load_css("styles.css")


# ── Caching ──── Giúp giữ lại mô hình trong bộ nhớ, tránh load lại nhiều lần tăng tốc độ.
@st.cache_resource
def get_embedder():
    return embedder

@st.cache_resource
def get_llm():
    return llm

# ── Session State ───── Lưu trữ trạng thái
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None # nơi chứa FAISS index của tài liệu.
if "messages" not in st.session_state:
    st.session_state.messages = [] # lịch sử hội thoại.
if "current_file" not in st.session_state:
    st.session_state.current_file = None #  tên file PDF hiện tại.
# ── Session State bổ sung cho chat sessions + persistence (tính năng lịch sử)───────────────────
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_history()  # load các chat cũ nếu có
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None # lưu id của đoạn chat đang hoạt động
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # giữ câu hỏi khi qua rerun để hiển thị tên của nó ở history


# ── Hàm hỗ trợ tách biệt các đoạn lặp ───────────────────────────────────────

def save_current_session_to_disk():
    """Lưu messages, file, vectorstore của chat hiện tại xuống disk"""
    if st.session_state.current_chat_id is None:
        return
    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id:
            s["messages"] = st.session_state.messages.copy()
            s["file"] = st.session_state.current_file
            if st.session_state.vectorstore is not None:
                save_vectorstore(s["id"], st.session_state.vectorstore)
            break
    save_history(st.session_state.chat_sessions)


def load_session_to_state(target_id):
    """Load đúng chat mà người dùng vừa bấm vào từ disk"""
    fresh_sessions = load_history()
    target = next((s for s in fresh_sessions if s["id"] == target_id), None)

    if target:
        st.session_state.current_chat_id = target_id
        st.session_state.messages = target["messages"].copy()
        st.session_state.current_file = target.get("file")
    else:
        # Nếu không có trên disk → giữ nguyên state hiện tại hoặc reset
        st.session_state.current_chat_id = target_id
        st.session_state.messages = []
        st.session_state.current_file = None
        logger.warning(f"Không tìm thấy chat {target_id} trên disk, reset state")

    # Luôn load vectorstore từ disk (quan trọng nhất để hỏi đáp đúng tài liệu)
    st.session_state.vectorstore = load_vectorstore(target_id, get_embedder())


def create_new_chat_session(title="Cuộc trò chuyện mới"):
    """Tạo session chat mới và cập nhật state"""
    new_id = max([s["id"] for s in st.session_state.chat_sessions], default=-1) + 1
    st.session_state.chat_sessions.append({
        "id": new_id,
        "title": title,
        "messages": [],
        "vectorstore": None,
        "file": None
    })
    st.session_state.current_chat_id = new_id
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.current_file = None
    return new_id


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT_SIDEBAR};'>SmartDoc AI</h2>", unsafe_allow_html=True)

    # Instructions section
    with st.expander("Hướng dẫn sử dụng"):
        st.markdown("""
        **Các bước hoạt động** 
        1. Tải lên file PDF ở phần chính giữa  
        2. Chờ xử lý xong (thấy thông báo xanh)  
        3. Đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh  
        4. Hệ thống chỉ trả lời dựa trên nội dung tài liệu  

        **Tùy chỉnh để có được câu trả lời tốt nhất**  
        • **Chunk Size** & **Chunk Overlap**: Tăng lên nếu tài liệu dài hoặc phức tạp (thường 1000-1500 & 200-300)  
        • **Top-k** & **Fetch-k**: Tăng nếu câu trả lời không chính xác (khuyến nghị Top-k = 5-7)  
        • **Search Type = MMR** → sẽ tự động hiện **Lambda Mult** (0.7 là giá trị tốt nhất)  
        
        **Lưu ý quan trọng**:  
        • Thông số càng lớn **không phải lúc nào cũng tốt hơn**. Cần thử nghiệm để tìm giá trị phù hợp.  
        • Nếu câu trả lời vẫn sai → thử tăng Chunk Size + Top-k hoặc chuyển sang MMR.
        """)

    # Model configuration display
    with st.expander("Cấu hình mô hình"):
        st.markdown(f"""
        • **Embedding**: paraphrase-multilingual-mpnet-base-v2  
        • **LLM**: Qwen2.5 7B (Ollama)  
        • **Retriever**: FAISS – top 4 chunks  
        • **Temperature**: 0.7  
        • **Ngày chạy**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """)
        
    # Cho phép người dùng chỉnh chunk_size, chunk_overlap, k
    with st.expander("Thiết lập & Tùy chọn"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 1200, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        top_k = st.slider("Top-k", 1, 10, 3, 1)
        fetch_k = st.slider("Fetch-k", min_value=top_k, max_value=100, value=30, step=5)
        search_type = st.selectbox("Search Type", ["similarity", "mmr"], index=0)

        # Chỉ hiện lambda khi chọn MMR
        if search_type == "mmr":
            lambda_mult = st.slider("Lambda Mult (Diversity)", 0.0, 1.0, 0.7, 0.05,
                                help="0.7 là giá trị cân bằng tốt nhất cho tiếng Việt")
        else:
            lambda_mult = 0.7  # giá trị mặc định, không dùng

    # ── Lịch sử cuộc trò chuyện ─────────────────────────────────────────────
    st.subheader("Lịch sử cuộc trò chuyện")

    # Nút xóa tất cả lịch sử
    # Nút bấm để mở dialog
    if st.button("🔄 Đặt lại lịch sử", type="tertiary", use_container_width=True):
        if len(st.session_state.chat_sessions) > 0:
            st.session_state.show_confirm = True

    # Hiển thị dialog xác nhận
    if st.session_state.get("show_confirm", False):
        st.caption("**Bạn có chắc chắn muốn xóa toàn bộ lịch sử không?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Đồng ý"):
                delete_all_vectorstores()
                st.session_state.chat_sessions   = []
                st.session_state.messages        = []
                st.session_state.current_chat_id = None
                st.session_state.vectorstore     = None
                st.session_state.current_file    = None
                save_history([])
                st.session_state.show_confirm = False
                st.rerun()
        with col2:
            if st.button("❌ Hủy"):
                st.session_state.show_confirm = False
                st.rerun()


    # ── Nút New Chat ─────────────────────────────────────────────────────────
    if st.button(
        "✨ Hộp thoại mới",
        type="secondary",
        use_container_width=True,
    ):
        # Chống spam: nếu đang ở session rỗng (chưa chat) thì không tạo thêm — chỉ rerun
        if len(st.session_state.messages) == 0 and st.session_state.current_chat_id is not None:
            st.rerun()

        # Lưu session hiện tại trước khi tạo mới
        save_current_session_to_disk()

        # Tạo session mới
        new_id = create_new_chat_session("Cuộc trò chuyện mới")
        st.rerun()


    # ── Hiển thị danh sách các cuộc chat — mới nhất lên đầu ─────────────────
    for session in reversed(st.session_state.chat_sessions):
        title     = session["title"]
        is_active = session["id"] == st.session_state.current_chat_id

        if st.button(
            f"📩 {title}",
            key=f"chat_{session['id']}",
            use_container_width=True,
            type="primary"
        ):
            # Lưu session hiện tại trước khi chuyển
            save_current_session_to_disk()

            # Load session mục tiêu
            load_session_to_state(session["id"])
            st.rerun()


# ── Main Area ────────────────────────────────────────────────────────────────
st.title("📄 Hỏi đáp thông minh với tài liệu PDF")
st.markdown("**SmartDoc AI** – Upload PDF → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")

if st.session_state.vectorstore is None:
    st.info("Vui lòng tải file PDF lên ở ô bên trái (hàng dưới) để bắt đầu hỏi đáp.")
else:
    st.success(f"Đang làm việc với tài liệu: **{st.session_state.current_file}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Input hỏi đáp + File uploader ngang hàng ────────────────────────────────
col_upload, col_chat = st.columns([1, 4])

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

with col_upload:
    # Key động để tránh cache file cũ khi chuyển chat
    uploader_key = f"uploader_{st.session_state.current_chat_id or 'new'}"
    uploaded_file = st.file_uploader(
        "TẢI PDF",
        type="pdf",
        accept_multiple_files=False,
        help="Chỉ hỗ trợ PDF có text layer (không phải scan/image-only). Tối đa ~50MB.",
        label_visibility="collapsed",
        key=uploader_key
    )

with col_chat:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...") # ô nhập liệu

if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ── Xử lý upload & indexing ──────────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ File **{uploaded_file.name}** quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Đang xử lý tài liệu..."):
            st.write("📖 Đang đọc file PDF...")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")  # ← LOG 1

                st.write("✂️ Splitting — Đang chia nhỏ văn bản...")
                chunks = load_and_split(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap) ######### CHỈNH TẠI ĐÂY
                if not chunks:
                    raise ValueError("PDF_EMPTY")
                st.write(f"✅ Splitting xong — {len(chunks)} đoạn")
                logger.info(f"Splitting xong: {len(chunks)} chunks | chunk_size={chunk_size} | overlap={chunk_overlap}")  # ← LOG 2

                st.write("🧠 Creating embeddings — Đang tạo vector...")
                st.session_state.vectorstore = create_vectorstore(chunks, get_embedder())
                st.session_state.current_file = uploaded_file.name
                logger.info(f"Vectorstore tạo xong: {uploaded_file.name}")  # ← LOG 3

                # Nếu chưa có chat → tạo mới
                if st.session_state.current_chat_id is None:
                    create_new_chat_session(uploaded_file.name[:50])
                # Lưu vectorstore cho chat hiện tại
                save_vectorstore(st.session_state.current_chat_id, st.session_state.vectorstore)
                save_history(st.session_state.chat_sessions)

                st.success(f"Hoàn tất! Tài liệu **{uploaded_file.name}** đã xử lý ({len(chunks)} đoạn).")
                st.info(f"""
                📄 **{uploaded_file.name}**  
                📦 Kích thước: {file_size_mb:.2f} MB  
                🔢 Số đoạn (chunks): {len(chunks)}  
                """)

            except ValueError as e:
                if "PDF_EMPTY" in str(e):
                    st.warning("⚠️ Không thể đọc nội dung từ file này. PDF có thể là bản scan hoặc chỉ chứa hình ảnh.")
                else:
                    st.error(f"❌ File không hợp lệ: {str(e)}")
            except Exception as e:
                err = str(e).lower()
                if "connection" in err or "refused" in err or "ollama" in err:
                    st.error("🔌 Lỗi kết nối Ollama. Chạy lệnh `ollama serve` rồi thử lại.")
                elif "timeout" in err:
                    st.error("⏱️ Ollama phản hồi quá chậm.")
                elif "cuda" in err or "gpu" in err:
                    st.error("🖥️ Lỗi GPU/CUDA.")
                else:
                    st.error(f"❌ Lỗi không xác định: {str(e)}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# Xử lý câu hỏi
if prompt:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
        st.stop()

    # ── Tạo session tự động nếu user hỏi mà chưa nhấn New Chat ──────────────
    if st.session_state.current_chat_id is None:
        new_id = create_new_chat_session(prompt[:40] + ("..." if len(prompt) > 40 else ""))
        # Lưu nếu đã có file
        if st.session_state.current_file is not None and st.session_state.vectorstore is not None:
            save_vectorstore(new_id, st.session_state.vectorstore)
            save_history(st.session_state.chat_sessions)
        st.session_state.pending_prompt = prompt
        st.rerun()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Cập nhật title nếu cần
    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id:
            if s["title"] == "Cuộc trò chuyện mới":
                s["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
            break

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và suy nghĩ..."):
            if st.session_state.vectorstore is None:
                response = "Vui lòng upload tài liệu trước khi hỏi."
            else:
                try:
                    logger.info(f"Query: '{prompt[:80]}...'" if len(prompt) > 80 else f"Query: '{prompt}'")  # ← LOG 4
                    if search_type == "mmr":
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": top_k,
                                "lambda_mult": lambda_mult,
                                "fetch_k": fetch_k
                            }
                        )
                    else:
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": top_k, "fetch_k": fetch_k}
                        )
                    retrieved_docs = retriever.invoke(prompt)

                    if search_type == "mmr":
                        logger.info(f"Retriever: MMR | k={top_k} | lambda_mult={lambda_mult} | fetch_k={fetch_k}")
                    else:
                        logger.info(f"Retriever: similarity | k={top_k}")
                    logger.info(
                        f"Retrieved {len(retrieved_docs)} docs | "
                        f"search_type={search_type} | "
                        f"sources: {[doc.metadata.get('source', 'unknown') for doc in retrieved_docs]}"
                    )

                    if not retrieved_docs:
                        response = "Không tìm thấy thông tin liên quan trong tài liệu."
                    else:
                        context = format_docs(retrieved_docs)
                        response = rag_chain.invoke({
                            "context": context,
                            "question": prompt,
                            "language_instruction": get_language_instruction(prompt)
                        }).strip()

                        if "không tìm thấy" in response.lower() or len(response.strip()) < 15:
                            response = "Không tìm thấy thông tin phù hợp trong tài liệu."

                    logger.info(f"Response length: {len(response)} chars")  # ← LOG 6

                except Exception as e:
                    err = str(e).lower()
                    if "connection" in err or "refused" in err:
                        response = "🔌 Mất kết nối đến Ollama. Kiểm tra `ollama serve` và thử lại."
                    elif "timeout" in err:
                        response = "⏱️ Ollama phản hồi quá chậm."
                    else:
                        response = f"❌ Lỗi xử lý câu hỏi: {str(e)}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Lưu sau mỗi lượt trả lời
        save_current_session_to_disk()

        streamlit_js_eval(
            js_expressions="""
                parent.document.querySelectorAll('*').forEach(function(el) {
                    el.scrollTop = el.scrollHeight;
                });
            """,
            key=f"scroll_{int(time.time() * 100000)}"
        )