# app.py
import streamlit as st
import os
import tempfile
from datetime import datetime
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import add_to_vectorstore, create_vectorstore, get_uploaded_sources
from src.rag_chain import get_language_instruction, llm, format_docs, rag_chain, self_rag_query
from streamlit_js_eval import streamlit_js_eval
from src.conversational import rewrite_with_history
from src.reranker import rerank
import time


from src.logger import setup_logger
# ── Import persistence (lưu/load lịch sử + FAISS ra disk) ───────────────────
from src.persistence import (
    save_history, load_history,
    save_vectorstore, load_vectorstore,
    delete_all_vectorstores, delete_vectorstore
)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

logger = setup_logger()


# ── Load CSS từ file riêng ───────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ── Theme & Color Palette ────────────────────────────────────────────────────
TEXT_SIDEBAR = "#FFFFFF"  # Màu chữ trắng cho sidebar

st.set_page_config(  # Cấu hình giao diện
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
    st.session_state.vectorstore = None  # nơi chứa FAISS index của tài liệu.
if "messages" not in st.session_state:
    st.session_state.messages = []  # lịch sử hội thoại.
if "current_file" not in st.session_state:
    st.session_state.current_file = None  # tên file PDF hiện tại.
# ── Session State bổ sung cho chat sessions + persistence (tính năng lịch sử)───────────────────
if "chat_sessions" not in st.session_state:
    sessions = load_history()
    sessions = [
        s for s in sessions
        if not (s["title"] == "Cuộc trò chuyện mới" and not s["messages"] and not s["file"])
    ]
    st.session_state.chat_sessions = sessions
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  # lưu id của đoạn chat đang hoạt động
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # giữ câu hỏi khi qua rerun để hiển thị tên của nó ở history
# Session State bổ sung cho upload feedback
if "upload_success_msg" not in st.session_state:
    st.session_state.upload_success_msg = None
if "upload_info_msg" not in st.session_state:
    st.session_state.upload_info_msg = None
if "upload_warning_msgs" not in st.session_state:
    st.session_state.upload_warning_msgs = []
if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0

# Trạng thái bật/tắt Self-RAG (8.2.10)
if "self_rag_enabled" not in st.session_state:
    st.session_state.self_rag_enabled = False  # mặc định tắt để tiết kiệm tài nguyên


# ── Hàm hỗ trợ tách biệt các đoạn lặp ───────────────────────────────────────

def save_current_session_to_disk():
    """Lưu messages, file, vectorstore của chat hiện tại xuống disk"""
    if st.session_state.current_chat_id is None:
        return

    uploaded_sources = get_uploaded_sources(st.session_state.vectorstore)

    for s in st.session_state.chat_sessions:
        if s["id"] == st.session_state.current_chat_id:
            s["messages"] = st.session_state.messages.copy()
            s["file"] = st.session_state.current_file
            s["files"] = uploaded_sources
            if st.session_state.vectorstore is not None:
                save_vectorstore(s["id"], st.session_state.vectorstore)
            break

    # Lọc bỏ các session rỗng trước khi lưu
    st.session_state.chat_sessions = [
        s for s in st.session_state.chat_sessions
        if not (s["title"] == "Cuộc trò chuyện mới" and not s["messages"] and not s.get("file") and not s.get("files"))
    ]
    save_history(st.session_state.chat_sessions)


def load_session_to_state(target_id):
    """Load đúng chat mà người dùng vừa bấm vào từ disk"""
    fresh_sessions = load_history()
    st.session_state.chat_sessions = fresh_sessions
    target = next((s for s in fresh_sessions if s["id"] == target_id), None)

    if target:
        st.session_state.current_chat_id = target_id
        st.session_state.messages = target["messages"].copy()
        files = target.get("files", [])
        st.session_state.current_file = target.get("file") or (files[-1] if files else None)
    else:
        # Nếu không có trên disk → giữ nguyên state hiện tại hoặc reset
        st.session_state.current_chat_id = target_id
        st.session_state.messages = []
        st.session_state.current_file = None
        logger.warning(f"Không tìm thấy chat {target_id} trên disk, reset state")

    # Luôn load vectorstore từ disk (quan trọng nhất để hỏi đáp đúng tài liệu)
    st.session_state.vectorstore = load_vectorstore(target_id, get_embedder())


def create_new_chat_session(title="Cuoc tro chuyen moi", keep_current_context=False):
    """Tạo session chat mới và cập nhật state"""
    new_id = max([s["id"] for s in st.session_state.chat_sessions], default=-1) + 1
    st.session_state.chat_sessions.append({
        "id": new_id,
        "title": title,
        "messages": [],
        "vectorstore": None,
        "file": None,
        "files": []
    })
    st.session_state.current_chat_id = new_id
    if not keep_current_context:
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.current_file = None
    return new_id


def render_self_rag_meta(meta: dict):
    """Render khối thông tin Self-RAG gọn, đều dòng và dễ đọc hơn."""
    if not meta:
        return

    attempts = meta.get("attempts", 1)
    confidence = meta.get("confidence", "?")
    query_used = meta.get("query_used", "")
    reason = meta.get("evaluation", {}).get("reason", "")

    blocks = [
        f"<div><strong>Số lần thử:</strong> {attempts} | <strong>Độ tin cậy:</strong> {confidence}/10</div>"
    ]

    if attempts > 1 and query_used:
        blocks.append(
            f"<div><strong>Câu hỏi được viết lại:</strong> <em>{query_used}</em></div>"
        )

    if reason:
        blocks.append(f"<div><strong>Lý do đánh giá:</strong> {reason}</div>")

    st.markdown(
        f"<div class='self-rag-meta'>{''.join(blocks)}</div>",
        unsafe_allow_html=True
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT_SIDEBAR};'>SmartDoc AI</h2>", unsafe_allow_html=True)

    # Instructions section
    with st.expander("Hướng dẫn sử dụng"):
        st.markdown("""
        **Các bước hoạt động** 
        1. Tải lên file tài liệu ở phần chính giữa  
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

        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["faiss", "bm25", "hybrid"],
            index=2  # default hybrid
        )
        use_reranker = st.checkbox(
            "Use Cross-Encoder Reranking",
            value=True
        )

        # Toggle bật/tắt chế độ Self-RAG (8.2.10)
        st.divider()
        st.session_state.self_rag_enabled = st.checkbox(
            "Bật Self-RAG",
            value=st.session_state.self_rag_enabled,
            help="Self-RAG cho phép tự đánh giá và cải thiện câu trả lời. Chậm hơn nhưng chính xác hơn."
        )

    # ── Hiển thị danh sách tài liệu đã upload + filter (8.2.8) ────────────────────────────────────
    with st.expander("Tài liệu đã upload", expanded=True):
        uploaded_sources = get_uploaded_sources(st.session_state.vectorstore)
        if uploaded_sources:
            st.markdown(
                f"<div style='background-color:#FFFFFF; color:#31333F; border-radius:8px; padding:8px 12px; font-size:0.875rem;'>"
                f"Đang index: <b>{len(uploaded_sources)}</b> tài liệu"
                f"</div>",
                unsafe_allow_html=True
            )

            for src in uploaded_sources:
                st.caption(f"• {src}")

            source_options = ["Tất cả tài liệu"] + uploaded_sources
            st.selectbox(
                "Tìm kiếm trong:",
                options=source_options,
                index=0,
                key="source_filter_select"
            )
        else:
            st.markdown(
                "<div style='background-color:#FFFFFF; color:#31333F; border-radius:8px; padding:8px 12px; font-size:0.875rem;'>"
                "Chưa có tài liệu nào được upload."
                "</div>",
                unsafe_allow_html=True
            )

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
                st.session_state.chat_sessions = []
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                st.session_state.vectorstore = None
                st.session_state.current_file = None
                save_history([])
                st.session_state.show_confirm = False
                st.rerun()
        with col2:
            if st.button("❌ Hủy"):
                st.session_state.show_confirm = False
                st.rerun()

        # ── Nút Clear Vector Store ─────────────────────────────────────────────
    if st.session_state.vectorstore is not None and st.session_state.current_chat_id is not None:
        if st.button("🗑️ Xóa tài liệu hiện tại", type="tertiary", use_container_width=True, key="quart_btn"):
            st.session_state.show_confirm_clear_vs = True

    if st.session_state.get("show_confirm_clear_vs", False):
        st.caption("**Bạn có chắc chắn muốn xóa tài liệu hiện tại không?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Đồng ý", key="confirm_clear_vs_yes"):
                delete_vectorstore(st.session_state.current_chat_id)
                for s in st.session_state.chat_sessions:
                    if s["id"] == st.session_state.current_chat_id:
                        s["file"] = None
                        s["files"] = []
                        break
                save_history(st.session_state.chat_sessions)
                st.session_state.vectorstore = None
                st.session_state.current_file = None
                st.session_state.show_confirm_clear_vs = False
                st.rerun()
        with col2:
            if st.button("❌ Hủy", key="confirm_clear_vs_no"):
                st.session_state.show_confirm_clear_vs = False
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
        title = session["title"]
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
st.title("📄 Hỏi đáp thông minh với tài liệu")
st.markdown(
    "**SmartDoc AI** – Upload PDF hoặc DOCX → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")

current_sources = get_uploaded_sources(st.session_state.vectorstore)

if st.session_state.vectorstore is None:
    st.info("Vui lòng tải file PDF hoặc DOCX lên ở ô bên trái (hàng dưới) để bắt đầu hỏi đáp.")
else:
    if len(current_sources) <= 1:
        st.success(f"Đang làm việc với tài liệu: **{st.session_state.current_file}**")
    else:
        st.success(f"Đang làm việc với **{len(current_sources)} tài liệu** trong cùng 1 index.")
        st.caption(" | ".join(current_sources))

# Hiển thị thông báo upload thành công sau rerun (8.2.8)
if st.session_state.get("upload_success_msg"):
    st.success(st.session_state.upload_success_msg)
    st.session_state.upload_success_msg = None
if st.session_state.get("upload_info_msg"):
    st.info(st.session_state.upload_info_msg)
    st.session_state.upload_info_msg = None
if st.session_state.get("upload_warning_msgs"):
    for msg in st.session_state.upload_warning_msgs:
        st.warning(msg)
    st.session_state.upload_warning_msgs = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if "self_rag_meta" in message and message["self_rag_meta"]:
        meta = message["self_rag_meta"]
        with st.expander("🔁 Thông tin Self-RAG"):
            render_self_rag_meta(meta)

    if "citations" in message and message["citations"]:
        with st.expander(" Xem nguồn trích dẫn (Citations) & Highlight"):
            for cite in message["citations"]:
                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                st.info(f'"{cite["snippet"]}"')

# ── Input hỏi đáp + File uploader ngang hàng ────────────────────────────────
col_upload, col_chat = st.columns([1, 4])

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

with col_upload:
    # Key động để tránh cache file cũ khi chuyển chat
    uploader_key = (
        f"uploader_{st.session_state.current_chat_id if st.session_state.current_chat_id is not None else 'new'}"
        f"_{st.session_state.uploader_nonce}"
    )
    uploaded_file = st.file_uploader(
        "TẢI TÀI LIỆU (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Chỉ hỗ trợ tài liệu có text layer (không phải scan/image-only). Tối đa ~50MB.",
        label_visibility="collapsed",
        key=uploader_key
    )

with col_chat:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...")  # ô nhập liệu

if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ── Xử lý upload nhiều file & indexing ─────────────────────────────────────────
if uploaded_file:
    # Nếu chưa có chat session → tạo mới dựa trên tên file
    pending_session_title = None
    if st.session_state.current_chat_id is None:
        if len(uploaded_file) == 1:
            pending_session_title = uploaded_file[0].name[:50]
        else:
            pending_session_title = f"{uploaded_file[0].name[:35]} +{len(uploaded_file) - 1} file"

    processed_names = []  # Danh sách file xử lý thành công
    error_messages = []  # Danh sách lỗi
    total_chunks = 0  # Tổng số đoạn (chunks)
    total_size_mb = 0.0  # Tổng dung lượng (MB)

    with st.spinner("Đang xử lý tài liệu..."):
        # ── Lặp qua từng file upload ───────────────────────────────────────────
        for uploaded_file in uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            # Kiểm tra kích thước file
            if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                error_messages.append(
                    f"{uploaded_file.name}: quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB."
                )
                continue

            # Lấy đuôi file (.pdf, .docx)
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            tmp_path = None

            try:
                # ── Lưu file tạm để xử lý ──────────────────────────────────────
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")  # ← LOG 1

                # ── Splitting — Chia nhỏ văn bản thành chunks ─────────────────
                chunks = load_and_split(
                    tmp_path,
                    display_name=uploaded_file.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Nếu chua co chat session, chi tao khi file dau tien da xu ly thanh cong
                if st.session_state.current_chat_id is None:
                    create_new_chat_session(pending_session_title or uploaded_file.name[:50])

                # ── Creating embeddings — Tạo vector & lưu vào vectorstore ────
                st.session_state.vectorstore = add_to_vectorstore(
                    st.session_state.vectorstore,
                    chunks,
                    get_embedder()
                )

                # Cập nhật thông tin sau khi xử lý thành công
                st.session_state.current_file = uploaded_file.name
                processed_names.append(uploaded_file.name)
                total_chunks += len(chunks)
                total_size_mb += file_size_mb

                logger.info(
                    f"Vectorstore cập nhật xong: {uploaded_file.name} | chunks={len(chunks)}"
                )  # ← LOG 2

            except ValueError as e:
                # ── Lỗi dữ liệu file (rỗng, scan, không đọc được) ─────────────
                if "FILE_EMPTY" in str(e):
                    error_messages.append(
                        f"{uploaded_file.name}: không đọc được nội dung. File có thể rỗng, là bản scan hoặc chỉ chứa hình ảnh."
                    )
                else:
                    error_messages.append(f"{uploaded_file.name}: {str(e)}")

            except Exception as e:
                # ── Lỗi hệ thống / Ollama / GPU ─────────
                err = str(e).lower()
                if "connection" in err or "refused" in err or "ollama" in err:
                    error_messages.append(f"🔌 {uploaded_file.name}: lỗi kết nối Ollama.")
                elif "timeout" in err:
                    error_messages.append(f"⏱️ {uploaded_file.name}: Ollama phản hồi quá chậm.")
                elif "cuda" in err or "gpu" in err:
                    error_messages.append(f"💻 {uploaded_file.name}: lỗi GPU/CUDA.")
                else:
                    error_messages.append(f"❌ {uploaded_file.name}: {str(e)}")

            finally:
                # ── Xóa file tạm sau khi xử lý ────────────────────────────────
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # ── Sau khi xử lý xong tất cả file ─────────────────────────────────────────
    if processed_names:
        # Cập nhật lại tiêu đề chat nếu đang là mặc định
        for s in st.session_state.chat_sessions:
            if s["id"] == st.session_state.current_chat_id and s["title"] == "Cuộc trò chuyện mới":
                if len(processed_names) == 1:
                    s["title"] = processed_names[0][:50]
                else:
                    s["title"] = f"{processed_names[0][:35]} +{len(processed_names) - 1} file"
                break

        # Lưu session hiện tại
        save_current_session_to_disk()

        # Tổng hợp thông tin hiển thị
        all_sources = get_uploaded_sources(st.session_state.vectorstore)

        st.session_state.upload_success_msg = (
            f"Hoàn tất! Đã xử lý {len(processed_names)} tài liệu ({total_chunks} chunks)."
        )
        st.session_state.upload_info_msg = (
            f"📄 Tài liệu mới: {', '.join(processed_names)}  \n"
            f"📦 Tổng dung lượng: {total_size_mb:.2f} MB  \n"
            f"🔢 Số đoạn (chunks): {total_chunks}  \n"
            f"📚 Tổng tài liệu đang index: {len(all_sources)}"
        )

        # Lưu cảnh báo (nếu có)
        st.session_state.upload_warning_msgs = error_messages

        # Reset uploader (tránh giữ file cũ)
        st.session_state.uploader_nonce += 1

        st.rerun()

    if not processed_names and error_messages:
        st.session_state.upload_warning_msgs = error_messages
        st.session_state.uploader_nonce += 1
        st.rerun()

# Xử lý câu hỏi
if prompt:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Vui lòng upload tài liệu trước khi hỏi.")
        st.stop()

    # ── Tạo session tự động nếu user hỏi mà chưa nhấn New Chat ──────────────
    if st.session_state.current_chat_id is None:
        create_new_chat_session(
            prompt[:40] + ("..." if len(prompt) > 40 else ""),
            keep_current_context=True
        )
        save_current_session_to_disk()
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
                citations = []
                self_rag_meta = None
            else:
                try:
                    logger.info(f"Query: '{prompt[:80]}...'" if len(prompt) > 80 else f"Query: '{prompt}'")  # ← LOG 4

                    # Áp dụng filter tài liệu nếu người dùng chọn (8.2.8)
                    source_filter = None
                    if "source_filter_select" in st.session_state:
                        selected = st.session_state.source_filter_select
                        if selected != "Tất cả tài liệu":
                            source_filter = selected

                    # Cấu hình retriever với filter (nếu có)
                    retriever_kwargs = {"k": top_k, "fetch_k": fetch_k}
                    if source_filter:
                        retriever_kwargs["filter"] = {"source": source_filter}
                    if search_type == "mmr":
                        retriever_kwargs["lambda_mult"] = lambda_mult

                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type=search_type,
                        search_kwargs=retriever_kwargs
                    )

                    all_docs = list(st.session_state.vectorstore.docstore._dict.values())
                    current_doc_count = len(all_docs)
                    ## Tạo BM25 (nếu chưa có)
                    if "bm25_retriever" not in st.session_state or st.session_state.get(
                            "bm25_doc_count") != current_doc_count:
                        logger.info("Đang tạo index BM25 (Chỉ chạy 1 lần duy nhất)...")
                        st.session_state.bm25_retriever = BM25Retriever.from_documents(all_docs)
                        st.session_state.bm25_doc_count = current_doc_count

                    bm25_retriever = st.session_state.bm25_retriever
                    bm25_retriever.k = top_k
                    ## implement ensemble retriever
                    retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, retriever],
                        weights=[0.3, 0.7]
                    )
                    if search_type == "mmr":
                        logger.info(f"Retriever: MMR | k={top_k} | lambda_mult={lambda_mult} | fetch_k={fetch_k}")
                    else:
                        logger.info(f"Retriever: similarity | k={top_k}")

                    self_rag_meta = None
                    if st.session_state.self_rag_enabled:
                        # Chế độ Self-RAG: tự đánh giá và retry nếu cần
                        result = self_rag_query(prompt, retriever, max_retries=2)
                        response = result["answer"]
                        retrieved_docs = result["docs"]
                        self_rag_meta = {
                            "attempts": result["attempts"],
                            "confidence": result["confidence"],
                            "query_used": result["query_used"],
                            "evaluation": result["evaluation"]
                        }
                        logger.info(
                            f"Self-RAG | attempts={result['attempts']} | "
                            f"confidence={result['confidence']} | "
                            f"query_used='{result['query_used'][:60]}'"
                        )
                    else:
                        # Chế độ thường: pipeline RAG chuẩn
                        
                        # retrieved_docs = retriever.invoke(prompt)
                        # testing conversational
                        query_for_retrieval = rewrite_with_history(
                            prompt,
                            st.session_state.messages
                        )
                        # Dense retriever (FAISS)
                        faiss_retriever = st.session_state.vectorstore.as_retriever(
                            search_type=search_type,
                            search_kwargs=retriever_kwargs
                        )
                        # BM25 setup (already cached)
                        bm25_retriever = st.session_state.bm25_retriever
                        bm25_retriever.k = top_k
                        # Switch logic
                        if retrieval_mode == "faiss":
                            retriever = faiss_retriever
                        elif retrieval_mode == "bm25":
                            retriever = bm25_retriever
                        elif retrieval_mode == "hybrid":
                            retriever = EnsembleRetriever(
                                retrievers=[bm25_retriever, faiss_retriever],
                                weights=[0.3, 0.7]
                            )
                        retrieved_docs = retriever.invoke(query_for_retrieval)
                        if use_reranker:
                            retrieved_docs = rerank(
                                query_for_retrieval,
                                retrieved_docs,
                                top_k=top_k
                            )


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

                    # Tạo citations từ retrieved_docs
                    citations = []
                    for i, doc in enumerate(retrieved_docs):
                        page_num = doc.metadata.get("page", -1) + 1

                        raw_source = doc.metadata.get("source", "Unknown")
                        file_name = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown document"

                        snippet = doc.page_content[:250].replace('\n', ' ') + "..."

                        citations.append({
                            "index": i + 1,
                            "page": page_num if page_num > 0 else "N/A",
                            "file": file_name,
                            "snippet": snippet
                        })

                    logger.info(f"Response length: {len(response)} chars")  # ← LOG 6

                except Exception as e:
                    err = str(e).lower()
                    if "connection" in err or "refused" in err:
                        response = "🔌 Mất kết nối đến Ollama. Kiểm tra `ollama serve` và thử lại."
                    elif "timeout" in err:
                        response = "⏱️ Ollama phản hồi quá chậm."
                    else:
                        response = f"❌ Lỗi xử lý câu hỏi: {str(e)}"
                    citations = []
                    self_rag_meta = None

        st.markdown(response)

    # Hiển thị thông tin Self-RAG nếu có ngay sau câu trả lời (8.2.10)
    if self_rag_meta:
        with st.expander("🔁 Thông tin Self-RAG"):
            render_self_rag_meta(self_rag_meta)

    if citations:
        with st.expander("Xem nguồn trích dẫn (Citations) & Highlight"):
            st.markdown("Hệ thống đã dựa các đoạn văn bản sau để tạo câu trả lời:")

            for cite in citations:
                st.markdown(f"**Nguồn {cite['index']}:** File `{cite['file']}` — Trang **{cite['page']}**")
                st.info(f'"{cite["snippet"]}"')

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "citations": citations,
        "self_rag_meta": self_rag_meta  # None nếu không bật Self-RAG
    })

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
