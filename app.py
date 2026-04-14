# app.py — Entry point. Chỉ khởi tạo, không chứa logic.
import streamlit as st
from src.embedder import embedder
from src.rag_chain import llm
from src.logger import setup_logger
from src.persistence import load_history
from src.vectorstore import get_uploaded_sources
from src.config import SELF_RAG_ENABLED


from ui.sidebar import render_sidebar, save_current_session_to_disk, create_new_chat_session
from ui.upload_panel import render_upload_panel
from ui.chat_panel import render_chat_history, handle_query

logger = setup_logger()


# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css("styles.css")


# ── Caching ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return embedder

@st.cache_resource
def get_llm():
    return llm


# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "vectorstore":         None,
    "messages":            [],
    "current_file":        None,
    "current_chat_id":     None,
    "pending_prompt":      None,
    "upload_success_msg":  None,
    "upload_info_msg":     None,
    "upload_warning_msgs": [],
    "uploader_nonce":      0,
    "self_rag_enabled":    SELF_RAG_ENABLED,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "chat_sessions" not in st.session_state:
    sessions = load_history()
    st.session_state.chat_sessions = [
        s for s in sessions
        if not (s["title"] == "Cuộc trò chuyện mới" and not s["messages"] and not s["file"])
    ]


# ── Sidebar (settings + lịch sử) ─────────────────────────────────────────────
settings = render_sidebar(get_embedder())


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("📄 Hỏi đáp thông minh với tài liệu")
st.markdown(
    "**SmartDoc AI** – Upload PDF hoặc DOCX → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. "
    "Hệ thống chạy cục bộ, bảo mật cao."
)

current_sources = get_uploaded_sources(st.session_state.vectorstore)

if st.session_state.vectorstore is None:
    st.info("Vui lòng tải file PDF hoặc DOCX lên ở ô bên trái để bắt đầu hỏi đáp.")

else:
    # Lấy session hiện tại
    session = next((s for s in st.session_state.chat_sessions 
                   if s["id"] == st.session_state.current_chat_id), None)
    
    files_meta = session.get("files_metadata", {}) if session else {}

    if len(current_sources) <= 1:
        file_name = st.session_state.current_file or (current_sources[0] if current_sources else "")
        date = files_meta.get(file_name, {}).get("upload_date", "N/A")
        st.success(f"Đang làm việc với tài liệu: **{file_name}**")
        st.caption(f"Ngày upload: {date}")
    else:
        st.success(f"Đang làm việc với **{len(current_sources)} tài liệu** trong cùng phiên.")
        
        caption = " | ".join(
            f"**{src}** ({files_meta.get(src, {}).get('upload_date', 'N/A')})"
            for src in current_sources
        )
        st.caption(caption)

# Thông báo upload sau rerun
if st.session_state.upload_success_msg:
    st.success(st.session_state.upload_success_msg)
    st.session_state.upload_success_msg = None
if st.session_state.upload_info_msg:
    st.info(st.session_state.upload_info_msg)
    st.session_state.upload_info_msg = None
if st.session_state.upload_warning_msgs:
    for msg in st.session_state.upload_warning_msgs:
        st.warning(msg)
    st.session_state.upload_warning_msgs = []

# Lịch sử chat
render_chat_history()

# Upload + chat input ngang hàng
col_chat = render_upload_panel(
    embedder=get_embedder(),
    chunk_size=settings["chunk_size"],
    chunk_overlap=settings["chunk_overlap"],
    ocr_enabled=settings["ocr_enabled"],
    create_new_chat_session_fn=create_new_chat_session,
    save_current_session_fn=save_current_session_to_disk,
)

with col_chat:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...")

if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# Xử lý câu hỏi
if prompt:
    handle_query(
        prompt=prompt,
        settings=settings,
        save_current_session_fn=save_current_session_to_disk,
        create_new_chat_session_fn=create_new_chat_session,
    )