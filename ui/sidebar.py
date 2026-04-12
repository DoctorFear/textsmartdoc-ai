# Sidebar: lịch sử, settings, clear buttons → 8.2.2, 8.2.3
# ui/sidebar.py
import streamlit as st
from src.vectorstore import get_uploaded_sources
from src.persistence import (
    save_history, load_history,
    save_vectorstore, load_vectorstore,
    delete_all_vectorstores, delete_vectorstore
)
from src.config import SIDEBAR_TEXT_COLOR
from src.logger import setup_logger
from ui.settings_panel import render_model_info, render_settings_panel

logger = setup_logger()


# ── Helpers (cũng được import bởi upload_panel, chat_panel) ──────────────────

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
        if not (
            s["title"] == "Cuộc trò chuyện mới"
            and not s["messages"]
            and not s.get("file")
            and not s.get("files")
        )
    ]
    save_history(st.session_state.chat_sessions)


def load_session_to_state(target_id, embedder):
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
        st.session_state.current_chat_id = target_id
        st.session_state.messages = []
        st.session_state.current_file = None
        logger.warning(f"Không tìm thấy chat {target_id} trên disk, reset state")

    # Luôn load vectorstore từ disk (quan trọng nhất để hỏi đáp đúng tài liệu)
    st.session_state.vectorstore = load_vectorstore(target_id, embedder)

    #if st.session_state.vectorstore is None and session.get("files"): FIX 8.2.2
    #st.warning(f"⚠️ Không thể load vectorstore cho chat này. Tài liệu có thể bị xóa hoặc lỗi index.") FIX 8.2.2


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


# ── Render sidebar ────────────────────────────────────────────────────────────

def render_sidebar(embedder) -> dict:
    """
    Render toàn bộ sidebar.
    Settings UI đã tách sang ui/settings_panel.py.
    Returns:
        dict settings từ render_settings_panel()
    """
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{SIDEBAR_TEXT_COLOR};'>SmartDoc AI</h2>",
            unsafe_allow_html=True
        )

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

        # Model info (tách sang settings_panel.py)
        render_model_info()

        # Settings sliders/selects (tách sang settings_panel.py)
        settings = render_settings_panel()

        # ── Tài liệu đã upload + filter (8.2.8) ──────────────────────────────
        with st.expander("Tài liệu đã upload", expanded=True):
            uploaded_sources = get_uploaded_sources(st.session_state.vectorstore)
            if uploaded_sources:
                st.markdown(
                    f"<div style='background-color:#FFFFFF; color:#31333F; border-radius:8px; "
                    f"padding:8px 12px; font-size:0.875rem;'>"
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
                    "<div style='background-color:#FFFFFF; color:#31333F; border-radius:8px; "
                    "padding:8px 12px; font-size:0.875rem;'>"
                    "Chưa có tài liệu nào được upload."
                    "</div>",
                    unsafe_allow_html=True
                )

        # ── Lịch sử cuộc trò chuyện ──────────────────────────────────────────
        st.subheader("Lịch sử cuộc trò chuyện")

        # Nút xóa tất cả lịch sử (8.2.3)
        if st.button("🔄 Đặt lại lịch sử", type="tertiary", use_container_width=True):
            if len(st.session_state.chat_sessions) > 0:
                st.session_state.show_confirm = True

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

        # Nút Clear Vector Store (8.2.3)
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
                    # === RESET TRIỆT ĐỂ session_state ===
                    st.session_state.vectorstore = None
                    st.session_state.current_file = None
                    
                    # Xóa cache BM25 (rất quan trọng)
                    if "bm25_retriever" in st.session_state:
                        st.session_state.pop("bm25_retriever", None)
                    if "bm25_doc_count" in st.session_state:
                        st.session_state.pop("bm25_doc_count", None)

                    # Reset source filter nếu có
                    if "source_filter_select" in st.session_state:
                        st.session_state.source_filter_select = "Tất cả tài liệu"

                    # Lưu lại history
                    save_history(st.session_state.chat_sessions)

                    st.session_state.show_confirm_clear_vs = False
                    
                    st.success("✅ Đã xóa tài liệu và Vector Store thành công!")
                    st.rerun()
            with col2:
                if st.button("❌ Hủy", key="confirm_clear_vs_no"):
                    st.session_state.show_confirm_clear_vs = False
                    st.rerun()

        # Nút New Chat
        if st.button("✨ Hộp thoại mới", type="secondary", use_container_width=True):
            # Chống spam: nếu đang ở session rỗng thì không tạo thêm
            if len(st.session_state.messages) == 0 and st.session_state.current_chat_id is not None:
                st.rerun()
            save_current_session_to_disk()
            create_new_chat_session("Cuộc trò chuyện mới")
            st.rerun()

        # Danh sách các cuộc chat — mới nhất lên đầu
        for session in reversed(st.session_state.chat_sessions):
            title = session["title"]
            # is_active = session["id"] == st.session_state.current_chat_id fix 8.2.2s
            if st.button(
                f"📩 {title}",
                key=f"chat_{session['id']}",
                use_container_width=True,
                type="primary"
                # type="primary" if is_active else "secondary" fix 8.2.2
            ):
                save_current_session_to_disk()
                load_session_to_state(session["id"], embedder)
                st.rerun()


                    

    return settings