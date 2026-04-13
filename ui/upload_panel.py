# Khu vực upload file (PDF/DOCX) → 8.2.1, 8.2.8


# ui/upload_panel.py
import os
import tempfile
import streamlit as st
from src.loader import load_and_split
from src.vectorstore import add_to_vectorstore, get_uploaded_sources
from src.config import MAX_FILE_SIZE_MB, OCR_MODE
from src.logger import setup_logger
from datetime import datetime

logger = setup_logger()

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def render_upload_panel(embedder, chunk_size, chunk_overlap, ocr_enabled, create_new_chat_session_fn, save_current_session_fn):
    col_upload, col_chat = st.columns([1, 4])

    with col_upload:
        if ocr_enabled:
            st.info("🔍 Chế độ OCR đang BẬT")

        # Key động để tránh cache file cũ
        uploader_key = (
            f"uploader_{st.session_state.current_chat_id if st.session_state.current_chat_id is not None else 'new'}"
            f"_{st.session_state.uploader_nonce}"
        )
        uploaded_files = st.file_uploader(
            "TẢI TÀI LIỆU (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            # Cập nhật help text
            help="Hỗ trợ cả PDF văn bản và PDF scan (nếu bật OCR).",
            label_visibility="collapsed",
            key=uploader_key
        )

    # ── Xử lý upload nhiều file & indexing ───────────────────────────────────
    if uploaded_files:
        pending_session_title = None
        if st.session_state.current_chat_id is None:
            if len(uploaded_files) == 1:
                pending_session_title = uploaded_files[0].name[:50]
            else:
                pending_session_title = f"{uploaded_files[0].name[:35]} +{len(uploaded_files) - 1} file"

        processed_names = []
        error_messages = []
        total_chunks = 0
        total_size_mb = 0.0

        with st.spinner("Đang xử lý tài liệu..."):


            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_size_mb = len(file_bytes) / (1024 * 1024)

                # Kiểm tra kích thước file
                if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                    error_messages.append(
                        f"{uploaded_file.name}: quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB."
                    )
                    continue

                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                tmp_path = None

                try:
                    # Lưu file tạm để xử lý
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")  # ← LOG 1

                    # Splitting — Chia nhỏ văn bản thành chunks
                    chunks = load_and_split(
                        tmp_path,
                        display_name=uploaded_file.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        ocr_enabled=ocr_enabled
                    )

                    # Nếu chưa có chat session, chỉ tạo khi file đầu tiên xử lý thành công
                    if st.session_state.current_chat_id is None:
                        create_new_chat_session_fn(pending_session_title or uploaded_file.name[:50])

                    # Creating embeddings — Tạo vector & lưu vào vectorstore
                    st.session_state.vectorstore = add_to_vectorstore(
                        st.session_state.vectorstore,
                        chunks,
                        embedder
                    )

                    st.session_state.current_file = uploaded_file.name
                    processed_names.append(uploaded_file.name)
                    total_chunks += len(chunks)
                    total_size_mb += file_size_mb

                    logger.info(f"Vectorstore cập nhật xong: {uploaded_file.name} | chunks={len(chunks)}")  # ← LOG 2

                except ValueError as e:
                    if "FILE_EMPTY" in str(e):
                        error_messages.append(
                            f"{uploaded_file.name}: không đọc được nội dung. File có thể rỗng, là bản scan hoặc chỉ chứa hình ảnh."
                        )
                    else:
                        error_messages.append(f"{uploaded_file.name}: {str(e)}")

                except Exception as e:
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
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # ── Sau khi xử lý xong tất cả file ──────────────────────────────────
        if processed_names:
            # Tạo metadata cho các file vừa upload
            current_metadata = {}
            upload_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            upload_timestamp = datetime.now().isoformat()

            for name in processed_names:
                current_metadata[name] = {
                    "upload_date": upload_time_str,
                    "upload_timestamp": upload_timestamp,
                    "file_type": os.path.splitext(name)[1].lower().replace(".", ""),
                    "size_mb": round(len(uploaded_file.getvalue()) / (1024 * 1024), 2) if 'uploaded_file' in locals() else None
                }

            # Cập nhật metadata vào session hiện tại
            for s in st.session_state.chat_sessions:
                if s["id"] == st.session_state.current_chat_id:
                    if "files_metadata" not in s:
                        s["files_metadata"] = {}
                    # Merge metadata mới (không ghi đè metadata cũ của file khác)
                    s["files_metadata"].update(current_metadata)
                    break

            # === QUY TẮC ĐẶT TIÊU ĐỀ MỚI ===
            if st.session_state.current_chat_id is None:
                # Trường hợp chưa có chat nào → tạo mới với tên file
                if len(processed_names) == 1:
                    session_title = processed_names[0][:50]
                else:
                    session_title = f"{processed_names[0][:35]} +{len(processed_names)-1} file"
                
                create_new_chat_session_fn(
                    title="Cuộc trò chuyện mới",
                    keep_current_context=True,
                    initial_title=session_title
                )
            else:
                # Đã có chat → kiểm tra xem có tin nhắn nào chưa
                current_session = next((s for s in st.session_state.chat_sessions 
                                    if s["id"] == st.session_state.current_chat_id), None)
                
                if current_session and not current_session.get("messages"):
                    # Chưa có tin nhắn nào → đặt tên theo file
                    if len(processed_names) == 1:
                        new_title = processed_names[0][:50]
                    else:
                        new_title = f"{processed_names[0][:35]} +{len(processed_names)-1} file"
                    
                    current_session["title"] = new_title

            # Cập nhật lại tiêu đề nếu đang là mặc định và đã có tin nhắn 
            # (trường hợp chat trước rồi upload thêm)
            for s in st.session_state.chat_sessions:
                if s["id"] == st.session_state.current_chat_id:
                    if s["title"] in ["Cuộc trò chuyện mới", "Cuoc tro chuyen moi"] and s.get("messages"):
                        # Nếu đã có tin nhắn thì giữ nguyên tiêu đề từ tin nhắn đầu
                        pass
                    break

            save_current_session_fn()

            all_sources = get_uploaded_sources(st.session_state.vectorstore)

            st.session_state.upload_success_msg = (
                f"Hoàn tất! Đã xử lý {len(processed_names)} tài liệu ({total_chunks} chunks)."
            )
            st.session_state.upload_info_msg = (
                f"Tài liệu mới: {', '.join(processed_names)} \n  "
                f"Ngày upload: {upload_time_str}\n"
                f"Tổng dung lượng: {total_size_mb:.2f} MB \n"
                f"Số đoạn (chunks): {total_chunks} \n"
                f"Tổng tài liệu đang index: {len(all_sources)}"
            )
            st.session_state.upload_warning_msgs = error_messages
            st.session_state.uploader_nonce += 1
            st.rerun()

        if not processed_names and error_messages:
            st.session_state.upload_warning_msgs = error_messages
            st.session_state.uploader_nonce += 1
            st.rerun()

    return col_chat