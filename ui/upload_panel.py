# ui/upload_panel.py
import os
import tempfile
import streamlit as st
from src.loader import load_and_split
from src.vectorstore import add_to_vectorstore, get_uploaded_sources
from src.config import MAX_FILE_SIZE_MB
from src.logger import setup_logger
from datetime import datetime

logger = setup_logger()

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Chỉ cho phép PDF và DOCX
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def validate_file_format(filename):
    """
    Kiểm tra định dạng file - chỉ cho phép PDF và DOCX.
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"❌ {filename}: Định dạng không được phép. Chỉ chấp nhận PDF (.pdf) hoặc DOCX (.docx)."
    
    return True, None


def render_upload_panel(embedder, chunk_size, chunk_overlap, ocr_enabled,
                        create_new_chat_session_fn, save_current_session_fn):
    
    col_upload, col_chat = st.columns([1, 4])

    with col_upload:
        # Key động để tránh cache file cũ khi rerun
        uploader_key = (
            f"uploader_{st.session_state.current_chat_id if st.session_state.current_chat_id is not None else 'new'}"
            f"_{st.session_state.uploader_nonce}"
        )

        uploaded_files = st.file_uploader(
            "TẢI TÀI LIỆU (PDF/DOCX)",
            accept_multiple_files=True,
            help="Hỗ trợ cả PDF văn bản và PDF scan (nếu bật OCR). Chỉ chấp nhận PDF hoặc DOCX.",
            label_visibility="collapsed",
            key=uploader_key
        )

    # ── Xử lý upload nhiều file & indexing ───────────────────────────────────
    if uploaded_files:
        processed_names = []
        error_messages = []
        total_chunks = 0
        total_size_mb = 0.0
        
        # Kiểm tra định dạng ngay trước khi xử lý (nếu có lỗi format sẽ hiển thị ngay)
        immediate_errors = []
        for uploaded_file in uploaded_files:
            is_valid_format, format_error = validate_file_format(uploaded_file.name)
            if not is_valid_format:
                immediate_errors.append(format_error)
        
        # Hiển thị lỗi định dạng ngay (nếu có)
        if immediate_errors:
            with col_upload:
                st.divider()
                for error in immediate_errors:
                    st.warning(error)
                st.divider()

        with st.spinner("Đang xử lý tài liệu..."):
            for uploaded_file in uploaded_files:
                # ── Bước 1: Kiểm tra định dạng file ──
                # (đã kiểm tra ở trên, bỏ qua file invalid)
                is_valid_format, _ = validate_file_format(uploaded_file.name)
                if not is_valid_format:
                    logger.warning(f"Invalid format rejected: {uploaded_file.name}")
                    continue
                
                file_bytes = uploaded_file.getvalue()
                file_size_mb = len(file_bytes) / (1024 * 1024)

                # ── Bước 2: Kiểm tra kích thước file ──
                if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                    error_messages.append(
                        f"{uploaded_file.name}: quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB."
                    )
                    continue

                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                tmp_path = None

                try:
                    # Lưu file tạm
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")
                    logger.info(f"Chunk Size: {chunk_size} | Chunk Overlap: {chunk_overlap}")
                    
                    # ── Bước 3: Splitting ──
                    with st.status(f"📄 Xử lý: {uploaded_file.name}") as status:
                        status.update(label=f"🔄 Splitting...", state="running")
                        
                        # Load và split tài liệu
                        chunks = load_and_split(
                            tmp_path,
                            display_name=uploaded_file.name,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            ocr_enabled=ocr_enabled
                        )
                        
                        status.update(label=f"✅ Splitting: {len(chunks)} chunks", state="running")
                        logger.info(f"Split xong: {uploaded_file.name} | chunks={len(chunks)}")

                        # Tạo session mới nếu chưa có
                        if st.session_state.current_chat_id is None:
                            session_title = (
                                uploaded_file.name[:50] if len(uploaded_files) == 1
                                else f"{uploaded_files[0].name[:35]} +{len(uploaded_files)-1} file"
                            )
                            create_new_chat_session_fn(
                                title="Cuộc trò chuyện mới",
                                keep_current_context=True,
                                initial_title=session_title
                            )

                        # ── Bước 4: Creating embeddings ──
                        status.update(label=f"🧠 Creating embeddings...", state="running")
                        
                        # Thêm vào vectorstore
                        st.session_state.vectorstore = add_to_vectorstore(
                            st.session_state.vectorstore, chunks, embedder
                        )
                        
                        status.update(label=f"✅ Done!", state="complete")
                        logger.info(f"Embeddings tạo xong: {uploaded_file.name}")

                    st.session_state.current_file = uploaded_file.name
                    processed_names.append(uploaded_file.name)
                    total_chunks += len(chunks)
                    total_size_mb += file_size_mb

                    logger.info(f"Vectorstore cập nhật xong: {uploaded_file.name} | chunks={len(chunks)}")
                    logger.info(f"Tổng tài liệu đang index:{len(get_uploaded_sources(st.session_state.vectorstore))}")

                except ValueError as e:
                    if "FILE_EMPTY" in str(e):
                        error_messages.append(
                            f"{uploaded_file.name}: không đọc được nội dung. "
                            f"File có thể rỗng, là bản scan thuần hoặc chỉ chứa hình ảnh."
                        )
                    else:
                        error_messages.append(f"{uploaded_file.name}: {str(e)}")

                except Exception as e:
                    # Chỉ bắt lỗi liên quan đến upload/loader/OCR, KHÔNG bắt lỗi Ollama
                    err_lower = str(e).lower()
                    if "poppler" in err_lower or "pdf" in err_lower or "ocr" in err_lower:
                        error_messages.append(f"❌ {uploaded_file.name}: Lỗi xử lý file PDF/OCR - {str(e)}")
                    else:
                        error_messages.append(f"❌ {uploaded_file.name}: {str(e)}")

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # ── Sau khi xử lý xong ───────────────────────────────────────────────
        # Gộp tất cả lỗi (format + size + processing)
        all_errors = immediate_errors + error_messages
        if processed_names:
            # Tạo metadata
            upload_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            upload_timestamp = datetime.now().isoformat()

            current_metadata = {}
            for name in processed_names:
                current_metadata[name] = {
                    "upload_date": upload_time_str,
                    "upload_timestamp": upload_timestamp,
                    "file_type": os.path.splitext(name)[1].lower().replace(".", ""),
                    "size_mb": round([f for f in uploaded_files if f.name == name][0].size / (1024*1024), 2)
                    if any(f.name == name for f in uploaded_files) else None
                }

            # Cập nhật metadata vào session hiện tại
            for s in st.session_state.chat_sessions:
                if s["id"] == st.session_state.current_chat_id:
                    if "files_metadata" not in s:
                        s["files_metadata"] = {}
                    s["files_metadata"].update(current_metadata)
                    break

            # === Đặt tiêu đề session ===
            if st.session_state.current_chat_id is not None:
                current_session = next((s for s in st.session_state.chat_sessions 
                                      if s["id"] == st.session_state.current_chat_id), None)
                
                if current_session and not current_session.get("messages"):
                    # Chưa có tin nhắn → đặt tên theo file
                    if len(processed_names) == 1:
                        new_title = processed_names[0][:50]
                    else:
                        new_title = f"{processed_names[0][:35]} +{len(processed_names)-1} file"
                    current_session["title"] = new_title

            save_current_session_fn()

            all_sources = get_uploaded_sources(st.session_state.vectorstore)

            # Lưu thông báo thành công
            st.session_state.upload_success_msg = (
                f"✅ Hoàn tất! Đã xử lý {len(processed_names)} tài liệu ({total_chunks} chunks)."
            )
            st.session_state.upload_info_msg = (
                f"Tài liệu mới: {', '.join(processed_names)}\n"
                f"Ngày upload: {upload_time_str}\n"
                f"Tổng dung lượng: {total_size_mb:.2f} MB\n"
                f"Số đoạn (chunks): {total_chunks}\n"
                f"Tổng tài liệu đang index: {len(all_sources)}"
            )
            st.session_state.upload_warning_msgs = all_errors
            st.session_state.uploader_nonce += 1
            st.rerun()

        # Trường hợp không xử lý được file nào nhưng có lỗi
        elif all_errors:
            st.session_state.upload_warning_msgs = all_errors
            st.session_state.uploader_nonce += 1
            st.rerun()

    return col_chat