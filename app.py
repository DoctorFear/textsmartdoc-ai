# app.py
import streamlit as st
import os
import tempfile
from datetime import datetime
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
from src.rag_chain import llm, format_docs, rag_chain
from streamlit_js_eval import streamlit_js_eval
import time
from src.logger import setup_logger
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

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{TEXT_SIDEBAR};'>SmartDoc AI</h2>", unsafe_allow_html=True)

    if st.button("Xóa tài liệu & Reset", type="primary", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.rerun()

    # Instructions section
    with st.expander("Hướng dẫn sử dụng"):
        st.markdown("""
        1. Tải lên file PDF ở phần chính giữa  
        2. Chờ xử lý xong (thấy thông báo xanh)  
        3. Đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh  
        4. Hệ thống chỉ trả lời dựa trên nội dung tài liệu
        5. **Lưu ý**: Tùy chỉnh Chunk size, Chunk overlap trước khi upload tài liệu
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
        chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=1200, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        top_k = st.slider("Search Kwargs - Top-k", min_value=1, max_value=10, value=3, step=1)
        fetch_k = st.slider("Search Kwargs - Fetch-k", min_value=top_k, max_value=100, value=30, step=5)
        lambda_mult = st.slider("Lambda Mult", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        search_type = st.selectbox("Search Type", options=["similarity", "mmr"], index=0)



# ── Main Area ────────────────────────────────────────────────────────────────
# Title và header
st.title("📄 Hỏi đáp thông minh với tài liệu PDF")
st.markdown("**SmartDoc AI** – Upload PDF → hỏi bất kỳ câu gì liên quan đến nội dung tài liệu. Hệ thống chạy cục bộ, bảo mật cao.")

# Trạng thái tài liệu
if st.session_state.vectorstore is None:
    st.info("Vui lòng tải file PDF lên ở ô bên trái (hàng dưới) để bắt đầu hỏi đáp.")
else:
    st.success(f"Đang làm việc với tài liệu: **{st.session_state.current_file}**")

# Hiển thị lịch sử chat
# Hiển thị lịch sử chat từ session_state.messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Input hỏi đáp + File uploader ngang hàng ────────────────────────────────
col_upload, col_chat = st.columns([1, 4])

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

with col_upload:
    uploaded_file = st.file_uploader(
        "TẢI PDF",
        type="pdf",
        accept_multiple_files=False,
        help="Chỉ hỗ trợ PDF có text layer (không phải scan/image-only). Tối đa ~50MB.",
        label_visibility="collapsed"
    )

with col_chat:
    prompt = st.chat_input("Nhập câu hỏi về tài liệu...") # ô nhập liệu

# ── Xử lý upload & indexing ──────────────────────────────────────────────────
if uploaded_file is not None:

    # ── Validation file size ──────────────────────────────────────────────
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ File **{uploaded_file.name}** quá lớn ({file_size_mb:.1f}MB). Giới hạn tối đa là {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Đang xử lý tài liệu..."):
            st.write("📖 Đang đọc file PDF...")
            # Bước 1
            try: # Lưu tạm file bằng tempfile.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                logger.info(f"Upload nhận file: {uploaded_file.name} | Size: {file_size_mb:.2f}MB")  # ← LOG 1


                # Load + split + embed + create FAISS
                # Gọi load_and_split để chia nhỏ văn bản thành các đoạn (chunk).
                # Bước 2
                st.write("✂️ Splitting — Đang chia nhỏ văn bản...")
                chunks = load_and_split(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap) ######### CHỈNH TẠI ĐÂY
                if not chunks:
                    raise ValueError("PDF_EMPTY")
                st.write(f"✅ Splitting xong — {len(chunks)} đoạn")
                logger.info(f"Splitting xong: {len(chunks)} chunks | chunk_size={chunk_size} | overlap={chunk_overlap}")  # ← LOG 2


                # Dùng create_vectorstore để tạo FAISS index từ embedding.
                # Nếu thành công → lưu vào session_state.vectorstore.
                # Bước 3
                st.write("🧠 Creating embeddings — Đang tạo vector...")
                st.session_state.vectorstore = create_vectorstore(chunks, get_embedder())
                st.session_state.current_file = uploaded_file.name

                logger.info(f"Vectorstore tạo xong: {uploaded_file.name}")  # ← LOG 3

                # Done
                st.success(f"Hoàn tất! Tài liệu **{uploaded_file.name}** đã xử lý ({len(chunks)} đoạn).")
                st.info(f"""
                📄 **{uploaded_file.name}**  
                📦 Kích thước: {file_size_mb:.2f} MB  
                🔢 Số đoạn (chunks): {len(chunks)}  
                """)

            except ValueError as e:
                # ── Invalid format warning ────────────────────────────────
                if "PDF_EMPTY" in str(e):
                    st.warning("⚠️ Không thể đọc nội dung từ file này. PDF có thể là bản scan hoặc chỉ chứa hình ảnh. Hãy thử file PDF có text layer.")
                else:
                    st.error(f"❌ File không hợp lệ: {str(e)}")

            except ConnectionError:
                # ── Model connection error ────────────────────────────────
                st.error("🔌 Không thể kết nối đến Ollama. Vui lòng kiểm tra Ollama đang chạy chưa (`ollama serve`).")

            except MemoryError:
                st.error("💾 Không đủ RAM để xử lý file này. Hãy thử file nhỏ hơn hoặc tăng chunk_size.")

            except Exception as e:
                # ── Phân loại lỗi theo message ────────────────────────────
                err = str(e).lower()
                if "connection" in err or "refused" in err or "ollama" in err:
                    st.error("🔌 Lỗi kết nối Ollama. Chạy lệnh `ollama serve` rồi thử lại.")
                elif "timeout" in err:
                    st.error("⏱️ Ollama phản hồi quá chậm. Thử lại sau hoặc kiểm tra tài nguyên máy.")
                elif "cuda" in err or "gpu" in err:
                    st.error("🖥️ Lỗi GPU/CUDA. Kiểm tra driver hoặc chạy Ollama ở chế độ CPU.")
                else:
                    st.error(f"❌ Lỗi không xác định: {str(e)}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Xử lý câu hỏi
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Lưu câu hỏi vào session_state.messages để hiển thị lại sau này.
    with st.chat_message("user"): # Hiển thị tin nhắn của người dùng trong khung chat.
        st.markdown(prompt)
    with st.chat_message("assistant"): # Tạo khung chat cho assistant.
        with st.spinner("Đang tìm kiếm và suy nghĩ..."): # Hiện spinner (vòng xoay) báo hiệu đang xử lý.
            if st.session_state.vectorstore is None:
                response = "Vui lòng upload tài liệu trước khi hỏi."
            else:
                try:
                    logger.info(f"Query: '{prompt[:80]}...'" if len(prompt) > 80 else f"Query: '{prompt}'")  # ← LOG 4
                    # Retrieve & Tạo retriever từ FAISS index.
                    # Thay đoạn retriever hiện tại
                    if search_type == "mmr":
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": top_k,
                                "lambda_mult": lambda_mult,  # 0.7 = nghiêng về relevance hơn
                                "fetch_k": fetch_k        # fetch nhiều hơn rồi MMR lọc lại
                            }
                        )
                    else:
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": top_k}
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
                        # Nếu không tìm thấy đoạn liên quan → trả lời mặc định.
                    else:
                        context = format_docs(retrieved_docs)
                        # Invoke chain
                        response = rag_chain.invoke({
                            "context": context,
                            "question": prompt
                        }).strip()

                        # Nếu LLM trả về rỗng hoặc quá ngắn → fallback
                        if len(response) < 10:
                            response = "Không tìm thấy thông tin phù hợp trong tài liệu."
                    logger.info(f"Response length: {len(response)} chars")  # ← LOG 6
                    
                except Exception as e:
                    # ── Model connection error khi query ─────────────────
                    err = str(e).lower()
                    if "connection" in err or "refused" in err:
                        response = "🔌 Mất kết nối đến Ollama. Kiểm tra `ollama serve` và thử lại."
                    elif "timeout" in err:
                        response = "⏱️ Ollama phản hồi quá chậm. Câu hỏi có thể quá phức tạp, thử rút gọn lại."
                    else:
                        response = f"❌ Lỗi xử lý câu hỏi: {str(e)}"

                    
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        streamlit_js_eval(
            js_expressions="""
                parent.document.querySelectorAll('*').forEach(function(el) {
                    el.scrollTop = el.scrollHeight;
                });
            """,
            key=f"scroll_{int(time.time() * 100000)}"
        )