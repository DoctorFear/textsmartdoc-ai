# app.py
import streamlit as st
import os
import tempfile
from datetime import datetime
from src.loader import load_and_split
from src.embedder import embedder
from src.vectorstore import create_vectorstore
from src.rag_chain import llm, PROMPT, format_docs, rag_chain

# ── Theme & Color Palette ────────────────────────────────────────────────────
PRIMARY = "#007BFF"
ACCENT = "#FFC107"
DARK = "#2C2F33"
LIGHT = "#F8F9FA"
TEXT_SIDEBAR = "#FFFFFF"     # Màu chữ trắng cho sidebar

st.set_page_config( # Cấu hình giao diện
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
:root {{
    --primary: {PRIMARY};
    --accent: {ACCENT};
    --dark: {DARK};
    --light: {LIGHT};
}}
/* Nền ứng dụng */
.stApp {{
    background-color: var(--light);
}}
/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: var(--dark);
    color: white;
}}
/* Nút bấm chung */
.stButton > button {{
    background-color: var(--primary);
    color: white;
    border: none;
    border-radius: 6px !important;
    border: none !important;
    padding: 6px 12px !important;
    transition: 0.3s;
}}
.stButton > button:hover {{
    background-color: {ACCENT};
    color: black;
    transform: scale(1.01);
}}
/* Tiêu đề */
h1, h2, h3 {{
    color: var(--dark);
}}
/* Tin nhắn chat */
.stChatMessage.user {{
    background-color: #E3F2FD;
}}
.stChatMessage.assistant {{
    background-color: #FFF3E0;
}}
/* Nút Browse files */
button[kind="secondary"][data-testid="stBaseButton-secondary"] {{
    background-color: #28a745 !important; /* xanh lá */
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 6px 12px !important;
    transition: 0.3s;
    margin-top: 4px;
    margin-left: 4px;
}}
button[kind="secondary"][data-testid="stBaseButton-secondary"]:hover {{
    background-color: #218838 !important; /* xanh lá đậm */
    color: #fff !important;
    transform: scale(1.05);
}}
/* Nút Remove file */
button[kind="minimal"][data-testid="stBaseButton-minimal"] {{
    background-color: #dc3545 !important; /* đỏ */
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 4px 8px !important;
    transition: 0.3s;
    margin-left: 8px;
}}
/* Hiệu ứng hover */
button[kind="minimal"][data-testid="stBaseButton-minimal"]:hover {{
    background-color: #c82333 !important; /* đỏ đậm */
    color: #fff !important;
    transform: scale(1.05);
}}
/* Đổi màu tên file */
div[data-testid="stFileUploaderFileName"] {{
    color: #ffffff !important; /* xanh dương */
    font-weight: bold;
}}
/* Đổi màu dung lượng file */
small.st-emotion-cache-1rpn56r {{
    color: #FF5722 !important; /* cam */
    font-style: italic;
}}
.st-emotion-cache-1l4firl {{
    display: flex;
    -webkit-box-align: baseline;
    align-items: baseline;
    flex: 1 1 0%;
    padding-left: 5px;
    overflow: hidden;
}}
.st-emotion-cache-1h1td79 h2{{
    font-size: 2.25rem;
    font-weight: 800;
    padding: 1rem 0px;
    text-align: center;
}}
.st-emotion-cache-1cl4umz {{
  background-color: #ff4d4f; /* màu nền đỏ */
  color: white;              /* chữ trắng */
  border-radius: 8px;        /* bo góc */
  padding: 10px 16px;        /* khoảng cách trong */
  font-weight: bold;
  cursor: pointer;
}}
.st-emotion-cache-1cl4umz:hover {{
  background-color: #d9363e; /* màu khi hover */
}}
/* Ẩn label mặc định của file uploader khi dùng inline */
div[data-testid="stFileUploader"] label {{
    display: none;
}}
/* Căn chỉnh file uploader ngang với chat input */
div[data-testid="stFileUploader"] {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}
div[data-testid="stFileUploader"] section {{
    padding: 4px 8px !important;
    min-height: unset !important;
    border-radius: 8px !important;
}}

.st-emotion-cache-i0nc7r {{
    display: none;
}}
.st-emotion-cache-36dndz .ewslnz90 {{
    display: flex;
    height: 3.5rem;
    gap: 0.5rem;
}}
.st-emotion-cache-36dndz .ewslnz90 {{
    display: flex;
    width: 8rem;
    height: 3.5rem;
    gap: 0.5rem;
}}
.st-emotion-cache-ukvpxj {{
    width: auto;
    flex: none;
}}

.st-emotion-cache-fis6aj ewslnz97 {{
    display: none;
}}



.st-emotion-cache-1permvm {{
    position: fixed;
    bottom: 0;
    right: 0;
    width: 100%;
    z-index: 100;
    padding: 0.5rem 4.5rem 0.5rem 33rem;
    background-color: white;
    display: flex;
    flex-direction: row;
    align-items: center;
    box-sizing: border-box;
}}

.st-emotion-cache-18kf3ut:nth-last-of-type(3) {{
    margin-top: -1rem;
}}



/* Responsive cho màn hình nhỏ */
@media (max-width: 768px) {{
    .st-emotion-cache-1permvm {{
        flex-direction: column; /* xếp dọc khi màn hình hẹp */
        padding: 0.25rem;
    }}
}}

.st-emotion-cache-6mn6c9 {{
    background-color: rgb(240, 242, 246);
    border: 1px solid transparent;
    position: relative;
    display: flex;
    flex-direction: column;
    -webkit-box-align: stretch;
    align-items: stretch;
    flex: 1 1 0%;
    padding: 0.75rem 1rem;
    gap: 0.5rem;
    border-radius: 0.5rem;
    box-sizing: border-box;

}}

.st-emotion-cache-36dndz .ewslnz97{{
    display: none;
}}


</style>



""", unsafe_allow_html=True)



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
    st.subheader("Hướng dẫn sử dụng")
    st.markdown("""
    1. Tải lên file PDF ở phần chính giữa  
    2. Chờ xử lý xong (thấy thông báo xanh)  
    3. Đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh  
    4. Hệ thống chỉ trả lời dựa trên nội dung tài liệu
    """)

    # Settings information
    st.subheader("Thiết lập & Tùy chọn")
    st.markdown("""
    - Chunk size: 1200  
    - Chunk overlap: 200  
    - Số đoạn lấy lại: 4  
    - Temperature: 0.7  
    """)

    # Model configuration display
    st.subheader("Cấu hình mô hình")
    st.markdown(f"""
    • **Embedding**: paraphrase-multilingual-mpnet-base-v2  
    • **LLM**: Qwen2.5 7B (Ollama)  
    • **Retriever**: FAISS – top 4 chunks  
    • **Temperature**: 0.7  
    • **Ngày chạy**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    """)




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
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Đang đọc, chia nhỏ và lập chỉ mục tài liệu... (có thể mất 20–120 giây)"):
            try: # Lưu tạm file bằng tempfile.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                # Load + split + embed + create FAISS
                # Gọi load_and_split để chia nhỏ văn bản thành các đoạn (chunk).
                chunks = load_and_split(tmp_path, chunk_size=1200, chunk_overlap=200) ######### CHỈNH TẠI ĐÂY
                if not chunks:
                    raise ValueError("Không trích xuất được nội dung từ PDF (có thể là PDF scan hoặc rỗng).")
               
                # Dùng create_vectorstore để tạo FAISS index từ embedding.
                # Nếu thành công → lưu vào session_state.vectorstore.
                st.session_state.vectorstore = create_vectorstore(chunks, get_embedder())
                st.session_state.current_file = uploaded_file.name
                st.success(f"Hoàn tất! Tài liệu **{uploaded_file.name}** đã xử lý ({len(chunks)} đoạn).")
            except Exception as e:
                st.error(f"Lỗi khi xử lý PDF: {str(e)}\nKiểm tra:\n- Ollama đang chạy chưa?\n- File PDF có chứa text không?\n- RAM đủ không?")
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
                    # Retrieve & Tạo retriever từ FAISS index.
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4} ######### CHỈNH TẠI ĐÂY
                    )
                    retrieved_docs = retriever.invoke(prompt)
                   
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
                except Exception as e:
                    response = f"**Lỗi khi xử lý câu hỏi:** {str(e)}\n(Kiểm tra Ollama có đang chạy không?)"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


