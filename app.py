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
ACCENT  = "#FFC107"
DARK    = "#2C2F33"
LIGHT   = "#F8F9FA"

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
    .stApp {{ background-color: var(--light); }}
    section[data-testid="stSidebar"] {{ background-color: var(--dark); color: white; }}
    .stButton > button {{ background-color: var(--primary); color: white; border: none; }}
    .stButton > button:hover {{ background-color: {ACCENT}; color: black; }}
    h1, h2, h3 {{ color: var(--dark); }}
    .stChatMessage.user {{ background-color: #E3F2FD; }}
    .stChatMessage.assistant {{ background-color: #FFF3E0; }}
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
    st.markdown(f"<h2 style='color:white;'>SmartDoc AI</h2>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🗑️ Xóa tài liệu & Reset", type="primary", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.rerun()

    uploaded_file = st.file_uploader(
        "TẢI PDF LÊN",
        type="pdf",
        accept_multiple_files=False,
        help="Chỉ hỗ trợ PDF có text layer (không phải scan/image-only)"
    )

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

# ── Main Area – Chat ─────────────────────────────────────────────────────────
st.title("Hỏi đáp thông minh với tài liệu của bạn")


#Hiển thị tiêu đề và trạng thái (đã có tài liệu hay chưa).
if st.session_state.vectorstore is None:
    st.info("Upload file PDF ở sidebar để bắt đầu hỏi đáp.")
else:
    st.success(f"Đang làm việc với tài liệu: **{st.session_state.current_file}**")

# Hiển thị lịch sử chat
# Hiển thị lịch sử chat từ session_state.messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input hỏi đáp
if prompt := st.chat_input("Nhập câu hỏi về tài liệu..."): # ô nhập liệu
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