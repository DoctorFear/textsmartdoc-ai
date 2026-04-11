from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
import os

def load_and_split(file_path: str, display_name: str | None = None, chunk_size=1000, chunk_overlap=100):
    # Xác định định dạng file
    ext = os.path.splitext(file_path)[1].lower()

    # Load file dựa trên định dạng
    if ext == ".pdf":
        loader = PDFPlumberLoader(file_path)    # Load PDF bằng PDFPlumber
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)      # Load DOCX bằng Docx2txt
    else: 
        raise ValueError(f"Định dạng file không được hỗ trợ: '{ext}'. Chỉ chấp nhận PDF hoặc DOCX.")
    
    # Load tài liệu
    docs = loader.load()

    if not docs or not any(getattr(doc, "page_content", "").strip() for doc in docs):
        raise ValueError("FILE_EMPTY")
    
    # Gắn metadata cho từng trang/đoạn văn bản (8.2.8)
    source_name = display_name or os.path.basename(file_path)
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for doc in docs:
        doc.metadata.update({
            "source": source_name,              # Tên file
            "file_type": ext.replace(".", ""),  # Loại file (pdf, docx)
            "upload_date": upload_time,         # Thời gian upload
        })
    
    # Tách văn bản thành chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       
        chunk_overlap=chunk_overlap, 
        length_function=len,
        add_start_index=True,        
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("FILE_EMPTY")
    
    return chunks

    
