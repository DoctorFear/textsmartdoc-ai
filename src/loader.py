# src/loader.py
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
import os

from src.ocr_handler import OCRHandler, OCRConfig
from src.logger import setup_logger   
logger = setup_logger()


def load_and_split(
    file_path: str, 
    display_name: str | None = None, 
    chunk_size=1000, 
    chunk_overlap=100, 
    ocr_enabled=False
):
    """
    Load và split tài liệu từ PDF hoặc DOCX.
    
    Args:
        file_path: Đường dẫn file
        display_name: Tên hiển thị
        chunk_size: Kích thước chunk
        chunk_overlap: Overlap giữa chunks
        ocr_enabled: Toggle OCR từ UI
                    - True  → Luôn dùng OCR (xử lý text + ảnh)
                    - False → Chỉ PDFPlumber (nhanh, chỉ text)
    """
    
    ext = os.path.splitext(file_path)[1].lower()
    source_name = display_name or os.path.basename(file_path)

    # ==================== LOG ====================
    logger.info(f"📊 Bắt đầu xử lý file: {source_name}")
    logger.info(f"   → Chế độ OCR     = {'BẬT' if ocr_enabled else 'TẮT'}")
    logger.info(f"   → Chunk Size     = {chunk_size} ký tự")
    logger.info(f"   → Loại file      = {ext}")
    # =============================================

    docs = []
    extraction_method = "direct"

    # --- XỬ LÝ PDF ---
    if ext == ".pdf":
        if ocr_enabled:
            logger.info("→ OCR ON: Processing with EasyOCR...")
            try:
                ocr_handler = OCRHandler(
                    languages=OCRConfig.OCR_LANGUAGES,
                    use_gpu=OCRConfig.OCR_USE_GPU
                )

                docs = ocr_handler.process_pdf_to_docs(file_path)  # ✅ FIX
                extraction_method = "ocr"

            except Exception as e:
                logger.error(f"❌ OCR failed: {e}")
                raise
        else:
            # ❌ OFF: Chỉ PDFPlumber
            logger.info("→ OCR OFF: Using PDFPlumberLoader...")
            try:
                loader = PDFPlumberLoader(file_path)
                docs = loader.load()
                extraction_method = "direct"
            except Exception as e:
                logger.error(f"❌ PDFPlumberLoader failed: {e}")
                raise

    # --- XỬ LÝ DOCX ---
    elif ext == ".docx":
        logger.info("→ Processing DOCX with UnstructuredWordDocumentLoader...")
        try:
            loader = UnstructuredWordDocumentLoader(
                file_path,
                mode="elements",
            )
            docs = loader.load()
            extraction_method = "direct"
        except Exception as e:
            logger.error(f"❌ DOCX processing failed: {e}")
            raise

    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: '{ext}'. Chỉ chấp nhận PDF hoặc DOCX.")
    
    # --- KIỂM TRA DỮ LIỆU ---
    if not docs or not any(getattr(doc, "page_content", "").strip() for doc in docs):
        raise ValueError("FILE_EMPTY")
    
    # --- GẮN METADATA ---
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upload_timestamp = datetime.now().isoformat()

    for doc in docs:

        current_page = doc.metadata.get("page", 0)
        if extraction_method == "direct":
            current_page += 1

        doc.metadata.update({
            "source": source_name,
            "file_type": ext.replace(".", ""),
            "upload_date": upload_time,
            "upload_timestamp": upload_timestamp,
            "extraction_method": extraction_method,  # "ocr" hoặc "direct"
            "page": current_page
        })
    
    # --- SPLITTING ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       
        chunk_overlap=chunk_overlap, 
        length_function=len,
        add_start_index=True,        
    )
    chunks = splitter.split_documents(docs)

    logger.info(f"✓ Splitting done | Chunks: {len(chunks)} | Method: {extraction_method}")

    if not chunks:
        raise ValueError("FILE_EMPTY")
    
    return chunks