from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import os
import tempfile
from docx2pdf import convert
import pythoncom

# === IMPORT SAU KHI GỘP ===
from src.ocr_handler import OCRHandler, OCRConfig
from src.logger import setup_logger   

logger = setup_logger()


def convert_docx_to_pdf(input_path: str) -> str:
    output_dir = tempfile.mkdtemp()
    
    output_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(input_path))[0] + ".pdf"
    )

    try:
        pythoncom.CoInitialize()   # 🔥 FIX CHÍNH
        convert(input_path, output_path)
        return output_path

    except Exception as e:
        raise RuntimeError(f"Lỗi convert DOCX → PDF: {e}")

    finally:
        pythoncom.CoUninitialize()  # 🔥 cleanup

def load_and_split(
    file_path: str, 
    display_name: str | None = None, 
    chunk_size=1000, 
    chunk_overlap=100, 
    ocr_enabled=False
):
    """
    Load và split tài liệu từ PDF hoặc DOCX.
    """
    
    ext = os.path.splitext(file_path)[1].lower()
    source_name = display_name or os.path.basename(file_path)

    logger.info(f"📊 Bắt đầu xử lý file: {source_name}")
    logger.info(f"   → Chế độ OCR     = {'BẬT' if ocr_enabled else 'TẮT'}")
    logger.info(f"   → Chunk Size     = {chunk_size} ký tự")
    logger.info(f"   → Loại file      = {ext}")

    docs = []
    extraction_method = "direct"

    # ====================== FIX QUAN TRỌNG ======================
    # Convert DOCX → PDF TRƯỚC khi xử lý
    if ext == ".docx":
        logger.info("→ Converting DOCX → PDF...")
        try:
            file_path = convert_docx_to_pdf(file_path)
            ext = ".pdf"
            logger.info("→ DOCX converted → continue as PDF")
        except Exception as e:
            logger.error(f"❌ DOCX → PDF pipeline failed: {e}")
            raise

    # ====================== XỬ LÝ PDF ======================
    if ext == ".pdf":
        if ocr_enabled:
            logger.info("→ OCR ON: Processing PDF with Hybrid OCR...")
            try:
                ocr_handler = OCRHandler(
                    languages=OCRConfig.LANGUAGES,
                    use_gpu=OCRConfig.USE_GPU
                )
                docs = ocr_handler.process_pdf_to_docs(file_path)
                extraction_method = "ocr"

            except Exception as e:
                logger.error(f"❌ PDF OCR failed: {e}")
                raise
        else:
            logger.info("→ OCR OFF: Using PDFPlumberLoader...")
            try:
                loader = PDFPlumberLoader(file_path)
                docs = loader.load()
                extraction_method = "direct"
            except Exception as e:
                logger.error(f"❌ PDFPlumberLoader failed: {e}")
                raise

        try:
            from PyPDF2 import PdfReader
            total_pages = len(PdfReader(file_path).pages)

            pages_in_docs = set(d.metadata.get("page", 0) for d in docs)
            missing_pages = set(range(1, total_pages + 1)) - pages_in_docs

            print("📄 Total pages:", total_pages)
            print("📑 Pages extracted:", sorted(pages_in_docs))
            print("❌ Missing pages:", sorted(missing_pages))
        except Exception as e:
            print("⚠️ Debug page error:", e)

    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: '{ext}'. Chỉ chấp nhận PDF hoặc DOCX.")
    
    # ====================== KIỂM TRA ======================
    if not docs or not any(getattr(doc, "page_content", "").strip() for doc in docs):
        raise ValueError("FILE_EMPTY")
    
    # ====================== GẮN METADATA ======================
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    upload_timestamp = datetime.now().isoformat()
    original_ext = os.path.splitext(source_name)[1].lower()

    for doc in docs:
        raw_p = doc.metadata.get("page", 0)

        # PDFPlumber: 0-indexed → +1 để ra trang thật
        if ext == ".pdf" and not ocr_enabled:
            actual_page = raw_p + 1
        else:
            actual_page = raw_p if raw_p > 0 else 1

        doc.metadata["page"] = actual_page
        doc.metadata.update({
            "source": source_name,
            "file_type": ext.replace(".", ""),                 # pdf
            "original_file_type": original_ext.replace(".", ""),  # docx/pdf
            "upload_date": upload_time,
            "upload_timestamp": upload_timestamp,
            "extraction_method": extraction_method,
            "page": actual_page
        })
    
    # ====================== SPLITTING ======================
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

    # --- DEBUG SECTION START ---
    if chunks:
        first_chunk = chunks[0]
        last_chunk = chunks[-1]
        
        print("\n" + "="*50)
        print("🔍 DEBUG METADATA:")
        print(f"📄 File: {source_name}")
        print(f"🛠️ Method: {extraction_method}")
        print(f"1️⃣ Trang đầu tiên (Chunk 0) lưu là: {first_chunk.metadata.get('page')}")
        print(f"🔟 Trang cuối cùng (Last Chunk) lưu là: {last_chunk.metadata.get('page')}")
        print("="*50 + "\n")

        if ext == ".pdf" and 'total_pages' in locals():
            missing_pages = set(range(1, total_pages+1)) - set(d.metadata["page"] for d in docs)
            print("Missing pages:", missing_pages)
    # --- DEBUG SECTION END ---
    
    return chunks