import os
import gc
import torch
import numpy as np
from typing import List
import concurrent.futures
from tqdm import tqdm

from pdf2image import convert_from_path
from PIL import Image

from langchain.schema import Document as LangChainDocument
from PyPDF2 import PdfReader

from src.logger import setup_logger


class OCRConfig:
    """Cấu hình OCR chung"""
    CONFIDENCE_THRESHOLD = 0.3
    DPI = 220
    LANGUAGES = ['vi', 'en']
    USE_GPU = False


class OCRHandler:
    """
    OCR cho PDF
    (DOCX đã được convert sang PDF từ loader)
    """

    def __init__(self, languages=None, use_gpu=None):
        self.logger = setup_logger("smartdoc")
        self.languages = languages or OCRConfig.LANGUAGES
        self.use_gpu = use_gpu if use_gpu is not None else OCRConfig.USE_GPU
        self.reader = self._init_reader()

    def _init_reader(self):
        try:
            import easyocr

            if self.use_gpu and not torch.cuda.is_available():
                self.logger.warning("GPU không khả dụng → chuyển sang CPU")
                self.use_gpu = False

            reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            self.logger.info(f"✓ EasyOCR khởi tạo trên {'GPU' if self.use_gpu else 'CPU'}")
            return reader

        except Exception as e:
            self.logger.error(f"❌ Khởi tạo EasyOCR thất bại: {e}")
            raise

    def ocr_image(self, image: Image.Image) -> str:
        """OCR một ảnh PIL"""
        try:
            results = self.reader.readtext(np.array(image), batch_size=8)
            lines = [r[1] for r in results if r[2] >= OCRConfig.CONFIDENCE_THRESHOLD]
            return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"Lỗi OCR ảnh: {e}")
            return ""

        finally:
            if self.use_gpu:
                torch.cuda.empty_cache()

    # ====================== PDF ======================
    def process_pdf_to_docs(self, pdf_path: str) -> List[LangChainDocument]:
        """
        OCR toàn bộ PDF:
        - Convert toàn bộ PDF → images 1 lần (NHANH hơn rất nhiều)
        - OCR từng trang (đa luồng nếu CPU)
        """

        poppler_path = r"D:\textsmartdoc-ai\poppler-bin\poppler-24.08.0\Library\bin"
        filename = os.path.basename(pdf_path)

        try:
            total_pages = len(PdfReader(pdf_path).pages)
        except Exception as e:
            self.logger.error(f"Không đọc được PDF: {e}")
            return []

        self.logger.info(f"🚀 OCR PDF: {filename} ({total_pages} trang)")

        # ================= CONVERT PDF → IMAGES =================
        try:
            images = convert_from_path(
                pdf_path,
                dpi=OCRConfig.DPI,
                poppler_path=poppler_path
            )
        except Exception as e:
            self.logger.error(f"❌ Lỗi convert PDF → images: {e}")
            return []

        docs = []

        # ================= OCR =================
        def process_page(idx_img):
            idx, img = idx_img
            try:
                image = img.convert("RGB")
                text = self.ocr_image(image)

                if text.strip():
                    return LangChainDocument(
                        page_content=text,
                        metadata={
                            "page": idx + 1,
                            "source": filename,
                            "extraction_method": "ocr"
                        }
                    )
                else:
                    self.logger.warning(f"Trang {idx+1}: Không nhận diện được chữ")
                    return None

            except Exception as e:
                self.logger.error(f"Lỗi OCR trang {idx+1}: {e}")
                return None

            finally:
                try:
                    img.close()
                except:
                    pass

        # ================= MULTI THREAD =================
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1 if self.use_gpu else 3
        ) as executor:

            results = list(tqdm(
                executor.map(process_page, enumerate(images)),
                total=len(images),
                desc="OCR PDF Pages"
            ))

        # lọc None
        docs = [doc for doc in results if doc is not None]

        # ================= CLEAN MEMORY =================
        del images
        gc.collect()

        self.logger.info(f"✅ OCR xong: {len(docs)}/{total_pages} trang")
        return docs