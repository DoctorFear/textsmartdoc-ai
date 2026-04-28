import os
import gc
import time
import torch
import tempfile  # Phải import ở đây
import numpy as np
import concurrent.futures
from typing import List
from tqdm import tqdm

from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from langchain.schema import Document as LangChainDocument

# Giả định setup_logger đã có sẵn
from src.logger import setup_logger

class OCRConfig:
    CONFIDENCE_THRESHOLD = 0.3
    DPI = 220 # 200 là "điểm ngọt" giữa tốc độ và độ chính xác OCR
    LANGUAGES = ['vi', 'en']
    USE_GPU = False

class OCRHandler:
    def __init__(self, languages=None, use_gpu=None, dpi=None, batch_size=8, confidence_threshold=0.3):
        self.logger = setup_logger("smartdoc")
        self.languages = languages or OCRConfig.LANGUAGES
        self.use_gpu = use_gpu if use_gpu is not None else OCRConfig.USE_GPU
        self.dpi = min(max(dpi or OCRConfig.DPI, 100), 400)
        self.batch_size = min(max(batch_size, 1), 32)
        self.confidence_threshold = min(max(confidence_threshold, 0.0), 1.0)
        self.reader = self._init_reader()

    def _init_reader(self):
        try:
            import easyocr
            if self.use_gpu and not torch.cuda.is_available():
                self.logger.warning("GPU không khả dụng → chuyển sang CPU")
                self.use_gpu = False
            return easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
        except Exception as e:
            self.logger.error(f"Khởi tạo EasyOCR thất bại: {e}")
            raise

    def ocr_image(self, image: Image.Image) -> str:
        try:
            results = self.reader.readtext(np.array(image), batch_size=self.batch_size)
            lines = [r[1] for r in results if r[2] >= self.confidence_threshold]
            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Lỗi OCR ảnh: {e}")
            return ""
        finally:
            if self.use_gpu:
                torch.cuda.empty_cache()


    def _process_single_page(self, page_idx: int, img: Image.Image, filename: str):
        # Log ở cấp độ luồng nhỏ (Optional - có thể bỏ nếu quá rác log)
        self.logger.debug(f"Đang OCR trang {page_idx}...") 
        try:
            image_rgb = img.convert("RGB")
            text = self.ocr_image(image_rgb)
            if text.strip():
                return LangChainDocument(
                    page_content=text,
                    metadata={"page": page_idx, "source": filename}
                )
        except Exception as e:
            self.logger.error(f"Lỗi tại trang {page_idx}: {e}")
        return None

    def process_pdf_to_docs(self, pdf_path: str) -> List[LangChainDocument]:
        poppler_path = r"D:\textsmartdoc-ai\poppler-bin\poppler-24.08.0\Library\bin"
        filename = os.path.basename(pdf_path)
        
        try:
            total_pages = len(PdfReader(pdf_path).pages)
        except Exception as e:
            self.logger.error(f"Không đọc được PDF: {e}")
            return []

        self.logger.info(f"BẮT ĐẦU XỬ LÝ: {filename} | Tổng: {total_pages} trang")
        
        all_docs = []
        pdf_batch_size = 10 
        ocr_workers = 1 if self.use_gpu else min(4, os.cpu_count())

        with tempfile.TemporaryDirectory() as temp_dir:
            for start in range(1, total_pages + 1, pdf_batch_size):
                end = min(start + pdf_batch_size - 1, total_pages)
                
                # --- TẦNG 3: LOG TIẾN TRÌNH CHUNG (BATCHING) ---
                self.logger.info(f"[BATCH] Đang xử lý cụm trang: {start} -> {end}")
                batch_start_time = time.time()

                # --- TẦNG 1: LOG ĐA LUỒNG RENDER (POPPLER) ---
                self.logger.info(f"[P2I] Đang render PDF sang ảnh (thread_count=4)...")
                t1_start = time.time()
                try:
                    batch_images = convert_from_path(
                        pdf_path,
                        dpi=self.dpi,
                        first_page=start,
                        last_page=end,
                        fmt="jpeg",
                        thread_count=4, 
                        use_pdftocairo=True,
                        poppler_path=poppler_path
                    )
                    self.logger.info(f"[P2I] Render xong {len(batch_images)} trang trong {time.time()-t1_start:.2f}s")
                except Exception as e:
                    self.logger.error(f"[P2I] Lỗi: {e}")
                    continue

                # --- TẦNG 2: LOG ĐA LUỒNG OCR (PYTHON THREADS) ---
                self.logger.info(f"[OCR] Đang chạy OCR đa luồng ({ocr_workers} workers)...")
                t2_start = time.time()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=ocr_workers) as executor:
                    page_indices = list(range(start, end + 1))
                    results = list(executor.map(
                        self._process_single_page, 
                        page_indices, 
                        batch_images,
                        [filename] * len(batch_images)
                    ))
                
                valid_batch_docs = [d for d in results if d is not None]
                all_docs.extend(valid_batch_docs)
                self.logger.info(f"[OCR] OCR xong cụm trang trong {time.time()-t2_start:.2f}s")

                # --- GIẢI PHÓNG BỘ NHỚ ---
                for img in batch_images:
                    img.close()
                del batch_images
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()
                
                self.logger.info(f"[BATCH] Hoàn tất cụm {start}-{end} sau {time.time()-batch_start_time:.2f}s. Tổng tích lũy: {len(all_docs)} trang.")

        self.logger.info(f"MODULE OCR DONE: Đã trích xuất {len(all_docs)}/{total_pages} trang.")
        return all_docs

