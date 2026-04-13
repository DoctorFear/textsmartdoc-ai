# src/ocr_handler.py
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class OCRConfig:
    """Cấu hình tập trung cho OCR"""
    OCR_CONFIDENCE_THRESHOLD = 0.3
    OCR_DPI = 300
    OCR_LANGUAGES = ['vi', 'en']
    OCR_USE_GPU = True


class OCRHandler:
    """
    Xử lý OCR cho PDF/DOCX bằng EasyOCR.
    
    Chế độ:
    - ocr_enabled=True  → Luôn dùng OCR (xử lý text + ảnh)
    - ocr_enabled=False → Chỉ PDFPlumber (nhanh, chỉ text)
    """
    
    def _initialize_reader(self):
        """Khởi tạo EasyOCR reader với cấu hình tối ưu"""
        try:
            import easyocr
            import torch
            
            # Kiểm tra xem CUDA có thực sự khả dụng không
            cuda_available = torch.cuda.is_available()
            if self.use_gpu and not cuda_available:
                logger.warning("⚠️ GPU được chọn nhưng CUDA không khả dụng. Chuyển về CPU.")
                self.use_gpu = False

            self.reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu, 
                verbose=False,
            )
            
            device_name = "GPU (CUDA)" if self.use_gpu else "CPU"
            logger.info(f"✓ EasyOCR initialized on {device_name} with {self.languages}")
            
        except ImportError:
            logger.error("❌ Thiếu thư viện. Chạy: pip install easyocr torch torchvision torchaudio")
            raise

    def extract_text_from_image(self, image: np.ndarray) -> str:
        """
        Trích xuất text từ ảnh numpy array.
        
        Args:
            image: numpy array từ cv2.imread() hoặc PIL.Image
        
        Returns:
            str: Text trích xuất, lọc theo confidence threshold
        """
        if self.reader is None:
            raise RuntimeError("OCR Reader not initialized")
        
        results = self.reader.readtext(image)
        
        # Lọc theo confidence threshold
        filtered_lines = [
            r[1] for r in results 
            if r[2] >= OCRConfig.OCR_CONFIDENCE_THRESHOLD
        ]
        
        text = "\n".join(filtered_lines)
        return text
    
    def process_pdf_to_docs(self, pdf_path: str, batch_size: int = 5):
        from pdf2image import convert_from_path
        from langchain.schema import Document

        docs = []
        page = 1

        while True:
            images = convert_from_path(
                pdf_path,
                dpi=OCRConfig.OCR_DPI,
                first_page=page,
                last_page=page + batch_size - 1
            )

            if not images:
                break

            if len(images) == 0:
                break

            for i, img in enumerate(images):
                actual_page = page + i
                logger.info(f"  Page {actual_page}...")

                text = self.extract_text_from_image(np.array(img))

                if text.strip():  # bỏ trang rỗng
                    docs.append(Document(
                        page_content=text,
                        metadata={"page": actual_page}
                    ))

            page += batch_size

        logger.info(f"✓ OCR done: {len(docs)} pages")
        return docs