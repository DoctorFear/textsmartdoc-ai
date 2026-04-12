# src/logger.py Logging
import logging
import os
from datetime import datetime

def setup_logger(name: str = "smartdoc") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:  # tránh add handler trùng khi Streamlit rerun
        return logger
    
    logger.setLevel(logging.INFO)
    
    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # ── File handler ──────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/smartdoc_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # ── Format ────────────────────────────────────────────────────────────
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger