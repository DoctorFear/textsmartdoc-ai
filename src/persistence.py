# persistence.py Save/load FAISS + chat JSON → 8.2.2
# ── Xử lý toàn bộ việc lưu/load lịch sử chat + FAISS vectorstore ra disk ────
#
# Cấu trúc thư mục được tạo tự động:
#   chat_history.json              ← danh sách sessions + messages
#   faiss_store/
#       chat_0/
#           index.faiss
#           index.pkl
#       chat_1/
#           ...

import json
import os
import shutil
from langchain_community.vectorstores import FAISS

# ── Đường dẫn mặc định ───────────────────────────────────────────────────────
HISTORY_FILE = "chat_history.json"
FAISS_DIR    = "faiss_store"


# ────────────────────────────────────────────────────────────────────────────
# 1. MESSAGES  →  chat_history.json
# ────────────────────────────────────────────────────────────────────────────

def save_history(chat_sessions: list) -> None:
    """
    Lưu toàn bộ danh sách sessions xuống chat_history.json.

    Chỉ lưu các field có thể JSON-serialize được:
        id, title, messages, file
    Bỏ qua field 'vectorstore' (object Python — lưu riêng bằng save_vectorstore).
    """
    serializable = []
    for s in chat_sessions:
            serializable.append({
                "id":               s["id"],
                "title":            s["title"],
                "messages":         s["messages"],
                "file":             s.get("file"),
                "files":            s.get("files", []),
                "files_metadata":   s.get("files_metadata", {}),   # ← Quan trọng: lưu metadata theo file
            })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_history() -> list:
    """
    Đọc chat_history.json → trả về list sessions khi app khởi động.

    Field 'vectorstore' được set = None vì object FAISS không lưu trong JSON.
    Dùng load_vectorstore() riêng khi user click vào một session.

    Trả về [] nếu file chưa tồn tại hoặc bị hỏng.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        for s in sessions:
            s.setdefault("messages", [])
            s.setdefault("file", None)
            s.setdefault("files", [])
            s.setdefault("files_metadata", {})
            s.setdefault("vectorstore", None)
        return sessions
    except (json.JSONDecodeError, KeyError):
        return []


# ────────────────────────────────────────────────────────────────────────────
# 2. FAISS VECTORSTORE  →  faiss_store/chat_<id>/
# ────────────────────────────────────────────────────────────────────────────

def save_vectorstore(session_id: int, vectorstore) -> None:
    if vectorstore is None:
        return
    folder = os.path.join(FAISS_DIR, f"chat_{session_id}")
    os.makedirs(folder, exist_ok=True)
    vectorstore.save_local(folder)


def load_vectorstore(session_id: int, embedder) -> object | None:
    folder = os.path.join(FAISS_DIR, f"chat_{session_id}")
    faiss_file = os.path.join(folder, "index.faiss")

    if not os.path.exists(faiss_file):
        return None
    try:
        return FAISS.load_local(
            folder,
            embedder,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Lỗi load FAISS cho session {session_id}: {e}")
        # delete_vectorstore(session_id) FIX 8.2.2
        return None


# ────────────────────────────────────────────────────────────────────────────
# 3. DỌN DẸP
# ────────────────────────────────────────────────────────────────────────────

def delete_all_vectorstores() -> None:
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)


def delete_vectorstore(session_id: int) -> None:
    """
    Xóa vectorstore của một session cụ thể (chat_<id>).
    """
    folder = os.path.join(FAISS_DIR, f"chat_{session_id}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
