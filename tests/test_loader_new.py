from src.loader import load_and_split

file_path = "tests/test_docx_table.docx"

print("=== Đang test load DOCX với UnstructuredWordDocumentLoader ===")

chunks = load_and_split(
    file_path=file_path,
    display_name="test_docx_table.docx",
    chunk_size=1200,      # dùng giá trị mặc định trong config của bạn
    chunk_overlap=200
)

print(f"\nTổng số chunks được tạo: {len(chunks)}\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content[:1200])   # In ra để xem bảng có giữ cấu trúc không
    print("\nMetadata:")
    print(chunk.metadata)
    print("-" * 90)
    print("\n")