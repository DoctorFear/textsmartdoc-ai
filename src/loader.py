from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(pdf_path: str, chunk_size=1000, chunk_overlap=100):
    # Load PDF bằng PDFPlumber
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    
    # Tách văn bản thành chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    return chunks
