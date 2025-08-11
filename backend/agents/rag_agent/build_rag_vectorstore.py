# build_rag_vectorstore_from_dir.py

import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from agents.llm_loader import get_embedding_model  # your own loader

# === Load all PDFs from a directory ===
def load_all_pdfs_from_directory(directory_path):
    all_text = ""
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                print(f"✅ Loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")
    return all_text

# === Split text into chunks ===
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# === Build and persist vectorstore ===
def create_vectorstore_from_directory(pdf_dir, persist_directory="./rag_db"):
    print("📂 Reading PDF files from:", pdf_dir)
    combined_text = load_all_pdfs_from_directory(pdf_dir)
    
    if not combined_text.strip():
        print("❌ No valid text extracted from PDFs.")
        return
    
    print("✂️ Splitting text into chunks...")
    documents = split_text(combined_text)
    
    print("📌 Embedding and storing in Chroma DB...")
    embedding = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Vector DB created at {persist_directory}")

if __name__ == "__main__":
    # 👇 Update this path to your actual PDF directory
    pdf_folder = "CP Act 2019_1732700731"
    os.makedirs("rag_db", exist_ok=True)
    create_vectorstore_from_directory(pdf_folder, persist_directory="./rag_db")
