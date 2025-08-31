# build_rag_vectorstore.py

import os
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the parent directory to Python path so we can import from agents
current_dir = Path(__file__).parent.absolute()
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from llm_loader import get_embedding_model

@dataclass
class LegalMetadata:
    """Metadata for legal document chunks"""
    filename: str
    page_number: int
    chapter: str = ""
    section: str = ""
    referenced_sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        metadata = {}
        for k, v in self.__dict__.items():
            if v:  # Only include non-empty values
                if isinstance(v, list):
                    metadata[k] = ', '.join(str(item) for item in v)  # Convert list to comma-separated string
                else:
                    metadata[k] = v
        return metadata

class LegalDocumentSplitter:
    """Structure-aware splitter for legal documents"""
    
    def __init__(self):
        # Structure detection patterns
        self.patterns = {
            'chapter': r'CHAPTER [A-Z]+',
            'section': r'(?:SECTION|SEC\.) \d+(?:\([a-z\d]+\))?',
            'definition': r'(?:^|\n)"[^"]+"\s+means',
            'numbered_clause': r'(?:^|\n)\d+\.\s+',
            'roman_numeral': r'(?:^|\n)[ivxIVX]+\.\s+',
            'cross_reference': r'(?:Section|Sec\.) \d+(?:\([a-z\d]+\))?'
        }
        
        # Different splitters for different content types
        self.base_splitter_params = {
            'separators': ["\n\n", "\n", ". ", " "],
            'length_function': len,
        }
        
        self.dense_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            **self.base_splitter_params
        )
        
        self.narrative_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            **self.base_splitter_params
        )

    def is_dense_content(self, text: str) -> bool:
        """Determine if text contains dense content like definitions or lists"""
        # Check for definitions, numbered lists, or dense subsections
        definition_pattern = self.patterns['definition']
        numbered_pattern = self.patterns['numbered_clause']
        subsection_pattern = r'\([a-z\d]+\)'
        
        return bool(
            re.search(definition_pattern, text) or
            len(re.findall(numbered_pattern, text)) > 2 or
            len(re.findall(subsection_pattern, text)) > 2
        )
    
    def extract_metadata(self, text: str, filename: str, page_number: int) -> LegalMetadata:
        """Extract structural metadata from text"""
        # Find chapter and section headers
        chapter_match = re.search(self.patterns['chapter'], text)
        section_match = re.search(self.patterns['section'], text)
        
        # Extract cross-references
        references = list(set(re.findall(self.patterns['cross_reference'], text)))
        
        return LegalMetadata(
            filename=filename,
            page_number=page_number,
            chapter=chapter_match.group(0) if chapter_match else "",
            section=section_match.group(0) if section_match else "",
            referenced_sections=references
        )
    
    def split_document(self, text: str, filename: str, page_number: int) -> List[Document]:
        """Split document while preserving structure and adding metadata"""
        chunks = []
        
        # Find all structural boundaries
        boundaries = []
        for pattern_name, pattern in self.patterns.items():
            if pattern_name not in ['cross_reference']:  # Skip reference pattern for splitting
                for match in re.finditer(pattern, text):
                    boundaries.append(match.start())
        
        # Sort boundaries and add end of text
        boundaries = sorted(list(set(boundaries)))
        boundaries.append(len(text))
        
        # Split at each boundary
        for i in range(len(boundaries) - 1):
            chunk_text = text[boundaries[i]:boundaries[i + 1]].strip()
            if not chunk_text:
                continue
                
            # Extract metadata for this chunk
            metadata = self.extract_metadata(chunk_text, filename, page_number)
            
            # Choose appropriate splitter and split into smaller chunks
            current_splitter = self.dense_splitter if self.is_dense_content(chunk_text) else self.narrative_splitter
            sub_chunks = current_splitter.split_text(chunk_text)
            
            # Create documents with metadata
            metadata_dict = metadata.to_dict()
            
            # Ensure referenced_sections is a string
            if 'referenced_sections' in metadata_dict and isinstance(metadata_dict['referenced_sections'], list):
                metadata_dict['referenced_sections'] = ', '.join(metadata_dict['referenced_sections'])
            
            for sub_chunk in sub_chunks:
                chunks.append(Document(
                    page_content=sub_chunk,
                    metadata=metadata_dict
                ))
        
        return chunks

def process_pdfs(directory_path: str) -> List[Document]:
    """Load and process PDFs with structure-aware splitting"""
    logging.info(f"Processing PDFs from directory: {directory_path}")
    documents = []
    splitter = LegalDocumentSplitter()
    
    try:
        for filename in os.listdir(directory_path):
            if not filename.lower().endswith(".pdf"):
                continue
                
            pdf_path = os.path.join(directory_path, filename)
            logging.info(f"Processing file: {pdf_path}")
            
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, 1):
                    logging.info(f"Processing page {page_num}")
                    text = page.extract_text()
                    if text:
                        # Log first few characters of text for debugging
                        preview = text[:100].replace('\n', ' ')
                        logging.info(f"Page {page_num} content preview: {preview}...")
                        
                        chunks = splitter.split_document(text, filename, page_num)
                        documents.extend(chunks)
                        
                        # Log chunk information
                        logging.info(f"Created {len(chunks)} chunks from page {page_num}")
                        if chunks:
                            sample_chunk = chunks[0]
                            logging.info(f"Sample chunk metadata: {sample_chunk.metadata}")
                            
                logging.info(f"✅ Successfully processed {filename} - Total chunks: {len(documents)}")
            except Exception as e:
                logging.error(f"❌ Failed to process {filename}: {str(e)}")
                logging.exception("Full traceback:")
        
        logging.info(f"Total documents processed: {len(documents)}")
        return documents
        
    except Exception as e:
        logging.error(f"❌ Critical error processing documents: {str(e)}")
        logging.exception("Full traceback:")
        return []

def create_vectorstore_from_directory(pdf_dir: str, persist_directory: str = "./rag_db"):
    """Create a vector store from legal documents with structure-aware processing"""
    print("📂 Reading and processing PDF files...")
    documents = process_pdfs(pdf_dir)
    
    if not documents:
        print("❌ No valid documents processed.")
        return
    
    # Print some statistics
    total_chunks = len(documents)
    avg_chunk_size = sum(len(doc.page_content) for doc in documents) / total_chunks
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
    
    # Sample metadata from first chunk
    if documents:
        print("\n📑 Sample chunk metadata:")
        for key, value in documents[0].metadata.items():
            print(f"   {key}: {value}")
    
    print("\n📌 Embedding and storing in Chroma DB...")
    embedding = get_embedding_model()
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Vector DB created at {persist_directory}")

if __name__ == "__main__":
    try:
        # Get current directory and set up paths
        current_dir = Path(__file__).parent.absolute()
        pdf_path = current_dir / "CP Act 2019_1732700731.pdf"
        db_path = current_dir / "rag_db"
        
        print(f"PDF path: {pdf_path}")
        print(f"DB path: {db_path}")
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        # Ensure DB directory exists
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Create vectorstore
        create_vectorstore_from_directory(
            str(current_dir),
            persist_directory=str(db_path)
        )
    except Exception as e:
        print(f"Error building vectorstore: {str(e)}")
        import traceback
        traceback.print_exc()


