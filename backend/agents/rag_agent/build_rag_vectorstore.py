# build_rag_vectorstore.py

import os
import re
import sys
import logging
import functools
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path
import unicodedata

# Add the parent directory to Python path so we can import from agents
current_dir = Path(__file__).parent.absolute()
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from llm_loader import get_embedding_model

# Content type definitions
CONTENT_TYPES = {
    'definition': (r'(?:^|\n)"[^"]+"\s+means', 4),
    'penalty': (r'penalty|fine', 5),
    'regulation': (r'shall|must|required|prohibited', 4),
    'guideline': (r'may|optional|recommended', 3),
    'general': (r'', 2)
}

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    text = unicodedata.normalize('NFKC', text)
    replacements = [
        (r'\s+', ' '),  # Normalize whitespace
        (r'[\u200b\u200c\u200d]', ''),  # Remove zero-width spaces
        (r'[\u2018\u2019]', "'"),  # Normalize quotes
        (r'[\u201C\u201D]', '"'),
        (r'(\w)-\s+(\w)', r'\1\2'),  # Fix hyphenation
        (r'([.!?])\1+', r'\1'),  # Remove repeated punctuation
        (r'[Ss]ec\.\s*(\d+)', r'Section \1'),  # Normalize section references
        (r'[Ss]ection\s+(\d+)(?:\s*\(([a-z\d]+)\))?', 
         lambda m: f"Section {m.group(1)}" + (f"({m.group(2)})" if m.group(2) else ""))
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text.strip()

@dataclass
class LegalMetadata:
    """Metadata for legal document chunks"""
    filename: str
    page_number: int
    chapter: str = ""
    section: str = ""
    section_title: str = ""
    subsection: str = ""
    paragraph: str = ""
    referenced_sections: List[str] = field(default_factory=list)
    content_type: str = ""  # e.g., 'definition', 'regulation', 'penalty'
    key_terms: List[str] = field(default_factory=list)
    hierarchy_path: str = ""  # e.g., "chapter_1/section_2/subsection_a"
    chunk_importance: int = 2  # Default importance

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary, joining lists into strings"""
        return {
            k: ', '.join(map(str, v)) if isinstance(v, list) else v
            for k, v in self.__dict__.items()
            if v or v == 0
        }

class LegalDocumentSplitter:
    """Structure-aware splitter for legal documents"""

    # Document structure patterns
    PATTERNS = {
        'chapter': r'CHAPTER [A-Z]+',
        'section': r'^\d+\.\s',  # Matches section numbers like '28. '
        'section_title': r'^\d+\.\s.*',  # Matches section titles like '28. (1) The State Government...'
        'subsection': r'^\s*\([a-z\d]+\)',
        'paragraph': r'^\s*\(\d+\)',
        'definition': r'(?:^|\n)"[^"]+"\s+means',
        'cross_reference': r'(?:Section|Sec\.) \d+(?:\([a-z\d]+\))?',
        'key_terms': r'"([^"]+)"\s+(?:means|refers to|includes)'
    }

    # Define chunk sizes for different content types
    CHUNK_CONFIG = {
        'dense': {'size': 500, 'overlap': 75},
        'narrative': {'size': 1200, 'overlap': 100}
    }

    def __init__(self):
        """Initialize text splitters for different content types"""
        splitter_params = {'separators': ["\n\n", "\n", ". ", " "], 'length_function': len}
        self.splitters = {
            key: RecursiveCharacterTextSplitter(
                chunk_size=config['size'],
                chunk_overlap=config['overlap'],
                **splitter_params
            ) for key, config in self.CHUNK_CONFIG.items()
        }

    def remove_unwanted_headers(self, text: str) -> str:
        """Remove unwanted headers like 'THE GAZETTE OF INDIA EXTRAORDINARY' but keep chapter names."""
        lines = text.split("\n")
        filtered_lines = [line for line in lines if not line.strip().startswith("THE GAZETTE OF INDIA")]
        return "\n".join(filtered_lines)

    def extract_metadata(self, text: str, filename: str, page_number: int) -> LegalMetadata:
        """Extract structural metadata from text"""
        text = self.remove_unwanted_headers(clean_text(text))  # Remove unwanted headers before processing
        matches = {
            pattern: re.search(regex, text)
            for pattern, regex in self.PATTERNS.items()
            if pattern not in ['cross_reference', 'key_terms']
        }
        references = list(set(re.findall(self.PATTERNS['cross_reference'], text)))
        key_terms = list(set(m.group(1) for m in re.finditer(self.PATTERNS['key_terms'], text)))
        hierarchy = []

        # Ensure hierarchy is built correctly
        for component in ['chapter', 'section', 'subsection', 'paragraph']:
            if matches.get(component):
                value = matches[component].group(0)
                if component in ['subsection', 'paragraph']:
                    value = value.strip('()')
                hierarchy.append(f"{component}_{value.split()[-1]}")

        # Validate and correct section metadata
        section = matches['section'].group(0).strip() if matches.get('section') else ""
        if not section:
            logging.warning(f"Missing section metadata on page {page_number} in file {filename}")

        # Deduplicate and validate references
        references = [ref for ref in references if ref.strip()]
        if not references:
            logging.warning(f"No references found on page {page_number} in file {filename}")

        content_type = next(
            (ctype for ctype, (pattern, _) in CONTENT_TYPES.items()
             if pattern and re.search(pattern, text, re.I)),
            'general'
        )
        importance = CONTENT_TYPES[content_type][1]

        return LegalMetadata(
            filename=filename,
            page_number=page_number,
            chapter=matches['chapter'].group(0) if matches.get('chapter') else "",
            section=section,
            section_title=matches['section_title'].group(0).strip() if matches.get('section_title') else "",
            subsection=matches['subsection'].group(0) if matches.get('subsection') else "",
            paragraph=matches['paragraph'].group(0) if matches.get('paragraph') else "",
            referenced_sections=references,
            content_type=content_type,
            key_terms=key_terms,
            hierarchy_path='/'.join(hierarchy),
            chunk_importance=importance
        )

    def split_document(self, text: str, filename: str, page_number: int) -> List[Document]:
        """Split document while preserving structure and adding metadata"""
        text = self.remove_unwanted_headers(clean_text(text))  # Remove unwanted headers before splitting
        boundaries = [
            (match.start(), pattern_name, match.group(0))
            for pattern_name, pattern in self.PATTERNS.items()
            if pattern_name not in ['cross_reference', 'key_terms']
            for match in re.finditer(pattern, text)
        ]
        if not boundaries:
            metadata = self.extract_metadata(text, filename, page_number)
            return self._process_chunk(text, metadata)
        boundaries.sort(key=lambda x: x[0])
        chunks = []
        for i, (start, _, _) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i < len(boundaries) - 1 else len(text)
            chunk_text = text[start:end].strip()
            if chunk_text:
                metadata = self.extract_metadata(chunk_text, filename, page_number)
                chunks.extend(self._process_chunk(chunk_text, metadata))
        return chunks

    def _process_chunk(self, text: str, metadata: LegalMetadata) -> List[Document]:
        """Process text chunk into valid documents with metadata"""
        splitter = self.splitters['dense' if re.search(self.PATTERNS['definition'], text) else 'narrative']
        chunks = []
        seen_content = set()
        for sub_chunk in splitter.split_text(text):
            if len(sub_chunk.strip()) >= 50 and any(p in sub_chunk for p in '.!?') and sub_chunk.count('(') == sub_chunk.count(')'):
                content_key = ' '.join(sorted(sub_chunk.split()))
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    chunks.append(Document(
                        page_content=sub_chunk,
                        metadata=metadata.to_dict()
                    ))
        return chunks

def validate_pdf(pdf_path: str) -> Tuple[bool, str]:
    """Validate PDF file before processing"""
    if not os.path.exists(pdf_path):
        return False, "File does not exist"
    if not os.access(pdf_path, os.R_OK):
        return False, "File is not readable"
    if os.path.getsize(pdf_path) == 0:
        return False, "File is empty"
    try:
        with open(pdf_path, 'rb') as f:
            if f.read(4) != b'%PDF':
                return False, "Not a valid PDF file"
    except Exception as e:
        return False, f"Error reading file header: {str(e)}"
    return True, "PDF is valid"

def process_pdf_page(page: Any, filename: str, page_num: int, splitter: LegalDocumentSplitter) -> List[Document]:
    """Process a single PDF page"""
    text = page.extract_text()
    if text and len(text.strip()) >= 50:
        # Replace '\n' with a space in the preview text
        preview_text = text[:100].replace("\n", " ")
        logging.info(f"Page {page_num} preview: {preview_text}...")
        return splitter.split_document(clean_text(text), filename, page_num)
    return []

def process_pdfs(directory_path: str) -> List[Document]:
    """Load and process PDFs with structure-aware splitting"""
    logging.info(f"Processing PDFs from directory: {directory_path}")
    documents = []
    splitter = LegalDocumentSplitter()
    stats = {'total_pages': 0, 'processed_pages': 0, 'total_chunks': 0}
    try:
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.warning("No PDF files found in directory")
            return []
        for filename in pdf_files:
            pdf_path = os.path.join(directory_path, filename)
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                logging.error(f"❌ Invalid PDF file: {filename}")
                continue
            try:
                reader = PdfReader(pdf_path)
                if not reader.pages:
                    continue
                file_chunks = []
                stats['total_pages'] += len(reader.pages)
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_chunks = process_pdf_page(page, filename, page_num, splitter)
                        if page_chunks:
                            file_chunks.extend(page_chunks)
                            stats['processed_pages'] += 1
                    except Exception as e:
                        logging.error(f"Error on page {page_num}: {e}")
                if file_chunks:
                    documents.extend(file_chunks)
                    stats['total_chunks'] += len(file_chunks)
                    logging.info(f"✅ Processed {filename}: {len(file_chunks)} chunks")
            except Exception as e:
                logging.error(f"❌ Failed to process {filename}: {e}")
        if documents:
            success_rate = (stats['processed_pages'] / stats['total_pages']) * 100
            avg_chunk_size = sum(len(doc.page_content) for doc in documents) / len(documents)
            logging.info(f"\n📊 Processing Results:")
            logging.info(f"   Pages processed: {stats['processed_pages']}/{stats['total_pages']} ({success_rate:.1f}%)")
            logging.info(f"   Total chunks: {stats['total_chunks']}")
            logging.info(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        return documents
    except Exception as e:
        logging.error(f"❌ Critical error: {e}")
        logging.exception("Traceback:")
        return []

def create_vectorstore_from_directory(pdf_dir: str, persist_directory: str = "./rag_db") -> None:
    """Create a vector store from legal documents"""
    logging.info("📂 Processing PDF files...")
    documents = process_pdfs(pdf_dir)
    if not documents:
        logging.error("❌ No valid documents to process")
        return
    try:
        logging.info("\n📦 Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=get_embedding_model(),
            persist_directory=persist_directory
        )
        vectordb.persist()
        logging.info(f"✅ Vector DB created at {persist_directory}")
    except Exception as e:
        logging.error(f"❌ Failed to create vector database: {e}")
        logging.exception("Traceback:")

if __name__ == "__main__":
    try:
        current_dir = Path(__file__).parent.absolute()
        db_path = current_dir / "rag_db"
        db_path.mkdir(parents=True, exist_ok=True)
        create_vectorstore_from_directory(str(current_dir), str(db_path))
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.exception("Traceback:")


