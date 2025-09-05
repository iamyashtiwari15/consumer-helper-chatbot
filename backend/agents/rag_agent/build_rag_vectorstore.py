# build_rag_vectorstore_pdf.py

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from llm_loader import get_embedding_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from typing import Optional

@dataclass
class Section:
    number: str
    title: str
    content: str
    start_page: int
    subsections: dict = None  # Will store subsections like (a), (b), etc.
    
@dataclass
class Chunk:
    content: str
    metadata: dict

def clean_text(text: str) -> str:
    """Remove headers, footers, and clean up text."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, page numbers, and common headers
        if not line or re.match(r'^\d+$', line) or \
           re.match(r'^Page \d+ of \d+$', line) or \
           line.startswith('THE GAZETTE OF INDIA'):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def split_into_paragraphs(text: str, max_chars: int = 1000) -> List[str]:
    """Split text into paragraphs without breaking sentences."""
    # First split by double newlines to respect paragraph boundaries
    paragraphs = text.split('\n\n')
    results = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If single paragraph exceeds max length, split by sentences
        if len(para) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_length + len(sentence) > max_chars and current_chunk:
                    results.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            if current_length + len(para) > max_chars and current_chunk:
                results.append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)
    
    if current_chunk:
        results.append(' '.join(current_chunk))
    
    return results

def extract_section_info(text: str) -> Tuple[str, str]:
    """Extract section number and title from the section start."""
    # Match patterns like "Section 1." or "1." at start of line
    section_match = re.match(r'^(?:Section\s+)?(\d+)\.\s*(.*)$', text.strip())
    if section_match:
        number = section_match.group(1)
        title = section_match.group(2).strip()
        return f"Section {number}", title
    return "", ""

def split_into_sections(text: str, page_num: int) -> List[Section]:
    """Split text into sections based on section markers."""
    sections = []
    current_section = None
    current_content = []
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for new section start
        if re.match(r'^(?:Section\s+)?\d+\.', line):
            # Save previous section if it exists
            if current_section:
                current_section.content = '\n'.join(current_content).strip()
                sections.append(current_section)
            
            # Start new section
            number, title = extract_section_info(line)
            current_section = Section(number=number, title=title, content='', start_page=page_num)
            current_content = [line]
        elif current_section:
            # Only add non-empty lines
            if line:
                current_content.append(line)
    
    # Add the last section
    if current_section and current_content:
        current_section.content = '\n'.join(current_content).strip()
        sections.append(current_section)
    
    return sections

def identify_unit(text: str) -> tuple[str, str, str]:
    """
    Identify the legal unit type from the given text.
    Returns: (unit_type, unit_number, content)
    """
    # Section (e.g., "Section 1: Title")
    section_match = re.match(r'^\s*Section\s+(\d+)[\.:]?\s*(.*)', text, re.IGNORECASE)
    if section_match:
        return 'section', section_match.group(1), section_match.group(2).strip()

    # Subsection (e.g., "(1) Something...")
    subsection_match = re.match(r'^\s*\((\d+)\)\s*(.*)', text)
    if subsection_match:
        return 'subsection', subsection_match.group(1), subsection_match.group(2).strip()

    # Clause (e.g., "(i) Something...")
    clause_match = re.match(r'^\s*\(([ivx]+)\)\s*(.*)', text, re.IGNORECASE)
    if clause_match:
        return 'clause', clause_match.group(1), clause_match.group(2).strip()

    # Subclause (e.g., "(a) Something...")
    subclause_match = re.match(r'^\s*\(([a-z])\)\s*(.*)', text)
    if subclause_match:
        return 'subclause', subclause_match.group(1), subclause_match.group(2).strip()

    # Default: plain paragraph
    return 'paragraph', '', text.strip()


def split_hierarchical(section: Section, source_file: str) -> List[Document]:
    """Split section into hierarchical chunks with rich metadata."""
    documents = []
    base_metadata = {
        'section_number': section.number,
        'section_title': section.title,
        'start_page': section.start_page,
        'source_file': source_file,
    }

    # Use the content attribute of the Section object
    lines = section.content.splitlines()

    current = {
        "subsection": None,
        "clause": None,
        "subclause": None,
        "content": []
    }

    def flush_current():
        """Save the current accumulated unit into documents."""
        if current["content"]:
            unit_text = " ".join(current["content"]).strip()
            chunk_metadata = base_metadata.copy()
            if current["subsection"]:
                chunk_metadata['subsection'] = current["subsection"]
            if current["clause"]:
                chunk_metadata['clause'] = current["clause"]
            if current["subclause"]:
                chunk_metadata['subclause'] = current["subclause"]
            documents.append(Document(
                page_content=unit_text,
                metadata=chunk_metadata
            ))
            current["content"] = []

    for line in lines:
        if not line.strip():
            continue

        unit_type, unit_number, content = identify_unit(line)

        if unit_type == "subsection":
            flush_current()
            current["subsection"] = unit_number
            current["clause"] = None
            current["subclause"] = None
            current["content"] = [f"({unit_number}) {content}"]

        elif unit_type == "clause":
            flush_current()
            current["clause"] = unit_number
            current["subclause"] = None
            current["content"] = [f"({unit_number}) {content}"]

        elif unit_type == "subclause":
            flush_current()
            current["subclause"] = unit_number
            current["content"] = [f"({unit_number}) {content}"]

        else:  # paragraph continuation
            current["content"].append(content)

        # If the chunk is getting too long, flush it early
        if sum(len(c) for c in current["content"]) > 1200:
            flush_current()

    # Flush any leftover
    flush_current()

    return documents

def process_pdf_files(directory_path: str) -> list:
    logging.info(f"Processing PDF files from directory: {directory_path}")
    documents = []

    try:
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.warning("No PDF files found in directory")
            return []

        for filename in pdf_files:
            pdf_path = os.path.join(directory_path, filename)
            try:
                reader = PdfReader(pdf_path)
                all_sections = []

                # First pass: extract and clean text, identify sections
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and len(text.strip()) > 0:
                        cleaned_text = clean_text(text)
                        sections = split_into_sections(cleaned_text, page_num)
                        all_sections.extend(sections)

                # Second pass: process sections into hierarchical documents
                for section in all_sections:
                    if section.content.strip():
                        # Use the new hierarchical splitting
                        section_docs = split_hierarchical(section, filename)
                        documents.extend(section_docs)

                logging.info(f"✅ Processed {filename}: Created {len(documents)} chunks across {len(all_sections)} sections")

            except Exception as e:
                logging.error(f"❌ Failed to process {filename}: {e}")

        return documents

    except Exception as e:
        logging.error(f"❌ Critical error: {e}")
        logging.exception("Traceback:")
        return []

def create_vectorstore_from_pdf_directory(pdf_dir: str, persist_directory: str = "./rag_db") -> None:
    logging.info("📂 Processing PDF files...")
    documents = process_pdf_files(pdf_dir)

    if not documents:
        logging.error("❌ No valid documents to process")
        return

    try:
        logging.info("\n📦 Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=get_embedding_model(),
            persist_directory=persist_directory,
            collection_name="rag_db"  # Specify a fixed collection name
        )

        # Enhance metadata for better retrieval
        for doc in documents:
            if "section_number" in doc.metadata:
                doc.metadata["section_number"] = doc.metadata["section_number"].upper()

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
        create_vectorstore_from_pdf_directory(str(current_dir), str(db_path))
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.exception("Traceback:")
