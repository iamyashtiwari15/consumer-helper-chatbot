import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain_chroma import Chroma
from .llm_loader import get_embedding_model

logger = logging.getLogger(__name__)

class SectionRetriever:
    """
    Specialized retriever for handling section-specific queries in the Consumer Protection Act.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.embedding_model = get_embedding_model()
        self.vectordb = Chroma(
            persist_directory=str(db_path),
            embedding_function=self.embedding_model,
            collection_name="rag_db"
        )
        logger.debug(f"Initialized SectionRetriever with database at {db_path}")
    
    def get_section(self, section_number: str) -> List[Dict[str, Any]]:
        """
        Retrieve a specific section and all its subsections.
        """
        try:
            # Ensure section number is in correct format
            if not section_number.startswith("Section "):
                section_number = f"Section {section_number}"
            
            logger.debug(f"Searching for section: {section_number}")
            
            # Query the vector store for the exact section
            results = self.vectordb.similarity_search_with_score(
                query=f"full text of {section_number}",
                k=10,  # Get more results to ensure we catch all subsections
                filter={"section_number": section_number}
            )
            
            if not results:
                logger.warning(f"No results found for section {section_number}")
                return []
            
            # Process and format results
            documents = []
            for doc, score in results:
                documents.append({
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "score": score
                    }
                })
                logger.debug(f"Found content: {doc.page_content[:100]}...")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving section {section_number}: {str(e)}")
            logger.exception("Full traceback:")
            return []
    
    def extract_section_number(self, query: str) -> Optional[str]:
        """
        Extract and normalize section number from query.
        """
        import re
        patterns = [
            r'section\s+(\d+(?:\([a-z\d]+\))?)',  # Section 33(1)
            r'sec\.?\s+(\d+(?:\([a-z\d]+\))?)',   # Sec. 33
            r'section\s+(\d+)',                    # Section 33
        ]
        
        query = query.lower().strip()
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                section_num = match.group(1).strip()
                normalized = f"Section {section_num}"
                logger.debug(f"Extracted section {normalized} from query: {query}")
                return normalized
        
        logger.debug(f"No section number found in query: {query}")
        return None
