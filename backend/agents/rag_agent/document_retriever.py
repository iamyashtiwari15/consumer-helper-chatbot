def format_docs(results):
    return [
        {
            "content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "source": "local_db",
                "score": score
            }
        }
        for doc, score in results
    ]
# agents/document_retriever.py

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from backend.agents.rag_agent.query_expander import QueryExpander
from backend.agents.rag_agent.query_classifier import QueryClassifier
from backend.agents.web_search.web_search_agent import WebSearchAgent
from langchain.docstore.document import Document
from backend.agents.rag_agent.llm_loader import get_llm, get_embedding_model
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_section_number(query: str) -> Optional[str]:
    """Extract section number from query"""
    import re
    patterns = [
        r'section\s+(\d+(?:\([a-z\d]+\))?)',  # Section 33(1)
        r'sec\.?\s+(\d+(?:\([a-z\d]+\))?)',   # Sec. 33
        r'section\s+(\d+)',                    # Section 33
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            # Normalize the extracted section number to match the format in our database
            section_num = match.group(1).strip()
            normalized = f"Section {section_num}"
            logger.info(f"Extracted section number: {normalized}")
            return normalized  # Format must match exactly what's in the database
    logger.info("No section number found in query")
    return None

# Initialize components with proper error handling
try:
    logger.info("Initializing components")
    expander = QueryExpander()
    classifier = QueryClassifier()
    
    # Setup Chroma with Langchain
    current_dir = Path(__file__).parent.absolute()
    db_path = current_dir / "rag_db"
    
    # Initialize logger with debug level for development
    logger.setLevel(logging.DEBUG)
    
    # Check if the database directory exists
    if not db_path.exists():
        logger.error(f"❌ Database path does not exist: {db_path}")
        raise FileNotFoundError(f"Database directory not found at {db_path}")

    logger.info(f"Initializing Chroma at path: {db_path}")
    vectordb = Chroma(
        persist_directory=str(db_path),
        embedding_function=get_embedding_model(),
        collection_name="rag_db" # Ensure this matches the builder script
    )
    
    # Log collection info
    try:
        # Get collection stats using Langchain's Chroma
        collection_stats = vectordb._collection.count()
        logger.info(f"✅ Successfully connected to ChromaDB. Collection contains {collection_stats} documents")
        if collection_stats == 0:
            logger.warning("⚠️ WARNING: ChromaDB collection is empty. Please run build_rag_vectorstore.py to index documents.")
    except Exception as e:
        logger.warning(f"⚠️ Could not get collection count: {str(e)}")
        
except Exception as e:
    logger.error("❌ Failed to initialize document retriever components")
    logger.exception("Initialization error details:")
    # Re-raise the exception to stop the application from running with a broken state
    raise

# Initialize web search agent

# Hybrid retrieve_documents function
def retrieve_documents(query: str, k: int = 5, query_classification: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve relevant documents using a hybrid approach:
    - Metadata filtering for section-specific queries
    - Semantic search for general queries
    - Web search for additional context when needed
    
    Returns:
        Tuple of (retrieved documents, query classification)
    """
    logger.info(f"🔍 Retrieving documents for query: {query}")
    
    # Use provided query_classification or fallback to classifier
    if query_classification is None:
        try:
            query_classification = classifier.classify_query(query)
            logger.info(f"Query classification: {query_classification}")
        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}")
            query_classification = {
                "query_type": "general-info",
                "topics": [],
                "required_info_types": []
            }
    
    # Initialize list to store all documents
    all_documents = []
    
    
    # Extract section number if present
    section_num = extract_section_number(query)
    results = []
    try:
        # Get expanded query for semantic search
        expanded_query_dict = expander.expand_query(query)
        search_query = expanded_query_dict["original_query"]

        # Try section-specific retrieval first if section number is present
        section_docs = []
        if section_num:
            filter_dict = {"section_number": {"$eq": section_num}}
            logger.info(f"Applied section filter: {filter_dict}")
            try:
                section_results = vectordb.similarity_search_with_score(
                    search_query,
                    k=k,
                    filter=filter_dict
                )
                section_docs = format_docs(section_results)
                if section_docs:
                    logger.info(f"Found {len(section_docs)} section-specific results")
                    all_documents.extend(section_docs)
            except Exception as e:
                logger.warning(f"Section-specific search failed: {str(e)}")
                # Continue to general search

        # If no section-specific results, or for general queries, do semantic search
        if not section_docs or (query_classification and query_classification.get("query_type", "general-info") != "section-specific"):
            logger.info("Performing general semantic search")
            results = vectordb.similarity_search_with_score(
                search_query,
                k=k
            )
            general_docs = format_docs(results)
            if general_docs:
                logger.info(f"Found {len(general_docs)} results through general semantic search")
                all_documents.extend(general_docs)

        if not all_documents:
            logger.warning("⚠️ No documents found from any source.")
            return [], query_classification

        # Sort all documents by score
        all_documents.sort(key=lambda x: x.get("metadata", {}).get("score", 0), reverse=True)
        logger.info(f"Returning total of {len(all_documents)} documents from all sources")
        return all_documents, query_classification

    except Exception as e:
        logger.error(f"❌ Error retrieving documents: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return [], query_classification

# The `AgenticDocumentRetriever` class and its methods
# (`_validate_documents`, `_deduplicate_and_rank`) were complex and
# depended on the `rank_documents` function which was removed.
# To make the code runnable and correct, these have been
# commented out or removed entirely.
# A simpler, more reliable retrieval function is provided above.

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_query_with_section = "What is the penalty for fraud in Section 33?"
    test_query_general = "What are the rules regarding data privacy?"
    
    print(f"\n--- Running search for: '{test_query_with_section}' ---")
    docs = retrieve_documents(test_query_with_section)
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1} (Score: {doc['score']}):")
        print(f"Content: {doc['content'][:200]}...")
        print(f"Metadata: {doc['metadata']}")

    print(f"\n--- Running search for: '{test_query_general}' ---")
    docs = retrieve_documents(test_query_general)
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1} (Score: {doc['score']}):")
        print(f"Content: {doc['content'][:200]}...")
        print(f"Metadata: {doc['metadata']}")