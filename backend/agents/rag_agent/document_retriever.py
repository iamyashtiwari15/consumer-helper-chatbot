# agents/document_retriever.py

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .query_expander import QueryExpander
from agents.rag_agent.llm_loader import get_llm, get_embedding_model
from langchain_community.vectorstores import Chroma

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
        r'sec\.?\s+(\d+(?:\([a-z\d]+\))?)',  # Sec. 33
        r'section\s+(\d+)',                  # Section 33
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            # Normalize the extracted section number
            return match.group(1).upper() 
    return None

# Initialize components with proper error handling
try:
    logger.info("Initializing QueryExpander")
    expander = QueryExpander()
    
    # Setup Chroma with Langchain
    current_dir = Path(__file__).parent.absolute()
    db_path = current_dir / "rag_db"
    
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

# Hybrid retrieve_documents function
def retrieve_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using a hybrid approach:
    - Metadata filtering for section-specific queries
    - Semantic search for general queries
    - Combined results when appropriate
    """
    logger.info(f"🔍 Retrieving documents for query: {query}")
    
    # Extract section number if present
    section_num = extract_section_number(query)
    results = []
    
    try:
        # Get expanded query for semantic search
        expanded_query_dict = expander.expand_query(query)
        search_query = expanded_query_dict["original_query"]
        
        if section_num:
            # For section-specific queries, use metadata filtering
            filter_dict = {
                "where": {
                    "$eq": {
                        "section_number": section_num
                    }
                }
            }
            logger.info(f"Applied section filter: {filter_dict}")
            
            try:
                # Get section-specific results
                section_results = vectordb.similarity_search_with_score(
                    search_query,
                    k=k,
                    filter=filter_dict
                )
                
                # If we find good section-specific results, return them
                if section_results:
                    logger.info(f"Found {len(section_results)} section-specific results")
                    return section_results
            except Exception as e:
                logger.warning(f"Section-specific search failed: {str(e)}")
                # Continue to general search
                
        # For general queries or if section-specific search yields no results
        # Use semantic search with a filter that matches all sections
        filter_dict = {
            "where": {
                "$contains": {
                    "section_number": "Section"  # Match all sections
                }
            }
        }
        logger.info("Performing general semantic search")
        
        # Get results using semantic search
        results = vectordb.similarity_search_with_score(
            search_query,
            k=k,
            filter=filter_dict
        )

        if not results:
            logger.warning("⚠️ No documents found matching the search criteria.")
            return []
            
        logger.info(f"Found {len(results)} results through semantic search")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error retrieving documents: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

    try:
        # Perform the search using the constructed filter
        expanded_query_dict = expander.expand_query(query)
        logger.info(f"Expanded query: {expanded_query_dict}")
        
        # Use the original query for vector search since the expanded queries are too verbose
        # TODO: Implement better query selection strategy from expanded queries
        search_query = expanded_query_dict["original_query"]
        
        results = vectordb.similarity_search_with_score(
            search_query,
            k=k,
            filter=filter_dict
        )

        if not results:
            logger.warning("⚠️ No documents found matching the search criteria.")
            return []

        logger.info(f"✅ Found {len(results)} documents.")
        
        # Format the results into a list of dictionaries
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return formatted_results

    except Exception as e:
        logger.error(f"❌ Error retrieving documents: {str(e)}")
        logger.exception("Full traceback:")
        return []

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