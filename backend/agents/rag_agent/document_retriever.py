# agents/document_retriever.py

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .query_expander import QueryExpander
from agents.rag_agent.llm_loader import get_llm
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Step 1: Define a wrapper that matches ChromaDB's required interface
class ChromaCompatibleEmbeddingFunction:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(input)

    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

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
            return match.group(1)
    return None

# Step 2: Initialize components with proper error handling
try:
    logger.info("Initializing QueryExpander")
    expander = QueryExpander()
    
    logger.info("Initializing ChromaCompatibleEmbeddingFunction")
    embedding_function = ChromaCompatibleEmbeddingFunction()
    
    # Step 3: Setup ChromaDB client and collection with absolute path
    current_dir = Path(__file__).parent.absolute()
    db_path = current_dir / "rag_db"
    db_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at path: {db_path}")
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Try to get collection count to verify database access
    logger.info("Setting up ChromaDB collection")
    collection = chroma_client.get_or_create_collection(
        name="rag_db",
        embedding_function=embedding_function
    )
    
    # Log collection info
    try:
        count = collection.count()
        logger.info(f"Successfully connected to ChromaDB. Collection contains {count} documents")
        if count == 0:
            logger.warning("WARNING: ChromaDB collection is empty. Please run build_rag_vectorstore.py to index documents.")
    except Exception as e:
        logger.warning(f"Could not get collection count: {str(e)}")
        
except Exception as e:
    logger.error("Failed to initialize document retriever components")
    logger.exception("Initialization error details:")
collection = chroma_client.get_or_create_collection(
    name="rag_db",
    embedding_function=embedding_function
)

def retrieve_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant documents with section-aware search"""
    logger.info(f"Retrieving documents for query: {query}")
    
    # Extract section number if present
    section_num = extract_section_number(query)
    logger.info(f"Extracted section number: {section_num}")
    
    try:
        # If section number found, prioritize exact matches
        if section_num:
            logger.info(f"Attempting exact section match for Section {section_num}")
            # Try exact section match using string pattern
            section_pattern = f"SECTION {section_num}|SEC. {section_num}"
            section_filter = {
                "section": {"$regex": section_pattern}
            }
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=section_filter
                )
                
                if results and len(results['documents']) > 0 and results['documents'][0]:
                    logger.info(f"Found exact section match. First result metadata: {results.get('metadatas', [[]])[0]}")
                    return results['documents'][0]
                else:
                    logger.info("No exact section match found")
            except Exception as e:
                logger.error(f"Error during exact match search: {str(e)}")
        
        # Fallback to semantic search
        logger.info("Falling back to semantic search")
        expanded_query = expander.expand_query(query)
        logger.info(f"Expanded query: {expanded_query}")
        
        try:
            results = collection.query(
                query_texts=[expanded_query],
                n_results=k
            )
            
            if results and len(results['documents']) > 0 and results['documents'][0]:
                logger.info(f"Found {len(results['documents'][0])} documents via semantic search")
                logger.info(f"First result metadata: {results.get('metadatas', [[]])[0]}")
                return results['documents'][0]
            else:
                logger.warning("No documents found in semantic search")
                return []
                
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            return []
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        logger.exception("Full traceback:")
        return []

class AgenticDocumentRetriever:
    def __init__(self):
        self.validator = get_llm(role="validator")
        
    def retrieve_documents(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Agentic document retrieval with query planning and result validation.
        
        Args:
            user_query: Raw user input text
        
        Returns:
            A list of validated and ranked documents
        """
        try:
            # Step 1: Get query plan and expansions
            query_info = expander.expand_query(user_query)
            
            # Step 2: Retrieve documents for each sub-query
            all_documents = []
            for expanded_query in query_info["expanded_queries"]:
                results = collection.query(
                    query_texts=[expanded_query],
                    n_results=3  # Fewer per sub-query as we'll combine them
                )
                
                for i in range(len(results["ids"][0])):
                    doc = {
                        "content": results["documents"][0][i],
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                        "source_path": results["metadatas"][0][i].get("source_path", ""),
                        "score": results["distances"][0][i],
                        "query_matched": expanded_query
                    }
                    all_documents.append(doc)
            
            # Step 3: Validate and rank documents
            validated_docs = self._validate_documents(
                all_documents, 
                user_query, 
                query_info["query_plan"]
            )
            
            # Step 4: Remove duplicates and sort by relevance
            final_docs = self._deduplicate_and_rank(validated_docs)
            
            return final_docs[:5]  # Return top 5 most relevant documents

        except Exception as e:
            print(f"[AgenticDocumentRetriever Error] {e}")
            return []
            
    def _validate_documents(
        self, 
        documents: List[Dict[str, Any]], 
        query: str,
        query_plan: str
    ) -> List[Dict[str, Any]]:
        """Validate document relevance and quality."""
        prompt = f"""
        Validate the following retrieved documents for the query:
        
        Query: {query}
        Query Plan: {query_plan}
        
        For each document, assess:
        1. Relevance to query intent
        2. Information completeness
        3. Reliability of source
        4. Potential missing context
        
        Return a validation score (0-1) for each document.
        """
        
        # Process documents in batches to avoid context length issues
        validated_docs = []
        for doc in documents:
            validation = self.validator.invoke(
                prompt + f"\n\nDocument Content: {doc['content'][:500]}..."
            )
            try:
                # Extract validation score from response
                score = float(validation.content.strip())
                doc["validation_score"] = score
                validated_docs.append(doc)
            except:
                doc["validation_score"] = 0.0
                validated_docs.append(doc)
                
        return validated_docs
        
    def _deduplicate_and_rank(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicates and rank by combined relevance score."""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            # Create a content hash for deduplication
            content_hash = hash(doc["content"])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                # Combine retrieval and validation scores
                doc["combined_score"] = (
                    (1 - doc["score"]) * 0.6 +  # Convert distance to similarity
                    doc["validation_score"] * 0.4
                )
                unique_docs.append(doc)
        
        # Sort by combined score
        return sorted(unique_docs, key=lambda x: x["combined_score"], reverse=True)

# Initialize the retriever
document_retriever = AgenticDocumentRetriever()

# Update the retrieve_documents function to use the new class
def retrieve_documents(user_query: str) -> List[Dict[str, Any]]:
    return document_retriever.retrieve_documents(user_query)
