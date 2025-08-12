# agents/document_retriever.py

from typing import List, Dict, Any
from agents.rag_agent.query_expander import QueryExpander
from agents.rag_agent.llm_loader import get_llm
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Step 1: Define a wrapper that matches ChromaDB's required interface
class ChromaCompatibleEmbeddingFunction:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(input)

    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

# Step 2: Initialize query expander and embedding model
expander = QueryExpander()
embedding_function = ChromaCompatibleEmbeddingFunction()

# Step 3: Setup ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="agents/rag_agent/rag_db")  # use a persistent path
collection = chroma_client.get_or_create_collection(
    name="rag_db",
    embedding_function=embedding_function
)

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
