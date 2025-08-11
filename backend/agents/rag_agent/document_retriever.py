# agents/document_retriever.py

from typing import List, Dict, Any
from agents.rag_agent.query_expander import QueryExpander
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

# Step 4: Define document retrieval logic
def retrieve_documents(user_query: str) -> List[Dict[str, Any]]:
    """
    Expands a user query and retrieves relevant documents from ChromaDB.
    
    Args:
        user_query: Raw user input text
    
    Returns:
        A list of documents with relevant content and metadata
    """
    try:
        # Expand the query
        expanded = expander.expand_query(user_query)["expanded_query"]

        # Query the vectorstore
        results = collection.query(
            query_texts=[expanded],
            n_results=5
        )

        # Format the results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "Unknown"),
                "source_path": results["metadatas"][0][i].get("source_path", ""),
                "score": results["distances"][0][i]
            })

        return documents

    except Exception as e:
        print(f"[DocumentRetriever Error] {e}")
        return []
