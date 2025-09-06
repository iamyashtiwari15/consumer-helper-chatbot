import sys, os
# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# test_rag_pipeline.py
"""
Standalone script to test RAG pipeline: document retrieval + response generation.
"""
from backend.agents.rag_agent.document_retriever import retrieve_documents
from backend.agents.rag_agent.response_generator import ResponseGenerator

# Sample queries to test
QUERIES = [
    "What does Section 33 say?",
    "Tell me about consumer rights",
    "How to file a complaint?",
    "What is the penalty for fraud?",
    "hello"
]

def test_rag(query):
    print(f"\n=== Testing query: {query} ===")
    docs, classification = retrieve_documents(query)
    print(f"Retrieved {len(docs)} documents.")
    print(f"Classification: {classification}")
    rg = ResponseGenerator()
    response = rg.generate_response(query, docs, classification)
    print(f"\nResponse:\n{response['response']}")
    print(f"Sources: {response.get('sources', [])}")
    print(f"Confidence: {response.get('confidence', None)}")
    print(f"Verification: {response.get('verification_result', None)}")

if __name__ == "__main__":
    for q in QUERIES:
        test_rag(q)
