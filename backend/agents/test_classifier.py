from agents.rag_agent.query_classifier import QueryClassifier

def test_new_classifier():
    """Quick test of the simplified classifier"""
    print("🧪 Testing Enhanced Query Classifier\n")
    
    classifier = QueryClassifier()
    
    test_queries = [
        "Hello there",
        "My phone is broken", 
        "I want a refund for defective product",
        "I was scammed online",
        "What does Section 15 say?",
        "How to file complaint?",
        "My consumer rights?",
        "Phone number of consumer court?",
        "Thanks for helping"
    ]
    
    for query in test_queries:
        result = classifier.classify_query(query)
        strategy = classifier.get_response_strategy(result)
        
        print(f"Query: '{query}'")
        print(f"  Type: {result.query_type}")
        print(f"  Topics: {result.topics}")
        print(f"  Strategy: RAG={strategy['use_rag']}, Web={strategy['use_web']}, Skip={strategy['skip_response']}")
        if result.requires_external_sources:
            print(f"  🌐 Needs external sources")
        print()

if __name__ == "__main__":
    test_new_classifier()