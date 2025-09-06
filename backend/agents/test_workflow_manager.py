from pathlib import Path
from backend.agents.workflow_manager import WorkflowManager

def main():
    db_path = Path("c:/Users/Yash Tiwari/Desktop/consumer helper chat bot/backend/agents/rag_agent/rag_db")
    manager = WorkflowManager(db_path)

    # Test 1: Greeting
    response = manager.process_query("hello")
    print("Test 1 - Greeting Response:")
    print(response)
    print("\n" + "-"*60 + "\n")

    # Test 2: General legal query
    response = manager.process_query("How to file a consumer complaint?")
    print("Test 2 - Legal Query Response:")
    print(response)
    print("\n" + "-"*60 + "\n")

    # Test 3: Section-specific query
    response = manager.process_query("Explain Section 33 of Consumer Protection Act 2019")
    print("Test 3 - Section Query Response:")
    print(response)
    print("\n" + "-"*60 + "\n")

    # Test 4: (Optional) Image + query (uncomment when image agent is integrated)
    # response = manager.process_query("Is this bill overcharging?", image_path="path/to/bill.jpg")
    # print("Test 4 - Image Query Response:")
    # print(response)
    # print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()
