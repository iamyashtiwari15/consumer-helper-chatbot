
from backend.agents.vision_agents.image_analysis_agent import analyze_image

image_path = "backend/agents/vision_agents/test_img_3.png"

# If you want to test with workflow manager, import and instantiate it here
# from agents.workflow_manager import WorkflowManager
# workflow_manager = WorkflowManager(db_path="your_db_path")
workflow_manager = None  # Set to actual instance if needed

result = analyze_image(image_path, workflow_manager=workflow_manager)
print("OCR Text:")
print(result["ocr_text"])
print("\nSummary:")
print(result["ocr_summary"])
if result["workflow_response"]:
    print("\nWorkflow Response:")
    print(result["workflow_response"])
