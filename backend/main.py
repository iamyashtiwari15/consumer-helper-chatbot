from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from agents.llm_loader import get_llm
from agents.guardrails import LocalGuardrails
from agents.workflow_manager import WorkflowManager
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Memory per session
session_memory: dict[str, list] = {}

llm = get_llm()
guard = LocalGuardrails(llm=llm)
manager = WorkflowManager("agents/rag_agent/rag_db")


def convert_history(messages):
    result = []
    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            role = "user" if msg.type == "human" else "assistant"
            result.append({"role": role, "content": msg.content})
    return result

def handle_user_input(
    user_input: Optional[str],
    session_id: str,
    image_bytes: Optional[bytes] = None,
    image_type: Optional[str] = None
) -> dict:
    """Unified handler using LangGraph agent for text and image input."""
    messages = session_memory.get(session_id, [])
    chat_history = convert_history(messages)
    input_type = "image" if image_bytes else "text"
    image_path = None
    if image_bytes and image_type:
        ext = image_type.split("/")[-1]
        unique_filename = f"{uuid.uuid4()}.{ext}"
        image_path = f"temp_{unique_filename}"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

    input_state = {
        "input": user_input or "",
        "image": image_bytes,
        "image_path": image_path,
        "image_type": image_type or "",
        "input_type": input_type,
        "agent_name": "",
        "response": "",
        "involved_agents": [],
        "bypass_guardrails": False,
        "messages": messages,
        "chat_history": chat_history,
    }

    from agents.agent_decision import consumer_rights_agent_graph
    output = consumer_rights_agent_graph.invoke(input_state)
    session_memory[session_id] = output["messages"]
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    return output


@app.post("/chat")
async def chat_endpoint(message: str = Form(...), session_id: str = Form(...)):
    try:
        agent_output = handle_user_input(message, session_id)
        inner_response_obj = agent_output.get("response", {})
        final_text = inner_response_obj.get("response", "Error: Could not find a response.")
        final_sources = inner_response_obj.get("sources", [])
        return JSONResponse(content={"response": final_text, "sources": final_sources})
    except Exception as e:
        return JSONResponse(content={"response": f"⚠️ Error: {str(e)}"}, status_code=500)


@app.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: str = Form("")
):
    try:
        file_bytes = await file.read()
        file_type = file.content_type
        agent_output = handle_user_input(message, session_id, image_bytes=file_bytes, image_type=file_type)
        inner_response_obj = agent_output.get("response", {})
        final_text = inner_response_obj.get("response", "Error: Could not find a response.")
        final_sources = inner_response_obj.get("sources", [])
        return JSONResponse(content={"response": final_text, "sources": final_sources})
    except Exception as e:
        return JSONResponse(content={"response": f"⚠️ Error: {str(e)}"}, status_code=500)
