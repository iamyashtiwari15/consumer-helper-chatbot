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
import ast

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

    # Append user message to session history
    if user_input:
        messages.append(HumanMessage(content=user_input))

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


    # Post-process agent messages to ensure only main text response is stored
    processed_messages = []
    for msg in output["messages"]:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            if msg.type == "ai":
                content = msg.content
                # If content is a dict, get 'response' key
                if isinstance(content, dict) and "response" in content:
                    msg = AIMessage(content=content["response"])
                # If content is a string that looks like a dict, try to parse and extract 'response'
                elif isinstance(content, str) and content.strip().startswith("{"):
                    try:
                        import json
                        obj = json.loads(content)
                        if isinstance(obj, dict) and "response" in obj:
                            msg = AIMessage(content=obj["response"])
                    except Exception:
                        # Try to parse as Python dict string
                        try:
                            obj = ast.literal_eval(content)

                            if isinstance(obj, dict) and "response" in obj:
                                msg = AIMessage(content=obj["response"])
                        except Exception:
                            pass
            processed_messages.append(msg)
        else:
            processed_messages.append(msg)
    session_memory[session_id] = processed_messages
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    return output


@app.post("/chat")
async def chat_endpoint(message: str = Form(...), session_id: str = Form(...)):
    try:
        import asyncio
        # Run the blocking handle_user_input in a thread to avoid blocking the async event loop
        agent_output = await asyncio.to_thread(handle_user_input, message, session_id)
        inner_response_obj = agent_output.get("response", {})
        final_text = inner_response_obj.get("response", "Error: Could not find a response.")
        final_sources = inner_response_obj.get("sources", [])
        return JSONResponse(content={"response": final_text, "sources": final_sources})
    except Exception as e:
        return JSONResponse(content={"response": f"⚠️ Error: {str(e)}"}, status_code=500)


# New endpoint to fetch chat history for a session
from fastapi import Request
@app.post("/history")
async def history_endpoint(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        if not session_id:
            return JSONResponse(content={"error": "Missing session_id"}, status_code=400)
        messages = session_memory.get(session_id, [])
        # Convert messages to frontend format
        history = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = "user" if msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
        return JSONResponse(content={"history": history})
    except Exception as e:
        return JSONResponse(content={"error": f"Error: {str(e)}"}, status_code=500)


@app.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: str = Form("")
):
    try:
        import asyncio
        file_bytes = await file.read()
        file_type = file.content_type
        # Run the blocking handle_user_input in a thread for async performance
        agent_output = await asyncio.to_thread(handle_user_input, message, session_id, image_bytes=file_bytes, image_type=file_type)
        inner_response_obj = agent_output.get("response", {})
        final_text = inner_response_obj.get("response", "Error: Could not find a response.")
        final_sources = inner_response_obj.get("sources", [])
        return JSONResponse(content={"response": final_text, "sources": final_sources})
    except Exception as e:
        return JSONResponse(content={"response": f"⚠️ Error: {str(e)}"}, status_code=500)
