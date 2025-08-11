

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
import shutil

from agents.guardrails import LocalGuardrails
from agents.agent_decision import consumer_rights_agent_graph

from agents.llm_loader import get_llm
from langchain_core.messages import BaseMessage

import os

load_dotenv()

llm = get_llm()
guard = LocalGuardrails(llm=llm)

app = FastAPI()

# ✅ CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Memory for session chat history
session_memory: dict[str, list[BaseMessage]] = {}

# ✅ Core input handler
def handle_user_input(
    user_input: Optional[str],
    session_id: str,
    image_bytes: Optional[bytes] = None,
    image_type: Optional[str] = None
) -> str:
    messages = session_memory.get(session_id, [])

    # Detect type
    input_type = "image" if image_bytes else \
                 "text"

    image_path = None
    if image_bytes:
        ext = image_type.split("/")[-1]
        image_path = f"temp_uploaded_image.{ext}"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

    # Prepare input state
    input_state = {
        "input": user_input or "",
        "image": image_bytes,  # raw bytes
        "image_path": image_path,  # path for analysis agent
        "image_type": image_type or "",
        "input_type": input_type,
        "agent_name": "",
        "response": "",
        "involved_agents": [],
        "bypass_guardrails": False,
        "messages": messages,
    }

    # Run through decision graph
    output = consumer_rights_agent_graph.invoke(input_state)

    # Update chat history
    session_memory[session_id] = output["messages"]

    # Clean up temporary image file
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return output["response"]

# ✅ Route: Text-only chat
@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(...)):
    try:
        reply = handle_user_input(user_input=message, session_id=session_id)
        return {"reply": reply}
    except Exception as e:
        print("Error in /chat:", e)
        return JSONResponse(status_code=500, content={"reply": "Server error while processing your message."})

# ✅ Route: Image + optional message
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        image_type = file.content_type

        # Combined image + optional message
        reply = handle_user_input(
            user_input=message,
            session_id=session_id,
            image_bytes=contents,
            image_type=image_type
        )
        return {"reply": reply}
    except Exception as e:
        print("Error in /upload:", e)
        return JSONResponse(status_code=500, content={"reply": "Server error while processing your image."})
