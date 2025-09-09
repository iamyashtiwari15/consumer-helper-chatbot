
# --- AGENT DECISION MODULE ---
import logging
import os
import uuid
from agents.vision_agents.image_analysis_agent import analyze_image
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from agents.llm_loader import get_llm
from agents.workflow_manager import WorkflowManager
from agents.guardrails import LocalGuardrails

logging.basicConfig(level=logging.DEBUG)
llm = get_llm()
manager = WorkflowManager("agents/rag_agent/rag_db")
guard = LocalGuardrails(llm)

class GraphState(TypedDict):
    input: str
    image: Optional[bytes]
    image_type: str
    agent_name: str
    response: str
    workflow_response: dict
    involved_agents: List[str]
    input_type: Literal["text", "image"]
    bypass_guardrails: bool
    messages: List[BaseMessage]
    chat_history: Optional[List[dict]]

# Node 1: Guardrails check
def guardrails_node(state: GraphState):
    if state["input_type"] == "image":
        return {**state, "bypass_guardrails": True}
    is_safe, result = guard.check_input(state["input"])
    if not is_safe:
        return {
            **state,
            "bypass_guardrails": False,
            "agent_name": "GUARDRAILS_BLOCK",
            "response": result.content,
            "involved_agents": state.get("involved_agents", []) + ["GUARDRAILS_BLOCK"],
            "messages": state["messages"] + [AIMessage(content=result.content)],
        }
    return {**state, "bypass_guardrails": True}

# Node 2: Image detection
def image_detection_node(state: GraphState):
    if state["input_type"] != "image":
        return state
    return {**state, "image_type": "generic", "agent_name": "DOCUMENT_ANALYSIS_AGENT"}

# Node 3: Agent Routing
def route_to_agent(state: GraphState):
    if state["agent_name"] == "GUARDRAILS_BLOCK" or state["input_type"] == "image":
        return state
    # Pass chat_history to WorkflowManager
    chat_history = state.get("chat_history", None)
    result = manager.process_query(state["input"], chat_history=chat_history)
    if result.get("query_type") in ["greeting", "chitchat"]:
        agent_name = "CONVERSATION_AGENT"
    elif result.get("response") is not None:
        agent_name = "RAG_AGENT"
    else:
        agent_name = "CONVERSATION_AGENT"
    logging.info(f"WorkflowManager selected agent: {agent_name}")
    return {**state, "agent_name": agent_name, "workflow_response": result}

# Node 4: Call the routed agent
def call_agent(state: GraphState):
    if state["agent_name"] == "GUARDRAILS_BLOCK":
        return state
    input_text = state.get("input", "")
    messages = state["messages"] + [HumanMessage(content=input_text)]
    agent = state["agent_name"]
    try:
        if agent in ["CONVERSATION_AGENT", "RAG_AGENT"]:
            workflow_response = state.get("workflow_response", {})
            logging.debug(f"[DEBUG] workflow_response: {workflow_response}")
            output = workflow_response if isinstance(workflow_response, dict) else {}
            if not output or not output.get("response"):
                output = {
                    "response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.",
                    "sources": [],
                    "confidence": 0.0
                }
            # Ensure consistent typing
            if "sources" not in output:
                output["sources"] = []
            if "confidence" not in output:
                output["confidence"] = 0.0
            logging.debug(f"[DEBUG] output after fallback: {output}")
        elif agent == "DOCUMENT_ANALYSIS_AGENT":
            try:
                temp_filename = f"temp_{uuid.uuid4().hex}.png"
                with open(temp_filename, "wb") as f:
                    f.write(state["image"])
                result = analyze_image(temp_filename)
                os.remove(temp_filename)
                # Always return a dict
                output = {
                    "response": result if isinstance(result, str) else str(result),
                    "sources": [],
                    "confidence": 0.0
                }
            except Exception as e:
                logging.error(f"Document/image analysis failed: {str(e)}")
                output = {
                    "response": f"❌ Document/image analysis failed: {str(e)}",
                    "sources": [],
                    "confidence": 0.0
                }
        else:
            output = {"response": "⚠️ Could not process your request.", "sources": [], "confidence": 0.0}
        main_response = output["response"]
        messages.append(AIMessage(content=main_response))
        updated_agents = state.get("involved_agents", []) + [agent]
        return {
            **state,
            "response": output,
            "involved_agents": updated_agents,
            "agent_name": agent,
            "messages": messages,
        }
    except Exception as e:
        logging.error(f"Exception in call_agent: {str(e)}")
        output = {
            "response": f"❌ Internal error: {str(e)}",
            "sources": [],
            "confidence": 0.0
        }
        messages.append(AIMessage(content=output["response"]))
        updated_agents = state.get("involved_agents", []) + [agent]
        return {
            **state,
            "response": output,
            "involved_agents": updated_agents,
            "agent_name": agent,
            "messages": messages,
        }

# Build the graph
def build_consumer_rights_agent_graph():
    builder = StateGraph(GraphState)
    builder.add_node("Guardrails", guardrails_node)
    builder.add_node("ImageDetection", image_detection_node)
    builder.add_node("RouteAgent", route_to_agent)
    builder.add_node("CallAgent", call_agent)
    builder.set_entry_point("Guardrails")
    builder.add_edge("Guardrails", "ImageDetection")
    builder.add_edge("ImageDetection", "RouteAgent")
    builder.add_edge("RouteAgent", "CallAgent")
    builder.add_edge("CallAgent", END)
    return builder.compile()

consumer_rights_agent_graph = build_consumer_rights_agent_graph()
