from agents.vision_agents.image_analysis_agent import analyze_image
import logging
import os
import uuid

import json
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from agents.llm_loader import get_llm
from agents.consumer_rights_chat_agent import get_consumer_rights_response  # renamed from medical
from agents.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
llm = get_llm()  # Use planner role for decision making
manager = WorkflowManager("agents/rag_agent/rag_db")
from agents.guardrails import LocalGuardrails
guard = LocalGuardrails(llm)
# Define shared state
class GraphState(TypedDict):
    input: str
    image: Optional[bytes]
    image_type: str
    agent_name: str
    response: str
    involved_agents: List[str]
    input_type: Literal["text", "image"]
    bypass_guardrails: bool
    messages: List[BaseMessage]

# Node 1: Guardrails check
def guardrails_node(state: GraphState):
    if state["input_type"] == "image":
        return {**state, "bypass_guardrails": True}  # Skip guardrails on images (or change as needed)

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

# Node 2: Image detection stub
def image_detection_node(state: GraphState):
    if state["input_type"] != "image":
        return state
    return {**state, "image_type": "generic", "agent_name": "DOCUMENT_ANALYSIS_AGENT"}  # renamed agent for consumer rights domain

# Node 3: Agent Routing
def route_to_agent(state: GraphState):
    if state["agent_name"] == "GUARDRAILS_BLOCK" or state["input_type"] == "image":
        return state  # Already routed in image detection or guardrails block

    # Delegate all text query handling to WorkflowManager
    result = manager.process_query(state["input"])
    # Always pass WorkflowManager's result as workflow_response
    if result.get("query_type") in ["greeting", "chitchat"] or (result.get("response") and not result.get("sources")):
        agent_name = "CONVERSATION_AGENT"
    elif result.get("sources"):
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
        if agent == "CONVERSATION_AGENT" or agent == "RAG_AGENT":
            workflow_response = state.get("workflow_response", {})
            logging.info(f"[DEBUG] workflow_response: {workflow_response}")
            output = workflow_response.get("response", "")
            logging.info(f"[DEBUG] output before fallback: {output}")
            if not output:
                output = "Sorry, no information was found for your query. Please try rephrasing or ask about another topic."
            logging.info(f"[DEBUG] output after fallback: {output}")
        elif agent == "DOCUMENT_ANALYSIS_AGENT":
            try:
                temp_filename = f"temp_{uuid.uuid4().hex}.png"
                with open(temp_filename, "wb") as f:
                    f.write(state["image"])

                output = analyze_image(temp_filename)

                os.remove(temp_filename)
            except Exception as e:
                logging.error(f"Document/image analysis failed: {str(e)}")
                output = f"❌ Document/image analysis failed: {str(e)}"
        else:
            output = "⚠️ Could not process your request."

        messages.append(AIMessage(content=output))
        updated_agents = state.get("involved_agents", []) + [agent]

        return {
            **state,
            "response": output,
            "involved_agents": updated_agents,
            "agent_name": agent,
            "messages": messages,
        }
    except AttributeError as e:
        logging.error(f"AttributeError in call_agent: {str(e)} | output: {locals().get('output', None)}")
        output = f"❌ Internal error: {str(e)}"
        messages.append(AIMessage(content=output))
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
        output = f"❌ Internal error: {str(e)}"
        messages.append(AIMessage(content=output))
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

# Compile the final graph
consumer_rights_agent_graph = build_consumer_rights_agent_graph()
