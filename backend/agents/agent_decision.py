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
from agents.rag_agent.response_generator import ResponseGenerator
from agents.web_search.web_search_processor import WebSearchProcessor
from agents.guardrails import LocalGuardrails
from agents.rag_agent.document_retriever import retrieve_documents

logging.basicConfig(level=logging.INFO)
# Initialize agents with specific roles
llm = get_llm()  # Use planner role for decision making
rag_agent = ResponseGenerator()
web_agent = WebSearchProcessor()
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

    # First check if relevant documents are available
    try:
        docs = retrieve_documents(state["input"])
        has_relevant_docs = len(docs) > 0
    except Exception as e:
        logging.error(f"Error checking document availability: {str(e)}")
        has_relevant_docs = False

    if has_relevant_docs:
        logging.info("Found relevant documents in RAG system, using RAG_AGENT")
        return {**state, "agent_name": "RAG_AGENT"}

    prompt = f"""
You are a decision-making agent that decides which specialist agent should respond to the user's consumer rights query.

List of agents:
- CONVERSATION_AGENT: Handles general consumer rights queries, policy clarifications, or ambiguous input.
- WEB_SEARCH_PROCESSOR_AGENT: Searches trusted consumer rights websites and complaint portals for real-time information, recent cases, or specific examples.

Given the user's input: \"{state['input']}\", respond ONLY as a JSON object like: {{"agent_name": "CONVERSATION_AGENT"}}

Choose WEB_SEARCH_PROCESSOR_AGENT if the query:
- Requires recent/current information
- Asks about specific cases or examples
- References specific companies or products

Choose CONVERSATION_AGENT if the query:
- Seeks general advice or explanations
- Asks about basic consumer rights concepts
- Is conversational or unclear
"""
    # Send only the prompt text instead of message history
    decision = llm.invoke(prompt)

    chosen = "CONVERSATION_AGENT"  # default fallback
    try:
        response_text = decision.content if hasattr(decision, 'content') else str(decision)
        decision_dict = json.loads(response_text)
        chosen = decision_dict.get("agent_name", "CONVERSATION_AGENT")
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"⚠️ Error parsing agent decision: {str(e)}\nResponse: {repr(decision)}")

    logging.info(f"Selected agent: {chosen} (no relevant documents found in RAG system)")
    return {**state, "agent_name": chosen}

# Node 4: Call the routed agent
def call_agent(state: GraphState):
    if state["agent_name"] == "GUARDRAILS_BLOCK":
        return state

    input_text = state.get("input", "")
    messages = state["messages"] + [HumanMessage(content=input_text)]
    agent = state["agent_name"]

    try:
        if agent == "CONVERSATION_AGENT":
            response = get_consumer_rights_response(messages)  # renamed function for domain
            output = getattr(response, "content", response)  # fallback to str if needed

        elif agent == "RAG_AGENT":
            retrieved_docs = retrieve_documents(input_text)
            if not retrieved_docs:
                logging.warning("No documents retrieved from RAG system despite initial check")
                # Fallback to web search with explanation
                fallback = web_agent.process_web_results(input_text)
                fallback_text = getattr(fallback, "content", fallback)
                output = f"""### Web Search Results
*Note: While we typically use our legal document database, we've searched trusted online sources for this query.*

{fallback_text}

---"""
                agent = "WEB_SEARCH_PROCESSOR_AGENT"
            else:
                result = rag_agent.generate_response(input_text, retrieved_docs)
                output = result.get("response", "")
                
                # If RAG response indicates insufficient information, enhance with web search
                if "insufficient information" in output.lower():
                    fallback = web_agent.process_web_results(input_text)
                    fallback_text = getattr(fallback, "content", fallback)
                    output = f"""### Local Document Information
{output}

### Additional Web Search Results
*To provide more comprehensive information, we've also searched trusted online sources:*

{fallback_text}

---"""

        elif agent == "WEB_SEARCH_PROCESSOR_AGENT":
            response = web_agent.process_web_results(input_text)
            response_text = getattr(response, "content", response)
            # Format web search results in markdown
            output = f"""### Web Search Results\n\n{response_text}\n\n---\n"""

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
