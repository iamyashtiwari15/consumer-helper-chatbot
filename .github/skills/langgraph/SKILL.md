---
name: langgraph
description: 'LangGraph workflow for building stateful agents with nodes, edges, reducers, tools, checkpointers, and human-in-the-loop. Use for branching, loops, multi-step decisions, and multi-agent orchestration.'
argument-hint: 'What LangGraph agent workflow do you want to build?'
user-invocable: true
disable-model-invocation: false
---

# LangGraph

## What Is LangGraph?
LangGraph is an orchestration framework built on top of LangChain.
It models workflows as directed graphs with shared typed state.
Use it when you need loops, conditional routing, human approvals, or multi-agent coordination.

## Core Primitives
StateGraph = State + Nodes + Edges

- State: shared typed dictionary
- Node: function that reads and updates state
- Edge: transition between nodes
- Reducer: merge logic for state updates
- Checkpointer: persistence across steps and sessions

## 1. Define State
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    step_count: int
    retrieved_docs: list
```

## 2. Build the Graph
```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_llm)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_relevance)
workflow.add_node("generate", generate_answer)

workflow.add_edge(START, "agent")
workflow.add_edge("generate", END)

workflow.add_conditional_edges(
    "agent",
    decide_next_step,
    {
        "retrieve": "retrieve",
        "respond": END,
    }
)

graph = workflow.compile()
```

## 3. Node Functions
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def retrieve_documents(state: AgentState) -> dict:
    query = state["messages"][-1].content
    docs = vectorstore.similarity_search(query, k=4)
    return {"retrieved_docs": docs}

def generate_answer(state: AgentState) -> dict:
    context = "\n".join(d.page_content for d in state["retrieved_docs"])
    prompt = f"Context:\n{context}\n\nQuestion: {state['messages'][-1].content}"
    answer = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [answer]}
```

## 4. Routing
```python
def decide_next_step(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "retrieve"
    return "respond"
```

Helper:
```python
from langgraph.prebuilt import tools_condition
workflow.add_conditional_edges("agent", tools_condition)
```

## 5. Tools in LangGraph
```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return web_search(query)

tools = [search_web]
llm_with_tools = llm.bind_tools(tools)
workflow.add_node("tools", ToolNode(tools))
```

## 6. Memory and Persistence
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [HumanMessage("Hello")]}, config=config)
```

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    graph = workflow.compile(checkpointer=memory)
```

```python
from langgraph.checkpoint.redis import RedisSaver

saver = RedisSaver.from_conn_string("redis://localhost:6379")
```

## 7. Human-in-the-Loop
```python
from langgraph.types import interrupt

def human_approval(state: AgentState) -> dict:
    decision = interrupt({
        "question": "Approve this action?",
        "proposed_action": state["messages"][-1].content
    })
    return {"messages": [HumanMessage(content=decision)]}
```

## 8. Multi-Agent Patterns
- Supervisor: one router agent delegates to specialists.
- Swarm: peers collaborate directly.
- Pipeline: strict stage-by-stage handoff.

## 9. Agentic RAG Pattern
Typical flow:
- Agent decides retrieve vs direct response.
- Retrieval tool fetches docs.
- Grading step checks relevance.
- Query may be rewritten.
- Generation produces grounded answer.

## Production Checklist
- Define typed state with TypedDict.
- Use add_messages reducer for messages.
- Assign unique thread_id per user/session.
- Use SqliteSaver for dev and Redis/Postgres for production.
- Keep state JSON-serializable and compact.
- Avoid sensitive data in raw state.
- Enable LangSmith tracing.
- Visualize graph with draw_mermaid.

## Debugging
```python
print(graph.get_graph().draw_mermaid())

for step in graph.stream(inputs, config):
    print(step)

state_snapshot = graph.get_state_history(config)
```

## Dependencies
```bash
pip install langgraph langchain-core langchain-openai
pip install langgraph-checkpoint-sqlite
pip install langgraph-checkpoint-redis
pip install langsmith
```
