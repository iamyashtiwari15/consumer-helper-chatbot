---
name: memory-agent
description: 'Agentic memory workflow for short-term, long-term, episodic, and semantic memory using LangGraph and vector stores. Use for personalization, cross-session continuity, and safe memory lifecycle management.'
argument-hint: 'What memory behavior do you want the agent to support?'
user-invocable: true
disable-model-invocation: false
---

# Memory Agent

## What Is a Memory Agent?
A memory agent stores, retrieves, and reasons over information across interactions.
It enables personalization and continuity across turns and sessions.
Without memory, every interaction starts from zero.

## Memory Taxonomy
- Short-term: in-context buffer or sliding window
- Long-term: episodic, semantic, procedural memory
- Cross-session: external database and vector store
- Cross-agent: shared memory store for multiple agents

## 1. Short-Term Memory

### Buffer memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### Sliding window memory
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10, return_messages=True)
```

### Summary memory
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)
```

## 2. Long-Term Memory (LangGraph recommended)
Use checkpointers so memory is persisted as part of graph state.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.redis import RedisSaver

memory = SqliteSaver.from_conn_string("agent_memory.db")
memory = RedisSaver.from_conn_string("redis://localhost:6379")

graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user-abc-session-1"}}
graph.invoke(inputs, config=config)
```

### Episodic memory
```python
def save_episode(thread_id: str, summary: str):
    episodes_db.upsert({
        "thread_id": thread_id,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    })

def load_episodes(thread_id: str, k: int = 5) -> list[str]:
    return episodes_db.query(thread_id, limit=k)
```

### Semantic memory
```python
from langchain_community.vectorstores import Qdrant

def remember_preference(user_id: str, fact: str):
    vectorstore.add_texts(
        texts=[fact],
        metadatas=[{"user_id": user_id, "type": "preference"}]
    )

def recall_relevant(user_id: str, query: str, k: int = 3) -> list[str]:
    results = vectorstore.similarity_search(
        query,
        k=k,
        filter={"user_id": user_id}
    )
    return [r.page_content for r in results]
```

## 3. Memory Management Patterns

### Retrieve -> Inject -> Update
```python
def agent_with_memory(user_id: str, user_message: str) -> str:
    past_context = recall_relevant(user_id, user_message)

    system = f"""You are a helpful assistant.

What you remember about this user:
{chr(10).join(past_context)}"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_message)
    ])

    new_facts = extract_facts(user_message, response.content)
    for fact in new_facts:
        remember_preference(user_id, fact)

    return response.content
```

### LangGraph memory node flow
- Recall node fetches relevant memories.
- Agent node answers with memory-augmented prompt.
- Save node extracts and persists new facts.

## 4. External Memory Libraries
Mem0 can act as a persistent memory layer:

```python
from mem0 import Memory

m = Memory()
m.add("User prefers bullet point summaries", user_id="alice")
results = m.search("How does Alice like content formatted?", user_id="alice")
```

## 5. Fact Extraction
```python
def extract_facts_with_llm(messages: list) -> list[str]:
    extraction_prompt = f"""
    Given this conversation, extract 0-5 specific facts about the user
    that would be useful to remember in future conversations.
    Return only a JSON list of strings. Return [] if nothing notable.

    Conversation:
    {format_messages(messages)}
    """
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    return json.loads(response.content)
```

## 6. Memory Security
- Encrypt sensitive fields before storing.
- Filter by user_id on every retrieval.
- Never store secrets (API keys, passwords) in raw memory state.
- Avoid logging full raw state containing PII.

## 7. Memory Health
Common issues and fixes:
- Memory bloat: summarize old turns.
- Context overflow: trim and externalize memory.
- Serialization errors: keep state JSON-serializable.
- Cross-user leakage: enforce user_id filtering.
- Stale memory: add timestamps and expiry policies.

## 8. Choosing Memory Type
- Single-turn QnA: no memory required.
- Multi-turn same session: buffer or sliding window.
- Long sessions: summary memory.
- Returning users: semantic plus episodic memory.
- High concurrency production: Redis or Postgres checkpointing.

## Dependencies
```bash
pip install langgraph langchain langchain-community
pip install langgraph-checkpoint-sqlite langgraph-checkpoint-redis
pip install mem0ai
pip install qdrant-client chromadb pgvector
```
