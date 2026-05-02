---
name: langchain
description: 'LangChain development workflow for models, prompts, LCEL chains, retrieval chains, tools, agents, memory, and streaming. Use for linear LLM applications and simple RAG implementations.'
argument-hint: 'What LangChain flow are you building (chain, RAG, tools, or agent)?'
user-invocable: true
disable-model-invocation: false
---

# LangChain

## What Is LangChain?
LangChain is an open-source framework for building LLM-powered applications.
It provides composable units: models, prompts, chains, retrievers, tools, and memory.
Use LangChain for linear or sequential workflows. For complex looping stateful agents, use LangGraph.

## Core Concepts

Prompt -> LLM -> Output Parser

Chain = sequence of the above.
Agent = chain plus tools and a decide-next-step loop.

## 1. Models and Chat Models
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o", temperature=0)
```

## 2. Prompts
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

partial_prompt = prompt.partial(context="Company handbook v3.2")
```

## 3. Chains (LCEL)
Use the pipe operator to connect components.

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is our leave policy?", "history": []})
```

Useful parsers:
- `StrOutputParser`
- `JsonOutputParser`
- `PydanticOutputParser`
- `CommaSeparatedListOutputParser`

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

parser = JsonOutputParser(pydantic_object=Answer)
chain = prompt | llm | parser
```

## 4. Retrieval Chains (RAG)
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)
result = rag_chain.invoke({"input": "What are the refund terms?"})
```

## 5. Tools and Tool Calling
```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the internal knowledge base for relevant documents."""
    return db.query(query)

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    return email_client.send(to=to, subject=subject, body=body)

llm_with_tools = llm.bind_tools([search_database, send_email])
```

Tool rules:
- Use descriptive snake_case names.
- Keep docstrings explicit because models read them.
- Add type hints for schema generation.

## 6. Agents (ReAct style)
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What's the weather in Delhi?", "chat_history": []})
```

## 7. Memory
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
```

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)
```

```python
from langchain.memory import VectorStoreRetrieverMemory

memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever(k=5))
```

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",
    token_counter=llm,
    include_system=True
)
messages = trimmer.invoke(all_messages)
```

## 8. Document Loaders
```python
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    NotionDirectoryLoader,
    UnstructuredFileLoader,
)

loader = PyPDFLoader("report.pdf")
docs = loader.load()
```

## 9. Streaming
```python
for chunk in chain.stream({"question": "Explain RAG"}):
    print(chunk, end="", flush=True)

async for chunk in chain.astream({"question": "Explain RAG"}):
    print(chunk, end="", flush=True)
```

## 10. LangSmith Tracing
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

## LangChain vs LangGraph
Use LangChain for sequential QnA, summarization, and simple RAG pipelines.
Use LangGraph for looping, branching, human approvals, and multi-agent workflows.

## Dependencies
```bash
pip install langchain langchain-core langchain-community
pip install langchain-openai langchain-anthropic
pip install langsmith
```
