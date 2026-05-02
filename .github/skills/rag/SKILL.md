---
name: rag
description: 'RAG (retrieval-augmented generation) workflow for ingesting documents, chunking, embedding, retrieval, reranking, and grounded answer generation. Use for building or improving document QnA systems with citations and production quality checks.'
argument-hint: 'What RAG use case or pipeline do you want to build or improve?'
user-invocable: true
disable-model-invocation: false
---

# RAG (Retrieval-Augmented Generation)

## What Is RAG?
RAG grounds LLM responses in external, trusted documents.
The model reads before it writes: retrieve relevant passages, inject them as context, then generate.
RAG keeps model weights unchanged and updates knowledge through the data layer.

## Core Pipeline

INGEST -> INDEX -> RETRIEVE -> GENERATE

## 1. Ingest and Chunk
Split source documents into semantically meaningful chunks.

| Strategy | When to use |
|---|---|
| Fixed-size (for example 512 tokens, 50 overlap) | Quick baseline for uniform text |
| Recursive character splitter | General prose; respects paragraphs and sentences |
| Semantic chunking | Better recall by grouping by meaning |
| Heading-aware structural chunking | PDFs and sectioned documents |
| Agentic chunking | Highest quality, highest cost |

Rules:
- Keep chunks short enough for precision and long enough for meaning.
- Use overlap around 10-20 percent.
- Attach metadata: source, page, date, owner, sensitivity.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)
```

## 2. Embed and Index
Convert chunks to vectors and store in a vector database.

Common embedding choices:
- `voyage-3-large`
- `text-embedding-3-large` (OpenAI)
- `embed-v3-english` (Cohere)
- `bge-m3` (open source)

Common vector database choices:
- Pinecone for managed production
- Weaviate for hybrid plus graph use cases
- Milvus for billion-scale workloads
- Qdrant for strong metadata filtering
- pgvector for Postgres-native deployments
- ChromaDB for local prototyping
- FAISS for in-memory local search

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Qdrant.from_documents(chunks, embeddings, location=":memory:")
```

## 3. Retrieve

### Dense Retrieval (baseline)
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### Hybrid Retrieval (recommended)
Combine BM25 and dense vectors, then fuse rankings.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=5)
dense = vectorstore.as_retriever(search_kwargs={"k": 5})
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.3, 0.7])
```

### Re-ranking (precision boost)
```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="ms-marco-MiniLM-L-6-v2"),
    top_n=3
)
```

### Query Expansion (multi-query)
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

## 4. Generate
Stuff retrieved chunks into the prompt and call the LLM.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    chain_type="stuff",
    retriever=hybrid,
    return_source_documents=True
)
result = qa.invoke({"query": "What is our refund policy?"})
```

Chain type options:
- `stuff`: default when all chunks fit.
- `map_reduce`: many chunks.
- `refine`: iterative improvement.

## Advanced Patterns
- Agentic RAG: agent decides whether to retrieve, rewrite query, or validate answer.
- GraphRAG: graph-based retrieval for entity-heavy domains.
- Embedding/query caching for latency and cost control.

## Production Checklist
- Define end-to-end latency target.
- Use hybrid retrieval (not vector-only) for production.
- Add reranker for precision-sensitive tasks.
- Re-embed on content change events.
- Store source metadata and sensitivity tags.
- Track retrieval metrics (NDCG, MRR, relevance).
- Defend against prompt injection from retrieved text.
- Version corpus and keep audit trails for regulated domains.

## Performance Targets
- High-quality optimized RAG can substantially improve accuracy vs non-grounded responses.
- Retrieval latency under 100 ms is feasible with indexing and caching.

## Dependencies
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters
pip install qdrant-client chromadb faiss-cpu
pip install rank-bm25 sentence-transformers
```
