import logging
from typing import List, Dict, Any, Optional
from agents.llm_loader import get_llm

class ResponseGenerator:
    """
    Generates consumer rights responses based on retrieved legal context and user query.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = get_llm()  # Load LLM from llm_loader
        self.include_sources = True

    def _build_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        table_instructions = """
        Some of the retrieved information is presented in table format (e.g., timelines, refund rates, warranty durations). When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers.
        2. Reformat the table for clarity, ensuring legal terms are clear.
        3. If adding an interpretation column (e.g., "Meaning"), mention it explicitly.
        4. Summarize legal requirements or obligations shown in the tables.
        """

        response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, reply exactly in this format (and nothing else):
           {
               "Insufficient Information", "I don't have enough information."
           }
        3. Do not provide legal advice; only summarize the rights and rules from the provided sources.
        4. Be concise, accurate, and avoid making assumptions not supported by the context.
        5. Format the answer with headings, subheadings, and tables (if applicable) in markdown.
        6. Include a short disclaimer: "This information is for educational purposes and may not be legally binding."
        7. If monetary or date values are provided, use them exactly as they appear in the context.
        """

        history_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history]) if chat_history else ""

        prompt = f"""You are a consumer rights assistant providing accurate and verified information based only on trusted legal sources such as official government websites, consumer protection acts, and related documentation.

Here are the last few messages from our conversation:
{history_text}

The user has asked the following question:
{query}

I've retrieved the following information to help answer this question:
{context}

{table_instructions}
{response_format_instructions}

Based on the provided information, please answer the user's question thoroughly but concisely.
If the information doesn't contain the answer, follow the insufficient information format exactly.
Do not provide any source link that is not present in the context.
"""

        return prompt

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        picture_paths: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        try:
            doc_texts = [doc["content"] for doc in retrieved_docs]
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            prompt = self._build_prompt(query, context, chat_history)
            response = self.response_generator_model.invoke(prompt)

            response_text = response.content.strip()

            # Detect insufficient info case
            if response_text.startswith("{") and "Insufficient Information" in response_text:
                return {
                    "response": response_text,
                    "sources": [],
                    "confidence": 0.0
                }

            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            confidence = self._calculate_confidence(retrieved_docs)

            if self.include_sources:
                response_text += "\n\n##### Source documents:"
                for src in sources:
                    response_text += f"\n- [{src['title']}]({src['path']})"

            if picture_paths:
                response_text += "\n\n##### Reference images:"
                for path in picture_paths:
                    response_text += f"\n- [{path.split('/')[-1]}]({path})"

            return {
                "response": response_text,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": '{ "Insufficient Information", "I don\'t have enough information." }',
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen = set()

        for doc in documents:
            title = doc.get("source")
            path = doc.get("source_path")
            if not title or not path:
                continue

            source_id = f"{title}|{path}"
            if source_id in seen:
                continue

            sources.append({
                "title": title,
                "path": path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            })
            seen.add(source_id)

        sources.sort(key=lambda x: x["score"], reverse=True)
        return [{"title": s["title"], "path": s["path"]} for s in sources]

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0

        keys = ["combined_score", "rerank_score", "score"]
        for key in keys:
            if key in documents[0]:
                scores = [doc.get(key, 0) for doc in documents[:3]]
                return sum(scores) / len(scores)

        return 0.0
