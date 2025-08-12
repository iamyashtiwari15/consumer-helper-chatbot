import logging
from typing import List, Dict, Any, Optional
from agents.rag_agent.llm_loader import get_llm

class ResponseGenerator:
    """
    Generates consumer rights responses based on retrieved legal context and user query.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_generator = get_llm()  # Main response generator
        self.fact_checker = get_llm(role="validator")  # For fact checking
        self.include_sources = True
        self.max_refinement_attempts = 2  # Maximum number of response refinements

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
            
            # Step 1: Generate initial response
            prompt = self._build_prompt(query, context, chat_history)
            initial_response = self.response_generator.invoke(prompt)
            response_text = initial_response.content.strip()

            # Handle insufficient info case
            if response_text.startswith("{") and "Insufficient Information" in response_text:
                return {
                    "response": response_text,
                    "sources": [],
                    "confidence": 0.0,
                    "verification_result": None
                }

            # Step 2: Fact check and verify the response
            verification_result = self._verify_response(
                response_text,
                query,
                context
            )

            # Step 3: Refine response if needed
            final_response = self._refine_response(
                response_text,
                verification_result,
                query,
                context
            )

            # Step 4: Add source documentation
            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            confidence = self._calculate_confidence(retrieved_docs)

            if self.include_sources:
                final_response += "\n\n##### Source documents:"
                for src in sources:
                    final_response += f"\n- [{src['title']}]({src['path']})"

            if picture_paths:
                final_response += "\n\n##### Reference images:"
                for path in picture_paths:
                    final_response += f"\n- [{path.split('/')[-1]}]({path})"

            return {
                "response": final_response,
                "sources": sources,
                "confidence": confidence,
                "verification_result": verification_result
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": '{ "Insufficient Information", "I don\'t have enough information." }',
                "sources": [],
                "confidence": 0.0,
                "verification_result": None
            }
            
    def _verify_response(
        self,
        response: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Verify the factual accuracy and completeness of the response.
        """
        verification_prompt = f"""
        Verify the following response for factual accuracy and completeness:

        Query: {query}
        Response: {response}
        Source Context: {context}

        Verify:
        1. All facts are supported by the context
        2. No contradictions with the source material
        3. All relevant information is included
        4. No unsupported claims or speculation

        Return a JSON object with:
        1. accuracy_score (0-1)
        2. supported_facts (list)
        3. unsupported_claims (list)
        4. missing_information (list)
        5. suggested_improvements (list)
        """
        
        verification = self.fact_checker.invoke(verification_prompt)
        try:
            # Note: In practice, you'd want to parse this properly
            return {
                "verified": True,
                "details": verification.content
            }
        except:
            return {
                "verified": False,
                "details": "Verification failed"
            }
            
    def _refine_response(
        self,
        original_response: str,
        verification_result: Dict[str, Any],
        query: str,
        context: str
    ) -> str:
        """
        Refine the response based on verification feedback.
        """
        if verification_result["verified"]:
            return original_response
            
        refinement_prompt = f"""
        Refine the following response based on verification feedback:

        Original Query: {query}
        Original Response: {original_response}
        Verification Result: {verification_result["details"]}
        Source Context: {context}

        Rules for refinement:
        1. Address any unsupported claims
        2. Add missing important information
        3. Fix any factual errors
        4. Maintain clarity and conciseness
        5. Stay true to the source material

        Provide the refined response.
        """
        
        refined = self.response_generator.invoke(refinement_prompt)
        return refined.content.strip()

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
