# processors/web_search_processor.py
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

from agents.web_search.web_search_agent import WebSearchAgent
from agents.llm_loader import get_llm  # Loads default LLM


class WebSearchProcessor:
    """
    Processes web search results for consumer rights queries and routes them to the appropriate LLM for response generation.
    """
    def __init__(self):
        self.web_search_agent = WebSearchAgent()
        self.llm = get_llm()

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """
        Convert chat history list of dicts into a readable string.
        """
        if not chat_history:
            return ""
        return "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
        )

    def _build_prompt_for_web_search(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build a prompt to summarize user intent for web search.
        """
        formatted_history = self._format_chat_history(chat_history)
        prompt = f"""Here is the recent conversation history:

{formatted_history}

The user has now asked:

{query}

If the conversation history contains relevant context, merge it with the latest question to form a single, clear, well-formed query for a web search.
This query should focus on Indian consumer rights, consumer court judgments, and legal remedies for consumers.
If there is no relevant past context, simply return the latest question.
Keep it concise and factual.
"""
        return prompt

    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Retrieves, summarizes, and returns web search results for consumer rights domain.
        """
        web_search_query_prompt = self._build_prompt_for_web_search(query, chat_history)
        web_search_query = self.llm.invoke(web_search_query_prompt).content

        web_results = self.web_search_agent.search(web_search_query)

        llm_prompt = (
            "You are an AI assistant specialized in Indian consumer rights. "
            "Below are web search results retrieved for a user query. "
            "Summarize and generate a helpful, concise, and legally accurate response. "
            "Use authentic sources only (such as government portals, consumer court websites, and trusted news sources). "
            "If you find court judgments, explain them in simple terms.\n\n"
            f"Original User Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )

        response = self.llm.invoke(llm_prompt)
        return response.content
