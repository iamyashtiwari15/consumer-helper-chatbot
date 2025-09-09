import logging
from .tavily_search import FlexibleTavilySearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time consumer rights and court-related information
    from trusted web sources or unrestricted sources.
    """
    def __init__(self):
        self.flexible_agent = FlexibleTavilySearchAgent()
        self.logger = logging.getLogger(__name__)

    def search(self, query: str, trusted_sites_only: bool = True) -> str:
        """
        Perform web search using FlexibleTavilySearchAgent.
        """
        self.logger.info(f"[WebSearchAgent] Received query: {query}")
        results = self.flexible_agent.search(query, trusted_sites_only=trusted_sites_only)
        self.logger.info(f"[WebSearchAgent] Search results: {results[:200]}")
        return f"Web Search Results:\n{results}\n"
