from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time consumer rights and court-related information
    from trusted web sources only.
    """
    def __init__(self):
        self.tavily_search_agent = TavilySearchAgent()

    def search(self, query: str) -> str:
        """
        Perform trusted site-restricted web search.
        """
        tavily_results = self.tavily_search_agent.search_tavily(query=query)
        return f"Trusted Source Results:\n{tavily_results}\n"
