
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()  # Load TAVILY_API_KEY from .env file

TRUSTED_SITES = [
    "consumerhelpline.gov.in",
    "ncdrc.nic.in",
    "supremecourtofindia.nic.in",
    "indiancourts.nic.in",
    "districts.ecourts.gov.in",
    "consumeraffairs.nic.in",
    "pgportal.gov.in",
    "ccpa.gov.in",
    "fcraonline.nic.in",
    "legalserviceindia.com",
    "vakilno1.com",
    "indiankanoon.org",
    "legitquest.com",
    "barandbench.com",
    "livelaw.in",
    "lawsisto.com"
]

class TavilySearchAgent:
    """
    Handles general web search using Tavily API, restricted to trusted sources only.
    """
    def __init__(self):
        self.tavily_search = TavilySearch(max_results=5)  # Uses env var TAVILY_API_KEY

    def search_tavily(self, query: str) -> str:
        """Perform a trusted site-restricted web search using Tavily API."""
        try:
            query = query.strip('"\'')
            
            # Build a query with site restrictions
            site_filter = " OR ".join([f"site:{site}" for site in TRUSTED_SITES])
            restricted_query = f"{query} ({site_filter})"

            result = self.tavily_search.invoke({"query": restricted_query})

            if result and isinstance(result, str):
                return result.strip()
            return "No relevant results found from trusted sources."

        except Exception as e:
            return f"Error retrieving web search results: {e}"
