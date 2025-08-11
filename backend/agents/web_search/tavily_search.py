'''import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()  # Load TAVILY_API_KEY from .env file

class TavilySearchAgent:
    """
    Handles general web search using Tavily API, restricted to authentic domains.
    """
    def __init__(self):
        self.tavily_search = TavilySearch(max_results=5)  # Uses env var TAVILY_API_KEY
        # List of trusted domains (Govt, Courts, Consumer Forums, Legal Analysis)
        self.authentic_domains = [
            # Government consumer help & policy
            "consumerhelpline.gov.in",
            "mca.gov.in",
            "doj.gov.in",
            "lawmin.gov.in",
            "nhrc.nic.in",
            "prsindia.org",

            # Supreme Court & High Court portals
            "supremecourtofindia.nic.in",
            "indiancourts.nic.in",
            "scobserver.in",

            # National, State & District Consumer Commissions
            "ncdrc.nic.in",  # National Consumer Disputes Redressal Commission
            "confonet.nic.in",  # Consumer Forums of India
            "cms.nic.in",  # Case Management System
            "consumer.tn.gov.in",  # Tamil Nadu consumer protection

            # Legal aid & resources
            "legalaidservicesindia.org",
            "legalserviceindia.com",

            # Trusted legal news & analysis
            "barandbench.com",
            "livelaw.in",
            "vakilno1.com",
        ]

    def search_tavily(self, query: str) -> str:
        """Perform a web search restricted to authentic domains."""
        try:
            query = query.strip('"\'')
            # Append site filters to query
            site_filter = " OR ".join([f"site:{domain}" for domain in self.authentic_domains])
            filtered_query = f"{query} ({site_filter})"

            result = self.tavily_search.invoke({"query": filtered_query})

            if result and isinstance(result, str):
                return result.strip()
            return "No relevant results found from authentic sites."

        except Exception as e:
            return f"Error retrieving web search results: {e}"
'''
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
