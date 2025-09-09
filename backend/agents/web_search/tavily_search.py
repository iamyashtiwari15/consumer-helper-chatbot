


import logging
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()  # Load TAVILY_API_KEY from .env file

TRUSTED_SITES = [
    "consumerhelpline.gov.in",
    "ncdrc.nic.in",
    "consumeraffairs.nic.in",
    "pgportal.gov.in",
    "indiankanoon.org",
    "supremecourtofindia.nic.in",
    "districts.ecourts.gov.in"
]

class FlexibleTavilySearchAgent:
    """
    Flexible Tavily search agent supporting trusted-only and unrestricted search.
    """
    def __init__(self):
        self.tavily_search = TavilySearch(max_results=5)
        self.logger = logging.getLogger(__name__)

    def search(self, query: str, trusted_sites_only: bool = False) -> str:
        """
        Perform a web search using Tavily API.
        If trusted_sites_only is True, restrict to trusted sites.
        Returns a clean, readable string of results.
        """
        self.logger.info(f"Starting search for query: {query} | trusted_sites_only={trusted_sites_only}")
        clean_query = query.strip('"\'')
        final_query = clean_query

        if trusted_sites_only:
            self.logger.info(f"Performing TRUSTED site search for: {clean_query}")
            site_filter = " OR ".join([f"site:{site}" for site in TRUSTED_SITES])
            final_query = f"{clean_query} ({site_filter})"
        else:
            self.logger.info(f"Performing GENERAL web search for: {clean_query}")

        try:
            self.logger.info(f"Executing query: {final_query}")
            if len(final_query) > 400:
                final_query = final_query[:400]

            # The API returns a dictionary, so we store it as such.
            result_dict = self.tavily_search.invoke({"query": final_query})
            self.logger.info(f"Raw result: {str(result_dict)[:200]}")

            # The actual search results are in the "results" key of the dictionary.
            results_list = result_dict.get("results", [])

            # Now we check and process the list we extracted.
            if results_list and isinstance(results_list, list):
                formatted_results = "\n\n".join(
                    [f"Title: {item.get('title', 'N/A')}\nURL: {item.get('url', 'N/A')}\nSnippet: {item.get('content', 'N/A')}" for item in results_list]
                )
                return formatted_results.strip()

            return "No relevant results were found from the web search."

        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            return f"Error retrieving web search results: {e}"
