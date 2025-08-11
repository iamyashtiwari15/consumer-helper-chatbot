import logging
from typing import List, Dict, Any
from agents.llm_loader import get_llm
class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = get_llm()

    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand the original query with relevant medical terms.
        
        Args:
            original_query: The user's original query
            
        Returns:
            Dictionary with original and expanded queries
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # Generate expansions - implement one of the strategies below
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with medical terminology."""
        prompt = f"""
        As a medical expert, expand the following query with relevant medical terminology, 
        synonyms, and related concepts that would help in retrieving relevant medical information:
        
        User Query: {query}
        
        Expand the query only if you feel like it is required, otherwise keep the user query intact.
        Be specific to the medical or any other domain mentioned in the ueer query, do not add other medical domains.
        If the user query asks about answering in tabular format, include that in the expanded query and do not answer in tabular format yourself.
        Provide only the expanded query without explanations.
        """
        expansion = self.model.invoke(prompt)
        
        return expansion