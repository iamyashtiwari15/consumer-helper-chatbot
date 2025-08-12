import logging
from typing import List, Dict, Any
from agents.rag_agent.llm_loader import get_llm
class QueryExpander:
    """
    Agentic query expansion with planning and decomposition capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.planner = get_llm(role="planner")
        self.expander = get_llm(role="expander")

    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Plan and expand the query using multi-agent approach.
        
        Args:
            original_query: The user's original query
            
        Returns:
            Dictionary with query plan and expansions
        """
        self.logger.info(f"Planning and expanding query: {original_query}")
        
        # Step 1: Create query plan
        query_plan = self._create_query_plan(original_query)
        
        # Step 2: Decompose complex queries
        sub_queries = self._decompose_query(original_query, query_plan)
        
        # Step 3: Expand each sub-query
        expanded_queries = [self._generate_expansions(q) for q in sub_queries]
        
        return {
            "original_query": original_query,
            "query_plan": query_plan.content,
            "sub_queries": sub_queries,
            "expanded_queries": [eq.content for eq in expanded_queries]
        }
    
    def _create_query_plan(self, query: str) -> str:
        """Create a strategic plan for query processing."""
        prompt = f"""
        Analyze the following query and create a strategic plan:
        
        User Query: {query}
        
        1. Identify main concepts and entities
        2. Determine information requirements
        3. Specify retrieval strategy
        4. List potential challenges
        
        Return the plan in a structured format.
        """
        return self.planner.invoke(prompt)
    
    def _decompose_query(self, query: str, query_plan: str) -> List[str]:
        """Break down complex queries into simpler sub-queries."""
        prompt = f"""
        Decompose the following query into simpler sub-queries:
        
        User Query: {query}
        Query Plan: {query_plan}
        
        Rules:
        1. Each sub-query should focus on a single aspect
        2. Preserve the original intent
        3. Ensure completeness
        4. Maximum 3 sub-queries
        
        Return a list of sub-queries.
        """
        response = self.planner.invoke(prompt)
        # Extract sub-queries from response
        sub_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        return sub_queries if sub_queries else [query]
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with domain-specific terminology."""
        prompt = f"""
        Expand the following query with relevant domain-specific terminology and concepts:
        
        User Query: {query}
        
        Rules:
        1. Expand only if necessary, otherwise keep original query
        2. Stay focused on the specific domain mentioned
        3. Include synonyms and related terms
        4. Preserve the original query intent
        5. Return ONLY the expanded query, no explanations
        
        Expanded Query:
        """
        
        expansion = self.expander.invoke(prompt)
        return expansion