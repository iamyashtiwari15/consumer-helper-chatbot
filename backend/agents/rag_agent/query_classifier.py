import logging
from typing import Dict, Any, List
from .llm_loader import get_llm

class QueryClassifier:
    """
    Classifies user queries to determine the best response strategy.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = get_llm(role="classifier")
        
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify the query to determine:
        1. Query type (section-specific, general-info, procedure, rights, complaint)
        2. Topics involved
        3. Required information types
        4. Ambiguity detection (clarification_needed, clarification_question)
        """
        
        prompt = f"""Analyze the following consumer query and classify it:

Query: {query}

Provide classification in this exact JSON format:
{{
    "query_type": "one of: greeting, chitchat, section-specific, general-info, procedure, rights, complaint",
    "topics": ["list", "of", "relevant", "topics"],
    "required_info_types": ["list", "of", "required", "information", "types"],
    "has_actionable_request": true/false,
    "requires_external_sources": true/false,
    "clarification_needed": true/false,
    "clarification_question": "If the query is ambiguous or too broad, suggest a follow-up question to clarify the user's intent. Otherwise, return an empty string."
}}

Example classifications:
1. "Hello" -> greeting, clarification_needed: false, clarification_question: ""
2. "How are you?" -> chitchat, clarification_needed: false, clarification_question: ""
3. "What does Section 33 say?" -> section-specific, clarification_needed: false, clarification_question: ""
4. "How to file a complaint?" -> procedure, clarification_needed: false, clarification_question: ""
5. "What are my rights as a consumer?" -> rights, clarification_needed: true, clarification_question: "Could you please specify which rights you are interested in? For example, rights related to product returns, data privacy, or service contracts?"
6. "Steps after online fraud" -> procedure + complaint, clarification_needed: false, clarification_question: ""
Focus on consumer protection and legal context."""

        result = self.classifier.invoke(prompt)
        try:
            # The LLM should return a JSON string
            import json
            classification = result.content
            if isinstance(classification, str):
                classification = json.loads(classification)
            # Ensure new fields are present
            if "clarification_needed" not in classification:
                classification["clarification_needed"] = False
            if "clarification_question" not in classification:
                classification["clarification_question"] = ""
            return classification
        except Exception as e:
            self.logger.error(f"Error parsing classification: {e}")
            return {
                "query_type": "general-info",
                "topics": [],
                "required_info_types": [],
                "has_actionable_request": False,
                "requires_external_sources": False,
                "clarification_needed": False,
                "clarification_question": ""
            }

    def get_response_strategy(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best strategy to answer the query based on its classification.
        """
        strategy = {
            "use_rag": True,
            "needs_web_search": False,
            "needs_examples": False,
            "needs_step_by_step": False,
            "response_format": "default"
        }
        
        query_type = classification.get("query_type", "general-info")
        
        if query_type == "procedure":
            strategy.update({
                "needs_step_by_step": True,
                "response_format": "steps",
                "needs_examples": True
            })
            
        elif query_type == "complaint":
            strategy.update({
                "needs_web_search": True,
                "needs_step_by_step": True,
                "response_format": "steps"
            })
            
        if classification.get("requires_external_sources", False):
            strategy["needs_web_search"] = True
            
        if classification.get("has_actionable_request", False):
            strategy["needs_step_by_step"] = True
            
        return strategy
