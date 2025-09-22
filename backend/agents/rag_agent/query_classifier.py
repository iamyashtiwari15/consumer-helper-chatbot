import re
from typing import Any, Dict
from agents.rag_agent.role_llm_loader import get_llm
from agents.rag_agent.classifier_schema import QueryClassification
from langchain_core.output_parsers import PydanticOutputParser


class QueryClassifier:
    """
    Simple, fast query classifier with rule-based pre-filtering + LLM fallback.
    """

    def __init__(self):
        self.llm = get_llm()
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)

    def _get_prompt(self, query: str) -> str:
        """Create classification prompt"""
        return f"""Classify this consumer query:

Query: "{query}"

Categories:
- greeting: Hello, hi, thanks, bye
- chitchat: How are you, what's up, casual talk  
- section-specific: "What does Section X say?"
- product-issue: Broken/defective products, warranty problems
- refund-return: Want money back, return items
- fraud-scam: Fake products, scams, cheated
- procedure: How to file complaints, legal steps
- rights: Consumer rights questions
- contact-info: Phone numbers, addresses, helplines
- general-info: Everything else, news, updates

{self.parser.get_format_instructions()}

Be quick and accurate."""

    def classify_query(self, query: str) -> QueryClassification:
        """
        Fast classification with rule-based pre-filtering + LLM fallback
        """
        # Rule-based classification for obvious cases (90% of queries)
        rule_result = self._rule_classify(query)
        if rule_result:
            return rule_result
            
        # LLM classification for complex cases (10% of queries)
        try:
            prompt = self._get_prompt(query)
            result = self.llm.invoke(prompt)
            # Extract content from result
            content = result.content if hasattr(result, 'content') else str(result)
            return self.parser.parse(content)
        except Exception:
            # Simple fallback
            return QueryClassification(
                query_type="general-info",
                topics=[],
                has_actionable_request=True,
                requires_external_sources=True,
                clarification_needed=False,
                clarification_question=""
            )

    def _rule_classify(self, query: str) -> QueryClassification:
        """Fast rule-based classification for obvious patterns"""
        q = query.lower().strip()
        
        # Greeting patterns
        if re.search(r'\b(hello|hi|hey|good morning|good afternoon|good evening|namaste)\b', q) or len(q) < 8:
            return QueryClassification(
                query_type="greeting", topics=["greeting"], 
                has_actionable_request=False, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Chitchat patterns  
        if re.search(r'\b(how are you|what\'?s up|thanks?|thank you|bye|goodbye|ok|okay|fine)\b', q):
            return QueryClassification(
                query_type="chitchat", topics=["conversation"],
                has_actionable_request=False, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Section-specific patterns
        if re.search(r'\b(section|clause|article|part|chapter|rule)\s*\d+\b', q):
            return QueryClassification(
                query_type="section-specific", topics=["legal", "section"],
                has_actionable_request=True, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Product issue patterns
        if re.search(r'\b(broken|defective|not working|faulty|damaged|poor quality|warranty|guarantee)\b', q):
            return QueryClassification(
                query_type="product-issue", topics=["product", "defect", "quality"],
                has_actionable_request=True, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Refund/return patterns
        if re.search(r'\b(refund|return|money back|get back|reimburse)\b', q):
            return QueryClassification(
                query_type="refund-return", topics=["refund", "return", "money"],
                has_actionable_request=True, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Fraud/scam patterns  
        if re.search(r'\b(fraud|scam|fake|cheat|duplicate|counterfeit|unauthorized)\b', q):
            return QueryClassification(
                query_type="fraud-scam", topics=["fraud", "scam", "security"],
                has_actionable_request=True, requires_external_sources=True,
                clarification_needed=False, clarification_question=""
            )
        
        # Procedure patterns
        if re.search(r'\b(how to|steps|process|procedure|file|complaint|court)\b', q):
            return QueryClassification(
                query_type="procedure", topics=["procedure", "complaint"],
                has_actionable_request=True, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Rights patterns
        if re.search(r'\b(rights|consumer rights|my rights|what rights)\b', q):
            return QueryClassification(
                query_type="rights", topics=["rights", "consumer"],
                has_actionable_request=True, requires_external_sources=False,
                clarification_needed=False, clarification_question=""
            )
        
        # Contact info patterns
        if re.search(r'\b(phone|number|contact|address|helpline)\b', q):
            return QueryClassification(
                query_type="contact-info", topics=["contact", "information"],
                has_actionable_request=True, requires_external_sources=True,
                clarification_needed=False, clarification_question=""
            )
        
        # General info patterns (news, latest, current info)
        if re.search(r'\b(latest|current|news|who is|recent|update)\b', q):
            return QueryClassification(
                query_type="general-info", topics=["information", "news"],
                has_actionable_request=True, requires_external_sources=True,
                clarification_needed=False, clarification_question=""
            )
        
        return None  # Let LLM handle complex cases

    def get_response_strategy(self, classification: QueryClassification) -> Dict[str, Any]:
        """
        Simple strategy mapping - much cleaner than before
        """
        strategies = {
            "greeting": {"use_rag": False, "use_web": False, "skip_response": True},
            "chitchat": {"use_rag": False, "use_web": False, "skip_response": True},
            "section-specific": {"use_rag": False, "use_web": False, "use_section": True, "skip_response": False},
            "product-issue": {"use_rag": True, "use_web": False, "skip_response": False},
            "refund-return": {"use_rag": True, "use_web": False, "skip_response": False},
            "fraud-scam": {"use_rag": True, "use_web": True, "skip_response": False},  # Needs web for latest scam info
            "procedure": {"use_rag": True, "use_web": True,"skip_response": False},
            "rights": {"use_rag": True, "use_web": False, "skip_response": False}, 
            "contact-info": {"use_rag": False, "use_web": True, "skip_response": False},  # Web only for contacts
            "general-info": {"use_rag": True, "use_web": True, "skip_response": False},
        }
        
        base_strategy = strategies.get(classification.query_type, strategies["general-info"])
        
        # Add standard fields
        strategy = {
            "use_section": False,
            "use_image": False,
            **base_strategy
        }
        
        # Override web search if explicitly needed
        if getattr(classification, "requires_external_sources", False):
            strategy["use_web"] = True
            
        return strategy
