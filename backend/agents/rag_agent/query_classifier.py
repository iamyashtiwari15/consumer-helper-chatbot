import logging
from typing import Any, Dict
from agents.rag_agent.llm_loader import get_llm
from agents.rag_agent.classifier_schema import QueryClassification
from langchain_core.output_parsers import PydanticOutputParser


class QueryClassifier:
    """
    Classifies user queries to determine the best response strategy.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = get_llm(role="classifier")
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify the query and return a validated QueryClassification object.
        """
        format_instructions = self.parser.get_format_instructions()
        prompt = f"""
Classify the following consumer query. Use the format below.

Query: {query}

{format_instructions}

**Example Classifications:**
1. "Hello" -> greeting, clarification_needed: false, clarification_question: ""
2. "How are you?" -> chitchat, clarification_needed: false, clarification_question: ""
3. "What does Section 33 say?" -> section-specific, clarification_needed: false, clarification_question: ""
4. "How to file a complaint?" -> procedure, clarification_needed: false, clarification_question: ""
5. "What are my rights as a consumer?" -> rights, clarification_needed: true, clarification_question: "Could you please specify which rights you are interested in? For example, rights related to product returns, data privacy, or service contracts?"
6. "Steps after online fraud" -> procedure + complaint, clarification_needed: false, clarification_question: ""
7. "What is the phone number of the consumer court in Delhi?" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
8. "Find the latest amendments to CPA 2019" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
9. "Who is the current president of the National Consumer Disputes Redressal Commission?" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
10. "Show me recent news about consumer protection" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
11. "Explain the process for filing a complaint online" -> procedure, requires_external_sources: false, clarification_needed: false, clarification_question: ""
12. "Give me the address of the consumer helpline" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
13. "What are the penalties for misleading advertisements?" -> general-info, requires_external_sources: false, clarification_needed: false, clarification_question: ""
14. "Who are the members of the Central Consumer Protection Authority?" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
15. "What was the consumer law before CPA 2019?" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
16. "Show me amendments after 2020" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
17. "Compare CPA 1986 and CPA 2019" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
18. "What changed before and after the 2020 amendments?" -> general-info, requires_external_sources: true, clarification_needed: false, clarification_question: ""
"""
        result = self.classifier.invoke(prompt)
        try:
            # Extract only the JSON part from the LLM output
            content = result.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                content = content[json_start:json_end]
            return self.parser.parse(content)
        except Exception as e:
            self.logger.error(f"Error parsing classification: {e}")
            return QueryClassification(
                query_type="general-info",
                topics=[],
                has_actionable_request=False,
                requires_external_sources=False,
                clarification_needed=False,
                clarification_question=""
            )

    def get_response_strategy(self, classification: QueryClassification) -> Dict[str, Any]:
        """
        Determine the best strategy to answer the query based on its classification.
        This is now the single source of truth for workflow strategy.
        """
        strategy = {
            "use_rag": False,
            "use_web": False,
            "use_section": False,
            "use_image": False,
            "skip_response": False
        }

        query_type = getattr(classification, "query_type", "general-info")

        if query_type in ["chitchat", "greeting"]:
            strategy["skip_response"] = True
        elif query_type == "section-specific":
            strategy["use_section"] = True
            strategy["use_rag"] = True
        elif query_type in ["procedure", "rights", "complaint", "general-info"]:
            strategy["use_rag"] = True
            web_keywords = [
                "internet", "web", "online", "search", "google", "phone number", "contact", "website",
                "latest", "current", "news", "address", "who is", "members", "recent", "find", "lookup",
                "before", "after", "since", "amendment", "update", "change", "2020", "2021", "2022", "2023", "2024",
                "compare", "difference", "vs", "versus"
            ]
            if getattr(classification, "requires_external_sources", False) or any(kw in getattr(classification, "query_text", "").lower() for kw in web_keywords):
                strategy["use_web"] = True
        return strategy
