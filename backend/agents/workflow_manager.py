import logging
from typing import TypedDict, Optional, Dict, Any, List
from urllib import response
from agents.rag_agent.query_classifier import QueryClassifier
from agents.rag_agent.document_retriever import retrieve_documents
from agents.rag_agent.response_generator import ResponseGenerator
from agents.rag_agent.section_retriever import SectionRetriever
from agents.web_search.tavily_search import FlexibleTavilySearchAgent
from agents.query_merger import AdvancedQueryMerger
from agents.rag_agent.classifier_schema import QueryClassification
# from backend.agents.vision_agents.image_analysis_agent import extract_text_from_image  # Uncomment when integrating image agent

logger = logging.getLogger(__name__)

class WorkflowStrategy(TypedDict, total=False):
    use_rag: bool
    use_web: bool
    use_section: bool
    use_image: bool
    skip_response: bool

class WorkflowResponse(TypedDict, total=False):
    response: str
    sources: List[str]
    confidence: float
    verification_result: Any

class WorkflowManager:
    """
    Orchestrates the query handling workflow for the consumer helper chatbot.
    Decides which agents to use and how to combine their outputs.
    """
    def __init__(self, db_path):
        self.classifier = QueryClassifier()
        self.response_generator = ResponseGenerator()
        self.section_retriever = SectionRetriever(db_path)
        self.web_search_agent = FlexibleTavilySearchAgent()
        # self.image_agent = extract_text_from_image  # Uncomment when integrating image agent

    def _determine_strategy(self, classification: QueryClassification, query: str, image_path: Optional[str]) -> WorkflowStrategy:
        """
        Decide which modules to activate based on a hybrid of LLM classification and deterministic rules.
        """
        strategy: WorkflowStrategy = {
            "use_rag": False,
            "use_web": False,
            "use_section": False,
            "use_image": bool(image_path),
            "skip_response": False
        }
        qtype = classification.query_type
        # Handle greetings and chitchat efficiently
        if qtype == "chitchat" or qtype == "greeting":
            strategy["skip_response"] = True
            strategy["use_rag"] = False
            strategy["use_web"] = False
            strategy["use_section"] = False
            strategy["use_image"] = False
        elif qtype == "section-specific":
            strategy["use_section"] = True
            strategy["use_rag"] = True
        elif qtype in ["procedure", "rights", "complaint", "general-info"]:
            strategy["use_rag"] = True
            # Hybrid logic for web search
            web_keywords = [
                "internet", "web", "online", "search", "google", "phone number", "contact", "website",
                "latest", "current", "news", "address", "who is", "members", "recent", "find", "lookup"
            ]
            if classification.requires_external_sources or any(kw in query.lower() for kw in web_keywords):
                strategy["use_web"] = True
        return strategy

    def _gather_context(self, query: str, strategy: WorkflowStrategy, image_path: Optional[str], query_classification: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Gather context from RAG, section retriever, web search, and image agent as needed."""
        docs = []
        # Image context
        if strategy.get("use_image") and image_path:
            try:
                # extracted_text = self.image_agent(image_path)
                extracted_text = ""  # Placeholder
                if extracted_text:
                    query = f"{query}\n\n[Image Context]: {extracted_text}"
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")
        # Section context
        if strategy.get("use_section"):
            try:
                section_num = self.section_retriever.extract_section_number(query)
                docs = self.section_retriever.get_section(section_num) if section_num else []
            except Exception as e:
                logger.warning(f"Section retrieval failed: {e}")
        # RAG context
        elif strategy.get("use_rag"):
            try:
                docs, _ = retrieve_documents(query, query_classification=query_classification)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        # Web context
        if strategy.get("use_web"):
            try:
                # Use unrestricted search for general-info queries, restricted otherwise
                trusted_sites = True
                if query_classification and hasattr(query_classification, "query_type"):
                    if getattr(query_classification, "query_type", "") == "general-info":
                        trusted_sites = False
                web_results = self.web_search_agent.search(query, trusted_sites_only=trusted_sites)
                if web_results:
                    docs.append({
                        "content": web_results,
                        "metadata": {"source": "web_search"},
                        "score": 0.9
                    })
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        logger.info(f"[DEBUG] Gathered {len(docs)} docs for query: {query}")
        for i, doc in enumerate(docs):
            logger.info(f"[DEBUG] Doc {i+1}: Content: {doc.get('content', '')[:200]} | Metadata: {doc.get('metadata', {})}")
        return docs

    def _generate_response(self, query: str, docs: List[Dict[str, Any]], classification: QueryClassification, chat_history: Optional[List[Dict[str, str]]]) -> WorkflowResponse:
        """Generate the final response using the response generator."""
        try:
            logger.info(f"[DEBUG] Passing {len(docs)} docs to ResponseGenerator for query: {query}")
            for i, doc in enumerate(docs):
                logger.info(f"[DEBUG] Doc {i+1}: Content: {doc.get('content', '')[:200]} | Metadata: {doc.get('metadata', {})}")
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=docs,
                query_classification=classification,
                chat_history=chat_history
            )
            logger.info(f"[DEBUG] ResponseGenerator.generate_response output: {response}")
            # Always return a valid response dictionary
            if not response or not response.get("response"):
                return {"response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.", "sources": [], "confidence": 0.0}
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {"response": "Sorry, something went wrong. Please try again later.", "sources": [], "confidence": 0.0}

    def process_query(self, query: str, image_path: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> WorkflowResponse:
        logger.info(f"Received query: {query}")
        # Use AdvancedQueryMerger to rewrite the query for context-aware input
        merged_query = AdvancedQueryMerger.merge(query, chat_history or [])
        # Classify the merged query
        classification: QueryClassification = self.classifier.classify_query(merged_query)
        logger.info(f"Query classification: {classification}")

        # Early exit for greeting/chitchat queries
        if classification.query_type in ["greeting", "chitchat"]:
            return {"response": "Hello! How can I assist you with consumer rights or complaints today?", "sources": [], "confidence": 1.0, "query_type": classification.query_type}

        strategy = self._determine_strategy(classification, merged_query, image_path)
        logger.info(f"Workflow strategy: {strategy}")

        docs = self._gather_context(merged_query, strategy, image_path, query_classification=classification)

        # Only exit early for clarification if no docs are found
        if classification.clarification_needed and not docs:
            return {
                "response": classification.clarification_question or "Could you please clarify your query?",
                "source_docs": [],
                "strategy": strategy,
                "classification": classification,
                "query_type": classification.query_type
            }

        # If docs are found, generate response and include clarification question if present
        response = self._generate_response(merged_query, docs, classification, chat_history)
        if classification.clarification_needed:
            clarification = classification.clarification_question or "Could you please clarify your query?"
            response["clarification"] = clarification
        response["query_type"] = classification.query_type
        return response

