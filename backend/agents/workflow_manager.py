import logging
from typing import TypedDict, Optional, Dict, Any, List
from agents.rag_agent.query_classifier import QueryClassifier
from agents.rag_agent.document_retriever import retrieve_documents
from agents.rag_agent.response_generator import ResponseGenerator
from agents.rag_agent.section_retriever import SectionRetriever
from agents.web_search.web_search_agent import WebSearchAgent
# from backend.agents.vision_agents.image_analysis_agent import extract_text_from_image  # Uncomment when integrating image agent

logger = logging.getLogger(__name__)

class QueryClassification(TypedDict, total=False):
    query_type: str
    topics: List[str]
    required_info_types: List[str]
    has_actionable_request: bool
    requires_external_sources: bool

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
        self.web_search_agent = WebSearchAgent()
        # self.image_agent = extract_text_from_image  # Uncomment when integrating image agent

    def _determine_strategy(self, classification: QueryClassification, image_path: Optional[str]) -> WorkflowStrategy:
        """Decide which modules to activate based on query classification and image presence."""
        strategy: WorkflowStrategy = {
            "use_rag": False,
            "use_web": False,
            "use_section": False,
            "use_image": bool(image_path),
            "skip_response": False
        }
        qtype = classification.get("query_type", "general-info")
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
            if classification.get("requires_external_sources", False):
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
                web_results = self.web_search_agent.search(query)
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
        # Classify the query
        classification: QueryClassification = self.classifier.classify_query(query)
        logger.info(f"Query classification: {classification}")

        # Early exit for greeting/chitchat queries
        if classification.get("query_type") in ["greeting", "chitchat"]:
            return {"response": "Hello! How can I assist you with consumer rights or complaints today?", "sources": [], "confidence": 1.0}

        strategy = self._determine_strategy(classification, image_path)
        logger.info(f"Workflow strategy: {strategy}")

        docs = self._gather_context(query, strategy, image_path, query_classification=classification)

        # Only exit early for clarification if no docs are found
        if classification.get("clarification_needed", False) and not docs:
            return {
                "response": classification.get("clarification_question", "Could you please clarify your query?"),
                "source_docs": [],
                "strategy": strategy,
                "classification": classification
            }

        # If docs are found, generate response and include clarification question if present
        response = self._generate_response(query, docs, classification, chat_history)
        if classification.get("clarification_needed", False):
            # Append clarification to response if not already present
            clarification = classification.get("clarification_question", "Could you please clarify your query?")
            response["clarification"] = clarification
        return response