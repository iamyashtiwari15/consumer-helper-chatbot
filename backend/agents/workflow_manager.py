import logging
from typing import TypedDict, Optional, Dict, Any, List
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
from agents.rag_agent.query_classifier import QueryClassifier
from agents.rag_agent.document_retriever import retrieve_documents
from agents.rag_agent.response_generator import ResponseGenerator
from agents.rag_agent.section_retriever import SectionRetriever
from agents.web_search.tavily_search import FlexibleTavilySearchAgent
from agents.query_merger import AdvancedQueryMerger
from agents.rag_agent.classifier_schema import QueryClassification
from agents.vision_agents.image_analysis_agent import analyze_image  # Uncomment when integrating image agent

logger = logging.getLogger(__name__)

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

    def _determine_strategy(self, classification: QueryClassification, query: str, image_path: Optional[str]) -> WorkflowStrategy:
        """
        Use QueryClassifier.get_response_strategy as the single source of truth for workflow strategy.
        """
        strategy = self.classifier.get_response_strategy(classification)
        # Optionally, add image logic here if needed
        strategy["use_image"] = bool(image_path)
        return strategy

    def _gather_context(self, query: str, strategy: WorkflowStrategy, image_path: Optional[str], query_classification: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Gather context from RAG, section retriever, web search, and image agent as needed."""
        docs = []
        # NOTE: The direct image context is now handled in process_query.
        # This section could be used if you need to re-analyze the image for some reason.
        
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
        return docs

    def _generate_response(self, query: str, docs: List[Dict[str, Any]], classification: QueryClassification, chat_history: Optional[List[Dict[str, str]]]) -> WorkflowResponse:
        """Generate the final response using the response generator."""
        logger.info(f"[LOG] Starting response generation for query: {query}")
        try:
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=docs,
                query_classification=classification,
                chat_history=chat_history
            )
            logger.info(f"[LOG] Finished response generation. Response: {response}")
            if not response or not response.get("response"):
                logger.warning(f"[LOG] No response generated for query: {query}")
                return {"response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.", "sources": [], "confidence": 0.0}
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {"response": "Sorry, something went wrong. Please try again later.", "sources": [], "confidence": 0.0}

    def process_query(self, query: str, image_path: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None, image_context: Optional[str] = None) -> WorkflowResponse:
        """
        Processes a user's query, merges contexts, and generates a response.
        """
        logger.info(f"Received query: '{query}'")

        # Step 1: Merge user query with chat history
        merged_query = AdvancedQueryMerger.merge(query, chat_history or [])

        # Step 2: Image context already handled upstream by the graph
        # merged_query = query

        # Step 3: Classify the final merged query
        classification: QueryClassification = self.classifier.classify_query(merged_query)
        logger.info(f"Query classification: {classification}")

        # Step 4: Handle simple cases like greetings
        if classification.query_type in ["greeting", "chitchat"]:
            return {"response": "Hello! How can I assist you with consumer rights or complaints today?", "sources": [], "confidence": 1.0, "query_type": classification.query_type}

        # Step 5: Determine the strategy and gather documents
        strategy = self._determine_strategy(classification, merged_query, image_path)
        logger.info(f"Workflow strategy: {strategy}")
        docs = self._gather_context(merged_query, strategy, image_path, query_classification=classification)

        # Step 6: Handle cases where clarification is needed and no documents were found
        if classification.clarification_needed and not docs:
            return {
                "response": classification.clarification_question or "Could you please clarify your query?",
                "source_docs": [], "strategy": strategy, "classification": classification,
                "query_type": classification.query_type
            }

        # Step 7: Generate the final response
        response = self._generate_response(merged_query, docs, classification, chat_history)
        if classification.clarification_needed:
            response["clarification"] = classification.clarification_question or "Could you please clarify your query?"
        response["query_type"] = classification.query_type
        
        return response
