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
        """Simplified context gathering with parallel execution for speed"""
        docs = []
        
        # Section retrieval (synchronous, usually fast)
        if strategy.get("use_section"):
            try:
                section_num = self.section_retriever.extract_section_number(query)
                docs = self.section_retriever.get_section(section_num) if section_num else []
            except Exception as e:
                logger.warning(f"Section retrieval failed: {e}")
        
        # Parallel RAG and Web search for better performance
        elif strategy.get("use_rag") or strategy.get("use_web"):
            import concurrent.futures
            
            def get_rag_docs():
                if strategy.get("use_rag"):
                    try:
                        rag_docs, _ = retrieve_documents(query, query_classification=query_classification)
                        return rag_docs
                    except Exception as e:
                        logger.warning(f"RAG retrieval failed: {e}")
                return []
            
            def get_web_docs():
                if strategy.get("use_web"):
                    try:
                        # Optimize web search based on query type
                        trusted_only = True
                        if query_classification and hasattr(query_classification, "query_type"):
                            # Use broader search for general info and fraud cases
                            if getattr(query_classification, "query_type", "") in ["general-info", "fraud-scam"]:
                                trusted_only = False
                        
                        web_results = self.web_search_agent.search(query, trusted_sites_only=trusted_only)
                        if web_results:
                            return [{
                                "content": web_results,
                                "metadata": {"source": "web_search", "query_type": getattr(query_classification, "query_type", "unknown")},
                                "score": 0.9
                            }]
                    except Exception as e:
                        logger.warning(f"Web search failed: {e}")
                return []
            
            # Execute in parallel with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                if strategy.get("use_rag"):
                    futures.append(executor.submit(get_rag_docs))
                if strategy.get("use_web"):
                    futures.append(executor.submit(get_web_docs))
                
                for future in concurrent.futures.as_completed(futures, timeout=15):
                    try:
                        result = future.result()
                        if result:
                            docs.extend(result)
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"Context gathering timed out")
                    except Exception as e:
                        logger.warning(f"Error in parallel context gathering: {e}")
        
        logger.info(f"Gathered {len(docs)} documents for query type: {getattr(query_classification, 'query_type', 'unknown')}")
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
            responses = {
                "greeting": "Hello! How can I assist you with consumer rights or complaints today?",
                "chitchat": "I'm here to help you with consumer rights and protection matters. What can I assist you with?"
            }
            return {
                "response": responses.get(classification.query_type, responses["greeting"]), 
                "sources": [], "confidence": 1.0, "query_type": classification.query_type
            }

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
