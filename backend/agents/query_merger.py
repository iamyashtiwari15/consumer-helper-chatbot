from agents.rag_agent.llm_loader import get_llm


class AdvancedQueryMerger:
    @staticmethod
    def merge(current_query, chat_history, max_history=2):
        """
        Use an LLM to rewrite the user's query into a self-contained question based on chat history.
        Args:
            current_query (str): The latest user query.
            chat_history (list): List of previous exchanges (dicts with 'role' and 'content').
            max_history (int): Number of previous turns to include.
        Returns:
            str: Rewritten, self-contained query string.
        """
        relevant_history = chat_history[-max_history:] if chat_history else []
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in relevant_history
        ])
        prompt = f"""
Given the following chat history and the user's latest query, rewrite the query as a self-contained, contextually complete question. If the query is already self-contained, return it as is.

Chat History:
{history_str}

User's Latest Query:
{current_query}

Rewritten Query:
"""
        llm = get_llm(role="query_rewriter")
        result = llm.invoke(prompt)
        rewritten_query = result.content.strip() if hasattr(result, "content") else str(result).strip()
        return rewritten_query
