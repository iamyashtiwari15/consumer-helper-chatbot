from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agents.llm_loader import get_llm
import os

# Initialize with conversation role
llm = get_llm()

def get_consumer_rights_response(messages: list) -> AIMessage:
    """
    Generates a legal customer rights response using the LLM and message history.

    Args:
        messages: A list of LangChain BaseMessages including HumanMessage(s) and prior AIMessage(s)

    Returns:
        AIMessage with the model's response
    """
   
    full_messages = [SystemMessage(content=(
    "You are a helpful, ethical customer rights assistant. "
    "You provide general information about consumer protection laws, rights, and complaint processes. "
    "Do NOT provide legal advice or representation. "
    "Base your responses only on publicly available, trusted legal sources such as official government sites, "
    "consumer helpline portals, or reputable legal information resources. "
    "Always clarify that the information is for educational purposes only."
    "\n\n"
    "Format all responses using this structure (no markdown headings, use dividers instead):\n"
    "— Summary —\n"
    "Short summary (2–3 sentences) of the situation or query.\n"
    "\n"
    "— General Information —\n"
    "1. Use numbered steps for possible actions.\n"
    "• Use bullet points for details under each step.\n"
    "\n"
    "— Next Steps —\n"
    "Short suggestions for what the user can do next.\n"
    "\n"
    "— Disclaimer —\n"
    "State that the information is educational and not legal advice."
))] + messages

    
    try:
        result = llm.invoke(full_messages)
        return AIMessage(content=result.content)

    except Exception as e:
        print(f"[Legal Agent Error] {e}")
        return AIMessage(content="⚠️ Sorry, I encountered an issue generating a response.")
