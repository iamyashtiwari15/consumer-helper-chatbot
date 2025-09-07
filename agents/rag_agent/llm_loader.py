# agents/llm_loader.py

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

# agents/llm_loader.py

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

# agents/llm_loader.py

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

def get_llm(role: str = "default"):
    """
    Loads the LLM with specific system prompts based on role.
    
    Args:
        role: Role for the LLM (default, planner, validator, etc.)
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file.")
    
    system_prompts = {
        "default": "You are a helpful assistant.",
        "planner": """You are a query planning expert. Analyze user queries to:
            1. Identify key concepts and relationships
            2. Break down complex queries into sub-queries
            3. Determine required information types
            4. Plan the retrieval strategy""",
        "validator": """You are a fact-checking expert. Your role is to:
            1. Verify response accuracy against source documents
            2. Identify unsupported claims
            3. Ensure response completeness
            4. Suggest improvements""",
        "expander": """You are a query expansion expert. Your role is to:
            1. Identify key concepts in user queries
            2. Add relevant domain-specific terminology
            3. Include synonyms and related terms
            4. Maintain query intent and context"""
    }
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
    )
    
    # Create a wrapped version that includes the system prompt in each call
    system_prompt = system_prompts.get(role, system_prompts["default"])
    
    class WrappedLLM:
        def __init__(self, base_llm, system_prompt):
            self.base_llm = base_llm
            self.system_prompt = system_prompt
            
        def invoke(self, prompt: str):
            # Create a chat template that includes the system prompt
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            
            # Format the messages with the actual prompt
            messages = chat_template.format_messages(input=prompt)
            
            # Invoke the base LLM with the formatted messages
            return self.base_llm.invoke(messages)
    
    return WrappedLLM(llm, system_prompt)
class NamedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Loads the default sentence embedding model using HuggingFace.
    """
    # model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # return embedding
    return NamedHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
