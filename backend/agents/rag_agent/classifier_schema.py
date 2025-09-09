from typing import List, Literal
from pydantic import BaseModel, Field

class QueryClassification(BaseModel):
    query_type: Literal[
        "greeting", "chitchat", "section-specific", "general-info", "procedure", "rights", "complaint"
    ] = Field(..., description="Type of query, e.g., greeting, chitchat, section-specific, general-info, procedure, rights, complaint.")
    topics: List[str] = Field(default_factory=list, description="List of relevant topics identified in the query.")
    has_actionable_request: bool = Field(..., description="True if the query contains an actionable request.")
    requires_external_sources: bool = Field(..., description="True if the query requires information from external sources (e.g., web search, real-time info, contact details).")
    clarification_needed: bool = Field(..., description="True if the query is ambiguous or needs clarification.")
    clarification_question: str = Field("", description="A clarifying question to ask the user if the query is ambiguous; empty string if not needed.")
