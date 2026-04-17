from  pydantic import BaseModel, Field  
from typing import List 

class RetrieverInputSchema(BaseModel):
    """Input schema for the document retriever tool."""
    query: str = Field(..., description="Search query string to find relevant documents")
    
class WebSearchInputSchema(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(..., description="Search query string for web search")

class AddInputSchema(BaseModel):
    """Input schema for the add tool."""
    a: int = Field(..., description="First number to add")
    b: int = Field(..., description="Second number to add")