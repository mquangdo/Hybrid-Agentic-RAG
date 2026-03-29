import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from db.vector_db import VectorDB
from functools import lru_cache
from schema import RetrieverInputSchema, WebSearchInputSchema

class RAGTools:
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_vector_db():
        """
        Initializes and returns the VectorDB instance.

        This function creates an instance of the VectorDB class, which manages
        the vector database for document retrieval. It can be used to access
        the retriever tool for searching relevant documents based on queries.

        Args:
            None
            
        Returns:
            An instance of the VectorDB class for managing the vector database
        """
        return VectorDB()
    
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_retriever(k: int = 3):
        vector_db = RAGTools.get_vector_db()
        return vector_db.get_retriever(k=k)


    @staticmethod
    def document_retriever(query: str) -> List[str]:
        """
        Searches through documents to find relevant information.

        Use this tool when you need to answer questions about the content
        in the documents. It will search the vector database for the most
        relevant document chunks based on your query.

        Args:
            query: Search query string to find relevant documents

        Returns:
            List of document contents as strings that are relevant to the query
        """
        retriever = RAGTools.get_retriever()
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]
    
    
def get_tools() -> List[StructuredTool]:
    """
    Returns a list of tools for use in the workflow.
    
        Args:
            None
        
        Returns:
            List of tool instances to be used in the workflow
    """
    
    rag_tools = [
        StructuredTool.from_function(
            func=RAGTools.document_retriever,
            name="document_retriever",
            description="Search through documents to find relevant information. Use this tool when you need to answer questions about the content in the documents.",
            input_schema=RetrieverInputSchema
        )
    ]
    
    return rag_tools


