from typing import List
from langchain_core.tools import tool
from db.vector_db import VectorDB


def setup_retriever_tool():
    """
    Sets up and returns the document retriever tool.

    Initializes VectorDB internally to avoid circular imports
    and provides a tool interface for document search.

    Returns:
        List of tools with the document_retriever tool
    """
    vector_db = VectorDB()
    retriever = vector_db.get_retriever(k=4)

    @tool
    def document_retriever(query: str) -> List[str]:
        """Searches through documents to find relevant information.

        Use this tool when you need to answer questions about the content
        in the documents. It will search the vector database for the most
        relevant document chunks based on your query.

        Args:
            query: Search query string to find relevant documents

        Returns:
            List of document contents as strings that are relevant to the query
        """
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]

    return [document_retriever]
