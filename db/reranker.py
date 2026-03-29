import os
from langchain_core.documents import Document
from langchain_cohere.rerank import CohereRerank
from dotenv import load_dotenv

load_dotenv()  

if __name__ == "__main__":

    query = "What is normalization in databases?"

    docs = [
        Document(page_content="Normalization reduces redundancy in relational databases..."),
        Document(page_content="Python is a programming language..."),
        Document(page_content="Database normalization organizes columns and tables to reduce duplication..."),
    ]

    reranker = CohereRerank(
        model="rerank-english-v3.0",  
        top_n=2,
    )

    reranked_docs = reranker.compress_documents(docs, query)

    for i, d in enumerate(reranked_docs, 1):
        print(i, d.page_content[:80])