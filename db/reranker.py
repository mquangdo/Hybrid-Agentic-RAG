import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_cohere.rerank import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv

load_dotenv()  


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="documents", embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is attention?")
print("Retrieved documents BEFORE reranking:")
pretty_print_docs(docs)

print('*' * 80)

reranker = CohereRerank(model="rerank-english-v4.0")

retriever_with_reranking = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
docs = retriever_with_reranking.invoke("What is attention?")
print("\nRetrieved documents AFTER reranking:")
pretty_print_docs(docs)
