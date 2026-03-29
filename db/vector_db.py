import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """A class for managing a vector database using ChromaDB and HuggingFace embeddings."""

    def __init__(
        self, db_path: str = "./chroma_db", collection_name: str = "documents"
    ):
        """
        Initializes the VectorDB instance.

        Args:
            db_path: Path where the ChromaDB database will be stored
            collection_name: Name of the collection in ChromaDB

        Returns:
            None
        """
        self.data_dir = Path("../data")
        self.db_path = db_path
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def load_pdfs(self) -> List[Any]:
        """
        Loads all PDF files from the data directory.

        Args:
            None

        Returns:
            List of Document objects loaded from all PDF files in the data directory
        """
        documents = []
        for pdf_file in self.data_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(pdf_file.name)
            documents.extend(docs)
        return documents

    def split_documents(
        self, documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Any]:
        """
        Splits documents into smaller chunks for embedding.

        Args:
            documents: List of Document objects to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of Document chunks split from the input documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def build_db(self):
        """
        Builds the vector database from PDF documents.

        Args:
            None

        Returns:
            None
        """
        docs = self.load_pdfs()
        texts = self.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name,
        )
        print(f"Vector DB created at {self.db_path} with {len(texts)} chunks")

    def get_retriever(self, k: int = 4):
        """
        Creates a retriever for querying the vector database.

        Args:
            k: Number of documents to retrieve per query

        Returns:
            A retriever object for querying the vector database
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.db_path,
                collection_name=self.collection_name,
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
