from vector_db import VectorDB


def test_build_db():
    """
    Tests building a fresh vector database from PDF documents.

    Args:
        None

    Returns:
        None
    """
    print("=" * 60)
    print("Test: Building Vector Database")
    print("=" * 60)

    vec_db = VectorDB()
    print("Loading PDFs from data directory...")
    docs = vec_db.load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs")

    print("Splitting documents into chunks...")
    texts = vec_db.split_documents(docs)
    print(f"Created {len(texts)} text chunks")

    print("Building vector database...")
    vec_db.build_db()

    print("\n✓ Vector DB created successfully!")
    print("=" * 60)


def test_retrieve():
    """
    Tests retrieving documents from an existing vector database.

    Args:
        None

    Returns:
        None
    """
    print("=" * 60)
    print("Test: Retrieving from Existing Database")
    print("=" * 60)

    vec_db = VectorDB()
    retriever = vec_db.get_retriever(k=3)

    print("Testing with sample query...")
    test_query = "What is RAG?"
    print(f"Query: '{test_query}'\n")

    docs = retriever.invoke(test_query)

    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Preview: {doc.page_content[:200]}...")
        print()

    print("✓ Retrieval test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_vector_db.py [build|retrieve]")
        print("  build    - Build a new vector database from PDFs")
        print("  retrieve - Test retrieval from existing database")
        sys.exit(1)

    test_type = sys.argv[1].lower()

    if test_type == "build":
        test_build_db()
    elif test_type == "retrieve":
        test_retrieve()
    else:
        print(f"Unknown test type: {test_type}")
        print("Use 'build' or 'retrieve'")
        sys.exit(1)


def test_vector_db():
    """
    Legacy combined test that runs both build and retrieve.

    For backward compatibility.

    Args:
        None

    Returns:
        None
    """
    test_build_db()
    test_retrieve()
