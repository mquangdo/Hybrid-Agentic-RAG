# Hybrid Agentic RAG: ChromaDB + Neo4j

A hybrid retrieval-augmented generation system that combines vector similarity search (ChromaDB) with knowledge graph relationships (Neo4j) for more intelligent and context-aware AI responses.

## Overview

This system implements an **Agentic RAG** architecture that goes beyond traditional RAG by:
- Using **ChromaDB** for semantic similarity search across document embeddings
- Leveraging **Neo4j** to capture and query entity relationships and knowledge graphs
- Orchestrating intelligent retrieval agents that can reason about when and how to use each database

## 🏗️ Architecture

```
User Query
    ↓
Router Agent (Decides retrieval strategy)
    ↓
├──→ Vector Search (ChromaDB) ←───┐
│         ↓                       │
│   Relevant Chunks               │
│         ↓                       │
│   Context Builder               │
│         ↓                       │
└────→ Response Generator ←─────┘
         ↓
    Final Answer

Knowledge Path (Neo4j)
    ↓
Entity Extraction
    ↓
Graph Query
    ↓
Relationship Context
```

## ✨ Key Features

- **Hybrid Retrieval**: Combines dense vector search with structured graph relationships
- **Agent-Driven**: Intelligent routing between vector and graph databases
- **Entity Extraction**: Automatically extracts entities and builds knowledge graphs
- **Multi-modal**: Supports both unstructured documents and structured knowledge
- **Context Enrichment**: Relationships from Neo4j enrich vector search results

## 🛠️ Setup

### Prerequisites

- Python 3.9+
- Neo4j 5.x
- ChromaDB

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Hybrid-Agentic-RAG

# Install dependencies
pip install -r requirements.txt

# Start Neo4j
docker run -d \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Start ChromaDB
docker run -d -p 8000:8000 chromadb/chroma:latest
```

### Configuration

Create a `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=your_key_here
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

## 🔍 How It Works

### Vector Store (ChromaDB)
- Stores document chunks as embeddings
- Performs semantic similarity search
- Handles unstructured text retrieval

### Knowledge Graph (Neo4j)
- Stores entities and relationships
- Enables complex graph traversals
- Provides structured knowledge context

### Agent Flow

1. **Query Analysis**: Router agent analyzes query intent
2. **Entity Detection**: Extracts named entities from query
3. **Strategy Selection**: Decides vector vs graph vs hybrid approach
4. **Parallel Retrieval**: Fetches from ChromaDB and/or Neo4j
5. **Context Fusion**: Combines results intelligently
6. **Response Generation**: LLM generates final answer

## 🤖 Use Cases

- **Enterprise Knowledge**: Connect documents with organizational hierarchies
- **Research Analysis**: Link papers, authors, citations, and concepts
- **Customer Support**: Map issues to solutions and product relationships
- **Legal Analysis**: Connect cases, statutes, and legal precedents

## 🔧 Configuration Options

```python
{
    "vector_weight": 0.7,           # Weight given to vector results
    "graph_weight": 0.3,             # Weight given to graph results
    "entity_threshold": 0.8,         # Confidence threshold for entity extraction
    "max_graph_depth": 2,           # Max relationship depth to traverse
    "chunk_size": 1000,              # Document chunk size
    "chunk_overlap": 200,            # Chunk overlap
    "embedding_model": "text-embedding-ada-002"
}
```

## 🚀 Future Enhancements

- [ ] Multi-modal support (images, audio)
- [ ] Cypher query generation from natural language
- [ ] Graph schema auto-discovery
- [ ] Incremental learning capabilities
- [ ] Streaming responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Resources

- [ChromaDB Documentation](https://docs.trychroma.com)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
