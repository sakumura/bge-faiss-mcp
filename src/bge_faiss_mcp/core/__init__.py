"""
Core components for BGE-FAISS MCP semantic search.

This module provides the core functionality including:
- BGE-M3 embeddings
- FAISS vector store
- Semantic retrieval
- Search management
"""

from .embedder import BGE_M3Embedder
from .manager import HybridSearchManager, SearchMode
from .retriever import SemanticRetriever
from .vector_store import FAISSVectorStore, Document

__all__ = [
    "BGE_M3Embedder",
    "HybridSearchManager",
    "SearchMode",
    "SemanticRetriever",
    "FAISSVectorStore",
    "Document",
]
