"""
BGE-FAISS MCP: Semantic Search Server for Model Context Protocol

A high-performance semantic search server using BGE-M3 embeddings and FAISS vector store.
Provides fast, accurate semantic search capabilities through Model Context Protocol.
"""

__version__ = "1.0.5"
__author__ = "AI Dev Companion Project"
__email__ = "your-email@example.com"

from .core.manager import HybridSearchManager, SearchMode
from .core.embedder import BGE_M3Embedder
from .core.vector_store import FAISSVectorStore
from .core.retriever import SemanticRetriever

__all__ = [
    "HybridSearchManager",
    "SearchMode",
    "BGE_M3Embedder",
    "FAISSVectorStore",
    "SemanticRetriever",
]
