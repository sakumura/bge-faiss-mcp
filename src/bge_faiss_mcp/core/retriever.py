"""
Semantic Retriever Module

High-level retrieval interface combining embedder and vector store.
"""

import numpy as np
import torch
import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from bge_faiss_mcp.core.embedder import BGE_M3Embedder
from bge_faiss_mcp.core.vector_store import FAISSVectorStore, Document
from bge_faiss_mcp.utils.gpu import GPUOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic search retriever combining embeddings and vector store."""

    def __init__(
        self,
        embedder: Optional[BGE_M3Embedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        use_gpu: bool = True,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize semantic retriever.

        Args:
            embedder: BGE-M3 embedder instance
            vector_store: FAISS vector store instance
            use_gpu: Use GPU acceleration
            store_path: Path for vector store persistence
        """
        # Initialize GPU optimizer
        self.gpu_optimizer = GPUOptimizer() if use_gpu else None

        # Initialize embedder
        if embedder is None:
            device = (
                self.gpu_optimizer.auto_select_device() if self.gpu_optimizer else "cpu"
            )
            self.embedder = BGE_M3Embedder(
                device=device, use_fp16=use_gpu, batch_size=32 if use_gpu else 8
            )
        else:
            self.embedder = embedder

        # Initialize vector store
        if vector_store is None:
            dimension = self.embedder.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(
                dimension=dimension,
                index_type="IVF",
                use_gpu=self._get_gpu_setting(),  # 環境変数対応GPU設定
                store_path=store_path,
            )
        else:
            self.vector_store = vector_store

    def _get_gpu_setting(self) -> bool:
        """
        環境変数とCUDA利用可能性に基づいてGPU設定を決定

        環境変数BGE_FAISS_DEVICE:
        - 'gpu': GPU強制使用
        - 'cpu': CPU強制使用
        - 'auto' (default): CUDA利用可能性で自動選択

        Returns:
            bool: GPU使用の可否
        """
        force_device = os.environ.get('BGE_FAISS_DEVICE', 'auto').lower()

        if force_device == 'cpu':
            logger.info("GPU usage disabled by BGE_FAISS_DEVICE=cpu")
            return False
        elif force_device == 'gpu':
            if torch.cuda.is_available():
                logger.info("GPU usage forced by BGE_FAISS_DEVICE=gpu")
                return True
            else:
                logger.warning("GPU forced but CUDA not available, using CPU")
                return False
        else:  # auto
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                logger.info("CUDA detected, using GPU acceleration")
            else:
                logger.info("CUDA not available, using CPU")
            return gpu_available

        logger.info("Semantic Retriever initialized")

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Add documents to the retriever.

        Args:
            texts: List of document texts
            ids: Optional document IDs
            metadata: Optional metadata for each document
            batch_size: Batch size for embedding generation

        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Initialize metadata if not provided
        if metadata is None:
            metadata = [{} for _ in texts]

        # Create Document objects
        documents = [
            Document(id=doc_id, content=text, metadata=meta)
            for doc_id, text, meta in zip(ids, texts, metadata)
        ]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        if self.gpu_optimizer:
            self.gpu_optimizer.monitor_memory("Before embedding")

        embeddings = self.embedder.encode_batch(texts, batch_size=batch_size)
        embeddings_array = np.array(embeddings)

        if self.gpu_optimizer:
            self.gpu_optimizer.monitor_memory("After embedding")
            self.gpu_optimizer.clear_cache()

        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents, embeddings_array)

        logger.info(f"Added {len(doc_ids)} documents to retriever")
        return doc_ids

    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            return_scores: Include similarity scores

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding, k=k, filter_metadata=filter_metadata
        )

        # Format results
        formatted_results = []
        for doc, score in results:
            result = {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
            if return_scores:
                result["score"] = score
            formatted_results.append(result)

        return formatted_results

    def search_batch(
        self, queries: List[str], k: int = 10, batch_size: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries in batch.

        Args:
            queries: List of search queries
            k: Number of results per query
            batch_size: Batch size for embedding generation

        Returns:
            List of search results for each query
        """
        # Generate query embeddings
        query_embeddings = self.embedder.encode_batch(queries, batch_size=batch_size)

        # Search for each query
        all_results = []
        for query, embedding in zip(queries, query_embeddings):
            results = self.vector_store.search(embedding, k=k)

            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )
            all_results.append(formatted_results)

        return all_results

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on semantic similarity.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        # Generate embeddings
        query_embedding = self.embedder.encode(query)
        doc_embeddings = self.embedder.encode_batch(documents)
        doc_embeddings_array = np.array(doc_embeddings)

        # Calculate similarities
        similarities = self.embedder.similarity(
            query_embedding.reshape(1, -1), doc_embeddings_array
        )

        # Sort by similarity
        scores = similarities[0]
        ranked_indices = np.argsort(scores)[::-1]

        # Return top k if specified
        if top_k:
            ranked_indices = ranked_indices[:top_k]

        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def save(self, path: Optional[Path] = None):
        """Save the vector store to disk."""
        self.vector_store.save(path)
        logger.info("Retriever saved")

    def load(self, path: Optional[Path] = None):
        """Load the vector store from disk."""
        self.vector_store.load(path)
        logger.info("Retriever loaded")

    def clear(self):
        """Clear the vector store."""
        self.vector_store.clear()
        if self.gpu_optimizer:
            self.gpu_optimizer.clear_cache()
        logger.info("Retriever cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "vector_store": self.vector_store.get_stats(),
            "embedder": {
                "model": self.embedder.model_name,
                "device": self.embedder.device,
                "dimension": self.embedder.get_embedding_dimension(),
                "batch_size": self.embedder.batch_size,
            },
        }

        if self.gpu_optimizer:
            stats["gpu"] = self.gpu_optimizer.get_memory_stats()

        return stats

    def find_similar(self, document_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.

        Args:
            document_id: ID of the reference document
            k: Number of similar documents to return

        Returns:
            List of similar documents
        """
        # Get document
        doc = self.vector_store.get_document_by_id(document_id)
        if not doc:
            logger.warning(f"Document {document_id} not found")
            return []

        # Use document's embedding if available, otherwise generate
        if doc.embedding is not None:
            embedding = doc.embedding
        else:
            embedding = self.embedder.encode(doc.content)

        # Search for similar documents (excluding the source)
        results = self.vector_store.search(embedding, k=k + 1)

        # Filter out the source document and format results
        formatted_results = []
        for result_doc, score in results:
            if result_doc.id != document_id:
                formatted_results.append(
                    {
                        "id": result_doc.id,
                        "content": result_doc.content,
                        "metadata": result_doc.metadata,
                        "score": score,
                    }
                )

            if len(formatted_results) >= k:
                break

        return formatted_results
