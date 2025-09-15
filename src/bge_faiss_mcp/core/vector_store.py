"""
FAISS Vector Store with GPU Acceleration

Efficient vector storage and similarity search using FAISS with CPU fallback.
"""

import faiss  # type: ignore
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with metadata."""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class FAISSVectorStore:
    """FAISS-based vector store with GPU optimization support."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "IVF",
        use_gpu: bool = False,
        gpu_id: int = 0,
        nlist: int = 100,
        nprobe: int = 10,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of index ('Flat', 'IVF', 'HNSW')
            use_gpu: Attempt to use GPU (falls back to CPU if unavailable)
            gpu_id: GPU device ID
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search
            store_path: Path to save/load index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.nlist = nlist
        self.nprobe = nprobe
        # Handle store_path (may already be Path type)
        if store_path is not None:
            self.store_path = (
                store_path if isinstance(store_path, Path) else Path(store_path)
            )
        else:
            self.store_path = Path(".search/vectors")

        # Document storage
        self.documents: List[Document] = []
        self.id_to_index: Dict[str, int] = {}

        # Index metadata for change detection
        import time

        self.index_metadata = {
            "last_scan_time": 0.0,
            "file_count": 0,
            "search_count": 0,
            "created_at": time.time(),
            "update_strategy": "auto",  # auto, manual, time, count
        }

        # Initialize index
        self._initialize_index()

    def _initialize_index(self, expected_vectors: Optional[int] = None):
        """Initialize FAISS index based on configuration and expected data size.

        Args:
            expected_vectors: Expected number of vectors (for dynamic index selection)
        """
        # Determine actual index type based on data size
        if expected_vectors is not None:
            if expected_vectors < 100:
                # Small dataset: use Flat index
                actual_type = "Flat"
                logger.info(
                    f"Small dataset ({expected_vectors} vectors), using Flat index"
                )
            elif expected_vectors < 1000:
                # Medium dataset: use IVF with adjusted nlist
                actual_type = "IVF"
                # Adjust nlist for medium datasets
                self.nlist = min(10, max(4, expected_vectors // 25))
                logger.info(
                    f"Medium dataset ({expected_vectors} vectors), using IVF with nlist={self.nlist}"
                )
            else:
                # Large dataset: use configured settings
                actual_type = self.index_type
                logger.info(
                    f"Large dataset ({expected_vectors} vectors), using {actual_type} index"
                )
        else:
            actual_type = self.index_type

        logger.info(
            f"Initializing FAISS {actual_type} index (dimension={self.dimension})"
        )

        # Create base index
        if actual_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            self.actual_index_type = "Flat"
        elif actual_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.actual_index_type = "IVF"
        elif actual_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.actual_index_type = "HNSW"
        else:
            raise ValueError(f"Unknown index type: {actual_type}")

        # GPU configuration
        self.gpu_index = None
        self.gpu_resource = None
        if self.use_gpu:
            try:
                # Check if FAISS GPU is available
                gpu_available = hasattr(faiss, 'StandardGpuResources')
                if gpu_available:
                    self.gpu_resource = faiss.StandardGpuResources()
                    logger.info(f"FAISS GPU support available, moving index to GPU {self.gpu_id}")
                    # Move index to GPU after training (if needed)
                    self._gpu_ready = True
                else:
                    logger.warning("FAISS GPU build not available, using CPU index")
                    self.use_gpu = False
                    self._gpu_ready = False
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS GPU: {e}, using CPU")
                self.use_gpu = False
                self._gpu_ready = False
        else:
            self._gpu_ready = False

        # Set search parameters for IVF
        actual_type = getattr(self, "actual_index_type", self.index_type)
        if actual_type == "IVF":
            self.index.nprobe = self.nprobe

        logger.info(f"Index initialized: {type(self.index).__name__}")

    def add_documents(
        self, documents: List[Document], embeddings: np.ndarray
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store.

        Args:
            documents: List of documents
            embeddings: Numpy array of embeddings

        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Check if we need to adjust index type for small datasets
        actual_type = getattr(self, "actual_index_type", self.index_type)

        # Train index if needed (for IVF)
        if actual_type == "IVF" and not self.index.is_trained:
            if len(embeddings) < self.nlist:
                # Not enough vectors for IVF, fallback to Flat
                logger.warning(
                    f"Not enough vectors ({len(embeddings)}) for IVF with nlist={self.nlist}"
                )
                logger.info("Falling back to Flat index")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.actual_index_type = "Flat"
            else:
                try:
                    logger.info(f"Training IVF index with {len(embeddings)} vectors...")
                    self.index.train(embeddings.astype(np.float32))

                    # Move to GPU after training if GPU is enabled
                    if self._gpu_ready and self.gpu_resource:
                        try:
                            self.gpu_index = faiss.index_cpu_to_gpu(
                                self.gpu_resource, self.gpu_id, self.index
                            )
                            logger.info(f"Index successfully moved to GPU {self.gpu_id}")
                        except Exception as e:
                            logger.warning(f"Failed to move index to GPU: {e}, using CPU")
                            self.gpu_index = None

                except Exception as e:
                    # Training failed, fallback to Flat
                    logger.error(f"IVF training failed: {e}")
                    logger.info("Falling back to Flat index")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.actual_index_type = "Flat"

        # Move Flat index to GPU if enabled and not already done
        if (self._gpu_ready and self.gpu_resource and
            self.gpu_index is None and self.actual_index_type == "Flat"):
            try:
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resource, self.gpu_id, self.index
                )
                logger.info(f"Flat index moved to GPU {self.gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to move Flat index to GPU: {e}")

        # Add embeddings to index (use GPU index if available)
        start_idx = len(self.documents)
        active_index = self.gpu_index if self.gpu_index else self.index
        active_index.add(embeddings.astype(np.float32))

        # Store documents and update mappings
        doc_ids = []
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            self.documents.append(doc)
            self.id_to_index[doc.id] = start_idx + i
            doc_ids.append(doc.id)

        logger.info(f"Added {len(documents)} documents to vector store")
        return doc_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        if len(self.documents) == 0:
            return []

        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search index (use GPU index if available)
        k_search = min(k * 3, len(self.documents)) if filter_metadata else k
        active_index = self.gpu_index if self.gpu_index else self.index
        distances, indices = active_index.search(
            query_embedding.astype(np.float32), k_search
        )

        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]

                # Apply metadata filter if provided
                if filter_metadata:
                    if not all(
                        doc.metadata.get(key) == value
                        for key, value in filter_metadata.items()
                    ):
                        continue

                # Convert L2 distance to similarity score (1 / (1 + distance))
                score = 1.0 / (1.0 + float(dist))
                results.append((doc, score))

                if len(results) >= k:
                    break

        return results

    def search_by_text(
        self, query_text: str, embedder, k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Search using text query (requires embedder).

        Args:
            query_text: Text query
            embedder: Embedder instance to generate embedding
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        query_embedding = embedder.encode(query_text)
        return self.search(query_embedding, k)

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        idx = self.id_to_index.get(doc_id)
        if idx is not None and idx < len(self.documents):
            return self.documents[idx]
        return None

    def save(self, path: Optional[Path] = None):
        """
        Save index and documents to disk.

        Args:
            path: Save path (uses store_path if not provided)
        """
        save_path = path or self.store_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save documents and mappings
        data_file = save_path / "documents.pkl"
        import time

        self.index_metadata["last_scan_time"] = time.time()
        self.index_metadata["file_count"] = len(self.documents)

        with open(data_file, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "id_to_index": self.id_to_index,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "index_metadata": self.index_metadata,
                },
                f,
            )

        logger.info(f"Vector store saved to {save_path}")

    def load(self, path: Optional[Path] = None):
        """
        Load index and documents from disk.

        Args:
            path: Load path (uses store_path if not provided)
        """
        if path is not None:
            load_path = path if isinstance(path, Path) else Path(path)
        else:
            load_path = self.store_path

        # Load FAISS index
        index_file = load_path / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))

            # Load documents and mappings
            data_file = load_path / "documents.pkl"
            if data_file.exists():
                try:
                    # Try to modify sys.path temporarily for pickle loading
                    import sys

                    original_path = sys.path.copy()

                    # Add common paths that might contain the required modules
                    additional_paths = [
                        str(Path(__file__).parent.parent.parent.parent),  # Project root
                        str(Path(__file__).parent.parent.parent),  # src/python
                        str(Path(__file__).parent.parent),  # semantic_rag
                        str(Path(__file__).parent),  # core
                    ]

                    for p in additional_paths:
                        if p not in sys.path:
                            sys.path.insert(0, p)

                    try:
                        with open(data_file, "rb") as f:
                            data = pickle.load(f)
                            self.documents = data["documents"]
                            self.id_to_index = data["id_to_index"]

                            # Load metadata if available
                            if "index_metadata" in data:
                                self.index_metadata.update(data["index_metadata"])
                            else:
                                # Legacy support: initialize with default values
                                import time

                                self.index_metadata = {
                                    "last_scan_time": time.time(),
                                    "file_count": len(self.documents),
                                    "search_count": 0,
                                    "created_at": time.time(),
                                    "update_strategy": "auto",
                                }

                        logger.info(f"Vector store loaded from {load_path}")
                        logger.info(f"Loaded {len(self.documents)} documents")
                        logger.info(
                            f"Last scan: {self.index_metadata.get('last_scan_time', 'Unknown')}"
                        )
                    finally:
                        # Restore original sys.path
                        sys.path = original_path

                except (ModuleNotFoundError, ImportError, pickle.UnpicklingError) as e:
                    logger.warning(
                        f"Failed to load pickled documents (likely due to module path changes): {e}"
                    )
                    logger.info("Rebuilding index to resolve module path issues...")

                    # Clear the existing but problematic index
                    self.documents = []
                    self.id_to_index = {}

                    # Delete the problematic pickle file so it gets regenerated
                    try:
                        data_file.unlink()
                        logger.info(f"Removed problematic pickle file: {data_file}")
                    except Exception as del_e:
                        logger.warning(f"Could not remove pickle file: {del_e}")

                except Exception as e:
                    logger.error(f"Unexpected error loading documents: {e}")
                    self.documents = []
                    self.id_to_index = {}
            else:
                logger.warning(f"Documents file not found at {data_file}")
        else:
            logger.warning(f"Index file not found at {index_file}")

    def clear(self):
        """Clear the vector store."""
        self._initialize_index()
        self.documents.clear()
        self.id_to_index.clear()
        logger.info("Vector store cleared")

    def increment_search_count(self):
        """Increment search counter for update triggers."""
        self.index_metadata["search_count"] += 1

    def _get_time_since_last_scan(self) -> float:
        """Get time elapsed since last scan in seconds."""
        import time

        return time.time() - self.index_metadata.get("last_scan_time", 0)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_size": self.index.ntotal if hasattr(self.index, "ntotal") else 0,
            "is_trained": (
                self.index.is_trained if hasattr(self.index, "is_trained") else True
            ),
            "actual_index_type": getattr(self, "actual_index_type", self.index_type),
            "metadata": self.index_metadata,
            "time_since_last_scan": self._get_time_since_last_scan(),
            "use_gpu": self.use_gpu,
        }

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.
        Note: This requires rebuilding the index.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        if doc_id not in self.id_to_index:
            return False

        # Get all documents except the one to remove
        remaining_docs = []
        remaining_embeddings = []

        for i, doc in enumerate(self.documents):
            if doc.id != doc_id:
                remaining_docs.append(doc)
                if doc.embedding is not None:
                    remaining_embeddings.append(doc.embedding)

        # Clear and rebuild
        self.clear()

        if remaining_docs and remaining_embeddings:
            embeddings_array = np.array(remaining_embeddings)
            self.add_documents(remaining_docs, embeddings_array)

        logger.info(f"Document {doc_id} removed")
        return True
