"""Tests for FAISS vector store operations."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from bge_faiss_mcp.core.vector_store import FAISSVectorStore, Document


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.temp_dir) / "test_store"
        self.dimension = 1024

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        assert store.store_path == self.store_path
        assert store.dimension == self.dimension
        assert store.index is not None
        assert store.documents == []

    def test_add_vectors_and_search(self):
        """Test adding vectors and searching."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        # Create test vectors
        vectors = np.random.rand(10, self.dimension).astype(np.float32)
        documents = [
            Document(
                id=f"doc_{i}",
                content=f"Document {i}",
                metadata={"source": f"test{i}.txt"},
                embedding=vectors[i],
            )
            for i in range(10)
        ]

        # Add vectors
        store.add_documents(documents, vectors)

        # Search
        query_vector = vectors[0]  # Use first vector as query
        results = store.search(query_vector, k=3)

        assert len(results) == 3
        assert results[0][0].content == "Document 0"  # Should find exact match first
        assert results[0][1] > 0.99  # High similarity score

    def test_save_and_load(self):
        """Test saving and loading index."""
        # Create first store and add data
        store1 = FAISSVectorStore(store_path=self.store_path, dimension=self.dimension)

        vectors = np.random.rand(5, self.dimension).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=f"Document {i}", metadata={}, embedding=vectors[i])
            for i in range(5)
        ]

        store1.add_documents(documents, vectors)
        store1.save()

        # Create second store and load
        store2 = FAISSVectorStore(store_path=self.store_path, dimension=self.dimension)
        store2.load()

        assert store2.index.ntotal == 5
        assert len(store2.documents) == 5

        # Test search on loaded index
        query_vector = vectors[0]
        results = store2.search(query_vector, k=1)
        assert len(results) == 1
        assert results[0][0].content == "Document 0"

    def test_empty_search(self):
        """Test search on empty index."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        query_vector = np.random.rand(self.dimension).astype(np.float32)
        results = store.search(query_vector, k=5)

        assert results == []

    def test_clear_index(self):
        """Test clearing the index."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        # Add some data
        vectors = np.random.rand(3, self.dimension).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=f"Doc {i}", metadata={}, embedding=vectors[i])
            for i in range(3)
        ]
        store.add_documents(documents, vectors)

        assert store.index.ntotal == 3
        assert len(store.documents) == 3

        # Clear
        store.clear()

        assert store.index.ntotal == 0
        assert len(store.documents) == 0

    def test_get_stats(self):
        """Test getting index statistics."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        # Empty index
        stats = store.get_stats()
        assert stats["num_documents"] == 0
        assert stats["dimension"] == self.dimension
        assert stats["index_type"] == "IVF"

        # Add some data
        vectors = np.random.rand(5, self.dimension).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=f"Doc {i}", metadata={}, embedding=vectors[i])
            for i in range(5)
        ]
        store.add_documents(documents, vectors)

        stats = store.get_stats()
        assert stats["num_documents"] == 5

    def test_invalid_dimension(self):
        """Test handling of invalid dimensions."""
        store = FAISSVectorStore(dimension=self.dimension, store_path=self.store_path)

        # Try to add vectors with wrong dimension
        wrong_vectors = np.random.rand(2, 512).astype(np.float32)  # Wrong dimension
        documents = [
            Document(id="doc_1", content="Doc 1", metadata={}, embedding=wrong_vectors[0]),
            Document(id="doc_2", content="Doc 2", metadata={}, embedding=wrong_vectors[1]),
        ]

        with pytest.raises(Exception):
            store.add_documents(documents, wrong_vectors)
