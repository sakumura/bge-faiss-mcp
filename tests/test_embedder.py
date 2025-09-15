"""Tests for BGE-M3 embedder functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from bge_faiss_mcp.core.embedder import BGE_M3Embedder


class TestBGE_M3Embedder:
    """Test cases for BGE_M3Embedder."""

    def test_embedder_initialization(self):
        """Test embedder can be initialized."""
        with patch("sentence_transformers.SentenceTransformer"):
            embedder = BGE_M3Embedder()
            assert embedder is not None
            assert embedder.model_name == "BAAI/bge-m3"
            assert embedder.get_embedding_dimension() == 1024

    @patch("sentence_transformers.SentenceTransformer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_encode_single(self, mock_cuda, mock_transformer):
        """Test encoding single text."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 1024)  # Shape for single text
        mock_transformer.return_value = mock_model

        with patch.object(BGE_M3Embedder, '_initialize_model'):
            embedder = BGE_M3Embedder()
            embedder.model = mock_model  # Directly assign mock model
            result = embedder.encode("Hello world")

        assert isinstance(result, np.ndarray)
        # encode() returns shape (1, 1024) for single text
        assert result.shape == (1, 1024)
        mock_model.encode.assert_called_once_with(
            ["Hello world"], batch_size=32, normalize_embeddings=True, 
            show_progress_bar=False, convert_to_numpy=True, device="cpu"
        )

    @patch("sentence_transformers.SentenceTransformer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_encode_batch(self, mock_cuda, mock_transformer):
        """Test encoding batch of texts."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 1024)
        mock_transformer.return_value = mock_model

        with patch.object(BGE_M3Embedder, '_initialize_model'):
            embedder = BGE_M3Embedder()
            embedder.model = mock_model  # Directly assign mock model
            texts = ["Hello", "World", "Test"]
            result = embedder.encode(texts)  # Use encode, not encode_batch

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)
        mock_model.encode.assert_called_once_with(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, device="cpu")

    @patch("sentence_transformers.SentenceTransformer")
    def test_similarity_calculation(self, mock_transformer):
        """Test similarity calculation between embeddings."""
        # Mock the transformer
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        embedder = BGE_M3Embedder()
        
        # Test similarity calculation
        query_emb = np.random.rand(1, 1024).astype(np.float32)
        doc_embs = np.random.rand(2, 1024).astype(np.float32)
        
        with patch.object(embedder, 'model', mock_model):
            similarities = embedder.similarity(query_emb, doc_embs)
            
            assert isinstance(similarities, np.ndarray)
            assert similarities.shape == (1, 2)

    @patch("sentence_transformers.SentenceTransformer")
    def test_empty_text_handling(self, mock_transformer):
        """Test handling of empty text."""
        mock_model = Mock()
        mock_model.encode.return_value = np.zeros((1, 1024))
        mock_transformer.return_value = mock_model

        embedder = BGE_M3Embedder()
        result = embedder.encode("")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1024)  # Single text returns shape (1, 1024)

    @patch("sentence_transformers.SentenceTransformer")
    def test_device_detection(self, mock_transformer):
        """Test device detection (CPU/GPU)."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False):
            embedder = BGE_M3Embedder()
            # Should fallback to CPU when GPU not available
            assert embedder is not None
