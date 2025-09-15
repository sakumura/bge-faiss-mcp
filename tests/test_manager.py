"""Tests for hybrid search manager functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from bge_faiss_mcp.core.manager import HybridSearchManager, SearchMode, SearchResult


class TestHybridSearchManager:
    """Test cases for HybridSearchManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.semantic_index_path = str(Path(self.temp_dir) / "vectors")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_manager_initialization(self, mock_analyzer, mock_rag, mock_retriever):
        """Test manager initialization."""
        # Mock dependencies
        mock_analyzer.return_value = Mock()
        mock_retriever_instance = Mock()
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 0}
        mock_retriever.return_value = mock_retriever_instance
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(
            semantic_index_path=self.semantic_index_path,
            default_mode=SearchMode.AUTO,
            enable_cache=True,
        )

        assert str(manager.semantic_index_path) == str(self.semantic_index_path)
        assert manager.default_mode == SearchMode.AUTO
        assert manager.enable_cache is True
        assert manager.semantic_retriever is not None
        assert manager.query_analyzer is not None

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_search_semantic_mode(self, mock_analyzer, mock_rag, mock_retriever):
        """Test semantic search mode."""
        # Mock query analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_query.return_value = {
            "is_semantic": True,
            "complexity": 0.8,
            "keywords": ["test", "search"],
        }
        mock_analyzer.return_value = mock_analyzer_instance

        # Mock semantic retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.search.return_value = [
            {
                "id": "doc1",
                "content": "Test document 1",
                "metadata": {"source": "test1.txt"},
                "score": 0.95
            },
            {
                "id": "doc2", 
                "content": "Test document 2",
                "metadata": {"source": "test2.txt"},
                "score": 0.85
            },
        ]
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 2}
        mock_retriever_instance.rerank.return_value = [(0, 0.95), (1, 0.85)]  # (index, score) tuples
        mock_retriever.return_value = mock_retriever_instance
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(
            semantic_index_path=self.semantic_index_path,
            default_mode=SearchMode.SEMANTIC,
        )

        results = manager.search("semantic search query", k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].mode == SearchMode.SEMANTIC
        assert results[0].score == 0.95
        assert "test1.txt" in str(results[0].metadata)

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_search_pattern_mode(self, mock_analyzer, mock_rag, mock_retriever):
        """Test pattern search mode."""
        mock_analyzer.return_value = Mock()
        mock_retriever_instance = Mock()
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 0}
        mock_retriever.return_value = mock_retriever_instance
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(
            semantic_index_path=self.semantic_index_path,
            default_mode=SearchMode.PATTERN,
        )

        # Mock pattern search (simplified)
        with patch.object(manager, "_pattern_search") as mock_pattern:
            mock_pattern.return_value = [
                SearchResult(
                    content="Pattern match result",
                    score=1.0,
                    metadata={"source": "pattern.txt", "line": 42},
                    source="pattern.txt",
                    mode=SearchMode.PATTERN,
                )
            ]

            results = manager.search("function\\s+\\w+", k=1, mode=SearchMode.PATTERN)

            assert len(results) == 1
            assert results[0].mode == SearchMode.PATTERN
            assert results[0].score == 1.0

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_search_auto_mode_selection(self, mock_analyzer, mock_rag, mock_retriever):
        """Test automatic mode selection."""
        # Mock query analyzer to suggest semantic search
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_query.return_value = {
            "is_semantic": True,
            "complexity": 0.7,
            "keywords": ["natural", "language"],
        }
        mock_analyzer.return_value = mock_analyzer_instance

        # Mock semantic retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.search.return_value = [
            {
                "id": "auto_doc",
                "content": "Auto mode result", 
                "metadata": {"source": "auto.txt"}, 
                "score": 0.9
            }
        ]
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 1}
        mock_retriever_instance.rerank.return_value = [(0, 0.9)]  # (index, score) tuples
        mock_retriever.return_value = mock_retriever_instance
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(
            semantic_index_path=self.semantic_index_path, default_mode=SearchMode.AUTO
        )

        results = manager.search("natural language query", k=1)

        # Should automatically choose semantic mode
        assert len(results) == 1
        assert results[0].mode == SearchMode.SEMANTIC

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_get_available_modes(self, mock_analyzer, mock_rag, mock_retriever):
        """Test getting available search modes."""
        mock_analyzer.return_value = Mock()
        mock_retriever_instance = Mock()
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 0}
        mock_retriever.return_value = mock_retriever_instance
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(semantic_index_path=self.semantic_index_path)

        modes = manager.get_available_modes()

        assert SearchMode.SEMANTIC in modes
        assert SearchMode.PATTERN in modes
        assert SearchMode.AUTO in modes
        assert len(modes) == 3

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_get_stats(self, mock_analyzer, mock_rag, mock_retriever):
        """Test getting search statistics."""
        # Mock retriever with stats
        mock_retriever_instance = Mock()
        mock_retriever_instance.get_stats.return_value = {
            "num_documents": 100,
            "index_size": 1024000,
            "last_update": "2025-01-12T10:00:00Z",
        }
        mock_retriever.return_value = mock_retriever_instance
        mock_analyzer.return_value = Mock()
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(semantic_index_path=self.semantic_index_path)

        stats = manager.get_stats()

        assert "semantic" in stats
        assert stats["semantic"]["num_documents"] == 100
        assert stats["semantic"]["index_size"] == 1024000

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_build_initial_index(self, mock_analyzer, mock_rag, mock_retriever):
        """Test building initial search index."""
        # Mock retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.build_index.return_value = True
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 0}
        mock_retriever.return_value = mock_retriever_instance
        mock_analyzer.return_value = Mock()
        mock_rag.return_value = Mock()

        # Mock the _scan_project_files to return some files
        with patch.object(HybridSearchManager, '_scan_project_files') as mock_scan:
            mock_scan.return_value = [Path(self.temp_dir) / 'test.py']
            with patch.object(HybridSearchManager, '_read_file') as mock_read:
                mock_read.return_value = 'test content'
                with patch('pathlib.Path.cwd', return_value=Path(self.temp_dir)):
                    manager = HybridSearchManager(semantic_index_path=self.semantic_index_path)
                    success = manager.build_initial_index()

                    assert success is True

    @patch("bge_faiss_mcp.core.manager.SemanticRetriever")
    @patch("bge_faiss_mcp.core.manager.RAGChain")
    @patch("bge_faiss_mcp.core.manager.QueryAnalyzer")
    def test_error_handling(self, mock_analyzer, mock_rag, mock_retriever):
        """Test error handling in search operations."""
        # Mock retriever to raise exception
        mock_retriever_instance = Mock()
        mock_retriever_instance.search.side_effect = Exception("Search failed")
        mock_retriever_instance.vector_store = Mock()
        mock_retriever_instance.vector_store.get_stats.return_value = {"num_documents": 1}
        mock_retriever.return_value = mock_retriever_instance
        mock_analyzer.return_value = Mock()
        mock_rag.return_value = Mock()

        manager = HybridSearchManager(semantic_index_path=self.semantic_index_path)

        # Should handle exception gracefully by returning empty results
        results = manager.search("test query", k=1, mode=SearchMode.SEMANTIC)
        assert results == []  # Should return empty list when search fails
