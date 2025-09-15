"""Tests for MCP server functionality."""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from bge_faiss_mcp.core.manager import SearchMode, SearchResult


class TestMCPServer:
    """Test cases for MCP server functionality."""

    @patch("bge_faiss_mcp.server.initialize_search_manager")
    @patch("bge_faiss_mcp.server.search_manager", new=Mock())
    def test_search_function_basic(self, mock_init):
        """Test basic search function."""
        from bge_faiss_mcp.server import search

        # Mock search manager
        mock_manager = Mock()
        mock_manager.search.return_value = [
            SearchResult(
                content="Test result",
                score=0.85,
                metadata={"source": "test.txt"},
                source="test.txt",
                mode=SearchMode.SEMANTIC,
            )
        ]

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            result = search("test query", k=1, mode="semantic")

            assert result["query"] == "test query"
            assert result["total"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "Test result"
            assert result["results"][0]["score"] == 0.85
            assert result["results"][0]["mode"] == "semantic"

    @patch("bge_faiss_mcp.server.initialize_search_manager")
    def test_search_function_no_results(self, mock_init):
        """Test search function with no results."""
        from bge_faiss_mcp.server import search

        # Mock search manager with no results
        mock_manager = Mock()
        mock_manager.search.return_value = []

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            result = search("no match query", k=5)

            assert result["query"] == "no match query"
            assert result["total"] == 0
            assert result["results"] == []
            assert result["mode_used"] == "none"

    @patch("bge_faiss_mcp.server.initialize_search_manager")
    def test_search_function_error_handling(self, mock_init):
        """Test search function error handling."""
        from bge_faiss_mcp.server import search

        # Mock search manager to raise exception
        mock_manager = Mock()
        mock_manager.search.side_effect = Exception("Search error")

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            result = search("error query", k=1)

            assert "error" in result
            assert result["error"] == "Search error"

    def test_get_stats_function(self):
        """Test get_stats function."""
        from bge_faiss_mcp.server import get_stats

        # Mock search manager with stats
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            "semantic": {"num_documents": 50, "index_size": 512000}
        }
        mock_manager.get_available_modes.return_value = [
            SearchMode.SEMANTIC,
            SearchMode.PATTERN,
            SearchMode.AUTO,
        ]

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            result = get_stats()

            assert result["status"] == "ready"
            assert len(result["available_modes"]) == 3
            assert "semantic" in result["available_modes"]
            assert "pattern" in result["available_modes"]
            assert "auto" in result["available_modes"]
            assert result["statistics"]["semantic"]["num_documents"] == 50

    def test_get_stats_not_initialized(self):
        """Test get_stats when manager not initialized."""
        from bge_faiss_mcp.server import get_stats

        with patch("bge_faiss_mcp.server.search_manager", None), \
             patch("bge_faiss_mcp.server.initialize_search_manager"):
            result = get_stats()

            assert result["status"] == "not_initialized"
            assert "error" in result

    @patch("bge_faiss_mcp.server.initialize_search_manager_for_path")
    @patch("bge_faiss_mcp.server.get_working_directory")
    def test_build_index_function(self, mock_get_wd, mock_init_path):
        """Test build_index function."""
        from bge_faiss_mcp.server import build_index
        from pathlib import Path

        # Mock working directory
        mock_get_wd.return_value = Path("/test/project")

        # Mock search manager
        mock_manager = Mock()
        mock_manager.build_initial_index.return_value = True
        mock_manager.get_stats.return_value = {
            "semantic": {"vector_store": {"num_documents": 25}}
        }

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            with patch("os.chdir"):
                result = build_index()

                assert result["status"] == "success"
                assert result["documents_indexed"] == 25
                assert "Index built with 25 documents" in result["message"]

    @patch("bge_faiss_mcp.server.get_working_directory")
    def test_clear_index_function(self, mock_get_wd):
        """Test clear_index function."""
        from bge_faiss_mcp.server import clear_index
        from pathlib import Path
        import tempfile
        import os

        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            search_dir = test_dir / ".search"
            search_dir.mkdir()
            (search_dir / "test_file.txt").write_text("test")

            mock_get_wd.return_value = test_dir

            result = clear_index()

            assert result["status"] == "success"
            assert "Index cleared successfully" in result["message"]
            assert not search_dir.exists()

    def test_configure_update_strategy_function(self):
        """Test configure_update_strategy function."""
        from bge_faiss_mcp.server import configure_update_strategy

        # Mock search manager and vector store
        mock_vector_store = Mock()
        mock_vector_store.index_metadata = {}
        mock_vector_store.save = Mock()

        mock_retriever = Mock()
        mock_retriever.vector_store = mock_vector_store

        mock_manager = Mock()
        mock_manager.semantic_retriever = mock_retriever

        with patch("bge_faiss_mcp.server.search_manager", mock_manager):
            result = configure_update_strategy(
                strategy="time", time_threshold=7200, count_threshold=20
            )

            assert result["status"] == "success"
            assert result["strategy"] == "time"
            assert result["time_threshold"] == 7200
            assert result["count_threshold"] == 20

            # Verify metadata was updated
            assert mock_vector_store.index_metadata["update_strategy"] == "time"
            assert mock_vector_store.index_metadata["time_threshold"] == 7200
            assert mock_vector_store.index_metadata["count_threshold"] == 20
            mock_vector_store.save.assert_called_once()

    def test_configure_update_strategy_invalid(self):
        """Test configure_update_strategy with invalid strategy."""
        from bge_faiss_mcp.server import configure_update_strategy

        result = configure_update_strategy(strategy="invalid")

        assert result["status"] == "error"
        assert "Invalid strategy" in result["message"]

    @patch("bge_faiss_mcp.server.Path")
    def test_get_working_directory_env_var(self, mock_path):
        """Test get_working_directory with environment variables."""
        from bge_faiss_mcp.server import get_working_directory
        import os

        # Mock Path.cwd()
        mock_path.cwd.return_value = Path("/fallback/dir")
        mock_path.return_value = Path("/env/dir")

        # Test MCP_WORKING_DIR
        with patch.dict(os.environ, {"MCP_WORKING_DIR": "/env/dir"}):
            result = get_working_directory()
            assert str(result).replace("\\", "/") == "/env/dir"

        # Test WORKING_DIR
        with patch.dict(os.environ, {"WORKING_DIR": "/env/dir"}, clear=True):
            result = get_working_directory()
            assert str(result).replace("\\", "/") == "/env/dir"

    @patch("bge_faiss_mcp.server.sys.argv", ["script.py", "--working-dir", "/cmd/dir"])
    @patch("bge_faiss_mcp.server.Path")
    def test_get_working_directory_cmd_line(self, mock_path):
        """Test get_working_directory with command line args."""
        from bge_faiss_mcp.server import get_working_directory

        mock_path.return_value = Path("/cmd/dir")

        result = get_working_directory()
        assert str(result).replace("\\", "/") == "/cmd/dir"
