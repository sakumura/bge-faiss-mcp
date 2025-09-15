#!/usr/bin/env python3
"""
Semantic Search MCP Server

Provides semantic search capabilities through Model Context Protocol (MCP).
Uses BGE-M3 embeddings and FAISS vector store for high-performance search.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

# Import MCP SDK
from mcp.server import FastMCP

# Import our semantic search components
from bge_faiss_mcp.core.manager import HybridSearchManager, SearchMode

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("semantic-search")

# Global search manager
search_manager = None


def get_working_directory():
    """Get the actual working directory dynamically"""
    # 1. Check MCP-specific environment variables
    if "MCP_WORKING_DIR" in os.environ:
        logger.info(f"Using MCP_WORKING_DIR: {os.environ['MCP_WORKING_DIR']}")
        return Path(os.environ["MCP_WORKING_DIR"])

    # 2. Check common working directory environment variables
    if "WORKING_DIR" in os.environ:
        logger.info(f"Using WORKING_DIR: {os.environ['WORKING_DIR']}")
        return Path(os.environ["WORKING_DIR"])

    # 3. Check for command-line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--working-dir" and i + 1 < len(sys.argv):
                working_dir = Path(sys.argv[i + 1])
                logger.info(f"Using command-line working dir: {working_dir}")
                return working_dir

    # 4. Try to detect from process environment
    # MCP might set the current directory when launching the process
    cwd = Path.cwd()
    logger.info(f"Using current working directory: {cwd}")

    # Log all environment variables for debugging
    logger.debug("Environment variables:")
    for key, value in os.environ.items():
        if "DIR" in key.upper() or "PATH" in key.upper() or "MCP" in key.upper():
            logger.debug(f"  {key}: {value}")

    return cwd


def initialize_search_manager():
    """Initialize the search manager"""
    global search_manager

    if search_manager is not None:
        return

    try:
        # Get the actual working directory
        working_dir = get_working_directory()
        logger.info(f"Working directory: {working_dir}")

        # Use default config - completely independent from Serena MCP
        config = {
            "semantic_index_path": ".search/vectors",
            "default_mode": "auto",
            "enable_cache": True,
        }
        logger.info(f"Config: {config}")

        # Create search manager - use the detected working directory for index paths
        semantic_path = working_dir / config["semantic_index_path"]
        logger.info(f"Semantic index path: {semantic_path}")

        # Debug: Check if paths exist
        logger.info(f"Semantic path exists: {semantic_path.exists()}")

        # Debug: Try importing dependencies first
        logger.info("Attempting to import dependencies...")
        try:
            from bge_faiss_mcp.core.manager import HybridSearchManager, SearchMode

            logger.info("✅ HybridSearchManager imported successfully")
        except ImportError as ie:
            logger.error(f"❌ Failed to import HybridSearchManager: {ie}")
            raise

        logger.info("Creating HybridSearchManager instance...")
        manager = HybridSearchManager(
            semantic_index_path=str(semantic_path),
            default_mode=SearchMode.AUTO,
            enable_cache=config["enable_cache"],
        )

        search_manager = manager  # Assign to global variable
        logger.info("✅ Search manager initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize search manager: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Don't raise - allow partial functionality
        search_manager = None


@mcp.tool()
def search(
    query: str, k: int = 5, mode: str = "auto", project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform semantic, keyword, or pattern search with automatic mode selection

    Args:
        query: Search query
        k: Number of results to return
        mode: Search mode (auto, semantic, lightweight, pattern)
        project_path: Optional path to project directory

    Returns:
        Search results with metadata
    """
    try:
        logger.info(f"Search request: query='{query}', k={k}, mode='{mode}'")

        # Initialize search manager for specific project path if provided
        if project_path:
            working_dir = Path(project_path).resolve()
            original_cwd = Path.cwd()
            try:
                os.chdir(working_dir)
                initialize_search_manager_for_path(working_dir)
            finally:
                os.chdir(original_cwd)
        elif search_manager is None:
            logger.info("Initializing search manager...")
            initialize_search_manager()

        if search_manager is None:
            logger.error("Search manager is None after initialization")
            return {"error": "Search manager initialization failed"}

        logger.info(f"Search manager ready: {type(search_manager)}")

        if not query:
            return {"error": "Query parameter is required"}

        # Convert mode string to SearchMode
        search_mode = None
        if mode != "auto":
            try:
                search_mode = SearchMode(mode)
                logger.info(f"Using search mode: {search_mode}")
            except ValueError:
                logger.warning(f"Invalid search mode: {mode}, falling back to auto")
                search_mode = None

        # Execute search
        logger.info("Executing search...")
        results = search_manager.search(query=query, k=k, mode=search_mode)

        logger.info(f"Search completed, found {len(results) if results else 0} results")

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "content": result.content[:500],  # Truncate for readability
                    "score": result.score,
                    "source": result.source,
                    "mode": result.mode.value,
                    "metadata": result.metadata,
                }
            )

        return {
            "query": query,
            "results": formatted_results,
            "total": len(formatted_results),
            "mode_used": results[0].mode.value if results else "none",
        }

    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}


@mcp.tool()
def get_stats() -> Dict[str, Any]:
    """
    Get search engine statistics and capabilities

    Returns:
        Statistics and available modes
    """
    try:
        # Initialize search manager if needed
        if search_manager is None:
            initialize_search_manager()

        if search_manager is None:
            return {
                "status": "not_initialized",
                "error": "Failed to initialize search manager",
            }

        stats = search_manager.get_stats()
        modes = search_manager.get_available_modes()

        return {
            "status": "ready",
            "available_modes": [mode.value for mode in modes],
            "statistics": stats,
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return {"error": str(e)}


def initialize_search_manager_for_path(working_dir: Path):
    """Initialize search manager for a specific path"""
    global search_manager

    try:
        logger.info(f"🎯 Initializing search manager for specific path: {working_dir}")

        # Use default config - completely independent from Serena MCP
        config = {
            "semantic_index_path": ".search/vectors",
            "default_mode": "auto",
            "enable_cache": True,
        }

        # Create search manager for the specific path
        semantic_path = working_dir / config["semantic_index_path"]

        logger.info(f"📁 Semantic path: {semantic_path}")
        logger.info(f"📁 Semantic path exists: {semantic_path.exists()}")

        # 修正: SearchMode.AUTOを直接使用
        search_manager = HybridSearchManager(
            semantic_index_path=str(semantic_path),
            default_mode=SearchMode.AUTO,  # 修正箇所
            enable_cache=config["enable_cache"],
        )

        logger.info(f"✅ Search manager initialized for path: {working_dir}")
        logger.info(f"🔍 Search manager type: {type(search_manager)}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize search manager for {working_dir}: {e}")
        import traceback

        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        raise


@mcp.tool()
def build_index(project_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Build or rebuild the search index for a project

    Args:
        project_path: Path to the project directory (uses current dir if not specified)

    Returns:
        Index building status and statistics
    """
    try:
        # Determine project path
        if project_path:
            working_dir = Path(project_path).resolve()
        else:
            working_dir = get_working_directory()

        logger.info(f"Building index for: {working_dir}")

        # Try to initialize search manager if it failed before
        if search_manager is None:
            try:
                initialize_search_manager()
                if search_manager is None:
                    return {
                        "status": "failed",
                        "project_path": str(working_dir),
                        "message": "Search manager initialization failed",
                    }
            except Exception as init_e:
                logger.error(f"Search manager re-initialization failed: {init_e}")
                return {
                    "status": "failed",
                    "project_path": str(working_dir),
                    "message": f"Search manager initialization failed: {str(init_e)}",
                }

        # Save current directory
        original_cwd = Path.cwd()

        try:
            # Change to project directory
            os.chdir(working_dir)

            # Initialize search manager for this path
            initialize_search_manager_for_path(working_dir)

            # Build index
            if search_manager and search_manager.semantic_retriever:
                success = search_manager.build_initial_index()
                stats = search_manager.get_stats()

                num_docs = 0
                if "semantic" in stats and "vector_store" in stats["semantic"]:
                    num_docs = stats["semantic"]["vector_store"].get("num_documents", 0)

                return {
                    "status": "success" if success else "failed",
                    "project_path": str(working_dir),
                    "documents_indexed": num_docs,
                    "message": (
                        f"Index built with {num_docs} documents"
                        if success
                        else "Failed to build index"
                    ),
                }
            else:
                # Try to initialize if not already done
                if search_manager is None:
                    logger.info(
                        "Search manager not initialized, attempting initialization..."
                    )
                    initialize_search_manager_for_path(working_dir)

                if search_manager and search_manager.semantic_retriever:
                    success = search_manager.build_initial_index()
                    stats = search_manager.get_stats()

                    num_docs = 0
                    if "semantic" in stats and "vector_store" in stats["semantic"]:
                        num_docs = stats["semantic"]["vector_store"].get(
                            "num_documents", 0
                        )

                    return {
                        "status": "success" if success else "failed",
                        "project_path": str(working_dir),
                        "documents_indexed": num_docs,
                        "message": (
                            f"Index built with {num_docs} documents"
                            if success
                            else "Failed to build index"
                        ),
                    }
                else:
                    return {
                        "status": "failed",
                        "project_path": str(working_dir),
                        "message": "Search manager initialization failed",
                    }

        finally:
            # Restore original directory
            os.chdir(original_cwd)

    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def clear_index(project_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear the search index for a project

    Args:
        project_path: Path to the project directory

    Returns:
        Clear status
    """
    try:
        # Determine project path
        if project_path:
            working_dir = Path(project_path).resolve()
        else:
            working_dir = get_working_directory()

        logger.info(f"Clearing index for: {working_dir}")

        # Clear index directory
        index_path = working_dir / ".search"
        if index_path.exists():
            import shutil

            shutil.rmtree(index_path)
            logger.info(f"Cleared index at {index_path}")

            return {
                "status": "success",
                "project_path": str(working_dir),
                "message": "Index cleared successfully",
            }
        else:
            return {
                "status": "success",
                "project_path": str(working_dir),
                "message": "No index to clear",
            }

    except Exception as e:
        logger.error(f"Clear index failed: {e}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def configure_update_strategy(
    strategy: str = "auto",
    project_path: Optional[str] = None,
    time_threshold: Optional[int] = None,
    count_threshold: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Configure the auto-reindexing strategy

    Args:
        strategy: Update strategy ('auto', 'manual', 'time', 'count')
        project_path: Path to the project directory
        time_threshold: Time threshold in seconds (for 'time' strategy)
        count_threshold: Search count threshold (for 'count' strategy)

    Returns:
        Configuration status
    """
    try:
        # Validate strategy
        valid_strategies = ["auto", "manual", "time", "count"]
        if strategy not in valid_strategies:
            return {
                "status": "error",
                "message": f"Invalid strategy. Must be one of: {valid_strategies}",
            }

        # Initialize search manager for specific project path if provided
        if project_path:
            working_dir = Path(project_path).resolve()
            original_cwd = Path.cwd()
            try:
                os.chdir(working_dir)
                initialize_search_manager_for_path(working_dir)
            finally:
                os.chdir(original_cwd)
        elif search_manager is None:
            logger.info("Initializing search manager...")
            initialize_search_manager()

        if search_manager is None or not search_manager.semantic_retriever:
            return {"status": "error", "message": "Search manager not available"}

        # Update configuration
        vector_store = search_manager.semantic_retriever.vector_store
        metadata = vector_store.index_metadata

        metadata["update_strategy"] = strategy

        # Set thresholds if provided
        if time_threshold is not None:
            metadata["time_threshold"] = time_threshold
        if count_threshold is not None:
            metadata["count_threshold"] = count_threshold

        # Save updated metadata
        vector_store.save()

        logger.info(f"Update strategy configured: {strategy}")

        return {
            "status": "success",
            "strategy": strategy,
            "time_threshold": metadata.get("time_threshold", 3600),
            "count_threshold": metadata.get("count_threshold", 10),
            "message": f"Update strategy set to '{strategy}'",
        }

    except Exception as e:
        logger.error(f"Configure update strategy failed: {e}")
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Semantic Search MCP Server")
    try:
        # Initialize search manager at startup
        logger.info("Initializing search manager...")
        initialize_search_manager()
        logger.info("Search manager initialized successfully")

        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
