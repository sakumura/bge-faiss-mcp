"""
HybridSearchManager - セマンティック検索と軽量検索の統合マネージャー

2つの検索システムを統合し、クエリに応じて最適なシステムを選択。
セマンティック検索（BGE-M3）と軽量検索（MiniLM）を効率的に使い分ける。
"""

import logging
import os
import json
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from bge_faiss_mcp.core.retriever import SemanticRetriever
from bge_faiss_mcp.core.rag import RAGChain
from bge_faiss_mcp.utils.analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """検索モードの定義"""

    SEMANTIC = "semantic"  # BGE-M3による高精度セマンティック検索
    PATTERN = "pattern"  # 正規表現パターン検索
    AUTO = "auto"  # 自動選択


@dataclass
class SearchResult:
    """統一検索結果フォーマット"""

    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # ファイルパス
    mode: SearchMode  # 使用した検索モード


class HybridSearchManager:
    """
    セマンティック検索とパターン検索を統合管理するマネージャー

    クエリの複雑度に応じて最適な検索エンジンを自動選択し、
    統一されたインターフェースで検索結果を提供する。
    """

    def __init__(
        self,
        semantic_index_path: Optional[str] = None,
        default_mode: SearchMode = SearchMode.AUTO,
        enable_cache: bool = True,
    ):
        """
        Args:
            semantic_index_path: BGE-M3インデックスのパス
            default_mode: デフォルトの検索モード
            enable_cache: 検索結果のキャッシュを有効にするか
        """
        # Convert to Path objects for consistent handling
        self.semantic_index_path = (
            Path(semantic_index_path)
            if semantic_index_path
            else Path(".search/vectors")
        )
        self.default_mode = default_mode
        self.enable_cache = enable_cache

        # 各コンポーネントの初期化
        self.query_analyzer = QueryAnalyzer()
        self.semantic_retriever = None
        self.rag_chain = None

        # キャッシュ
        self._cache = {} if enable_cache else None

        # 検索エンジンの初期化
        self._initialize_engines()

        logger.info(f"HybridSearchManager initialized with mode: {default_mode}")

    def _initialize_engines(self):
        """検索エンジンの遅延初期化"""
        logger.info("🔧 Starting engine initialization...")

        # セマンティック検索の初期化（常に実行）
        try:
            logger.info("📦 Importing semantic search components...")
            from bge_faiss_mcp.core.embedder import BGE_M3Embedder
            from bge_faiss_mcp.core.vector_store import FAISSVectorStore

            logger.info("✅ Semantic imports successful")

            # ディレクトリ作成
            logger.info(
                f"📁 Creating semantic index directory: {self.semantic_index_path}"
            )
            self.semantic_index_path.mkdir(parents=True, exist_ok=True)
            logger.info("✅ Directory created successfully")

            # GPU/CPU自動選択
            logger.info("🚀 Initializing BGE-M3 embedder...")
            embedder = BGE_M3Embedder(
                model_name="BAAI/bge-m3", use_fp16=True, batch_size=32
            )
            logger.info("✅ BGE-M3 embedder initialized")

            # ベクトルストア初期化
            logger.info("🗄️ Initializing FAISS vector store...")
            dimension = embedder.get_embedding_dimension()
            logger.info(f"📐 Embedding dimension: {dimension}")
            vector_store = FAISSVectorStore(
                dimension=dimension,
                store_path=self.semantic_index_path,
                index_type="IVF",  # デフォルトはIVF（後で動的に調整される）
                nlist=100,  # デフォルト値（後で動的に調整される）
            )
            logger.info("✅ FAISS vector store initialized")

            # インデックスのロード（既存の場合）
            if (self.semantic_index_path / "index.faiss").exists():
                vector_store.load(self.semantic_index_path)
                logger.info("Semantic index loaded successfully")
            else:
                logger.info("No existing index found, will create on first search")

            # Retriever初期化
            self.semantic_retriever = SemanticRetriever(
                embedder=embedder, vector_store=vector_store
            )

            # RAGチェーン初期化
            self.rag_chain = RAGChain(
                retriever=self.semantic_retriever, max_context_length=2000
            )

            logger.info("Semantic search engine initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic search: {e}")
            import traceback

            logger.error(f"📋 Full traceback: {traceback.format_exc()}")
            self.semantic_retriever = None

        logger.info("✅ Hybrid search manager initialization complete")

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Optional[SearchMode] = None,
        rerank: bool = True,
    ) -> List[SearchResult]:
        """
        統合検索インターフェース

        Args:
            query: 検索クエリ
            k: 取得する結果数
            mode: 検索モード（None の場合は自動選択）
            rerank: リランキングを実行するか

        Returns:
            統一フォーマットの検索結果リスト
        """
        # 初回検索時にインデックスが空なら構築
        if self.semantic_retriever and not self._has_index():
            logger.info("No index found, building initial index...")
            self.build_initial_index()
        # 既存インデックスがある場合、ファイル変更をチェック
        elif self.semantic_retriever and self._check_files_changed():
            logger.info("File changes detected, updating index...")
            self.build_initial_index()

        # 検索カウンターを増加（トリガー判定用）
        if self.semantic_retriever and self.semantic_retriever.vector_store:
            self.semantic_retriever.vector_store.increment_search_count()

        # キャッシュチェック
        cache_key = f"{query}_{k}_{mode}_{rerank}"
        if self._cache is not None and cache_key in self._cache:
            logger.debug(f"Cache hit for query: {query}")
            return self._cache[cache_key]

        # 検索モードの決定
        if mode is None:
            mode = self._determine_search_mode(query)

        logger.info(f"Searching with mode: {mode.value} for query: {query}")

        # 検索実行
        results = []

        if mode == SearchMode.SEMANTIC and self.semantic_retriever:
            results = self._semantic_search(query, k, rerank)
        elif mode == SearchMode.PATTERN:
            results = self._pattern_search(query, k)
        elif mode == SearchMode.AUTO:
            # 自動モードでフォールバック戦略
            results = self._auto_search(query, k, rerank)
        else:
            logger.warning(f"Search mode {mode} not available")
            # フォールバック
            results = self._fallback_search(query, k)

        # キャッシュ保存
        if self._cache is not None:
            self._cache[cache_key] = results

        return results

    def _determine_search_mode(self, query: str) -> SearchMode:
        """クエリを分析して最適な検索モードを決定"""
        if self.default_mode != SearchMode.AUTO:
            return self.default_mode

        # クエリ分析
        analysis = self.query_analyzer.analyze(query)

        # 分析結果に基づいてモード選択
        if analysis.needs_semantic_understanding:
            if self.semantic_retriever:
                return SearchMode.SEMANTIC
        elif analysis.is_simple_keyword:
            if self.semantic_retriever:
                return SearchMode.SEMANTIC
        elif analysis.has_regex_pattern:
            return SearchMode.PATTERN

        # デフォルトはセマンティック検索
        return SearchMode.SEMANTIC if self.semantic_retriever else SearchMode.PATTERN

    def _semantic_search(self, query: str, k: int, rerank: bool) -> List[SearchResult]:
        """BGE-M3によるセマンティック検索"""
        try:
            raw_results = self.semantic_retriever.search(
                query, k=k * 2 if rerank else k
            )

            # リランキング
            if rerank and len(raw_results) > 0:
                # rerank用にdocument textを抽出
                doc_texts = [item.get("content", "") for item in raw_results]
                rerank_results = self.semantic_retriever.rerank(
                    query, doc_texts, top_k=k
                )

                # rerankの結果を元のdict形式に戻す
                reranked_raw_results = []
                for idx, score in rerank_results:
                    if idx < len(raw_results):
                        item = raw_results[idx].copy()
                        item["score"] = score  # rerank scoreで更新
                        reranked_raw_results.append(item)
                raw_results = reranked_raw_results

            # 結果を統一フォーマットに変換
            results = []
            for item in raw_results[:k]:
                # itemはdict形式のはず
                if isinstance(item, dict):
                    result = SearchResult(
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                        metadata=item.get("metadata", {}),
                        source=item.get("metadata", {}).get("source", ""),
                        mode=SearchMode.SEMANTIC,
                    )
                    results.append(result)
                else:
                    logger.error(f"Unexpected result format: {type(item)} - {item}")

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _pattern_search(self, query: str, k: int) -> List[SearchResult]:
        """パターン検索（将来的にSerenaのsearch_for_patternと連携）"""
        logger.info("Pattern search not yet implemented, falling back")
        return []

    def _auto_search(self, query: str, k: int, rerank: bool) -> List[SearchResult]:
        """
        自動モードでの検索（優先順位に従って実行）
        1. セマンティック検索
        2. 軽量版検索
        3. パターン検索
        """
        # まずセマンティック検索を試す
        if self.semantic_retriever:
            results = self._semantic_search(query, k, rerank)
            if results:
                return results

        # 最後にパターン検索
        results = self._pattern_search(query, k)
        return results

    def _fallback_search(self, query: str, k: int) -> List[SearchResult]:
        """フォールバック検索（何も利用できない場合）"""
        logger.warning("No search engines available, returning empty results")
        return []

    def query_with_context(
        self, query: str, k: int = 5, mode: Optional[SearchMode] = None
    ) -> Dict[str, Any]:
        """
        RAGチェーンを使用してコンテキスト付き回答を生成

        Args:
            query: 質問
            k: 検索する文書数
            mode: 検索モード

        Returns:
            回答とソース情報を含む辞書
        """
        if not self.rag_chain:
            # RAGチェーンが利用できない場合は通常の検索
            results = self.search(query, k, mode)
            return {
                "answer": "RAG chain not available",
                "sources": [r.source for r in results],
                "documents": [r.content for r in results],
            }

        # 検索モードの決定
        if mode is None:
            mode = self._determine_search_mode(query)

        # モードに応じてRAGチェーンを使用
        if mode == SearchMode.SEMANTIC:
            return self.rag_chain.answer_with_sources(query, k=k)
        else:
            # 他のモードでは通常の検索結果を返す
            results = self.search(query, k, mode)
            return {
                "answer": None,
                "sources": [r.source for r in results],
                "documents": [r.content for r in results],
            }

    def _scan_project_files(self, max_files: int = 1000) -> List[Path]:
        """プロジェクトファイルをスキャン"""
        import os
        from pathlib import Path

        cwd = Path.cwd()
        files = []

        # 除外するディレクトリ
        exclude_dirs = {
            ".git",
            ".search",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
        }

        # 対象とする拡張子
        include_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".css",
            ".scss",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
        }

        for root, dirs, filenames in os.walk(cwd):
            # 除外ディレクトリをスキップ
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                if len(files) >= max_files:
                    break

                file_path = Path(root) / filename

                # ファイルサイズチェック（10MB以下）
                if file_path.stat().st_size > 10 * 1024 * 1024:
                    continue

                # 拡張子チェック
                if file_path.suffix.lower() in include_extensions:
                    files.append(file_path)

        logger.info(f"Found {len(files)} files to index")
        return files

    def _read_file(self, file_path: Path) -> str:
        """ファイルを読み込み"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # 最大文字数制限（100KB）
                if len(content) > 100000:
                    content = content[:100000]
                return content
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""

    def build_initial_index(self) -> bool:
        """Build initial index"""
        if not self.semantic_retriever:
            logger.warning("Semantic retriever not initialized")
            return False

        try:
            logger.info("Building initial index...")

            # プロジェクトファイルをスキャン
            files = self._scan_project_files()

            if not files:
                logger.warning("No files found to index")
                return False

            logger.info(f"Found {len(files)} files to index")

            # ファイル数に基づいてインデックスタイプを動的に調整
            if len(files) < 100:
                logger.info(
                    f"Small dataset detected ({len(files)} files), reinitializing with optimized settings"
                )
                # 小規模データセット用に再初期化
                self._reinitialize_for_small_dataset(len(files))

            # バッチ処理用にデータを準備
            texts = []
            ids = []
            metadata = []

            for file_path in files:
                content = self._read_file(file_path)
                if content:
                    # ファイル名と内容を結合
                    relative_path = file_path.relative_to(Path.cwd())
                    text = f"File: {relative_path}\n\n{content}"

                    texts.append(text)
                    ids.append(str(relative_path))
                    metadata.append(
                        {
                            "source": str(relative_path),
                            "file_type": file_path.suffix,
                            "size": len(content),
                        }
                    )

            if texts:
                # インデックスに追加
                logger.info(f"Indexing {len(texts)} documents...")
                self.semantic_retriever.add_documents(
                    texts=texts,
                    ids=ids,
                    metadata=metadata,
                    batch_size=16,  # バッチサイズを小さめに
                )

                # インデックスを保存
                self.semantic_retriever.vector_store.save()
                logger.info(f"Initial index built with {len(texts)} documents")
                return True
            else:
                logger.warning("No valid content found to index")
                return False

        except Exception as e:
            logger.error(f"Failed to build initial index: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _reinitialize_for_small_dataset(self, file_count: int):
        """Reinitialize vector store for small datasets"""
        try:
            from bge_faiss_mcp.core.embedder import BGE_M3Embedder
            from bge_faiss_mcp.core.vector_store import FAISSVectorStore

            # 既存のembedderを保持
            embedder = self.semantic_retriever.embedder

            # 小規模データセット用に最適化されたベクトルストアを作成
            dimension = embedder.get_embedding_dimension()

            # ファイル数に基づいて適切な設定を選択
            if file_count < 10:
                # 非常に小さいデータセット: Flatインデックス
                index_type = "Flat"
                nlist = 1
            elif file_count < 50:
                # 小さいデータセット: 小さいnlistでIVF
                index_type = "IVF"
                nlist = 4
            else:
                # 中規模データセット: 調整されたnlist
                index_type = "IVF"
                nlist = min(10, file_count // 10)

            logger.info(
                f"Reinitializing vector store for {file_count} files with {index_type} index (nlist={nlist})"
            )

            # 新しいベクトルストアを作成
            vector_store = FAISSVectorStore(
                dimension=dimension,
                store_path=self.semantic_index_path,
                index_type=index_type,
                nlist=nlist,
            )

            # expected_vectors パラメータで最適化を呼び出す
            vector_store._initialize_index(expected_vectors=file_count)

            # Retrieverのベクトルストアを更新
            self.semantic_retriever.vector_store = vector_store
            logger.info(f"Vector store reinitialized successfully for small dataset")

        except Exception as e:
            logger.error(f"Failed to reinitialize for small dataset: {e}")
            # エラーが発生してもそのまま続行（既存の設定を使用）

    def _check_files_changed(self) -> bool:
        """Check if project files have changed since last scan."""
        if not self.semantic_retriever or not self.semantic_retriever.vector_store:
            return False

        try:
            import time
            import os
            from pathlib import Path

            metadata = self.semantic_retriever.vector_store.index_metadata
            last_scan_time = metadata.get("last_scan_time", 0)
            search_count = metadata.get("search_count", 0)
            update_strategy = metadata.get("update_strategy", "auto")

            # Manual mode: never auto-update
            if update_strategy == "manual":
                return False

            # Time-based check (default: 1 hour)
            if update_strategy == "time":
                time_threshold = metadata.get("time_threshold", 3600)  # 1 hour
                if time.time() - last_scan_time > time_threshold:
                    logger.info(
                        f"Time threshold exceeded ({time_threshold}s), checking for changes"
                    )
                    return self._scan_for_file_changes(last_scan_time)
                return False

            # Count-based check (default: every 10 searches)
            if update_strategy == "count":
                count_threshold = metadata.get("count_threshold", 10)
                if search_count >= count_threshold:
                    logger.info(
                        f"Search count threshold reached ({search_count}/{count_threshold})"
                    )
                    # Reset counter
                    metadata["search_count"] = 0
                    return self._scan_for_file_changes(last_scan_time)
                return False

            # Auto mode: intelligent checks based on project size
            file_count = metadata.get("file_count", 0)
            if file_count < 100:
                # Small projects: check every 5 searches
                if search_count >= 5:
                    metadata["search_count"] = 0
                    return self._scan_for_file_changes(last_scan_time)
            elif file_count < 1000:
                # Medium projects: check every 15 searches
                if search_count >= 15:
                    metadata["search_count"] = 0
                    return self._scan_for_file_changes(last_scan_time)
            else:
                # Large projects: time-based (30 minutes)
                if time.time() - last_scan_time > 1800:
                    return self._scan_for_file_changes(last_scan_time)

            return False

        except Exception as e:
            logger.warning(f"Failed to check file changes: {e}")
            return False

    def _scan_for_file_changes(self, last_scan_time: float) -> bool:
        """Scan project files for changes since last_scan_time."""
        try:
            import os
            from pathlib import Path

            # Get current project files
            current_files = self._scan_project_files()

            # Check for new or modified files
            changes_detected = False

            for file_path in current_files:
                try:
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime > last_scan_time:
                        logger.debug(f"File changed: {file_path}")
                        changes_detected = True
                        break  # Early exit for performance
                except (OSError, FileNotFoundError):
                    # File might have been deleted or inaccessible
                    continue

            if changes_detected:
                logger.info("File changes detected")
                return True

            # Check if files were deleted (current count vs indexed count)
            metadata = self.semantic_retriever.vector_store.index_metadata
            indexed_count = metadata.get("file_count", 0)
            current_count = len(current_files)

            if current_count != indexed_count:
                logger.info(f"File count changed: {indexed_count} -> {current_count}")
                return True

            logger.debug("No file changes detected")
            return False

        except Exception as e:
            logger.warning(f"Failed to scan for file changes: {e}")
            return False

    def _has_index(self) -> bool:
        """Check if index exists"""
        if not self.semantic_retriever:
            return False

        stats = self.semantic_retriever.vector_store.get_stats()
        return stats.get("num_documents", 0) > 0

    def clear_cache(self):
        """検索キャッシュをクリア"""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Search cache cleared")

    def get_available_modes(self) -> List[SearchMode]:
        """利用可能な検索モードを取得"""
        modes = [SearchMode.AUTO]

        if self.semantic_retriever:
            modes.append(SearchMode.SEMANTIC)

        # パターン検索は常に利用可能（将来実装）
        modes.append(SearchMode.PATTERN)

        return modes

    def get_stats(self) -> Dict[str, Any]:
        """検索エンジンの統計情報を取得"""
        stats = {
            "available_modes": [m.value for m in self.get_available_modes()],
            "default_mode": self.default_mode.value,
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._cache) if self._cache else 0,
        }

        # セマンティック検索の統計
        if self.semantic_retriever:
            stats["semantic"] = self.semantic_retriever.get_stats()

        return stats


# エクスポート
__all__ = ["HybridSearchManager", "SearchMode", "SearchResult"]
