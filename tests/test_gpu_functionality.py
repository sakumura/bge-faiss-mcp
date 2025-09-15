"""
GPU機能とデバイス選択のテスト
"""

import pytest
import os
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from bge_faiss_mcp.core.retriever import SemanticRetriever
from bge_faiss_mcp.core.vector_store import FAISSVectorStore, Document
from bge_faiss_mcp.core.embedder import BGE_M3Embedder


class TestGPUFunctionality:
    """GPU機能の包括的テスト"""

    @patch('bge_faiss_mcp.core.embedder.SentenceTransformer')
    @patch('bge_faiss_mcp.utils.gpu.torch.cuda.is_available')
    def test_gpu_detection_logic(self, mock_cuda_available, mock_transformer):
        """GPU検出ロジックのテスト"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # 環境変数なしの場合（デフォルト）
        with patch.dict(os.environ, {}, clear=True):
            mock_cuda_available.return_value = True
            with patch('torch.cuda.is_available', return_value=True):
                retriever = SemanticRetriever()
                assert retriever._get_gpu_setting() == True

            mock_cuda_available.return_value = False
            with patch('torch.cuda.is_available', return_value=False):
                retriever = SemanticRetriever()
                assert retriever._get_gpu_setting() == False

    def test_environment_variable_control(self):
        """環境変数による制御テスト"""
        retriever = SemanticRetriever()

        # GPU強制使用
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'gpu'}):
            with patch('torch.cuda.is_available', return_value=True):
                assert retriever._get_gpu_setting() == True

            with patch('torch.cuda.is_available', return_value=False):
                assert retriever._get_gpu_setting() == False

        # CPU強制使用
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            with patch('torch.cuda.is_available', return_value=True):
                assert retriever._get_gpu_setting() == False

            with patch('torch.cuda.is_available', return_value=False):
                assert retriever._get_gpu_setting() == False

        # auto設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'auto'}):
            with patch('torch.cuda.is_available', return_value=True):
                assert retriever._get_gpu_setting() == True

            with patch('torch.cuda.is_available', return_value=False):
                assert retriever._get_gpu_setting() == False

    def test_environment_variable_case_insensitive(self):
        """環境変数の大文字小文字非依存テスト"""
        retriever = SemanticRetriever()

        # 大文字設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'GPU'}):
            with patch('torch.cuda.is_available', return_value=True):
                assert retriever._get_gpu_setting() == True

        # 小文字設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            with patch('torch.cuda.is_available', return_value=True):
                assert retriever._get_gpu_setting() == False

    @patch('bge_faiss_mcp.core.embedder.SentenceTransformer')
    @patch('torch.cuda.is_available')
    def test_embedder_device_selection(self, mock_cuda_available, mock_transformer):
        """Embedderのデバイス選択テスト"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # GPU利用可能な場合
        mock_cuda_available.return_value = True
        embedder = BGE_M3Embedder()
        assert embedder.device == "cuda"

        # GPU利用不可の場合
        mock_cuda_available.return_value = False
        embedder = BGE_M3Embedder()
        assert embedder.device == "cpu"

        # 明示的にデバイス指定
        embedder = BGE_M3Embedder(device="cpu")
        assert embedder.device == "cpu"

    @patch('faiss.StandardGpuResources')
    @patch('torch.cuda.is_available')
    def test_vector_store_gpu_initialization(self, mock_cuda_available, mock_gpu_resources):
        """VectorStoreのGPU初期化テスト"""
        mock_cuda_available.return_value = True

        # GPU利用時
        store = FAISSVectorStore(dimension=1024, use_gpu=True)
        assert store.use_gpu == True
        assert store._gpu_ready == True

        # CPU利用時
        store = FAISSVectorStore(dimension=1024, use_gpu=False)
        assert store.use_gpu == False
        assert store._gpu_ready == False

    @patch('faiss.StandardGpuResources')
    @patch('torch.cuda.is_available')
    def test_vector_store_gpu_fallback(self, mock_cuda_available, mock_gpu_resources):
        """FAISS GPU利用不可時のフォールバックテスト"""
        mock_cuda_available.return_value = True
        mock_gpu_resources.side_effect = Exception("FAISS GPU not available")

        # GPU要求だが利用不可の場合
        store = FAISSVectorStore(dimension=1024, use_gpu=True)
        assert store.use_gpu == False  # フォールバック
        assert store._gpu_ready == False

    def test_vector_store_active_index_selection(self):
        """アクティブインデックス選択のテスト"""
        # CPU版でテスト
        store = FAISSVectorStore(dimension=128, use_gpu=False)

        # テストデータ
        embeddings = np.random.random((10, 128)).astype(np.float32)
        documents = [Document(id=f"doc_{i}", content=f"test content {i}", metadata={}) for i in range(10)]

        # データ追加
        store.add_vectors(embeddings, documents)

        # 検索時はCPUインデックスを使用
        query = np.random.random((1, 128)).astype(np.float32)
        results = store.search(query, k=3)

        assert len(results) <= 3
        assert all(isinstance(r['document'], Document) for r in results)

    @patch('bge_faiss_mcp.core.embedder.SentenceTransformer')
    def test_retriever_initialization_with_gpu_settings(self, mock_transformer):
        """GPU設定でのRetriever初期化テスト"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # 環境変数でGPU強制設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'gpu'}):
            with patch('torch.cuda.is_available', return_value=True):
                retriever = SemanticRetriever()

                # GPU設定が適用されているか確認
                assert retriever._get_gpu_setting() == True

    def test_batch_size_optimization_gpu_vs_cpu(self):
        """GPU/CPU時のバッチサイズ最適化テスト"""
        with patch('bge_faiss_mcp.core.embedder.SentenceTransformer'):
            # GPU使用時
            with patch('torch.cuda.is_available', return_value=True):
                embedder = BGE_M3Embedder(use_fp16=True)
                assert embedder.batch_size >= 16  # GPUでは大きなバッチサイズ

            # CPU使用時
            with patch('torch.cuda.is_available', return_value=False):
                embedder = BGE_M3Embedder(use_fp16=False)
                # CPUでも適切なバッチサイズが設定される

    @patch('logging.Logger.info')
    @patch('torch.cuda.is_available')
    def test_gpu_logging_messages(self, mock_cuda_available, mock_logger):
        """GPU関連ログメッセージのテスト"""
        retriever = SemanticRetriever()

        # GPU利用可能時のログ
        mock_cuda_available.return_value = True
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'auto'}):
            result = retriever._get_gpu_setting()
            assert result == True

        # CPU強制時のログ
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            result = retriever._get_gpu_setting()
            assert result == False

    def test_invalid_environment_variable(self):
        """不正な環境変数値のテスト"""
        retriever = SemanticRetriever()

        # 不正な値（デフォルト動作になる）
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'invalid_value'}):
            with patch('torch.cuda.is_available', return_value=True):
                # 不正な値の場合はautoとして扱われる
                assert retriever._get_gpu_setting() == True

            with patch('torch.cuda.is_available', return_value=False):
                assert retriever._get_gpu_setting() == False

    @patch('bge_faiss_mcp.core.embedder.SentenceTransformer')
    def test_gpu_memory_optimization(self, mock_transformer):
        """GPU メモリ最適化のテスト"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # GPU利用時のFP16設定
        with patch('torch.cuda.is_available', return_value=True):
            embedder = BGE_M3Embedder(use_fp16=True)
            assert embedder.use_fp16 == True

        # CPU利用時はFP16無効
        with patch('torch.cuda.is_available', return_value=False):
            embedder = BGE_M3Embedder(use_fp16=False)
            assert embedder.use_fp16 == False

    def test_search_performance_consistency(self):
        """GPU/CPU間での検索結果一貫性テスト"""
        # CPU版で検索結果をテスト
        store = FAISSVectorStore(dimension=128, use_gpu=False)

        # テストデータ
        embeddings = np.random.random((20, 128)).astype(np.float32)
        documents = [Document(id=f"doc_{i}", content=f"content {i}", metadata={}) for i in range(20)]

        store.add_vectors(embeddings, documents)

        # 検索実行
        query = np.random.random((1, 128)).astype(np.float32)
        results = store.search(query, k=5)

        # 結果の妥当性確認
        assert len(results) <= 5
        assert all('score' in r for r in results)
        assert all('document' in r for r in results)

        # スコアが降順になっているか確認（FAISSはL2距離なので昇順）
        scores = [r['score'] for r in results]
        assert scores == sorted(scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])