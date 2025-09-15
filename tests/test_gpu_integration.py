"""
GPU機能の統合テスト - 実際の動作を確認
"""

import pytest
import os
import numpy as np
from unittest.mock import patch
from pathlib import Path

from bge_faiss_mcp.core.retriever import SemanticRetriever
from bge_faiss_mcp.core.vector_store import FAISSVectorStore, Document


class TestGPUIntegration:
    """GPU機能の統合テスト"""

    def test_environment_variable_setting(self):
        """環境変数設定のテスト"""
        # CPU強制設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            os.environ['BGE_FAISS_DEVICE'] = 'cpu'
            assert os.environ.get('BGE_FAISS_DEVICE') == 'cpu'

        # GPU強制設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'gpu'}):
            os.environ['BGE_FAISS_DEVICE'] = 'gpu'
            assert os.environ.get('BGE_FAISS_DEVICE') == 'gpu'

        # auto設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'auto'}):
            os.environ['BGE_FAISS_DEVICE'] = 'auto'
            assert os.environ.get('BGE_FAISS_DEVICE') == 'auto'

    def test_vector_store_cpu_mode(self):
        """VectorStoreのCPUモードテスト"""
        # CPU強制モード
        store = FAISSVectorStore(dimension=128, use_gpu=False)

        # 基本設定確認
        assert store.use_gpu == False
        assert store.dimension == 128

        # テストデータ
        embeddings = np.random.random((5, 128)).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=f"test content {i}", metadata={})
            for i in range(5)
        ]

        # データ追加
        store.add_vectors(embeddings, documents)
        assert len(store.documents) == 5

        # 検索テスト
        query = np.random.random((1, 128)).astype(np.float32)
        results = store.search(query, k=3)

        assert len(results) <= 3
        assert all('score' in r for r in results)
        assert all('document' in r for r in results)

    def test_vector_store_stats(self):
        """VectorStoreの統計情報テスト"""
        store = FAISSVectorStore(dimension=64, use_gpu=False)

        # 初期状態
        stats = store.get_stats()
        assert stats['total_documents'] == 0
        assert stats['dimension'] == 64

        # データ追加後
        embeddings = np.random.random((3, 64)).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=f"content {i}", metadata={})
            for i in range(3)
        ]

        store.add_vectors(embeddings, documents)
        stats = store.get_stats()
        assert stats['total_documents'] == 3

    def test_semantic_retriever_cpu_initialization(self):
        """SemanticRetrieverのCPU初期化テスト"""
        # CPU強制設定でRetrieverを初期化
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()

            # GPU設定メソッドの確認
            assert hasattr(retriever, '_get_gpu_setting')
            gpu_setting = retriever._get_gpu_setting()
            # CPU強制なのでFalse
            assert gpu_setting == False

    def test_document_operations(self):
        """文書操作の統合テスト"""
        store = FAISSVectorStore(dimension=32, use_gpu=False)

        # 複数の文書を追加
        texts = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "FAISS is for similarity search",
            "BGE-M3 provides embeddings"
        ]

        embeddings = np.random.random((len(texts), 32)).astype(np.float32)
        documents = [
            Document(id=f"doc_{i}", content=text, metadata={"type": "test"})
            for i, text in enumerate(texts)
        ]

        store.add_vectors(embeddings, documents)

        # 検索実行
        query_embedding = np.random.random((1, 32)).astype(np.float32)
        results = store.search(query_embedding, k=2)

        assert len(results) == 2
        for result in results:
            assert isinstance(result['document'], Document)
            assert 'score' in result
            assert result['document'].metadata['type'] == 'test'

    def test_save_and_load_functionality(self):
        """保存・読み込み機能のテスト"""
        temp_dir = Path("temp_test_store")

        try:
            # 初期データ作成
            store1 = FAISSVectorStore(dimension=16, use_gpu=False, store_path=temp_dir)

            embeddings = np.random.random((2, 16)).astype(np.float32)
            documents = [
                Document(id="doc_1", content="first document", metadata={}),
                Document(id="doc_2", content="second document", metadata={})
            ]

            store1.add_vectors(embeddings, documents)
            store1.save()

            # 新しいインスタンスで読み込み
            store2 = FAISSVectorStore(dimension=16, use_gpu=False, store_path=temp_dir)
            store2.load()

            assert len(store2.documents) == 2
            assert store2.documents[0].content == "first document"
            assert store2.documents[1].content == "second document"

        finally:
            # クリーンアップ
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_gpu_detection_environment_scenarios(self):
        """環境設定シナリオのテスト"""
        # シナリオ1: 環境変数未設定（デフォルト）
        with patch.dict(os.environ, {}, clear=True):
            if 'BGE_FAISS_DEVICE' in os.environ:
                del os.environ['BGE_FAISS_DEVICE']

            retriever = SemanticRetriever()
            # デフォルトは自動検出
            setting = retriever._get_gpu_setting()
            # CUDAが利用可能かどうかに依存
            assert isinstance(setting, bool)

        # シナリオ2: 明示的なCPU設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()
            setting = retriever._get_gpu_setting()
            assert setting == False

    def test_index_operations(self):
        """インデックス操作のテスト"""
        store = FAISSVectorStore(dimension=8, use_gpu=False)

        # 初期状態
        assert len(store.documents) == 0

        # データ追加
        embeddings = np.random.random((3, 8)).astype(np.float32)
        documents = [
            Document(id=f"test_{i}", content=f"content {i}", metadata={})
            for i in range(3)
        ]

        store.add_vectors(embeddings, documents)
        assert len(store.documents) == 3

        # クリア操作
        store.clear()
        assert len(store.documents) == 0

    def test_batch_operations(self):
        """バッチ操作のテスト"""
        store = FAISSVectorStore(dimension=16, use_gpu=False)

        # 大量データ（相対的に）の処理
        batch_size = 10
        embeddings = np.random.random((batch_size, 16)).astype(np.float32)
        documents = [
            Document(id=f"batch_doc_{i}", content=f"batch content {i}", metadata={"batch": True})
            for i in range(batch_size)
        ]

        store.add_vectors(embeddings, documents)

        # バッチ検索
        query_embeddings = np.random.random((3, 16)).astype(np.float32)
        for i, query in enumerate(query_embeddings):
            results = store.search(query.reshape(1, -1), k=2)
            assert len(results) <= 2

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        store = FAISSVectorStore(dimension=32, use_gpu=False)

        # 次元数不一致のテスト
        with pytest.raises(Exception):
            wrong_embeddings = np.random.random((2, 16)).astype(np.float32)  # 16次元（32次元であるべき）
            documents = [
                Document(id="doc_1", content="test", metadata={}),
                Document(id="doc_2", content="test", metadata={})
            ]
            store.add_vectors(wrong_embeddings, documents)

        # 空の検索
        empty_store = FAISSVectorStore(dimension=32, use_gpu=False)
        query = np.random.random((1, 32)).astype(np.float32)
        results = empty_store.search(query, k=5)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])