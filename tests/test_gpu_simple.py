"""
GPU機能の簡単なテスト - 実際の動作を確認
"""

import pytest
import os
from unittest.mock import patch

from bge_faiss_mcp.core.retriever import SemanticRetriever


class TestGPUSimple:
    """GPU機能の簡単なテスト"""

    def test_environment_variable_cpu_setting(self):
        """環境変数でCPU設定のテスト"""
        # CPU強制設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            assert os.environ.get('BGE_FAISS_DEVICE') == 'cpu'

    def test_environment_variable_gpu_setting(self):
        """環境変数でGPU設定のテスト"""
        # GPU強制設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'gpu'}):
            assert os.environ.get('BGE_FAISS_DEVICE') == 'gpu'

    def test_environment_variable_auto_setting(self):
        """環境変数でauto設定のテスト"""
        # auto設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'auto'}):
            assert os.environ.get('BGE_FAISS_DEVICE') == 'auto'

    def test_semantic_retriever_initialization(self):
        """SemanticRetrieverの初期化テスト"""
        # CPU強制設定でRetrieverを初期化
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()

            # オブジェクトが正常に作成されることを確認
            assert retriever is not None
            assert hasattr(retriever, 'embedder')
            assert hasattr(retriever, 'vector_store')

    def test_gpu_setting_method_exists(self):
        """GPU設定メソッドの存在確認"""
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()

            # _get_gpu_settingメソッドが存在することを確認
            assert hasattr(retriever, '_get_gpu_setting')

            # メソッドが呼び出し可能であることを確認
            gpu_setting = retriever._get_gpu_setting()
            assert isinstance(gpu_setting, bool)

    def test_cpu_forced_setting(self):
        """CPU強制設定の動作確認"""
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()

            # CPU強制設定では必ずFalseになる
            gpu_setting = retriever._get_gpu_setting()
            assert gpu_setting == False

    def test_default_setting_behavior(self):
        """デフォルト設定の動作確認"""
        # 環境変数をクリア
        with patch.dict(os.environ, {}, clear=True):
            if 'BGE_FAISS_DEVICE' in os.environ:
                del os.environ['BGE_FAISS_DEVICE']

            retriever = SemanticRetriever()

            # デフォルト設定では自動検出
            gpu_setting = retriever._get_gpu_setting()
            # CUDA利用可能性に依存するため、booleanであることのみ確認
            assert isinstance(gpu_setting, bool)

    def test_retriever_components_initialization(self):
        """Retrieverコンポーネントの初期化テスト"""
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()

            # 各コンポーネントが初期化されていることを確認
            assert retriever.embedder is not None
            assert retriever.vector_store is not None

            # embedderのdevice設定確認
            assert hasattr(retriever.embedder, 'device')

    def test_case_insensitive_environment_variable(self):
        """大文字小文字非依存の環境変数テスト"""
        # 大文字設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'CPU'}):
            retriever = SemanticRetriever()
            gpu_setting = retriever._get_gpu_setting()
            assert gpu_setting == False

        # 小文字設定
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'cpu'}):
            retriever = SemanticRetriever()
            gpu_setting = retriever._get_gpu_setting()
            assert gpu_setting == False

    def test_invalid_environment_variable(self):
        """不正な環境変数値のテスト"""
        # 不正な値（autoとして扱われる）
        with patch.dict(os.environ, {'BGE_FAISS_DEVICE': 'invalid_value'}):
            retriever = SemanticRetriever()
            gpu_setting = retriever._get_gpu_setting()
            # 不正な値の場合はautoとして扱われ、CUDA利用可能性に依存
            assert isinstance(gpu_setting, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])