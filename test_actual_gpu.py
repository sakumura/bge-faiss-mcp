#!/usr/bin/env python
"""
実際にモデルをロードしてGPU使用を確認
"""
import os
import torch

print("=" * 60)
print("Direct GPU Test")
print("=" * 60)

# 環境変数を設定
os.environ['BGE_FAISS_DEVICE'] = 'gpu'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # テストテンソルをGPUに送る
    test_tensor = torch.randn(100, 100).cuda()
    print(f"Test tensor on GPU: {test_tensor.device}")

    # メモリ使用量
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
else:
    print("GPU not available - this is the problem!")

print("\nTrying to import bge_faiss_mcp...")
try:
    from bge_faiss_mcp.core.embedder import BGE_M3Embedder
    print("Import successful")

    print("\nInitializing embedder...")
    embedder = BGE_M3Embedder()

    print(f"Embedder device: {embedder.device}")
    print(f"Model on GPU: {next(embedder.model.parameters()).device}")

    # テスト埋め込み
    test = embedder.encode(["test"])
    print(f"Embedding generated: shape {test[0].shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()