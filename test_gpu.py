import torch
import sys
import os

# GPU状態を確認
print("=" * 50)
print("PyTorch GPU Status")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU not available")

print("\n" + "=" * 50)
print("Testing bge-faiss-mcp GPU usage")
print("=" * 50)

# BGE_FAISS_DEVICEを設定
os.environ['BGE_FAISS_DEVICE'] = 'gpu'

# bge-faiss-mcpのモジュールをインポート
try:
    from bge_faiss_mcp.core.embedder import BGE_M3Embedder
    print("[OK] BGE_M3Embedder imported successfully")

    # Embedderを初期化
    print("\nInitializing BGE_M3Embedder...")
    embedder = BGE_M3Embedder()

    # デバイスを確認
    print(f"Embedder device: {embedder.device}")
    print(f"Model device: {next(embedder.model.parameters()).device}")

    # テスト埋め込みを実行
    test_text = "This is a test for GPU usage"
    print(f"\nTesting embedding generation for: '{test_text}'")

    embedding = embedder.encode([test_text])[0]
    print(f"[OK] Embedding generated successfully")
    print(f"Embedding shape: {embedding.shape}")

    # GPU使用状況を確認
    if torch.cuda.is_available():
        print(f"\nGPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()