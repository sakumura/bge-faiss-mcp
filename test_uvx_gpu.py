#!/usr/bin/env python
"""
uvx版 bge-faiss-mcp のGPU使用状況を確認するテストスクリプト
"""
import os
import sys
import subprocess
import time
import json
import torch

def check_gpu_status():
    """現在のGPU状態を確認"""
    print("=" * 60)
    print("System GPU Status Check")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # GPUメモリ使用状況
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")
    else:
        print("GPU not available")

    print()

def test_uvx_gpu():
    """uvx版でGPUが使用されているか確認"""
    print("=" * 60)
    print("Testing uvx version GPU usage")
    print("=" * 60)

    # BGE_FAISS_DEVICE環境変数を設定
    env = os.environ.copy()
    env['BGE_FAISS_DEVICE'] = 'gpu'

    # uvx版のbge-faiss-mcpを短時間実行してログを取得
    cmd = [
        'uvx', '--from', 'git+https://github.com/sakumura/bge-faiss-mcp',
        'bge-faiss-mcp', '--help'
    ]

    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment: BGE_FAISS_DEVICE=gpu")
    print()

    try:
        # プロセスを開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 5秒待って出力を収集
        time.sleep(5)

        # プロセスを終了
        process.terminate()

        # 出力を取得
        stdout, stderr = process.communicate(timeout=2)

        # GPU関連のログを探す
        gpu_found = False
        cuda_found = False

        all_output = stdout + stderr
        lines = all_output.split('\n')

        print("GPU-related log messages:")
        print("-" * 40)

        for line in lines:
            if any(keyword in line.lower() for keyword in ['gpu', 'cuda', 'device', 'vram']):
                print(f"  {line}")
                if 'gpu' in line.lower() and ('using' in line.lower() or 'available' in line.lower()):
                    gpu_found = True
                if 'cuda' in line.lower():
                    cuda_found = True

        print("-" * 40)

        # 結果判定
        if gpu_found or cuda_found:
            print("\n[OK] GPU-related messages found in uvx version")
        else:
            print("\n[WARNING] No clear GPU usage indication in uvx version")
            print("\nFull stderr output (first 20 lines):")
            for i, line in enumerate(stderr.split('\n')[:20]):
                print(f"  {i+1}: {line}")

        # GPUメモリ使用状況を再確認
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\nGPU memory after test: {allocated_after:.2f} GB")

    except subprocess.TimeoutExpired:
        print("[INFO] Process timed out (expected for MCP server)")
    except Exception as e:
        print(f"[ERROR] {e}")

def test_local_gpu():
    """ローカル開発版でGPUが使用されているか確認"""
    print("\n" + "=" * 60)
    print("Testing local development version GPU usage")
    print("=" * 60)

    # 環境変数を設定
    os.environ['BGE_FAISS_DEVICE'] = 'gpu'

    try:
        # ローカルモジュールをインポート
        from bge_faiss_mcp.core.embedder import BGE_M3Embedder

        print("[OK] BGE_M3Embedder imported")
        print("\nInitializing embedder...")

        # GPUメモリ使用前
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU memory before: {mem_before:.2f} GB")

        # Embedderを初期化
        embedder = BGE_M3Embedder()

        # デバイスを確認
        print(f"Embedder device: {embedder.device}")

        # モデルのデバイスを確認
        model_device = next(embedder.model.parameters()).device
        print(f"Model device: {model_device}")

        # テスト埋め込みを実行
        test_text = "Testing GPU usage with uvx version"
        print(f"\nGenerating embedding for: '{test_text}'")

        embedding = embedder.encode([test_text])[0]
        print(f"[OK] Embedding generated, shape: {embedding.shape}")

        # GPUメモリ使用後
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU memory after: {mem_after:.2f} GB")
            print(f"Memory increase: {(mem_after - mem_before):.2f} GB")

            if model_device.type == 'cuda':
                print("\n[SUCCESS] Local version is using GPU!")
            else:
                print("\n[WARNING] Local version is NOT using GPU")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン実行"""
    print("\n" + "=" * 60)
    print("BGE-FAISS-MCP GPU Usage Test")
    print("=" * 60)

    # システムGPU状態を確認
    check_gpu_status()

    # uvx版をテスト
    test_uvx_gpu()

    # ローカル版をテスト（比較用）
    test_local_gpu()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    main()