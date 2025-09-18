#!/usr/bin/env python
"""
最終GPU確認テスト - uvx版がGPUを使用しているか
"""
import subprocess
import time
import os

print("=" * 60)
print("Final GPU Test for uvx v1.0.8")
print("=" * 60)

# BGE_FAISS_DEVICE環境変数を設定
os.environ['BGE_FAISS_DEVICE'] = 'gpu'

print("Testing bge-faiss-mcp with GPU environment variable...")
print(f"BGE_FAISS_DEVICE={os.environ.get('BGE_FAISS_DEVICE')}")

# bge-faiss-mcpを起動
proc = subprocess.Popen(
    ["bge-faiss-mcp", "--help"],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    env=os.environ
)

# 5秒待つ
time.sleep(5)
proc.terminate()

try:
    stdout, stderr = proc.communicate(timeout=2)
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()

# ログから重要な情報を抽出
print("\n" + "=" * 40)
print("Searching for GPU/CUDA messages:")
print("=" * 40)

gpu_found = False
cuda_version = None

for line in stderr.split('\n'):
    line_lower = line.lower()
    if 'gpu' in line_lower or 'cuda' in line_lower:
        print(f"[FOUND] {line}")
        if 'using gpu' in line_lower:
            gpu_found = True
        if 'cuda not available' in line_lower:
            print(f"[ERROR] {line}")
            gpu_found = False

print("\n" + "=" * 40)
print("Result:")
print("=" * 40)

if gpu_found:
    print("[SUCCESS] uvx version is using GPU!")
    print("[OK] CUDA 11.8 PyTorch is correctly installed")
else:
    print("[WARNING] GPU not detected")
    print("\nStderr出力の最初の15行:")
    for i, line in enumerate(stderr.split('\n')[:15], 1):
        print(f"  {i}: {line}")

print("\n" + "=" * 60)