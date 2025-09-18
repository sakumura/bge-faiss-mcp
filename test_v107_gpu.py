#!/usr/bin/env python
"""
v1.0.7でGPUが使用されるかテスト
"""
import subprocess
import sys
import time

print("Testing v1.0.7 GPU support")
print("=" * 60)

# 1. uvツールの場所を確認
result = subprocess.run(["where", "bge-faiss-mcp"], capture_output=True, text=True)
print(f"bge-faiss-mcp location: {result.stdout.strip()}")

# 2. uvツール版をテスト
print("\nTesting uv tool version with GPU...")
env = {"BGE_FAISS_DEVICE": "gpu"}
proc = subprocess.Popen(
    ["bge-faiss-mcp", "--help"],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    env={**subprocess.os.environ, **env}
)

# 5秒待って出力を収集
time.sleep(5)
proc.terminate()

try:
    stdout, stderr = proc.communicate(timeout=2)
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, stderr = proc.communicate()

# GPU関連のメッセージを探す
print("\nSearching for GPU-related messages...")
print("-" * 40)

found_gpu = False
for line in stderr.split('\n'):
    line_lower = line.lower()
    if 'gpu' in line_lower or 'cuda' in line_lower:
        print(f"  {line}")
        if 'using gpu' in line_lower or 'gpu:' in line_lower:
            found_gpu = True

print("-" * 40)

if found_gpu:
    print("\n[SUCCESS] v1.0.7 is using GPU!")
else:
    print("\n[WARNING] No clear GPU usage in v1.0.7")
    print("\nShowing first 10 error lines for debugging:")
    for i, line in enumerate(stderr.split('\n')[:10]):
        print(f"  {i+1}: {line}")

print("\n" + "=" * 60)
print("Test complete")