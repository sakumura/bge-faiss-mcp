#!/usr/bin/env python
"""
uv tool環境のPyTorchバージョンを確認
"""
import subprocess
import sys

print("Checking PyTorch version in uv tool environment...")
print("=" * 60)

# uv toolの環境でPythonスクリプトを実行
script = """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.version, 'cuda'):
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('CUDA version: None (CPU-only build)')
"""

# uv tool環境の場所を探す
result = subprocess.run(
    ["python", "-c", "import site; print(site.getsitepackages())"],
    capture_output=True,
    text=True
)
print(f"Site packages: {result.stdout}")

# uv tool runで実行
print("\nTrying to check PyTorch in uv tool environment...")
result = subprocess.run(
    ["uv", "tool", "run", "bge-faiss-mcp", "--version"],
    capture_output=True,
    text=True
)
print(f"bge-faiss-mcp version: {result.stdout}")

# 直接Python環境を確認
print("\nChecking installed packages...")
result = subprocess.run(
    ["uv", "tool", "list"],
    capture_output=True,
    text=True
)
print(f"Installed tools:\n{result.stdout}")

# uv toolの環境パスを取得
import os
uv_tools_path = os.path.join(os.environ['USERPROFILE'], '.local', 'share', 'uv', 'tools')
print(f"\nuv tools path: {uv_tools_path}")

# bge-faiss-mcpのPython実行可能ファイルを探す
bge_path = os.path.join(uv_tools_path, 'bge-faiss-mcp')
if os.path.exists(bge_path):
    print(f"bge-faiss-mcp tool directory exists")
    # Scriptsフォルダを探す
    for root, dirs, files in os.walk(bge_path):
        if 'python.exe' in files:
            python_exe = os.path.join(root, 'python.exe')
            print(f"\nFound Python: {python_exe}")
            # PyTorchバージョンを確認
            result = subprocess.run(
                [python_exe, "-c", script],
                capture_output=True,
                text=True
            )
            print(f"\nPyTorch in uv tool environment:")
            print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}")
            break