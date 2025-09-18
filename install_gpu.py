#!/usr/bin/env python
"""
GPU-enabled PyTorch installation script for uvx.
This ensures the correct PyTorch version with CUDA support is installed.
"""
import sys
import subprocess
import platform

def install_gpu_pytorch():
    """Install GPU-enabled PyTorch."""
    system = platform.system().lower()

    if system == "windows":
        # Windows用CUDA 12.1版PyTorch
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--upgrade"
        ])
    elif system == "linux":
        # Linux用CUDA 12.1版PyTorch
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--upgrade"
        ])
    else:
        # macOSはCPU版のみ
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--upgrade"
        ])

    print("PyTorch installation completed.")

    # 確認
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

if __name__ == "__main__":
    install_gpu_pytorch()