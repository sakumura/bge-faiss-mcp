# GPU Support for BGE-FAISS MCP

## Problem
When installing `bge-faiss-mcp` via `uvx`, it may install CPU-only PyTorch even if you have a GPU available.

## Solution

### Option 1: Set environment variables in .mcp.json (Recommended)
```json
{
  "mcpServers": {
    "local-search": {
      "command": "uvx",
      "args": ["bge-faiss-mcp"],
      "env": {
        "BGE_FAISS_DEVICE": "cuda",
        "PIP_INDEX_URL": "https://download.pytorch.org/whl/cu121"
      }
    }
  }
}
```

### Option 2: Use pre-installed environment
Instead of using `uvx`, create a conda/venv environment with GPU PyTorch pre-installed:

```bash
# Create conda environment
conda create -n bge-faiss-mcp python=3.10
conda activate bge-faiss-mcp

# Install GPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install bge-faiss-mcp
pip install bge-faiss-mcp
```

Then in .mcp.json:
```json
{
  "mcpServers": {
    "local-search": {
      "command": "python",
      "args": ["-m", "bge_faiss_mcp.server"],
      "env": {
        "BGE_FAISS_DEVICE": "cuda"
      }
    }
  }
}
```

### Option 3: Manual GPU PyTorch installation
After installing with uvx, manually upgrade PyTorch:

```bash
# Find uvx installation
uvx --python-preference managed run --with bge-faiss-mcp python -c "import sys; print(sys.executable)"

# Use that Python to install GPU PyTorch
/path/to/uvx/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

## Verification
To check if GPU is being used:
```bash
# Check PyTorch version
uvx --python-preference managed run --with bge-faiss-mcp python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Notes
- The package will automatically detect and use GPU if available
- Falls back to CPU if GPU is not available or if there's an error
- Set `BGE_FAISS_DEVICE=cpu` to force CPU mode