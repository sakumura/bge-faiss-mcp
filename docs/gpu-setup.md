# GPU設定ガイド

BGE-FAISS MCPのGPU設定について詳しく説明します。

## 概要

BGE-FAISS MCPは以下の2つのGPU加速をサポートしています：

1. **BGE-M3埋め込み生成**: PyTorchによるGPU加速
2. **FAISS検索**: FAISS GPUライブラリによる高速化

## 自動検出機能

デフォルトでは、システムのCUDA利用可能性を自動検出し、適切なデバイスを選択します：

```python
# 自動検出ロジック
if torch.cuda.is_available() and faiss_gpu_available:
    use_gpu = True  # GPU使用
else:
    use_gpu = False  # CPUフォールバック
```

## 環境変数による制御

`BGE_FAISS_DEVICE`環境変数で動作を制御できます：

### 設定値

- `auto` (デフォルト): 自動検出
- `gpu`: GPU強制使用
- `cpu`: CPU強制使用

### 使用例

```bash
# GPU強制使用
export BGE_FAISS_DEVICE=gpu
uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp

# CPU強制使用
export BGE_FAISS_DEVICE=cpu
uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp

# Claude Code MCP設定での使用
{
  "mcpServers": {
    "bge-faiss-search": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sakumura/bge-faiss-mcp", "bge-faiss-mcp"],
      "env": {
        "BGE_FAISS_DEVICE": "gpu"
      }
    }
  }
}
```

## GPU環境のセットアップ

### 1. CUDA環境の確認

```bash
# CUDA利用可能性を確認
nvidia-smi

# PyTorchからCUDA確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. PyTorch GPU版のインストール

```bash
# CUDA 11.8の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. FAISS GPU版のインストール

```bash
# FAISS GPU版
pip install faiss-gpu

# または一括インストール
pip install git+https://github.com/sakumura/bge-faiss-mcp.git[gpu]
```

## パフォーマンス比較

### RTX 4060 Ti (8GB)での測定例

| タスク | CPU時間 | GPU時間 | 高速化倍率 |
|--------|---------|---------|------------|
| 埋め込み生成 (100文書) | 15.2秒 | 1.8秒 | 8.4倍 |
| FAISS検索 (10,000文書) | 45ms | 5ms | 9倍 |
| インデックス構築 (1,000文書) | 180秒 | 12秒 | 15倍 |

### メモリ使用量

| 設定 | VRAM使用量 | システムRAM |
|------|------------|-------------|
| BGE-M3のみ | 2.1GB | 1.5GB |
| BGE-M3 + FAISS(1K文書) | 2.3GB | 0.8GB |
| BGE-M3 + FAISS(10K文書) | 2.8GB | 0.8GB |

## トラブルシューティング

### 1. CUDA not available

**症状**: `CUDA not available, using CPU`のメッセージ

**解決策**:
```bash
# CUDA環境の確認
nvidia-smi
nvcc --version

# PyTorchの再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. FAISS GPU not available

**症状**: `FAISS GPU build not available, using CPU index`

**解決策**:
```bash
# FAISS GPU版のインストール
pip uninstall faiss-cpu faiss
pip install faiss-gpu
```

### 3. Out of Memory エラー

**症状**: `CUDA out of memory`

**解決策**:
```bash
# バッチサイズを小さくして実行
export BGE_FAISS_BATCH_SIZE=8
# または CPU使用に切り替え
export BGE_FAISS_DEVICE=cpu
```

### 4. Mixed CUDA versions

**症状**: CUDAバージョンの不整合エラー

**解決策**:
```bash
# 環境を完全にクリア
pip uninstall torch torchvision faiss-gpu faiss-cpu
pip cache purge

# 統一バージョンで再インストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

## 最適化のヒント

### 1. バッチサイズの調整

GPU使用時はバッチサイズを大きくすることで処理効率が向上します：

- RTX 4060 Ti: バッチサイズ 32-64
- RTX 3060: バッチサイズ 16-32
- RTX 2060: バッチサイズ 8-16

### 2. データサイズに応じた設定

- **小規模** (< 1,000文書): GPU効果は限定的、CPUでも十分
- **中規模** (1,000-10,000文書): GPU推奨、大幅な高速化
- **大規模** (> 10,000文書): GPU必須、メモリ使用量に注意

### 3. 混合精度の活用

FP16演算による高速化（自動有効化）：

```python
# BGE-M3では自動的にFP16が使用される（GPU使用時）
use_fp16=True if device == "cuda" else False
```

## システム要件

### 最小要件（GPU使用時）
- NVIDIA GPU (Compute Capability 3.5以上)
- CUDA 11.0以上
- 4GB VRAM
- 8GB システムRAM

### 推奨要件
- NVIDIA RTX 3060以上
- CUDA 11.8または12.1
- 8GB VRAM以上
- 16GB システムRAM

## 参考資料

- [PyTorch CUDA インストール](https://pytorch.org/get-started/locally/)
- [FAISS GPU ドキュメント](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [NVIDIA CUDA インストール](https://developer.nvidia.com/cuda-downloads)