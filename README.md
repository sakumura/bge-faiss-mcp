# BGE-FAISS MCP

🚀 **BGE-M3埋め込みとFAISSベクトルストアを使用した高性能セマンティック検索サーバー**

最先端のBGE-M3埋め込みとFAISSによる高速検索を提供するModel Context Protocol (MCP)サーバーです。

## ✨ 特徴

- **🎯 BGE-M3埋め込み**: 優れた多言語セマンティック理解
- **⚡ FAISS高速検索**: GPU自動検出・CPU自動フォールバック
- **🔍 ハイブリッド検索**: セマンティック/パターン検索の自動切り替え
- **🔧 MCP統合**: Claude Codeとシームレスに連携
- **🚀 GPU高速化**: RTX 4060 Ti等で10-100倍の高速化

## 📦 インストール

### uvx実行（推奨）
Python環境の構築不要で直接実行できます：

```bash
# GitHub経由で直接実行
uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp
```

### 従来方式（pip）
```bash
# GitHubから直接インストール
pip install git+https://github.com/sakumura/bge-faiss-mcp.git

# または uvを使用
uv pip install git+https://github.com/sakumura/bge-faiss-mcp.git
```

> **詳細**: [uvxインストール方法](docs/uvx-installation.md)を参照

## 🚀 クイックスタート

### Claude Code設定

`.mcp.json`に以下を追加:

#### uvx使用時（推奨）
```json
{
  "mcpServers": {
    "bge-faiss-search": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/sakumura/bge-faiss-mcp",
        "bge-faiss-mcp"
      ]
    }
  }
}
```

#### pip使用時
```json
{
  "mcpServers": {
    "bge-faiss-search": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "bge_faiss_mcp.server"],
      "cwd": "."
    }
  }
}
```

サーバーが自動的に:
- プロジェクトのセマンティックインデックスを構築
- 高速セマンティック検索を提供
- ファイル変更時にインデックスを自動更新

## 🔧 環境変数

- `WORKING_DIR`: インデックス対象ディレクトリ（デフォルト: 現在のディレクトリ）  
- `DEFAULT_MODE`: 検索モード - `auto`、`semantic`、`pattern`（デフォルト: `auto`）

## 📖 使用可能な機能

### 検索
```
search(query, k=5, mode="auto")
```
自然言語でコードやドキュメントを検索

### インデックス管理
- `build_index()` - インデックスの構築/再構築
- `clear_index()` - インデックスのクリア
- `get_stats()` - 統計情報の取得

## 📋 必要環境

### uvx使用時
- [uv](https://docs.astral.sh/uv/)がインストール済み
- 4GB以上のRAM（推奨: 8GB以上）

### pip使用時
- Python 3.10以上
- 4GB以上のRAM（推奨: 8GB以上）

## 🎮 GPU設定

BGE-FAISS MCPは自動的にGPUを検出・利用しますが、手動制御も可能です。

### 自動GPU検出（デフォルト）
```bash
# CUDA利用可能な場合はGPU、不可の場合はCPU
uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp
```

### 手動制御
```bash
# GPU強制使用
BGE_FAISS_DEVICE=gpu uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp

# CPU強制使用
BGE_FAISS_DEVICE=cpu uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp
```

### GPU版のインストール（pip使用時）
GPU高速化を利用する場合:
```bash
# CUDA 11.8の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu

# または
pip install git+https://github.com/sakumura/bge-faiss-mcp.git[gpu]
```

## 📚 ドキュメント

- [uvxインストール方法](docs/uvx-installation.md)
- [技術仕様](docs/technical-spec.md)
- [GPU設定詳細](docs/gpu-setup.md)

## 📄 ライセンス

MIT License
