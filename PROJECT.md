# BGE-FAISS MCP プロジェクト

## プロジェクト概要

**BGE-FAISS MCP** - BGE-M3埋め込みとFAISSベクトルストアを使用した高性能セマンティック検索MCPサーバー

### 特徴
- **BGE-M3埋め込み**: BAAI/bge-m3 多言語埋め込みモデル
- **FAISS高速検索**: Facebook AI類似度検索（CPU/GPU自動選択）
- **MCP統合**: Model Context Protocol 1.12.0+ 対応
- **ハイブリッド検索**: セマンティック/パターン検索の自動切り替え

### パッケージ情報
- **パッケージ名**: `bge-faiss-mcp`
- **現在バージョン**: 1.0.6
- **PyPI公開**: 準備中
- **GitHub**: https://github.com/sakumura/bge-faiss-mcp

## 技術アーキテクチャ

### ディレクトリ構造
```
bge-faiss-mcp/
├── src/bge_faiss_mcp/
│   ├── server.py           # MCPサーバーエントリーポイント
│   ├── core/
│   │   ├── manager.py      # ハイブリッド検索マネージャー
│   │   ├── embedder.py     # BGE-M3埋め込み処理
│   │   ├── vector_store.py # FAISSベクトルストレージ
│   │   ├── retriever.py    # 検索エンジン
│   │   └── rag.py          # RAGチェーン
│   └── utils/
│       ├── gpu.py          # GPU/CPU最適化・自動選択
│       ├── parser.py       # ファイルパーサー
│       └── analyzer.py     # クエリ解析
├── tests/                  # テストスイート（32テスト）
├── .claude/               # Claude Code設定・インデックス
└── .serena/memories/      # プロジェクト知識ベース
```

### 依存関係
- **Python**: 3.10～3.12（3.13は未対応）
- **PyTorch**: GPU/CPU自動選択（プラットフォーム別、uvx対応）
- **MCP**: 1.12.0+
- **FAISS**: faiss-cpu（全プラットフォーム）、オプションでfaiss-gpu

## 🚨 重要：uvx互換性問題と解決策

### 問題の概要
**uvx経由でのインストールが失敗する重大な問題が存在**

#### 根本原因
- PyTorchのCUDA版がPython 3.13に対応していない
- uvxのデフォルト環境がPython 3.13を使用
- 直接的なPyTorch依存関係指定により、バージョン解決が失敗

#### 失敗した対策（v1.0.4～v1.0.5）
1. **CUDA 12.1版への移行** → Python 3.13非対応で失敗
2. **tool.uv.indexでの指定** → 直接依存関係との競合で失敗
3. **faiss-gpu/cpu の条件分岐** → PyTorch問題は未解決

#### 最新の解決策（v1.0.6）
**プラットフォーム別条件付き依存関係管理**に変更：

```toml
# dependenciesからtorch/torchvisionを除外
dependencies = [
    # PyTorch関連を削除、tool.uv.sourcesで管理
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu118", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cu118", marker = "sys_platform == 'linux'" },
]
```

**効果**：
- uvxインストール時にプラットフォーム別の適切なPyTorchを自動選択
- Python 3.10～3.12での動作を保証（3.13は明示的に除外）
- GPU/CPU自動フォールバック機能を維持

## 現在の進行状況

### ✅ 完了済みタスク
- [x] 親プロジェクト（Claude Hub）からの独立
- [x] import文の修正完了（`semantic_rag` → `bge_faiss_mcp`）
- [x] パッケージ構造の整備
- [x] pyproject.toml作成（uv対応）
- [x] テストスイート作成（32テスト、100%パス）
- [x] MCP名称変更（`bge-faiss-search` → `local-search`）
- [x] uvx版でのGPU/CPU自動選択機能実装
- [x] clear_index のファイルロック問題修正
- [x] uvx互換性問題の解決（v1.0.6: 条件付き依存関係管理）

### 🚧 進行中タスク
- [ ] PyPIへの公開準備
- [ ] Claude Code設定例のドキュメント整備
- [ ] パフォーマンステストの追加

### 📋 今後の予定
- [ ] PyPIへの正式公開
- [ ] GitHub Actions CI/CD設定
- [ ] ベンチマーク結果の公開
- [ ] 多言語ドキュメント作成

## 開発ワークフロー

### 環境セットアップ
```bash
# リポジトリクローン
git clone https://github.com/sakumura/bge-faiss-mcp.git
cd bge-faiss-mcp

# 仮想環境作成
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 開発環境インストール
uv pip install -e ".[dev]"
```

### 品質チェック（コミット前に必ず実行）
```bash
# 型チェック
mypy src/bge_faiss_mcp/

# コードフォーマット
black src/ tests/ --target-version py310

# Linting
flake8 src/ tests/

# テスト実行
pytest --cov=bge_faiss_mcp --cov-report=term-missing
```

### MCP サーバー起動
```bash
# 直接起動
python -m bge_faiss_mcp.server

# uvx経由（GPU/CPU自動選択）
uvx bge-faiss-mcp
```

## 品質基準

### 必須要件
- ✅ 全テスト合格（32/32テスト）
- ✅ 型チェック通過（mypy）
- ✅ コードフォーマット（black）
- ✅ Lintチェック（flake8）
- ✅ テストカバレッジ 80%以上

### リリース要件
- [ ] PyPIパッケージビルド成功
- [ ] Claude Code統合テスト完了
- [ ] セキュリティチェック（bandit, safety）
- [ ] ドキュメント完備
- [ ] サンプルコード動作確認

## Claude Code 統合

### .mcp.json 設定例
```json
{
  "mcpServers": {
    "local-search": {
      "command": "uvx",
      "args": ["bge-faiss-mcp"],
      "env": {
        "BGE_FAISS_DEVICE": "auto"
      }
    }
  }
}
```

### 環境変数
- `BGE_FAISS_DEVICE`: GPU/CPU選択（`auto`, `cuda`, `cpu`）
- `BGE_FAISS_INDEX_PATH`: インデックス保存先
- `BGE_FAISS_BATCH_SIZE`: バッチサイズ（デフォルト: 32）

## トラブルシューティング

### GPU が使用されない場合
```bash
# GPU確認
nvidia-smi

# 環境変数設定
set BGE_FAISS_DEVICE=cuda  # Windows
export BGE_FAISS_DEVICE=cuda  # macOS/Linux
```

### インデックスエラー
```bash
# インデックス再構築
python -c "from bge_faiss_mcp import HybridSearchManager; m = HybridSearchManager(); m.build_index('.')"
```

## 貢献ガイドライン

1. **Issue作成**: バグ報告や機能提案
2. **Fork & Clone**: リポジトリをフォーク
3. **Branch作成**: `feature/` or `fix/` プレフィックス
4. **テスト追加**: 新機能には必ずテストを追加
5. **品質チェック**: 上記の品質チェックを実行
6. **Pull Request**: mainブランチへPR作成

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

## 連絡先

- **GitHub Issues**: https://github.com/sakumura/bge-faiss-mcp/issues
- **プロジェクト**: AI Dev Companion Project

---

**最終更新**: 2025-01-18
**作成元**: HANDOVER.mdから統合・更新