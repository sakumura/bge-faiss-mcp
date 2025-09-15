# 技術仕様

## アーキテクチャ概要

### コンポーネント構成
```
bge-faiss-mcp/
├── core/
│   ├── embedder.py       # BGE-M3埋め込み
│   ├── vector_store.py   # FAISSベクトルストア
│   ├── retriever.py      # セマンティック検索
│   └── manager.py        # ハイブリッド検索管理
├── utils/
│   ├── gpu.py           # GPU最適化
│   ├── parser.py        # ドキュメントパーサー
│   └── analyzer.py      # クエリ解析
└── server.py            # MCPサーバー
```

## BGE-M3埋め込み

### モデル仕様
- **モデル名**: `BAAI/bge-m3`
- **次元数**: 1024次元
- **対応言語**: 多言語（日本語含む）
- **最大トークン長**: 8192トークン

### 埋め込み処理
```python
# 文書の埋め込み生成
embeddings = model.encode(documents, batch_size=32)
```

## FAISS ベクトルストア

### インデックス種類
- **デフォルト**: IVFFlat（逆ファイルインデックス）
- **GPU対応**: GPU利用時はGPUインデックス自動選択
- **精度**: 高精度検索対応

### インデックス構成
```python
# インデックス初期化
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(training_vectors)
```

## 検索モード

### 1. AUTO（自動選択）
- クエリ内容に基づき最適なモードを自動選択
- 短いクエリ: セマンティック検索
- 具体的なファイル名等: パターン検索

### 2. SEMANTIC（セマンティック検索）
- BGE-M3埋め込みを使用
- 意味的類似性に基づく検索
- Top-K結果を返却

### 3. PATTERN（パターン検索）
- 正規表現ベースの検索
- ファイル名やパス検索に最適
- 高速処理

## パフォーマンス指標

### 処理速度
- **CPU**: 約100文書/秒（埋め込み生成）
- **GPU**: 約500文書/秒（CUDA対応時）
- **検索**: <100ms（1000文書インデックス）

### メモリ使用量
- **モデル**: 約4GB（BGE-M3）
- **インデックス**: 約1MB/1000文書
- **最小推奨RAM**: 8GB

## ファイル対応形式

### サポート対象
- Python (.py)
- Markdown (.md)
- Text (.txt)
- JSON (.json)
- YAML (.yml, .yaml)

### 除外対象
- バイナリファイル
- 大容量ファイル（>10MB）
- .gitignore指定ファイル

## 設定項目

### 主要設定
```python
config = {
    "semantic_index_path": ".search/vectors",
    "default_mode": "auto",
    "enable_cache": True,
    "batch_size": 32,
    "max_file_size": 10 * 1024 * 1024  # 10MB
}
```

### 環境変数
- `BGE_FAISS_GPU`: GPU使用強制（1で有効）
- `BGE_FAISS_BATCH_SIZE`: バッチサイズ設定
- `BGE_FAISS_LOG_LEVEL`: ログレベル設定

## API仕様

### MCP Tools

#### search
```json
{
  "name": "search",
  "arguments": {
    "query": "検索クエリ",
    "k": 5,
    "mode": "auto"
  }
}
```

#### build_index
```json
{
  "name": "build_index",
  "arguments": {
    "project_path": "."
  }
}
```

#### get_stats
```json
{
  "name": "get_stats",
  "arguments": {}
}
```

## セキュリティ考慮事項

### データ保護
- ローカル処理のみ（外部送信なし）
- インデックスはプロジェクト内に保存
- 機密ファイルの自動除外

### プライバシー
- 個人情報の埋め込みを避ける設計
- ログにファイル内容を出力しない

## 拡張性

### プラグイン対応
- カスタム埋め込みモデル追加可能
- ファイルパーサーの拡張対応
- 検索モードの追加可能