# uvxでのインストール・実行方法

## 概要
`uvx`を使用することで、Python環境の構築なしに`bge-faiss-mcp`を直接実行できます。

## 前提条件
- [uv](https://docs.astral.sh/uv/)がインストール済みであること

## インストールと実行

### 1. GitHub経由での実行（推奨）
```bash
# 最新版を実行
uvx --from git+https://github.com/sakumura/bge-faiss-mcp bge-faiss-mcp

# 特定のブランチを指定
uvx --from git+https://github.com/sakumura/bge-faiss-mcp@feat/uvx-support bge-faiss-mcp
```

### 2. Claude Code設定
`.mcp.json`に以下の設定を追加：

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

### 3. 初回実行
- BGE-M3モデルの自動ダウンロード（約4GB）
- FAISSインデックスの自動構築
- 完了後、セマンティック検索が利用可能

## トラブルシューティング

### よくある問題

#### GPU関連の警告
```
WARNING:bge_faiss_mcp.core.embedder:GPU not available, using CPU (will be slower)
```
**解決方法**: CPUでも動作するため、警告は無視可能

#### モデルダウンロードエラー
**解決方法**: インターネット接続を確認し、再実行

#### メモリ不足エラー
**解決方法**:
- 他のアプリケーションを終了
- 最低8GBのRAMを推奨

### パフォーマンス最適化

#### CPU使用時
- 検索対象ファイル数を制限
- バッチサイズを小さく設定

#### メモリ使用量削減
- インデックスサイズが大きい場合は分割処理

## 関連情報
- [技術仕様](technical-spec.md)
- [GitHub リポジトリ](https://github.com/sakumura/bge-faiss-mcp)