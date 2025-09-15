"""
QueryAnalyzer - クエリ分析モジュール

クエリの複雑度を分析し、最適な検索エンジンを推薦する。
セマンティック理解が必要なクエリ、シンプルなキーワード検索、
正規表現パターンなどを識別する。
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """クエリ分析結果"""

    query: str
    word_count: int
    has_question_words: bool
    has_technical_terms: bool
    has_regex_pattern: bool
    is_simple_keyword: bool
    needs_semantic_understanding: bool
    confidence: float
    recommended_mode: str


class QueryAnalyzer:
    """
    クエリを分析して最適な検索モードを推薦
    """

    # 質問を示す単語
    QUESTION_WORDS = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "explain",
        "describe",
        "understand",
        "implement",
        "なぜ",
        "どのように",
        "いつ",
        "どこ",
        "誰が",
        "何を",
    }

    # 技術用語のパターン
    TECHNICAL_PATTERNS = [
        r"\b(function|class|method|variable|module|package)\b",
        r"\b(async|await|promise|callback|closure)\b",
        r"\b(api|rest|graphql|grpc|websocket)\b",
        r"\b(docker|kubernetes|container|pod|service)\b",
        r"\b(git|commit|branch|merge|rebase)\b",
        r"\b(database|query|index|transaction|schema)\b",
        r"\b(machine learning|deep learning|neural network|transformer)\b",
    ]

    # 正規表現パターンの検出
    REGEX_INDICATORS = [
        r"[\.\*\+\?\[\]\{\}\(\)\|\\]",  # 正規表現メタ文字
        r"\b(regex|pattern|match|search)\b",  # 正規表現関連キーワード
    ]

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: カスタム設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        logger.info("QueryAnalyzer initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイルをロード"""
        # デフォルト設定
        config = {
            "semantic_threshold": 0.6,
            "keyword_threshold": 0.3,
            "min_words_for_semantic": 3,
            "max_words_for_keyword": 2,
        }

        # カスタム設定があれば上書き
        if config_path and Path(config_path).exists():
            try:
                import json

                with open(config_path, "r", encoding="utf-8") as f:
                    custom_config = json.load(f)
                    config.update(custom_config)
                logger.info(f"Custom config loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")

        return config

    def analyze(self, query: str) -> QueryAnalysis:
        """
        クエリを分析して検索モードを推薦

        Args:
            query: 分析対象のクエリ

        Returns:
            クエリ分析結果
        """
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)

        # 各種チェック
        has_question_words = self._has_question_words(query_lower)
        has_technical_terms = self._has_technical_terms(query_lower)
        has_regex_pattern = self._has_regex_pattern(query)
        is_simple_keyword = self._is_simple_keyword(query, word_count)

        # セマンティック理解の必要性を判定
        needs_semantic = self._needs_semantic_understanding(
            word_count, has_question_words, has_technical_terms, query_lower
        )

        # 信頼度の計算
        confidence = self._calculate_confidence(
            word_count, has_question_words, has_technical_terms, has_regex_pattern
        )

        # 推奨モードの決定
        recommended_mode = self._determine_recommended_mode(
            needs_semantic, is_simple_keyword, has_regex_pattern
        )

        analysis = QueryAnalysis(
            query=query,
            word_count=word_count,
            has_question_words=has_question_words,
            has_technical_terms=has_technical_terms,
            has_regex_pattern=has_regex_pattern,
            is_simple_keyword=is_simple_keyword,
            needs_semantic_understanding=needs_semantic,
            confidence=confidence,
            recommended_mode=recommended_mode,
        )

        logger.debug(f"Query analysis: {analysis}")
        return analysis

    def _has_question_words(self, query_lower: str) -> bool:
        """質問語を含むかチェック"""
        return any(word in query_lower for word in self.QUESTION_WORDS)

    def _has_technical_terms(self, query_lower: str) -> bool:
        """技術用語を含むかチェック"""
        for pattern in self.TECHNICAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        return False

    def _has_regex_pattern(self, query: str) -> bool:
        """正規表現パターンを含むかチェック"""
        for pattern in self.REGEX_INDICATORS:
            if re.search(pattern, query):
                return True
        return False

    def _is_simple_keyword(self, query: str, word_count: int) -> bool:
        """シンプルなキーワード検索かチェック"""
        # 単語数が少ない
        if word_count > self.config["max_words_for_keyword"]:
            return False

        # 特殊文字が少ない
        special_chars = re.findall(r"[^a-zA-Z0-9\s\-_]", query)
        if len(special_chars) > 2:
            return False

        return True

    def _needs_semantic_understanding(
        self,
        word_count: int,
        has_question_words: bool,
        has_technical_terms: bool,
        query_lower: str,
    ) -> bool:
        """セマンティック理解が必要かどうか判定"""

        # 質問形式の場合
        if has_question_words:
            return True

        # 長いクエリの場合
        if word_count >= self.config["min_words_for_semantic"]:
            return True

        # 複雑な技術的質問
        if has_technical_terms and word_count > 2:
            return True

        # 自然言語的な表現を含む場合
        natural_language_indicators = [
            "implementation",
            "example",
            "usage",
            "purpose",
            "difference between",
            "similar to",
            "related to",
            "実装",
            "例",
            "使い方",
            "目的",
            "違い",
            "類似",
        ]

        if any(indicator in query_lower for indicator in natural_language_indicators):
            return True

        return False

    def _calculate_confidence(
        self,
        word_count: int,
        has_question_words: bool,
        has_technical_terms: bool,
        has_regex_pattern: bool,
    ) -> float:
        """分析の信頼度を計算（0.0〜1.0）"""
        confidence = 0.5  # ベースライン

        # 明確な指標がある場合は信頼度を上げる
        if has_question_words:
            confidence += 0.2

        if has_technical_terms:
            confidence += 0.1

        if has_regex_pattern:
            confidence += 0.2

        # 単語数による調整
        if word_count == 1:
            confidence += 0.1  # 単一キーワードは明確
        elif word_count >= 5:
            confidence += 0.1  # 長いクエリも意図が明確

        return min(confidence, 1.0)

    def _determine_recommended_mode(
        self, needs_semantic: bool, is_simple_keyword: bool, has_regex_pattern: bool
    ) -> str:
        """推奨検索モードを決定"""
        if has_regex_pattern:
            return "pattern"
        elif needs_semantic:
            return "semantic"
        elif is_simple_keyword:
            return "lightweight"
        else:
            return "auto"

    def batch_analyze(self, queries: List[str]) -> List[QueryAnalysis]:
        """
        複数のクエリを一括分析

        Args:
            queries: クエリのリスト

        Returns:
            分析結果のリスト
        """
        results = []
        for query in queries:
            try:
                analysis = self.analyze(query)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze query '{query}': {e}")
                # エラー時はデフォルト値を返す
                results.append(
                    QueryAnalysis(
                        query=query,
                        word_count=len(query.split()),
                        has_question_words=False,
                        has_technical_terms=False,
                        has_regex_pattern=False,
                        is_simple_keyword=False,
                        needs_semantic_understanding=True,
                        confidence=0.0,
                        recommended_mode="auto",
                    )
                )

        return results

    def explain_analysis(self, analysis: QueryAnalysis) -> str:
        """
        分析結果を人間が読みやすい形式で説明

        Args:
            analysis: クエリ分析結果

        Returns:
            説明文字列
        """
        explanation = []
        explanation.append(f"Query: '{analysis.query}'")
        explanation.append(f"Word count: {analysis.word_count}")

        if analysis.has_question_words:
            explanation.append("✓ Contains question words")

        if analysis.has_technical_terms:
            explanation.append("✓ Contains technical terms")

        if analysis.has_regex_pattern:
            explanation.append("✓ Contains regex patterns")

        if analysis.is_simple_keyword:
            explanation.append("✓ Simple keyword search")

        if analysis.needs_semantic_understanding:
            explanation.append("✓ Needs semantic understanding")

        explanation.append(f"Confidence: {analysis.confidence:.2f}")
        explanation.append(f"Recommended mode: {analysis.recommended_mode}")

        return "\n".join(explanation)


# エクスポート
__all__ = ["QueryAnalyzer", "QueryAnalysis"]
