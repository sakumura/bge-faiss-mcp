"""
RAG Chain Module

Retrieval-Augmented Generation chain for context-aware responses.
"""

from typing import List, Dict, Any, Optional
import logging
from bge_faiss_mcp.core.retriever import SemanticRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain for context-aware generation."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        max_context_length: int = 3000,
        top_k: int = 5,
    ):
        """
        Initialize RAG chain.

        Args:
            retriever: Semantic retriever instance
            max_context_length: Maximum context length in characters
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.max_context_length = max_context_length
        self.top_k = top_k

    def build_context(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build context from retrieved documents.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            Formatted context string
        """
        k = k or self.top_k

        # Retrieve relevant documents
        results = self.retriever.search(
            query=query, k=k, filter_metadata=filter_metadata
        )

        # Build context
        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            # Format document
            doc_text = f"[Document {i}]\n{result['content']}\n"

            # Check length
            if total_length + len(doc_text) > self.max_context_length:
                # Truncate if necessary
                remaining = self.max_context_length - total_length
                if remaining > 100:  # Only add if substantial
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n".join(context_parts)

    def create_prompt(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a prompt with context for generation.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt
        """
        if system_prompt is None:
            system_prompt = (
                "以下のコンテキストを使用して、質問に正確に答えてください。"
                "コンテキストに答えがない場合は、その旨を明記してください。"
            )

        prompt = f"""{system_prompt}

コンテキスト:
{context}

質問: {query}

回答:"""

        return prompt

    def query(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute RAG query.

        Args:
            query: User query
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            system_prompt: Optional system prompt
            return_context: Include context in response

        Returns:
            Response dictionary
        """
        # Build context
        context = self.build_context(query, k, filter_metadata)

        # Create prompt
        prompt = self.create_prompt(query, context, system_prompt)

        # Prepare response
        response = {
            "query": query,
            "prompt": prompt,
            "num_documents": min(k or self.top_k, len(context.split("[Document"))) - 1,
        }

        if return_context:
            response["context"] = context
            response["documents"] = self.retriever.search(
                query=query, k=k or self.top_k, filter_metadata=filter_metadata
            )

        return response

    def batch_query(
        self,
        queries: List[str],
        k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple RAG queries.

        Args:
            queries: List of queries
            k: Number of documents per query
            system_prompt: Optional system prompt

        Returns:
            List of response dictionaries
        """
        responses = []

        for query in queries:
            response = self.query(
                query=query, k=k, system_prompt=system_prompt, return_context=False
            )
            responses.append(response)

        return responses

    def answer_with_sources(
        self, query: str, k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Answer query with source attribution.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            Response with sources
        """
        k = k or self.top_k

        # Retrieve documents
        results = self.retriever.search(query=query, k=k)

        # Build response with sources
        response = {"query": query, "sources": []}

        for result in results:
            source = {
                "id": result["id"],
                "content": (
                    result["content"][:500] + "..."
                    if len(result["content"]) > 500
                    else result["content"]
                ),
                "score": result.get("score", 0),
                "metadata": result.get("metadata", {}),
            }
            response["sources"].append(source)

        # Create context for answer
        context = self.build_context(query, k)
        response["context"] = context
        response["prompt"] = self.create_prompt(query, context)

        return response
