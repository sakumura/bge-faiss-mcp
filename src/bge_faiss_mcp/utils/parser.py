"""
Document Parser Module

Parse and chunk documents for semantic indexing.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_line: int
    end_line: int


class DocumentParser:
    """Parse and chunk documents for semantic processing."""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 100
    ):
        """
        Initialize document parser.

        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def parse_markdown(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            Parsed document dictionary
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata
        metadata = self._extract_markdown_metadata(content)

        # Parse structure
        sections = self._parse_markdown_sections(content)

        # Get file info
        stat = file_path.stat()

        return {
            "path": str(file_path),
            "name": file_path.name,
            "content": content,
            "sections": sections,
            "metadata": {
                **metadata,
                "file_type": "markdown",
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            },
        }

    def _extract_markdown_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from Markdown frontmatter."""
        metadata = {}

        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match and "title" not in metadata:
            metadata["title"] = title_match.group(1).strip()

        # Extract date patterns
        date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", content)
        if date_match and "date" not in metadata:
            metadata["date"] = date_match.group(1)

        return metadata

    def _parse_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse Markdown sections."""
        sections = []
        lines = content.split("\n")

        current_section = None
        section_content: list[str] = []

        for i, line in enumerate(lines):
            # Check for heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section
                if current_section:
                    current_section["content"] = "\n".join(section_content).strip()
                    current_section["end_line"] = i - 1
                    sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                current_section = {
                    "level": level,
                    "heading": heading,
                    "start_line": i,
                    "content": "",
                }
                section_content = []
            else:
                section_content.append(line)

        # Save last section
        if current_section:
            current_section["content"] = "\n".join(section_content).strip()
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)

        return sections

    def chunk_document(
        self, document: Dict[str, Any], strategy: str = "sliding_window"
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Parsed document
            strategy: Chunking strategy ('sliding_window' or 'section')

        Returns:
            List of document chunks
        """
        if strategy == "section":
            return self._chunk_by_sections(document)
        else:
            return self._chunk_sliding_window(document)

    def _chunk_sliding_window(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document using sliding window."""
        content = document["content"]
        chunks = []

        # Simple token approximation (words)
        words = content.split()

        # Create chunks
        start = 0
        chunk_num = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))

            # Get chunk content
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Skip if too small
            if len(chunk_words) < self.min_chunk_size and start > 0:
                break

            # Create chunk
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    **document["metadata"],
                    "chunk_num": chunk_num,
                    "total_chunks": -1,  # Will be updated
                    "file_path": document["path"],
                },
                chunk_id=f"{Path(document['path']).stem}_chunk_{chunk_num}",
                start_line=0,  # Approximate
                end_line=0,  # Approximate
            )
            chunks.append(chunk)

            # Move window
            start += self.chunk_size - self.chunk_overlap
            chunk_num += 1

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _chunk_by_sections(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document by sections."""
        chunks = []
        sections = document.get("sections", [])

        if not sections:
            # Fall back to sliding window
            return self._chunk_sliding_window(document)

        for i, section in enumerate(sections):
            # Combine small sections
            content = f"# {section['heading']}\n\n{section['content']}"

            # Check if section is too large
            words = content.split()
            if len(words) > self.chunk_size:
                # Split large section
                sub_chunks = self._split_large_section(content, section, i, document)
                chunks.extend(sub_chunks)
            else:
                # Use section as chunk
                chunk = DocumentChunk(
                    content=content,
                    metadata={
                        **document["metadata"],
                        "section": section["heading"],
                        "section_level": section["level"],
                        "section_num": i,
                        "file_path": document["path"],
                    },
                    chunk_id=f"{Path(document['path']).stem}_section_{i}",
                    start_line=section["start_line"],
                    end_line=section["end_line"],
                )
                chunks.append(chunk)

        return chunks

    def _split_large_section(
        self,
        content: str,
        section: Dict[str, Any],
        section_num: int,
        document: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Split a large section into smaller chunks."""
        chunks = []
        words = content.split()

        start = 0
        sub_chunk_num = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    **document["metadata"],
                    "section": section["heading"],
                    "section_level": section["level"],
                    "section_num": section_num,
                    "sub_chunk": sub_chunk_num,
                    "file_path": document["path"],
                },
                chunk_id=f"{Path(document['path']).stem}_section_{section_num}_{sub_chunk_num}",
                start_line=section["start_line"],
                end_line=section["end_line"],
            )
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap
            sub_chunk_num += 1

        return chunks

    def parse_directory(
        self, directory: Path, pattern: str = "*.md", recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse all matching files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Search recursively

        Returns:
            List of parsed documents
        """
        documents = []

        # Find matching files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file_path in files:
            # Skip hidden files and directories
            if any(part.startswith(".") for part in file_path.parts):
                continue

            try:
                if file_path.suffix.lower() == ".md":
                    doc = self.parse_markdown(file_path)
                    documents.append(doc)
                    logger.info(f"Parsed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        logger.info(f"Parsed {len(documents)} documents")
        return documents
