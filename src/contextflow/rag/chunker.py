"""
Smart Chunker for ContextFlow RAG.

Intelligent text chunking with:
- Paragraph/sentence boundary detection
- Code block preservation
- Configurable overlap
- Metadata tracking per chunk
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contextflow.utils.logging import get_logger
from contextflow.utils.tokens import TokenEstimator

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ChunkingStrategy(Enum):
    """Chunking strategies."""

    FIXED = "fixed"  # Fixed size chunks
    SENTENCE = "sentence"  # Split on sentence boundaries
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    SEMANTIC = "semantic"  # Split on semantic boundaries (headers, etc)
    CODE = "code"  # Preserve code blocks
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class Chunk:
    """A single chunk of text."""

    id: str
    content: str
    start_index: int
    end_index: int
    token_count: int
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # Overlap info
    overlap_before: int = 0
    overlap_after: int = 0

    def __len__(self) -> int:
        """Return content length."""
        return len(self.content)

    def __str__(self) -> str:
        """Return string representation."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk({self.chunk_index + 1}/{self.total_chunks}, {self.token_count} tokens): {preview}"


@dataclass
class ChunkingResult:
    """Result from chunking operation."""

    chunks: list[Chunk]
    total_tokens: int
    total_chunks: int
    strategy_used: ChunkingStrategy
    average_chunk_size: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __iter__(self):
        """Iterate over chunks."""
        return iter(self.chunks)

    def __getitem__(self, index: int) -> Chunk:
        """Get chunk by index."""
        return self.chunks[index]


# =============================================================================
# Regex Patterns
# =============================================================================

# Sentence endings (handles abbreviations better)
SENTENCE_END_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|'  # Period/!/? followed by space and capital
    r'(?<=[.!?])\s*$'  # End of text
)

# Paragraph boundaries (double newline or more)
PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')

# Fenced code blocks (``` or ~~~)
CODE_BLOCK_PATTERN = re.compile(
    r'(```[\w]*\n.*?```|~~~[\w]*\n.*?~~~)',
    re.DOTALL
)

# Indented code blocks (4+ spaces or tab at line start)
INDENTED_CODE_PATTERN = re.compile(
    r'(?:^(?:[ ]{4,}|\t)[^\n]*\n)+',
    re.MULTILINE
)

# Markdown headers
HEADER_PATTERN = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)

# Horizontal rules
HR_PATTERN = re.compile(r'^(?:[-*_]){3,}\s*$', re.MULTILINE)


# =============================================================================
# Smart Chunker Implementation
# =============================================================================


class SmartChunker:
    """
    Intelligent text chunker for RAG systems.

    Features:
    - Respects natural text boundaries
    - Preserves code blocks intact
    - Configurable chunk size and overlap
    - Metadata tracking

    Usage:
        chunker = SmartChunker(chunk_size=4000, overlap=500)
        result = chunker.chunk(long_text)

        for chunk in result.chunks:
            # Process each chunk
            pass
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        overlap: int = 500,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        preserve_code_blocks: bool = True,
        min_chunk_size: int = 100,
        token_estimator: TokenEstimator | None = None,
    ):
        """
        Initialize SmartChunker.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            strategy: Chunking strategy to use
            preserve_code_blocks: Keep code blocks intact
            min_chunk_size: Minimum chunk size
            token_estimator: Token counter (creates default if None)

        Raises:
            ValueError: If overlap >= chunk_size or chunk_size < min_chunk_size
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        if chunk_size < min_chunk_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be >= min_chunk_size ({min_chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.preserve_code_blocks = preserve_code_blocks
        self.min_chunk_size = min_chunk_size
        self._estimator = token_estimator or TokenEstimator()

        logger.debug(
            "SmartChunker initialized",
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy.value,
        )

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkingResult:
        """
        Chunk text into manageable pieces.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            ChunkingResult with list of chunks
        """
        if not text or not text.strip():
            return ChunkingResult(
                chunks=[],
                total_tokens=0,
                total_chunks=0,
                strategy_used=self.strategy,
                average_chunk_size=0.0,
                metadata=metadata or {},
            )

        # Select chunking method based on strategy
        strategy_map = {
            ChunkingStrategy.FIXED: self._chunk_fixed,
            ChunkingStrategy.SENTENCE: self._chunk_sentence,
            ChunkingStrategy.PARAGRAPH: self._chunk_paragraph,
            ChunkingStrategy.SEMANTIC: self._chunk_semantic,
            ChunkingStrategy.CODE: self._chunk_code,
            ChunkingStrategy.HYBRID: self._chunk_hybrid,
        }

        chunk_method = strategy_map.get(self.strategy, self._chunk_hybrid)
        boundaries = chunk_method(text)

        # Apply overlap
        if self.overlap > 0 and len(boundaries) > 1:
            boundaries = self._apply_overlap(boundaries, text)

        # Create chunk objects
        chunks = self._create_chunks(text, boundaries, metadata)

        # Calculate stats
        total_tokens = sum(c.token_count for c in chunks)
        avg_size = total_tokens / len(chunks) if chunks else 0.0

        logger.info(
            "Chunking complete",
            strategy=self.strategy.value,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            average_size=round(avg_size, 1),
        )

        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            total_chunks=len(chunks),
            strategy_used=self.strategy,
            average_chunk_size=avg_size,
            metadata=metadata or {},
        )

    def _chunk_fixed(self, text: str) -> list[tuple[int, int]]:
        """
        Fixed-size chunking based on token count.

        Splits text into chunks of approximately chunk_size tokens,
        without regard to natural boundaries.
        """
        boundaries: list[tuple[int, int]] = []
        text_len = len(text)

        if text_len == 0:
            return boundaries

        # Estimate chars per token for this text
        total_tokens = self._estimator.count_tokens(text)
        chars_per_token = text_len / max(total_tokens, 1)
        target_chars = int(self.chunk_size * chars_per_token)

        start = 0
        while start < text_len:
            end = min(start + target_chars, text_len)

            # Adjust to not cut in middle of word
            if end < text_len:
                # Look for space backwards
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos + 1

            boundaries.append((start, end))
            start = end

        return boundaries

    def _chunk_sentence(self, text: str) -> list[tuple[int, int]]:
        """
        Sentence-boundary chunking.

        Splits on sentence boundaries, grouping sentences until
        chunk_size is reached.
        """
        sentence_boundaries = self._find_sentence_boundaries(text)

        if not sentence_boundaries:
            return [(0, len(text))] if text else []

        boundaries: list[tuple[int, int]] = []
        chunk_start = 0
        current_tokens = 0

        for sent_end in sentence_boundaries:
            sentence = text[chunk_start:sent_end] if not boundaries else text[
                boundaries[-1][1] if boundaries else chunk_start:sent_end
            ]
            sent_tokens = self._estimator.count_tokens(sentence)

            if current_tokens + sent_tokens > self.chunk_size and current_tokens > 0:
                # End current chunk before this sentence
                if boundaries:
                    boundaries.append((boundaries[-1][1], chunk_start))
                else:
                    boundaries.append((0, chunk_start))
                current_tokens = 0

            current_tokens += sent_tokens
            chunk_start = sent_end

        # Add final chunk
        if chunk_start < len(text) or not boundaries:
            last_start = boundaries[-1][1] if boundaries else 0
            boundaries.append((last_start, len(text)))

        # Clean up overlapping boundaries
        return self._clean_boundaries(boundaries, text)

    def _chunk_paragraph(self, text: str) -> list[tuple[int, int]]:
        """
        Paragraph-boundary chunking.

        Splits on paragraph boundaries (double newlines),
        grouping paragraphs until chunk_size is reached.
        """
        para_boundaries = self._find_paragraph_boundaries(text)

        if not para_boundaries:
            return [(0, len(text))] if text else []

        boundaries: list[tuple[int, int]] = []
        chunk_start = 0
        current_tokens = 0

        prev_end = 0
        for para_end in para_boundaries:
            para_text = text[prev_end:para_end]
            para_tokens = self._estimator.count_tokens(para_text)

            if current_tokens + para_tokens > self.chunk_size and current_tokens > 0:
                # End current chunk before this paragraph
                boundaries.append((chunk_start, prev_end))
                chunk_start = prev_end
                current_tokens = 0

            current_tokens += para_tokens
            prev_end = para_end

        # Add final chunk
        if chunk_start < len(text):
            boundaries.append((chunk_start, len(text)))

        return boundaries

    def _chunk_semantic(self, text: str) -> list[tuple[int, int]]:
        """
        Semantic-boundary chunking (headers, sections).

        Splits on semantic boundaries like markdown headers,
        horizontal rules, and major section breaks.
        """
        # Find all semantic boundaries
        semantic_positions: list[int] = []

        # Headers
        for match in HEADER_PATTERN.finditer(text):
            semantic_positions.append(match.start())

        # Horizontal rules
        for match in HR_PATTERN.finditer(text):
            semantic_positions.append(match.start())

        # Sort and deduplicate
        semantic_positions = sorted(set(semantic_positions))

        if not semantic_positions:
            # Fall back to paragraph chunking
            return self._chunk_paragraph(text)

        boundaries: list[tuple[int, int]] = []
        chunk_start = 0
        current_tokens = 0

        for sem_pos in semantic_positions:
            if sem_pos <= chunk_start:
                continue

            section_text = text[chunk_start:sem_pos]
            section_tokens = self._estimator.count_tokens(section_text)

            if current_tokens + section_tokens > self.chunk_size and current_tokens > 0:
                # End current chunk
                boundaries.append((chunk_start, sem_pos))
                chunk_start = sem_pos
                current_tokens = section_tokens
            else:
                current_tokens += section_tokens

        # Add final chunk
        if chunk_start < len(text):
            boundaries.append((chunk_start, len(text)))

        return boundaries

    def _chunk_code(self, text: str) -> list[tuple[int, int]]:
        """
        Code-aware chunking that preserves code blocks.

        Identifies code blocks and keeps them intact,
        chunking the surrounding text normally.
        """
        code_blocks = self._find_code_blocks(text)

        if not code_blocks:
            return self._chunk_paragraph(text)

        # Create protected regions
        protected = set()
        for start, end in code_blocks:
            for i in range(start, end):
                protected.add(i)

        boundaries: list[tuple[int, int]] = []
        chunk_start = 0
        current_tokens = 0

        i = 0
        while i < len(text):
            # Check if we're entering a code block
            in_code_block = False
            code_end = i
            for cb_start, cb_end in code_blocks:
                if cb_start <= i < cb_end:
                    in_code_block = True
                    code_end = cb_end
                    break

            if in_code_block:
                # Include entire code block
                code_text = text[i:code_end]
                code_tokens = self._estimator.count_tokens(code_text)

                if current_tokens + code_tokens > self.chunk_size and current_tokens > 0:
                    # End current chunk before code block
                    boundaries.append((chunk_start, i))
                    chunk_start = i
                    current_tokens = 0

                current_tokens += code_tokens
                i = code_end
            else:
                # Regular text - find next paragraph break or code block
                next_break = len(text)

                # Find next paragraph
                para_match = PARAGRAPH_PATTERN.search(text, i)
                if para_match:
                    next_break = min(next_break, para_match.end())

                # Find next code block
                for cb_start, cb_end in code_blocks:
                    if cb_start > i:
                        next_break = min(next_break, cb_start)
                        break

                segment_text = text[i:next_break]
                segment_tokens = self._estimator.count_tokens(segment_text)

                if current_tokens + segment_tokens > self.chunk_size and current_tokens > 0:
                    boundaries.append((chunk_start, i))
                    chunk_start = i
                    current_tokens = 0

                current_tokens += segment_tokens
                i = next_break

        # Add final chunk
        if chunk_start < len(text):
            boundaries.append((chunk_start, len(text)))

        return boundaries

    def _chunk_hybrid(self, text: str) -> list[tuple[int, int]]:
        """
        Hybrid strategy combining multiple approaches.

        1. Detect code blocks, mark as protected
        2. Split on paragraphs first
        3. If paragraph too large, split on sentences
        4. Apply overlap
        """
        # Step 1: Find and protect code blocks
        code_blocks = self._find_code_blocks(text) if self.preserve_code_blocks else []

        # Step 2: Find paragraph boundaries
        para_boundaries = self._find_paragraph_boundaries(text)

        boundaries: list[tuple[int, int]] = []
        chunk_start = 0
        current_tokens = 0

        def is_in_code_block(pos: int) -> bool:
            """Check if position is inside a code block."""
            for cb_start, cb_end in code_blocks:
                if cb_start <= pos < cb_end:
                    return True
            return False

        def get_code_block_end(pos: int) -> int:
            """Get end of code block containing position."""
            for cb_start, cb_end in code_blocks:
                if cb_start <= pos < cb_end:
                    return cb_end
            return pos

        prev_end = 0
        for para_end in para_boundaries:
            # Skip if we're inside a code block
            if is_in_code_block(prev_end):
                cb_end = get_code_block_end(prev_end)
                para_end = max(para_end, cb_end)

            para_text = text[prev_end:para_end]
            para_tokens = self._estimator.count_tokens(para_text)

            # If paragraph is too large, split on sentences
            if para_tokens > self.chunk_size:
                # Split large paragraph into sentences
                sent_boundaries = self._find_sentence_boundaries(para_text)
                sent_start = 0

                for sent_offset in sent_boundaries:
                    sent_text = para_text[sent_start:sent_offset]
                    sent_tokens = self._estimator.count_tokens(sent_text)

                    if current_tokens + sent_tokens > self.chunk_size and current_tokens > 0:
                        boundaries.append((chunk_start, prev_end + sent_start))
                        chunk_start = prev_end + sent_start
                        current_tokens = 0

                    current_tokens += sent_tokens
                    sent_start = sent_offset
            else:
                if current_tokens + para_tokens > self.chunk_size and current_tokens > 0:
                    boundaries.append((chunk_start, prev_end))
                    chunk_start = prev_end
                    current_tokens = 0

                current_tokens += para_tokens

            prev_end = para_end

        # Add final chunk
        if chunk_start < len(text):
            boundaries.append((chunk_start, len(text)))

        return boundaries

    def _find_code_blocks(self, text: str) -> list[tuple[int, int]]:
        """
        Find code block boundaries (``` or indented).

        Returns:
            List of (start, end) tuples for each code block
        """
        blocks: list[tuple[int, int]] = []

        # Find fenced code blocks (``` or ~~~)
        for match in CODE_BLOCK_PATTERN.finditer(text):
            blocks.append((match.start(), match.end()))

        # Find indented code blocks
        for match in INDENTED_CODE_PATTERN.finditer(text):
            # Check if this overlaps with a fenced block
            overlaps = False
            for start, end in blocks:
                if not (match.end() <= start or match.start() >= end):
                    overlaps = True
                    break
            if not overlaps:
                blocks.append((match.start(), match.end()))

        # Sort by start position
        blocks.sort(key=lambda x: x[0])
        return blocks

    def _find_sentence_boundaries(self, text: str) -> list[int]:
        """
        Find sentence end positions.

        Returns:
            List of character positions where sentences end
        """
        boundaries: list[int] = []

        # Simple sentence detection
        i = 0
        while i < len(text):
            # Look for sentence-ending punctuation
            if text[i] in ".!?":
                # Check if followed by space and capital or end of text
                if i + 1 >= len(text):
                    boundaries.append(i + 1)
                elif i + 2 < len(text) and text[i + 1].isspace() and text[i + 2].isupper() or text[i + 1].isspace() and i + 2 >= len(text):
                    boundaries.append(i + 2)
            i += 1

        # If no sentences found, treat entire text as one
        if not boundaries:
            boundaries.append(len(text))

        return boundaries

    def _find_paragraph_boundaries(self, text: str) -> list[int]:
        """
        Find paragraph boundaries (double newlines).

        Returns:
            List of character positions where paragraphs end
        """
        boundaries: list[int] = []

        for match in PARAGRAPH_PATTERN.finditer(text):
            boundaries.append(match.end())

        # Always include end of text
        if not boundaries or boundaries[-1] != len(text):
            boundaries.append(len(text))

        return boundaries

    def _apply_overlap(
        self,
        boundaries: list[tuple[int, int]],
        text: str,
    ) -> list[tuple[int, int]]:
        """
        Apply overlap between chunks.

        Extends each chunk's start to include overlap_tokens from
        the previous chunk.
        """
        if len(boundaries) <= 1:
            return boundaries

        # Estimate chars per token
        total_tokens = self._estimator.count_tokens(text)
        chars_per_token = len(text) / max(total_tokens, 1)
        overlap_chars = int(self.overlap * chars_per_token)

        new_boundaries: list[tuple[int, int]] = []

        for i, (start, end) in enumerate(boundaries):
            if i == 0:
                new_boundaries.append((start, end))
            else:
                # Extend start backwards for overlap
                new_start = max(0, start - overlap_chars)

                # Try to align to word boundary
                if new_start > 0:
                    space_pos = text.rfind(" ", new_start - 50, start)
                    if space_pos > new_start - 50:
                        new_start = space_pos + 1

                new_boundaries.append((new_start, end))

        return new_boundaries

    def _clean_boundaries(
        self,
        boundaries: list[tuple[int, int]],
        text: str,
    ) -> list[tuple[int, int]]:
        """Clean up boundaries to remove gaps and overlaps."""
        if not boundaries:
            return [(0, len(text))] if text else []

        # Sort by start
        boundaries.sort(key=lambda x: x[0])

        # Merge adjacent/overlapping
        cleaned: list[tuple[int, int]] = []
        for start, end in boundaries:
            if start >= end:
                continue
            if cleaned and start <= cleaned[-1][1]:
                # Merge with previous
                cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], end))
            else:
                cleaned.append((start, end))

        return cleaned

    def _create_chunks(
        self,
        text: str,
        boundaries: list[tuple[int, int]],
        metadata: dict[str, Any] | None,
    ) -> list[Chunk]:
        """Create Chunk objects from boundaries."""
        chunks: list[Chunk] = []
        total_chunks = len(boundaries)

        for i, (start, end) in enumerate(boundaries):
            content = text[start:end]
            token_count = self._estimator.count_tokens(content)

            # Calculate overlap info
            overlap_before = 0
            overlap_after = 0

            if i > 0:
                prev_end = boundaries[i - 1][1]
                if start < prev_end:
                    overlap_before = self._estimator.count_tokens(text[start:prev_end])

            if i < total_chunks - 1:
                next_start = boundaries[i + 1][0]
                if end > next_start:
                    overlap_after = self._estimator.count_tokens(text[next_start:end])

            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=content,
                start_index=start,
                end_index=end,
                token_count=token_count,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata={**(metadata or {}), "strategy": self.strategy.value},
                overlap_before=overlap_before,
                overlap_after=overlap_after,
            )
            chunks.append(chunk)

        return chunks

    def rechunk(
        self,
        chunks: list[Chunk],
        new_size: int,
    ) -> list[Chunk]:
        """
        Re-chunk existing chunks to new size.

        Args:
            chunks: Existing chunks to re-chunk
            new_size: New target chunk size in tokens

        Returns:
            List of re-chunked Chunks
        """
        if not chunks:
            return []

        # Combine all chunks into single text
        # Remove overlapping portions
        combined_text = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                combined_text += chunk.content
            else:
                # Skip overlap portion
                start_offset = chunk.overlap_before
                if start_offset > 0:
                    # Estimate character offset
                    chars_per_token = len(chunk.content) / max(chunk.token_count, 1)
                    char_offset = int(start_offset * chars_per_token)
                    combined_text += chunk.content[char_offset:]
                else:
                    combined_text += chunk.content

        # Create new chunker with new size
        new_chunker = SmartChunker(
            chunk_size=new_size,
            overlap=self.overlap,
            strategy=self.strategy,
            preserve_code_blocks=self.preserve_code_blocks,
            min_chunk_size=self.min_chunk_size,
            token_estimator=self._estimator,
        )

        result = new_chunker.chunk(combined_text)
        return result.chunks

    def merge_small_chunks(
        self,
        chunks: list[Chunk],
        min_size: int,
    ) -> list[Chunk]:
        """
        Merge chunks smaller than min_size.

        Args:
            chunks: List of chunks
            min_size: Minimum chunk size in tokens

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged: list[Chunk] = []
        buffer_content = ""
        buffer_start = 0
        buffer_tokens = 0

        for chunk in chunks:
            if chunk.token_count < min_size:
                # Add to buffer
                if not buffer_content:
                    buffer_start = chunk.start_index
                buffer_content += chunk.content
                buffer_tokens += chunk.token_count
            else:
                # Flush buffer if exists
                if buffer_content:
                    merged.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            content=buffer_content,
                            start_index=buffer_start,
                            end_index=buffer_start + len(buffer_content),
                            token_count=buffer_tokens,
                            chunk_index=len(merged),
                            total_chunks=0,  # Will update later
                            metadata=chunk.metadata,
                        )
                    )
                    buffer_content = ""
                    buffer_tokens = 0

                merged.append(
                    Chunk(
                        id=chunk.id,
                        content=chunk.content,
                        start_index=chunk.start_index,
                        end_index=chunk.end_index,
                        token_count=chunk.token_count,
                        chunk_index=len(merged),
                        total_chunks=0,  # Will update later
                        metadata=chunk.metadata,
                    )
                )

        # Flush remaining buffer
        if buffer_content:
            merged.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=buffer_content,
                    start_index=buffer_start,
                    end_index=buffer_start + len(buffer_content),
                    token_count=buffer_tokens,
                    chunk_index=len(merged),
                    total_chunks=0,
                    metadata={},
                )
            )

        # Update total_chunks
        for chunk in merged:
            chunk.total_chunks = len(merged)

        return merged


# =============================================================================
# Convenience Functions
# =============================================================================

_default_chunker: SmartChunker | None = None


def _get_chunker() -> SmartChunker:
    """Get default chunker instance."""
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = SmartChunker()
    return _default_chunker


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 500,
) -> list[Chunk]:
    """
    Quick chunking without creating instance.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of Chunk objects
    """
    chunker = SmartChunker(chunk_size=chunk_size, overlap=overlap)
    result = chunker.chunk(text)
    return result.chunks


def chunk_for_embedding(
    text: str,
    max_tokens: int = 512,
) -> list[str]:
    """
    Chunk text optimized for embedding models.

    Uses smaller chunks with no overlap, optimized for
    typical embedding model context limits (512-8192 tokens).

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk (default 512)

    Returns:
        List of text strings (just content, no metadata)
    """
    chunker = SmartChunker(
        chunk_size=max_tokens,
        overlap=0,  # No overlap for embeddings
        strategy=ChunkingStrategy.SENTENCE,
        min_chunk_size=50,
    )
    result = chunker.chunk(text)
    return [chunk.content for chunk in result.chunks]
