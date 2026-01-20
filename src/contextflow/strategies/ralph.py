"""
RALPH Strategy - Recursive Abstraction for Large-scale Processing with Hierarchy.

For medium context windows (10K-100K tokens). Iterative, chunked processing.

Two modes:
- RALPH_ITERATIVE: Sequential chunk processing with accumulation
- RALPH_STRUCTURED: Hierarchical processing with multi-level summarization

This strategy implements Boris Step 13 verification for quality assurance.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextflow.core.types import Message
from contextflow.strategies.base import (
    BaseStrategy,
    CostEstimate,
    StrategyResult,
    StrategyType,
    VerificationResult,
)
from contextflow.strategies.verification import VerificationProtocol
from contextflow.utils.errors import ProviderError, StrategyExecutionError
from contextflow.utils.logging import StrategyLogger, get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider


# =============================================================================
# Constants and Model Pricing
# =============================================================================

# Approximate pricing per 1K tokens (input/output) for common models
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "default": {"input": 0.003, "output": 0.015},
}

# Default thresholds for mode selection
ITERATIVE_THRESHOLD_TOKENS = 50_000
STRUCTURED_DENSITY_THRESHOLD = 0.6

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""

    chunk_id: int
    content: str
    summary: str
    key_points: list[str]
    relevance_score: float
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the chunk result
        """
        return {
            "chunk_id": self.chunk_id,
            "content_length": len(self.content),
            "summary": self.summary,
            "key_points": self.key_points,
            "relevance_score": self.relevance_score,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class HierarchyLevel:
    """Represents a level in the hierarchical summarization."""

    level: int
    summaries: list[str]
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Prompts
# =============================================================================


RALPH_CHUNK_PROMPT = """You are processing a portion of a larger document. Your task is to:
1. Extract the key information relevant to the given task
2. Create a concise summary of this chunk
3. Identify the most important points

TASK: {task}

CHUNK {chunk_id} of {total_chunks}:
{chunk_content}

{accumulated_context}

Provide your analysis in the following format:
<summary>
A concise summary of the key information in this chunk (2-4 sentences)
</summary>

<key_points>
- First key point relevant to the task
- Second key point
- Third key point (if applicable)
</key_points>

<relevance>
Score from 0.0-1.0 indicating how relevant this chunk is to the task
</relevance>

<findings>
Any specific findings, data, or conclusions related to the task
</findings>"""


RALPH_SYNTHESIS_PROMPT = """You are synthesizing information from multiple chunks of a document to answer a task.

TASK: {task}

{constraints_text}

CHUNK SUMMARIES AND FINDINGS:
{chunk_summaries}

Based on all the information gathered from the chunks above, provide a comprehensive answer to the task.

Your response should:
1. Directly address the task/question
2. Synthesize information across all relevant chunks
3. Be well-organized and coherent
4. Include specific evidence or data from the chunks when relevant

FINAL ANSWER:"""


RALPH_HIERARCHICAL_PROMPT = """Create a higher-level summary from the following summaries.
Preserve the most important information while reducing redundancy.

TASK CONTEXT: {task}

SUMMARIES TO SYNTHESIZE:
{summaries}

Provide a consolidated summary that:
1. Captures all essential information
2. Eliminates redundancy
3. Maintains logical flow
4. Highlights key findings relevant to the task

CONSOLIDATED SUMMARY:"""


RALPH_SECTION_SUMMARY_PROMPT = """You are creating a section summary from multiple related chunk summaries.

TASK: {task}

SECTION {section_id} - CHUNK SUMMARIES:
{chunk_summaries}

Create a cohesive section summary that:
1. Integrates information from all chunks in this section
2. Identifies cross-chunk themes or patterns
3. Notes any contradictions or gaps
4. Preserves task-relevant details

SECTION SUMMARY:"""


# =============================================================================
# RALPH Strategy Implementation
# =============================================================================


class RALPHStrategy(BaseStrategy):
    """
    RALPH Strategy for medium-sized contexts.

    Recursive Abstraction for Large-scale Processing with Hierarchy.

    Optimal for:
    - Context 10K-100K tokens
    - Documents that can be chunked logically
    - Tasks requiring synthesis across sections

    Modes:
    - ITERATIVE: Process chunks sequentially, accumulate findings
    - STRUCTURED: Hierarchical summarization (chunk -> section -> document)

    Example:
        provider = ClaudeProvider(...)
        strategy = RALPHStrategy(
            provider=provider,
            mode="auto",
            chunk_size=4000,
            enable_verification=True
        )

        result = await strategy.execute(
            task="Summarize the key arguments in this document",
            context=large_document
        )
    """

    def __init__(
        self,
        provider: BaseProvider,
        mode: str = "auto",
        chunk_size: int = 4000,
        chunk_overlap: int = 500,
        enable_verification: bool = True,
        max_parallel_chunks: int = 5,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        verification_threshold: float = 0.8,
    ) -> None:
        """
        Initialize RALPH Strategy.

        Args:
            provider: LLM provider instance for completions
            mode: Processing mode ("iterative", "structured", or "auto")
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks to maintain context
            enable_verification: Whether to verify results
            max_parallel_chunks: Maximum chunks to process in parallel
            model: Model to use (overrides provider default)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in generated output
            verification_threshold: Minimum score to pass verification

        Raises:
            ValueError: If mode is invalid or chunk settings are invalid
        """
        super().__init__(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verification_threshold=verification_threshold,
        )

        # Validate mode
        valid_modes = ("auto", "iterative", "structured")
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {valid_modes}"
            )

        # Validate chunk settings
        if chunk_size < 500:
            raise ValueError(
                f"chunk_size must be at least 500 tokens, got {chunk_size}"
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be between 0 and chunk_size-1, got {chunk_overlap}"
            )
        if max_parallel_chunks < 1:
            raise ValueError(
                f"max_parallel_chunks must be at least 1, got {max_parallel_chunks}"
            )

        self._mode = mode
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_verification = enable_verification
        self._max_parallel_chunks = max_parallel_chunks

        # Strategy logger
        self._strategy_logger = StrategyLogger("ralph")

        # Verification protocol
        if enable_verification:
            self._verifier = VerificationProtocol(
                provider=provider,
                min_confidence=verification_threshold,
            )
        else:
            self._verifier = None

        logger.debug(
            "RALPHStrategy initialized",
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_verification=enable_verification,
            max_parallel_chunks=max_parallel_chunks,
        )

    # =========================================================================
    # Properties (from BaseStrategy ABC)
    # =========================================================================

    @property
    def name(self) -> str:
        """
        Strategy identifier.

        Returns:
            Strategy name "ralph"
        """
        return "ralph"

    @property
    def strategy_type(self) -> StrategyType:
        """
        Strategy type based on current mode.

        Returns:
            StrategyType.RALPH_ITERATIVE or RALPH_STRUCTURED
        """
        if self._mode == "iterative":
            return StrategyType.RALPH_ITERATIVE
        return StrategyType.RALPH_STRUCTURED

    @property
    def max_tokens(self) -> int:
        """
        Maximum tokens this strategy can handle.

        Returns:
            100,000 tokens maximum
        """
        return 100_000

    @property
    def min_tokens(self) -> int:
        """
        Minimum tokens for this strategy.

        Returns:
            10,000 tokens minimum
        """
        return 10_000

    # =========================================================================
    # Main Execute Method
    # =========================================================================

    async def execute(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute RALPH strategy on the given task and context.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints for verification
            **kwargs: Additional parameters
                - force_mode: Override auto mode selection
                - sections: Pre-defined section boundaries

        Returns:
            StrategyResult with answer and metadata

        Raises:
            StrategyExecutionError: If execution fails
        """
        start_time = time.time()

        # Estimate token count
        context_tokens = self._estimate_tokens(context)

        self._strategy_logger.log_start(
            token_count=context_tokens,
            mode=self._mode,
            constraints=len(constraints) if constraints else 0,
        )

        try:
            # Determine mode if auto
            effective_mode = kwargs.get("force_mode", self._mode)
            if effective_mode == "auto":
                density = self._estimate_density(context)
                effective_mode = self._determine_mode(context_tokens, density)
                logger.debug(
                    "Auto mode selected",
                    effective_mode=effective_mode,
                    context_tokens=context_tokens,
                    density=round(density, 3),
                )

            # Execute based on mode
            if effective_mode == "iterative":
                result = await self._execute_iterative(
                    task=task,
                    context=context,
                    constraints=constraints,
                )
            else:
                result = await self._execute_structured(
                    task=task,
                    context=context,
                    constraints=constraints,
                    sections=kwargs.get("sections"),
                )

            # Update execution time
            result.execution_time = time.time() - start_time

            self._strategy_logger.log_complete(
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                duration_seconds=result.execution_time,
                iterations=result.iterations,
                verification_passed=result.verification_passed,
            )

            return result

        except Exception as e:
            logger.error(
                "RALPH execution failed",
                error=str(e),
                mode=effective_mode if "effective_mode" in dir() else self._mode,
            )
            raise StrategyExecutionError(
                strategy="ralph",
                message=f"Execution failed: {str(e)}",
                cause=e,
            ) from e

    # =========================================================================
    # Iterative Mode Implementation
    # =========================================================================

    async def _execute_iterative(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> StrategyResult:
        """
        Execute iterative mode: Process chunks sequentially with accumulation.

        Workflow:
        1. Split context into overlapping chunks
        2. Process each chunk with the task, accumulating findings
        3. Synthesize final answer from accumulated results
        4. Verify if enabled

        Args:
            task: The task/question to process
            context: The context to analyze
            constraints: Optional constraints

        Returns:
            StrategyResult with synthesized answer
        """
        logger.info("Starting RALPH iterative mode")

        total_input_tokens = 0
        total_output_tokens = 0
        chunk_results: list[ChunkResult] = []

        # Split into chunks
        chunks = self._split_into_chunks(
            context=context,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )

        logger.debug(
            "Context split into chunks",
            chunk_count=len(chunks),
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )

        # Process chunks sequentially with accumulation
        accumulated_context = ""
        for i, chunk in enumerate(chunks):
            chunk_result = await self._process_chunk(
                chunk=chunk,
                chunk_id=i,
                task=task,
                accumulated_context=accumulated_context,
                total_chunks=len(chunks),
            )
            chunk_results.append(chunk_result)
            total_input_tokens += chunk_result.token_count

            # Accumulate high-relevance findings
            if chunk_result.relevance_score >= 0.5:
                accumulated_context += f"\n[Previous finding from chunk {i}]: {chunk_result.summary}"

            logger.debug(
                "Chunk processed",
                chunk_id=i,
                relevance=round(chunk_result.relevance_score, 2),
            )

        # Synthesize final answer
        final_answer, synthesis_tokens = await self._synthesize_results(
            task=task,
            chunk_results=chunk_results,
            constraints=constraints,
        )
        total_output_tokens += synthesis_tokens

        # Verify if enabled
        verification_passed = True
        verification_score = 1.0
        verification_details: dict[str, Any] = {}

        if self._enable_verification and self._verifier:
            verification_result = await self._verifier.verify(
                task=task,
                output=final_answer,
                constraints=constraints,
                context=context[:10000],  # Use truncated context for verification
            )
            verification_passed = verification_result.passed
            verification_score = verification_result.confidence
            verification_details = {
                "checks": [c.check_type.value for c in verification_result.checks],
                "issues": verification_result.issues,
                "suggestions": verification_result.suggestions,
            }

        # Calculate cost
        model = self._model or self._provider.model
        cost = self._calculate_cost(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=model,
        )

        return StrategyResult(
            answer=final_answer,
            strategy_used=StrategyType.RALPH_ITERATIVE,
            iterations=len(chunks),
            token_usage={
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            execution_time=0.0,  # Will be set by execute()
            verification_passed=verification_passed,
            verification_score=verification_score,
            metadata={
                "mode": "iterative",
                "chunks_processed": len(chunks),
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "chunk_results": [cr.to_dict() for cr in chunk_results],
                "verification_details": verification_details,
                "cost_usd": cost,
            },
        )

    # =========================================================================
    # Structured Mode Implementation
    # =========================================================================

    async def _execute_structured(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        sections: list[tuple[int, int]] | None = None,
    ) -> StrategyResult:
        """
        Execute structured mode: Hierarchical processing with multi-level summarization.

        Workflow:
        1. Split into chunks
        2. Process chunks in parallel (Level 1)
        3. Group results into sections, create section summaries (Level 2)
        4. Synthesize document-level answer (Level 3)
        5. Verify if enabled

        Args:
            task: The task/question to process
            context: The context to analyze
            constraints: Optional constraints
            sections: Optional pre-defined section boundaries

        Returns:
            StrategyResult with hierarchically synthesized answer
        """
        logger.info("Starting RALPH structured mode")

        total_input_tokens = 0
        total_output_tokens = 0
        hierarchy_levels: list[HierarchyLevel] = []

        # Split into chunks
        chunks = self._split_into_chunks(
            context=context,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )

        logger.debug(
            "Context split for structured processing",
            chunk_count=len(chunks),
        )

        # Level 1: Process chunks in parallel
        chunk_results = await self._process_chunks_parallel(
            chunks=chunks,
            task=task,
        )

        level1_summaries = [cr.summary for cr in chunk_results]
        level1_tokens = sum(cr.token_count for cr in chunk_results)
        total_input_tokens += level1_tokens

        hierarchy_levels.append(
            HierarchyLevel(
                level=1,
                summaries=level1_summaries,
                token_count=level1_tokens,
                metadata={"chunk_count": len(chunks)},
            )
        )

        # Level 2: Group into sections and summarize
        section_summaries: list[str] = []
        section_size = max(3, len(chunk_results) // 4)  # ~4 sections

        for section_id, i in enumerate(range(0, len(chunk_results), section_size)):
            section_chunks = chunk_results[i : i + section_size]
            if section_chunks:
                section_summary, tokens = await self._create_section_summary(
                    task=task,
                    chunk_results=section_chunks,
                    section_id=section_id,
                )
                section_summaries.append(section_summary)
                total_output_tokens += tokens

        hierarchy_levels.append(
            HierarchyLevel(
                level=2,
                summaries=section_summaries,
                token_count=total_output_tokens,
                metadata={"section_count": len(section_summaries)},
            )
        )

        # Level 3: Final synthesis
        final_answer, synthesis_tokens = await self._synthesize_hierarchical(
            task=task,
            section_summaries=section_summaries,
            constraints=constraints,
        )
        total_output_tokens += synthesis_tokens

        hierarchy_levels.append(
            HierarchyLevel(
                level=3,
                summaries=[final_answer],
                token_count=synthesis_tokens,
                metadata={"type": "final_synthesis"},
            )
        )

        # Verify if enabled
        verification_passed = True
        verification_score = 1.0
        verification_details: dict[str, Any] = {}

        if self._enable_verification and self._verifier:
            verification_result = await self._verifier.verify(
                task=task,
                output=final_answer,
                constraints=constraints,
                context=context[:10000],
            )
            verification_passed = verification_result.passed
            verification_score = verification_result.confidence
            verification_details = {
                "checks": [c.check_type.value for c in verification_result.checks],
                "issues": verification_result.issues,
                "suggestions": verification_result.suggestions,
            }

        # Calculate cost
        model = self._model or self._provider.model
        cost = self._calculate_cost(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=model,
        )

        return StrategyResult(
            answer=final_answer,
            strategy_used=StrategyType.RALPH_STRUCTURED,
            iterations=len(chunks) + len(section_summaries) + 1,
            token_usage={
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            execution_time=0.0,
            verification_passed=verification_passed,
            verification_score=verification_score,
            metadata={
                "mode": "structured",
                "chunks_processed": len(chunks),
                "sections_created": len(section_summaries),
                "hierarchy_levels": [
                    {
                        "level": hl.level,
                        "summary_count": len(hl.summaries),
                        "token_count": hl.token_count,
                    }
                    for hl in hierarchy_levels
                ],
                "verification_details": verification_details,
                "cost_usd": cost,
            },
        )

    # =========================================================================
    # Chunk Processing Methods
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _process_chunk(
        self,
        chunk: str,
        chunk_id: int,
        task: str,
        accumulated_context: str,
        total_chunks: int = 1,
    ) -> ChunkResult:
        """
        Process a single chunk and extract relevant information.

        Args:
            chunk: The chunk content to process
            chunk_id: Identifier for this chunk
            task: The task/question
            accumulated_context: Context from previous chunks
            total_chunks: Total number of chunks being processed

        Returns:
            ChunkResult with summary and key points

        Raises:
            ProviderError: If LLM call fails after retries
        """
        # Build accumulated context section
        acc_section = ""
        if accumulated_context:
            acc_section = f"\nPREVIOUS FINDINGS:\n{accumulated_context}\n"

        prompt = RALPH_CHUNK_PROMPT.format(
            task=task,
            chunk_id=chunk_id + 1,
            total_chunks=total_chunks,
            chunk_content=chunk,
            accumulated_context=acc_section,
        )

        # Call LLM
        response = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=1024,
            temperature=self._temperature,
        )

        # Parse response
        content = response.content
        summary = self._extract_tag_content(content, "summary")
        key_points = self._extract_key_points(content)
        relevance = self._extract_relevance(content)
        findings = self._extract_tag_content(content, "findings")

        return ChunkResult(
            chunk_id=chunk_id,
            content=chunk,
            summary=summary or findings or content[:500],
            key_points=key_points,
            relevance_score=relevance,
            token_count=response.input_tokens + response.output_tokens,
            metadata={
                "findings": findings,
                "raw_response_length": len(content),
            },
        )

    async def _process_chunks_parallel(
        self,
        chunks: list[str],
        task: str,
    ) -> list[ChunkResult]:
        """
        Process multiple chunks in parallel with rate limiting.

        Args:
            chunks: List of chunk contents to process
            task: The task/question

        Returns:
            List of ChunkResult objects in original order
        """
        results: list[ChunkResult] = []
        semaphore = asyncio.Semaphore(self._max_parallel_chunks)

        async def process_with_semaphore(
            chunk: str, idx: int
        ) -> ChunkResult:
            async with semaphore:
                return await self._process_chunk(
                    chunk=chunk,
                    chunk_id=idx,
                    task=task,
                    accumulated_context="",  # No accumulation in parallel mode
                    total_chunks=len(chunks),
                )

        # Create tasks for all chunks
        tasks = [
            process_with_semaphore(chunk, idx)
            for idx, chunk in enumerate(chunks)
        ]

        # Gather results, handling exceptions
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, filtering out errors
        for idx, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Chunk processing failed",
                    chunk_id=idx,
                    error=str(result),
                )
                # Create fallback result
                results.append(
                    ChunkResult(
                        chunk_id=idx,
                        content=chunks[idx],
                        summary="[Processing failed for this chunk]",
                        key_points=[],
                        relevance_score=0.0,
                        token_count=0,
                        metadata={"error": str(result)},
                    )
                )
            else:
                results.append(result)

        return results

    # =========================================================================
    # Synthesis Methods
    # =========================================================================

    async def _synthesize_results(
        self,
        task: str,
        chunk_results: list[ChunkResult],
        constraints: list[str] | None = None,
    ) -> tuple[str, int]:
        """
        Synthesize final answer from chunk results.

        Args:
            task: Original task
            chunk_results: Results from chunk processing
            constraints: Optional constraints

        Returns:
            Tuple of (synthesized answer, output tokens used)
        """
        # Build chunk summaries text
        summaries_text = ""
        for cr in chunk_results:
            if cr.relevance_score > 0.2:  # Include semi-relevant chunks
                summaries_text += f"\n=== Chunk {cr.chunk_id + 1} (Relevance: {cr.relevance_score:.2f}) ===\n"
                summaries_text += f"Summary: {cr.summary}\n"
                if cr.key_points:
                    summaries_text += "Key Points:\n"
                    for point in cr.key_points:
                        summaries_text += f"  - {point}\n"

        # Build constraints text
        constraints_text = ""
        if constraints:
            constraints_text = "CONSTRAINTS:\n" + "\n".join(
                f"- {c}" for c in constraints
            )

        prompt = RALPH_SYNTHESIS_PROMPT.format(
            task=task,
            constraints_text=constraints_text,
            chunk_summaries=summaries_text,
        )

        response = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )

        return response.content, response.output_tokens

    async def _create_section_summary(
        self,
        task: str,
        chunk_results: list[ChunkResult],
        section_id: int,
    ) -> tuple[str, int]:
        """
        Create a summary for a section of chunks.

        Args:
            task: Original task
            chunk_results: Chunk results in this section
            section_id: Section identifier

        Returns:
            Tuple of (section summary, output tokens used)
        """
        # Build chunk summaries for this section
        chunk_summaries_text = ""
        for cr in chunk_results:
            chunk_summaries_text += f"\nChunk {cr.chunk_id + 1}: {cr.summary}\n"
            if cr.key_points:
                for point in cr.key_points[:3]:  # Limit key points
                    chunk_summaries_text += f"  - {point}\n"

        prompt = RALPH_SECTION_SUMMARY_PROMPT.format(
            task=task,
            section_id=section_id + 1,
            chunk_summaries=chunk_summaries_text,
        )

        response = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=1024,
            temperature=self._temperature,
        )

        return response.content, response.output_tokens

    async def _synthesize_hierarchical(
        self,
        task: str,
        section_summaries: list[str],
        constraints: list[str] | None = None,
    ) -> tuple[str, int]:
        """
        Create final synthesis from section summaries.

        Args:
            task: Original task
            section_summaries: List of section summaries
            constraints: Optional constraints

        Returns:
            Tuple of (final answer, output tokens used)
        """
        # Build summaries text
        summaries_text = ""
        for i, summary in enumerate(section_summaries):
            summaries_text += f"\n=== Section {i + 1} ===\n{summary}\n"

        # Build constraints text
        constraints_text = ""
        if constraints:
            constraints_text = "CONSTRAINTS:\n" + "\n".join(
                f"- {c}" for c in constraints
            )

        prompt = RALPH_SYNTHESIS_PROMPT.format(
            task=task,
            constraints_text=constraints_text,
            chunk_summaries=summaries_text,
        )

        response = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )

        return response.content, response.output_tokens

    # =========================================================================
    # Verification Method
    # =========================================================================

    async def verify(
        self,
        task: str,
        output: str,
        constraints: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify output using VerificationProtocol.

        Boris Step 13: "Give Claude a way to verify its work"

        Args:
            task: Original task description
            output: Generated output to verify
            constraints: Constraints to check against

        Returns:
            VerificationResult with pass/fail and details

        Raises:
            StrategyExecutionError: If verification fails
        """
        if not self._verifier:
            # Return passing result if verification disabled
            return VerificationResult(
                passed=True,
                confidence=1.0,
                issues=[],
                suggestions=[],
                checks_performed=["verification_disabled"],
            )

        try:
            result = await self._verifier.verify(
                task=task,
                output=output,
                constraints=constraints,
            )

            return VerificationResult(
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                suggestions=result.suggestions,
                checks_performed=[c.check_type.value for c in result.checks],
            )

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            raise StrategyExecutionError(
                strategy="ralph",
                message=f"Verification failed: {str(e)}",
                cause=e,
            ) from e

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    def estimate_cost(
        self,
        context_tokens: int,
        model: str | None = None,
    ) -> CostEstimate:
        """
        Estimate execution cost before running.

        RALPH typically uses 2-4x the context tokens due to:
        - Chunk processing (each chunk analyzed)
        - Synthesis passes (section + final)
        - Verification (if enabled)

        Args:
            context_tokens: Number of tokens in context
            model: Model to estimate for

        Returns:
            CostEstimate with min/max/expected costs

        Raises:
            ValueError: If context_tokens is invalid
        """
        if context_tokens < 0:
            raise ValueError(f"context_tokens must be non-negative, got {context_tokens}")

        model = model or self._model or self._provider.model

        # Determine mode
        density = 0.5  # Assume medium density for estimation
        effective_mode = self._determine_mode(context_tokens, density)

        # Calculate estimated passes
        num_chunks = max(1, context_tokens // self._chunk_size)

        if effective_mode == "iterative":
            # Iterative: process each chunk + synthesis
            input_multiplier = 1.5  # Some accumulation overhead
            output_multiplier = 0.3  # ~30% output per chunk + synthesis
        else:
            # Structured: parallel chunks + sections + synthesis
            input_multiplier = 1.2  # Less overhead due to parallel
            output_multiplier = 0.4  # Section summaries + synthesis

        # Verification adds ~20% overhead
        if self._enable_verification:
            input_multiplier *= 1.1
            output_multiplier *= 1.1

        estimated_input = int(context_tokens * input_multiplier)
        estimated_output = int(context_tokens * output_multiplier)

        # Get pricing
        pricing = self._get_model_pricing(model)
        input_cost_per_k = pricing["input"]
        output_cost_per_k = pricing["output"]

        # Calculate costs
        expected_cost = (
            (estimated_input / 1000) * input_cost_per_k
            + (estimated_output / 1000) * output_cost_per_k
        )

        # Min/max ranges
        min_cost = expected_cost * 0.7
        max_cost = expected_cost * 1.5

        return CostEstimate(
            min_cost=round(min_cost, 6),
            max_cost=round(max_cost, 6),
            expected_cost=round(expected_cost, 6),
            model=model,
            context_tokens=context_tokens,
            estimated_output_tokens=estimated_output,
            metadata={
                "mode": effective_mode,
                "estimated_chunks": num_chunks,
                "input_multiplier": input_multiplier,
                "output_multiplier": output_multiplier,
            },
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _split_into_chunks(
        self,
        context: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """
        Split context into overlapping chunks.

        Uses approximate token counting (4 chars per token) for efficiency.
        Attempts to split on paragraph/sentence boundaries.

        Args:
            context: Text to split
            chunk_size: Target size in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of chunk strings
        """
        # Convert token sizes to char sizes (approximate)
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4

        chunks: list[str] = []
        start = 0
        context_len = len(context)

        while start < context_len:
            # Calculate end position
            end = min(start + char_chunk_size, context_len)

            # Try to find a good break point (paragraph or sentence)
            if end < context_len:
                # Look for paragraph break
                para_break = context.rfind("\n\n", start + char_overlap, end)
                if para_break != -1:
                    end = para_break + 2

                # If no paragraph, look for sentence break
                elif (sent_break := context.rfind(". ", start + char_overlap, end)) != -1:
                    end = sent_break + 2

                # If no sentence, look for newline
                elif (line_break := context.rfind("\n", start + char_overlap, end)) != -1:
                    end = line_break + 1

            # Extract chunk
            chunk = context[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start for next chunk (with overlap)
            start = end - char_overlap

            # Ensure we make progress
            if start <= 0 or (len(chunks) > 0 and start <= len(context) - char_chunk_size):
                if end >= context_len:
                    break

        return chunks

    def _determine_mode(
        self,
        context_tokens: int,
        density: float,
    ) -> str:
        """
        Determine processing mode based on context characteristics.

        Args:
            context_tokens: Number of tokens in context
            density: Information density score (0.0-1.0)

        Returns:
            "iterative" or "structured"
        """
        # Smaller contexts or sparse content -> iterative
        if context_tokens < ITERATIVE_THRESHOLD_TOKENS:
            return "iterative"

        # Dense content -> structured for better summarization
        if density >= STRUCTURED_DENSITY_THRESHOLD:
            return "structured"

        # Larger sparse content -> also structured for efficiency
        return "structured"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses provider's count_tokens if available, otherwise approximates.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        try:
            return self._provider.count_tokens(text)
        except Exception:
            # Fallback: ~4 characters per token
            return len(text) // 4

    def _estimate_density(self, text: str) -> float:
        """
        Estimate information density of text.

        Higher density = more information per token.

        Args:
            text: Text to analyze

        Returns:
            Density score from 0.0 to 1.0
        """
        if not text:
            return 0.0

        # Simple heuristics for density
        lines = text.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]

        if not non_empty_lines:
            return 0.0

        # Factors that indicate density
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        has_structure = any(
            l.strip().startswith(("- ", "* ", "1.", "#", "|"))
            for l in non_empty_lines
        )
        code_blocks = text.count("```")
        data_indicators = sum(
            1 for keyword in ["table", "data", "figure", "chart", "result"]
            if keyword.lower() in text.lower()
        )

        # Calculate density score
        density = 0.3  # Base density

        # Longer lines suggest more content
        if avg_line_length > 100:
            density += 0.2
        elif avg_line_length > 50:
            density += 0.1

        # Structured content is denser
        if has_structure:
            density += 0.15

        # Code and data are dense
        if code_blocks > 0:
            density += 0.15

        if data_indicators > 2:
            density += 0.1

        return min(1.0, density)

    def _get_model_pricing(self, model: str) -> dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model identifier

        Returns:
            Dict with 'input' and 'output' prices per 1K tokens
        """
        # Check for exact match
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Check for partial match
        for model_prefix, pricing in MODEL_PRICING.items():
            if model_prefix in model.lower():
                return pricing

        return MODEL_PRICING["default"]

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """
        Calculate actual cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used

        Returns:
            Cost in USD
        """
        pricing = self._get_model_pricing(model)
        return (
            (input_tokens / 1000) * pricing["input"]
            + (output_tokens / 1000) * pricing["output"]
        )

    def _extract_tag_content(self, text: str, tag: str) -> str:
        """
        Extract content between XML-style tags.

        Args:
            text: Text to search
            tag: Tag name (without brackets)

        Returns:
            Content between tags, or empty string if not found
        """
        import re

        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_key_points(self, text: str) -> list[str]:
        """
        Extract key points from response.

        Args:
            text: Text containing key points

        Returns:
            List of key point strings
        """
        key_points: list[str] = []

        # Try to extract from tag
        content = self._extract_tag_content(text, "key_points")
        if not content:
            # Look for bullet points anywhere
            content = text

        # Extract bullet points
        import re

        bullet_pattern = r"^[-*]\s+(.+)$"
        for line in content.split("\n"):
            match = re.match(bullet_pattern, line.strip())
            if match:
                key_points.append(match.group(1).strip())

        return key_points[:5]  # Limit to 5 key points

    def _extract_relevance(self, text: str) -> float:
        """
        Extract relevance score from response.

        Args:
            text: Text containing relevance score

        Returns:
            Relevance score between 0.0 and 1.0
        """
        import re

        content = self._extract_tag_content(text, "relevance")
        if content:
            # Try to extract number
            match = re.search(r"(\d+\.?\d*)", content)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass

        # Default to medium relevance
        return 0.5

    def supports_streaming(self) -> bool:
        """
        Check if this strategy supports streaming.

        RALPH does not support streaming due to its multi-pass nature.

        Returns:
            False
        """
        return False
