"""
Result Aggregator for ContextFlow.

Combines and synthesizes results from multiple SubAgents.

Features:
- Multiple aggregation strategies
- Conflict resolution
- Quality scoring
- LLM-assisted synthesis

Based on Boris' Best Practices:
- Step 8: Result aggregation from multiple subagents
- Step 13: Quality verification of aggregated results
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextflow.agents.sub_agent import AgentResult
from contextflow.core.types import Message
from contextflow.utils.errors import AgentError
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class AggregationStrategy(Enum):
    """
    Strategies for combining results from multiple agents.

    Each strategy is optimized for different use cases:
    - CONCATENATE: Simple joining, preserves all content
    - MERGE: Removes duplicates, combines similar content
    - SYNTHESIZE: LLM-based intelligent synthesis
    - VOTE: Majority voting for same-task results
    - BEST: Selects highest-scoring result
    - HIERARCHICAL: Creates structured hierarchical summary
    """

    CONCATENATE = "concatenate"
    MERGE = "merge"
    SYNTHESIZE = "synthesize"
    VOTE = "vote"
    BEST = "best"
    HIERARCHICAL = "hierarchical"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AggregationConfig:
    """
    Configuration for result aggregation.

    Attributes:
        strategy: Default aggregation strategy
        min_confidence: Minimum confidence for including results
        max_output_tokens: Maximum tokens in aggregated output
        resolve_conflicts: Whether to detect and resolve conflicts
        include_sources: Whether to track source information
        deduplication_threshold: Similarity threshold for deduplication (0.0-1.0)
        quality_threshold: Minimum quality score for inclusion
    """

    strategy: AggregationStrategy = AggregationStrategy.SYNTHESIZE
    min_confidence: float = 0.7
    max_output_tokens: int = 8000
    resolve_conflicts: bool = True
    include_sources: bool = True
    deduplication_threshold: float = 0.8
    quality_threshold: float = 0.5


@dataclass
class ConflictInfo:
    """
    Information about a detected conflict.

    Attributes:
        source_indices: Indices of conflicting results
        content_a: First conflicting content
        content_b: Second conflicting content
        conflict_type: Type of conflict (contradiction, inconsistency, etc.)
        resolution: How the conflict was resolved
    """

    source_indices: tuple[int, int]
    content_a: str
    content_b: str
    conflict_type: str
    resolution: str | None = None


@dataclass
class AggregatedResult:
    """
    Result from aggregation process.

    Contains the combined content along with metadata about the
    aggregation process and quality metrics.

    Attributes:
        content: Aggregated content
        strategy_used: Strategy that was applied
        source_count: Number of source results
        sources: Original AgentResults (if include_sources=True)
        conflicts_resolved: Number of conflicts detected and resolved
        quality_score: Overall quality score (0.0-1.0)
        token_count: Estimated token count of result
        metadata: Additional metadata
    """

    content: str
    strategy_used: AggregationStrategy
    source_count: int
    sources: list[AgentResult] = field(default_factory=list)
    conflicts_resolved: int = 0
    quality_score: float = 0.0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "strategy_used": self.strategy_used.value,
            "source_count": self.source_count,
            "conflicts_resolved": self.conflicts_resolved,
            "quality_score": round(self.quality_score, 4),
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


# =============================================================================
# Result Aggregator
# =============================================================================


class ResultAggregator:
    """
    Aggregates results from multiple agents.

    Combines outputs from multiple SubAgents using various strategies,
    with support for conflict resolution and quality scoring.

    Usage:
        aggregator = ResultAggregator(provider)

        # Aggregate results with default strategy
        results = [agent1_result, agent2_result, agent3_result]
        final = await aggregator.aggregate(
            results,
            task="Find all security issues",
            strategy=AggregationStrategy.SYNTHESIZE
        )

        print(final.content)
        print(f"Combined from {final.source_count} sources")
        print(f"Quality score: {final.quality_score}")

        # Quick aggregation without LLM
        merged = await aggregator.aggregate(
            results,
            strategy=AggregationStrategy.MERGE
        )
    """

    def __init__(
        self,
        provider: BaseProvider | None = None,
        config: AggregationConfig | None = None,
    ) -> None:
        """
        Initialize ResultAggregator.

        Args:
            provider: LLM provider for synthesis (optional for non-LLM strategies)
            config: Aggregation configuration
        """
        self._provider = provider
        self._config = config or AggregationConfig()

        # Statistics
        self._aggregation_count = 0
        self._total_sources_processed = 0
        self._total_conflicts_resolved = 0

        logger.info(
            "ResultAggregator initialized",
            strategy=self._config.strategy.value,
            has_provider=provider is not None,
            resolve_conflicts=self._config.resolve_conflicts,
        )

    # =========================================================================
    # Main Aggregation Method
    # =========================================================================

    async def aggregate(
        self,
        results: list[AgentResult],
        task: str | None = None,
        strategy: AggregationStrategy | None = None,
    ) -> AggregatedResult:
        """
        Aggregate multiple results.

        Args:
            results: List of AgentResults to aggregate
            task: Original task (for context in synthesis)
            strategy: Override strategy (uses config default if None)

        Returns:
            AggregatedResult with combined content

        Raises:
            AgentError: If aggregation fails
            ValueError: If no results provided
        """
        if not results:
            raise ValueError("No results provided for aggregation")

        start_time = time.time()
        used_strategy = strategy or self._config.strategy

        # Filter by confidence if configured
        filtered_results = self._filter_by_confidence(results)

        if not filtered_results:
            logger.warning("All results filtered out by confidence threshold")
            filtered_results = results  # Fall back to all results

        logger.info(
            "Starting aggregation",
            strategy=used_strategy.value,
            total_results=len(results),
            filtered_results=len(filtered_results),
            has_task=task is not None,
        )

        # Detect conflicts if enabled
        conflicts: list[ConflictInfo] = []
        if self._config.resolve_conflicts and len(filtered_results) > 1:
            conflicts = self._detect_conflicts(filtered_results)
            if conflicts:
                logger.info(
                    "Conflicts detected",
                    count=len(conflicts),
                )

        # Execute strategy
        try:
            content = await self._execute_strategy(filtered_results, task, used_strategy, conflicts)
        except Exception as e:
            logger.error(
                "Aggregation failed",
                strategy=used_strategy.value,
                error=str(e),
            )
            raise AgentError(
                message=f"Aggregation failed: {e}",
                details={"strategy": used_strategy.value},
            )

        # Deduplicate if not already handled by strategy
        if used_strategy not in (AggregationStrategy.SYNTHESIZE, AggregationStrategy.VOTE):
            content = self._deduplicate(content)

        # Calculate quality score
        quality_score = self._calculate_quality_score(filtered_results, content)

        # Build result
        execution_time = time.time() - start_time
        self._aggregation_count += 1
        self._total_sources_processed += len(results)
        self._total_conflicts_resolved += len(conflicts)

        result = AggregatedResult(
            content=content,
            strategy_used=used_strategy,
            source_count=len(filtered_results),
            sources=filtered_results if self._config.include_sources else [],
            conflicts_resolved=len(conflicts),
            quality_score=quality_score,
            token_count=self._estimate_tokens(content),
            metadata={
                "execution_time": round(execution_time, 3),
                "original_result_count": len(results),
                "filtered_result_count": len(filtered_results),
                "conflicts_detected": len(conflicts),
                "task_provided": task is not None,
            },
        )

        logger.info(
            "Aggregation completed",
            strategy=used_strategy.value,
            quality_score=round(quality_score, 3),
            token_count=result.token_count,
            execution_time=round(execution_time, 3),
        )

        return result

    async def _execute_strategy(
        self,
        results: list[AgentResult],
        task: str | None,
        strategy: AggregationStrategy,
        conflicts: list[ConflictInfo],
    ) -> str:
        """Execute the selected aggregation strategy."""
        # Handle conflicts first if synthesis is used
        conflict_context = ""
        if conflicts and strategy == AggregationStrategy.SYNTHESIZE:
            conflict_context = await self._resolve_conflicts(conflicts, task or "")

        if strategy == AggregationStrategy.CONCATENATE:
            return await self._concatenate(results)
        elif strategy == AggregationStrategy.MERGE:
            return await self._merge(results)
        elif strategy == AggregationStrategy.SYNTHESIZE:
            if not self._provider:
                logger.warning("No provider for synthesis, falling back to merge")
                return await self._merge(results)
            return await self._synthesize(results, task or "", conflict_context)
        elif strategy == AggregationStrategy.VOTE:
            return await self._vote(results)
        elif strategy == AggregationStrategy.BEST:
            return await self._select_best(results)
        elif strategy == AggregationStrategy.HIERARCHICAL:
            if not self._provider:
                logger.warning("No provider for hierarchical, falling back to concatenate")
                return await self._concatenate(results)
            return await self._hierarchical(results, task or "")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # =========================================================================
    # Strategy Implementations
    # =========================================================================

    async def _concatenate(self, results: list[AgentResult]) -> str:
        """
        Simple concatenation with separators.

        Joins all results with clear separators, preserving all content.
        Best for: Preserving complete outputs from all agents.
        """
        parts: list[str] = []

        for i, result in enumerate(results):
            if not result.success or not result.output.strip():
                continue

            header = f"=== Source {i + 1}: {result.role.value} ==="
            parts.append(header)
            parts.append(result.output.strip())
            parts.append("")  # Empty line separator

        return "\n".join(parts).strip()

    async def _merge(self, results: list[AgentResult]) -> str:
        """
        Merge similar content, remove duplicates.

        Combines results while eliminating redundant information.
        Best for: Combining results that may have overlapping content.
        """
        # Collect all unique paragraphs/sections
        seen_hashes: set[str] = set()
        unique_content: list[str] = []

        for result in results:
            if not result.success or not result.output.strip():
                continue

            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(result.output)

            for para in paragraphs:
                para_stripped = para.strip()
                if not para_stripped:
                    continue

                # Hash for deduplication
                para_hash = self._content_hash(para_stripped)

                if para_hash not in seen_hashes:
                    seen_hashes.add(para_hash)
                    unique_content.append(para_stripped)

        # Join unique content
        return "\n\n".join(unique_content)

    async def _synthesize(
        self,
        results: list[AgentResult],
        task: str,
        conflict_context: str = "",
    ) -> str:
        """
        LLM-based synthesis of results.

        Uses the LLM to create a coherent summary that integrates
        all results intelligently.
        Best for: Creating polished, coherent final outputs.
        """
        if not self._provider:
            raise AgentError(message="Provider required for synthesis strategy")

        # Build synthesis prompt
        source_texts: list[str] = []
        for i, result in enumerate(results):
            if not result.success or not result.output.strip():
                continue

            source_texts.append(
                f"### Source {i + 1} ({result.role.value})\n{result.output.strip()}"
            )

        if not source_texts:
            return ""

        sources_combined = "\n\n".join(source_texts)

        system_prompt = """You are an expert at synthesizing information from multiple sources.
Your task is to combine the provided source materials into a single coherent response.

Guidelines:
- Integrate information without redundancy
- Maintain accuracy from all sources
- Resolve any contradictions by noting them or choosing the most supported view
- Create a well-structured, flowing narrative
- Preserve important details and nuances
- Do not add information not present in the sources"""

        user_content_parts = [f"## Original Task\n{task}\n"] if task else []
        user_content_parts.append(f"## Source Materials\n\n{sources_combined}")

        if conflict_context:
            user_content_parts.append(f"\n## Conflict Resolutions\n{conflict_context}")

        user_content_parts.append(
            "\n## Instructions\nSynthesize the above sources into a single, "
            "comprehensive response that addresses the original task."
        )

        user_content = "\n".join(user_content_parts)

        response = await self._provider.complete(
            messages=[Message(role="user", content=user_content)],
            system=system_prompt,
            max_tokens=self._config.max_output_tokens,
            temperature=0.5,
        )

        return response.content

    async def _vote(self, results: list[AgentResult]) -> str:
        """
        Majority voting for same-task results.

        Selects the most common response when multiple agents
        produce similar outputs.
        Best for: Tasks with discrete answers where consensus matters.
        """
        if not results:
            return ""

        # Normalize and count responses
        response_counts: Counter[str] = Counter()
        response_map: dict[str, str] = {}  # normalized -> original

        for result in results:
            if not result.success or not result.output.strip():
                continue

            normalized = self._normalize_for_voting(result.output)
            response_counts[normalized] += 1

            # Keep first occurrence as representative
            if normalized not in response_map:
                response_map[normalized] = result.output.strip()

        if not response_counts:
            return ""

        # Get most common response
        most_common = response_counts.most_common(1)[0]
        winning_normalized, vote_count = most_common

        logger.debug(
            "Voting result",
            winning_votes=vote_count,
            total_votes=sum(response_counts.values()),
            unique_responses=len(response_counts),
        )

        return response_map[winning_normalized]

    async def _select_best(self, results: list[AgentResult]) -> str:
        """
        Select best result by verification score.

        Picks the single highest-quality result based on
        verification scores.
        Best for: When only the highest-quality response is needed.
        """
        if not results:
            return ""

        best_result: AgentResult | None = None
        best_score = -1.0

        for result in results:
            if not result.success or not result.output.strip():
                continue

            # Calculate score
            score = self._calculate_result_score(result)

            if score > best_score:
                best_score = score
                best_result = result

        if best_result:
            logger.debug(
                "Best result selected",
                agent_id=best_result.agent_id,
                score=round(best_score, 3),
            )
            return best_result.output.strip()

        return ""

    async def _hierarchical(
        self,
        results: list[AgentResult],
        task: str,
    ) -> str:
        """
        Build hierarchical summary.

        Creates a structured output with main points and supporting
        details in a hierarchical format.
        Best for: Complex topics requiring organization.
        """
        if not self._provider:
            raise AgentError(message="Provider required for hierarchical strategy")

        # Combine all content
        all_content: list[str] = []
        for result in results:
            if not result.success or not result.output.strip():
                continue
            all_content.append(result.output.strip())

        if not all_content:
            return ""

        combined = "\n\n---\n\n".join(all_content)

        system_prompt = """You are an expert at organizing information hierarchically.
Your task is to structure the provided content into a clear hierarchy.

Guidelines:
- Create clear main sections with descriptive headings
- Use subsections for detailed points
- Use bullet points for lists of related items
- Maintain logical flow from general to specific
- Preserve all important information
- Use markdown formatting"""

        user_content = f"""## Original Task
{task}

## Content to Organize

{combined}

## Instructions
Create a hierarchical summary with:
1. An executive summary (2-3 sentences)
2. Main sections with clear headings
3. Subsections and bullet points as needed
4. Key takeaways at the end"""

        response = await self._provider.complete(
            messages=[Message(role="user", content=user_content)],
            system=system_prompt,
            max_tokens=self._config.max_output_tokens,
            temperature=0.4,
        )

        return response.content

    # =========================================================================
    # Conflict Detection and Resolution
    # =========================================================================

    def _detect_conflicts(
        self,
        results: list[AgentResult],
    ) -> list[ConflictInfo]:
        """
        Detect conflicting information between results.

        Looks for contradictions, inconsistencies, and opposing
        statements across different agent outputs.
        """
        conflicts: list[ConflictInfo] = []

        # Extract key statements from each result
        statements_by_result: list[list[str]] = []
        for result in results:
            if result.success and result.output.strip():
                statements = self._extract_key_statements(result.output)
                statements_by_result.append(statements)
            else:
                statements_by_result.append([])

        # Compare statements between results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                for stmt_a in statements_by_result[i]:
                    for stmt_b in statements_by_result[j]:
                        conflict_type = self._check_conflict(stmt_a, stmt_b)
                        if conflict_type:
                            conflicts.append(
                                ConflictInfo(
                                    source_indices=(i, j),
                                    content_a=stmt_a,
                                    content_b=stmt_b,
                                    conflict_type=conflict_type,
                                )
                            )

        return conflicts[:10]  # Limit to top 10 conflicts

    def _extract_key_statements(self, text: str) -> list[str]:
        """Extract key statements from text for conflict detection."""
        statements: list[str] = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 500:
                # Include sentences that make assertions
                if any(
                    indicator in sentence.lower()
                    for indicator in [
                        "is",
                        "are",
                        "was",
                        "were",
                        "should",
                        "must",
                        "will",
                        "can",
                        "cannot",
                        "always",
                        "never",
                        "all",
                        "none",
                        "only",
                        "best",
                        "worst",
                    ]
                ):
                    statements.append(sentence)

        return statements[:20]  # Limit statements per result

    def _check_conflict(self, stmt_a: str, stmt_b: str) -> str | None:
        """Check if two statements conflict."""
        a_lower = stmt_a.lower()
        b_lower = stmt_b.lower()

        # Check for direct negation patterns
        negation_pairs = [
            ("is", "is not"),
            ("are", "are not"),
            ("can", "cannot"),
            ("should", "should not"),
            ("will", "will not"),
            ("always", "never"),
            ("all", "none"),
        ]

        for pos, neg in negation_pairs:
            if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                # Check if they're about the same subject (simple heuristic)
                a_words = set(a_lower.split())
                b_words = set(b_lower.split())
                overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
                if overlap > 0.3:
                    return "contradiction"

        # Check for numeric conflicts (e.g., "5 issues" vs "3 issues")
        nums_a = re.findall(r"\b(\d+)\b", stmt_a)
        nums_b = re.findall(r"\b(\d+)\b", stmt_b)
        if nums_a and nums_b:
            # Check if same context but different numbers
            a_words = set(a_lower.split()) - set(nums_a)
            b_words = set(b_lower.split()) - set(nums_b)
            overlap = len(a_words & b_words) / max(len(a_words), len(b_words), 1)
            if overlap > 0.5 and nums_a[0] != nums_b[0]:
                return "numeric_inconsistency"

        return None

    async def _resolve_conflicts(
        self,
        conflicts: list[ConflictInfo],
        task: str,
    ) -> str:
        """
        Resolve detected conflicts using LLM.

        Creates context about how conflicts should be handled
        in the synthesis.
        """
        if not conflicts or not self._provider:
            return ""

        conflict_descriptions: list[str] = []
        for i, conflict in enumerate(conflicts[:5]):  # Limit to top 5
            conflict_descriptions.append(
                f"{i + 1}. {conflict.conflict_type.replace('_', ' ').title()}:\n"
                f"   - View A: {conflict.content_a[:200]}...\n"
                f"   - View B: {conflict.content_b[:200]}..."
            )

        prompt = f"""The following conflicts were detected in the source materials:

{chr(10).join(conflict_descriptions)}

Task context: {task}

For each conflict, provide a brief resolution guidance (which view to prefer, or how to present both)."""

        response = await self._provider.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are an expert at resolving conflicting information. Be concise.",
            max_tokens=500,
            temperature=0.3,
        )

        return response.content

    # =========================================================================
    # Quality Scoring
    # =========================================================================

    def _calculate_quality_score(
        self,
        results: list[AgentResult],
        aggregated: str,
    ) -> float:
        """
        Calculate quality score for aggregation.

        Factors:
        - Source verification scores
        - Content coverage
        - Output coherence (length-based heuristic)
        """
        if not results or not aggregated:
            return 0.0

        scores: list[float] = []

        # Factor 1: Average source verification score
        verification_scores: list[float] = []
        for result in results:
            if result.verification_result:
                verification_scores.append(result.verification_result.confidence)
        if verification_scores:
            scores.append(sum(verification_scores) / len(verification_scores))

        # Factor 2: Content coverage
        source_content_length = sum(len(r.output) for r in results if r.success and r.output)
        if source_content_length > 0:
            coverage = min(len(aggregated) / source_content_length, 1.0)
            # Optimal coverage around 40-80%
            if 0.3 < coverage < 0.9:
                coverage_score = 1.0 - abs(coverage - 0.6) / 0.6
            else:
                coverage_score = coverage if coverage < 0.3 else 0.5
            scores.append(coverage_score)

        # Factor 3: Success rate of sources
        success_rate = sum(1 for r in results if r.success) / len(results)
        scores.append(success_rate)

        # Factor 4: Output has reasonable length
        if len(aggregated) > 100:
            length_score = min(len(aggregated) / 1000, 1.0)
            scores.append(length_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_result_score(self, result: AgentResult) -> float:
        """Calculate score for a single result."""
        score = 0.0

        # Verification score
        if result.verification_result:
            score += result.verification_result.confidence * 0.5

        # Success factor
        if result.success:
            score += 0.3

        # Output length factor (prefer substantial outputs)
        if result.output:
            length_factor = min(len(result.output) / 2000, 1.0) * 0.2
            score += length_factor

        return score

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _filter_by_confidence(
        self,
        results: list[AgentResult],
    ) -> list[AgentResult]:
        """Filter results by minimum confidence threshold."""
        filtered: list[AgentResult] = []

        for result in results:
            if not result.success:
                continue

            # Check verification confidence
            if result.verification_result:
                if result.verification_result.confidence >= self._config.min_confidence:
                    filtered.append(result)
            else:
                # No verification, include if successful
                filtered.append(result)

        return filtered

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or markdown headers
        paragraphs = re.split(r"\n\n+|(?=^#+\s)", text, flags=re.MULTILINE)
        return [p.strip() for p in paragraphs if p.strip()]

    def _content_hash(self, text: str) -> str:
        """Create hash for content deduplication."""
        # Normalize before hashing
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _normalize_for_voting(self, text: str) -> str:
        """Normalize text for voting comparison."""
        # Remove extra whitespace, lowercase, remove punctuation
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized

    def _deduplicate(self, text: str) -> str:
        """Remove duplicate content from text."""
        paragraphs = self._split_into_paragraphs(text)
        seen_hashes: set[str] = set()
        unique: list[str] = []

        for para in paragraphs:
            para_hash = self._content_hash(para)
            if para_hash not in seen_hashes:
                seen_hashes.add(para_hash)
                unique.append(para)

        return "\n\n".join(unique)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self._provider:
            try:
                return self._provider.count_tokens(text)
            except Exception:
                pass
        # Fallback: ~4 chars per token
        return len(text) // 4

    # =========================================================================
    # Statistics and Configuration
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        return {
            "aggregation_count": self._aggregation_count,
            "total_sources_processed": self._total_sources_processed,
            "total_conflicts_resolved": self._total_conflicts_resolved,
            "avg_sources_per_aggregation": (
                self._total_sources_processed / self._aggregation_count
                if self._aggregation_count > 0
                else 0.0
            ),
            "config": {
                "strategy": self._config.strategy.value,
                "min_confidence": self._config.min_confidence,
                "max_output_tokens": self._config.max_output_tokens,
                "resolve_conflicts": self._config.resolve_conflicts,
            },
        }

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug("Config updated", key=key, value=value)

    def __repr__(self) -> str:
        return (
            f"ResultAggregator("
            f"strategy={self._config.strategy.value!r}, "
            f"has_provider={self._provider is not None})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def aggregate_results(
    results: list[AgentResult],
    provider: BaseProvider | None = None,
    strategy: AggregationStrategy = AggregationStrategy.MERGE,
) -> str:
    """
    Quick aggregation without full setup.

    Args:
        results: List of AgentResults to aggregate
        provider: Optional LLM provider (required for synthesis)
        strategy: Aggregation strategy

    Returns:
        Aggregated content string
    """
    aggregator = ResultAggregator(provider=provider)
    result = await aggregator.aggregate(results, strategy=strategy)
    return result.content


async def synthesize_with_llm(
    results: list[AgentResult],
    task: str,
    provider: BaseProvider,
) -> str:
    """
    Direct LLM synthesis of results.

    Args:
        results: List of AgentResults to synthesize
        task: Original task description
        provider: LLM provider

    Returns:
        Synthesized content string
    """
    aggregator = ResultAggregator(provider=provider)
    result = await aggregator.aggregate(
        results,
        task=task,
        strategy=AggregationStrategy.SYNTHESIZE,
    )
    return result.content


async def select_best_result(
    results: list[AgentResult],
) -> AgentResult | None:
    """
    Select the best result from a list.

    Args:
        results: List of AgentResults to evaluate

    Returns:
        Best AgentResult or None if all failed
    """
    aggregator = ResultAggregator()
    # Use internal scoring to find best
    best: AgentResult | None = None
    best_score = -1.0

    for result in results:
        if result.success:
            score = aggregator._calculate_result_score(result)
            if score > best_score:
                best_score = score
                best = result

    return best


async def merge_unique_content(
    results: list[AgentResult],
) -> str:
    """
    Merge results keeping only unique content.

    Args:
        results: List of AgentResults to merge

    Returns:
        Merged content with duplicates removed
    """
    aggregator = ResultAggregator()
    result = await aggregator.aggregate(
        results,
        strategy=AggregationStrategy.MERGE,
    )
    return result.content


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "AggregationStrategy",
    # Data Classes
    "AggregationConfig",
    "AggregatedResult",
    "ConflictInfo",
    # Main Class
    "ResultAggregator",
    # Convenience Functions
    "aggregate_results",
    "synthesize_with_llm",
    "select_best_result",
    "merge_unique_content",
]
