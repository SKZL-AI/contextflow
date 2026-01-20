"""
Context Analyzer - Intelligent context analysis for strategy selection.

Provides comprehensive analysis of documents/context to determine:
1. Token count estimation
2. Information density scoring
3. Complexity level assessment
4. Optimal strategy recommendation
5. Cost estimation

Supports both heuristic-based analysis (fast) and LLM-assisted analysis
(more accurate) through SubAgent integration.

Based on Strategy Matrix:
Token Count | Density | Complexity | Strategy
<10K        | *       | Simple     | GSD_DIRECT
<10K        | *       | Complex    | GSD_GUIDED
10K-50K     | <0.5    | *          | RALPH_ITERATIVE
10K-50K     | >=0.5   | *          | RALPH_STRUCTURED
50K-100K    | <0.7    | *          | RALPH_STRUCTURED
50K-100K    | >=0.7   | *          | RLM_BASIC
>100K       | *       | *          | RLM_FULL
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from contextflow.core.types import StrategyType
from contextflow.utils.errors import (
    ContextFlowError,
    ValidationError,
)
from contextflow.utils.logging import ProviderLogger, get_logger
from contextflow.utils.tokens import (
    MODEL_CONTEXT_LIMITS,
    MODEL_PRICING,
    TokenEstimator,
)

if TYPE_CHECKING:
    from contextflow.agents.sub_agent import SubAgent
    from contextflow.providers.base import BaseProvider

logger = get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ComplexityLevel(str, Enum):
    """Task complexity levels for analysis."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXHAUSTIVE = "exhaustive"


class DensityLevel(str, Enum):
    """Information density levels."""

    SPARSE = "sparse"  # < 0.3 - lots of whitespace, comments
    MEDIUM = "medium"  # 0.3-0.7 - normal code/documentation
    DENSE = "dense"  # > 0.7 - compressed data, minified code


class ContentType(str, Enum):
    """Detected content types."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    PROSE = "prose"
    DATA = "data"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# Complexity indicator keywords (from router.py patterns)
COMPLEXITY_INDICATORS: dict[str, list[str]] = {
    "exhaustive": [
        "all", "every", "exhaustive", "comprehensive", "complete",
        "entire", "thorough", "full", "detailed analysis", "in-depth",
        "everything", "nothing missed", "cover all", "leave nothing out",
    ],
    "complex": [
        "analyze", "compare", "contrast", "evaluate", "synthesize",
        "explain why", "implications", "how does this relate",
        "critique", "assess", "investigate", "examine",
        "multiple aspects", "step by step", "elaborate", "justify",
        "argue", "prove", "demonstrate", "connect", "patterns",
        "relationship between", "cause and effect", "impact of",
    ],
    "moderate": [
        "summarize", "describe", "explain", "outline", "overview",
        "key points", "main ideas", "important", "significant",
        "breakdown", "structure", "organize", "categorize",
    ],
    "simple": [
        "what is", "who is", "when was", "where is", "list",
        "name", "define", "identify", "find", "extract",
        "yes or no", "true or false", "how many", "which one",
        "simple", "quick", "brief", "short",
    ],
}

# Code indicators for content type detection
CODE_INDICATORS: list[str] = [
    "```", "def ", "class ", "import ", "function ",
    "const ", "var ", "let ", "return ", "if (", "for (",
    "->", "=>", "&&", "||", "==", "!=", "async ", "await ",
    "public ", "private ", "protected ", "static ",
    "interface ", "struct ", "enum ", "trait ",
]

# Data indicators
DATA_INDICATORS: list[str] = [
    "{", "[", "<", ":", ",",
    '":', "': ", "null", "true", "false",
]


# =============================================================================
# Data Classes and Pydantic Models
# =============================================================================


class ChunkSuggestion(BaseModel):
    """Suggested chunking parameters."""

    size: int = Field(default=4000, description="Recommended chunk size in tokens")
    overlap: int = Field(default=500, description="Recommended overlap between chunks")
    total_chunks: int = Field(default=1, description="Estimated number of chunks")


class CostEstimate(BaseModel):
    """Cost estimates for different models."""

    model: str = Field(description="Model name")
    input_cost: float = Field(description="Estimated input cost in USD")
    output_cost: float = Field(description="Estimated output cost in USD")
    total_cost: float = Field(description="Total estimated cost in USD")


@dataclass
class ContextAnalysis:
    """
    Result of context analysis.

    Contains comprehensive analysis including token count, density,
    complexity, strategy recommendation, and cost estimates.

    Attributes:
        token_count: Total tokens in the context
        density: Information density score (0.0-1.0)
        density_level: Categorical density level
        complexity: Task complexity level
        content_type: Detected content type
        recommended_strategy: Best strategy for this context
        reasoning: Explanation for the recommendation
        alternative_strategies: Other viable strategies
        estimated_costs: Cost estimates per model
        chunk_suggestion: Recommended chunking parameters
        warnings: Any warnings about the analysis
        metadata: Additional analysis metadata
        analysis_time_ms: Time taken for analysis in milliseconds
    """

    token_count: int
    density: float
    density_level: DensityLevel
    complexity: ComplexityLevel
    content_type: ContentType
    recommended_strategy: StrategyType
    reasoning: str
    alternative_strategies: list[StrategyType] = field(default_factory=list)
    estimated_costs: dict[str, CostEstimate] = field(default_factory=dict)
    chunk_suggestion: ChunkSuggestion | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    analysis_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "token_count": self.token_count,
            "density": round(self.density, 3),
            "density_level": self.density_level.value,
            "complexity": self.complexity.value,
            "content_type": self.content_type.value,
            "recommended_strategy": self.recommended_strategy.value,
            "reasoning": self.reasoning,
            "alternative_strategies": [s.value for s in self.alternative_strategies],
            "estimated_costs": {
                k: v.model_dump() for k, v in self.estimated_costs.items()
            },
            "chunk_suggestion": (
                self.chunk_suggestion.model_dump() if self.chunk_suggestion else None
            ),
            "warnings": self.warnings,
            "metadata": self.metadata,
            "analysis_time_ms": round(self.analysis_time_ms, 2),
        }


@dataclass
class AnalyzerConfig:
    """
    Configuration for ContextAnalyzer.

    Attributes:
        gsd_max_tokens: Maximum tokens for GSD strategy
        ralph_min_tokens: Minimum tokens for RALPH strategy
        ralph_max_tokens: Maximum tokens for RALPH strategy
        rlm_min_tokens: Minimum tokens for RLM strategy
        density_threshold_low: Threshold between sparse and medium
        density_threshold_high: Threshold between medium and dense
        enable_llm_analysis: Enable LLM-assisted analysis
        default_model: Default model for cost estimation
        chunk_target_size: Target chunk size for suggestions
        chunk_overlap_ratio: Overlap ratio for chunking
    """

    gsd_max_tokens: int = 10_000
    ralph_min_tokens: int = 10_000
    ralph_max_tokens: int = 100_000
    rlm_min_tokens: int = 100_000
    density_threshold_low: float = 0.3
    density_threshold_high: float = 0.7
    enable_llm_analysis: bool = False
    default_model: str = "claude-3-5-sonnet-20241022"
    chunk_target_size: int = 4000
    chunk_overlap_ratio: float = 0.125  # 12.5% overlap

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gsd_max_tokens": self.gsd_max_tokens,
            "ralph_min_tokens": self.ralph_min_tokens,
            "ralph_max_tokens": self.ralph_max_tokens,
            "rlm_min_tokens": self.rlm_min_tokens,
            "density_threshold_low": self.density_threshold_low,
            "density_threshold_high": self.density_threshold_high,
            "enable_llm_analysis": self.enable_llm_analysis,
            "default_model": self.default_model,
            "chunk_target_size": self.chunk_target_size,
            "chunk_overlap_ratio": self.chunk_overlap_ratio,
        }


# =============================================================================
# Context Analyzer Class
# =============================================================================


class ContextAnalyzer:
    """
    Intelligent context analyzer for strategy selection.

    Analyzes documents to determine optimal processing strategy based on:
    - Token count estimation
    - Information density scoring
    - Complexity level assessment
    - Content type detection

    Supports two modes:
    1. Heuristic analysis (fast, no LLM calls)
    2. LLM-assisted analysis (more accurate, requires provider)

    Usage:
        # Basic usage with heuristics only
        analyzer = ContextAnalyzer()
        analysis = analyzer.analyze(task="Summarize this", context=document)
        print(f"Recommended: {analysis.recommended_strategy}")

        # With LLM-assisted analysis
        analyzer = ContextAnalyzer(provider=claude_provider)
        analysis = await analyzer.analyze_with_llm(
            task="Analyze codebase",
            context=code_content
        )

        # Access detailed results
        print(f"Token count: {analysis.token_count}")
        print(f"Density: {analysis.density_level.value}")
        print(f"Complexity: {analysis.complexity.value}")
        print(f"Strategy: {analysis.recommended_strategy.value}")
        print(f"Reasoning: {analysis.reasoning}")
    """

    def __init__(
        self,
        provider: BaseProvider | None = None,
        config: AnalyzerConfig | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        """
        Initialize ContextAnalyzer.

        Args:
            provider: Optional LLM provider for LLM-assisted analysis
            config: Analyzer configuration (uses defaults if None)
            token_estimator: Token counter (creates default if None)
        """
        self._provider = provider
        self._config = config or AnalyzerConfig()
        self._token_estimator = token_estimator or TokenEstimator()
        self._logger = ProviderLogger("analyzer")

        # SubAgent for LLM-based analysis (lazy initialized)
        self._analysis_agent: SubAgent | None = None

        # Analysis cache for repeated queries
        self._cache: dict[int, ContextAnalysis] = {}
        self._cache_max_size = 100

        logger.info(
            "ContextAnalyzer initialized",
            has_provider=provider is not None,
            config=self._config.to_dict(),
        )

    @property
    def config(self) -> AnalyzerConfig:
        """Get analyzer configuration."""
        return self._config

    @property
    def has_provider(self) -> bool:
        """Check if LLM provider is available."""
        return self._provider is not None

    # =========================================================================
    # Main Analysis Methods
    # =========================================================================

    def analyze(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        use_cache: bool = True,
    ) -> ContextAnalysis:
        """
        Analyze context using heuristics (synchronous).

        Fast analysis method that doesn't require LLM calls.
        Uses linguistic patterns and statistical measures.

        Args:
            task: Task description to analyze
            context: Context/document to analyze
            constraints: Optional task constraints
            use_cache: Whether to use/update analysis cache

        Returns:
            ContextAnalysis with strategy recommendation

        Raises:
            ValidationError: If inputs are invalid
        """
        start_time = time.time()

        # Validate inputs
        if not task or not task.strip():
            raise ValidationError("Task cannot be empty", field="task")
        if not context:
            raise ValidationError("Context cannot be empty", field="context")

        # Check cache
        cache_key = hash(f"{task}:{context[:1000]}")
        if use_cache and cache_key in self._cache:
            logger.debug("Returning cached analysis")
            return self._cache[cache_key]

        logger.debug(
            "Starting heuristic analysis",
            task_length=len(task),
            context_length=len(context),
        )

        # Perform analysis
        token_count = self._count_tokens(task + context)
        density = self._estimate_density(context)
        density_level = self._classify_density(density)
        complexity = self._assess_complexity(task, constraints)
        content_type = self._detect_content_type(context)

        # Select strategy
        strategy, reasoning = self._select_strategy(
            token_count, density, complexity
        )

        # Get alternatives
        alternatives = self._get_alternative_strategies(strategy, token_count)

        # Estimate costs
        cost_estimates = self._estimate_costs(token_count, strategy)

        # Generate chunk suggestion
        chunk_suggestion = self._suggest_chunking(token_count, content_type)

        # Generate warnings
        warnings = self._generate_warnings(token_count, density, complexity)

        analysis_time = (time.time() - start_time) * 1000

        analysis = ContextAnalysis(
            token_count=token_count,
            density=density,
            density_level=density_level,
            complexity=complexity,
            content_type=content_type,
            recommended_strategy=strategy,
            reasoning=reasoning,
            alternative_strategies=alternatives,
            estimated_costs=cost_estimates,
            chunk_suggestion=chunk_suggestion,
            warnings=warnings,
            metadata={
                "task_length": len(task),
                "context_length": len(context),
                "has_constraints": constraints is not None,
                "constraints_count": len(constraints) if constraints else 0,
                "analysis_method": "heuristic",
            },
            analysis_time_ms=analysis_time,
        )

        # Update cache
        if use_cache:
            self._update_cache(cache_key, analysis)

        logger.info(
            "Analysis complete",
            token_count=token_count,
            density=round(density, 3),
            complexity=complexity.value,
            strategy=strategy.value,
            analysis_time_ms=round(analysis_time, 2),
        )

        return analysis

    async def analyze_async(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        use_llm: bool = False,
    ) -> ContextAnalysis:
        """
        Analyze context asynchronously.

        Can optionally use LLM for more accurate analysis.

        Args:
            task: Task description
            context: Context to analyze
            constraints: Optional constraints
            use_llm: Whether to use LLM-assisted analysis

        Returns:
            ContextAnalysis with recommendation

        Raises:
            ContextFlowError: If LLM analysis requested but no provider
        """
        if use_llm and self._provider:
            return await self._analyze_with_llm(task, context, constraints)

        # Run synchronous analysis in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.analyze(task, context, constraints),
        )

    async def _analyze_with_llm(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> ContextAnalysis:
        """
        Perform LLM-assisted analysis for more accurate results.

        Uses a SubAgent with specialized prompt to analyze the content
        and provide more nuanced assessment.

        Args:
            task: Task description
            context: Context to analyze
            constraints: Optional constraints

        Returns:
            ContextAnalysis with LLM-enhanced reasoning
        """
        if not self._provider:
            raise ContextFlowError(
                "LLM analysis requires a provider",
                details={"hint": "Initialize ContextAnalyzer with a provider"},
            )

        start_time = time.time()

        # Get base heuristic analysis first
        base_analysis = self.analyze(task, context, constraints, use_cache=False)

        # Initialize analysis agent if needed
        if self._analysis_agent is None:
            self._analysis_agent = await self._create_analysis_agent()

        # Build analysis prompt
        analysis_prompt = self._build_llm_analysis_prompt(
            task=task,
            context_preview=context[:2000],  # Sample for efficiency
            base_analysis=base_analysis,
            constraints=constraints,
        )

        logger.debug("Executing LLM analysis", prompt_length=len(analysis_prompt))

        try:
            # Execute LLM analysis
            result = await self._analysis_agent.execute(
                task=analysis_prompt,
                context=None,  # Context is in the prompt
            )

            if result.success:
                # Parse LLM response and enhance analysis
                enhanced_analysis = self._parse_llm_analysis(
                    result.output,
                    base_analysis,
                )
                enhanced_analysis.analysis_time_ms = (time.time() - start_time) * 1000
                enhanced_analysis.metadata["analysis_method"] = "llm_assisted"
                enhanced_analysis.metadata["llm_tokens_used"] = result.token_usage.get(
                    "total_tokens", 0
                )

                logger.info(
                    "LLM analysis complete",
                    strategy=enhanced_analysis.recommended_strategy.value,
                    llm_tokens=result.token_usage.get("total_tokens", 0),
                )

                return enhanced_analysis
            else:
                # Fall back to heuristic analysis
                logger.warning(
                    "LLM analysis failed, using heuristic",
                    error=result.error,
                )
                base_analysis.warnings.append(
                    f"LLM analysis failed: {result.error}. Using heuristic analysis."
                )
                return base_analysis

        except Exception as e:
            logger.error("LLM analysis error", error=str(e))
            base_analysis.warnings.append(f"LLM analysis error: {str(e)}")
            return base_analysis

    # =========================================================================
    # Core Analysis Methods
    # =========================================================================

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        try:
            return self._token_estimator.count_tokens(text)
        except Exception as e:
            logger.warning("Token counting failed, using estimate", error=str(e))
            # Fallback: ~4 chars per token
            return len(text) // 4

    def _estimate_density(self, text: str) -> float:
        """
        Estimate information density of text.

        Higher density indicates more information per token.
        Uses multiple factors to calculate density score.

        Factors:
        - Whitespace ratio (less = denser)
        - Average word length (longer = denser)
        - Code content (code is denser)
        - Structure indicators (tables, lists)
        - Numeric content (more numbers = denser)
        - Unique word ratio (higher = less repetition = denser)

        Args:
            text: Text to analyze

        Returns:
            Density score from 0.0 (sparse) to 1.0 (dense)
        """
        if not text or len(text) < 10:
            return 0.0

        density_score = 0.0
        factors_weight = 0.0

        # Factor 1: Whitespace ratio (weight: 0.15)
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / len(text)
        # Less whitespace = denser
        whitespace_factor = max(0.0, 1.0 - (whitespace_ratio * 3))
        density_score += whitespace_factor * 0.15
        factors_weight += 0.15

        # Factor 2: Average word length (weight: 0.15)
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            # Longer words = denser vocabulary
            word_length_factor = min(1.0, avg_word_length / 8.0)
            density_score += word_length_factor * 0.15
            factors_weight += 0.15

        # Factor 3: Code content detection (weight: 0.2)
        code_matches = sum(1 for ind in CODE_INDICATORS if ind in text)
        code_factor = min(1.0, code_matches / 5.0)
        density_score += code_factor * 0.2
        factors_weight += 0.2

        # Factor 4: Structure indicators (weight: 0.15)
        structure_indicators = [
            "|", "- ", "* ", "1.", "##", "###",
            "<table", "<tr>", "<td>", "| ---",
        ]
        structure_matches = sum(text.count(ind) for ind in structure_indicators)
        structure_factor = min(1.0, structure_matches / 20.0)
        density_score += structure_factor * 0.15
        factors_weight += 0.15

        # Factor 5: Numeric content (weight: 0.15)
        numeric_chars = sum(1 for c in text if c.isdigit())
        numeric_ratio = numeric_chars / len(text)
        numeric_factor = min(1.0, numeric_ratio * 10)
        density_score += numeric_factor * 0.15
        factors_weight += 0.15

        # Factor 6: Unique word ratio (weight: 0.2)
        if words:
            unique_words = set(w.lower() for w in words)
            unique_ratio = len(unique_words) / len(words)
            unique_factor = unique_ratio
            density_score += unique_factor * 0.2
            factors_weight += 0.2

        # Normalize
        if factors_weight > 0:
            final_density = density_score / factors_weight
        else:
            final_density = 0.3

        return min(1.0, max(0.0, final_density))

    def _classify_density(self, density: float) -> DensityLevel:
        """
        Classify density score into categorical level.

        Args:
            density: Density score (0.0-1.0)

        Returns:
            DensityLevel enum value
        """
        if density < self._config.density_threshold_low:
            return DensityLevel.SPARSE
        elif density < self._config.density_threshold_high:
            return DensityLevel.MEDIUM
        else:
            return DensityLevel.DENSE

    def _assess_complexity(
        self,
        task: str,
        constraints: list[str] | None = None,
    ) -> ComplexityLevel:
        """
        Assess task complexity based on linguistic indicators.

        Analyzes task description and constraints to determine
        how complex the required processing will be.

        Args:
            task: Task description
            constraints: Optional constraints

        Returns:
            ComplexityLevel enum value
        """
        task_lower = task.lower()

        # Count matches for each complexity level
        scores: dict[str, int] = {
            "exhaustive": 0,
            "complex": 0,
            "moderate": 0,
            "simple": 0,
        }

        for level, keywords in COMPLEXITY_INDICATORS.items():
            for keyword in keywords:
                if keyword in task_lower:
                    scores[level] += 1

        # Additional complexity signals

        # Multi-part questions
        question_count = task.count("?")
        if question_count > 2:
            scores["complex"] += 2
        elif question_count > 1:
            scores["complex"] += 1

        # Conjunctions suggesting multiple requirements
        conjunction_words = ["and", "also", "additionally", "furthermore", "plus", "as well"]
        for word in conjunction_words:
            if f" {word} " in task_lower:
                scores["moderate"] += 1

        # Task length as complexity indicator
        if len(task) > 500:
            scores["complex"] += 1
        elif len(task) > 200:
            scores["moderate"] += 1

        # Constraints add complexity
        if constraints:
            scores["moderate"] += len(constraints)
            if len(constraints) > 3:
                scores["complex"] += 1

        # Determine final complexity
        if scores["exhaustive"] >= 2:
            return ComplexityLevel.EXHAUSTIVE
        elif scores["exhaustive"] >= 1 or scores["complex"] >= 3:
            return ComplexityLevel.COMPLEX
        elif scores["complex"] >= 1 or scores["moderate"] >= 2:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    def _detect_content_type(self, text: str) -> ContentType:
        """
        Detect the primary content type of the text.

        Args:
            text: Text to analyze

        Returns:
            ContentType enum value
        """
        if not text or len(text) < 50:
            return ContentType.UNKNOWN

        # Count indicators
        code_score = sum(1 for ind in CODE_INDICATORS if ind in text)
        data_score = sum(1 for ind in DATA_INDICATORS if ind in text)

        # Check for documentation patterns
        doc_patterns = ["##", "###", "**", "__", "- ", "* ", "1.", "```"]
        doc_score = sum(text.count(p) for p in doc_patterns)

        # Normalize scores
        text_len_factor = len(text) / 1000
        code_normalized = code_score / max(1, text_len_factor)
        data_normalized = data_score / max(1, text_len_factor * 2)
        doc_normalized = doc_score / max(1, text_len_factor)

        # Determine type
        max_score = max(code_normalized, data_normalized, doc_normalized)

        if max_score < 1.0:
            return ContentType.PROSE
        elif code_normalized == max_score and code_normalized > 3:
            return ContentType.CODE
        elif data_normalized == max_score and data_normalized > 5:
            return ContentType.DATA
        elif doc_normalized == max_score and doc_normalized > 2:
            return ContentType.DOCUMENTATION
        elif code_normalized > 1 and doc_normalized > 1:
            return ContentType.MIXED
        else:
            return ContentType.PROSE

    def _select_strategy(
        self,
        token_count: int,
        density: float,
        complexity: ComplexityLevel,
    ) -> tuple[StrategyType, str]:
        """
        Select optimal strategy based on analysis results.

        Implements the strategy matrix from the specification.

        Args:
            token_count: Number of tokens
            density: Information density (0.0-1.0)
            complexity: Task complexity level

        Returns:
            Tuple of (StrategyType, reasoning string)
        """
        cfg = self._config

        # Small context (<10K tokens): GSD
        if token_count < cfg.gsd_max_tokens:
            if complexity in (ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE):
                return (
                    StrategyType.GSD_DIRECT,
                    f"Small context ({token_count:,} tokens) with {complexity.value} "
                    f"complexity. GSD_DIRECT provides efficient single-pass processing.",
                )
            else:
                return (
                    StrategyType.GSD_DIRECT,  # Map to GSD_DIRECT since GSD_GUIDED not in types
                    f"Small context ({token_count:,} tokens) with {complexity.value} "
                    f"complexity. GSD with structured approach handles complex tasks.",
                )

        # Medium context (10K-50K tokens): RALPH
        if token_count < 50_000:
            if density < cfg.density_threshold_low + 0.2:  # Below 0.5
                return (
                    StrategyType.RALPH_STRUCTURED,
                    f"Medium context ({token_count:,} tokens) with sparse density "
                    f"({density:.2f}). RALPH iterative mode handles this efficiently.",
                )
            else:
                return (
                    StrategyType.RALPH_STRUCTURED,
                    f"Medium context ({token_count:,} tokens) with moderate/high density "
                    f"({density:.2f}). RALPH_STRUCTURED provides hierarchical processing.",
                )

        # Large context (50K-100K tokens)
        if token_count < cfg.ralph_max_tokens:
            if density < cfg.density_threshold_high:
                return (
                    StrategyType.RALPH_STRUCTURED,
                    f"Large context ({token_count:,} tokens) with moderate density "
                    f"({density:.2f}). RALPH_STRUCTURED handles multi-level summarization.",
                )
            else:
                return (
                    StrategyType.RLM_FULL,
                    f"Large context ({token_count:,} tokens) with high density "
                    f"({density:.2f}). RLM provides recursive processing for dense content.",
                )

        # Very large context (>100K tokens): RLM_FULL
        if density < cfg.density_threshold_high:
            return (
                StrategyType.RLM_FULL,
                f"Very large context ({token_count:,} tokens). "
                f"RLM_FULL enables recursive processing with sub-agents.",
            )
        else:
            return (
                StrategyType.RLM_DENSE,
                f"Very large, dense context ({token_count:,} tokens, density {density:.2f}). "
                f"RLM_DENSE provides aggressive recursion for maximum information extraction.",
            )

    def _get_alternative_strategies(
        self,
        primary: StrategyType,
        token_count: int,
    ) -> list[StrategyType]:
        """
        Get alternative strategies that could handle this context.

        Args:
            primary: Primary recommended strategy
            token_count: Token count

        Returns:
            List of alternative strategies
        """
        alternatives: list[StrategyType] = []

        # Define strategy capabilities
        strategy_limits = {
            StrategyType.GSD_DIRECT: self._config.gsd_max_tokens,
            StrategyType.RALPH_STRUCTURED: self._config.ralph_max_tokens,
            StrategyType.RLM_FULL: 10_000_000,
            StrategyType.RLM_DENSE: 10_000_000,
        }

        for strategy in StrategyType:
            if strategy == StrategyType.AUTO:
                continue
            if strategy != primary and strategy_limits.get(strategy, 0) >= token_count:
                alternatives.append(strategy)

        return alternatives[:3]

    def _estimate_costs(
        self,
        token_count: int,
        strategy: StrategyType,
    ) -> dict[str, CostEstimate]:
        """
        Estimate costs for different models.

        Args:
            token_count: Input token count
            strategy: Selected strategy

        Returns:
            Dictionary of model name to CostEstimate
        """
        costs: dict[str, CostEstimate] = {}

        # Strategy multipliers for multi-pass processing
        strategy_multipliers = {
            StrategyType.GSD_DIRECT: 1.0,
            StrategyType.RALPH_STRUCTURED: 2.0,
            StrategyType.RLM_FULL: 3.0,
            StrategyType.RLM_DENSE: 4.0,
        }
        multiplier = strategy_multipliers.get(strategy, 1.5)

        # Output ratio estimates
        output_ratio = 0.3 if strategy == StrategyType.GSD_DIRECT else 0.4

        for model, pricing in MODEL_PRICING.items():
            if pricing.get("input", 0) == 0:  # Skip free/local models
                continue

            effective_input = token_count * multiplier
            estimated_output = int(token_count * output_ratio)

            input_cost = (effective_input / 1_000_000) * pricing["input"]
            output_cost = (estimated_output / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost

            # Add verification overhead
            total_cost *= 1.2

            costs[model] = CostEstimate(
                model=model,
                input_cost=round(input_cost, 6),
                output_cost=round(output_cost, 6),
                total_cost=round(total_cost, 6),
            )

        return costs

    def _suggest_chunking(
        self,
        token_count: int,
        content_type: ContentType,
    ) -> ChunkSuggestion:
        """
        Generate chunking suggestions based on context size and type.

        Args:
            token_count: Total token count
            content_type: Detected content type

        Returns:
            ChunkSuggestion with recommended parameters
        """
        base_chunk_size = self._config.chunk_target_size

        # Adjust based on content type
        if content_type == ContentType.CODE:
            chunk_size = int(base_chunk_size * 0.75)  # Smaller for code
        elif content_type == ContentType.DATA:
            chunk_size = int(base_chunk_size * 1.25)  # Larger for structured data
        else:
            chunk_size = base_chunk_size

        overlap = int(chunk_size * self._config.chunk_overlap_ratio)
        effective_size = chunk_size - overlap

        if effective_size <= 0:
            effective_size = chunk_size // 2

        total_chunks = max(1, (token_count + effective_size - 1) // effective_size)

        return ChunkSuggestion(
            size=chunk_size,
            overlap=overlap,
            total_chunks=total_chunks,
        )

    def _generate_warnings(
        self,
        token_count: int,
        density: float,
        complexity: ComplexityLevel,
    ) -> list[str]:
        """
        Generate warnings about potential issues.

        Args:
            token_count: Token count
            density: Density score
            complexity: Complexity level

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        # Check context size
        default_limit = MODEL_CONTEXT_LIMITS.get(
            self._config.default_model, 200_000
        )
        usage_ratio = token_count / default_limit

        if usage_ratio > 0.9:
            warnings.append(
                f"Context uses {usage_ratio:.0%} of model limit. "
                "Consider chunking or using a model with larger context."
            )
        elif usage_ratio > 0.75:
            warnings.append(
                f"Context uses {usage_ratio:.0%} of model limit. "
                "Leave room for output tokens."
            )

        # Check density + complexity combination
        if density > 0.8 and complexity in (ComplexityLevel.COMPLEX, ComplexityLevel.EXHAUSTIVE):
            warnings.append(
                "High density with complex task may require multiple iterations. "
                "Consider breaking into subtasks."
            )

        # Check for very large context
        if token_count > 500_000:
            warnings.append(
                f"Very large context ({token_count:,} tokens). "
                "Processing may be slow and costly."
            )

        return warnings

    # =========================================================================
    # LLM Analysis Support
    # =========================================================================

    async def _create_analysis_agent(self) -> SubAgent:
        """
        Create a SubAgent for LLM-assisted analysis.

        Returns:
            Configured SubAgent instance
        """
        from contextflow.agents.sub_agent import AgentConfig, AgentRole, SubAgent

        system_prompt = """You are a context analysis expert. Analyze documents to determine:
1. Token count estimation accuracy
2. Information density (sparse vs dense)
3. Content complexity (simple, moderate, complex, exhaustive)
4. Recommended processing strategy

Provide clear, structured analysis with reasoning for your recommendations.

Consider:
- How much information is packed per token
- Whether the content requires deep analysis or simple extraction
- The relationships between different parts of the content
- Optimal chunking strategies for the content type

Output your analysis in a structured format."""

        config = AgentConfig(
            role=AgentRole.ANALYZER,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3,  # Lower for consistent analysis
            enable_verification=False,  # Analysis doesn't need verification
        )

        return SubAgent(
            provider=self._provider,
            role=AgentRole.ANALYZER,
            name="context-analyzer-agent",
            config=config,
        )

    def _build_llm_analysis_prompt(
        self,
        task: str,
        context_preview: str,
        base_analysis: ContextAnalysis,
        constraints: list[str] | None,
    ) -> str:
        """
        Build prompt for LLM analysis.

        Args:
            task: Original task
            context_preview: Preview of context (first 2000 chars)
            base_analysis: Heuristic analysis results
            constraints: Optional constraints

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "## Analysis Request",
            "",
            "Analyze the following context and task to determine optimal processing strategy.",
            "",
            "### Task",
            task,
            "",
            "### Context Preview (first 2000 characters)",
            "```",
            context_preview,
            "```",
            "",
            "### Heuristic Analysis Results",
            f"- Token Count: {base_analysis.token_count:,}",
            f"- Estimated Density: {base_analysis.density:.3f} ({base_analysis.density_level.value})",
            f"- Detected Complexity: {base_analysis.complexity.value}",
            f"- Content Type: {base_analysis.content_type.value}",
            f"- Recommended Strategy: {base_analysis.recommended_strategy.value}",
            "",
        ]

        if constraints:
            prompt_parts.extend([
                "### Constraints",
                *[f"- {c}" for c in constraints],
                "",
            ])

        prompt_parts.extend([
            "### Your Analysis",
            "Please provide:",
            "1. DENSITY_ASSESSMENT: Is the heuristic density estimate accurate? (yes/no with explanation)",
            "2. COMPLEXITY_ASSESSMENT: Is the complexity level accurate? (yes/no with explanation)",
            "3. STRATEGY_RECOMMENDATION: Do you agree with the strategy? If not, what do you recommend?",
            "4. ADDITIONAL_INSIGHTS: Any other observations about the content?",
            "",
            "Format your response with these exact section headers.",
        ])

        return "\n".join(prompt_parts)

    def _parse_llm_analysis(
        self,
        llm_output: str,
        base_analysis: ContextAnalysis,
    ) -> ContextAnalysis:
        """
        Parse LLM analysis output and enhance base analysis.

        Args:
            llm_output: Raw LLM output
            base_analysis: Original heuristic analysis

        Returns:
            Enhanced ContextAnalysis
        """
        # Create a copy of base analysis
        enhanced = ContextAnalysis(
            token_count=base_analysis.token_count,
            density=base_analysis.density,
            density_level=base_analysis.density_level,
            complexity=base_analysis.complexity,
            content_type=base_analysis.content_type,
            recommended_strategy=base_analysis.recommended_strategy,
            reasoning=base_analysis.reasoning,
            alternative_strategies=base_analysis.alternative_strategies.copy(),
            estimated_costs=base_analysis.estimated_costs.copy(),
            chunk_suggestion=base_analysis.chunk_suggestion,
            warnings=base_analysis.warnings.copy(),
            metadata=base_analysis.metadata.copy(),
        )

        # Extract strategy recommendation from LLM output
        output_lower = llm_output.lower()

        # Check for strategy changes
        strategy_map = {
            "gsd_direct": StrategyType.GSD_DIRECT,
            "ralph_structured": StrategyType.RALPH_STRUCTURED,
            "rlm_full": StrategyType.RLM_FULL,
            "rlm_dense": StrategyType.RLM_DENSE,
        }

        for strategy_name, strategy_type in strategy_map.items():
            if strategy_name in output_lower and "recommend" in output_lower:
                enhanced.recommended_strategy = strategy_type
                break

        # Adjust density if LLM suggests it's different
        if "density" in output_lower:
            if "higher" in output_lower or "more dense" in output_lower:
                enhanced.density = min(1.0, enhanced.density + 0.15)
                enhanced.density_level = self._classify_density(enhanced.density)
            elif "lower" in output_lower or "sparse" in output_lower:
                enhanced.density = max(0.0, enhanced.density - 0.15)
                enhanced.density_level = self._classify_density(enhanced.density)

        # Adjust complexity if LLM suggests it's different
        if "complexity" in output_lower:
            if "more complex" in output_lower or "higher complexity" in output_lower:
                complexity_upgrade = {
                    ComplexityLevel.SIMPLE: ComplexityLevel.MODERATE,
                    ComplexityLevel.MODERATE: ComplexityLevel.COMPLEX,
                    ComplexityLevel.COMPLEX: ComplexityLevel.EXHAUSTIVE,
                    ComplexityLevel.EXHAUSTIVE: ComplexityLevel.EXHAUSTIVE,
                }
                enhanced.complexity = complexity_upgrade[enhanced.complexity]
            elif "simpler" in output_lower or "lower complexity" in output_lower:
                complexity_downgrade = {
                    ComplexityLevel.SIMPLE: ComplexityLevel.SIMPLE,
                    ComplexityLevel.MODERATE: ComplexityLevel.SIMPLE,
                    ComplexityLevel.COMPLEX: ComplexityLevel.MODERATE,
                    ComplexityLevel.EXHAUSTIVE: ComplexityLevel.COMPLEX,
                }
                enhanced.complexity = complexity_downgrade[enhanced.complexity]

        # Add LLM reasoning
        enhanced.reasoning = (
            f"{base_analysis.reasoning}\n\n"
            f"LLM Analysis: {llm_output[:500]}..."
            if len(llm_output) > 500
            else f"{base_analysis.reasoning}\n\nLLM Analysis: {llm_output}"
        )

        enhanced.metadata["llm_output"] = llm_output

        return enhanced

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _update_cache(self, key: int, analysis: ContextAnalysis) -> None:
        """
        Update analysis cache with size management.

        Args:
            key: Cache key
            analysis: Analysis to cache
        """
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entries
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 4]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = analysis

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.debug("Analysis cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_max_size,
            "utilization": len(self._cache) / self._cache_max_size,
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug("Config updated", key=key, value=value)

    def __repr__(self) -> str:
        return (
            f"ContextAnalyzer("
            f"has_provider={self.has_provider}, "
            f"gsd_max={self._config.gsd_max_tokens}, "
            f"ralph_max={self._config.ralph_max_tokens})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_context(
    task: str,
    context: str,
    constraints: list[str] | None = None,
    config: AnalyzerConfig | None = None,
) -> ContextAnalysis:
    """
    Quick context analysis without creating analyzer instance.

    Convenience function for one-off analysis operations.

    Args:
        task: Task description
        context: Context to analyze
        constraints: Optional constraints
        config: Optional analyzer configuration

    Returns:
        ContextAnalysis with recommendation

    Example:
        analysis = analyze_context(
            task="Summarize this document",
            context=document_content
        )
        print(f"Strategy: {analysis.recommended_strategy}")
    """
    analyzer = ContextAnalyzer(config=config)
    return analyzer.analyze(task, context, constraints)


async def analyze_context_async(
    task: str,
    context: str,
    provider: BaseProvider | None = None,
    use_llm: bool = False,
) -> ContextAnalysis:
    """
    Async context analysis with optional LLM assistance.

    Args:
        task: Task description
        context: Context to analyze
        provider: Optional LLM provider
        use_llm: Whether to use LLM-assisted analysis

    Returns:
        ContextAnalysis with recommendation
    """
    analyzer = ContextAnalyzer(provider=provider)
    return await analyzer.analyze_async(task, context, use_llm=use_llm)


def get_recommended_strategy(
    token_count: int,
    density: float = 0.5,
    complexity: str = "moderate",
    config: AnalyzerConfig | None = None,
) -> StrategyType:
    """
    Get strategy recommendation for given parameters.

    Simple function to get strategy without analyzing content.

    Args:
        token_count: Number of tokens
        density: Information density (0.0-1.0)
        complexity: Complexity level string

    Returns:
        Recommended StrategyType
    """
    config = config or AnalyzerConfig()

    # Map complexity string to enum
    complexity_map = {
        "simple": ComplexityLevel.SIMPLE,
        "moderate": ComplexityLevel.MODERATE,
        "complex": ComplexityLevel.COMPLEX,
        "exhaustive": ComplexityLevel.EXHAUSTIVE,
    }
    complexity_level = complexity_map.get(complexity.lower(), ComplexityLevel.MODERATE)

    analyzer = ContextAnalyzer(config=config)
    strategy, _ = analyzer._select_strategy(token_count, density, complexity_level)
    return strategy


def estimate_analysis_cost(
    token_count: int,
    model: str = "claude-3-5-sonnet-20241022",
) -> CostEstimate:
    """
    Estimate cost for processing given token count.

    Args:
        token_count: Number of tokens
        model: Model to estimate for

    Returns:
        CostEstimate with breakdown
    """
    pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})

    # Assume typical processing multiplier
    effective_input = token_count * 1.5
    estimated_output = int(token_count * 0.3)

    input_cost = (effective_input / 1_000_000) * pricing["input"]
    output_cost = (estimated_output / 1_000_000) * pricing["output"]

    return CostEstimate(
        model=model,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round((input_cost + output_cost) * 1.2, 6),  # With overhead
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ComplexityLevel",
    "DensityLevel",
    "ContentType",
    # Data Classes
    "ChunkSuggestion",
    "CostEstimate",
    "ContextAnalysis",
    "AnalyzerConfig",
    # Main Class
    "ContextAnalyzer",
    # Convenience Functions
    "analyze_context",
    "analyze_context_async",
    "get_recommended_strategy",
    "estimate_analysis_cost",
    # Constants
    "COMPLEXITY_INDICATORS",
    "CODE_INDICATORS",
    "DATA_INDICATORS",
]
