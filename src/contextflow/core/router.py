"""
Strategy Router - Automatic strategy selection for ContextFlow.

Routes tasks to optimal strategy based on:
- Token count
- Information density (sparse vs dense)
- Task complexity

Decision Matrix:
Token Count | Density | Complexity | Strategy
<10K        | *       | Low        | GSD_DIRECT
<10K        | *       | High       | GSD_GUIDED
10K-50K     | <0.5    | *          | RALPH_ITERATIVE
10K-50K     | >=0.5   | *          | RALPH_STRUCTURED
50K-100K    | <0.7    | *          | RALPH_STRUCTURED
50K-100K    | >=0.7   | *          | RLM_BASIC
>100K       | *       | *          | RLM_FULL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextflow.strategies.base import (
    BaseStrategy,
    StrategyResult,
    StrategyType,
)
from contextflow.utils.logging import StrategyLogger, get_logger
from contextflow.utils.tokens import MODEL_PRICING, TokenEstimator

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider


logger = get_logger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ComplexityLevel(Enum):
    """Task complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXHAUSTIVE = "exhaustive"


# Complexity indicator keywords
COMPLEXITY_INDICATORS: dict[str, list[str]] = {
    "exhaustive": [
        "all", "every", "exhaustive", "comprehensive", "complete",
        "entire", "thorough", "full", "detailed analysis", "in-depth",
        "everything", "nothing missed", "cover all", "leave nothing out",
    ],
    "high": [
        "analyze", "compare", "contrast", "evaluate", "synthesize",
        "explain why", "implications", "how does this relate",
        "critique", "assess", "investigate", "examine",
        "multiple aspects", "step by step", "elaborate", "justify",
        "argue", "prove", "demonstrate", "connect", "patterns",
        "relationship between", "cause and effect", "impact of",
    ],
    "medium": [
        "summarize", "describe", "explain", "outline", "overview",
        "key points", "main ideas", "important", "significant",
        "breakdown", "structure", "organize", "categorize",
    ],
    "low": [
        "what is", "who is", "when was", "where is", "list",
        "name", "define", "identify", "find", "extract",
        "yes or no", "true or false", "how many", "which one",
        "simple", "quick", "brief", "short",
    ],
}

# Default cost estimation factors
DEFAULT_COST_FACTORS = {
    "gsd_output_ratio": 0.3,  # Expected output as ratio of input
    "ralph_multiplier": 2.0,  # Cost multiplier for multi-pass
    "rlm_multiplier": 3.0,  # Cost multiplier for recursive calls
    "verification_overhead": 0.2,  # 20% overhead for verification
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ContextAnalysis:
    """Analysis of context for strategy selection."""

    token_count: int
    estimated_density: float  # 0.0-1.0 (sparse to dense)
    complexity: ComplexityLevel
    recommended_strategy: StrategyType
    reasoning: str
    alternative_strategies: list[StrategyType]
    estimated_cost: dict[str, float]  # Strategy -> estimated cost
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token_count": self.token_count,
            "estimated_density": round(self.estimated_density, 3),
            "complexity": self.complexity.value,
            "recommended_strategy": self.recommended_strategy.value,
            "reasoning": self.reasoning,
            "alternative_strategies": [s.value for s in self.alternative_strategies],
            "estimated_cost": self.estimated_cost,
            "metadata": self.metadata,
        }


@dataclass
class RouterConfig:
    """Configuration for strategy router."""

    # Token thresholds
    gsd_max_tokens: int = 10_000
    ralph_min_tokens: int = 10_000
    ralph_max_tokens: int = 100_000
    rlm_min_tokens: int = 100_000

    # Density thresholds
    density_threshold_low: float = 0.5
    density_threshold_high: float = 0.7

    # Behavioral settings
    enable_fallback: bool = True
    preferred_strategy: StrategyType | None = None
    enable_cost_estimation: bool = True

    # Model for cost estimation
    default_model: str = "claude-3-5-sonnet-20241022"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gsd_max_tokens": self.gsd_max_tokens,
            "ralph_min_tokens": self.ralph_min_tokens,
            "ralph_max_tokens": self.ralph_max_tokens,
            "rlm_min_tokens": self.rlm_min_tokens,
            "density_threshold_low": self.density_threshold_low,
            "density_threshold_high": self.density_threshold_high,
            "enable_fallback": self.enable_fallback,
            "preferred_strategy": (
                self.preferred_strategy.value if self.preferred_strategy else None
            ),
            "enable_cost_estimation": self.enable_cost_estimation,
            "default_model": self.default_model,
        }


# =============================================================================
# Strategy Router
# =============================================================================


class StrategyRouter:
    """
    Automatic strategy selection for ContextFlow.

    Routes tasks to the optimal strategy based on:
    - Token count (context size)
    - Information density (sparse vs dense content)
    - Task complexity (simple vs complex requests)

    Decision Matrix:
    Token Count | Density | Complexity | Strategy
    <10K        | *       | Low        | GSD_DIRECT
    <10K        | *       | High       | GSD_GUIDED
    10K-50K     | <0.5    | *          | RALPH_ITERATIVE
    10K-50K     | >=0.5   | *          | RALPH_STRUCTURED
    50K-100K    | <0.7    | *          | RALPH_STRUCTURED
    50K-100K    | >=0.7   | *          | RLM_BASIC
    >100K       | *       | *          | RLM_FULL

    Usage:
        router = StrategyRouter(provider)

        # Get recommendation
        analysis = router.analyze(task, context)
        print(f"Recommended: {analysis.recommended_strategy}")

        # Or route directly
        result = await router.route(task, context)
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: RouterConfig | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        """
        Initialize StrategyRouter.

        Args:
            provider: LLM provider for strategy execution
            config: Router configuration (uses defaults if None)
            token_estimator: Token counter (creates default if None)
        """
        self._provider = provider
        self._config = config or RouterConfig()
        self._token_estimator = token_estimator or TokenEstimator()
        self._logger = StrategyLogger("router")
        self._general_logger = get_logger(__name__)

        # Strategy instances cache
        self._strategy_cache: dict[StrategyType, BaseStrategy] = {}

        logger.info(
            "StrategyRouter initialized",
            provider=provider.name,
            config=self._config.to_dict(),
        )

    @property
    def config(self) -> RouterConfig:
        """Get router configuration."""
        return self._config

    @property
    def provider(self) -> BaseProvider:
        """Get the provider instance."""
        return self._provider

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> ContextAnalysis:
        """
        Analyze context and recommend strategy.

        Performs comprehensive analysis of the task and context to
        determine the optimal strategy for processing.

        Args:
            task: Task to perform
            context: Context to analyze
            constraints: Optional constraints

        Returns:
            ContextAnalysis with recommendation and reasoning
        """
        logger.debug(
            "Analyzing context",
            task_length=len(task),
            context_length=len(context),
        )

        # Count tokens
        token_count = self._count_tokens(task + context)

        # Estimate density
        density = self._estimate_density(context)

        # Assess complexity
        complexity = self._assess_complexity(task)

        # Select strategy
        strategy, reasoning = self._select_strategy(token_count, density, complexity)

        # Handle preferred strategy override
        if self._config.preferred_strategy:
            if self._can_handle_context(self._config.preferred_strategy, token_count):
                strategy = self._config.preferred_strategy
                reasoning = f"Using preferred strategy override. Original recommendation: {strategy.value}"
            else:
                reasoning += f" (Preferred strategy {self._config.preferred_strategy.value} cannot handle {token_count} tokens)"

        # Get alternatives
        alternatives = self._get_alternative_strategies(strategy, token_count)

        # Estimate costs
        estimated_costs = self._estimate_costs(
            token_count,
            [strategy] + alternatives,
        )

        analysis = ContextAnalysis(
            token_count=token_count,
            estimated_density=density,
            complexity=complexity,
            recommended_strategy=strategy,
            reasoning=reasoning,
            alternative_strategies=alternatives,
            estimated_cost=estimated_costs,
            metadata={
                "task_length": len(task),
                "context_length": len(context),
                "constraints_count": len(constraints) if constraints else 0,
            },
        )

        logger.info(
            "Context analysis complete",
            token_count=token_count,
            density=round(density, 3),
            complexity=complexity.value,
            recommended=strategy.value,
        )

        return analysis

    async def route(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        force_strategy: StrategyType | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Route task to appropriate strategy and execute.

        Analyzes the context, selects the optimal strategy, and
        executes it to produce a result.

        Args:
            task: Task to perform
            context: Context to process
            constraints: Optional constraints
            force_strategy: Override automatic selection
            **kwargs: Additional strategy-specific args

        Returns:
            StrategyResult from selected strategy

        Raises:
            StrategyExecutionError: If strategy execution fails
            ValueError: If forced strategy cannot handle context
        """
        # Analyze context (unless forcing)
        if force_strategy:
            token_count = self._count_tokens(task + context)
            if not self._can_handle_context(force_strategy, token_count):
                raise ValueError(
                    f"Strategy {force_strategy.value} cannot handle "
                    f"{token_count} tokens (max: {self._get_strategy_max_tokens(force_strategy)})"
                )
            strategy_type = force_strategy
            logger.info(
                "Using forced strategy",
                strategy=force_strategy.value,
                token_count=token_count,
            )
        else:
            analysis = self.analyze(task, context, constraints)
            strategy_type = analysis.recommended_strategy
            logger.info(
                "Strategy selected",
                strategy=strategy_type.value,
                reasoning=analysis.reasoning[:100],
            )

        # Get or create strategy instance
        strategy = self._create_strategy(strategy_type)

        # Log execution start
        self._logger.log_start(
            token_count=self._count_tokens(context),
            strategy=strategy_type.value,
        )

        # Execute strategy with verification
        try:
            result = await strategy.execute_with_verification(
                task=task,
                context=context,
                constraints=constraints,
                **kwargs,
            )

            self._logger.log_complete(
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                duration_seconds=result.execution_time,
                verification_passed=result.verification_passed,
            )

            return result

        except Exception as e:
            logger.error(
                "Strategy execution failed",
                strategy=strategy_type.value,
                error=str(e),
            )

            # Try fallback if enabled
            if self._config.enable_fallback and not force_strategy:
                fallback = self._get_fallback_strategy(strategy_type, self._count_tokens(context))
                if fallback:
                    logger.info(
                        "Attempting fallback strategy",
                        fallback=fallback.value,
                    )
                    fallback_strategy = self._create_strategy(fallback)
                    return await fallback_strategy.execute_with_verification(
                        task=task,
                        context=context,
                        constraints=constraints,
                        **kwargs,
                    )

            raise

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
        return self._token_estimator.count_tokens(text)

    def _estimate_density(self, text: str) -> float:
        """
        Estimate information density of text.

        Higher density indicates more information per token.

        Factors considered:
        - Whitespace ratio (less whitespace = denser)
        - Average word length (longer words = denser vocabulary)
        - Code vs prose ratio (code is denser)
        - Repetition (less repetition = denser)
        - Structure indicators (tables, lists = denser)

        Args:
            text: Text to analyze

        Returns:
            Density score from 0.0 (sparse) to 1.0 (dense)
        """
        if not text or len(text) < 10:
            return 0.0

        density_score = 0.0
        factors_evaluated = 0

        # Factor 1: Whitespace ratio (less whitespace = denser)
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / len(text)
        # Target: 10-20% whitespace is normal prose, less is denser
        whitespace_factor = max(0.0, 1.0 - (whitespace_ratio * 3))
        density_score += whitespace_factor * 0.15
        factors_evaluated += 0.15

        # Factor 2: Average word length
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            # Average English word is ~5 chars, technical content is longer
            word_length_factor = min(1.0, avg_word_length / 8.0)
            density_score += word_length_factor * 0.15
            factors_evaluated += 0.15

        # Factor 3: Code content detection
        code_indicators = [
            "```", "def ", "class ", "import ", "function ",
            "const ", "var ", "let ", "return ", "if (", "for (",
            "->", "=>", "&&", "||", "==", "!=",
        ]
        code_matches = sum(1 for ind in code_indicators if ind in text)
        code_factor = min(1.0, code_matches / 5.0)
        density_score += code_factor * 0.2
        factors_evaluated += 0.2

        # Factor 4: Structure indicators (tables, lists, headers)
        structure_indicators = [
            "|", "- ", "* ", "1.", "##", "###",
            "<table", "<tr>", "<td>", "| ---",
        ]
        structure_matches = sum(text.count(ind) for ind in structure_indicators)
        structure_factor = min(1.0, structure_matches / 20.0)
        density_score += structure_factor * 0.15
        factors_evaluated += 0.15

        # Factor 5: Data/numeric content
        numeric_chars = sum(1 for c in text if c.isdigit())
        numeric_ratio = numeric_chars / len(text)
        numeric_factor = min(1.0, numeric_ratio * 10)  # 10%+ numbers is dense
        density_score += numeric_factor * 0.15
        factors_evaluated += 0.15

        # Factor 6: Unique words ratio (higher = less repetitive = denser)
        if words:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            unique_factor = unique_ratio  # Already 0-1
            density_score += unique_factor * 0.2
            factors_evaluated += 0.2

        # Normalize to 0-1 range
        if factors_evaluated > 0:
            final_density = density_score / factors_evaluated
        else:
            final_density = 0.3  # Default

        return min(1.0, max(0.0, final_density))

    def _assess_complexity(self, task: str) -> ComplexityLevel:
        """
        Assess task complexity based on linguistic indicators.

        Analyzes the task description to determine how complex
        the required processing will be.

        Indicators:
        - Keywords: "all", "every", "exhaustive", "comprehensive"
        - Multi-part questions (multiple question marks, "and", "also")
        - Comparison/analysis requests
        - Length and specificity of task

        Args:
            task: Task description

        Returns:
            ComplexityLevel enum value
        """
        task_lower = task.lower()

        # Count matches for each complexity level
        scores: dict[str, int] = {
            "exhaustive": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for level, keywords in COMPLEXITY_INDICATORS.items():
            for keyword in keywords:
                if keyword in task_lower:
                    scores[level] += 1

        # Additional complexity signals
        # Multi-part questions
        question_count = task.count("?")
        if question_count > 2:
            scores["high"] += 2
        elif question_count > 1:
            scores["high"] += 1

        # Conjunctions suggesting multiple requirements
        conjunction_words = ["and", "also", "additionally", "furthermore", "plus"]
        for word in conjunction_words:
            if f" {word} " in task_lower:
                scores["medium"] += 1

        # Task length as complexity indicator
        if len(task) > 500:
            scores["high"] += 1
        elif len(task) > 200:
            scores["medium"] += 1

        # Determine final complexity
        if scores["exhaustive"] >= 2:
            return ComplexityLevel.EXHAUSTIVE
        elif scores["exhaustive"] >= 1 or scores["high"] >= 3:
            return ComplexityLevel.HIGH
        elif scores["high"] >= 1 or scores["medium"] >= 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW

    def _select_strategy(
        self,
        token_count: int,
        density: float,
        complexity: ComplexityLevel,
    ) -> tuple[StrategyType, str]:
        """
        Select optimal strategy based on analysis results.

        Implements the decision matrix:
        Token Count | Density | Complexity | Strategy
        <10K        | *       | Low        | GSD_DIRECT
        <10K        | *       | High       | GSD_GUIDED
        10K-50K     | <0.5    | *          | RALPH_ITERATIVE
        10K-50K     | >=0.5   | *          | RALPH_STRUCTURED
        50K-100K    | <0.7    | *          | RALPH_STRUCTURED
        50K-100K    | >=0.7   | *          | RLM_BASIC
        >100K       | *       | *          | RLM_FULL

        Args:
            token_count: Number of tokens in context
            density: Information density score (0.0-1.0)
            complexity: Task complexity level

        Returns:
            Tuple of (StrategyType, reasoning string)
        """
        config = self._config

        # Small context (<10K tokens): GSD
        if token_count < config.gsd_max_tokens:
            if complexity in (ComplexityLevel.LOW, ComplexityLevel.MEDIUM):
                return (
                    StrategyType.GSD_DIRECT,
                    f"Small context ({token_count} tokens) with {complexity.value} complexity: "
                    f"GSD_DIRECT is optimal for single-pass processing.",
                )
            else:
                return (
                    StrategyType.GSD_GUIDED,
                    f"Small context ({token_count} tokens) with {complexity.value} complexity: "
                    f"GSD_GUIDED provides structured approach for complex tasks.",
                )

        # Medium context (10K-50K tokens)
        if token_count < 50_000:
            if density < config.density_threshold_low:
                return (
                    StrategyType.RALPH_ITERATIVE,
                    f"Medium context ({token_count} tokens) with low density ({density:.2f}): "
                    f"RALPH_ITERATIVE handles sparse content efficiently through sequential processing.",
                )
            else:
                return (
                    StrategyType.RALPH_STRUCTURED,
                    f"Medium context ({token_count} tokens) with moderate/high density ({density:.2f}): "
                    f"RALPH_STRUCTURED provides hierarchical summarization for dense content.",
                )

        # Large context (50K-100K tokens)
        if token_count < config.ralph_max_tokens:
            if density < config.density_threshold_high:
                return (
                    StrategyType.RALPH_STRUCTURED,
                    f"Large context ({token_count} tokens) with moderate density ({density:.2f}): "
                    f"RALPH_STRUCTURED handles this size with multi-level summarization.",
                )
            else:
                return (
                    StrategyType.RLM_BASIC,
                    f"Large context ({token_count} tokens) with high density ({density:.2f}): "
                    f"RLM_BASIC provides recursive processing for dense, information-rich content.",
                )

        # Very large context (>100K tokens): RLM_FULL
        return (
            StrategyType.RLM_FULL,
            f"Very large context ({token_count} tokens): "
            f"RLM_FULL is required for full recursive processing with sub-agents.",
        )

    def _get_alternative_strategies(
        self,
        primary: StrategyType,
        token_count: int,
    ) -> list[StrategyType]:
        """
        Get alternative strategies that could handle this context.

        Returns strategies in order of preference that can handle
        the given token count.

        Args:
            primary: Primary recommended strategy
            token_count: Token count to handle

        Returns:
            List of alternative StrategyType values
        """
        alternatives: list[StrategyType] = []

        # Define strategy capabilities
        strategy_limits = {
            StrategyType.GSD_DIRECT: self._config.gsd_max_tokens,
            StrategyType.GSD_GUIDED: self._config.gsd_max_tokens,
            StrategyType.RALPH_ITERATIVE: self._config.ralph_max_tokens,
            StrategyType.RALPH_STRUCTURED: self._config.ralph_max_tokens,
            StrategyType.RLM_BASIC: 10_000_000,  # RLM can handle very large
            StrategyType.RLM_FULL: 10_000_000,
        }

        # Order of preference (more capable strategies later)
        strategy_order = [
            StrategyType.GSD_DIRECT,
            StrategyType.GSD_GUIDED,
            StrategyType.RALPH_ITERATIVE,
            StrategyType.RALPH_STRUCTURED,
            StrategyType.RLM_BASIC,
            StrategyType.RLM_FULL,
        ]

        for strategy in strategy_order:
            if strategy != primary and strategy_limits.get(strategy, 0) >= token_count:
                alternatives.append(strategy)

        # Limit to top 3 alternatives
        return alternatives[:3]

    def _get_fallback_strategy(
        self,
        failed_strategy: StrategyType,
        token_count: int,
    ) -> StrategyType | None:
        """
        Get fallback strategy when primary fails.

        Args:
            failed_strategy: Strategy that failed
            token_count: Context token count

        Returns:
            Fallback strategy or None
        """
        # Fallback hierarchy
        fallbacks = {
            StrategyType.GSD_DIRECT: StrategyType.GSD_GUIDED,
            StrategyType.GSD_GUIDED: StrategyType.RALPH_ITERATIVE,
            StrategyType.RALPH_ITERATIVE: StrategyType.RALPH_STRUCTURED,
            StrategyType.RALPH_STRUCTURED: StrategyType.RLM_BASIC,
            StrategyType.RLM_BASIC: StrategyType.RLM_FULL,
            StrategyType.RLM_FULL: None,  # No fallback for RLM_FULL
        }

        fallback = fallbacks.get(failed_strategy)
        if fallback and self._can_handle_context(fallback, token_count):
            return fallback
        return None

    # =========================================================================
    # Strategy Management
    # =========================================================================

    def _create_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
        """
        Create strategy instance for the given type.

        Uses lazy loading with caching for efficiency.

        Args:
            strategy_type: Type of strategy to create

        Returns:
            Initialized strategy instance

        Raises:
            ValueError: If strategy type is not supported
        """
        # Check cache first
        if strategy_type in self._strategy_cache:
            return self._strategy_cache[strategy_type]

        # Import strategies lazily to avoid circular imports
        from contextflow.strategies.gsd import GSDStrategy
        from contextflow.strategies.ralph import RALPHStrategy
        from contextflow.strategies.rlm import RLMStrategy

        # Create strategy based on type
        strategy: BaseStrategy

        if strategy_type == StrategyType.GSD_DIRECT:
            strategy = GSDStrategy(
                provider=self._provider,
                mode="direct",
                enable_verification=True,
            )
        elif strategy_type == StrategyType.GSD_GUIDED:
            strategy = GSDStrategy(
                provider=self._provider,
                mode="guided",
                enable_verification=True,
            )
        elif strategy_type == StrategyType.RALPH_ITERATIVE:
            strategy = RALPHStrategy(
                provider=self._provider,
                mode="iterative",
                enable_verification=True,
            )
        elif strategy_type == StrategyType.RALPH_STRUCTURED:
            strategy = RALPHStrategy(
                provider=self._provider,
                mode="structured",
                enable_verification=True,
            )
        elif strategy_type == StrategyType.RLM_BASIC:
            strategy = RLMStrategy(
                provider=self._provider,
                mode="basic",
                enable_verification=True,
            )
        elif strategy_type == StrategyType.RLM_FULL:
            strategy = RLMStrategy(
                provider=self._provider,
                mode="full",
                enable_verification=True,
            )
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

        # Cache the strategy
        self._strategy_cache[strategy_type] = strategy
        return strategy

    def _can_handle_context(
        self,
        strategy_type: StrategyType,
        token_count: int,
    ) -> bool:
        """
        Check if a strategy can handle the given token count.

        Args:
            strategy_type: Strategy to check
            token_count: Number of tokens

        Returns:
            True if strategy can handle the context
        """
        max_tokens = self._get_strategy_max_tokens(strategy_type)
        return token_count <= max_tokens

    def _get_strategy_max_tokens(self, strategy_type: StrategyType) -> int:
        """
        Get maximum tokens a strategy can handle.

        Args:
            strategy_type: Strategy type

        Returns:
            Maximum token count
        """
        limits = {
            StrategyType.GSD_DIRECT: self._config.gsd_max_tokens,
            StrategyType.GSD_GUIDED: self._config.gsd_max_tokens,
            StrategyType.RALPH_ITERATIVE: self._config.ralph_max_tokens,
            StrategyType.RALPH_STRUCTURED: self._config.ralph_max_tokens,
            StrategyType.RLM_BASIC: 10_000_000,
            StrategyType.RLM_FULL: 10_000_000,
        }
        return limits.get(strategy_type, 10_000)

    # =========================================================================
    # Cost Estimation
    # =========================================================================

    def _estimate_costs(
        self,
        token_count: int,
        strategies: list[StrategyType],
    ) -> dict[str, float]:
        """
        Estimate costs for different strategies.

        Uses model pricing and strategy characteristics to estimate
        the likely cost of processing.

        Args:
            token_count: Input token count
            strategies: Strategies to estimate for

        Returns:
            Dictionary of strategy name -> estimated cost in USD
        """
        if not self._config.enable_cost_estimation:
            return {}

        costs: dict[str, float] = {}
        model = self._config.default_model
        pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})

        input_price_per_m = pricing.get("input", 3.0)
        output_price_per_m = pricing.get("output", 15.0)

        for strategy in strategies:
            # Estimate based on strategy characteristics
            if strategy in (StrategyType.GSD_DIRECT, StrategyType.GSD_GUIDED):
                # Single pass, estimate output at 30% of input
                output_estimate = int(token_count * 0.3)
                multiplier = 1.2 if strategy == StrategyType.GSD_GUIDED else 1.0

            elif strategy in (StrategyType.RALPH_ITERATIVE, StrategyType.RALPH_STRUCTURED):
                # Multi-pass, 2x input due to chunking + synthesis
                output_estimate = int(token_count * 0.5)
                multiplier = 2.0 if strategy == StrategyType.RALPH_STRUCTURED else 1.5

            else:  # RLM strategies
                # Recursive, 3x input due to multiple passes
                output_estimate = int(token_count * 0.4)
                multiplier = 3.0 if strategy == StrategyType.RLM_FULL else 2.5

            # Calculate cost
            input_cost = (token_count * multiplier / 1_000_000) * input_price_per_m
            output_cost = (output_estimate / 1_000_000) * output_price_per_m

            # Add verification overhead
            total_cost = (input_cost + output_cost) * (1 + DEFAULT_COST_FACTORS["verification_overhead"])

            costs[strategy.value] = round(total_cost, 6)

        return costs

    # =========================================================================
    # Information Methods
    # =========================================================================

    def get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]:
        """
        Get information about a specific strategy.

        Args:
            strategy_type: Strategy to get info for

        Returns:
            Dictionary with strategy details
        """
        descriptions = {
            StrategyType.GSD_DIRECT: {
                "name": "GSD Direct",
                "description": "Single-pass processing for simple tasks with small context",
                "optimal_for": "Tasks < 10K tokens with simple requirements",
                "min_tokens": 0,
                "max_tokens": self._config.gsd_max_tokens,
            },
            StrategyType.GSD_GUIDED: {
                "name": "GSD Guided",
                "description": "Structured single-pass for complex tasks with small context",
                "optimal_for": "Complex tasks < 10K tokens requiring step-by-step approach",
                "min_tokens": 0,
                "max_tokens": self._config.gsd_max_tokens,
            },
            StrategyType.RALPH_ITERATIVE: {
                "name": "RALPH Iterative",
                "description": "Sequential chunk processing with accumulation",
                "optimal_for": "Sparse content 10K-50K tokens",
                "min_tokens": self._config.ralph_min_tokens,
                "max_tokens": 50_000,
            },
            StrategyType.RALPH_STRUCTURED: {
                "name": "RALPH Structured",
                "description": "Hierarchical summarization with multi-level processing",
                "optimal_for": "Dense content 10K-100K tokens",
                "min_tokens": self._config.ralph_min_tokens,
                "max_tokens": self._config.ralph_max_tokens,
            },
            StrategyType.RLM_BASIC: {
                "name": "RLM Basic",
                "description": "Simplified recursive processing with REPL",
                "optimal_for": "Very dense content 50K-100K tokens",
                "min_tokens": 50_000,
                "max_tokens": 10_000_000,
            },
            StrategyType.RLM_FULL: {
                "name": "RLM Full",
                "description": "Full recursive language model with sub-agents",
                "optimal_for": "Very large context > 100K tokens",
                "min_tokens": self._config.rlm_min_tokens,
                "max_tokens": 10_000_000,
            },
        }

        return descriptions.get(strategy_type, {
            "name": strategy_type.value,
            "description": "Unknown strategy",
            "optimal_for": "Unknown",
            "min_tokens": 0,
            "max_tokens": 0,
        })

    def list_strategies(self) -> list[dict[str, Any]]:
        """
        List all available strategies with their characteristics.

        Returns:
            List of strategy information dictionaries
        """
        return [
            self.get_strategy_info(strategy_type)
            for strategy_type in StrategyType
            if strategy_type.value not in ("auto",)  # Skip AUTO
        ]

    def __repr__(self) -> str:
        """String representation of router."""
        return (
            f"StrategyRouter("
            f"provider={self._provider.name!r}, "
            f"gsd_max={self._config.gsd_max_tokens}, "
            f"ralph_max={self._config.ralph_max_tokens})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def auto_route(
    provider: BaseProvider,
    task: str,
    context: str,
    **kwargs: Any,
) -> StrategyResult:
    """
    Quick routing without creating router instance.

    Convenience function for one-off routing operations.

    Args:
        provider: LLM provider
        task: Task to perform
        context: Context to process
        **kwargs: Additional arguments passed to route()

    Returns:
        StrategyResult from selected strategy
    """
    router = StrategyRouter(provider)
    return await router.route(task, context, **kwargs)


def analyze_context(
    task: str,
    context: str,
    provider: BaseProvider | None = None,
    config: RouterConfig | None = None,
) -> ContextAnalysis:
    """
    Quick analysis without full router setup.

    Performs context analysis without requiring a provider
    (uses only token estimation and heuristics).

    Args:
        task: Task to perform
        context: Context to analyze
        provider: Optional provider for more accurate token counting
        config: Optional router configuration

    Returns:
        ContextAnalysis with recommendation
    """
    # Create minimal router for analysis
    if provider:
        router = StrategyRouter(provider, config=config)
    else:
        # Create a mock provider for analysis only

        class _AnalysisProvider:
            """Minimal provider for analysis only."""

            name = "analysis"

            def count_tokens(self, text: str) -> int:
                return len(text) // 4  # Rough estimate

        # Use duck typing - analysis only needs count_tokens
        router = StrategyRouter(
            provider=_AnalysisProvider(),  # type: ignore
            config=config,
        )

    return router.analyze(task, context)


def get_recommended_strategy(
    token_count: int,
    density: float = 0.5,
    complexity: str = "medium",
    config: RouterConfig | None = None,
) -> StrategyType:
    """
    Get recommended strategy for given parameters.

    Simple function to get strategy recommendation without
    analyzing actual content.

    Args:
        token_count: Number of tokens
        density: Information density (0.0-1.0)
        complexity: Complexity level ("low", "medium", "high", "exhaustive")
        config: Optional router configuration

    Returns:
        Recommended StrategyType
    """
    config = config or RouterConfig()

    # Map complexity string to enum
    complexity_map = {
        "low": ComplexityLevel.LOW,
        "medium": ComplexityLevel.MEDIUM,
        "high": ComplexityLevel.HIGH,
        "exhaustive": ComplexityLevel.EXHAUSTIVE,
    }
    complexity_level = complexity_map.get(complexity.lower(), ComplexityLevel.MEDIUM)

    # Create minimal router and use selection logic
    class _MinimalProvider:
        name = "minimal"

        def count_tokens(self, text: str) -> int:
            return token_count

    router = StrategyRouter(
        provider=_MinimalProvider(),  # type: ignore
        config=config,
    )

    strategy, _ = router._select_strategy(token_count, density, complexity_level)
    return strategy
