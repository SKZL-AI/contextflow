"""
Abstract base class for all context processing strategies.

This module defines the BaseStrategy ABC that all strategy implementations
(GSD, RALPH, RLM) must inherit from, ensuring consistent behavior and
mandatory verification loops (Boris Step 13).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from contextflow.utils.errors import StrategyExecutionError
from contextflow.utils.logging import StrategyLogger, get_logger

# =============================================================================
# Strategy Type Enum
# =============================================================================


class StrategyType(str, Enum):
    """
    Strategy types for context processing.

    Strategies are selected based on context size and complexity:
    - GSD_DIRECT: Simple tasks with <10K tokens
    - GSD_GUIDED: Complex tasks with <10K tokens
    - RALPH_ITERATIVE: 10K-50K tokens, sparse content
    - RALPH_STRUCTURED: 10K-100K tokens, dense content
    - RLM_BASIC: 50K-100K tokens, very dense content
    - RLM_FULL: >100K tokens, full recursive processing
    """

    GSD_DIRECT = "gsd_direct"
    GSD_GUIDED = "gsd_guided"
    RALPH_ITERATIVE = "ralph_iterative"
    RALPH_STRUCTURED = "ralph_structured"
    RLM_BASIC = "rlm_basic"
    RLM_FULL = "rlm_full"


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class StrategyResult:
    """
    Result from strategy execution.

    Contains the generated answer along with metadata about the execution
    including token usage, timing, and verification results.

    Attributes:
        answer: The generated response text
        strategy_used: Which strategy produced this result
        iterations: Number of iterations taken
        token_usage: Dict with 'input', 'output', 'total' token counts
        execution_time: Time taken in seconds
        verification_passed: Whether verification checks passed
        verification_score: Confidence score from 0.0 to 1.0
        sub_results: Nested results from sub-strategies (RLM)
        metadata: Additional strategy-specific metadata
    """

    answer: str
    strategy_used: StrategyType
    iterations: int
    token_usage: dict[str, int]
    execution_time: float
    verification_passed: bool
    verification_score: float
    sub_results: list[StrategyResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        if not 0.0 <= self.verification_score <= 1.0:
            raise ValueError(
                f"verification_score must be between 0.0 and 1.0, " f"got {self.verification_score}"
            )
        if self.iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {self.iterations}")

    @property
    def total_tokens(self) -> int:
        """Get total token count including sub-results."""
        total = self.token_usage.get("total", 0)
        for sub_result in self.sub_results:
            total += sub_result.total_tokens
        return total

    @property
    def total_cost(self) -> float:
        """Get estimated cost in USD from metadata."""
        return self.metadata.get("cost_usd", 0.0)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "answer": self.answer,
            "strategy_used": self.strategy_used.value,
            "iterations": self.iterations,
            "token_usage": self.token_usage,
            "execution_time": self.execution_time,
            "verification_passed": self.verification_passed,
            "verification_score": self.verification_score,
            "sub_results": [sr.to_dict() for sr in self.sub_results],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class VerificationResult:
    """
    Result from verification check (Boris Step 13).

    Contains the outcome of self-verification including confidence
    scores and actionable feedback for improvement.

    Attributes:
        passed: Whether verification passed
        confidence: Confidence score from 0.0 to 1.0
        issues: List of identified issues
        suggestions: List of improvement suggestions
        checks_performed: List of verification checks that were run
        metadata: Additional verification metadata
    """

    passed: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate verification result data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def issue_count(self) -> int:
        """Get number of identified issues."""
        return len(self.issues)

    @property
    def has_suggestions(self) -> bool:
        """Check if there are improvement suggestions."""
        return len(self.suggestions) > 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the verification result
        """
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "checks_performed": self.checks_performed,
            "metadata": self.metadata,
        }


@dataclass
class CostEstimate:
    """
    Cost estimation for strategy execution.

    Provides min/max/expected cost estimates before execution
    to help with budget planning.

    Attributes:
        min_cost: Minimum expected cost in USD
        max_cost: Maximum expected cost in USD
        expected_cost: Most likely cost in USD
        model: Model used for estimation
        context_tokens: Input tokens considered
        estimated_output_tokens: Estimated output tokens
        metadata: Additional estimation metadata
    """

    min_cost: float
    max_cost: float
    expected_cost: float
    model: str
    context_tokens: int
    estimated_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cost estimate data."""
        if self.min_cost < 0:
            raise ValueError(f"min_cost must be non-negative, got {self.min_cost}")
        if self.max_cost < self.min_cost:
            raise ValueError(f"max_cost ({self.max_cost}) must be >= min_cost ({self.min_cost})")
        if not self.min_cost <= self.expected_cost <= self.max_cost:
            raise ValueError(
                f"expected_cost ({self.expected_cost}) must be between "
                f"min_cost ({self.min_cost}) and max_cost ({self.max_cost})"
            )

    def to_dict(self) -> dict[str, float]:
        """
        Convert to simple dictionary for API responses.

        Returns:
            Dictionary with min, max, expected costs
        """
        return {
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "expected_cost": self.expected_cost,
        }


# =============================================================================
# Base Strategy ABC
# =============================================================================


class BaseStrategy(ABC):
    """
    Abstract base class for all context processing strategies.

    All strategies must implement:
    - execute(): Main processing logic
    - verify(): Self-verification of output (Boris Step 13)
    - estimate_cost(): Cost estimation before execution

    Strategies are selected based on context characteristics:
    - Token count (size)
    - Content density (information per token)
    - Task complexity (simple vs multi-step)

    Example implementation:
        class GSDDirectStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "gsd_direct"

            @property
            def strategy_type(self) -> StrategyType:
                return StrategyType.GSD_DIRECT

            async def execute(self, task, context, **kwargs) -> StrategyResult:
                # Implementation here
                pass
    """

    def __init__(
        self,
        provider: Any,  # BaseProvider, but avoiding circular import
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        verification_threshold: float = 0.8,
    ):
        """
        Initialize the strategy.

        Args:
            provider: LLM provider instance for completions
            model: Model to use (overrides provider default)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in generated output
            verification_threshold: Minimum score to pass verification

        Raises:
            ValueError: If temperature or verification_threshold out of range
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")
        if not 0.0 <= verification_threshold <= 1.0:
            raise ValueError(
                f"verification_threshold must be between 0.0 and 1.0, "
                f"got {verification_threshold}"
            )

        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._verification_threshold = verification_threshold
        self._logger = StrategyLogger(self.name)
        self._general_logger = get_logger(__name__)

    # -------------------------------------------------------------------------
    # Abstract Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy identifier.

        Returns:
            Unique strategy name (e.g., "gsd_direct", "ralph_structured")
        """
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """
        Strategy type enum value.

        Returns:
            StrategyType enum for this strategy
        """
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        Maximum tokens this strategy can handle efficiently.

        Returns:
            Maximum context token count
        """
        pass

    @property
    @abstractmethod
    def min_tokens(self) -> int:
        """
        Minimum tokens for this strategy.

        If context is below this threshold, consider a simpler strategy.

        Returns:
            Minimum context token count
        """
        pass

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute the strategy on the given task and context.

        This is the main processing method. Implementations should:
        1. Analyze the task and context
        2. Process according to strategy logic
        3. Generate the answer
        4. Return a StrategyResult (verification can be separate)

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints for verification
            **kwargs: Strategy-specific parameters

        Returns:
            StrategyResult with answer and metadata

        Raises:
            StrategyExecutionError: If execution fails
            ContextOverflowError: If context exceeds strategy limits
        """
        pass

    @abstractmethod
    async def verify(
        self,
        task: str,
        output: str,
        constraints: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify that the output meets task requirements.

        Boris Step 13: "Give Claude a way to verify its work"
        This provides 2-3x quality improvement by enabling self-correction.

        Implementations should check:
        1. Completeness: Does the output address all parts of the task?
        2. Accuracy: Is the information factually correct?
        3. Consistency: Does the output align with the provided context?
        4. Constraints: Are all specified constraints satisfied?

        Args:
            task: Original task description
            output: Generated output to verify
            constraints: Constraints to check against

        Returns:
            VerificationResult with pass/fail and details

        Raises:
            StrategyExecutionError: If verification fails to complete
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        context_tokens: int,
        model: str | None = None,
    ) -> CostEstimate:
        """
        Estimate execution cost before running.

        Useful for budget planning and strategy selection.
        Estimates should account for:
        - Input tokens (context)
        - Expected output tokens
        - Verification iterations
        - Model pricing

        Args:
            context_tokens: Number of tokens in context
            model: Model to estimate for (uses default if not specified)

        Returns:
            CostEstimate with min/max/expected costs

        Raises:
            ValueError: If context_tokens is invalid
        """
        pass

    # -------------------------------------------------------------------------
    # Default Implementation Methods
    # -------------------------------------------------------------------------

    async def execute_with_verification(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        max_iterations: int = 3,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute strategy with automatic verification loop.

        Iterates until verification passes or max_iterations reached.
        This is the recommended entry point for production use.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints for verification
            max_iterations: Maximum verification iterations (default 3)
            **kwargs: Strategy-specific parameters

        Returns:
            StrategyResult with verification status

        Raises:
            StrategyExecutionError: If all iterations fail verification
        """
        start_time = time.time()
        self._logger.log_start(
            token_count=len(context) // 4,  # Rough estimate
            max_iterations=max_iterations,
        )

        best_result: StrategyResult | None = None
        best_score: float = 0.0

        for iteration in range(1, max_iterations + 1):
            self._logger.log_iteration(
                iteration=iteration,
                max_iterations=max_iterations,
            )

            try:
                # Execute strategy
                result = await self.execute(
                    task=task,
                    context=context,
                    constraints=constraints,
                    **kwargs,
                )

                # Verify output
                verification = await self.verify(
                    task=task,
                    output=result.answer,
                    constraints=constraints,
                )

                # Update result with verification info
                result.verification_passed = verification.passed
                result.verification_score = verification.confidence
                result.iterations = iteration
                result.metadata["verification_details"] = verification.to_dict()

                # Track best result
                if verification.confidence > best_score:
                    best_score = verification.confidence
                    best_result = result

                # Check if verification passed
                if verification.passed:
                    result.execution_time = time.time() - start_time
                    self._logger.log_complete(
                        total_tokens=result.total_tokens,
                        total_cost=result.total_cost,
                        duration_seconds=result.execution_time,
                        iterations=iteration,
                        verification_passed=True,
                    )
                    return result

                # Log verification failure for non-final iterations
                if iteration < max_iterations:
                    self._general_logger.debug(
                        "Verification failed, retrying",
                        iteration=iteration,
                        score=verification.confidence,
                        issues=verification.issues,
                    )

            except Exception as e:
                self._general_logger.warning(
                    "Iteration failed with error",
                    iteration=iteration,
                    error=str(e),
                )
                if iteration == max_iterations:
                    raise StrategyExecutionError(
                        strategy=self.name,
                        message=f"All {max_iterations} iterations failed: {e}",
                        cause=e,
                    ) from e

        # Return best result if no iteration passed verification
        if best_result is not None:
            best_result.execution_time = time.time() - start_time
            best_result.metadata["max_iterations_reached"] = True
            self._logger.log_complete(
                total_tokens=best_result.total_tokens,
                total_cost=best_result.total_cost,
                duration_seconds=best_result.execution_time,
                iterations=max_iterations,
                verification_passed=False,
            )
            return best_result

        # Should not reach here, but handle gracefully
        raise StrategyExecutionError(
            strategy=self.name,
            message=f"No result produced after {max_iterations} iterations",
        )

    def supports_streaming(self) -> bool:
        """
        Check if this strategy supports streaming output.

        Override in subclasses that support streaming.

        Returns:
            False by default, True if streaming is supported
        """
        return False

    async def stream(
        self,
        task: str,
        context: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream output chunks.

        Override if supports_streaming() returns True.
        Default implementation raises NotImplementedError.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            **kwargs: Strategy-specific parameters

        Yields:
            String chunks as they are generated

        Raises:
            NotImplementedError: If streaming not supported
        """
        raise NotImplementedError(
            f"Streaming not supported by {self.name} strategy. "
            f"Check supports_streaming() before calling stream()."
        )
        # Make this an async generator for type checking
        yield ""  # pragma: no cover

    def can_handle(self, token_count: int) -> bool:
        """
        Check if this strategy can handle the given token count.

        Args:
            token_count: Number of tokens in context

        Returns:
            True if token count is within strategy limits
        """
        return self.min_tokens <= token_count <= self.max_tokens

    def get_info(self) -> dict[str, Any]:
        """
        Get information about this strategy.

        Returns:
            Dictionary with strategy details
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming(),
            "verification_threshold": self._verification_threshold,
        }

    def __repr__(self) -> str:
        """String representation of strategy."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"type={self.strategy_type.value!r}, "
            f"tokens={self.min_tokens}-{self.max_tokens})"
        )


# =============================================================================
# Strategy Factory Protocol
# =============================================================================


class StrategyFactory:
    """
    Factory for creating strategy instances.

    Provides centralized strategy instantiation with proper
    provider injection and configuration.

    Example:
        factory = StrategyFactory()
        factory.register("gsd_direct", GSDDirectStrategy)

        strategy = factory.create("gsd_direct", provider=my_provider)
        result = await strategy.execute(task, context)
    """

    _registry: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: type[BaseStrategy]) -> None:
        """
        Register a strategy class.

        Args:
            name: Strategy name for lookup
            strategy_class: Strategy class to register

        Raises:
            ValueError: If name already registered
        """
        if name in cls._registry:
            raise ValueError(f"Strategy '{name}' already registered")
        cls._registry[name] = strategy_class

    @classmethod
    def create(
        cls,
        name: str,
        provider: Any,
        **kwargs: Any,
    ) -> BaseStrategy:
        """
        Create a strategy instance.

        Args:
            name: Strategy name to create
            provider: LLM provider instance
            **kwargs: Strategy initialization parameters

        Returns:
            Initialized strategy instance

        Raises:
            ValueError: If strategy name not found
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._registry[name](provider=provider, **kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        List all registered strategies.

        Returns:
            List of registered strategy names
        """
        return list(cls._registry.keys())

    @classmethod
    def get_strategy_class(cls, name: str) -> type[BaseStrategy]:
        """
        Get strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy name not found
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy '{name}'")
        return cls._registry[name]
