"""
GSD Strategy - Get Shit Done.

For small context windows (<10K tokens). Direct, single-pass processing.

Two modes:
- GSD_DIRECT: Simple tasks, single LLM call
- GSD_GUIDED: Complex tasks, structured prompting with guidance

Boris Step 13: All outputs are verified through VerificationProtocol.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
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
from contextflow.utils.logging import ProviderLogger, get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider


logger = get_logger(__name__)


# =============================================================================
# System Prompts
# =============================================================================


GSD_DIRECT_SYSTEM_PROMPT = """You are a highly capable assistant focused on providing direct, accurate answers.

GUIDELINES:
1. Answer the task directly and completely
2. Be concise but thorough - include all relevant information
3. Use the provided context as your primary source of truth
4. If the context doesn't contain enough information, acknowledge this clearly
5. Structure your response for clarity when appropriate

Your response should directly address the user's task without unnecessary preamble."""


GSD_GUIDED_SYSTEM_PROMPT = """You are a methodical assistant that breaks down complex tasks into structured steps.

APPROACH:
1. First, analyze what the task is asking for
2. Identify the key components that need to be addressed
3. Work through each component systematically
4. Synthesize your findings into a coherent response
5. Verify your response addresses all aspects of the task

Think step-by-step and be thorough in your analysis."""


GSD_GUIDED_USER_TEMPLATE = """## Task Analysis

**Task:** {task}

**Context Length:** {context_length} characters

**Constraints:** {constraints}

## Structured Approach

Please follow this structured approach to complete the task:

### Step 1: Understand the Task
- What is being asked?
- What are the key requirements?

### Step 2: Analyze the Context
- What relevant information does the context provide?
- Are there any gaps in the information?

### Step 3: Formulate Response
- Address each requirement systematically
- Use evidence from the context

### Step 4: Synthesize
- Combine your analysis into a clear, complete response

---

## Context

{context}

---

## Task (for reference)

{task}

Please provide your complete response following the structured approach above."""


# =============================================================================
# Task Complexity Indicators
# =============================================================================

COMPLEXITY_KEYWORDS = {
    "high": [
        "analyze",
        "compare",
        "contrast",
        "evaluate",
        "synthesize",
        "explain why",
        "what are the implications",
        "how does this relate",
        "critique",
        "assess",
        "investigate",
        "examine in detail",
        "multiple aspects",
        "comprehensive",
        "thorough analysis",
        "step by step",
        "elaborate",
        "justify",
        "argue",
    ],
    "low": [
        "what is",
        "who is",
        "when was",
        "where is",
        "list",
        "name",
        "define",
        "identify",
        "find",
        "extract",
        "summarize briefly",
        "yes or no",
        "true or false",
        "how many",
        "which one",
    ],
}

# Average costs per 1M tokens (USD) - used for cost estimation
MODEL_COSTS = {
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "default": {"input": 3.00, "output": 15.00},
}


# =============================================================================
# GSD Strategy Implementation
# =============================================================================


class GSDStrategy(BaseStrategy):
    """
    GSD (Get Shit Done) Strategy for small contexts.

    Optimal for:
    - Context < 10K tokens
    - Single-pass processing
    - Direct answers without iteration

    Modes:
    - DIRECT: Simple task, single LLM call
    - GUIDED: Complex task, structured approach with step-by-step guidance

    Example:
        provider = ClaudeProvider(model="claude-3-sonnet-20240229")
        strategy = GSDStrategy(provider, mode="auto")

        result = await strategy.execute(
            task="Summarize the key points",
            context="Document content here..."
        )

        # Or with verification loop
        result = await strategy.execute_with_verification(
            task="Summarize the key points",
            context="Document content here...",
            max_iterations=3
        )
    """

    def __init__(
        self,
        provider: BaseProvider,
        mode: str = "auto",
        enable_verification: bool = True,
        verification_threshold: float = 0.7,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
    ) -> None:
        """
        Initialize GSD Strategy.

        Args:
            provider: LLM provider for completions
            mode: Processing mode - "direct", "guided", or "auto"
            enable_verification: Whether to verify output (Boris Step 13)
            verification_threshold: Minimum score to pass verification (0.0-1.0)
            model: Model override (uses provider default if not specified)
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens in generated output

        Raises:
            ValueError: If mode is invalid or threshold out of range
        """
        if mode not in ("direct", "guided", "auto"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'direct', 'guided', or 'auto'.")

        super().__init__(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verification_threshold=verification_threshold,
        )

        self._mode = mode
        self._enable_verification = enable_verification
        self._provider_logger = ProviderLogger("gsd")
        self._verifier: VerificationProtocol | None = None

        if enable_verification:
            self._verifier = VerificationProtocol(
                provider=provider,
                min_confidence=verification_threshold,
            )

        logger.info(
            "GSDStrategy initialized",
            mode=mode,
            enable_verification=enable_verification,
            verification_threshold=verification_threshold,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "GSD"

    @property
    def strategy_type(self) -> StrategyType:
        """
        Strategy type based on current mode.

        Returns:
            GSD_DIRECT or GSD_GUIDED based on mode setting
        """
        if self._mode == "direct":
            return StrategyType.GSD_DIRECT
        elif self._mode == "guided":
            return StrategyType.GSD_GUIDED
        # For auto mode, return DIRECT as default (determined at runtime)
        return StrategyType.GSD_DIRECT

    @property
    def max_tokens(self) -> int:
        """Maximum tokens this strategy can handle efficiently."""
        return 10_000

    @property
    def min_tokens(self) -> int:
        """Minimum tokens for this strategy."""
        return 0

    @property
    def mode(self) -> str:
        """Current processing mode."""
        return self._mode

    # -------------------------------------------------------------------------
    # Main Execution Methods
    # -------------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute GSD strategy.

        For DIRECT mode: Single LLM call with task + context
        For GUIDED mode: Structured prompt with step-by-step guidance
        For AUTO mode: Automatically determines the best mode

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints for verification
            **kwargs: Additional parameters passed to the provider

        Returns:
            StrategyResult with answer and metadata

        Raises:
            StrategyExecutionError: If execution fails
            ProviderError: If LLM call fails
        """
        start_time = time.time()

        # Determine mode if auto
        effective_mode = self._mode
        if self._mode == "auto":
            effective_mode = self._determine_mode(task, context)
            logger.debug(
                "Auto-determined mode",
                mode=effective_mode,
                task_preview=task[:100],
            )

        # Log execution start
        token_estimate = self._provider.count_tokens(context)
        self._logger.log_start(token_count=token_estimate, mode=effective_mode)

        try:
            # Execute based on determined mode
            if effective_mode == "direct":
                result = await self._execute_direct(task, context, constraints)
            else:
                result = await self._execute_guided(task, context, constraints)

            # Update execution time
            result.execution_time = time.time() - start_time

            # Update strategy type based on actual mode used
            if effective_mode == "guided":
                result.strategy_used = StrategyType.GSD_GUIDED

            logger.info(
                "GSD execution completed",
                mode=effective_mode,
                execution_time_ms=round(result.execution_time * 1000, 2),
                total_tokens=result.total_tokens,
            )

            return result

        except ProviderError as e:
            logger.error("Provider error during GSD execution", error=str(e))
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"Provider error: {str(e)}",
                cause=e,
            ) from e
        except Exception as e:
            logger.error("Unexpected error during GSD execution", error=str(e))
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"Execution failed: {str(e)}",
                cause=e,
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _execute_direct(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> StrategyResult:
        """
        Direct mode: Single call, straightforward processing.

        Simple and efficient for tasks that don't require complex reasoning.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints

        Returns:
            StrategyResult with answer and metadata
        """
        logger.debug("Executing GSD direct mode", task_length=len(task))

        # Build the user message
        user_content = self._build_direct_prompt(task, context, constraints)

        # Log request
        input_tokens = self._provider.count_tokens(user_content)
        self._provider_logger.log_request(
            model=self._model or self._provider.model,
            input_tokens=input_tokens,
        )

        # Make LLM call
        start_time = time.time()
        response = await self._provider.complete(
            messages=[Message(role="user", content=user_content)],
            system=GSD_DIRECT_SYSTEM_PROMPT,
            model=self._model,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        self._provider_logger.log_response(
            model=response.model,
            output_tokens=response.output_tokens,
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
        )

        # Build result
        return StrategyResult(
            answer=response.content,
            strategy_used=StrategyType.GSD_DIRECT,
            iterations=1,
            token_usage={
                "input": response.input_tokens,
                "output": response.output_tokens,
                "total": response.tokens_used,
            },
            execution_time=latency_ms / 1000,
            verification_passed=False,  # Will be set by verify()
            verification_score=0.0,
            metadata={
                "mode": "direct",
                "model": response.model,
                "cost_usd": response.cost_usd,
                "finish_reason": response.finish_reason,
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _execute_guided(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> StrategyResult:
        """
        Guided mode: Structured approach for complex tasks.

        Uses a structured prompt template to guide the LLM through:
        1. Task analysis
        2. Context examination
        3. Response formulation
        4. Synthesis

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional list of constraints

        Returns:
            StrategyResult with answer and metadata
        """
        logger.debug("Executing GSD guided mode", task_length=len(task))

        # Build the structured prompt
        constraints_str = (
            "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"
        )

        user_content = GSD_GUIDED_USER_TEMPLATE.format(
            task=task,
            context_length=len(context),
            constraints=constraints_str,
            context=context,
        )

        # Log request
        input_tokens = self._provider.count_tokens(user_content)
        self._provider_logger.log_request(
            model=self._model or self._provider.model,
            input_tokens=input_tokens,
        )

        # Make LLM call with guided system prompt
        start_time = time.time()
        response = await self._provider.complete(
            messages=[Message(role="user", content=user_content)],
            system=GSD_GUIDED_SYSTEM_PROMPT,
            model=self._model,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        self._provider_logger.log_response(
            model=response.model,
            output_tokens=response.output_tokens,
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
        )

        # Extract the final synthesized response
        answer = self._extract_synthesis(response.content)

        # Build result
        return StrategyResult(
            answer=answer,
            strategy_used=StrategyType.GSD_GUIDED,
            iterations=1,
            token_usage={
                "input": response.input_tokens,
                "output": response.output_tokens,
                "total": response.tokens_used,
            },
            execution_time=latency_ms / 1000,
            verification_passed=False,
            verification_score=0.0,
            metadata={
                "mode": "guided",
                "model": response.model,
                "cost_usd": response.cost_usd,
                "finish_reason": response.finish_reason,
                "full_response": response.content,  # Keep full structured response
            },
        )

    # -------------------------------------------------------------------------
    # Verification (Boris Step 13)
    # -------------------------------------------------------------------------

    async def verify(
        self,
        task: str,
        output: str,
        constraints: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify output using VerificationProtocol (Boris Step 13).

        Checks:
        1. Task alignment - Does output address the task?
        2. Completeness - Is the answer complete?
        3. Quality - Overall response quality

        Args:
            task: Original task description
            output: Generated output to verify
            constraints: Constraints to check against

        Returns:
            VerificationResult with pass/fail and details

        Raises:
            StrategyExecutionError: If verification fails to complete
        """
        if not self._enable_verification or self._verifier is None:
            # Return default passing result if verification disabled
            return VerificationResult(
                passed=True,
                confidence=1.0,
                issues=[],
                suggestions=[],
                checks_performed=["verification_disabled"],
            )

        logger.debug("Verifying GSD output", output_length=len(output))

        try:
            result = await self._verifier.verify(
                task=task,
                output=output,
                constraints=constraints,
            )

            # Convert detailed verification result to base VerificationResult
            return VerificationResult(
                passed=result.passed,
                confidence=result.confidence,
                issues=result.issues,
                suggestions=result.suggestions,
                checks_performed=[c.check_type.value for c in result.checks],
                metadata={
                    "overall_score": result.overall_score,
                    "execution_time": result.execution_time,
                    "check_details": [
                        {
                            "type": c.check_type.value,
                            "passed": c.passed,
                            "score": c.score,
                            "message": c.message,
                        }
                        for c in result.checks
                    ],
                },
            )

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"Verification failed: {str(e)}",
                cause=e,
            ) from e

    # -------------------------------------------------------------------------
    # Cost Estimation
    # -------------------------------------------------------------------------

    def estimate_cost(
        self,
        context_tokens: int,
        model: str | None = None,
    ) -> CostEstimate:
        """
        Estimate cost for GSD execution.

        Estimates based on:
        - Input tokens (context + task overhead)
        - Expected output tokens (typically 500-2000 for GSD)
        - Model pricing

        Args:
            context_tokens: Number of tokens in context
            model: Model to estimate for (uses default if not specified)

        Returns:
            CostEstimate with min/max/expected costs

        Raises:
            ValueError: If context_tokens is invalid
        """
        if context_tokens < 0:
            raise ValueError(f"context_tokens must be non-negative, got {context_tokens}")

        # Get model for pricing lookup
        model_name = model or self._model or self._provider.model
        model_key = self._normalize_model_name(model_name)
        pricing = MODEL_COSTS.get(model_key, MODEL_COSTS["default"])

        # Estimate output tokens based on mode
        if self._mode == "guided":
            # Guided mode typically produces longer outputs
            min_output = 800
            max_output = 3000
            expected_output = 1500
        else:
            # Direct mode is more concise
            min_output = 200
            max_output = 1500
            expected_output = 600

        # Add task overhead (approximately 100-200 tokens)
        task_overhead = 150
        total_input = context_tokens + task_overhead

        # Calculate costs (convert from per 1M tokens)
        min_cost = (total_input * pricing["input"] / 1_000_000) + (
            min_output * pricing["output"] / 1_000_000
        )
        max_cost = (total_input * pricing["input"] / 1_000_000) + (
            max_output * pricing["output"] / 1_000_000
        )
        expected_cost = (total_input * pricing["input"] / 1_000_000) + (
            expected_output * pricing["output"] / 1_000_000
        )

        # If verification is enabled, add verification overhead (~20% extra)
        if self._enable_verification:
            min_cost *= 1.15
            max_cost *= 1.25
            expected_cost *= 1.20

        return CostEstimate(
            min_cost=min_cost,
            max_cost=max_cost,
            expected_cost=expected_cost,
            model=model_name,
            context_tokens=context_tokens,
            estimated_output_tokens=expected_output,
            metadata={
                "mode": self._mode,
                "verification_enabled": self._enable_verification,
                "pricing": pricing,
            },
        )

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    def supports_streaming(self) -> bool:
        """GSD strategy supports streaming output."""
        return True

    async def stream(
        self,
        task: str,
        context: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream GSD output.

        Streams the response as it's generated. Note that streaming
        does not support verification loop - use execute_with_verification
        for verified results.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            **kwargs: Additional parameters

        Yields:
            String chunks as they are generated

        Raises:
            StrategyExecutionError: If streaming fails
            ProviderError: If provider doesn't support streaming
        """
        # Determine mode
        effective_mode = self._mode
        if self._mode == "auto":
            effective_mode = self._determine_mode(task, context)

        logger.debug(
            "Streaming GSD response",
            mode=effective_mode,
            task_length=len(task),
        )

        try:
            # Build prompt based on mode
            if effective_mode == "guided":
                constraints_str = "None specified"
                user_content = GSD_GUIDED_USER_TEMPLATE.format(
                    task=task,
                    context_length=len(context),
                    constraints=constraints_str,
                    context=context,
                )
                system_prompt = GSD_GUIDED_SYSTEM_PROMPT
            else:
                user_content = self._build_direct_prompt(task, context, None)
                system_prompt = GSD_DIRECT_SYSTEM_PROMPT

            # Stream from provider
            async for chunk in self._provider.stream(
                messages=[Message(role="user", content=user_content)],
                system=system_prompt,
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            ):
                yield chunk.content

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"Streaming failed: {str(e)}",
                cause=e,
            ) from e

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _determine_mode(self, task: str, context: str) -> str:
        """
        Auto-determine whether to use direct or guided mode.

        Uses heuristics based on:
        - Task complexity keywords
        - Task length
        - Context length

        Args:
            task: The task description
            context: The context content

        Returns:
            "direct" or "guided"
        """
        task_lower = task.lower()

        # Count complexity indicators
        high_count = sum(1 for keyword in COMPLEXITY_KEYWORDS["high"] if keyword in task_lower)
        low_count = sum(1 for keyword in COMPLEXITY_KEYWORDS["low"] if keyword in task_lower)

        # Check task length (longer tasks tend to be more complex)
        task_length_factor = 1 if len(task) > 200 else 0

        # Check context length (larger contexts may need structured approach)
        context_length_factor = 1 if len(context) > 5000 else 0

        # Calculate complexity score
        complexity_score = high_count * 2 - low_count + task_length_factor + context_length_factor

        logger.debug(
            "Mode determination",
            high_keywords=high_count,
            low_keywords=low_count,
            task_length=len(task),
            context_length=len(context),
            complexity_score=complexity_score,
        )

        # Use guided mode for complexity score >= 2
        return "guided" if complexity_score >= 2 else "direct"

    def _build_direct_prompt(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
    ) -> str:
        """
        Build prompt for direct mode execution.

        Args:
            task: The task description
            context: The context content
            constraints: Optional constraints

        Returns:
            Formatted prompt string
        """
        parts = [
            "## Task",
            task,
            "",
            "## Context",
            context,
        ]

        if constraints:
            parts.extend(
                [
                    "",
                    "## Constraints",
                    "\n".join(f"- {c}" for c in constraints),
                ]
            )

        parts.extend(
            [
                "",
                "## Instructions",
                "Please complete the task based on the context provided above.",
            ]
        )

        return "\n".join(parts)

    def _extract_synthesis(self, full_response: str) -> str:
        """
        Extract the synthesized answer from guided mode response.

        Looks for the synthesis section or returns the full response
        if no clear synthesis is found.

        Args:
            full_response: The complete LLM response

        Returns:
            Extracted or cleaned answer
        """
        # Try to find synthesis section
        synthesis_markers = [
            r"(?:###?\s*)?(?:Step\s*4|Synthesis|Final\s*(?:Response|Answer)):?\s*\n+(.*)",
            r"(?:###?\s*)?(?:Summary|Conclusion):?\s*\n+(.*)",
        ]

        for pattern in synthesis_markers:
            match = re.search(pattern, full_response, re.IGNORECASE | re.DOTALL)
            if match:
                synthesis = match.group(1).strip()
                # Clean up any trailing headers
                synthesis = re.split(r"\n###?\s+", synthesis)[0].strip()
                if len(synthesis) > 100:  # Ensure meaningful content
                    return synthesis

        # If no clear synthesis, return the full response
        # but clean up the structural elements
        cleaned = re.sub(r"###?\s*Step\s*\d+:.*?\n", "", full_response)
        return cleaned.strip() or full_response

    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for pricing lookup.

        Args:
            model: Raw model name

        Returns:
            Normalized model key for pricing lookup
        """
        model_lower = model.lower()

        if "opus" in model_lower:
            return "claude-3-opus"
        elif "sonnet" in model_lower:
            if "3-5" in model_lower or "3.5" in model_lower:
                return "claude-3-5-sonnet"
            return "claude-3-sonnet"
        elif "haiku" in model_lower:
            return "claude-3-haiku"
        elif "gpt-4-turbo" in model_lower:
            return "gpt-4-turbo"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower:
            return "gpt-3.5-turbo"

        return "default"

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation of GSD strategy."""
        return (
            f"GSDStrategy("
            f"mode={self._mode!r}, "
            f"verification={self._enable_verification}, "
            f"threshold={self._verification_threshold})"
        )
