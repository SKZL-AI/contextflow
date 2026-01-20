"""
Verification Protocol for ContextFlow - Boris Step 13.

"Give Claude a way to verify its work - 2-3x quality improvement"

This module provides a comprehensive verification system that ensures
LLM outputs meet task requirements, constraints, and quality standards.
All strategies use this protocol to self-verify their outputs.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextflow.core.types import Message
from contextflow.utils.errors import ProviderError, StrategyExecutionError
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class VerificationCheckType(Enum):
    """Types of verification checks."""

    TASK_ALIGNMENT = "task_alignment"  # Does output match task?
    CONSTRAINT_CHECK = "constraint_check"  # All constraints met?
    COMPLETENESS = "completeness"  # Is answer complete?
    ACCURACY = "accuracy"  # No hallucinations?
    FORMAT_CHECK = "format_check"  # Correct format?
    QUALITY_CHECK = "quality_check"  # Overall quality


@dataclass
class VerificationCheck:
    """Single verification check result."""

    check_type: VerificationCheckType
    passed: bool
    score: float  # 0.0-1.0
    message: str
    details: dict[str, Any] | None = None


@dataclass
class VerificationResult:
    """Complete verification result."""

    passed: bool
    confidence: float  # 0.0-1.0
    overall_score: float  # 0.0-1.0
    checks: list[VerificationCheck] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    iteration: int = 0


# =============================================================================
# System Prompts for Verification
# =============================================================================


VERIFICATION_SYSTEM_PROMPT = """You are a verification assistant that evaluates LLM outputs for quality and correctness.
Your role is to objectively assess whether an output meets the requirements of a given task.

For each verification request, you will:
1. Analyze the TASK (what was asked)
2. Analyze the OUTPUT (what was generated)
3. Evaluate based on specific criteria
4. Return a structured JSON assessment

Be strict but fair. If the output substantially addresses the task, it passes.
If there are significant gaps, missing information, or errors, it fails.

IMPORTANT: Always respond with ONLY valid JSON. No additional text before or after."""


TASK_ALIGNMENT_PROMPT = """Evaluate if the OUTPUT properly addresses the TASK.

TASK:
{task}

OUTPUT:
{output}

Evaluate:
1. Does the output directly address what was asked?
2. Is the response on-topic and relevant?
3. Does it answer the core question/request?

Respond with ONLY this JSON format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "message": "Brief explanation",
    "issues": ["List any issues"],
    "details": {{"relevance": 0.0-1.0, "directness": 0.0-1.0}}
}}"""


CONSTRAINT_CHECK_PROMPT = """Evaluate if the OUTPUT meets all CONSTRAINTS.

OUTPUT:
{output}

CONSTRAINTS:
{constraints}

For EACH constraint, determine if it is met.

Respond with ONLY this JSON format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "message": "Brief summary",
    "constraint_results": [
        {{"constraint": "...", "met": true/false, "note": "..."}}
    ],
    "unmet_constraints": ["List any unmet constraints"]
}}"""


COMPLETENESS_PROMPT = """Evaluate if the OUTPUT is COMPLETE for the given TASK.

TASK:
{task}

OUTPUT:
{output}

Evaluate:
1. Are all parts of the task addressed?
2. Is the response thorough enough?
3. Are there any obvious gaps or missing information?

Respond with ONLY this JSON format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "message": "Brief explanation",
    "missing_elements": ["List anything missing"],
    "coverage_percentage": 0-100
}}"""


QUALITY_CHECK_PROMPT = """Evaluate the overall QUALITY of the OUTPUT.

OUTPUT:
{output}

Evaluate:
1. Clarity: Is it well-written and easy to understand?
2. Accuracy: Does the information appear correct and well-reasoned?
3. Structure: Is it well-organized?
4. Usefulness: Would this be helpful to the user?

Respond with ONLY this JSON format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "message": "Brief summary",
    "quality_aspects": {{
        "clarity": 0.0-1.0,
        "accuracy": 0.0-1.0,
        "structure": 0.0-1.0,
        "usefulness": 0.0-1.0
    }},
    "suggestions": ["List improvement suggestions"]
}}"""


ACCURACY_CHECK_PROMPT = """Evaluate if the OUTPUT is ACCURATE given the provided CONTEXT.

TASK:
{task}

CONTEXT (source of truth):
{context}

OUTPUT:
{output}

Evaluate:
1. Does the output accurately represent information from the context?
2. Are there any claims not supported by the context?
3. Are there any factual errors or hallucinations?

Respond with ONLY this JSON format:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "message": "Brief explanation",
    "unsupported_claims": ["List any claims not in context"],
    "accuracy_issues": ["List any accuracy problems"]
}}"""


IMPROVEMENT_PROMPT = """Improve the OUTPUT based on the identified ISSUES.

ORIGINAL TASK:
{task}

CURRENT OUTPUT:
{output}

ISSUES TO FIX:
{issues}

SUGGESTIONS:
{suggestions}

Please provide an improved version that:
1. Addresses all identified issues
2. Implements the suggestions where appropriate
3. Maintains the good aspects of the original output

Provide the improved output directly without any preamble or explanation."""


# =============================================================================
# VerificationProtocol Class
# =============================================================================


class VerificationProtocol:
    """
    Boris Step 13: Verification Protocol for output quality assurance.

    "Give Claude a way to verify its work - 2-3x quality improvement"

    Workflow:
    1. Self-Check: Does output address the original task?
    2. Constraint-Check: All requirements fulfilled?
    3. Completeness-Check: Is the answer complete?
    4. Quality-Check: Overall output quality assessment

    Usage:
        verifier = VerificationProtocol(provider)
        result = await verifier.verify(task, output, constraints)

        if not result.passed:
            # Iterate with feedback
            improved = await verifier.iterate_until_verified(
                task, output, constraints, max_iterations=3
            )
    """

    def __init__(
        self,
        provider: BaseProvider,
        min_confidence: float = 0.7,
        required_checks: list[VerificationCheckType] | None = None,
    ) -> None:
        """
        Initialize VerificationProtocol.

        Args:
            provider: LLM provider for verification calls
            min_confidence: Minimum confidence to pass (0.0-1.0)
            required_checks: Which checks must pass (default: all)

        Raises:
            ValueError: If min_confidence is not in valid range
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self.provider = provider
        self.min_confidence = min_confidence
        self.required_checks = required_checks or [
            VerificationCheckType.TASK_ALIGNMENT,
            VerificationCheckType.COMPLETENESS,
            VerificationCheckType.QUALITY_CHECK,
        ]

        logger.debug(
            "VerificationProtocol initialized",
            min_confidence=min_confidence,
            required_checks=[c.value for c in self.required_checks],
        )

    async def verify(
        self,
        task: str,
        output: str,
        constraints: list[str] | None = None,
        context: str | None = None,
    ) -> VerificationResult:
        """
        Verify that output meets task requirements.

        Args:
            task: Original task/question
            output: Generated output to verify
            constraints: Optional constraints to check
            context: Optional original context for accuracy check

        Returns:
            VerificationResult with detailed check results

        Raises:
            ProviderError: If LLM calls fail
            StrategyExecutionError: If verification logic fails
        """
        start_time = time.time()
        checks: list[VerificationCheck] = []
        issues: list[str] = []
        suggestions: list[str] = []

        logger.info(
            "Starting verification",
            task_length=len(task),
            output_length=len(output),
            has_constraints=constraints is not None,
            has_context=context is not None,
        )

        try:
            # Run checks based on required_checks and available data
            check_tasks: list[Coroutine[Any, Any, VerificationCheck]] = []

            # Task alignment is always checked
            if VerificationCheckType.TASK_ALIGNMENT in self.required_checks:
                check_tasks.append(self._check_task_alignment(task, output))

            # Constraint check only if constraints provided
            if VerificationCheckType.CONSTRAINT_CHECK in self.required_checks and constraints:
                check_tasks.append(self._check_constraints(output, constraints))

            # Completeness check
            if VerificationCheckType.COMPLETENESS in self.required_checks:
                check_tasks.append(self._check_completeness(task, output))

            # Quality check
            if VerificationCheckType.QUALITY_CHECK in self.required_checks:
                check_tasks.append(self._check_quality(output))

            # Accuracy check only if context provided
            if VerificationCheckType.ACCURACY in self.required_checks and context:
                check_tasks.append(self._check_accuracy(task, output, context))

            # Run all checks concurrently
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

            # Process results
            for check_result in check_results:
                if isinstance(check_result, BaseException):
                    logger.warning(
                        "Verification check failed",
                        error=str(check_result),
                    )
                    continue

                # Type assertion for mypy - check_result is VerificationCheck
                check: VerificationCheck = check_result
                checks.append(check)

                # Collect issues from failed checks
                if not check.passed:
                    issues.append(f"[{check.check_type.value}] {check.message}")
                    if check.details:
                        if "missing_elements" in check.details:
                            issues.extend(check.details["missing_elements"])
                        if "unmet_constraints" in check.details:
                            issues.extend(check.details["unmet_constraints"])
                        if "unsupported_claims" in check.details:
                            issues.extend(check.details["unsupported_claims"])

                # Collect suggestions
                if check.details and "suggestions" in check.details:
                    suggestions.extend(check.details["suggestions"])

            # Calculate overall scores
            if checks:
                overall_score = sum(c.score for c in checks) / len(checks)
                passed_count = sum(1 for c in checks if c.passed)
                confidence = passed_count / len(checks)

                # Must pass all required checks
                required_passed = all(
                    c.passed for c in checks if c.check_type in self.required_checks
                )
                passed = required_passed and confidence >= self.min_confidence
            else:
                overall_score = 0.0
                confidence = 0.0
                passed = False

            execution_time = time.time() - start_time

            verification_result = VerificationResult(
                passed=passed,
                confidence=confidence,
                overall_score=overall_score,
                checks=checks,
                issues=issues,
                suggestions=suggestions,
                execution_time=execution_time,
            )

            logger.info(
                "Verification completed",
                passed=passed,
                confidence=round(confidence, 3),
                overall_score=round(overall_score, 3),
                check_count=len(checks),
                issue_count=len(issues),
                execution_time_ms=round(execution_time * 1000, 2),
            )

            return verification_result

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            raise StrategyExecutionError(
                strategy="verification",
                message=f"Verification failed: {str(e)}",
                cause=e,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _check_task_alignment(
        self,
        task: str,
        output: str,
    ) -> VerificationCheck:
        """
        Check if output addresses the task.

        Args:
            task: Original task/question
            output: Generated output to verify

        Returns:
            VerificationCheck with alignment assessment
        """
        logger.debug("Running task alignment check")

        prompt = TASK_ALIGNMENT_PROMPT.format(task=task, output=output)
        response = await self._call_verification_llm(prompt)

        try:
            data = self._parse_json_response(response)
            return VerificationCheck(
                check_type=VerificationCheckType.TASK_ALIGNMENT,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.0)),
                message=data.get("message", "No message provided"),
                details={
                    "issues": data.get("issues", []),
                    **data.get("details", {}),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse task alignment response", error=str(e))
            return self._create_fallback_check(
                VerificationCheckType.TASK_ALIGNMENT,
                response,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _check_constraints(
        self,
        output: str,
        constraints: list[str],
    ) -> VerificationCheck:
        """
        Check if all constraints are met.

        Args:
            output: Generated output to verify
            constraints: List of constraints to check

        Returns:
            VerificationCheck with constraint assessment
        """
        logger.debug("Running constraint check", constraint_count=len(constraints))

        constraints_text = "\n".join(f"- {c}" for c in constraints)
        prompt = CONSTRAINT_CHECK_PROMPT.format(
            output=output,
            constraints=constraints_text,
        )
        response = await self._call_verification_llm(prompt)

        try:
            data = self._parse_json_response(response)
            return VerificationCheck(
                check_type=VerificationCheckType.CONSTRAINT_CHECK,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.0)),
                message=data.get("message", "No message provided"),
                details={
                    "constraint_results": data.get("constraint_results", []),
                    "unmet_constraints": data.get("unmet_constraints", []),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse constraint check response", error=str(e))
            return self._create_fallback_check(
                VerificationCheckType.CONSTRAINT_CHECK,
                response,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _check_completeness(
        self,
        task: str,
        output: str,
    ) -> VerificationCheck:
        """
        Check if answer is complete.

        Args:
            task: Original task/question
            output: Generated output to verify

        Returns:
            VerificationCheck with completeness assessment
        """
        logger.debug("Running completeness check")

        prompt = COMPLETENESS_PROMPT.format(task=task, output=output)
        response = await self._call_verification_llm(prompt)

        try:
            data = self._parse_json_response(response)
            return VerificationCheck(
                check_type=VerificationCheckType.COMPLETENESS,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.0)),
                message=data.get("message", "No message provided"),
                details={
                    "missing_elements": data.get("missing_elements", []),
                    "coverage_percentage": data.get("coverage_percentage", 0),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse completeness response", error=str(e))
            return self._create_fallback_check(
                VerificationCheckType.COMPLETENESS,
                response,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _check_quality(
        self,
        output: str,
    ) -> VerificationCheck:
        """
        Check overall output quality.

        Args:
            output: Generated output to verify

        Returns:
            VerificationCheck with quality assessment
        """
        logger.debug("Running quality check")

        prompt = QUALITY_CHECK_PROMPT.format(output=output)
        response = await self._call_verification_llm(prompt)

        try:
            data = self._parse_json_response(response)
            quality_aspects = data.get("quality_aspects", {})
            return VerificationCheck(
                check_type=VerificationCheckType.QUALITY_CHECK,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.0)),
                message=data.get("message", "No message provided"),
                details={
                    "quality_aspects": quality_aspects,
                    "suggestions": data.get("suggestions", []),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse quality check response", error=str(e))
            return self._create_fallback_check(
                VerificationCheckType.QUALITY_CHECK,
                response,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _check_accuracy(
        self,
        task: str,
        output: str,
        context: str,
    ) -> VerificationCheck:
        """
        Check if output is accurate against provided context.

        Args:
            task: Original task/question
            output: Generated output to verify
            context: Source context to verify against

        Returns:
            VerificationCheck with accuracy assessment
        """
        logger.debug("Running accuracy check")

        prompt = ACCURACY_CHECK_PROMPT.format(
            task=task,
            context=context,
            output=output,
        )
        response = await self._call_verification_llm(prompt)

        try:
            data = self._parse_json_response(response)
            return VerificationCheck(
                check_type=VerificationCheckType.ACCURACY,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.0)),
                message=data.get("message", "No message provided"),
                details={
                    "unsupported_claims": data.get("unsupported_claims", []),
                    "accuracy_issues": data.get("accuracy_issues", []),
                },
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse accuracy check response", error=str(e))
            return self._create_fallback_check(
                VerificationCheckType.ACCURACY,
                response,
            )

    async def iterate_until_verified(
        self,
        task: str,
        initial_output: str,
        constraints: list[str] | None = None,
        context: str | None = None,
        max_iterations: int = 3,
        improvement_callback: Callable[[str, str, list[str]], Coroutine[Any, Any, str]]
        | None = None,
    ) -> tuple[str, VerificationResult]:
        """
        Iterate on output until verification passes.

        Args:
            task: Original task
            initial_output: First attempt output
            constraints: Constraints to meet
            context: Original context
            max_iterations: Max improvement iterations
            improvement_callback: Async function to generate improved output
                                  Signature: async def(task, output, issues) -> str

        Returns:
            Tuple of (final_output, final_verification_result)

        Raises:
            StrategyExecutionError: If max iterations reached without passing
        """
        current_output = initial_output
        iteration = 0

        logger.info(
            "Starting iterative verification",
            max_iterations=max_iterations,
        )

        while iteration < max_iterations:
            iteration += 1

            logger.debug(
                "Verification iteration",
                iteration=iteration,
                output_length=len(current_output),
            )

            # Verify current output
            result = await self.verify(
                task=task,
                output=current_output,
                constraints=constraints,
                context=context,
            )
            result.iteration = iteration

            if result.passed:
                logger.info(
                    "Verification passed",
                    iteration=iteration,
                    confidence=round(result.confidence, 3),
                )
                return current_output, result

            # If not passed and more iterations available, improve
            if iteration < max_iterations:
                logger.debug(
                    "Verification failed, generating improvement",
                    issues=result.issues,
                )

                if improvement_callback:
                    # Use provided callback
                    current_output = await improvement_callback(task, current_output, result.issues)
                else:
                    # Use default improvement method
                    improvement_prompt = await self._generate_improvement_prompt(
                        task=task,
                        output=current_output,
                        issues=result.issues,
                        suggestions=result.suggestions,
                    )
                    current_output = await self._call_improvement_llm(improvement_prompt)

        # Max iterations reached
        logger.warning(
            "Max verification iterations reached",
            max_iterations=max_iterations,
            final_confidence=round(result.confidence, 3),
        )

        return current_output, result

    async def _generate_improvement_prompt(
        self,
        task: str,
        output: str,
        issues: list[str],
        suggestions: list[str],
    ) -> str:
        """
        Generate prompt for output improvement based on verification feedback.

        Args:
            task: Original task
            output: Current output to improve
            issues: List of identified issues
            suggestions: List of improvement suggestions

        Returns:
            Formatted improvement prompt
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues) or "None identified"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "None provided"

        return IMPROVEMENT_PROMPT.format(
            task=task,
            output=output,
            issues=issues_text,
            suggestions=suggestions_text,
        )

    def get_verification_summary(
        self,
        result: VerificationResult,
    ) -> str:
        """
        Get human-readable summary of verification result.

        Args:
            result: VerificationResult to summarize

        Returns:
            Formatted summary string
        """
        status = "PASSED" if result.passed else "FAILED"

        lines = [
            f"Verification {status}",
            f"  Confidence: {result.confidence:.1%}",
            f"  Overall Score: {result.overall_score:.1%}",
            f"  Execution Time: {result.execution_time:.2f}s",
            f"  Iteration: {result.iteration}",
            "",
            "Checks:",
        ]

        for check in result.checks:
            check_status = "OK" if check.passed else "FAIL"
            lines.append(
                f"  [{check_status}] {check.check_type.value}: "
                f"{check.score:.1%} - {check.message}"
            )

        if result.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in result.issues:
                lines.append(f"  - {issue}")

        if result.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in result.suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _call_verification_llm(self, prompt: str) -> str:
        """
        Call LLM for verification with appropriate settings.

        Args:
            prompt: Verification prompt to send

        Returns:
            LLM response content

        Raises:
            ProviderError: If LLM call fails
        """
        response = await self.provider.complete(
            messages=[Message(role="user", content=prompt)],
            system=VERIFICATION_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        return response.content

    async def _call_improvement_llm(self, prompt: str) -> str:
        """
        Call LLM for output improvement.

        Args:
            prompt: Improvement prompt to send

        Returns:
            Improved output content

        Raises:
            ProviderError: If LLM call fails
        """
        response = await self.provider.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=4096,
            temperature=0.7,  # Higher temperature for creative improvement
        )
        return response.content

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Try direct parsing first
        try:
            result: dict[str, Any] = json.loads(response)
            return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_pattern, response)
        if matches:
            try:
                result = json.loads(matches[0])
                return result
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
        brace_pattern = r"\{[\s\S]*\}"
        brace_matches = re.findall(brace_pattern, response)
        if brace_matches:
            # Try each match, starting with the largest
            for match in sorted(brace_matches, key=len, reverse=True):
                try:
                    result = json.loads(match)
                    return result
                except json.JSONDecodeError:
                    continue

        # Last resort: raise error
        raise json.JSONDecodeError(
            "Could not extract JSON from response",
            response,
            0,
        )

    def _create_fallback_check(
        self,
        check_type: VerificationCheckType,
        response: str,
    ) -> VerificationCheck:
        """
        Create a fallback check when JSON parsing fails.

        Uses simple heuristics to determine pass/fail based on response content.

        Args:
            check_type: Type of check being performed
            response: Raw response that failed to parse

        Returns:
            VerificationCheck with heuristic-based assessment
        """
        response_lower = response.lower()

        # Simple heuristics
        positive_indicators = [
            "pass",
            "correct",
            "complete",
            "good",
            "yes",
            "meets",
            "satisfied",
            "adequate",
            "sufficient",
        ]
        negative_indicators = [
            "fail",
            "incorrect",
            "incomplete",
            "bad",
            "no",
            "missing",
            "unsatisfied",
            "inadequate",
            "insufficient",
        ]

        positive_count = sum(1 for ind in positive_indicators if ind in response_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in response_lower)

        passed = positive_count > negative_count
        score = 0.6 if passed else 0.4

        return VerificationCheck(
            check_type=check_type,
            passed=passed,
            score=score,
            message="Fallback heuristic assessment (JSON parsing failed)",
            details={"raw_response": response[:500]},
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_verify(
    provider: BaseProvider,
    task: str,
    output: str,
    min_confidence: float = 0.7,
) -> bool:
    """
    Quick verification check - returns simple pass/fail.

    Args:
        provider: LLM provider
        task: Original task
        output: Output to verify
        min_confidence: Minimum confidence threshold

    Returns:
        True if verification passed, False otherwise
    """
    verifier = VerificationProtocol(provider, min_confidence=min_confidence)
    result = await verifier.verify(task, output)
    return result.passed


async def verified_completion(
    provider: BaseProvider,
    task: str,
    initial_output: str,
    max_iterations: int = 3,
    min_confidence: float = 0.7,
) -> tuple[str, VerificationResult]:
    """
    Get verified completion with automatic iteration.

    Args:
        provider: LLM provider
        task: Original task
        initial_output: First attempt output
        max_iterations: Maximum improvement iterations
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (final_output, verification_result)
    """
    verifier = VerificationProtocol(provider, min_confidence=min_confidence)
    return await verifier.iterate_until_verified(
        task=task,
        initial_output=initial_output,
        max_iterations=max_iterations,
    )
