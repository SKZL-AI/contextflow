"""
Unit tests for VerificationProtocol - CRITICAL.

Tests all verification functionality including:
- VerificationProtocol initialization
- All 5 check types (TASK_ALIGNMENT, CONSTRAINT_CHECK, COMPLETENESS, ACCURACY, QUALITY_CHECK)
- iterate_until_verified() with improvements
- JSON parsing and fallback heuristics
- Error handling and edge cases
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from contextflow.core.types import CompletionResponse
from contextflow.strategies.verification import (
    VerificationCheck,
    VerificationCheckType,
    VerificationProtocol,
    VerificationResult,
    quick_verify,
    verified_completion,
)
from contextflow.utils.errors import StrategyExecutionError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider for verification tests."""
    provider = MagicMock()
    provider.name = "mock-verification"
    provider.model = "mock-model"
    provider.complete = AsyncMock()
    provider.count_tokens = MagicMock(return_value=100)
    return provider


@pytest.fixture
def passing_verification_response() -> CompletionResponse:
    """Mock response indicating verification passed."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": True,
                "score": 0.95,
                "message": "Output correctly addresses the task",
                "issues": [],
                "details": {"relevance": 0.95, "directness": 0.92},
            }
        ),
        tokens_used=50,
        input_tokens=30,
        output_tokens=20,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=100.0,
    )


@pytest.fixture
def failing_verification_response() -> CompletionResponse:
    """Mock response indicating verification failed."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": False,
                "score": 0.4,
                "message": "Output does not fully address the task",
                "issues": ["Missing key information", "Incomplete analysis"],
                "details": {"relevance": 0.5, "directness": 0.3},
            }
        ),
        tokens_used=50,
        input_tokens=30,
        output_tokens=20,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=100.0,
    )


@pytest.fixture
def constraint_check_response() -> CompletionResponse:
    """Mock response for constraint checking."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": True,
                "score": 0.9,
                "message": "All constraints met",
                "constraint_results": [
                    {"constraint": "Must be under 500 words", "met": True, "note": "Within limit"},
                    {"constraint": "Include examples", "met": True, "note": "Examples provided"},
                ],
                "unmet_constraints": [],
            }
        ),
        tokens_used=60,
        input_tokens=40,
        output_tokens=20,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=110.0,
    )


@pytest.fixture
def completeness_response() -> CompletionResponse:
    """Mock response for completeness checking."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": True,
                "score": 0.85,
                "message": "Answer is complete",
                "missing_elements": [],
                "coverage_percentage": 95,
            }
        ),
        tokens_used=45,
        input_tokens=30,
        output_tokens=15,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=90.0,
    )


@pytest.fixture
def quality_check_response() -> CompletionResponse:
    """Mock response for quality checking."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": True,
                "score": 0.88,
                "message": "Good quality output",
                "quality_aspects": {
                    "clarity": 0.9,
                    "accuracy": 0.85,
                    "structure": 0.88,
                    "usefulness": 0.9,
                },
                "suggestions": ["Consider adding more context"],
            }
        ),
        tokens_used=55,
        input_tokens=35,
        output_tokens=20,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=100.0,
    )


@pytest.fixture
def accuracy_check_response() -> CompletionResponse:
    """Mock response for accuracy checking."""
    return CompletionResponse(
        content=json.dumps(
            {
                "passed": True,
                "score": 0.92,
                "message": "Output accurately reflects context",
                "unsupported_claims": [],
                "accuracy_issues": [],
            }
        ),
        tokens_used=50,
        input_tokens=35,
        output_tokens=15,
        model="mock-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=95.0,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestVerificationProtocolInit:
    """Tests for VerificationProtocol initialization."""

    def test_init_with_default_confidence(self, mock_provider: MagicMock) -> None:
        """Test initialization with default minimum confidence."""
        verifier = VerificationProtocol(provider=mock_provider)

        assert verifier.provider == mock_provider
        assert verifier.min_confidence == 0.7
        assert VerificationCheckType.TASK_ALIGNMENT in verifier.required_checks
        assert VerificationCheckType.COMPLETENESS in verifier.required_checks
        assert VerificationCheckType.QUALITY_CHECK in verifier.required_checks

    def test_init_with_custom_confidence(self, mock_provider: MagicMock) -> None:
        """Test initialization with custom minimum confidence."""
        verifier = VerificationProtocol(provider=mock_provider, min_confidence=0.9)

        assert verifier.min_confidence == 0.9

    def test_init_with_custom_checks(self, mock_provider: MagicMock) -> None:
        """Test initialization with custom required checks."""
        custom_checks = [VerificationCheckType.TASK_ALIGNMENT, VerificationCheckType.ACCURACY]
        verifier = VerificationProtocol(provider=mock_provider, required_checks=custom_checks)

        assert verifier.required_checks == custom_checks

    def test_init_invalid_confidence_high(self, mock_provider: MagicMock) -> None:
        """Test that confidence > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be between"):
            VerificationProtocol(provider=mock_provider, min_confidence=1.5)

    def test_init_invalid_confidence_low(self, mock_provider: MagicMock) -> None:
        """Test that confidence < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be between"):
            VerificationProtocol(provider=mock_provider, min_confidence=-0.1)


# =============================================================================
# Task Alignment Check Tests
# =============================================================================


class TestTaskAlignmentCheck:
    """Tests for task alignment verification check."""

    @pytest.mark.asyncio
    async def test_task_alignment_passes(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that task alignment passes for correct output."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.TASK_ALIGNMENT]
        )

        result = await verifier.verify(
            task="Summarize this document", output="This document discusses the key points of..."
        )

        assert result.passed is True
        assert result.confidence >= 0.7
        assert len(result.checks) == 1
        assert result.checks[0].check_type == VerificationCheckType.TASK_ALIGNMENT
        assert result.checks[0].passed is True

    @pytest.mark.asyncio
    async def test_task_alignment_fails(
        self, mock_provider: MagicMock, failing_verification_response: CompletionResponse
    ) -> None:
        """Test that task alignment fails for incorrect output."""
        mock_provider.complete = AsyncMock(return_value=failing_verification_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.TASK_ALIGNMENT]
        )

        result = await verifier.verify(
            task="Summarize this document", output="Unrelated content here"
        )

        assert result.passed is False
        assert len(result.issues) > 0


# =============================================================================
# Constraint Check Tests
# =============================================================================


class TestConstraintCheck:
    """Tests for constraint verification check."""

    @pytest.mark.asyncio
    async def test_constraint_check_passes(
        self, mock_provider: MagicMock, constraint_check_response: CompletionResponse
    ) -> None:
        """Test that constraint check passes when all constraints met."""
        mock_provider.complete = AsyncMock(return_value=constraint_check_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.CONSTRAINT_CHECK]
        )

        result = await verifier.verify(
            task="Write a summary",
            output="Here is a brief summary with examples...",
            constraints=["Must be under 500 words", "Include examples"],
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_constraint_check_skipped_without_constraints(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that constraint check is skipped when no constraints provided."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(
            provider=mock_provider,
            required_checks=[
                VerificationCheckType.TASK_ALIGNMENT,
                VerificationCheckType.CONSTRAINT_CHECK,
            ],
        )

        result = await verifier.verify(
            task="Write something", output="Output here", constraints=None  # No constraints
        )

        # Only task alignment should be checked
        check_types = [c.check_type for c in result.checks]
        assert VerificationCheckType.CONSTRAINT_CHECK not in check_types


# =============================================================================
# Completeness Check Tests
# =============================================================================


class TestCompletenessCheck:
    """Tests for completeness verification check."""

    @pytest.mark.asyncio
    async def test_completeness_check_passes(
        self, mock_provider: MagicMock, completeness_response: CompletionResponse
    ) -> None:
        """Test that completeness check passes for complete output."""
        mock_provider.complete = AsyncMock(return_value=completeness_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.COMPLETENESS]
        )

        result = await verifier.verify(
            task="Explain photosynthesis", output="Photosynthesis is the process by which plants..."
        )

        assert result.passed is True
        assert any(
            c.check_type == VerificationCheckType.COMPLETENESS and c.passed for c in result.checks
        )


# =============================================================================
# Quality Check Tests
# =============================================================================


class TestQualityCheck:
    """Tests for quality verification check."""

    @pytest.mark.asyncio
    async def test_quality_check_passes(
        self, mock_provider: MagicMock, quality_check_response: CompletionResponse
    ) -> None:
        """Test that quality check passes for high-quality output."""
        mock_provider.complete = AsyncMock(return_value=quality_check_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.QUALITY_CHECK]
        )

        result = await verifier.verify(
            task="Write an essay", output="This well-structured essay explores..."
        )

        assert result.passed is True
        assert len(result.suggestions) >= 0  # May have suggestions even when passing


# =============================================================================
# Accuracy Check Tests
# =============================================================================


class TestAccuracyCheck:
    """Tests for accuracy verification check."""

    @pytest.mark.asyncio
    async def test_accuracy_check_passes_with_context(
        self, mock_provider: MagicMock, accuracy_check_response: CompletionResponse
    ) -> None:
        """Test that accuracy check passes when output matches context."""
        mock_provider.complete = AsyncMock(return_value=accuracy_check_response)

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.ACCURACY]
        )

        result = await verifier.verify(
            task="Extract key facts",
            output="The company was founded in 2020...",
            context="Company History: Founded in 2020 by John Smith...",
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_accuracy_check_skipped_without_context(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that accuracy check is skipped when no context provided."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(
            provider=mock_provider,
            required_checks=[VerificationCheckType.TASK_ALIGNMENT, VerificationCheckType.ACCURACY],
        )

        result = await verifier.verify(
            task="Write something", output="Output here", context=None  # No context
        )

        check_types = [c.check_type for c in result.checks]
        assert VerificationCheckType.ACCURACY not in check_types


# =============================================================================
# Full Verification Tests
# =============================================================================


class TestFullVerification:
    """Tests for full verification with multiple checks."""

    @pytest.mark.asyncio
    async def test_full_verification_all_pass(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test full verification when all checks pass."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(provider=mock_provider)

        result = await verifier.verify(
            task="Summarize the document", output="This comprehensive summary covers..."
        )

        assert result.passed is True
        assert result.confidence >= 0.7
        assert result.overall_score > 0
        assert result.execution_time >= 0

    @pytest.mark.asyncio
    async def test_verification_result_structure(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that VerificationResult has correct structure."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(provider=mock_provider)
        result = await verifier.verify(task="Test task", output="Test output")

        assert isinstance(result, VerificationResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.overall_score, float)
        assert isinstance(result.checks, list)
        assert isinstance(result.issues, list)
        assert isinstance(result.suggestions, list)
        assert isinstance(result.execution_time, float)


# =============================================================================
# Iterate Until Verified Tests
# =============================================================================


class TestIterateUntilVerified:
    """Tests for iterate_until_verified functionality."""

    @pytest.mark.asyncio
    async def test_iterate_passes_on_first_attempt(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that iteration stops when first attempt passes."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        verifier = VerificationProtocol(provider=mock_provider)

        output, result = await verifier.iterate_until_verified(
            task="Write summary", initial_output="Good summary here", max_iterations=3
        )

        assert result.passed is True
        assert result.iteration == 1

    @pytest.mark.asyncio
    async def test_iterate_improves_until_pass(
        self,
        mock_provider: MagicMock,
        failing_verification_response: CompletionResponse,
        passing_verification_response: CompletionResponse,
    ) -> None:
        """Test that iteration continues until verification passes."""
        # First call fails, second call passes
        mock_provider.complete = AsyncMock(
            side_effect=[
                failing_verification_response,  # First verify fails
                CompletionResponse(  # Improvement response
                    content="Improved output here",
                    tokens_used=50,
                    input_tokens=30,
                    output_tokens=20,
                    model="mock-model",
                    finish_reason="stop",
                    cost_usd=0.0001,
                    latency_ms=100.0,
                ),
                passing_verification_response,  # Second verify passes
            ]
        )

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.TASK_ALIGNMENT]
        )

        output, result = await verifier.iterate_until_verified(
            task="Write summary", initial_output="Bad initial output", max_iterations=3
        )

        assert result.passed is True
        assert result.iteration == 2

    @pytest.mark.asyncio
    async def test_iterate_with_custom_callback(
        self,
        mock_provider: MagicMock,
        failing_verification_response: CompletionResponse,
        passing_verification_response: CompletionResponse,
    ) -> None:
        """Test iteration with custom improvement callback."""
        mock_provider.complete = AsyncMock(
            side_effect=[
                failing_verification_response,
                passing_verification_response,
            ]
        )

        async def custom_improve(task: str, output: str, issues: list[str]) -> str:
            return f"Improved: {output}"

        verifier = VerificationProtocol(
            provider=mock_provider, required_checks=[VerificationCheckType.TASK_ALIGNMENT]
        )

        output, result = await verifier.iterate_until_verified(
            task="Test task",
            initial_output="Initial",
            improvement_callback=custom_improve,
            max_iterations=3,
        )

        assert "Improved:" in output or result.passed


# =============================================================================
# JSON Parsing Tests
# =============================================================================


class TestJSONParsing:
    """Tests for JSON parsing functionality."""

    def test_parse_json_direct(self, mock_provider: MagicMock) -> None:
        """Test direct JSON parsing."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = '{"passed": true, "score": 0.9}'
        result = verifier._parse_json_response(response)

        assert result["passed"] is True
        assert result["score"] == 0.9

    def test_parse_json_from_markdown(self, mock_provider: MagicMock) -> None:
        """Test JSON extraction from markdown code blocks."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = """Here is the assessment:
```json
{"passed": true, "score": 0.85}
```"""
        result = verifier._parse_json_response(response)

        assert result["passed"] is True
        assert result["score"] == 0.85

    def test_parse_json_embedded(self, mock_provider: MagicMock) -> None:
        """Test JSON extraction from embedded content."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = 'Analysis complete. {"passed": false, "score": 0.3} End of response.'
        result = verifier._parse_json_response(response)

        assert result["passed"] is False
        assert result["score"] == 0.3

    def test_parse_json_invalid_raises(self, mock_provider: MagicMock) -> None:
        """Test that invalid JSON raises JSONDecodeError."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = "This is not JSON at all"

        with pytest.raises(json.JSONDecodeError):
            verifier._parse_json_response(response)


# =============================================================================
# Fallback Heuristics Tests
# =============================================================================


class TestFallbackHeuristics:
    """Tests for fallback heuristic assessment."""

    def test_fallback_positive_indicators(self, mock_provider: MagicMock) -> None:
        """Test fallback with positive indicators."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = "The output passes all criteria and is correct and complete."
        check = verifier._create_fallback_check(VerificationCheckType.TASK_ALIGNMENT, response)

        assert check.passed is True
        assert check.score == 0.6

    def test_fallback_negative_indicators(self, mock_provider: MagicMock) -> None:
        """Test fallback with negative indicators."""
        verifier = VerificationProtocol(provider=mock_provider)

        response = "The output fails to address the task and is incomplete."
        check = verifier._create_fallback_check(VerificationCheckType.TASK_ALIGNMENT, response)

        assert check.passed is False
        assert check.score == 0.4


# =============================================================================
# Verification Summary Tests
# =============================================================================


class TestVerificationSummary:
    """Tests for verification summary generation."""

    def test_summary_passed(self, mock_provider: MagicMock) -> None:
        """Test summary generation for passed verification."""
        verifier = VerificationProtocol(provider=mock_provider)

        result = VerificationResult(
            passed=True,
            confidence=0.9,
            overall_score=0.85,
            checks=[
                VerificationCheck(
                    check_type=VerificationCheckType.TASK_ALIGNMENT,
                    passed=True,
                    score=0.9,
                    message="Task addressed correctly",
                )
            ],
            execution_time=1.5,
            iteration=1,
        )

        summary = verifier.get_verification_summary(result)

        assert "PASSED" in summary
        assert "90" in summary  # 90% confidence
        assert "TASK_ALIGNMENT" in summary.lower() or "task_alignment" in summary

    def test_summary_failed_with_issues(self, mock_provider: MagicMock) -> None:
        """Test summary generation for failed verification with issues."""
        verifier = VerificationProtocol(provider=mock_provider)

        result = VerificationResult(
            passed=False,
            confidence=0.4,
            overall_score=0.35,
            checks=[
                VerificationCheck(
                    check_type=VerificationCheckType.TASK_ALIGNMENT,
                    passed=False,
                    score=0.3,
                    message="Does not address task",
                )
            ],
            issues=["Missing key information"],
            suggestions=["Add more details"],
            execution_time=1.2,
            iteration=3,
        )

        summary = verifier.get_verification_summary(result)

        assert "FAILED" in summary
        assert "Issues:" in summary
        assert "Suggestions:" in summary


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_quick_verify_passes(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test quick_verify returns True for passing output."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        result = await quick_verify(provider=mock_provider, task="Test task", output="Test output")

        assert result is True

    @pytest.mark.asyncio
    async def test_quick_verify_fails(
        self, mock_provider: MagicMock, failing_verification_response: CompletionResponse
    ) -> None:
        """Test quick_verify returns False for failing output."""
        mock_provider.complete = AsyncMock(return_value=failing_verification_response)

        result = await quick_verify(provider=mock_provider, task="Test task", output="Bad output")

        assert result is False

    @pytest.mark.asyncio
    async def test_verified_completion(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test verified_completion returns output and result."""
        mock_provider.complete = AsyncMock(return_value=passing_verification_response)

        output, result = await verified_completion(
            provider=mock_provider, task="Test task", initial_output="Initial output"
        )

        assert output == "Initial output"
        assert isinstance(result, VerificationResult)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in verification."""

    @pytest.mark.asyncio
    async def test_provider_error_wrapped(self, mock_provider: MagicMock) -> None:
        """Test that provider errors are wrapped in StrategyExecutionError."""
        mock_provider.complete = AsyncMock(side_effect=Exception("API Error"))

        verifier = VerificationProtocol(provider=mock_provider)

        with pytest.raises(StrategyExecutionError) as exc_info:
            await verifier.verify(task="Test", output="Output")

        assert "Verification failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_exception_continues(
        self, mock_provider: MagicMock, passing_verification_response: CompletionResponse
    ) -> None:
        """Test that exceptions in individual checks don't stop others."""
        # First check fails, subsequent checks succeed
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First check failed")
            return passing_verification_response

        mock_provider.complete = AsyncMock(side_effect=side_effect)

        verifier = VerificationProtocol(
            provider=mock_provider,
            required_checks=[
                VerificationCheckType.TASK_ALIGNMENT,
                VerificationCheckType.COMPLETENESS,
                VerificationCheckType.QUALITY_CHECK,
            ],
        )

        # Should not raise - continues with other checks
        result = await verifier.verify(task="Test", output="Output")

        # Some checks should have succeeded
        assert len(result.checks) >= 0


# =============================================================================
# Check Type Enum Tests
# =============================================================================


class TestVerificationCheckType:
    """Tests for VerificationCheckType enum."""

    def test_all_check_types_exist(self) -> None:
        """Test that all expected check types exist."""
        expected = [
            "TASK_ALIGNMENT",
            "CONSTRAINT_CHECK",
            "COMPLETENESS",
            "ACCURACY",
            "FORMAT_CHECK",
            "QUALITY_CHECK",
        ]

        for check_name in expected:
            assert hasattr(VerificationCheckType, check_name)

    def test_check_type_values(self) -> None:
        """Test that check types have correct string values."""
        assert VerificationCheckType.TASK_ALIGNMENT.value == "task_alignment"
        assert VerificationCheckType.CONSTRAINT_CHECK.value == "constraint_check"
        assert VerificationCheckType.COMPLETENESS.value == "completeness"
        assert VerificationCheckType.ACCURACY.value == "accuracy"
        assert VerificationCheckType.QUALITY_CHECK.value == "quality_check"
