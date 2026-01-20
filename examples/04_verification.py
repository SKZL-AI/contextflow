#!/usr/bin/env python
"""
Verification Protocol Example.

This example demonstrates ContextFlow's verification capabilities
(Boris Step 13: "Give Claude a way to verify its work"):
- Basic verification of outputs
- Verification with constraints
- Iterative verification loop
- Custom verification callbacks
- Verification result analysis

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable
    - Install contextflow: pip install -e .

Run:
    python examples/04_verification.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow
from contextflow.strategies.verification import (
    VerificationProtocol,
    VerificationCheckType,
    VerificationResult,
    quick_verify,
    verified_completion,
)
from contextflow.providers.factory import get_provider


# =============================================================================
# Sample Data
# =============================================================================

TECHNICAL_DOCUMENT = """
# API Authentication Guide

## Overview

Our API uses OAuth 2.0 for authentication. All requests must include
a valid access token in the Authorization header.

## Authentication Flow

1. Register your application to get client credentials
2. Redirect users to our authorization endpoint
3. Exchange the authorization code for tokens
4. Use the access token for API requests
5. Refresh tokens before they expire

## Token Types

- **Access Token**: Short-lived (1 hour), used for API calls
- **Refresh Token**: Long-lived (30 days), used to get new access tokens
- **ID Token**: Contains user identity claims (JWT format)

## Security Requirements

- All tokens must be stored securely
- Never expose tokens in client-side code
- Use HTTPS for all API communications
- Implement token rotation on detection of compromise

## Rate Limits

- 100 requests per minute per user
- 1000 requests per hour per application
- Exponential backoff required on 429 responses

## Error Codes

| Code | Description |
|------|-------------|
| 401  | Invalid or expired token |
| 403  | Insufficient permissions |
| 429  | Rate limit exceeded |
"""


# =============================================================================
# Example Functions
# =============================================================================


async def basic_verification_example() -> None:
    """
    Basic verification of LLM output.

    Shows how to verify that an output meets basic quality criteria.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Verification")
    print("=" * 60)

    provider = get_provider("claude")

    # Create verifier
    verifier = VerificationProtocol(
        provider=provider,
        min_confidence=0.7,  # 70% confidence threshold
    )

    # Task and output to verify
    task = "List the three types of tokens mentioned in the document"
    output = """
    The three types of tokens mentioned are:
    1. Access Token - Short-lived, used for API calls
    2. Refresh Token - Long-lived, used to get new access tokens
    3. ID Token - Contains user identity claims in JWT format
    """

    print(f"\nTask: {task}")
    print(f"Output to verify:\n{output}")

    # Verify the output
    result = await verifier.verify(
        task=task,
        output=output,
        context=TECHNICAL_DOCUMENT,  # Original context for accuracy check
    )

    print(f"\n--- Verification Results ---")
    print(f"Passed: {result.passed}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Execution Time: {result.execution_time:.2f}s")

    print(f"\nIndividual Checks:")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.check_type.value}: {check.score:.1%}")
        print(f"         {check.message}")


async def verification_with_constraints_example() -> None:
    """
    Verification with specific constraints.

    Shows how to verify that output meets specific requirements.
    """
    print("\n" + "=" * 60)
    print("Example 2: Verification with Constraints")
    print("=" * 60)

    provider = get_provider("claude")

    verifier = VerificationProtocol(
        provider=provider,
        min_confidence=0.8,
        required_checks=[
            VerificationCheckType.TASK_ALIGNMENT,
            VerificationCheckType.CONSTRAINT_CHECK,
            VerificationCheckType.COMPLETENESS,
        ],
    )

    task = "Explain the OAuth 2.0 authentication flow"
    output = """
    The OAuth 2.0 authentication flow works as follows:
    First, register your app to get credentials. Then redirect users
    to authorize. After authorization, exchange the code for tokens.
    Use the access token for API requests.
    """

    constraints = [
        "Must mention all 5 steps in the authentication flow",
        "Must explain each step briefly",
        "Must be under 200 words",
    ]

    print(f"\nTask: {task}")
    print(f"Constraints: {constraints}")
    print(f"\nOutput:\n{output}")

    result = await verifier.verify(
        task=task,
        output=output,
        constraints=constraints,
        context=TECHNICAL_DOCUMENT,
    )

    print(f"\n--- Constraint Verification ---")
    print(f"Passed: {result.passed}")
    print(f"Confidence: {result.confidence:.1%}")

    if result.issues:
        print(f"\nIssues Found:")
        for issue in result.issues:
            print(f"  - {issue}")

    if result.suggestions:
        print(f"\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")


async def iterative_verification_example() -> None:
    """
    Iterative verification with automatic improvement.

    Shows the verification loop that improves output until it passes.
    """
    print("\n" + "=" * 60)
    print("Example 3: Iterative Verification Loop")
    print("=" * 60)

    provider = get_provider("claude")

    verifier = VerificationProtocol(
        provider=provider,
        min_confidence=0.8,
    )

    task = "Summarize the security requirements from the document"
    initial_output = "Use HTTPS and store tokens securely."  # Incomplete

    print(f"\nTask: {task}")
    print(f"Initial (incomplete) output: {initial_output}")
    print("\nStarting iterative verification...")

    # Use iterate_until_verified to automatically improve
    final_output, final_result = await verifier.iterate_until_verified(
        task=task,
        initial_output=initial_output,
        context=TECHNICAL_DOCUMENT,
        constraints=[
            "Must cover all 4 security requirements",
            "Must be specific, not vague",
        ],
        max_iterations=3,
    )

    print(f"\n--- Results after {final_result.iteration} iteration(s) ---")
    print(f"Passed: {final_result.passed}")
    print(f"Confidence: {final_result.confidence:.1%}")
    print(f"\nFinal Output:\n{final_output}")


async def quick_verification_example() -> None:
    """
    Quick verification for simple pass/fail checks.

    Shows the simplified API for basic verification needs.
    """
    print("\n" + "=" * 60)
    print("Example 4: Quick Verification API")
    print("=" * 60)

    provider = get_provider("claude")

    # Good output
    good_output = """
    Rate limits for the API:
    - 100 requests per minute per user
    - 1000 requests per hour per application
    - Exponential backoff is required when receiving 429 responses
    """

    # Bad output (contains incorrect information)
    bad_output = """
    Rate limits for the API:
    - 500 requests per minute per user
    - No hourly limits
    - Retry immediately on any error
    """

    task = "What are the API rate limits?"

    # Quick check on good output
    print(f"\n--- Checking Good Output ---")
    is_good_valid = await quick_verify(
        provider=provider,
        task=task,
        output=good_output,
        min_confidence=0.7,
    )
    print(f"Good output verified: {is_good_valid}")

    # Quick check on bad output
    print(f"\n--- Checking Bad Output ---")
    is_bad_valid = await quick_verify(
        provider=provider,
        task=task,
        output=bad_output,
        min_confidence=0.7,
    )
    print(f"Bad output verified: {is_bad_valid}")


async def verification_summary_example() -> None:
    """
    Generate human-readable verification summary.

    Shows how to get a formatted verification report.
    """
    print("\n" + "=" * 60)
    print("Example 5: Verification Summary Report")
    print("=" * 60)

    provider = get_provider("claude")

    verifier = VerificationProtocol(
        provider=provider,
        min_confidence=0.75,
    )

    task = "Explain the error codes and their meanings"
    output = """
    API Error Codes:
    - 401: Your token is invalid or has expired. Get a new one.
    - 403: You don't have permission for this action.
    - 429: You've hit the rate limit. Wait and retry with backoff.
    """

    result = await verifier.verify(
        task=task,
        output=output,
        context=TECHNICAL_DOCUMENT,
    )

    # Get formatted summary
    summary = verifier.get_verification_summary(result)
    print(f"\n{summary}")


async def contextflow_integrated_verification() -> None:
    """
    Verification integrated with ContextFlow processing.

    Shows how verification works automatically within the pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 6: ContextFlow Integrated Verification")
    print("=" * 60)

    # ContextFlow automatically verifies outputs when enabled
    async with ContextFlow() as cf:
        # Process with verification enabled (default)
        result = await cf.process(
            task="Create a quick reference card for authentication",
            context=TECHNICAL_DOCUMENT,
            constraints=[
                "Include token types with lifetimes",
                "Include rate limits",
                "Format as bullet points",
            ],
        )

        print(f"\nStrategy Used: {result.strategy_used.value}")
        print(f"Verification Passed: {result.metadata.get('verification_passed', 'N/A')}")
        print(f"Verification Score: {result.metadata.get('verification_score', 'N/A'):.1%}")
        print(f"Iterations: {result.metadata.get('strategy_iterations', 'N/A')}")

        print(f"\n--- Result ---")
        print(result.answer)

        # Check trajectory for verification steps
        print(f"\n--- Execution Trajectory ---")
        for step in result.trajectory:
            if "verification" in step.step_type.lower():
                print(f"  {step.step_type}: {step.metadata}")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Verification Protocol Examples")
    print("(Boris Step 13: 'Give Claude a way to verify its work')")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples.")
        return

    try:
        await basic_verification_example()
        await verification_with_constraints_example()
        await iterative_verification_example()
        await quick_verification_example()
        await verification_summary_example()
        await contextflow_integrated_verification()

        print("\n" + "=" * 60)
        print("All verification examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
