"""
Strategy implementations for context processing.

This module provides the strategy pattern for handling different
context sizes and complexities:

- GSD (Get Shit Done): <10K tokens, simple to moderate tasks
- RALPH (Recursive Aggregation): 10K-100K tokens, iterative processing
- RLM (Recursive Language Model): >100K tokens, full recursive with sub-agents

All strategies implement the BaseStrategy ABC and include
mandatory verification loops (Boris Step 13).
"""

from contextflow.strategies.base import (
    BaseStrategy,
    CostEstimate,
    StrategyFactory,
    StrategyResult,
    StrategyType,
    VerificationResult,
)
from contextflow.strategies.gsd import GSDStrategy
from contextflow.strategies.ralph import ChunkResult, RALPHStrategy
from contextflow.strategies.rlm import (
    RLM_BASIC_SYSTEM_PROMPT,
    RLM_SYSTEM_PROMPT,
    CodeExecutionResult,
    REPLEnvironment,
    REPLVariable,
    RLMIteration,
    RLMMode,
    RLMState,
    RLMStrategy,
    execute_rlm,
    register_rlm_strategy,
)
from contextflow.strategies.verification import (
    VerificationCheck,
    VerificationCheckType,
    VerificationProtocol,
    quick_verify,
    verified_completion,
)
from contextflow.strategies.verification import (
    VerificationResult as DetailedVerificationResult,
)

__all__ = [
    # Core ABC
    "BaseStrategy",
    # Strategy Implementations
    "GSDStrategy",
    "RALPHStrategy",
    "ChunkResult",
    "RLMStrategy",
    # RLM Components
    "RLMState",
    "RLMMode",
    "REPLEnvironment",
    "REPLVariable",
    "CodeExecutionResult",
    "RLMIteration",
    "RLM_SYSTEM_PROMPT",
    "RLM_BASIC_SYSTEM_PROMPT",
    "execute_rlm",
    "register_rlm_strategy",
    # Result Types
    "StrategyResult",
    "VerificationResult",
    "DetailedVerificationResult",
    "VerificationCheck",
    "CostEstimate",
    # Enums
    "StrategyType",
    "VerificationCheckType",
    # Factory
    "StrategyFactory",
    # Verification Protocol (Boris Step 13)
    "VerificationProtocol",
    "quick_verify",
    "verified_completion",
]
