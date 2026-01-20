"""
ContextFlow - Intelligent LLM Context Orchestration

A framework for automatic strategy selection (GSD/RALPH/RLM) based on context size,
with in-memory RAG support and multi-provider LLM integration.

Basic Usage:
    from contextflow import ContextFlow

    cf = ContextFlow(provider="claude")
    result = await cf.process(
        task="Summarize this document",
        documents=["large_file.txt"],
        strategy="auto"
    )
    print(result.answer)

For more information, see: https://contextflow.ai/docs
"""

from contextflow.core.config import ContextFlowConfig
from contextflow.core.orchestrator import ContextFlow
from contextflow.core.types import (
    ContextAnalysis,
    ProcessResult,
    ProviderType,
    StrategyType,
    TaskStatus,
)

__version__ = "0.1.0"
__author__ = "SAI"
__all__ = [
    # Main class
    "ContextFlow",
    # Config
    "ContextFlowConfig",
    # Types
    "StrategyType",
    "TaskStatus",
    "ProviderType",
    "ProcessResult",
    "ContextAnalysis",
    # Version
    "__version__",
]
