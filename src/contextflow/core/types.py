"""
Core type definitions for ContextFlow.

This module contains all TypedDicts, Enums, and Dataclasses used throughout the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class StrategyType(str, Enum):
    """Available execution strategies."""

    AUTO = "auto"
    GSD_DIRECT = "gsd_direct"  # < 10K tokens - single LLM call
    RALPH_STRUCTURED = "ralph_structured"  # 10K-100K tokens - iterative
    RLM_FULL = "rlm_full"  # > 100K tokens - recursive with sub-agents
    RLM_DENSE = "rlm_dense"  # High density content - aggressive recursion


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProviderType(str, Enum):
    """LLM provider categories."""

    PROPRIETARY = "proprietary"  # Claude, GPT, Gemini
    OPEN_SOURCE = "open_source"  # Ollama, vLLM
    EXTERNAL_API = "external_api"  # Groq, Together AI
    LOCAL = "local"  # On-device models


class ChunkingStrategy(str, Enum):
    """Text chunking strategies."""

    FIXED = "fixed"  # Fixed token count
    SEMANTIC = "semantic"  # Semantic boundaries (paragraphs, sections)
    SLIDING = "sliding"  # Sliding window with overlap
    SMART = "smart"  # Adaptive based on content type


class AggregationStrategy(str, Enum):
    """Result aggregation strategies."""

    CONSENSUS = "consensus"  # Majority vote
    HIGHEST_CONFIDENCE = "highest_confidence"  # Highest confidence wins
    CHAIN_OF_EVIDENCE = "chain_of_evidence"  # Build evidence chain
    ALL = "all"  # Combine all results


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class Message:
    """Standardized message format for LLM communication."""

    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


# =============================================================================
# Provider Types
# =============================================================================


@dataclass
class CompletionResponse:
    """Standardized completion response from any provider."""

    content: str
    tokens_used: int
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str  # "stop" | "max_tokens" | "error"
    cost_usd: float
    latency_ms: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    raw_response: dict[str, Any] | None = None


@dataclass
class StreamChunk:
    """Single chunk from streaming response."""

    content: str
    is_final: bool = False
    chunk_index: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProviderCapabilities:
    """Provider capabilities and limits."""

    max_context_tokens: int
    max_output_tokens: int
    supports_streaming: bool
    supports_system_prompt: bool
    supports_tools: bool
    supported_models: list[str]
    rate_limit_rpm: int | None = None  # Requests per minute
    rate_limit_tpm: int | None = None  # Tokens per minute
    supports_batch_processing: bool = False
    supports_vision: bool = False
    latency_p50_ms: float = 1000.0  # Median latency
    latency_p99_ms: float = 3000.0  # 99th percentile


# =============================================================================
# Analysis Types
# =============================================================================


@dataclass
class ContextAnalysis:
    """Result of context analysis."""

    token_count: int
    complexity_score: float  # 0.0 - 1.0
    density_score: float  # 0.0 - 1.0
    structure_type: str  # "markdown", "paragraphs", "structured", etc.
    recommended_strategy: StrategyType
    estimated_cost: float  # USD
    estimated_time_seconds: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TrajectoryStep:
    """Single step in execution trajectory."""

    step_type: str  # "analysis", "strategy", "sub_agent", "aggregation"
    timestamp: datetime
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    """Final result of a ContextFlow process."""

    id: str
    answer: str
    strategy_used: StrategyType
    status: TaskStatus
    total_tokens: int
    total_cost: float  # USD
    execution_time: float  # seconds
    sub_agent_count: int = 0
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "answer": self.answer,
            "strategy_used": self.strategy_used.value,
            "status": self.status.value,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "execution_time": self.execution_time,
            "sub_agent_count": self.sub_agent_count,
            "trajectory": [
                {
                    "step_type": s.step_type,
                    "timestamp": s.timestamp.isoformat(),
                    "tokens_used": s.tokens_used,
                    "cost_usd": s.cost_usd,
                    "metadata": s.metadata,
                }
                for s in self.trajectory
            ],
            "warnings": self.warnings,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# RAG Types
# =============================================================================


@dataclass
class RAGDocument:
    """Document in RAG system."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Similarity score when retrieved


@dataclass
class Chunk:
    """Text chunk for processing."""

    id: str
    content: str
    tokens: int
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""

    embedding: list[float]
    dimension: int
    model: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Sub-Agent Types
# =============================================================================


@dataclass
class SubAgentResult:
    """Result from a sub-agent."""

    agent_id: str
    chunk_id: str
    answer: str
    confidence: float  # 0.0 - 1.0
    evidence: list[str] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    needs_further_investigation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated results from multiple sub-agents."""

    final_answer: str
    consensus_score: float  # 0.0 - 1.0
    total_agents: int
    supporting_agents: int
    evidence_chain: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
