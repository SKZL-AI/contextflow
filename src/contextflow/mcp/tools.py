"""
MCP Tool Definitions for ContextFlow.

This module defines all MCP tools exposed by the ContextFlow server.
Each tool follows the MCP specification with proper input schemas,
descriptions, and response types.

Tools:
- contextflow_process: Process documents with intelligent strategy selection
- contextflow_analyze: Analyze context complexity and recommend strategy
- contextflow_search: Search in session memory for relevant context
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from contextflow.core.types import StrategyType
from contextflow.utils.errors import ContextFlowError, ValidationError
from contextflow.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Local Enum Definitions (to avoid import cycles)
# =============================================================================


class ComplexityLevel:
    """Context complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# Tool Result Types
# =============================================================================


@dataclass
class ProcessResult:
    """Result from contextflow_process tool."""

    success: bool
    answer: str
    strategy_used: str
    token_usage: dict[str, Any]
    execution_time: float
    verification_passed: bool
    verification_score: float
    sub_agent_count: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "success": self.success,
            "answer": self.answer,
            "strategy_used": self.strategy_used,
            "token_usage": self.token_usage,
            "execution_time": self.execution_time,
            "verification_passed": self.verification_passed,
            "verification_score": self.verification_score,
            "sub_agent_count": self.sub_agent_count,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Result from contextflow_analyze tool."""

    token_count: int
    density: float
    complexity: str
    complexity_score: float
    recommended_strategy: str
    estimated_costs: dict[str, float]
    estimated_time: float
    structure_type: str
    chunk_suggestion: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "token_count": self.token_count,
            "density": self.density,
            "complexity": self.complexity,
            "complexity_score": self.complexity_score,
            "recommended_strategy": self.recommended_strategy,
            "estimated_costs": self.estimated_costs,
            "estimated_time": self.estimated_time,
            "structure_type": self.structure_type,
            "chunk_suggestion": self.chunk_suggestion,
            "warnings": self.warnings,
        }


@dataclass
class SearchResultItem:
    """Single search result item."""

    content: str
    score: float
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


# =============================================================================
# Tool Input Schemas (JSON Schema format for MCP)
# =============================================================================


PROCESS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task or question to process",
            "minLength": 1,
        },
        "documents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of file paths to process as context",
        },
        "context": {
            "type": "string",
            "description": "Direct context string (alternative to documents)",
        },
        "strategy": {
            "type": "string",
            "enum": ["auto", "gsd_direct", "ralph_structured", "rlm_full", "rlm_dense"],
            "default": "auto",
            "description": "Processing strategy: auto (recommended), gsd_direct (<10K tokens), ralph_structured (10K-100K), rlm_full (>100K)",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Additional constraints or requirements for verification",
        },
        "session_id": {
            "type": "string",
            "description": "Session ID for context persistence across calls",
        },
        "verify": {
            "type": "boolean",
            "default": True,
            "description": "Whether to verify the response (recommended)",
        },
    },
    "required": ["task"],
}


ANALYZE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of file paths to analyze",
        },
        "context": {
            "type": "string",
            "description": "Direct context string to analyze",
        },
        "include_chunk_suggestion": {
            "type": "boolean",
            "default": True,
            "description": "Whether to include chunking recommendations",
        },
    },
    "anyOf": [
        {"required": ["documents"]},
        {"required": ["context"]},
    ],
}


SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query for finding relevant context",
            "minLength": 1,
        },
        "max_results": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum number of results to return",
        },
        "session_id": {
            "type": "string",
            "description": "Session ID to search within",
        },
        "threshold": {
            "type": "number",
            "default": 0.0,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum similarity threshold for results",
        },
    },
    "required": ["query"],
}


# =============================================================================
# Tool Definitions (MCP Format)
# =============================================================================


@dataclass
class MCPToolDefinition:
    """MCP Tool definition structure."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


CONTEXTFLOW_TOOLS: list[MCPToolDefinition] = [
    MCPToolDefinition(
        name="contextflow_process",
        description=(
            "Process documents or context with intelligent strategy selection. "
            "Automatically selects the best processing strategy based on context size: "
            "GSD for <10K tokens (fast, single call), "
            "RALPH for 10K-100K tokens (iterative refinement), "
            "RLM for >100K tokens (recursive with sub-agents). "
            "Includes built-in verification for quality assurance."
        ),
        input_schema=PROCESS_INPUT_SCHEMA,
    ),
    MCPToolDefinition(
        name="contextflow_analyze",
        description=(
            "Analyze context complexity without executing any processing. "
            "Returns token count, information density, recommended strategy, "
            "estimated costs across providers, and chunking suggestions. "
            "Use this before processing to understand context characteristics "
            "and optimize your approach."
        ),
        input_schema=ANALYZE_INPUT_SCHEMA,
    ),
    MCPToolDefinition(
        name="contextflow_search",
        description=(
            "Search for relevant context within a session's memory. "
            "Uses semantic similarity to find the most relevant chunks "
            "from previously processed documents. Useful for follow-up "
            "questions or finding specific information in large document sets."
        ),
        input_schema=SEARCH_INPUT_SCHEMA,
    ),
]


# =============================================================================
# Tool Implementation Functions
# =============================================================================


async def contextflow_process(
    task: str,
    documents: list[str] | None = None,
    context: str | None = None,
    strategy: str = "auto",
    constraints: list[str] | None = None,
    session_id: str | None = None,
    verify: bool = True,
    contextflow_instance: Any | None = None,
) -> ProcessResult:
    """
    Process documents or context with intelligent strategy selection.

    This is the main processing tool for ContextFlow. It automatically
    selects the appropriate strategy based on context size and complexity,
    then executes the task with optional verification.

    Args:
        task: The task or question to process
        documents: List of file paths to process as context
        context: Direct context string (alternative to documents)
        strategy: Processing strategy (auto, gsd_direct, ralph_structured, rlm_full)
        constraints: Additional constraints for verification
        session_id: Session ID for context persistence
        verify: Whether to verify the response
        contextflow_instance: ContextFlow instance to use (injected by server)

    Returns:
        ProcessResult with answer, strategy used, and metadata

    Raises:
        ValidationError: If inputs are invalid
        ContextFlowError: If processing fails

    Example:
        result = await contextflow_process(
            task="Summarize the key findings",
            documents=["research.pdf", "data.csv"],
            strategy="auto",
            verify=True
        )
        print(result.answer)
    """
    start_time = time.time()

    logger.info(
        "contextflow_process called",
        task_length=len(task),
        has_documents=documents is not None,
        has_context=context is not None,
        strategy=strategy,
    )

    # Validate inputs
    if not task or not task.strip():
        raise ValidationError("Task cannot be empty", field="task")

    if documents is None and context is None:
        raise ValidationError(
            "Either 'documents' or 'context' must be provided",
            field="documents/context",
        )

    # Validate strategy
    try:
        strategy_type = StrategyType(strategy.lower())
    except ValueError:
        valid_strategies = [s.value for s in StrategyType]
        raise ValidationError(
            f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}",
            field="strategy",
        )

    try:
        # Use provided ContextFlow instance or import and create one
        if contextflow_instance is None:
            from contextflow.core.orchestrator import ContextFlow

            cf = ContextFlow()
            await cf.initialize()
        else:
            cf = contextflow_instance

        # Execute processing
        result = await cf.process(
            task=task,
            documents=documents,
            context=context,
            strategy=strategy_type,
            constraints=constraints,
        )

        execution_time = time.time() - start_time

        return ProcessResult(
            success=True,
            answer=result.answer,
            strategy_used=result.strategy_used.value,
            token_usage={
                "total_tokens": result.total_tokens,
                "cost_usd": result.total_cost,
            },
            execution_time=execution_time,
            verification_passed=result.metadata.get("verification_passed", True),
            verification_score=result.metadata.get("verification_score", 1.0),
            sub_agent_count=result.sub_agent_count,
            warnings=result.warnings,
            metadata={
                "execution_id": result.id,
                "trajectory_steps": len(result.trajectory),
            },
        )

    except ContextFlowError as e:
        logger.error("Process failed", error=str(e), exc_info=True)
        return ProcessResult(
            success=False,
            answer=f"Processing failed: {e.message}",
            strategy_used=strategy,
            token_usage={"total_tokens": 0, "cost_usd": 0.0},
            execution_time=time.time() - start_time,
            verification_passed=False,
            verification_score=0.0,
            warnings=[str(e)],
            metadata={"error": e.to_dict()},
        )

    except Exception as e:
        logger.error("Unexpected error in process", error=str(e), exc_info=True)
        return ProcessResult(
            success=False,
            answer=f"Unexpected error: {str(e)}",
            strategy_used=strategy,
            token_usage={"total_tokens": 0, "cost_usd": 0.0},
            execution_time=time.time() - start_time,
            verification_passed=False,
            verification_score=0.0,
            warnings=[str(e)],
            metadata={"error_type": type(e).__name__},
        )


async def contextflow_analyze(
    documents: list[str] | None = None,
    context: str | None = None,
    include_chunk_suggestion: bool = True,
    contextflow_instance: Any | None = None,
) -> AnalysisResult:
    """
    Analyze context complexity without executing any processing.

    This tool provides detailed analysis of context characteristics
    including token count, information density, recommended strategy,
    and cost estimates. Use this before processing to understand
    your context and plan accordingly.

    Args:
        documents: List of file paths to analyze
        context: Direct context string to analyze
        include_chunk_suggestion: Whether to include chunking recommendations
        contextflow_instance: ContextFlow instance to use (injected by server)

    Returns:
        AnalysisResult with context metrics and recommendations

    Raises:
        ValidationError: If no context provided

    Example:
        analysis = await contextflow_analyze(
            documents=["large_document.pdf"],
            include_chunk_suggestion=True
        )
        print(f"Recommended strategy: {analysis.recommended_strategy}")
        print(f"Estimated cost: ${analysis.estimated_costs['claude']:.4f}")
    """
    logger.info(
        "contextflow_analyze called",
        has_documents=documents is not None,
        has_context=context is not None,
    )

    # Validate inputs
    if documents is None and context is None:
        raise ValidationError(
            "Either 'documents' or 'context' must be provided",
            field="documents/context",
        )

    try:
        # Use provided ContextFlow instance or import and create one
        if contextflow_instance is None:
            from contextflow.core.orchestrator import ContextFlow

            cf = ContextFlow()
            await cf.initialize()
        else:
            cf = contextflow_instance

        # Execute analysis
        analysis = await cf.analyze(
            documents=documents,
            context=context,
        )

        # Determine complexity level from score
        if analysis.complexity_score < 0.3:
            complexity = ComplexityLevel.LOW.value
        elif analysis.complexity_score < 0.6:
            complexity = ComplexityLevel.MEDIUM.value
        elif analysis.complexity_score < 0.85:
            complexity = ComplexityLevel.HIGH.value
        else:
            complexity = ComplexityLevel.VERY_HIGH.value

        # Build chunk suggestion if requested
        chunk_suggestion = None
        if include_chunk_suggestion and analysis.metadata.get("chunk_suggestion"):
            cs = analysis.metadata["chunk_suggestion"]
            chunk_suggestion = {
                "strategy": cs.get("strategy", "semantic"),
                "chunk_size": cs.get("chunk_size", 4000),
                "overlap": cs.get("overlap", 200),
                "estimated_chunks": cs.get("estimated_chunks", 1),
                "rationale": cs.get("rationale", ""),
            }

        # Estimate costs for different providers
        base_cost = analysis.estimated_cost
        estimated_costs = {
            "claude": base_cost,
            "openai": base_cost * 1.2,  # Typically slightly more expensive
            "gemini": base_cost * 0.7,  # Typically less expensive
            "groq": base_cost * 0.3,  # Significantly cheaper
        }

        return AnalysisResult(
            token_count=analysis.token_count,
            density=analysis.density_score,
            complexity=complexity,
            complexity_score=analysis.complexity_score,
            recommended_strategy=analysis.recommended_strategy.value,
            estimated_costs=estimated_costs,
            estimated_time=analysis.estimated_time_seconds,
            structure_type=analysis.structure_type,
            chunk_suggestion=chunk_suggestion,
            warnings=analysis.warnings,
        )

    except ContextFlowError as e:
        logger.error("Analysis failed", error=str(e), exc_info=True)
        raise

    except Exception as e:
        logger.error("Unexpected error in analyze", error=str(e), exc_info=True)
        raise ContextFlowError(
            f"Analysis failed: {str(e)}",
            details={"error_type": type(e).__name__},
        )


async def contextflow_search(
    query: str,
    max_results: int = 10,
    session_id: str | None = None,
    threshold: float = 0.0,
    contextflow_instance: Any | None = None,
) -> list[SearchResultItem]:
    """
    Search for relevant context within session memory.

    This tool performs semantic search across previously processed
    documents to find the most relevant chunks for a given query.
    Useful for follow-up questions or finding specific information
    in large document sets.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        session_id: Session ID to search within (uses current if not provided)
        threshold: Minimum similarity threshold (0.0-1.0)
        contextflow_instance: ContextFlow instance to use (injected by server)

    Returns:
        List of SearchResultItem with matching content and scores

    Raises:
        ValidationError: If query is empty

    Example:
        results = await contextflow_search(
            query="machine learning algorithms",
            max_results=5,
            threshold=0.5
        )
        for result in results:
            print(f"[{result.score:.2f}] {result.content[:100]}...")
    """
    logger.info(
        "contextflow_search called",
        query_length=len(query),
        max_results=max_results,
        session_id=session_id,
        threshold=threshold,
    )

    # Validate inputs
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty", field="query")

    if max_results < 1 or max_results > 100:
        raise ValidationError(
            "max_results must be between 1 and 100",
            field="max_results",
        )

    try:
        # Check if RAG is available in the ContextFlow instance
        if contextflow_instance is not None and hasattr(contextflow_instance, "_rag"):
            rag = contextflow_instance._rag

            if rag is not None:
                # Perform search using RAG
                results = await rag.search(query, k=max_results)

                search_results = []
                for doc in results:
                    if doc.score >= threshold:
                        search_results.append(
                            SearchResultItem(
                                content=doc.content,
                                score=doc.score,
                                chunk_id=doc.id,
                                metadata=doc.metadata,
                            )
                        )

                logger.info(
                    "Search completed",
                    query=query[:50],
                    total_results=len(search_results),
                )

                return search_results

        # If no RAG available, return empty results with warning
        logger.warning("No RAG index available for search")
        return []

    except Exception as e:
        logger.error("Search failed", error=str(e), exc_info=True)
        raise ContextFlowError(
            f"Search failed: {str(e)}",
            details={"query": query[:100], "error_type": type(e).__name__},
        )


# =============================================================================
# Tool Handler Mapping
# =============================================================================


TOOL_HANDLERS: dict[str, Any] = {
    "contextflow_process": contextflow_process,
    "contextflow_analyze": contextflow_analyze,
    "contextflow_search": contextflow_search,
}


def get_tool_handler(tool_name: str) -> Any | None:
    """
    Get the handler function for a tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Handler function or None if not found
    """
    return TOOL_HANDLERS.get(tool_name)


def list_tool_names() -> list[str]:
    """
    Get list of all available tool names.

    Returns:
        List of tool names
    """
    return list(TOOL_HANDLERS.keys())


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Tool Definitions
    "CONTEXTFLOW_TOOLS",
    "MCPToolDefinition",
    # Tool Functions
    "contextflow_process",
    "contextflow_analyze",
    "contextflow_search",
    # Result Types
    "ProcessResult",
    "AnalysisResult",
    "SearchResultItem",
    # Input Schemas
    "PROCESS_INPUT_SCHEMA",
    "ANALYZE_INPUT_SCHEMA",
    "SEARCH_INPUT_SCHEMA",
    # Utilities
    "get_tool_handler",
    "list_tool_names",
    "TOOL_HANDLERS",
]
