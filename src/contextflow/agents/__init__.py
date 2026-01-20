"""
Sub-agent system for ContextFlow RLM strategy.

Provides:
- REPL Environment for safe code execution
- SubAgent for specialized task handling
- Agent pool for parallel task management
- Result aggregation from multiple agents
"""

from contextflow.agents.aggregator import (
    AggregatedResult,
    # Data Classes
    AggregationConfig,
    # Enums
    AggregationStrategy,
    ConflictInfo,
    # Main Class
    ResultAggregator,
    # Convenience Functions
    aggregate_results,
    merge_unique_content,
    select_best_result,
    synthesize_with_llm,
)
from contextflow.agents.pool import (
    # Main Class
    AgentPool,
    # Data Classes
    PoolConfig,
    PoolStats,
    PoolStatus,
    # Enums
    PoolStrategy,
    PoolTask,
    execute_with_roles,
    # Convenience Functions
    parallel_execute,
)
from contextflow.agents.repl import (
    BLOCKED_BUILTINS,
    # Constants
    SAFE_BUILTINS,
    CodeExecutionResult,
    # Enums
    ExecutionMode,
    # Main Class
    REPLEnvironment,
    # Data Classes
    REPLVariable,
    # Convenience Functions
    create_rlm_repl,
    execute_code_safely,
)
from contextflow.agents.sub_agent import (
    # Constants
    ROLE_PROMPTS,
    # Data Classes
    AgentConfig,
    AgentResult,
    # Enums
    AgentRole,
    AgentStatus,
    # Main Class
    SubAgent,
    # Factory Functions
    create_analyzer_agent,
    create_code_reviewer_agent,
    create_extractor_agent,
    create_researcher_agent,
    create_summarizer_agent,
    create_synthesizer_agent,
    create_verifier_agent,
    parallel_agent_tasks,
    # Convenience Functions
    quick_agent_task,
)

__all__ = [
    # REPL Enums
    "ExecutionMode",
    # REPL Data Classes
    "REPLVariable",
    "CodeExecutionResult",
    # REPL Constants
    "SAFE_BUILTINS",
    "BLOCKED_BUILTINS",
    # REPL Main Class
    "REPLEnvironment",
    # REPL Convenience Functions
    "create_rlm_repl",
    "execute_code_safely",
    # SubAgent Enums
    "AgentRole",
    "AgentStatus",
    # SubAgent Data Classes
    "AgentConfig",
    "AgentResult",
    # SubAgent Constants
    "ROLE_PROMPTS",
    # SubAgent Main Class
    "SubAgent",
    # SubAgent Factory Functions
    "create_analyzer_agent",
    "create_summarizer_agent",
    "create_extractor_agent",
    "create_verifier_agent",
    "create_code_reviewer_agent",
    "create_researcher_agent",
    "create_synthesizer_agent",
    # SubAgent Convenience Functions
    "quick_agent_task",
    "parallel_agent_tasks",
    # Pool Enums
    "PoolStrategy",
    "PoolStatus",
    # Pool Data Classes
    "PoolConfig",
    "PoolTask",
    "PoolStats",
    # Pool Main Class
    "AgentPool",
    # Pool Convenience Functions
    "parallel_execute",
    "execute_with_roles",
    # Aggregator Enums
    "AggregationStrategy",
    # Aggregator Data Classes
    "AggregationConfig",
    "AggregatedResult",
    "ConflictInfo",
    # Aggregator Main Class
    "ResultAggregator",
    # Aggregator Convenience Functions
    "aggregate_results",
    "synthesize_with_llm",
    "select_best_result",
    "merge_unique_content",
]
