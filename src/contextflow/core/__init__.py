"""Core orchestration module.

This module provides the main entry points and core components for ContextFlow:
- ContextFlow: Main orchestrator class
- ContextAnalyzer: Context analysis and strategy recommendation
- HooksManager: Lifecycle hooks for processing pipeline
- StrategyRouter: Automatic strategy selection
- SessionManager: Session and observation tracking
"""

from contextflow.core.analyzer import (
    AnalyzerConfig,
    ChunkSuggestion,
    ContentType,
    ContextAnalyzer,
    CostEstimate,
    DensityLevel,
    analyze_context,
    analyze_context_async,
    estimate_analysis_cost,
)
from contextflow.core.analyzer import (
    ContextAnalysis as AnalyzerContextAnalysis,
)
from contextflow.core.config import ContextFlowConfig
from contextflow.core.hooks import (
    HookContext,
    HookExecutionResult,
    HooksManager,
    HookType,
    RegisteredHook,
    compose_hooks,
    create_logging_hook,
    create_timing_hook,
    get_global_hooks_manager,
    on_error,
    on_verification_fail,
    post_process,
    post_strategy,
    pre_process,
    pre_strategy,
    reset_global_hooks_manager,
)
from contextflow.core.orchestrator import (
    ContextFlow,
    ExecutionContext,
    OrchestratorConfig,
    create_contextflow,
    quick_analyze,
    quick_process,
)
from contextflow.core.router import (
    ComplexityLevel,
    RouterConfig,
    StrategyRouter,
    auto_route,
    get_recommended_strategy,
)
from contextflow.core.router import (
    ContextAnalysis as RouterContextAnalysis,
)
from contextflow.core.router import (
    analyze_context as router_analyze_context,
)
from contextflow.core.session import (
    Observation,
    ObservationType,
    Session,
    SessionContext,
    SessionManager,
    get_default_session_manager,
    quick_session,
)
from contextflow.core.types import (
    CompletionResponse,
    ContextAnalysis,
    Message,
    ProcessResult,
    ProviderCapabilities,
    ProviderType,
    StrategyType,
    StreamChunk,
    TaskStatus,
)

__all__ = [
    # Types
    "StrategyType",
    "TaskStatus",
    "ProviderType",
    "ProcessResult",
    "ContextAnalysis",
    "Message",
    "CompletionResponse",
    "StreamChunk",
    "ProviderCapabilities",
    # Config
    "ContextFlowConfig",
    # Orchestrator (Main Entry Point)
    "ContextFlow",
    "OrchestratorConfig",
    "ExecutionContext",
    "quick_process",
    "quick_analyze",
    "create_contextflow",
    # Analyzer
    "ContextAnalyzer",
    "AnalyzerConfig",
    "AnalyzerContextAnalysis",
    "ChunkSuggestion",
    "CostEstimate",
    "DensityLevel",
    "ContentType",
    "analyze_context",
    "analyze_context_async",
    "estimate_analysis_cost",
    # Hooks
    "HookType",
    "HookContext",
    "HooksManager",
    "RegisteredHook",
    "HookExecutionResult",
    "pre_process",
    "post_process",
    "pre_strategy",
    "post_strategy",
    "on_error",
    "on_verification_fail",
    "get_global_hooks_manager",
    "reset_global_hooks_manager",
    "create_logging_hook",
    "create_timing_hook",
    "compose_hooks",
    # Router
    "StrategyRouter",
    "RouterConfig",
    "ComplexityLevel",
    "RouterContextAnalysis",
    "auto_route",
    "router_analyze_context",
    "get_recommended_strategy",
    # Session
    "SessionManager",
    "Session",
    "Observation",
    "ObservationType",
    "SessionContext",
    "get_default_session_manager",
    "quick_session",
]
