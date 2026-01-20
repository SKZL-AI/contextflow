"""
Main ContextFlow Orchestrator - Full Integration.

This is the primary entry point for the ContextFlow framework, orchestrating
all components for intelligent large-context processing.

Integrates:
- ContextAnalyzer: Analyze context before processing
- StrategyRouter: Select and execute strategies
- HooksManager: Lifecycle hooks execution
- VerificationProtocol: Verify outputs (Boris Step 13)
- SessionManager: Track sessions and observations
- TemporaryRAG: Document indexing for large contexts
- AgentPool: Parallel task execution

Based on Boris' Best Practices:
- Step 8: Sub-agents for parallel processing
- Step 9: Lifecycle hooks
- Step 13: Verification loop for 2-3x quality improvement
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from contextflow.core.analyzer import (
    AnalyzerConfig,
    ContextAnalyzer,
)
from contextflow.core.config import ContextFlowConfig, get_config
from contextflow.core.hooks import (
    HookContext,
    HooksManager,
    HookType,
    get_global_hooks_manager,
)
from contextflow.core.router import RouterConfig, StrategyRouter
from contextflow.core.session import (
    ObservationType,
    Session,
    SessionManager,
    get_default_session_manager,
)
from contextflow.core.types import (
    ContextAnalysis as CoreContextAnalysis,
)
from contextflow.core.types import (
    ProcessResult,
    StrategyType,
    TaskStatus,
    TrajectoryStep,
)
from contextflow.providers.base import BaseProvider
from contextflow.providers.factory import get_provider
from contextflow.strategies.base import StrategyResult
from contextflow.strategies.verification import (
    VerificationProtocol,
)
from contextflow.utils.errors import (
    ContextFlowError,
    StrategyExecutionError,
    ValidationError,
)
from contextflow.utils.logging import ProviderLogger, get_logger

if TYPE_CHECKING:
    from contextflow.agents.pool import AgentPool
    from contextflow.rag.embeddings.base import BaseEmbeddingProvider
    from contextflow.rag.temp_rag import TemporaryRAG


logger = get_logger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

ProcessCallback = Callable[[str, float], Coroutine[Any, Any, None]]
StreamCallback = Callable[[str], Coroutine[Any, Any, None]]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OrchestratorConfig:
    """
    Configuration for the ContextFlow orchestrator.

    Attributes:
        enable_verification: Enable output verification (Boris Step 13)
        verification_threshold: Minimum score to pass verification (0.0-1.0)
        max_verification_iterations: Max iterations for verification loop
        enable_sessions: Enable session tracking
        session_db_path: Path to session database
        enable_hooks: Enable lifecycle hooks
        enable_cost_tracking: Track and report costs
        enable_streaming: Enable streaming output
        default_timeout: Default timeout for operations (seconds)
        auto_select_strategy: Automatically select strategy when AUTO
        parallel_analysis: Run analysis in parallel where possible
    """

    enable_verification: bool = True
    verification_threshold: float = 0.7
    max_verification_iterations: int = 3
    enable_sessions: bool = True
    session_db_path: str = "~/.contextflow/sessions.db"
    enable_hooks: bool = True
    enable_cost_tracking: bool = True
    enable_streaming: bool = True
    default_timeout: float = 120.0
    auto_select_strategy: bool = True
    parallel_analysis: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_verification": self.enable_verification,
            "verification_threshold": self.verification_threshold,
            "max_verification_iterations": self.max_verification_iterations,
            "enable_sessions": self.enable_sessions,
            "session_db_path": self.session_db_path,
            "enable_hooks": self.enable_hooks,
            "enable_cost_tracking": self.enable_cost_tracking,
            "enable_streaming": self.enable_streaming,
            "default_timeout": self.default_timeout,
            "auto_select_strategy": self.auto_select_strategy,
            "parallel_analysis": self.parallel_analysis,
        }


@dataclass
class ExecutionContext:
    """
    Internal context for tracking execution state.

    Used to pass state through the processing pipeline.
    """

    execution_id: str
    task: str
    context: str
    documents: list[str] | None
    strategy: StrategyType
    start_time: float
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    session_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def add_trajectory_step(
        self,
        step_type: str,
        tokens: int = 0,
        cost: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a step to the execution trajectory."""
        self.trajectory.append(
            TrajectoryStep(
                step_type=step_type,
                timestamp=datetime.utcnow(),
                tokens_used=tokens,
                cost_usd=cost,
                metadata=metadata or {},
            )
        )
        self.total_tokens += tokens
        self.total_cost += cost


# =============================================================================
# Main ContextFlow Orchestrator
# =============================================================================


class ContextFlow:
    """
    Main ContextFlow Orchestrator.

    Provides automatic strategy selection and execution for processing
    large contexts with LLMs. Integrates all framework components for
    a seamless processing pipeline.

    Pipeline:
    1. Pre-process hooks
    2. Context analysis (for AUTO strategy)
    3. Pre-strategy hooks
    4. Strategy execution
    5. Post-strategy hooks
    6. Output verification (Boris Step 13)
    7. On-verification-fail hooks (if needed)
    8. Post-process hooks
    9. Session observation recording

    Example:
        # Basic usage
        cf = ContextFlow(provider="claude")
        result = await cf.process(
            task="Summarize this document",
            documents=["large_file.txt"],
            strategy="auto"
        )
        print(result.answer)

        # With RAG indexing
        result = await cf.process_with_rag(
            task="Find all API endpoints",
            documents=["api_docs.md", "code/*.py"],
            k=10  # Top 10 relevant chunks
        )

        # Streaming
        async for chunk in cf.stream(
            task="Explain this codebase",
            context=code_content
        ):
            print(chunk, end="", flush=True)

        # Context manager
        async with ContextFlow() as cf:
            result = await cf.process(task="Analyze", context=data)

    Attributes:
        provider: LLM provider instance
        config: ContextFlow configuration
        orchestrator_config: Orchestrator-specific configuration
    """

    def __init__(
        self,
        provider: str | BaseProvider | None = None,
        config: ContextFlowConfig | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        hooks_manager: HooksManager | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ContextFlow orchestrator.

        Args:
            provider: Provider name (e.g., "claude") or BaseProvider instance
            config: ContextFlow configuration (loads from env if None)
            orchestrator_config: Orchestrator-specific configuration
            hooks_manager: Custom hooks manager (uses global if None)
            session_manager: Custom session manager (uses default if None)
            **kwargs: Additional provider initialization arguments
        """
        # Configuration
        self.config = config or get_config()
        self._orchestrator_config = orchestrator_config or OrchestratorConfig()
        self._logger = ProviderLogger("orchestrator")

        # Provider initialization
        if isinstance(provider, BaseProvider):
            self._provider = provider
        else:
            provider_name = provider or self.config.default_provider
            self._provider = get_provider(
                name=provider_name,
                config=self.config,
                **kwargs,
            )

        # Component initialization
        self._analyzer: ContextAnalyzer | None = None
        self._router: StrategyRouter | None = None
        self._verifier: VerificationProtocol | None = None
        self._hooks_manager = hooks_manager or get_global_hooks_manager()
        self._session_manager = session_manager
        self._agent_pool: AgentPool | None = None
        self._rag: TemporaryRAG | None = None

        # State
        self._initialized = False
        self._current_session: Session | None = None
        self._execution_count = 0
        self._total_tokens_used = 0
        self._total_cost = 0.0

        logger.info(
            "ContextFlow initialized",
            provider=self._provider.name,
            orchestrator_config=self._orchestrator_config.to_dict(),
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def provider(self) -> BaseProvider:
        """Get the current LLM provider."""
        return self._provider

    @property
    def orchestrator_config(self) -> OrchestratorConfig:
        """Get orchestrator configuration."""
        return self._orchestrator_config

    @property
    def hooks_manager(self) -> HooksManager:
        """Get the hooks manager."""
        return self._hooks_manager

    @property
    def is_initialized(self) -> bool:
        """Check if all components are initialized."""
        return self._initialized

    @property
    def stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "provider": self._provider.name,
            "execution_count": self._execution_count,
            "total_tokens_used": self._total_tokens_used,
            "total_cost_usd": round(self._total_cost, 6),
            "has_session": self._current_session is not None,
            "has_rag": self._rag is not None,
            "has_agent_pool": self._agent_pool is not None,
        }

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """
        Initialize all components lazily.

        Called automatically on first use if not called explicitly.
        """
        if self._initialized:
            return

        logger.debug("Initializing ContextFlow components")

        # Initialize analyzer
        self._analyzer = ContextAnalyzer(
            provider=self._provider,
            config=AnalyzerConfig(
                gsd_max_tokens=self.config.strategy.gsd_max_tokens,
                ralph_max_tokens=self.config.strategy.ralph_max_tokens,
                rlm_min_tokens=self.config.strategy.rlm_min_tokens,
            ),
        )

        # Initialize router
        self._router = StrategyRouter(
            provider=self._provider,
            config=RouterConfig(
                gsd_max_tokens=self.config.strategy.gsd_max_tokens,
                ralph_max_tokens=self.config.strategy.ralph_max_tokens,
                rlm_min_tokens=self.config.strategy.rlm_min_tokens,
            ),
        )

        # Initialize verifier
        if self._orchestrator_config.enable_verification:
            self._verifier = VerificationProtocol(
                provider=self._provider,
                min_confidence=self._orchestrator_config.verification_threshold,
            )

        # Initialize session manager
        if self._orchestrator_config.enable_sessions:
            if self._session_manager is None:
                self._session_manager = get_default_session_manager()

        self._initialized = True
        logger.info("ContextFlow components initialized")

    async def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if not self._initialized:
            await self.initialize()

    # =========================================================================
    # Main Processing Entry Points
    # =========================================================================

    async def process(
        self,
        task: str,
        documents: list[str] | None = None,
        context: str | None = None,
        strategy: StrategyType | str = StrategyType.AUTO,
        constraints: list[str] | None = None,
        timeout: float | None = None,
        progress_callback: ProcessCallback | None = None,
        **kwargs: Any,
    ) -> ProcessResult:
        """
        Main processing entry point with full pipeline.

        Executes the complete processing pipeline:
        1. Pre-process hooks
        2. Context analysis (if AUTO strategy)
        3. Pre-strategy hooks
        4. Strategy execution
        5. Post-strategy hooks
        6. Output verification
        7. On-verification-fail hooks (if needed)
        8. Post-process hooks
        9. Session observation recording

        Args:
            task: Task description / query to process
            documents: List of document paths to process
            context: Direct context string (alternative to documents)
            strategy: Strategy to use (auto, gsd_direct, ralph_structured, rlm_full)
            constraints: Optional verification constraints
            timeout: Operation timeout in seconds
            progress_callback: Optional async callback for progress updates
            **kwargs: Additional strategy-specific arguments

        Returns:
            ProcessResult with answer, metadata, and trajectory

        Raises:
            ValidationError: If inputs are invalid
            StrategyExecutionError: If strategy execution fails
            ContextFlowError: If processing fails

        Example:
            result = await cf.process(
                task="Summarize the key points",
                documents=["report.pdf"],
                strategy="auto",
                constraints=["Include all statistics", "Keep under 500 words"]
            )
        """
        await self._ensure_initialized()

        # Validate inputs
        if not task or not task.strip():
            raise ValidationError("Task cannot be empty", field="task")

        if documents is None and context is None:
            raise ValidationError(
                "Either documents or context must be provided",
                field="documents/context",
            )

        # Parse strategy
        if isinstance(strategy, str):
            try:
                strategy = StrategyType(strategy.lower())
            except ValueError:
                raise ValidationError(
                    f"Unknown strategy: {strategy}. Valid: {[s.value for s in StrategyType]}",
                    field="strategy",
                )

        # Create execution context
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        timeout = timeout or self._orchestrator_config.default_timeout

        # Load documents if provided
        if documents:
            context = await self._load_documents(documents)
        elif context is None:
            context = ""

        exec_ctx = ExecutionContext(
            execution_id=execution_id,
            task=task,
            context=context,
            documents=documents,
            strategy=strategy,
            start_time=time.time(),
        )

        logger.info(
            "Starting process",
            execution_id=execution_id,
            task_length=len(task),
            context_length=len(context),
            strategy=strategy.value,
        )

        try:
            # Start session if enabled
            if self._orchestrator_config.enable_sessions and self._session_manager:
                session = await self._session_manager.start_session(
                    metadata={"execution_id": execution_id, "task": task[:200]}
                )
                exec_ctx.session_id = session.id
                self._current_session = session

            # Execute pipeline
            result = await self._execute_pipeline(
                exec_ctx=exec_ctx,
                constraints=constraints,
                timeout=timeout,
                progress_callback=progress_callback,
                **kwargs,
            )

            # Record success observation
            if exec_ctx.session_id and self._session_manager:
                await self._session_manager.add_observation(
                    session_id=exec_ctx.session_id,
                    obs_type=ObservationType.RESULT,
                    content=f"Successfully processed: {task[:100]}",
                    metadata={
                        "strategy": result.strategy_used.value,
                        "tokens": result.total_tokens,
                        "cost": result.total_cost,
                    },
                )

            # Update statistics
            self._execution_count += 1
            self._total_tokens_used += result.total_tokens
            self._total_cost += result.total_cost

            return result

        except Exception as e:
            # Record error observation
            if exec_ctx.session_id and self._session_manager:
                await self._session_manager.add_observation(
                    session_id=exec_ctx.session_id,
                    obs_type=ObservationType.ERROR,
                    content=f"Error: {str(e)}",
                    metadata={"error_type": type(e).__name__},
                )

            logger.error(
                "Process failed",
                execution_id=execution_id,
                error=str(e),
                exc_info=True,
            )
            raise

        finally:
            # End session if started in this call
            if exec_ctx.session_id and self._session_manager and self._current_session:
                await self._session_manager.end_session(exec_ctx.session_id)
                self._current_session = None

    async def stream(
        self,
        task: str,
        documents: list[str] | None = None,
        context: str | None = None,
        strategy: StrategyType | str = StrategyType.AUTO,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming processing with hooks.

        Executes the processing pipeline with streaming output.
        Note: Verification is performed on the complete output after streaming.

        Args:
            task: Task description
            documents: List of document paths
            context: Direct context string
            strategy: Strategy to use
            constraints: Verification constraints
            **kwargs: Additional arguments

        Yields:
            String chunks as they are generated

        Example:
            async for chunk in cf.stream(
                task="Explain this code",
                context=code_content
            ):
                print(chunk, end="", flush=True)
        """
        await self._ensure_initialized()

        # Validate inputs
        if not task or not task.strip():
            raise ValidationError("Task cannot be empty", field="task")

        # Parse strategy
        if isinstance(strategy, str):
            strategy = StrategyType(strategy.lower())

        # Load documents if provided
        if documents:
            context = await self._load_documents(documents)
        elif context is None:
            context = ""

        # Create execution context
        execution_id = f"stream-{uuid.uuid4().hex[:12]}"
        exec_ctx = ExecutionContext(
            execution_id=execution_id,
            task=task,
            context=context,
            documents=documents,
            strategy=strategy,
            start_time=time.time(),
        )

        logger.info(
            "Starting stream",
            execution_id=execution_id,
            strategy=strategy.value,
        )

        # Execute pre-process hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.PRE_PROCESS,
                task=task,
                context=context,
                strategy=strategy.value,
                execution_id=execution_id,
            )
            await self._hooks_manager.execute(HookType.PRE_PROCESS, hook_ctx)

        # Analyze and select strategy if AUTO
        if strategy == StrategyType.AUTO and self._router:
            analysis = self._router.analyze(task, context, constraints)
            strategy = analysis.recommended_strategy
            exec_ctx.strategy = strategy
            exec_ctx.add_trajectory_step("analysis", metadata={"strategy": strategy.value})

        # Get strategy from router
        if self._router:
            strategy_impl = self._router._create_strategy(strategy)

            # Check if strategy supports streaming
            if strategy_impl.supports_streaming():
                # Stream from strategy
                full_output = ""
                async for chunk in strategy_impl.stream(task, context, **kwargs):
                    full_output += chunk
                    yield chunk

                # Verify after streaming completes
                if self._orchestrator_config.enable_verification and self._verifier:
                    verification = await self._verifier.verify(
                        task=task,
                        output=full_output,
                        constraints=constraints,
                        context=context,
                    )
                    if not verification.passed:
                        logger.warning(
                            "Stream verification failed",
                            confidence=verification.confidence,
                            issues=verification.issues,
                        )
            else:
                # Fallback to non-streaming with simulated streaming
                result = await self._router.route(
                    task=task,
                    context=context,
                    constraints=constraints,
                    force_strategy=strategy,
                    **kwargs,
                )

                # Yield answer in chunks
                chunk_size = 50
                for i in range(0, len(result.answer), chunk_size):
                    yield result.answer[i : i + chunk_size]
                    await asyncio.sleep(0.01)  # Small delay for streaming effect

        # Execute post-process hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.POST_PROCESS,
                task=task,
                context=context,
                strategy=strategy.value,
                execution_id=execution_id,
            )
            await self._hooks_manager.execute(HookType.POST_PROCESS, hook_ctx)

    async def analyze(
        self,
        documents: list[str] | None = None,
        context: str | None = None,
        task: str | None = None,
        use_llm: bool = False,
    ) -> CoreContextAnalysis:
        """
        Analyze context without executing.

        Provides analysis of the context including token count,
        density, complexity, and strategy recommendation.

        Args:
            documents: List of document paths
            context: Direct context string
            task: Optional task for complexity assessment
            use_llm: Use LLM-assisted analysis (more accurate)

        Returns:
            ContextAnalysis with recommendations

        Raises:
            ValidationError: If no context provided

        Example:
            analysis = await cf.analyze(documents=["large_file.txt"])
            print(f"Recommended: {analysis.recommended_strategy}")
            print(f"Estimated cost: ${analysis.estimated_cost}")
        """
        await self._ensure_initialized()

        # Load documents if provided
        if documents:
            context = await self._load_documents(documents)
        elif context is None:
            raise ValidationError(
                "Either documents or context must be provided",
                field="documents/context",
            )

        task = task or "Analyze the provided context"

        if self._analyzer:
            if use_llm:
                analysis = await self._analyzer.analyze_async(
                    task=task,
                    context=context,
                    use_llm=True,
                )
            else:
                analysis = self._analyzer.analyze(task=task, context=context)

            # Convert to CoreContextAnalysis format
            return CoreContextAnalysis(
                token_count=analysis.token_count,
                complexity_score=analysis.density,
                density_score=analysis.density,
                structure_type=analysis.content_type.value,
                recommended_strategy=analysis.recommended_strategy,
                estimated_cost=sum(
                    c.total_cost for c in analysis.estimated_costs.values()
                )
                / max(len(analysis.estimated_costs), 1),
                estimated_time_seconds=analysis.token_count / 1000,  # Rough estimate
                warnings=analysis.warnings,
                metadata={
                    "reasoning": analysis.reasoning,
                    "alternatives": [s.value for s in analysis.alternative_strategies],
                    "chunk_suggestion": (
                        analysis.chunk_suggestion.model_dump()
                        if analysis.chunk_suggestion
                        else None
                    ),
                },
            )

        # Fallback if analyzer not available
        token_count = len(context) // 4
        return CoreContextAnalysis(
            token_count=token_count,
            complexity_score=0.5,
            density_score=0.5,
            structure_type="unknown",
            recommended_strategy=StrategyType.AUTO,
            estimated_cost=0.0,
            estimated_time_seconds=token_count / 1000,
            warnings=["Analyzer not available"],
        )

    async def process_with_rag(
        self,
        task: str,
        documents: list[str] | None = None,
        context: str | None = None,
        strategy: StrategyType | str = StrategyType.AUTO,
        k: int = 10,
        embedding_provider: BaseEmbeddingProvider | None = None,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> ProcessResult:
        """
        Process with document RAG indexing.

        Indexes documents using FAISS-based RAG and retrieves
        relevant chunks for processing. Useful for very large
        document collections where full context doesn't fit.

        Args:
            task: Task description
            documents: Document paths to index
            context: Direct context to index
            strategy: Strategy to use
            k: Number of relevant chunks to retrieve
            embedding_provider: Embedding provider for RAG
            constraints: Verification constraints
            **kwargs: Additional arguments

        Returns:
            ProcessResult with RAG-enhanced processing

        Example:
            result = await cf.process_with_rag(
                task="Find all security vulnerabilities",
                documents=["src/**/*.py"],
                k=20
            )
        """
        await self._ensure_initialized()

        # Validate inputs
        if not task:
            raise ValidationError("Task cannot be empty", field="task")

        # Load documents
        if documents:
            full_context = await self._load_documents(documents)
        elif context:
            full_context = context
        else:
            raise ValidationError(
                "Either documents or context must be provided",
                field="documents/context",
            )

        # Initialize RAG if not already done
        if self._rag is None:
            await self._initialize_rag(embedding_provider)

        # Index documents in RAG
        if self._rag:
            logger.info("Indexing documents in RAG", context_length=len(full_context))

            # Add document to RAG (auto-chunks if needed)
            await self._rag.add_document(
                content=full_context,
                metadata={"task": task[:200], "source": "process_with_rag"},
            )

            # Search for relevant chunks
            relevant_ids = await self._rag.search_compact(task, k=k)
            relevant_docs = await self._rag.get_full_documents(relevant_ids)

            # Combine relevant chunks as context
            rag_context = "\n\n---\n\n".join([doc.content for doc in relevant_docs])

            logger.info(
                "RAG retrieval complete",
                chunks_retrieved=len(relevant_docs),
                context_length=len(rag_context),
            )

            # Process with retrieved context
            return await self.process(
                task=task,
                context=rag_context,
                strategy=strategy,
                constraints=constraints,
                **kwargs,
            )

        # Fallback to regular processing
        return await self.process(
            task=task,
            context=full_context,
            strategy=strategy,
            constraints=constraints,
            **kwargs,
        )

    async def process_parallel(
        self,
        task: str,
        documents: list[str],
        strategy: StrategyType | str = StrategyType.AUTO,
        max_concurrent: int = 5,
        aggregation: str = "chain_of_evidence",
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> ProcessResult:
        """
        Process multiple documents in parallel using agent pool.

        Uses the AgentPool for parallel task execution across
        multiple documents, then aggregates results.

        Args:
            task: Task description to apply to each document
            documents: List of document paths
            strategy: Strategy to use
            max_concurrent: Maximum concurrent agents
            aggregation: Aggregation strategy (consensus, chain_of_evidence, all)
            constraints: Verification constraints
            **kwargs: Additional arguments

        Returns:
            ProcessResult with aggregated results
        """
        await self._ensure_initialized()

        # Initialize agent pool if needed
        if self._agent_pool is None:
            await self._initialize_agent_pool(max_concurrent)

        # Load each document
        document_contexts = []
        for doc_path in documents:
            doc_context = await self._load_single_document(doc_path)
            document_contexts.append(doc_context)

        # Create execution context
        execution_id = f"parallel-{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        logger.info(
            "Starting parallel processing",
            execution_id=execution_id,
            document_count=len(documents),
            max_concurrent=max_concurrent,
        )

        # Execute in parallel using agent pool
        if self._agent_pool:
            from contextflow.agents.sub_agent import AgentRole

            results = await self._agent_pool.map(
                tasks=[task] * len(document_contexts),
                contexts=document_contexts,
                role=AgentRole.ANALYZER,
                constraints=constraints,
            )

            # Aggregate results
            successful_results = [r for r in results if r.success]
            combined_answer = self._aggregate_results(
                successful_results, aggregation
            )

            total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in results)

            return ProcessResult(
                id=execution_id,
                answer=combined_answer,
                strategy_used=StrategyType.RLM_FULL,  # Parallel implies complex
                status=TaskStatus.COMPLETED,
                total_tokens=total_tokens,
                total_cost=0.0,  # Cost tracking TODO
                execution_time=time.time() - start_time,
                sub_agent_count=len(results),
                warnings=[r.error for r in results if not r.success and r.error],
                metadata={
                    "aggregation": aggregation,
                    "successful_agents": len(successful_results),
                    "total_agents": len(results),
                },
            )

        # Fallback to sequential processing
        logger.warning("Agent pool not available, falling back to sequential")
        combined_context = "\n\n---\n\n".join(document_contexts)
        return await self.process(
            task=task,
            context=combined_context,
            strategy=strategy,
            constraints=constraints,
            **kwargs,
        )

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    async def _execute_pipeline(
        self,
        exec_ctx: ExecutionContext,
        constraints: list[str] | None,
        timeout: float,
        progress_callback: ProcessCallback | None,
        **kwargs: Any,
    ) -> ProcessResult:
        """
        Execute the full processing pipeline.

        Args:
            exec_ctx: Execution context
            constraints: Verification constraints
            timeout: Operation timeout
            progress_callback: Progress callback
            **kwargs: Additional arguments

        Returns:
            ProcessResult from pipeline execution
        """
        # Step 1: Pre-process hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.PRE_PROCESS,
                task=exec_ctx.task,
                context=exec_ctx.context,
                execution_id=exec_ctx.execution_id,
            )
            hook_result = await self._hooks_manager.execute(
                HookType.PRE_PROCESS, hook_ctx
            )
            exec_ctx.add_trajectory_step(
                "pre_process_hooks",
                metadata={"hooks_executed": hook_result.hooks_executed},
            )

            # Update context if modified by hooks
            exec_ctx.task = hook_result.final_context.task or exec_ctx.task
            exec_ctx.context = hook_result.final_context.context or exec_ctx.context

        if progress_callback:
            await progress_callback("Pre-processing complete", 0.1)

        # Step 2: Analyze context (if AUTO strategy)
        analysis = None
        if (
            exec_ctx.strategy == StrategyType.AUTO
            and self._orchestrator_config.auto_select_strategy
            and self._router
        ):
            analysis = self._router.analyze(
                task=exec_ctx.task,
                context=exec_ctx.context,
                constraints=constraints,
            )
            exec_ctx.strategy = analysis.recommended_strategy
            exec_ctx.add_trajectory_step(
                "analysis",
                metadata={
                    "recommended_strategy": analysis.recommended_strategy.value,
                    "token_count": analysis.token_count,
                    "density": analysis.estimated_density,
                    "complexity": analysis.complexity.value,
                },
            )

            logger.info(
                "Strategy selected",
                execution_id=exec_ctx.execution_id,
                strategy=exec_ctx.strategy.value,
                reasoning=analysis.reasoning[:100],
            )

        if progress_callback:
            await progress_callback("Analysis complete", 0.2)

        # Step 3: Pre-strategy hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.PRE_STRATEGY,
                task=exec_ctx.task,
                context=exec_ctx.context,
                strategy=exec_ctx.strategy.value,
                execution_id=exec_ctx.execution_id,
            )
            await self._hooks_manager.execute(HookType.PRE_STRATEGY, hook_ctx)
            exec_ctx.add_trajectory_step("pre_strategy_hooks")

        if progress_callback:
            await progress_callback("Executing strategy", 0.3)

        # Step 4: Execute strategy
        strategy_result = await self._execute_strategy(
            exec_ctx=exec_ctx,
            constraints=constraints,
            timeout=timeout,
            **kwargs,
        )

        exec_ctx.add_trajectory_step(
            "strategy_execution",
            tokens=strategy_result.total_tokens,
            cost=strategy_result.total_cost,
            metadata={
                "strategy": exec_ctx.strategy.value,
                "iterations": strategy_result.iterations,
            },
        )

        if progress_callback:
            await progress_callback("Strategy execution complete", 0.6)

        # Step 5: Post-strategy hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.POST_STRATEGY,
                task=exec_ctx.task,
                context=exec_ctx.context,
                strategy=exec_ctx.strategy.value,
                result=strategy_result.answer,
                execution_id=exec_ctx.execution_id,
            )
            hook_result = await self._hooks_manager.execute(
                HookType.POST_STRATEGY, hook_ctx
            )
            exec_ctx.add_trajectory_step("post_strategy_hooks")

            # Allow hooks to modify result
            if hook_result.final_context.result:
                strategy_result.answer = str(hook_result.final_context.result)

        if progress_callback:
            await progress_callback("Post-processing", 0.7)

        # Step 6: Verification
        verification_passed = True
        verification_result = None

        if self._orchestrator_config.enable_verification and self._verifier:
            output, verification_result = await self._verifier.iterate_until_verified(
                task=exec_ctx.task,
                initial_output=strategy_result.answer,
                constraints=constraints,
                context=exec_ctx.context,
                max_iterations=self._orchestrator_config.max_verification_iterations,
            )

            strategy_result.answer = output
            strategy_result.verification_passed = verification_result.passed
            strategy_result.verification_score = verification_result.confidence
            verification_passed = verification_result.passed

            exec_ctx.add_trajectory_step(
                "verification",
                metadata={
                    "passed": verification_result.passed,
                    "confidence": verification_result.confidence,
                    "iterations": verification_result.iteration,
                },
            )

        if progress_callback:
            await progress_callback("Verification complete", 0.85)

        # Step 7: On-verification-fail hooks
        if not verification_passed and self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.ON_VERIFICATION_FAIL,
                task=exec_ctx.task,
                context=exec_ctx.context,
                result=strategy_result.answer,
                verification_result=(
                    verification_result.to_dict() if verification_result else None
                ),
                execution_id=exec_ctx.execution_id,
            )
            await self._hooks_manager.execute(HookType.ON_VERIFICATION_FAIL, hook_ctx)
            exec_ctx.add_trajectory_step("verification_fail_hooks")
            exec_ctx.warnings.append("Verification did not pass minimum threshold")

        # Step 8: Post-process hooks
        if self._orchestrator_config.enable_hooks:
            hook_ctx = HookContext(
                hook_type=HookType.POST_PROCESS,
                task=exec_ctx.task,
                context=exec_ctx.context,
                strategy=exec_ctx.strategy.value,
                result=strategy_result.answer,
                execution_id=exec_ctx.execution_id,
            )
            await self._hooks_manager.execute(HookType.POST_PROCESS, hook_ctx)
            exec_ctx.add_trajectory_step("post_process_hooks")

        if progress_callback:
            await progress_callback("Complete", 1.0)

        # Build final result
        execution_time = time.time() - exec_ctx.start_time

        return ProcessResult(
            id=exec_ctx.execution_id,
            answer=strategy_result.answer,
            strategy_used=exec_ctx.strategy,
            status=TaskStatus.COMPLETED,
            total_tokens=exec_ctx.total_tokens + strategy_result.total_tokens,
            total_cost=exec_ctx.total_cost + strategy_result.total_cost,
            execution_time=execution_time,
            sub_agent_count=len(strategy_result.sub_results),
            trajectory=exec_ctx.trajectory,
            warnings=exec_ctx.warnings,
            metadata={
                "verification_passed": verification_passed,
                "verification_score": (
                    verification_result.confidence if verification_result else 1.0
                ),
                "strategy_iterations": strategy_result.iterations,
                "analysis": analysis.to_dict() if analysis else None,
            },
        )

    async def _execute_strategy(
        self,
        exec_ctx: ExecutionContext,
        constraints: list[str] | None,
        timeout: float,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute the selected strategy.

        Args:
            exec_ctx: Execution context
            constraints: Verification constraints
            timeout: Timeout in seconds
            **kwargs: Additional arguments

        Returns:
            StrategyResult from strategy execution
        """
        if not self._router:
            raise ContextFlowError(
                "Router not initialized",
                details={"hint": "Call initialize() first"},
            )

        try:
            result = await asyncio.wait_for(
                self._router.route(
                    task=exec_ctx.task,
                    context=exec_ctx.context,
                    constraints=constraints,
                    force_strategy=(
                        exec_ctx.strategy
                        if exec_ctx.strategy != StrategyType.AUTO
                        else None
                    ),
                    **kwargs,
                ),
                timeout=timeout,
            )
            return result

        except TimeoutError:
            raise StrategyExecutionError(
                strategy=exec_ctx.strategy.value,
                message=f"Strategy execution timed out after {timeout}s",
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _load_documents(self, documents: list[str]) -> str:
        """
        Load content from document paths.

        Args:
            documents: List of document paths (can include globs)

        Returns:
            Combined document content
        """
        contents = []

        for doc_path in documents:
            content = await self._load_single_document(doc_path)
            if content:
                contents.append(content)

        return "\n\n---\n\n".join(contents)

    async def _load_single_document(self, doc_path: str) -> str:
        """
        Load content from a single document path.

        Args:
            doc_path: Document path

        Returns:
            Document content
        """
        path = Path(doc_path)

        # Check if it's a glob pattern
        if "*" in doc_path:
            contents = []
            base_path = Path(doc_path.split("*")[0]) or Path(".")
            pattern = doc_path[len(str(base_path)) :].lstrip("/\\")

            for file_path in base_path.glob(pattern.replace("**", "*")):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        contents.append(f"# {file_path}\n\n{content}")
                    except Exception as e:
                        logger.warning(
                            "Failed to read file",
                            path=str(file_path),
                            error=str(e),
                        )
            return "\n\n---\n\n".join(contents)

        # Regular file
        if not path.exists():
            logger.warning("Document not found", path=doc_path)
            return ""

        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read document", path=doc_path, error=str(e))
            return ""

    async def _initialize_rag(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> None:
        """Initialize RAG system."""
        try:
            from contextflow.rag.embeddings.sentence_transformers import (
                SentenceTransformersProvider,
            )
            from contextflow.rag.temp_rag import TemporaryRAG

            if embedding_provider is None:
                embedding_provider = SentenceTransformersProvider()

            self._rag = TemporaryRAG(
                embedding_provider=embedding_provider,
                auto_chunk=True,
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
            )

            logger.info("RAG initialized")

        except ImportError as e:
            logger.warning("Failed to initialize RAG", error=str(e))

    async def _initialize_agent_pool(self, max_concurrent: int = 5) -> None:
        """Initialize agent pool."""
        try:
            from contextflow.agents.pool import AgentPool, PoolConfig

            config = PoolConfig(
                max_agents=max_concurrent,
                max_concurrent_tasks=max_concurrent,
                auto_scale=True,
            )

            self._agent_pool = AgentPool(
                provider=self._provider,
                config=config,
                rag=self._rag,
            )

            logger.info("Agent pool initialized", max_concurrent=max_concurrent)

        except ImportError as e:
            logger.warning("Failed to initialize agent pool", error=str(e))

    def _aggregate_results(
        self,
        results: list[Any],
        aggregation: str,
    ) -> str:
        """
        Aggregate results from parallel execution.

        Args:
            results: List of agent results
            aggregation: Aggregation strategy

        Returns:
            Aggregated answer
        """
        if not results:
            return "No results to aggregate."

        if aggregation == "consensus":
            # Simple: take the most common answer (or first if all different)
            answers = [r.output for r in results]
            return max(set(answers), key=answers.count)

        elif aggregation == "chain_of_evidence":
            # Combine all answers with evidence chain
            parts = []
            for i, result in enumerate(results):
                parts.append(f"## Analysis {i + 1}\n\n{result.output}")
            return "\n\n---\n\n".join(parts)

        elif aggregation == "all":
            # Return all results
            return "\n\n---\n\n".join([r.output for r in results])

        else:
            # Default to chain_of_evidence
            return self._aggregate_results(results, "chain_of_evidence")

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def close(self) -> None:
        """
        Close all resources and cleanup.

        Should be called when done with the orchestrator.
        """
        logger.info("Closing ContextFlow")

        # Shutdown agent pool
        if self._agent_pool:
            await self._agent_pool.shutdown()
            self._agent_pool = None

        # Clear RAG
        if self._rag:
            self._rag.clear()
            self._rag = None

        # End any active session
        if self._current_session and self._session_manager:
            await self._session_manager.end_session(self._current_session.id)
            self._current_session = None

        self._initialized = False

    async def __aenter__(self) -> ContextFlow:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContextFlow("
            f"provider={self._provider.name!r}, "
            f"initialized={self._initialized}, "
            f"executions={self._execution_count})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_process(
    task: str,
    documents: list[str] | None = None,
    context: str | None = None,
    provider: str = "claude",
    strategy: str = "auto",
    **kwargs: Any,
) -> ProcessResult:
    """
    Quick processing without creating ContextFlow instance.

    Convenience function for one-off processing operations.

    Args:
        task: Task description
        documents: Document paths
        context: Direct context
        provider: Provider name
        strategy: Strategy to use
        **kwargs: Additional arguments

    Returns:
        ProcessResult from processing

    Example:
        result = await quick_process(
            task="Summarize this",
            documents=["file.txt"],
            provider="claude"
        )
    """
    async with ContextFlow(provider=provider) as cf:
        return await cf.process(
            task=task,
            documents=documents,
            context=context,
            strategy=StrategyType(strategy.lower()),
            **kwargs,
        )


async def quick_analyze(
    documents: list[str] | None = None,
    context: str | None = None,
    provider: str = "claude",
) -> CoreContextAnalysis:
    """
    Quick context analysis without full processing.

    Args:
        documents: Document paths
        context: Direct context
        provider: Provider name

    Returns:
        ContextAnalysis with recommendations
    """
    async with ContextFlow(provider=provider) as cf:
        return await cf.analyze(documents=documents, context=context)


def create_contextflow(
    provider: str | None = None,
    **kwargs: Any,
) -> ContextFlow:
    """
    Factory function for creating ContextFlow instances.

    Args:
        provider: Provider name
        **kwargs: Additional configuration

    Returns:
        Configured ContextFlow instance
    """
    return ContextFlow(provider=provider, **kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main Class
    "ContextFlow",
    # Configuration
    "OrchestratorConfig",
    "ExecutionContext",
    # Convenience Functions
    "quick_process",
    "quick_analyze",
    "create_contextflow",
]
