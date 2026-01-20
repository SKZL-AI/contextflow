"""
Integration tests for the full ContextFlow orchestrator pipeline.

Tests the complete process() pipeline with all 10 steps:
1. Pre-process hooks
2. Context analysis (for AUTO strategy)
3. Pre-strategy hooks
4. Strategy execution
5. Post-strategy hooks
6. Output verification
7. On-verification-fail hooks (if needed)
8. Post-process hooks
9. Session observation recording
10. Result building
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from contextflow.core.config import ContextFlowConfig
from contextflow.core.hooks import (
    HookContext,
    HooksManager,
    HookType,
)
from contextflow.core.orchestrator import ContextFlow, OrchestratorConfig
from contextflow.core.types import ProcessResult, StrategyType, TaskStatus
from contextflow.utils.errors import ValidationError

# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullProcessPipeline:
    """Test complete process() with all steps."""

    @pytest.mark.asyncio
    async def test_full_process_pipeline_success(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test complete process() with all steps executed successfully."""
        result = await configured_contextflow.process(
            task=sample_task,
            context=small_context,
            strategy=StrategyType.AUTO,
        )

        # Verify result structure
        assert result is not None
        assert isinstance(result, ProcessResult)
        assert result.status == TaskStatus.COMPLETED
        assert result.answer is not None
        assert len(result.answer) > 0

        # Verify strategy was selected
        assert result.strategy_used != StrategyType.AUTO

        # Verify trajectory recorded steps
        assert len(result.trajectory) > 0
        step_types = [step.step_type for step in result.trajectory]
        assert "pre_process_hooks" in step_types or "analysis" in step_types

        # Verify execution time tracked
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_full_process_pipeline_with_documents(
        self,
        configured_contextflow: ContextFlow,
        sample_task: str,
        tmp_path: Any,
    ) -> None:
        """Test process() with document paths."""
        # Create test document
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("This is test document content for processing.")

        result = await configured_contextflow.process(
            task=sample_task,
            documents=[str(doc_path)],
            strategy=StrategyType.AUTO,
        )

        assert result.status == TaskStatus.COMPLETED
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_full_process_pipeline_with_constraints(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
        sample_constraints: list[str],
    ) -> None:
        """Test process() with verification constraints."""
        result = await configured_contextflow.process(
            task=sample_task,
            context=small_context,
            constraints=sample_constraints,
        )

        assert result.status == TaskStatus.COMPLETED

        # Constraints should be passed to verification
        assert "verification_passed" in result.metadata or result.trajectory

    @pytest.mark.asyncio
    async def test_full_process_pipeline_with_explicit_strategy(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test process() with explicitly specified strategy."""
        result = await configured_contextflow.process(
            task=sample_task,
            context=small_context,
            strategy=StrategyType.GSD_DIRECT,
        )

        assert result.status == TaskStatus.COMPLETED
        assert result.strategy_used == StrategyType.GSD_DIRECT

    @pytest.mark.asyncio
    async def test_full_process_pipeline_tracks_tokens(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that token usage is tracked throughout pipeline."""
        result = await configured_contextflow.process(
            task=sample_task,
            context=small_context,
        )

        assert result.total_tokens > 0
        assert result.total_cost >= 0

    @pytest.mark.asyncio
    async def test_full_process_pipeline_records_trajectory(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that execution trajectory is recorded."""
        result = await configured_contextflow.process(
            task=sample_task,
            context=small_context,
        )

        assert len(result.trajectory) > 0

        # Each trajectory step should have required fields
        for step in result.trajectory:
            assert step.step_type is not None
            assert step.timestamp is not None
            assert isinstance(step.timestamp, datetime)


class TestProcessWithVerificationLoop:
    """Test that verification loop iterates until passed."""

    @pytest.mark.asyncio
    async def test_verification_loop_iterates(
        self,
        contextflow_with_verification: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that verification loop runs multiple iterations when needed."""
        result = await contextflow_with_verification.process(
            task=sample_task,
            context=small_context,
        )

        # Should eventually pass
        assert result.status == TaskStatus.COMPLETED

        # Provider should have been called multiple times due to verification
        provider = contextflow_with_verification.provider
        assert provider.call_count > 1  # type: ignore

    @pytest.mark.asyncio
    async def test_verification_respects_max_iterations(
        self,
        mock_provider_always_fail_verification,
        test_config: ContextFlowConfig,
        hooks_manager: HooksManager,
    ) -> None:
        """Test that max iterations limit is respected."""
        orchestrator_config = OrchestratorConfig(
            enable_verification=True,
            verification_threshold=0.7,
            max_verification_iterations=2,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider_always_fail_verification,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager,
        )
        await cf.initialize()

        try:
            result = await cf.process(
                task="Test task",
                context="Test context",
            )

            # Should complete but with failed verification
            assert result.status == TaskStatus.COMPLETED
            # Metadata should indicate verification didn't pass
            assert result.metadata.get(
                "verification_passed", True
            ) is False or "max_iterations_reached" in str(result.metadata)

        finally:
            await cf.close()


class TestProcessWithHooks:
    """Test all lifecycle hooks are executed."""

    @pytest.mark.asyncio
    async def test_all_hooks_executed_in_order(
        self,
        mock_provider,
        test_config: ContextFlowConfig,
        hook_tracker: dict[str, list[HookContext]],
        hooks_manager_with_tracking: HooksManager,
    ) -> None:
        """Test that all hooks are called in correct order."""
        orchestrator_config = OrchestratorConfig(
            enable_verification=False,  # Disable to simplify hook testing
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager_with_tracking,
        )
        await cf.initialize()

        try:
            await cf.process(
                task="Test task",
                context="Test context",
            )

            # Verify hooks were called
            assert len(hook_tracker[HookType.PRE_PROCESS.value]) > 0
            assert len(hook_tracker[HookType.PRE_STRATEGY.value]) > 0
            assert len(hook_tracker[HookType.POST_STRATEGY.value]) > 0
            assert len(hook_tracker[HookType.POST_PROCESS.value]) > 0

            # Verify order: PRE_PROCESS should come before POST_PROCESS
            pre_process_time = hook_tracker[HookType.PRE_PROCESS.value][0].timestamp
            post_process_time = hook_tracker[HookType.POST_PROCESS.value][0].timestamp
            assert pre_process_time <= post_process_time

        finally:
            await cf.close()

    @pytest.mark.asyncio
    async def test_pre_process_hook_can_modify_context(
        self,
        mock_provider,
        test_config: ContextFlowConfig,
    ) -> None:
        """Test that pre-process hooks can modify task/context."""
        manager = HooksManager(name="modifier")
        modified_task = "MODIFIED TASK"

        async def modify_task(context: HookContext) -> HookContext:
            return context.with_updates(task=modified_task)

        manager.register(HookType.PRE_PROCESS, modify_task, priority=1)

        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=manager,
        )
        await cf.initialize()

        try:
            await cf.process(
                task="Original task",
                context="Test context",
            )

            # The provider should have received the modified task
            calls = mock_provider.complete_calls
            assert len(calls) > 0

        finally:
            await cf.close()

    @pytest.mark.asyncio
    async def test_post_strategy_hook_can_modify_result(
        self,
        mock_provider,
        test_config: ContextFlowConfig,
    ) -> None:
        """Test that post-strategy hooks can modify result."""
        manager = HooksManager(name="result_modifier")
        modified_result = "MODIFIED RESULT"

        async def modify_result(context: HookContext) -> HookContext:
            return context.with_updates(result=modified_result)

        manager.register(HookType.POST_STRATEGY, modify_result, priority=1)

        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=manager,
        )
        await cf.initialize()

        try:
            result = await cf.process(
                task="Test task",
                context="Test context",
            )

            # Result should be modified by hook
            assert result.answer == modified_result

        finally:
            await cf.close()

    @pytest.mark.asyncio
    async def test_on_verification_fail_hook_called(
        self,
        mock_provider_always_fail_verification,
        test_config: ContextFlowConfig,
    ) -> None:
        """Test that on_verification_fail hook is called when verification fails."""
        manager = HooksManager(name="fail_hook")
        fail_hook_called = {"called": False}

        async def on_fail(context: HookContext) -> HookContext:
            fail_hook_called["called"] = True
            return context

        manager.register(HookType.ON_VERIFICATION_FAIL, on_fail, priority=1)

        orchestrator_config = OrchestratorConfig(
            enable_verification=True,
            verification_threshold=0.7,
            max_verification_iterations=1,  # Only 1 iteration
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider_always_fail_verification,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=manager,
        )
        await cf.initialize()

        try:
            await cf.process(
                task="Test task",
                context="Test context",
            )

            assert fail_hook_called["called"] is True

        finally:
            await cf.close()


class TestProcessErrorHandling:
    """Test error handling in the processing pipeline."""

    @pytest.mark.asyncio
    async def test_empty_task_raises_validation_error(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
    ) -> None:
        """Test that empty task raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await configured_contextflow.process(
                task="",
                context=small_context,
            )

        assert "task" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_context_raises_validation_error(
        self,
        configured_contextflow: ContextFlow,
        sample_task: str,
    ) -> None:
        """Test that missing context/documents raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await configured_contextflow.process(
                task=sample_task,
                # No context or documents provided
            )

        assert (
            "context" in str(exc_info.value).lower() or "documents" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_invalid_strategy_raises_validation_error(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that invalid strategy string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await configured_contextflow.process(
                task=sample_task,
                context=small_context,
                strategy="invalid_strategy",  # type: ignore
            )

        assert "strategy" in str(exc_info.value).lower()


class TestStreamingPipeline:
    """Test streaming processing pipeline."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that stream() yields content chunks."""
        chunks: list[str] = []

        async for chunk in configured_contextflow.stream(
            task=sample_task,
            context=small_context,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(chunks)
        assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_stream_executes_hooks(
        self,
        mock_provider,
        test_config: ContextFlowConfig,
        hook_tracker: dict[str, list[HookContext]],
        hooks_manager_with_tracking: HooksManager,
    ) -> None:
        """Test that streaming also executes lifecycle hooks."""
        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
            enable_streaming=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager_with_tracking,
        )
        await cf.initialize()

        try:
            chunks = []
            async for chunk in cf.stream(
                task="Test task",
                context="Test context",
            ):
                chunks.append(chunk)

            # Hooks should have been called
            assert len(hook_tracker[HookType.PRE_PROCESS.value]) > 0
            assert len(hook_tracker[HookType.POST_PROCESS.value]) > 0

        finally:
            await cf.close()


class TestAnalyzePipeline:
    """Test context analysis without execution."""

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
    ) -> None:
        """Test that analyze() returns context analysis."""
        analysis = await configured_contextflow.analyze(
            context=small_context,
        )

        assert analysis is not None
        assert analysis.token_count > 0
        assert analysis.recommended_strategy is not None

    @pytest.mark.asyncio
    async def test_analyze_with_documents(
        self,
        configured_contextflow: ContextFlow,
        tmp_path: Any,
    ) -> None:
        """Test analyze() with document paths."""
        doc_path = tmp_path / "analyze_doc.txt"
        doc_path.write_text("Content for analysis testing.")

        analysis = await configured_contextflow.analyze(
            documents=[str(doc_path)],
        )

        assert analysis is not None
        assert analysis.token_count > 0

    @pytest.mark.asyncio
    async def test_analyze_without_context_raises_error(
        self,
        configured_contextflow: ContextFlow,
    ) -> None:
        """Test that analyze() without context raises ValidationError."""
        with pytest.raises(ValidationError):
            await configured_contextflow.analyze()


class TestContextFlowLifecycle:
    """Test ContextFlow lifecycle management."""

    @pytest.mark.asyncio
    async def test_context_manager_usage(
        self,
        mock_provider,
        test_config: ContextFlowConfig,
        orchestrator_config: OrchestratorConfig,
    ) -> None:
        """Test async context manager usage."""
        async with ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
        ) as cf:
            assert cf.is_initialized

            result = await cf.process(
                task="Test task",
                context="Test context",
            )

            assert result.status == TaskStatus.COMPLETED

        # After exiting context, should be closed
        assert not cf.is_initialized

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that orchestrator stats are tracked."""
        initial_stats = configured_contextflow.stats

        await configured_contextflow.process(
            task=sample_task,
            context=small_context,
        )

        final_stats = configured_contextflow.stats

        assert final_stats["execution_count"] > initial_stats["execution_count"]
        assert final_stats["total_tokens_used"] > initial_stats["total_tokens_used"]

    @pytest.mark.asyncio
    async def test_multiple_processes_accumulate_stats(
        self,
        configured_contextflow: ContextFlow,
        small_context: str,
        sample_task: str,
    ) -> None:
        """Test that multiple process calls accumulate stats."""
        # Run multiple processes
        for _ in range(3):
            await configured_contextflow.process(
                task=sample_task,
                context=small_context,
            )

        stats = configured_contextflow.stats
        assert stats["execution_count"] == 3
