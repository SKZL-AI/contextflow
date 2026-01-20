"""
Unit tests for HooksManager.

Tests lifecycle hooks functionality including:
- Hook registration/unregistration
- Priority ordering
- Sync and async hooks
- Error isolation
- Hook execution
- Global hooks manager
"""

import asyncio
from datetime import datetime

import pytest

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
    post_process,
    pre_process,
    reset_global_hooks_manager,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hooks_manager() -> HooksManager:
    """Create a fresh HooksManager for testing."""
    return HooksManager(name="test")


@pytest.fixture
def sample_context() -> HookContext:
    """Create a sample hook context."""
    return HookContext(
        hook_type=HookType.PRE_PROCESS,
        task="Test task",
        context="Test context",
        strategy="gsd_direct",
        metadata={"key": "value"},
    )


@pytest.fixture
def sync_hook():
    """Create a synchronous hook callback."""

    def hook(ctx: HookContext) -> HookContext:
        ctx.metadata["sync_executed"] = True
        return ctx

    return hook


@pytest.fixture
def async_hook():
    """Create an asynchronous hook callback."""

    async def hook(ctx: HookContext) -> HookContext:
        await asyncio.sleep(0.01)
        ctx.metadata["async_executed"] = True
        return ctx

    return hook


@pytest.fixture
def failing_hook():
    """Create a hook that raises an exception."""

    def hook(ctx: HookContext) -> HookContext:
        raise ValueError("Hook failed intentionally")

    return hook


@pytest.fixture
def autouse_reset_global():
    """Reset global hooks manager before each test."""
    reset_global_hooks_manager()
    yield
    reset_global_hooks_manager()


# =============================================================================
# HookType Tests
# =============================================================================


class TestHookType:
    """Tests for HookType enum."""

    def test_all_hook_types_exist(self) -> None:
        """Test that all expected hook types exist."""
        expected = [
            "PRE_PROCESS",
            "POST_PROCESS",
            "PRE_STRATEGY",
            "POST_STRATEGY",
            "ON_ERROR",
            "ON_VERIFICATION_FAIL",
        ]

        for hook_name in expected:
            assert hasattr(HookType, hook_name)

    def test_hook_type_string_conversion(self) -> None:
        """Test hook type string conversion."""
        assert str(HookType.PRE_PROCESS) == "pre_process"
        assert str(HookType.POST_PROCESS) == "post_process"
        assert str(HookType.ON_ERROR) == "on_error"


# =============================================================================
# HookContext Tests
# =============================================================================


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_context_creation(self) -> None:
        """Test context creation with required fields."""
        ctx = HookContext(hook_type=HookType.PRE_PROCESS)

        assert ctx.hook_type == HookType.PRE_PROCESS
        assert ctx.task == ""
        assert ctx.context == ""
        assert ctx.strategy is None
        assert ctx.result is None
        assert ctx.error is None
        assert isinstance(ctx.metadata, dict)
        assert isinstance(ctx.timestamp, datetime)
        assert len(ctx.execution_id) > 0

    def test_context_with_all_fields(self, sample_context: HookContext) -> None:
        """Test context with all fields populated."""
        assert sample_context.task == "Test task"
        assert sample_context.context == "Test context"
        assert sample_context.strategy == "gsd_direct"
        assert sample_context.metadata["key"] == "value"

    def test_context_with_updates(self, sample_context: HookContext) -> None:
        """Test creating context with updated fields."""
        updated = sample_context.with_updates(task="Updated task", strategy="ralph_structured")

        assert updated.task == "Updated task"
        assert updated.strategy == "ralph_structured"
        # Original should be unchanged
        assert sample_context.task == "Test task"

    def test_context_to_dict(self, sample_context: HookContext) -> None:
        """Test context serialization to dictionary."""
        ctx_dict = sample_context.to_dict()

        assert "hook_type" in ctx_dict
        assert "task" in ctx_dict
        assert "context" in ctx_dict
        assert "timestamp" in ctx_dict
        assert "execution_id" in ctx_dict
        assert ctx_dict["hook_type"] == "pre_process"

    def test_context_truncates_long_content(self) -> None:
        """Test that long content is truncated in to_dict."""
        long_context = "x" * 1000
        ctx = HookContext(hook_type=HookType.PRE_PROCESS, context=long_context)
        ctx_dict = ctx.to_dict()

        # Should be truncated
        assert len(ctx_dict["context"]) < len(long_context)
        assert "..." in ctx_dict["context"]


# =============================================================================
# RegisteredHook Tests
# =============================================================================


class TestRegisteredHook:
    """Tests for RegisteredHook dataclass."""

    def test_registered_hook_creation(self, sync_hook) -> None:
        """Test registered hook creation."""
        hook = RegisteredHook(callback=sync_hook, priority=50, name="test_hook")

        assert hook.callback == sync_hook
        assert hook.priority == 50
        assert hook.name == "test_hook"
        assert hook.enabled is True
        assert isinstance(hook.tags, set)

    def test_hook_auto_name(self, sync_hook) -> None:
        """Test that hook gets auto-generated name."""
        hook = RegisteredHook(callback=sync_hook)

        assert len(hook.name) > 0

    def test_hook_hash_and_equality(self, sync_hook) -> None:
        """Test hook hashing and equality."""
        hook1 = RegisteredHook(callback=sync_hook)
        hook2 = RegisteredHook(callback=sync_hook)

        # Same callback means equal
        assert hook1 == hook2
        assert hash(hook1) == hash(hook2)


# =============================================================================
# HooksManager Registration Tests
# =============================================================================


class TestHookRegistration:
    """Tests for hook registration."""

    def test_register_sync_hook(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test registering a synchronous hook."""
        registered = hooks_manager.register(
            HookType.PRE_PROCESS, sync_hook, priority=50, name="my_sync_hook"
        )

        assert isinstance(registered, RegisteredHook)
        assert registered.name == "my_sync_hook"
        assert registered.priority == 50

        hooks = hooks_manager.get_hooks(HookType.PRE_PROCESS)
        assert len(hooks) == 1

    def test_register_async_hook(self, hooks_manager: HooksManager, async_hook) -> None:
        """Test registering an asynchronous hook."""
        registered = hooks_manager.register(HookType.POST_PROCESS, async_hook, name="my_async_hook")

        assert registered.name == "my_async_hook"

    def test_register_with_tags(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test registering hook with tags."""
        registered = hooks_manager.register(
            HookType.PRE_PROCESS, sync_hook, tags={"logging", "debug"}
        )

        assert "logging" in registered.tags
        assert "debug" in registered.tags

    def test_register_invalid_callback_raises(self, hooks_manager: HooksManager) -> None:
        """Test that non-callable raises ValueError."""
        with pytest.raises(ValueError, match="callable"):
            hooks_manager.register(HookType.PRE_PROCESS, "not a function")

    def test_register_decorator(self, hooks_manager: HooksManager) -> None:
        """Test decorator-style registration."""

        @hooks_manager.register_decorator(HookType.PRE_PROCESS, priority=10)
        def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        hooks = hooks_manager.get_hooks(HookType.PRE_PROCESS)
        assert len(hooks) == 1


# =============================================================================
# HooksManager Unregistration Tests
# =============================================================================


class TestHookUnregistration:
    """Tests for hook unregistration."""

    def test_unregister_by_callback(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test unregistering by callback reference."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)

        removed = hooks_manager.unregister(HookType.PRE_PROCESS, callback=sync_hook)

        assert removed == 1
        assert len(hooks_manager.get_hooks(HookType.PRE_PROCESS)) == 0

    def test_unregister_by_name(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test unregistering by name."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook, name="to_remove")

        removed = hooks_manager.unregister(HookType.PRE_PROCESS, name="to_remove")

        assert removed == 1

    def test_unregister_requires_callback_or_name(self, hooks_manager: HooksManager) -> None:
        """Test that unregister requires callback or name."""
        with pytest.raises(ValueError):
            hooks_manager.unregister(HookType.PRE_PROCESS)

    def test_unregister_all(self, hooks_manager: HooksManager, sync_hook, async_hook) -> None:
        """Test unregistering all hooks."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)
        hooks_manager.register(HookType.POST_PROCESS, async_hook)

        total = hooks_manager.unregister_all()

        assert total == 2
        assert len(hooks_manager.get_hooks(HookType.PRE_PROCESS)) == 0
        assert len(hooks_manager.get_hooks(HookType.POST_PROCESS)) == 0

    def test_unregister_all_specific_type(
        self, hooks_manager: HooksManager, sync_hook, async_hook
    ) -> None:
        """Test unregistering all hooks of a specific type."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)
        hooks_manager.register(HookType.POST_PROCESS, async_hook)

        removed = hooks_manager.unregister_all(HookType.PRE_PROCESS)

        assert removed == 1
        assert len(hooks_manager.get_hooks(HookType.PRE_PROCESS)) == 0
        assert len(hooks_manager.get_hooks(HookType.POST_PROCESS)) == 1


# =============================================================================
# Priority Ordering Tests
# =============================================================================


class TestPriorityOrdering:
    """Tests for priority-based hook ordering."""

    def test_hooks_sorted_by_priority(self, hooks_manager: HooksManager) -> None:
        """Test that hooks are sorted by priority (lowest first)."""

        def hook_a(ctx):
            return ctx

        def hook_b(ctx):
            return ctx

        def hook_c(ctx):
            return ctx

        hooks_manager.register(HookType.PRE_PROCESS, hook_a, priority=100, name="a")
        hooks_manager.register(HookType.PRE_PROCESS, hook_b, priority=50, name="b")
        hooks_manager.register(HookType.PRE_PROCESS, hook_c, priority=75, name="c")

        hooks = hooks_manager.get_hooks(HookType.PRE_PROCESS)

        assert hooks[0].name == "b"  # Priority 50
        assert hooks[1].name == "c"  # Priority 75
        assert hooks[2].name == "a"  # Priority 100

    @pytest.mark.asyncio
    async def test_execution_order_by_priority(
        self, hooks_manager: HooksManager, sample_context: HookContext
    ) -> None:
        """Test that hooks execute in priority order."""
        execution_order = []

        def hook_first(ctx: HookContext) -> HookContext:
            execution_order.append("first")
            return ctx

        def hook_second(ctx: HookContext) -> HookContext:
            execution_order.append("second")
            return ctx

        def hook_third(ctx: HookContext) -> HookContext:
            execution_order.append("third")
            return ctx

        hooks_manager.register(HookType.PRE_PROCESS, hook_third, priority=300)
        hooks_manager.register(HookType.PRE_PROCESS, hook_first, priority=100)
        hooks_manager.register(HookType.PRE_PROCESS, hook_second, priority=200)

        await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert execution_order == ["first", "second", "third"]


# =============================================================================
# Hook Execution Tests
# =============================================================================


class TestHookExecution:
    """Tests for hook execution."""

    @pytest.mark.asyncio
    async def test_execute_sync_hook(
        self, hooks_manager: HooksManager, sync_hook, sample_context: HookContext
    ) -> None:
        """Test executing a synchronous hook."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)

        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert result.hooks_executed == 1
        assert result.hooks_skipped == 0
        assert result.final_context.metadata.get("sync_executed") is True

    @pytest.mark.asyncio
    async def test_execute_async_hook(
        self, hooks_manager: HooksManager, async_hook, sample_context: HookContext
    ) -> None:
        """Test executing an asynchronous hook."""
        hooks_manager.register(HookType.PRE_PROCESS, async_hook)

        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert result.hooks_executed == 1
        assert result.final_context.metadata.get("async_executed") is True

    @pytest.mark.asyncio
    async def test_execute_no_hooks(
        self, hooks_manager: HooksManager, sample_context: HookContext
    ) -> None:
        """Test execution with no hooks registered."""
        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert result.hooks_executed == 0
        assert result.hooks_skipped == 0

    @pytest.mark.asyncio
    async def test_execute_simple(
        self, hooks_manager: HooksManager, sync_hook, sample_context: HookContext
    ) -> None:
        """Test execute_simple returns just the context."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)

        result_ctx = await hooks_manager.execute_simple(HookType.PRE_PROCESS, sample_context)

        assert isinstance(result_ctx, HookContext)
        assert result_ctx.metadata.get("sync_executed") is True


# =============================================================================
# Error Isolation Tests
# =============================================================================


class TestErrorIsolation:
    """Tests for error isolation in hooks."""

    @pytest.mark.asyncio
    async def test_failing_hook_continues_execution(
        self, hooks_manager: HooksManager, failing_hook, sync_hook, sample_context: HookContext
    ) -> None:
        """Test that a failing hook doesn't stop other hooks."""
        hooks_manager.register(HookType.PRE_PROCESS, failing_hook, priority=10)
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook, priority=20)

        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert result.hooks_executed == 1  # sync_hook succeeded
        assert result.hooks_skipped == 1  # failing_hook failed
        assert result.had_errors is True
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_stop_on_error_flag(
        self, hooks_manager: HooksManager, failing_hook, sync_hook, sample_context: HookContext
    ) -> None:
        """Test stop_on_error flag stops execution."""
        hooks_manager.register(HookType.PRE_PROCESS, failing_hook, priority=10)
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook, priority=20)

        result = await hooks_manager.execute(
            HookType.PRE_PROCESS, sample_context, stop_on_error=True
        )

        # Should stop after first error
        assert result.hooks_executed == 0
        assert result.hooks_skipped == 1
        assert result.had_errors is True

    @pytest.mark.asyncio
    async def test_error_details_captured(
        self, hooks_manager: HooksManager, failing_hook, sample_context: HookContext
    ) -> None:
        """Test that error details are captured."""
        hooks_manager.register(HookType.PRE_PROCESS, failing_hook)

        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert len(result.errors) == 1
        error_info = result.errors[0]
        assert "error_type" in error_info
        assert "message" in error_info
        assert error_info["error_type"] == "ValueError"


# =============================================================================
# Enable/Disable Hook Tests
# =============================================================================


class TestHookEnableDisable:
    """Tests for enabling and disabling hooks."""

    def test_disable_hook(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test disabling a hook by name."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook, name="to_disable")

        success = hooks_manager.disable_hook(HookType.PRE_PROCESS, "to_disable")

        assert success is True

        # Should not appear in enabled-only list
        enabled = hooks_manager.get_hooks(HookType.PRE_PROCESS, enabled_only=True)
        assert len(enabled) == 0

    def test_enable_hook(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test enabling a disabled hook."""
        registered = hooks_manager.register(HookType.PRE_PROCESS, sync_hook, name="to_toggle")
        registered.enabled = False

        success = hooks_manager.enable_hook(HookType.PRE_PROCESS, "to_toggle")

        assert success is True

        enabled = hooks_manager.get_hooks(HookType.PRE_PROCESS, enabled_only=True)
        assert len(enabled) == 1

    @pytest.mark.asyncio
    async def test_disabled_hooks_not_executed(
        self, hooks_manager: HooksManager, sync_hook, sample_context: HookContext
    ) -> None:
        """Test that disabled hooks are not executed."""
        registered = hooks_manager.register(HookType.PRE_PROCESS, sync_hook)
        registered.enabled = False

        result = await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        assert result.hooks_executed == 0


# =============================================================================
# Tag Filtering Tests
# =============================================================================


class TestTagFiltering:
    """Tests for tag-based hook filtering."""

    def test_filter_by_tags(self, hooks_manager: HooksManager) -> None:
        """Test filtering hooks by tags."""

        def hook_a(ctx):
            return ctx

        def hook_b(ctx):
            return ctx

        hooks_manager.register(HookType.PRE_PROCESS, hook_a, tags={"logging", "debug"})
        hooks_manager.register(HookType.PRE_PROCESS, hook_b, tags={"production"})

        logging_hooks = hooks_manager.get_hooks(HookType.PRE_PROCESS, tags={"logging"})
        assert len(logging_hooks) == 1

        debug_hooks = hooks_manager.get_hooks(HookType.PRE_PROCESS, tags={"debug"})
        assert len(debug_hooks) == 1

    @pytest.mark.asyncio
    async def test_execute_with_tag_filter(
        self, hooks_manager: HooksManager, sample_context: HookContext
    ) -> None:
        """Test execution with tag filtering."""
        execution_log = []

        def hook_logging(ctx: HookContext) -> HookContext:
            execution_log.append("logging")
            return ctx

        def hook_prod(ctx: HookContext) -> HookContext:
            execution_log.append("prod")
            return ctx

        hooks_manager.register(HookType.PRE_PROCESS, hook_logging, tags={"logging"})
        hooks_manager.register(HookType.PRE_PROCESS, hook_prod, tags={"production"})

        await hooks_manager.execute(HookType.PRE_PROCESS, sample_context, tags={"logging"})

        assert "logging" in execution_log
        assert "prod" not in execution_log


# =============================================================================
# Hook Statistics Tests
# =============================================================================


class TestHookStatistics:
    """Tests for hook statistics."""

    @pytest.mark.asyncio
    async def test_execution_stats(
        self, hooks_manager: HooksManager, sync_hook, sample_context: HookContext
    ) -> None:
        """Test execution statistics tracking."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)

        await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)
        await hooks_manager.execute(HookType.PRE_PROCESS, sample_context)

        stats = hooks_manager.stats

        assert stats["execution_count"] == 2
        assert stats["total_execution_time_ms"] > 0
        assert "average_execution_time_ms" in stats

    def test_list_hooks(self, hooks_manager: HooksManager, sync_hook, async_hook) -> None:
        """Test listing all hooks."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook, name="sync", priority=50)
        hooks_manager.register(HookType.POST_PROCESS, async_hook, name="async", priority=100)

        hooks_list = hooks_manager.list_hooks()

        assert "pre_process" in hooks_list
        assert "post_process" in hooks_list
        assert len(hooks_list["pre_process"]) == 1
        assert hooks_list["pre_process"][0]["name"] == "sync"


# =============================================================================
# Global Hooks Manager Tests
# =============================================================================


class TestGlobalHooksManager:
    """Tests for global hooks manager singleton."""

    def test_get_global_manager(self) -> None:
        """Test getting global hooks manager."""
        reset_global_hooks_manager()

        manager = get_global_hooks_manager()

        assert isinstance(manager, HooksManager)
        assert manager.name == "global"

    def test_global_manager_singleton(self) -> None:
        """Test that global manager is a singleton."""
        reset_global_hooks_manager()

        manager1 = get_global_hooks_manager()
        manager2 = get_global_hooks_manager()

        assert manager1 is manager2

    def test_reset_global_manager(self) -> None:
        """Test resetting global hooks manager."""
        manager1 = get_global_hooks_manager()
        reset_global_hooks_manager()
        manager2 = get_global_hooks_manager()

        assert manager1 is not manager2


# =============================================================================
# Decorator Factory Tests
# =============================================================================


class TestDecoratorFactories:
    """Tests for decorator factory functions."""

    def test_pre_process_decorator(self) -> None:
        """Test pre_process decorator."""
        reset_global_hooks_manager()

        @pre_process(priority=10, name="test_pre")
        def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        hooks = get_global_hooks_manager().get_hooks(HookType.PRE_PROCESS)
        assert len(hooks) == 1
        assert hooks[0].name == "test_pre"

    def test_post_process_decorator(self) -> None:
        """Test post_process decorator."""
        reset_global_hooks_manager()

        @post_process(priority=20)
        def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        hooks = get_global_hooks_manager().get_hooks(HookType.POST_PROCESS)
        assert len(hooks) == 1

    def test_on_error_decorator(self) -> None:
        """Test on_error decorator."""
        reset_global_hooks_manager()

        @on_error()
        def error_handler(ctx: HookContext) -> HookContext:
            return ctx

        hooks = get_global_hooks_manager().get_hooks(HookType.ON_ERROR)
        assert len(hooks) == 1


# =============================================================================
# Utility Hook Tests
# =============================================================================


class TestUtilityHooks:
    """Tests for utility hook functions."""

    @pytest.mark.asyncio
    async def test_logging_hook(self, sample_context: HookContext) -> None:
        """Test logging hook creation."""
        logging_hook = create_logging_hook(log_level="debug")

        result = await logging_hook(sample_context)

        assert result is sample_context  # Should return same context

    @pytest.mark.asyncio
    async def test_timing_hook(self, sample_context: HookContext) -> None:
        """Test timing hook creation."""
        timing_hook = create_timing_hook(metric_name="test_metric")

        # First call sets start time
        ctx1 = await timing_hook(sample_context)
        assert "test_metric_start" in ctx1.metadata

        # Second call calculates duration
        ctx2 = await timing_hook(ctx1)
        assert "test_metric_duration_ms" in ctx2.metadata
        assert ctx2.metadata["test_metric_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_compose_hooks(self, sample_context: HookContext) -> None:
        """Test composing multiple hooks."""

        def hook1(ctx: HookContext) -> HookContext:
            ctx.metadata["step1"] = True
            return ctx

        def hook2(ctx: HookContext) -> HookContext:
            ctx.metadata["step2"] = True
            return ctx

        composed = compose_hooks(hook1, hook2)
        result = await composed(sample_context)

        assert result.metadata.get("step1") is True
        assert result.metadata.get("step2") is True


# =============================================================================
# HookExecutionResult Tests
# =============================================================================


class TestHookExecutionResult:
    """Tests for HookExecutionResult."""

    def test_result_structure(self, sample_context: HookContext) -> None:
        """Test result structure."""
        result = HookExecutionResult(
            final_context=sample_context,
            hooks_executed=2,
            hooks_skipped=1,
            execution_time_ms=50.5,
            errors=[{"hook_name": "test", "error_type": "ValueError", "message": "test"}],
        )

        assert result.hooks_executed == 2
        assert result.hooks_skipped == 1
        assert result.execution_time_ms == 50.5
        assert result.had_errors is True

    def test_result_to_dict(self, sample_context: HookContext) -> None:
        """Test result serialization."""
        result = HookExecutionResult(final_context=sample_context, hooks_executed=1)
        result_dict = result.to_dict()

        assert "final_context" in result_dict
        assert "hooks_executed" in result_dict
        assert "execution_time_ms" in result_dict
        assert "had_errors" in result_dict


# =============================================================================
# HooksManager Repr Test
# =============================================================================


class TestHooksManagerRepr:
    """Tests for HooksManager string representation."""

    def test_repr(self, hooks_manager: HooksManager, sync_hook) -> None:
        """Test manager string representation."""
        hooks_manager.register(HookType.PRE_PROCESS, sync_hook)

        repr_str = repr(hooks_manager)

        assert "HooksManager" in repr_str
        assert "test" in repr_str  # name
        assert "1" in repr_str  # total hooks
