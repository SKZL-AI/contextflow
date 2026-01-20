"""
Lifecycle Hooks System for ContextFlow.

Provides hooks for intercepting and modifying behavior at various points
in the ContextFlow processing pipeline. Inspired by Boris Step 9 and Claude-Mem.

Use Cases:
    - PostProcess: Code formatting, output transformation
    - OnVerificationFail: Retry with adjusted prompt
    - OnError: Logging, notification, recovery actions
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Union

from contextflow.utils.logging import get_logger

logger = get_logger(__name__)

# Type definitions
SyncHookCallback = Callable[["HookContext"], "HookContext"]
AsyncHookCallback = Callable[["HookContext"], Awaitable["HookContext"]]
HookCallback = Union[SyncHookCallback, AsyncHookCallback]


class HookType(str, Enum):
    """Types of lifecycle hooks available in ContextFlow."""
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    PRE_STRATEGY = "pre_strategy"
    POST_STRATEGY = "post_strategy"
    ON_ERROR = "on_error"
    ON_VERIFICATION_FAIL = "on_verification_fail"

    def __str__(self) -> str:
        return self.value


@dataclass
class HookContext:
    """
    Context data passed to hook callbacks.

    Attributes:
        hook_type: The type of hook being executed
        task: The task/query being processed
        context: The context/documents being analyzed
        strategy: The strategy being used (if applicable)
        result: The result from processing (for post hooks)
        error: The exception that occurred (for error hooks)
        verification_result: Verification details (for verification hooks)
        metadata: Arbitrary metadata for hook communication
        timestamp: When this context was created
        execution_id: Unique identifier for the current execution
    """
    hook_type: HookType
    task: str = ""
    context: str = ""
    strategy: str | None = None
    result: Any | None = None
    error: Exception | None = None
    verification_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_id: str = ""

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = f"{self.hook_type.value}_{id(self)}_{int(time.time() * 1000)}"

    def with_updates(self, **kwargs: Any) -> HookContext:
        """Create a new HookContext with updated fields."""
        data = {
            "hook_type": self.hook_type, "task": self.task, "context": self.context,
            "strategy": self.strategy, "result": self.result, "error": self.error,
            "verification_result": self.verification_result,
            "metadata": self.metadata.copy(), "timestamp": self.timestamp,
            "execution_id": self.execution_id,
        }
        data.update(kwargs)
        return HookContext(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "hook_type": self.hook_type.value, "task": self.task,
            "context": self.context[:500] + "..." if len(self.context) > 500 else self.context,
            "strategy": self.strategy,
            "result": str(self.result)[:200] if self.result else None,
            "error": str(self.error) if self.error else None,
            "verification_result": self.verification_result,
            "metadata": self.metadata, "timestamp": self.timestamp.isoformat(),
            "execution_id": self.execution_id,
        }


@dataclass
class RegisteredHook:
    """Represents a registered hook callback with priority and metadata."""
    callback: HookCallback
    priority: int = 100
    name: str = ""
    enabled: bool = True
    tags: set[str] = field(default_factory=set)
    registered_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.callback, "__name__", f"hook_{id(self.callback)}")

    def __hash__(self) -> int:
        return hash(id(self.callback))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisteredHook):
            return NotImplemented
        return id(self.callback) == id(other.callback)


@dataclass
class HookExecutionResult:
    """Result from executing hooks with execution metadata."""
    final_context: HookContext
    hooks_executed: int = 0
    hooks_skipped: int = 0
    execution_time_ms: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def had_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_context": self.final_context.to_dict(),
            "hooks_executed": self.hooks_executed, "hooks_skipped": self.hooks_skipped,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "errors": self.errors, "had_errors": self.had_errors, "metadata": self.metadata,
        }


class HooksManager:
    """
    Lifecycle Hooks Manager for ContextFlow Processing.

    Features: Hook registration with priorities, sync/async callback support,
    error isolation, hook chaining, enable/disable, tag-based filtering.

    Example:
        manager = HooksManager()
        manager.register(HookType.PRE_PROCESS, my_hook, priority=10)
        result = await manager.execute(HookType.PRE_PROCESS, context)
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._hooks: dict[HookType, list[RegisteredHook]] = {ht: [] for ht in HookType}
        self._logger = get_logger(f"contextflow.hooks.{name}")
        self._execution_count: int = 0
        self._total_execution_time_ms: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            "name": self._name, "execution_count": self._execution_count,
            "total_execution_time_ms": round(self._total_execution_time_ms, 2),
            "average_execution_time_ms": (
                round(self._total_execution_time_ms / self._execution_count, 2)
                if self._execution_count > 0 else 0.0
            ),
            "hooks_registered": {ht.value: len(h) for ht, h in self._hooks.items()},
        }

    def register(
        self, hook_type: HookType, callback: HookCallback,
        priority: int = 100, name: str | None = None, tags: set[str] | None = None,
    ) -> RegisteredHook:
        """
        Register a hook callback for a specific hook type.

        Args:
            hook_type: The type of hook to register for
            callback: The callback function (sync or async)
            priority: Execution priority (default 100, lower = earlier)
            name: Optional human-readable name
            tags: Optional tags for filtering

        Returns:
            RegisteredHook instance

        Raises:
            ValueError: If callback is not callable
        """
        if not callable(callback):
            raise ValueError(f"Callback must be callable, got {type(callback)}")

        hook = RegisteredHook(callback=callback, priority=priority, name=name or "", tags=tags or set())
        self._hooks[hook_type].append(hook)
        self._hooks[hook_type].sort(key=lambda h: h.priority)
        self._logger.debug("Hook registered", hook_type=hook_type.value, name=hook.name, priority=priority)
        return hook

    def register_decorator(
        self, hook_type: HookType, priority: int = 100,
        name: str | None = None, tags: set[str] | None = None,
    ) -> Callable[[HookCallback], HookCallback]:
        """Decorator factory for registering hooks."""
        def decorator(callback: HookCallback) -> HookCallback:
            self.register(hook_type, callback, priority, name, tags)
            return callback
        return decorator

    def unregister(
        self, hook_type: HookType,
        callback: HookCallback | None = None, name: str | None = None,
    ) -> int:
        """Unregister hook(s) by callback or name. Returns count removed."""
        if callback is None and name is None:
            raise ValueError("Must provide either callback or name to unregister")

        original = len(self._hooks[hook_type])
        if callback is not None:
            self._hooks[hook_type] = [h for h in self._hooks[hook_type] if h.callback is not callback]
        elif name is not None:
            self._hooks[hook_type] = [h for h in self._hooks[hook_type] if h.name != name]
        return original - len(self._hooks[hook_type])

    def unregister_all(self, hook_type: HookType | None = None) -> int:
        """Unregister all hooks, optionally for a specific type."""
        total = 0
        types = [hook_type] if hook_type else list(HookType)
        for ht in types:
            total += len(self._hooks[ht])
            self._hooks[ht] = []
        return total

    def get_hooks(
        self, hook_type: HookType, tags: set[str] | None = None, enabled_only: bool = True,
    ) -> list[RegisteredHook]:
        """Get registered hooks for a type, optionally filtered."""
        hooks = self._hooks[hook_type]
        if enabled_only:
            hooks = [h for h in hooks if h.enabled]
        if tags:
            hooks = [h for h in hooks if tags.issubset(h.tags)]
        return hooks

    def enable_hook(self, hook_type: HookType, name: str) -> bool:
        """Enable a hook by name. Returns True if found."""
        for h in self._hooks[hook_type]:
            if h.name == name:
                h.enabled = True
                return True
        return False

    def disable_hook(self, hook_type: HookType, name: str) -> bool:
        """Disable a hook by name. Returns True if found."""
        for h in self._hooks[hook_type]:
            if h.name == name:
                h.enabled = False
                return True
        return False

    async def execute(
        self, hook_type: HookType, context: HookContext,
        tags: set[str] | None = None, stop_on_error: bool = False,
    ) -> HookExecutionResult:
        """
        Execute all registered hooks for a type.

        Hooks run in priority order. Errors are caught unless stop_on_error is True.

        Args:
            hook_type: The type of hook to execute
            context: The hook context to pass to callbacks
            tags: Optional tags to filter hooks
            stop_on_error: If True, stop execution on first error

        Returns:
            HookExecutionResult with final context and metadata
        """
        start_time = time.time()
        hooks = self.get_hooks(hook_type, tags=tags, enabled_only=True)
        result = HookExecutionResult(final_context=context, metadata={"hook_type": hook_type.value})

        if not hooks:
            return result

        current = context
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook.callback):
                    new_ctx = await hook.callback(current)
                else:
                    new_ctx = hook.callback(current)

                if not isinstance(new_ctx, HookContext):
                    self._logger.warning("Hook returned invalid type", hook_name=hook.name)
                    result.errors.append({
                        "hook_name": hook.name, "error_type": "InvalidReturnType",
                        "message": f"Expected HookContext, got {type(new_ctx).__name__}",
                    })
                    result.hooks_skipped += 1
                    continue

                current = new_ctx
                result.hooks_executed += 1

            except Exception as e:
                result.errors.append({"hook_name": hook.name, "error_type": type(e).__name__, "message": str(e)})
                result.hooks_skipped += 1
                self._logger.warning("Hook execution failed", hook_name=hook.name, error_message=str(e), exc_info=True)
                if stop_on_error:
                    break

        result.final_context = current
        result.execution_time_ms = (time.time() - start_time) * 1000
        self._execution_count += 1
        self._total_execution_time_ms += result.execution_time_ms
        return result

    async def execute_simple(self, hook_type: HookType, context: HookContext) -> HookContext:
        """Execute hooks and return just the final context."""
        return (await self.execute(hook_type, context)).final_context

    def list_hooks(self) -> dict[str, list[dict[str, Any]]]:
        """List all registered hooks with details."""
        return {
            ht.value: [{"name": h.name, "priority": h.priority, "enabled": h.enabled, "tags": list(h.tags)}
                       for h in hooks]
            for ht, hooks in self._hooks.items()
        }

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._hooks.values())
        return f"HooksManager(name={self._name!r}, total_hooks={total})"


# Global Hooks Manager Singleton
_global_hooks_manager: HooksManager | None = None


def get_global_hooks_manager() -> HooksManager:
    """Get the global hooks manager singleton."""
    global _global_hooks_manager
    if _global_hooks_manager is None:
        _global_hooks_manager = HooksManager(name="global")
    return _global_hooks_manager


def reset_global_hooks_manager() -> None:
    """Reset the global hooks manager. Useful for testing."""
    global _global_hooks_manager
    _global_hooks_manager = HooksManager(name="global")


# Convenience Decorator Factories
def _make_decorator(hook_type: HookType):
    def decorator_factory(
        priority: int = 100, name: str | None = None,
        tags: set[str] | None = None, manager: HooksManager | None = None,
    ) -> Callable[[HookCallback], HookCallback]:
        mgr = manager or get_global_hooks_manager()
        return mgr.register_decorator(hook_type, priority, name, tags)
    return decorator_factory


def pre_process(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering PRE_PROCESS hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.PRE_PROCESS, priority, name, tags)


def post_process(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering POST_PROCESS hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.POST_PROCESS, priority, name, tags)


def pre_strategy(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering PRE_STRATEGY hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.PRE_STRATEGY, priority, name, tags)


def post_strategy(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering POST_STRATEGY hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.POST_STRATEGY, priority, name, tags)


def on_error(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering ON_ERROR hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.ON_ERROR, priority, name, tags)


def on_verification_fail(
    priority: int = 100, name: str | None = None,
    tags: set[str] | None = None, manager: HooksManager | None = None,
) -> Callable[[HookCallback], HookCallback]:
    """Decorator for registering ON_VERIFICATION_FAIL hooks."""
    return (manager or get_global_hooks_manager()).register_decorator(HookType.ON_VERIFICATION_FAIL, priority, name, tags)


# Hook Utilities
def create_logging_hook(log_level: str = "info", include_context: bool = False) -> AsyncHookCallback:
    """Create a logging hook callback."""
    hook_logger = get_logger("contextflow.hooks.logging")
    log_func = getattr(hook_logger, log_level, hook_logger.info)

    async def logging_hook(context: HookContext) -> HookContext:
        data: dict[str, Any] = {"hook_type": context.hook_type.value, "execution_id": context.execution_id}
        if context.task:
            data["task"] = context.task[:100]
        if include_context and context.context:
            data["context_length"] = len(context.context)
        if context.error:
            data["error"] = str(context.error)
        if context.strategy:
            data["strategy"] = context.strategy
        log_func("Hook executed", **data)
        return context
    return logging_hook


def create_timing_hook(metric_name: str = "hook_execution") -> AsyncHookCallback:
    """Create a timing hook that tracks execution time in metadata."""
    async def timing_hook(context: HookContext) -> HookContext:
        key = f"{metric_name}_start"
        if key not in context.metadata:
            context.metadata[key] = time.time()
        else:
            start = context.metadata.pop(key)
            context.metadata[f"{metric_name}_duration_ms"] = round((time.time() - start) * 1000, 2)
        return context
    return timing_hook


def compose_hooks(*callbacks: HookCallback) -> AsyncHookCallback:
    """Compose multiple hook callbacks into a single callback."""
    async def composed(context: HookContext) -> HookContext:
        current = context
        for cb in callbacks:
            current = await cb(current) if asyncio.iscoroutinefunction(cb) else cb(current)
        return current
    return composed


__all__ = [
    "HookType", "HookContext", "RegisteredHook", "HookExecutionResult", "HooksManager",
    "get_global_hooks_manager", "reset_global_hooks_manager",
    "pre_process", "post_process", "pre_strategy", "post_strategy", "on_error", "on_verification_fail",
    "create_logging_hook", "create_timing_hook", "compose_hooks",
    "HookCallback", "SyncHookCallback", "AsyncHookCallback",
]
