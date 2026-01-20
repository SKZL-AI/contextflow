#!/usr/bin/env python
"""
Lifecycle Hooks Example.

This example demonstrates ContextFlow's hook system for intercepting
and modifying behavior at various points in the processing pipeline:
- PRE_PROCESS: Before processing starts
- POST_PROCESS: After processing completes
- PRE_STRATEGY: Before strategy execution
- POST_STRATEGY: After strategy execution
- ON_ERROR: When an error occurs
- ON_VERIFICATION_FAIL: When verification fails

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable
    - Install contextflow: pip install -e .

Run:
    python examples/06_hooks.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow
from contextflow.core.hooks import (
    HooksManager,
    HookType,
    HookContext,
    get_global_hooks_manager,
    reset_global_hooks_manager,
    pre_process,
    post_process,
    pre_strategy,
    post_strategy,
    on_error,
    on_verification_fail,
    create_logging_hook,
    create_timing_hook,
    compose_hooks,
)


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_DOCUMENT = """
# Product Requirements Document

## Feature: User Dashboard

### Overview
Create a personalized dashboard for users to view their activity.

### Requirements
1. Display recent orders (last 30 days)
2. Show account balance and pending payments
3. Provide quick links to common actions
4. Support mobile and desktop layouts

### Technical Notes
- Use React for frontend
- Cache data for 5 minutes
- Lazy load secondary components
"""


# =============================================================================
# Custom Hook Functions
# =============================================================================


async def logging_pre_process_hook(context: HookContext) -> HookContext:
    """
    Hook that logs when processing starts.
    """
    print(f"  [PRE_PROCESS] Starting task: {context.task[:50]}...")
    print(f"  [PRE_PROCESS] Context length: {len(context.context)} chars")
    return context


async def logging_post_process_hook(context: HookContext) -> HookContext:
    """
    Hook that logs when processing completes.
    """
    result_preview = str(context.result)[:100] if context.result else "None"
    print(f"  [POST_PROCESS] Completed! Result preview: {result_preview}...")
    return context


async def timing_hook(context: HookContext) -> HookContext:
    """
    Hook that tracks timing information.
    """
    if "start_time" not in context.metadata:
        # First call - record start time
        context.metadata["start_time"] = time.time()
        print(f"  [TIMING] Timer started")
    else:
        # Second call - calculate duration
        duration = time.time() - context.metadata["start_time"]
        context.metadata["duration_seconds"] = duration
        print(f"  [TIMING] Duration: {duration:.2f}s")
    return context


async def task_modifier_hook(context: HookContext) -> HookContext:
    """
    Hook that modifies the task before processing.
    """
    # Add instructions to the task
    modified_task = f"{context.task} Be concise and use bullet points."
    print(f"  [MODIFIER] Modified task to include formatting instructions")
    return context.with_updates(task=modified_task)


async def result_formatter_hook(context: HookContext) -> HookContext:
    """
    Hook that formats the result after processing.
    """
    if context.result:
        # Add a header to the result
        formatted = f"=== ContextFlow Output ===\n\n{context.result}\n\n=== End Output ==="
        print(f"  [FORMATTER] Added header/footer to result")
        return context.with_updates(result=formatted)
    return context


async def error_handler_hook(context: HookContext) -> HookContext:
    """
    Hook that handles errors gracefully.
    """
    if context.error:
        print(f"  [ERROR] Error occurred: {type(context.error).__name__}")
        print(f"  [ERROR] Message: {str(context.error)}")
        # Could add error reporting, notifications, etc.
    return context


async def verification_fail_hook(context: HookContext) -> HookContext:
    """
    Hook that handles verification failures.
    """
    if context.verification_result:
        print(f"  [VERIFICATION_FAIL] Verification did not pass")
        print(f"  [VERIFICATION_FAIL] Details: {context.verification_result}")
        # Could add retry logic, fallback strategies, etc.
    return context


# =============================================================================
# Example Functions
# =============================================================================


async def basic_hooks_example() -> None:
    """
    Basic hook registration and execution.

    Shows how to register and use hooks with ContextFlow.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Hooks Registration")
    print("=" * 60)

    # Reset global hooks for clean state
    reset_global_hooks_manager()

    # Get global hooks manager
    hooks = get_global_hooks_manager()

    # Register hooks
    hooks.register(HookType.PRE_PROCESS, logging_pre_process_hook, priority=10)
    hooks.register(HookType.POST_PROCESS, logging_post_process_hook, priority=10)

    print("\nRegistered hooks:")
    for hook_type, hook_list in hooks.list_hooks().items():
        if hook_list:
            print(f"  {hook_type}: {[h['name'] for h in hook_list]}")

    print("\nProcessing with hooks active...")

    async with ContextFlow() as cf:
        result = await cf.process(
            task="Summarize the key requirements",
            context=SAMPLE_DOCUMENT,
        )

    print(f"\nResult: {result.answer[:200]}...")


async def decorator_hooks_example() -> None:
    """
    Using decorators to register hooks.

    Shows the decorator syntax for hook registration.
    """
    print("\n" + "=" * 60)
    print("Example 2: Decorator-Based Hooks")
    print("=" * 60)

    # Create a fresh hooks manager for this example
    hooks = HooksManager(name="decorator_example")

    # Register using decorators
    @hooks.register_decorator(HookType.PRE_PROCESS, priority=5)
    async def my_pre_hook(ctx: HookContext) -> HookContext:
        print(f"  [DECORATOR] Pre-process hook called")
        ctx.metadata["custom_data"] = "added by hook"
        return ctx

    @hooks.register_decorator(HookType.POST_PROCESS, priority=5)
    async def my_post_hook(ctx: HookContext) -> HookContext:
        print(f"  [DECORATOR] Post-process hook called")
        print(f"  [DECORATOR] Custom data: {ctx.metadata.get('custom_data')}")
        return ctx

    print("\nHooks registered via decorators:")
    print(f"  Pre-process: {hooks.get_hooks(HookType.PRE_PROCESS)}")

    # Use the custom hooks manager with ContextFlow
    async with ContextFlow(hooks_manager=hooks) as cf:
        result = await cf.process(
            task="List the technical requirements",
            context=SAMPLE_DOCUMENT,
        )

    print(f"\nResult: {result.answer[:200]}...")


async def priority_hooks_example() -> None:
    """
    Hook priority and execution order.

    Shows how hooks are executed in priority order.
    """
    print("\n" + "=" * 60)
    print("Example 3: Hook Priority and Order")
    print("=" * 60)

    hooks = HooksManager(name="priority_example")

    # Register hooks with different priorities
    # Lower priority number = executed first

    async def hook_a(ctx: HookContext) -> HookContext:
        ctx.metadata.setdefault("execution_order", []).append("A")
        print(f"  [PRIORITY] Hook A executed (priority 10)")
        return ctx

    async def hook_b(ctx: HookContext) -> HookContext:
        ctx.metadata.setdefault("execution_order", []).append("B")
        print(f"  [PRIORITY] Hook B executed (priority 50)")
        return ctx

    async def hook_c(ctx: HookContext) -> HookContext:
        ctx.metadata.setdefault("execution_order", []).append("C")
        print(f"  [PRIORITY] Hook C executed (priority 100)")
        return ctx

    # Register in random order but with specific priorities
    hooks.register(HookType.PRE_PROCESS, hook_c, priority=100, name="hook_c")
    hooks.register(HookType.PRE_PROCESS, hook_a, priority=10, name="hook_a")
    hooks.register(HookType.PRE_PROCESS, hook_b, priority=50, name="hook_b")

    print("\nHooks will execute in priority order (10 -> 50 -> 100):")

    # Execute hooks manually to show order
    ctx = HookContext(
        hook_type=HookType.PRE_PROCESS,
        task="Test task",
        context="Test context",
    )

    result = await hooks.execute(HookType.PRE_PROCESS, ctx)
    print(f"\nExecution order: {result.final_context.metadata.get('execution_order')}")


async def modifying_hooks_example() -> None:
    """
    Hooks that modify task and result.

    Shows how hooks can transform data in the pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 4: Data-Modifying Hooks")
    print("=" * 60)

    hooks = HooksManager(name="modifying_example")

    # Pre-process hook that enhances the task
    async def enhance_task(ctx: HookContext) -> HookContext:
        enhanced = f"[ENHANCED] {ctx.task} - Include specific examples."
        print(f"  [ENHANCE] Original task: {ctx.task[:40]}...")
        print(f"  [ENHANCE] Enhanced task: {enhanced[:40]}...")
        return ctx.with_updates(task=enhanced)

    # Post-process hook that adds metadata
    async def add_metadata(ctx: HookContext) -> HookContext:
        if ctx.result:
            ctx.metadata["processed"] = True
            ctx.metadata["word_count"] = len(str(ctx.result).split())
            print(f"  [METADATA] Added word_count: {ctx.metadata['word_count']}")
        return ctx

    hooks.register(HookType.PRE_PROCESS, enhance_task, priority=10)
    hooks.register(HookType.POST_PROCESS, add_metadata, priority=10)

    async with ContextFlow(hooks_manager=hooks) as cf:
        result = await cf.process(
            task="What are the UI requirements?",
            context=SAMPLE_DOCUMENT,
        )

    print(f"\nFinal result: {result.answer[:200]}...")


async def utility_hooks_example() -> None:
    """
    Using built-in utility hooks.

    Shows the pre-built hook utilities.
    """
    print("\n" + "=" * 60)
    print("Example 5: Built-in Utility Hooks")
    print("=" * 60)

    hooks = HooksManager(name="utility_example")

    # Use built-in logging hook
    logging_hook = create_logging_hook(log_level="info", include_context=True)
    hooks.register(HookType.PRE_PROCESS, logging_hook, name="logging")

    # Use built-in timing hook
    timing_start = create_timing_hook(metric_name="process_time")
    timing_end = create_timing_hook(metric_name="process_time")
    hooks.register(HookType.PRE_PROCESS, timing_start, priority=1, name="timing_start")
    hooks.register(HookType.POST_PROCESS, timing_end, priority=1, name="timing_end")

    # Compose multiple hooks into one
    async def hook_1(ctx: HookContext) -> HookContext:
        ctx.metadata["step1"] = True
        return ctx

    async def hook_2(ctx: HookContext) -> HookContext:
        ctx.metadata["step2"] = True
        return ctx

    composed = compose_hooks(hook_1, hook_2)
    hooks.register(HookType.PRE_PROCESS, composed, name="composed")

    print("\nUtility hooks registered:")
    for h in hooks.get_hooks(HookType.PRE_PROCESS):
        print(f"  - {h.name} (priority {h.priority})")

    async with ContextFlow(hooks_manager=hooks) as cf:
        result = await cf.process(
            task="What technology stack is mentioned?",
            context=SAMPLE_DOCUMENT,
        )

    print(f"\nResult: {result.answer[:200]}...")


async def enable_disable_hooks_example() -> None:
    """
    Enabling and disabling hooks dynamically.

    Shows how to control hook execution at runtime.
    """
    print("\n" + "=" * 60)
    print("Example 6: Enable/Disable Hooks")
    print("=" * 60)

    hooks = HooksManager(name="toggle_example")

    async def verbose_hook(ctx: HookContext) -> HookContext:
        print(f"  [VERBOSE] Detailed logging: task={ctx.task[:30]}...")
        return ctx

    hooks.register(HookType.PRE_PROCESS, verbose_hook, name="verbose_logging")

    print("\n--- With verbose hook enabled ---")
    ctx = HookContext(hook_type=HookType.PRE_PROCESS, task="Test task", context="...")
    await hooks.execute(HookType.PRE_PROCESS, ctx)

    # Disable the hook
    hooks.disable_hook(HookType.PRE_PROCESS, "verbose_logging")
    print("\n--- With verbose hook disabled ---")
    await hooks.execute(HookType.PRE_PROCESS, ctx)
    print("  (No output - hook is disabled)")

    # Re-enable
    hooks.enable_hook(HookType.PRE_PROCESS, "verbose_logging")
    print("\n--- With verbose hook re-enabled ---")
    await hooks.execute(HookType.PRE_PROCESS, ctx)

    # Get stats
    print(f"\nHook Manager Stats: {hooks.stats}")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Lifecycle Hooks Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples.")
        return

    try:
        await basic_hooks_example()
        await decorator_hooks_example()
        await priority_hooks_example()
        await modifying_hooks_example()
        await utility_hooks_example()
        await enable_disable_hooks_example()

        print("\n" + "=" * 60)
        print("All hooks examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise
    finally:
        # Clean up global state
        reset_global_hooks_manager()


if __name__ == "__main__":
    asyncio.run(main())
