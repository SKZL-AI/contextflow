"""
RLM Strategy - Recursive Language Model.

Based on MIT CSAIL paper for processing 10M+ tokens.

Key concepts:
- REPL Environment: LLM can execute code to query context
- llm_query(): Recursive sub-calls for nested questions
- FINAL(): Signal completion and return result
- Background Verification: Verify while processing (Boris Step 12)

Two modes:
- RLM_BASIC: For 50K-100K tokens, simpler recursion
- RLM_FULL: For >100K tokens, full recursive capabilities
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextflow.core.types import Message
from contextflow.strategies.base import (
    BaseStrategy,
    CostEstimate,
    StrategyResult,
    StrategyType,
    VerificationResult,
)
from contextflow.strategies.verification import (
    VerificationProtocol,
)
from contextflow.strategies.verification import (
    VerificationResult as DetailedVerificationResult,
)
from contextflow.utils.errors import (
    ProviderError,
    RLMCodeExecutionError,
    RLMMaxIterationsError,
    StrategyExecutionError,
)
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider


logger = get_logger(__name__)


# =============================================================================
# RLM Enums
# =============================================================================


class RLMState(Enum):
    """RLM execution states."""

    INITIALIZING = "initializing"
    PROCESSING = "processing"
    AWAITING_CODE = "awaiting_code"
    EXECUTING_CODE = "executing_code"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class RLMMode(str, Enum):
    """RLM execution modes."""

    BASIC = "basic"  # 50K-100K tokens, simpler recursion
    FULL = "full"  # >100K tokens, full recursive capabilities
    AUTO = "auto"  # Automatically determine based on context size


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class REPLVariable:
    """Variable in REPL environment."""

    name: str
    value: Any
    var_type: str
    created_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        """String representation with truncated value."""
        value_str = str(self.value)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        return f"REPLVariable({self.name}: {self.var_type} = {value_str})"


@dataclass
class CodeExecutionResult:
    """Result from code execution in REPL."""

    success: bool
    output: str
    error: str | None = None
    execution_time: float = 0.0
    variables_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "output": self.output[:1000] if len(self.output) > 1000 else self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "variables_created": self.variables_created,
        }


@dataclass
class RLMIteration:
    """Single iteration of the RLM loop."""

    iteration: int
    state: RLMState
    llm_response: str
    code_blocks: list[str] = field(default_factory=list)
    execution_results: list[CodeExecutionResult] = field(default_factory=list)
    verification_result: DetailedVerificationResult | None = None
    tokens_used: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# System Prompts
# =============================================================================


RLM_SYSTEM_PROMPT = """You are an advanced AI assistant processing a large context using the RLM (Recursive Language Model) protocol.

## REPL Environment
You have access to a Python REPL environment with the following capabilities:

### Available Variables:
- `context`: The full context/document content (string)
- `task`: The task/question to answer (string)
- `constraints`: List of constraints to satisfy (list of strings)

### Available Functions:
- `llm_query(prompt: str) -> str`: Make a recursive LLM call for sub-questions
  Example: `result = llm_query("What is the summary of section 2?")`

- `search_context(query: str) -> list[str]`: Search context for relevant passages
  Example: `matches = search_context("financial data")`

- `count_tokens(text: str) -> int`: Count tokens in text
  Example: `token_count = count_tokens(context)`

- `chunk_context(chunk_size: int = 10000) -> list[str]`: Split context into chunks
  Example: `chunks = chunk_context(5000)`

### Code Execution Rules:
1. Write Python code in ```python code blocks
2. Code will be executed and output returned to you
3. Use `print()` to output results you want to see
4. Variables persist across code blocks
5. Keep code simple and focused

## Completion Signal
When you have the final answer, use the FINAL() signal:

```python
FINAL('''
Your complete, final answer here.
This should directly address the original task.
Include all relevant information.
''')
```

## Example Workflow:

### Iteration 1 (Analysis):
```python
# Analyze context size
print(f"Context length: {len(context)} characters")
print(f"Estimated tokens: {count_tokens(context)}")

# Search for relevant sections
relevant = search_context("main topic")
print(f"Found {len(relevant)} relevant passages")
```

### Iteration 2 (Processing):
```python
# Process in chunks if large
chunks = chunk_context(10000)
summaries = []
for i, chunk in enumerate(chunks[:5]):
    summary = llm_query(f"Summarize this section:\n{chunk}")
    summaries.append(summary)
    print(f"Chunk {i+1} summary: {summary[:100]}...")
```

### Iteration 3 (Synthesis):
```python
# Synthesize final answer
all_summaries = "\\n".join(summaries)
FINAL(f'''
Based on my analysis of the context:

{all_summaries}

Key conclusions:
1. ...
2. ...
''')
```

## Important Guidelines:
1. Start by analyzing the context to understand its structure
2. Use llm_query() for complex sub-questions that need LLM reasoning
3. Use search_context() to find specific information efficiently
4. Chunk large contexts for manageable processing
5. Always end with FINAL() when you have the complete answer
6. If stuck, explain your reasoning and what additional information you need"""


RLM_BASIC_SYSTEM_PROMPT = """You are an AI assistant processing a moderately large context (50K-100K tokens).

## Simplified REPL
You have a Python REPL with these functions:
- `search_context(query)`: Find relevant passages
- `llm_query(prompt)`: Ask a sub-question (use sparingly)
- `FINAL(answer)`: Return your final answer

## Quick Start:
1. Analyze what the task requires
2. Search for relevant information: `search_context("key terms")`
3. If needed, clarify with sub-queries: `llm_query("specific question")`
4. Provide final answer: `FINAL("your answer")`

Write Python code in ```python blocks. Always end with FINAL()."""


# =============================================================================
# REPL Environment
# =============================================================================


class REPLEnvironment:
    """
    REPL Environment for RLM code execution.

    Provides:
    - Variable storage and retrieval
    - Function registration (like llm_query)
    - Safe code execution
    - Async support

    The REPL creates a sandboxed execution environment where the LLM
    can write and execute Python code to explore the context.
    """

    def __init__(
        self,
        max_output_length: int = 50000,
        execution_timeout: float = 30.0,
    ) -> None:
        """
        Initialize REPL Environment.

        Args:
            max_output_length: Maximum length of captured output
            execution_timeout: Timeout for code execution in seconds
        """
        self._variables: dict[str, REPLVariable] = {}
        self._functions: dict[str, Callable[..., Any]] = {}
        self._async_functions: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._output_buffer: list[str] = []
        self._max_output_length = max_output_length
        self._execution_timeout = execution_timeout
        self._execution_count = 0

        logger.debug(
            "REPLEnvironment initialized",
            max_output_length=max_output_length,
            execution_timeout=execution_timeout,
        )

    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the environment.

        Args:
            name: Variable name
            value: Variable value
        """
        var_type = type(value).__name__
        self._variables[name] = REPLVariable(
            name=name,
            value=value,
            var_type=var_type,
        )
        logger.debug("Variable set", name=name, type=var_type)

    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the environment.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not found in REPL environment")
        return self._variables[name].value

    def has_variable(self, name: str) -> bool:
        """
        Check if variable exists.

        Args:
            name: Variable name

        Returns:
            True if variable exists
        """
        return name in self._variables

    def list_variables(self) -> list[str]:
        """
        List all variable names.

        Returns:
            List of variable names
        """
        return list(self._variables.keys())

    def set_function(self, name: str, func: Callable[..., Any]) -> None:
        """
        Register a synchronous function.

        Args:
            name: Function name for use in REPL
            func: Function to register
        """
        self._functions[name] = func
        logger.debug("Function registered", name=name)

    def set_async_function(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register an asynchronous function.

        Args:
            name: Function name for use in REPL
            func: Async function to register
        """
        self._async_functions[name] = func
        logger.debug("Async function registered", name=name)

    async def execute_async(self, code: str) -> CodeExecutionResult:
        """
        Execute code asynchronously in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            CodeExecutionResult with output and status
        """
        start_time = time.time()
        self._execution_count += 1
        self._output_buffer = []
        variables_created: list[str] = []

        logger.debug(
            "Executing code",
            execution_count=self._execution_count,
            code_length=len(code),
        )

        # Build execution namespace
        namespace = self._build_namespace()
        original_vars = set(namespace.keys())

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_code_internal(code, namespace),
                timeout=self._execution_timeout,
            )

            # Capture new variables
            for var_name in namespace:
                if var_name not in original_vars and not var_name.startswith("_"):
                    self.set_variable(var_name, namespace[var_name])
                    variables_created.append(var_name)

            execution_time = time.time() - start_time
            output = self._get_output()

            logger.debug(
                "Code executed successfully",
                execution_time=round(execution_time, 3),
                output_length=len(output),
                variables_created=variables_created,
            )

            return CodeExecutionResult(
                success=True,
                output=output if output else str(result) if result else "(no output)",
                execution_time=execution_time,
                variables_created=variables_created,
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Code execution timed out after {self._execution_timeout}s"
            logger.warning("Code execution timeout", timeout=self._execution_timeout)
            return CodeExecutionResult(
                success=False,
                output="",
                error=error_msg,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            logger.warning(
                "Code execution failed",
                error=error_msg,
                execution_time=round(execution_time, 3),
            )
            return CodeExecutionResult(
                success=False,
                output=self._get_output(),
                error=f"{error_msg}\n{tb}",
                execution_time=execution_time,
            )

    async def _execute_code_internal(
        self,
        code: str,
        namespace: dict[str, Any],
    ) -> Any:
        """
        Internal code execution with async support.

        Args:
            code: Code to execute
            namespace: Execution namespace

        Returns:
            Execution result (if any)
        """
        # Compile code
        try:
            compiled = compile(code, "<rlm_repl>", "exec")
        except SyntaxError as e:
            raise RLMCodeExecutionError(
                code=code,
                error=f"Syntax error: {e}",
            )

        # Execute in namespace
        # Note: For production, this should be sandboxed more securely
        exec(compiled, namespace)

        # Return last expression result if available
        return namespace.get("_result")

    def _build_namespace(self) -> dict[str, Any]:
        """
        Build the execution namespace with all variables and functions.

        Returns:
            Namespace dictionary
        """
        namespace: dict[str, Any] = {
            "__builtins__": self._get_safe_builtins(),
            "print": self._capture_print,
        }

        # Add variables
        for name, var in self._variables.items():
            namespace[name] = var.value

        # Add sync functions
        for name, func in self._functions.items():
            namespace[name] = func

        # Add async functions with sync wrapper
        for name, async_func in self._async_functions.items():
            namespace[name] = self._create_sync_wrapper(async_func)

        return namespace

    def _get_safe_builtins(self) -> dict[str, Any]:
        """
        Get safe builtins for sandboxed execution.

        Returns:
            Dictionary of safe builtin functions
        """
        # Allow only safe built-in functions
        safe_builtins = {
            # Type constructors
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            # Iteration
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            # Math
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            # String
            "chr": chr,
            "ord": ord,
            # Type checking
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            # Other safe functions
            "any": any,
            "all": all,
            "repr": repr,
            "format": format,
            "slice": slice,
            # Exceptions (for error handling)
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            # None and booleans
            "None": None,
            "True": True,
            "False": False,
        }
        return safe_builtins

    def _capture_print(self, *args: Any, **kwargs: Any) -> None:
        """
        Capture print output to buffer.

        Args:
            *args: Print arguments
            **kwargs: Print keyword arguments (sep, end)
        """
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        output = sep.join(str(arg) for arg in args) + end
        self._output_buffer.append(output)

    def _get_output(self) -> str:
        """
        Get and truncate captured output.

        Returns:
            Captured output string
        """
        output = "".join(self._output_buffer)
        if len(output) > self._max_output_length:
            output = output[: self._max_output_length] + "\n...(output truncated)"
        return output

    def _create_sync_wrapper(
        self,
        async_func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Callable[..., Any]:
        """
        Create a synchronous wrapper for an async function.

        This allows async functions to be called from synchronous code
        in the REPL by running them in the current event loop.

        Args:
            async_func: Async function to wrap

        Returns:
            Synchronous wrapper function
        """

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                asyncio.get_running_loop()
                # We're in an async context, use thread pool executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(asyncio.run, async_func(*args, **kwargs)).result(
                        timeout=self._execution_timeout
                    )
            except RuntimeError:
                # No running loop, just run directly
                return asyncio.run(async_func(*args, **kwargs))

        return sync_wrapper

    def clear(self) -> None:
        """Clear all variables and reset state."""
        self._variables.clear()
        self._output_buffer.clear()
        self._execution_count = 0
        logger.debug("REPL environment cleared")

    def get_state_summary(self) -> dict[str, Any]:
        """
        Get summary of current REPL state.

        Returns:
            Dictionary with state information
        """
        return {
            "variables": list(self._variables.keys()),
            "functions": list(self._functions.keys()),
            "async_functions": list(self._async_functions.keys()),
            "execution_count": self._execution_count,
        }


# =============================================================================
# RLM Strategy
# =============================================================================


class RLMStrategy(BaseStrategy):
    """
    RLM Strategy for very large contexts (>100K tokens).

    Based on MIT CSAIL "Recursive Language Models" paper.

    Core Algorithm:
    1. Initialize REPL with context as variable
    2. Define llm_query() for sub-calls
    3. Iterative loop:
       - Get LLM response
       - Extract and execute code blocks
       - Feed output back to LLM
       - Check for FINAL() signal
    4. Background verification during processing (Boris Step 12)

    Optimal for:
    - Context >100K tokens (up to 10M+)
    - Complex multi-step analysis
    - Tasks requiring code-based context exploration

    Example:
        strategy = RLMStrategy(provider, mode="full")
        result = await strategy.execute(
            task="Summarize all financial data",
            context=large_document,
            constraints=["Include revenue figures", "Mention risks"],
        )
    """

    def __init__(
        self,
        provider: BaseProvider,
        mode: str = "auto",
        max_iterations: int = 20,
        max_recursion_depth: int = 5,
        enable_verification: bool = True,
        enable_background_verification: bool = True,
        timeout: float = 300.0,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        verification_threshold: float = 0.8,
    ) -> None:
        """
        Initialize RLM Strategy.

        Args:
            provider: LLM provider for completions
            mode: Processing mode ("basic", "full", or "auto")
            max_iterations: Max main loop iterations
            max_recursion_depth: Max depth for llm_query calls
            enable_verification: Whether to verify final output
            enable_background_verification: Verify during processing (Boris Step 12)
            timeout: Max execution time in seconds
            model: Model to use (overrides provider default)
            temperature: Sampling temperature
            max_output_tokens: Max tokens in generated output
            verification_threshold: Minimum score to pass verification

        Raises:
            ValueError: If mode is invalid
        """
        super().__init__(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verification_threshold=verification_threshold,
        )

        # Validate mode
        valid_modes = ["basic", "full", "auto"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        self._mode = mode
        self._max_iterations = max_iterations
        self._max_recursion_depth = max_recursion_depth
        self._enable_verification = enable_verification
        self._enable_background_verification = enable_background_verification
        self._timeout = timeout

        # Internal state
        self._current_depth = 0
        self._iterations: list[RLMIteration] = []
        self._background_verifications: list[DetailedVerificationResult] = []
        self._state = RLMState.INITIALIZING
        self._total_tokens = 0
        self._total_cost = 0.0

        # Verification protocol
        self._verifier: VerificationProtocol | None = None
        if enable_verification:
            self._verifier = VerificationProtocol(
                provider=provider,
                min_confidence=verification_threshold,
            )

        logger.info(
            "RLMStrategy initialized",
            mode=mode,
            max_iterations=max_iterations,
            max_recursion_depth=max_recursion_depth,
            enable_verification=enable_verification,
            enable_background_verification=enable_background_verification,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "rlm"

    @property
    def strategy_type(self) -> StrategyType:
        """Strategy type enum value."""
        effective_mode = self._determine_mode(100000)  # Default estimate
        if effective_mode == "basic":
            return StrategyType.RLM_BASIC
        return StrategyType.RLM_FULL

    @property
    def max_tokens(self) -> int:
        """Maximum tokens this strategy can handle."""
        return 10_000_000  # 10M tokens

    @property
    def min_tokens(self) -> int:
        """Minimum tokens for this strategy."""
        return 50_000

    @property
    def current_state(self) -> RLMState:
        """Current execution state."""
        return self._state

    @property
    def iterations(self) -> list[RLMIteration]:
        """List of completed iterations."""
        return self._iterations.copy()

    async def execute(
        self,
        task: str,
        context: str,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> StrategyResult:
        """
        Execute RLM strategy on the given task and context.

        Args:
            task: The task/question to process
            context: The context/documents to analyze
            constraints: Optional constraints for verification
            **kwargs: Additional parameters

        Returns:
            StrategyResult with answer and metadata

        Raises:
            StrategyExecutionError: If execution fails
            RLMMaxIterationsError: If max iterations reached without FINAL
        """
        start_time = time.time()
        self._state = RLMState.INITIALIZING
        self._iterations = []
        self._background_verifications = []
        self._total_tokens = 0
        self._total_cost = 0.0

        # Estimate token count
        context_tokens = self._provider.count_tokens(context)

        self._logger.log_start(
            token_count=context_tokens,
            mode=self._mode,
            max_iterations=self._max_iterations,
        )

        # Determine effective mode
        effective_mode = self._determine_mode(context_tokens)
        logger.info(
            "RLM mode determined",
            effective_mode=effective_mode,
            context_tokens=context_tokens,
        )

        try:
            # Create and initialize REPL
            repl = REPLEnvironment()
            self._initialize_repl(repl, task, context, constraints)

            # Execute main RLM loop
            self._state = RLMState.PROCESSING
            result = await asyncio.wait_for(
                self._execute_rlm_loop(
                    task=task,
                    context=context,
                    repl=repl,
                    constraints=constraints,
                    mode=effective_mode,
                ),
                timeout=self._timeout,
            )

            # Final verification
            if self._enable_verification and self._verifier:
                self._state = RLMState.VERIFYING
                verification = await self.verify(
                    task=task,
                    output=result.answer,
                    constraints=constraints,
                )
                result.verification_passed = verification.passed
                result.verification_score = verification.confidence

            self._state = RLMState.COMPLETED
            result.execution_time = time.time() - start_time

            self._logger.log_complete(
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                duration_seconds=result.execution_time,
                iterations=len(self._iterations),
                verification_passed=result.verification_passed,
            )

            return result

        except TimeoutError:
            self._state = RLMState.FAILED
            execution_time = time.time() - start_time
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"RLM execution timed out after {self._timeout}s",
                details={
                    "iterations_completed": len(self._iterations),
                    "execution_time": execution_time,
                },
            )

        except RLMMaxIterationsError:
            self._state = RLMState.FAILED
            raise

        except Exception as e:
            self._state = RLMState.FAILED
            logger.error("RLM execution failed", error=str(e), exc_info=True)
            raise StrategyExecutionError(
                strategy=self.name,
                message=f"RLM execution failed: {str(e)}",
                cause=e,
            )

    def _initialize_repl(
        self,
        repl: REPLEnvironment,
        task: str,
        context: str,
        constraints: list[str] | None,
    ) -> None:
        """
        Initialize REPL with variables and functions.

        Args:
            repl: REPL environment to initialize
            task: Task string
            context: Context string
            constraints: Optional constraints
        """
        # Set core variables
        repl.set_variable("context", context)
        repl.set_variable("task", task)
        repl.set_variable("constraints", constraints or [])

        # Register FINAL function
        final_result: list[str] = []

        def final_func(answer: str) -> str:
            """Signal completion with final answer."""
            final_result.clear()
            final_result.append(answer)
            return f"[FINAL ANSWER RECORDED: {len(answer)} characters]"

        repl.set_function("FINAL", final_func)
        repl.set_variable("_final_result", final_result)

        # Register helper functions
        repl.set_function("count_tokens", self._provider.count_tokens)
        repl.set_function("search_context", self._create_search_function(context))
        repl.set_function("chunk_context", self._create_chunk_function(context))

        # Register llm_query as async function
        repl.set_async_function(
            "llm_query",
            self._create_llm_query_function(self._provider, depth=0),
        )

        logger.debug(
            "REPL initialized",
            variables=repl.list_variables(),
            state=repl.get_state_summary(),
        )

    async def _execute_rlm_loop(
        self,
        task: str,
        context: str,
        repl: REPLEnvironment,
        constraints: list[str] | None,
        mode: str,
    ) -> StrategyResult:
        """
        Main RLM execution loop.

        Algorithm:
        1. Send task + context to LLM with REPL instructions
        2. Extract code blocks from response
        3. Execute code in REPL
        4. Feed results back to LLM
        5. Repeat until FINAL() or max_iterations

        Args:
            task: Task to process
            context: Context to analyze
            repl: REPL environment
            constraints: Optional constraints
            mode: Execution mode ("basic" or "full")

        Returns:
            StrategyResult with final answer

        Raises:
            RLMMaxIterationsError: If max iterations reached
        """
        # Select system prompt based on mode
        system_prompt = RLM_BASIC_SYSTEM_PROMPT if mode == "basic" else RLM_SYSTEM_PROMPT

        # Build conversation history
        messages: list[Message] = []

        # Initial user message
        initial_content = self._build_initial_message(task, context, constraints)
        messages.append(Message(role="user", content=initial_content))

        final_answer: str | None = None
        background_verify_task: asyncio.Task[DetailedVerificationResult] | None = None

        for iteration in range(1, self._max_iterations + 1):
            iteration_start = time.time()
            self._state = RLMState.AWAITING_CODE

            logger.info(
                "RLM iteration starting",
                iteration=iteration,
                max_iterations=self._max_iterations,
            )

            # Get LLM response
            try:
                response = await self._call_llm(messages, system_prompt)
            except Exception as e:
                logger.error(
                    "LLM call failed",
                    iteration=iteration,
                    error=str(e),
                )
                raise

            llm_response = response.content
            self._total_tokens += response.tokens_used
            self._total_cost += response.cost_usd

            # Create iteration record
            rlm_iteration = RLMIteration(
                iteration=iteration,
                state=RLMState.EXECUTING_CODE,
                llm_response=llm_response,
                tokens_used=response.tokens_used,
            )

            # Check for FINAL signal in response text
            is_final, extracted_answer = self._check_final_signal(llm_response)
            if is_final and extracted_answer:
                final_answer = extracted_answer
                rlm_iteration.state = RLMState.COMPLETED
                rlm_iteration.duration_seconds = time.time() - iteration_start
                self._iterations.append(rlm_iteration)
                break

            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(llm_response)
            rlm_iteration.code_blocks = code_blocks

            if code_blocks:
                self._state = RLMState.EXECUTING_CODE
                execution_outputs: list[str] = []

                for i, code in enumerate(code_blocks):
                    logger.debug(
                        "Executing code block",
                        iteration=iteration,
                        block=i + 1,
                        code_length=len(code),
                    )

                    result = await repl.execute_async(code)
                    rlm_iteration.execution_results.append(result)

                    if result.success:
                        execution_outputs.append(f"[Code Block {i + 1}]\n{result.output}")
                    else:
                        execution_outputs.append(f"[Code Block {i + 1} ERROR]\n{result.error}")

                    # Check if FINAL was called in code
                    final_result_var = repl.get_variable("_final_result")
                    if final_result_var:
                        final_answer = final_result_var[0]
                        rlm_iteration.state = RLMState.COMPLETED
                        break

                # Build feedback message for LLM
                if not final_answer:
                    feedback = "\n\n".join(execution_outputs)
                    messages.append(Message(role="assistant", content=llm_response))
                    messages.append(
                        Message(
                            role="user",
                            content=f"Code execution results:\n\n{feedback}\n\nContinue with next step or provide FINAL answer.",
                        )
                    )
            else:
                # No code blocks - ask LLM to provide code or FINAL
                messages.append(Message(role="assistant", content=llm_response))
                messages.append(
                    Message(
                        role="user",
                        content="Please provide Python code to analyze the context, or use FINAL() if you have your answer.",
                    )
                )

            rlm_iteration.duration_seconds = time.time() - iteration_start
            self._iterations.append(rlm_iteration)

            # Background verification (Boris Step 12)
            if self._enable_background_verification and final_answer is None:
                # Start background verification every few iterations
                if iteration % 3 == 0:
                    partial_result = self._extract_partial_result(llm_response)
                    if partial_result:
                        background_verify_task = asyncio.create_task(
                            self._background_verify(task, partial_result, iteration)
                        )

            # Check for final answer from code execution
            if final_answer:
                break

        # Wait for any pending background verification
        if background_verify_task:
            try:
                bg_result = await asyncio.wait_for(background_verify_task, timeout=10.0)
                if bg_result:
                    self._background_verifications.append(bg_result)
            except TimeoutError:
                pass

        # Check if we got a final answer
        if final_answer is None:
            raise RLMMaxIterationsError(
                iterations=len(self._iterations),
                max_iterations=self._max_iterations,
                details={
                    "last_response": (llm_response[:500] if llm_response else "None"),
                },
            )

        return StrategyResult(
            answer=final_answer,
            strategy_used=StrategyType.RLM_FULL if mode == "full" else StrategyType.RLM_BASIC,
            iterations=len(self._iterations),
            token_usage={
                "input": self._total_tokens // 2,  # Rough estimate
                "output": self._total_tokens // 2,
                "total": self._total_tokens,
            },
            execution_time=0.0,  # Will be set by caller
            verification_passed=False,  # Will be updated after verification
            verification_score=0.0,
            metadata={
                "mode": mode,
                "iterations_detail": [
                    {
                        "iteration": it.iteration,
                        "state": it.state.value,
                        "code_blocks": len(it.code_blocks),
                        "tokens": it.tokens_used,
                        "duration": it.duration_seconds,
                    }
                    for it in self._iterations
                ],
                "background_verifications": len(self._background_verifications),
                "cost_usd": self._total_cost,
            },
        )

    def _build_initial_message(
        self,
        task: str,
        context: str,
        constraints: list[str] | None,
    ) -> str:
        """
        Build the initial message for the LLM.

        Args:
            task: Task to process
            context: Context to analyze
            constraints: Optional constraints

        Returns:
            Formatted initial message
        """
        # Truncate context for initial message if very large
        context_preview = context[:50000] if len(context) > 50000 else context
        truncated_note = (
            f"\n\n[Note: Context truncated in message. Full context ({len(context)} chars) available via `context` variable.]"
            if len(context) > 50000
            else ""
        )

        constraints_section = ""
        if constraints:
            constraints_section = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

        return f"""TASK: {task}{constraints_section}

CONTEXT:
{context_preview}{truncated_note}

Please analyze this context to complete the task. Use Python code to explore the data and call FINAL() when you have your answer."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[Message],
        system_prompt: str,
    ) -> Any:
        """
        Call the LLM with retry logic.

        Args:
            messages: Conversation messages
            system_prompt: System prompt to use

        Returns:
            CompletionResponse from provider
        """
        return await self._provider.complete(
            messages=messages,
            system=system_prompt,
            model=self._model,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )

    def _create_llm_query_function(
        self,
        provider: BaseProvider,
        depth: int = 0,
    ) -> Callable[[str], Coroutine[Any, Any, str]]:
        """
        Create llm_query() function for recursive sub-calls.

        Args:
            provider: LLM provider
            depth: Current recursion depth

        Returns:
            Async function for making LLM queries
        """

        async def llm_query(prompt: str) -> str:
            """
            Make a recursive LLM query.

            Args:
                prompt: The question/prompt to send

            Returns:
                LLM response text
            """
            if depth >= self._max_recursion_depth:
                logger.warning(
                    "Recursion limit reached",
                    depth=depth,
                    max_depth=self._max_recursion_depth,
                )
                return f"[Recursion limit reached at depth {depth}]"

            logger.debug("llm_query called", depth=depth, prompt_length=len(prompt))

            try:
                response = await provider.complete(
                    messages=[Message(role="user", content=prompt)],
                    max_tokens=2048,
                    temperature=0.5,
                )

                self._total_tokens += response.tokens_used
                self._total_cost += response.cost_usd

                return response.content

            except Exception as e:
                logger.error(
                    "llm_query failed",
                    depth=depth,
                    error=str(e),
                )
                return f"[Error in llm_query: {str(e)}]"

        return llm_query

    def _create_search_function(
        self,
        context: str,
    ) -> Callable[[str], list[str]]:
        """
        Create search_context function for finding relevant passages.

        Args:
            context: The full context to search

        Returns:
            Function that searches context
        """

        def search_context(query: str, max_results: int = 5) -> list[str]:
            """
            Search context for relevant passages.

            Args:
                query: Search query
                max_results: Maximum results to return

            Returns:
                List of relevant passages
            """
            query_lower = query.lower()
            query_words = set(query_lower.split())

            # Split context into paragraphs
            paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]

            # Score paragraphs by relevance
            scored: list[tuple[float, str]] = []
            for para in paragraphs:
                para_lower = para.lower()
                # Simple scoring: count matching words
                para_words = set(para_lower.split())
                overlap = len(query_words & para_words)
                # Bonus for exact phrase match
                if query_lower in para_lower:
                    overlap += 5
                if overlap > 0:
                    scored.append((overlap, para))

            # Sort by score and return top results
            scored.sort(key=lambda x: x[0], reverse=True)
            return [para for _, para in scored[:max_results]]

        return search_context

    def _create_chunk_function(
        self,
        context: str,
    ) -> Callable[[int], list[str]]:
        """
        Create chunk_context function for splitting context.

        Args:
            context: The full context

        Returns:
            Function that chunks context
        """

        def chunk_context(chunk_size: int = 10000) -> list[str]:
            """
            Split context into chunks.

            Args:
                chunk_size: Target size per chunk in characters

            Returns:
                List of context chunks
            """
            chunks: list[str] = []
            current_pos = 0

            while current_pos < len(context):
                # Find chunk end
                end_pos = min(current_pos + chunk_size, len(context))

                # Try to break at paragraph boundary
                if end_pos < len(context):
                    # Look for paragraph break
                    para_break = context.rfind("\n\n", current_pos, end_pos)
                    if para_break > current_pos + chunk_size // 2:
                        end_pos = para_break + 2

                chunk = context[current_pos:end_pos].strip()
                if chunk:
                    chunks.append(chunk)

                current_pos = end_pos

            return chunks

        return chunk_context

    def _extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract Python code blocks from LLM response.

        Args:
            response: LLM response text

        Returns:
            List of code block contents
        """
        # Match ```python ... ``` blocks
        pattern = r"```python\s*([\s\S]*?)```"
        matches = re.findall(pattern, response, re.IGNORECASE)

        # Also try ``` without language specifier
        if not matches:
            pattern = r"```\s*([\s\S]*?)```"
            matches = re.findall(pattern, response)
            # Filter out non-Python-looking code
            matches = [m for m in matches if not m.strip().startswith(("{", "<", "#!"))]

        return [m.strip() for m in matches if m.strip()]

    def _check_final_signal(self, response: str) -> tuple[bool, str | None]:
        """
        Check if response contains FINAL() signal and extract result.

        Args:
            response: LLM response text

        Returns:
            Tuple of (is_final, extracted_answer)
        """
        # Check for FINAL() call pattern
        patterns = [
            r"FINAL\s*\(\s*['\"]+([\s\S]*?)['\"]+\s*\)",  # FINAL("...")
            r"FINAL\s*\(\s*f?['\"]+([\s\S]*?)['\"]+\s*\)",  # FINAL(f"...")
            r"FINAL\s*\(\s*'''([\s\S]*?)'''\s*\)",  # FINAL('''...''')
            r'FINAL\s*\(\s*"""([\s\S]*?)"""\s*\)',  # FINAL("""...""")
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return True, match.group(1).strip()

        return False, None

    def _extract_partial_result(self, response: str) -> str | None:
        """
        Extract partial result from response for background verification.

        Args:
            response: LLM response text

        Returns:
            Partial result string if found
        """
        # Look for summary/conclusion sections
        patterns = [
            r"(?:Summary|Conclusion|Finding|Result)[:]\s*([\s\S]{100,500})",
            r"(?:Based on|According to|Analysis shows)[\s\S]{50,500}",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(0)[:500]

        return None

    async def _background_verify(
        self,
        task: str,
        partial_result: str,
        iteration: int,
    ) -> DetailedVerificationResult | None:
        """
        Background verification during processing (Boris Step 12).

        Runs verification in background without blocking main loop.

        Args:
            task: Original task
            partial_result: Partial result to verify
            iteration: Current iteration number

        Returns:
            VerificationResult if verification completes, None otherwise
        """
        if not self._verifier:
            return None

        logger.debug(
            "Starting background verification",
            iteration=iteration,
            result_length=len(partial_result),
        )

        try:
            result = await self._verifier.verify(
                task=task,
                output=partial_result,
            )

            logger.debug(
                "Background verification completed",
                iteration=iteration,
                passed=result.passed,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.warning(
                "Background verification failed",
                iteration=iteration,
                error=str(e),
            )
            return None

    async def verify(
        self,
        task: str,
        output: str,
        constraints: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify that output meets task requirements.

        Boris Step 13: Self-verification for quality improvement.

        Args:
            task: Original task description
            output: Generated output to verify
            constraints: Constraints to check against

        Returns:
            VerificationResult with pass/fail and details
        """
        if not self._verifier:
            # Return passing result if verification disabled
            return VerificationResult(
                passed=True,
                confidence=1.0,
                issues=[],
                suggestions=[],
                checks_performed=["verification_disabled"],
            )

        try:
            detailed_result = await self._verifier.verify(
                task=task,
                output=output,
                constraints=constraints,
            )

            return VerificationResult(
                passed=detailed_result.passed,
                confidence=detailed_result.confidence,
                issues=detailed_result.issues,
                suggestions=detailed_result.suggestions,
                checks_performed=[c.check_type.value for c in detailed_result.checks],
                metadata={
                    "overall_score": detailed_result.overall_score,
                    "execution_time": detailed_result.execution_time,
                },
            )

        except Exception as e:
            logger.error("Verification failed", error=str(e))
            return VerificationResult(
                passed=False,
                confidence=0.0,
                issues=[f"Verification error: {str(e)}"],
                suggestions=["Retry verification"],
                checks_performed=[],
            )

    def estimate_cost(
        self,
        context_tokens: int,
        model: str | None = None,
    ) -> CostEstimate:
        """
        Estimate cost for RLM execution.

        Note: RLM cost is harder to predict due to recursive nature.
        Returns conservative estimate based on expected iterations.

        Args:
            context_tokens: Number of tokens in context
            model: Model to estimate for

        Returns:
            CostEstimate with min/max/expected costs
        """
        # Determine mode
        mode = self._determine_mode(context_tokens)

        # Estimate iterations based on context size and mode
        if mode == "basic":
            expected_iterations = 3
            min_iterations = 2
            max_iterations = 6
        else:
            expected_iterations = min(7, max(3, context_tokens // 100000))
            min_iterations = 2
            max_iterations = self._max_iterations

        # Estimate output tokens per iteration
        output_per_iteration = 2000

        # Calculate token estimates
        # Each iteration processes context summary + conversation history
        # Context is passed once, then summaries used
        total_input_min = context_tokens + (min_iterations * 1000)
        total_input_expected = context_tokens + (expected_iterations * 2000)
        total_input_max = context_tokens + (max_iterations * 4000)

        total_output_min = min_iterations * output_per_iteration
        total_output_expected = expected_iterations * output_per_iteration
        total_output_max = max_iterations * output_per_iteration

        # Get model pricing (simplified - real implementation would look up actual prices)
        model_name = model or self._model or self._provider.model
        input_price_per_1k = 0.003  # $3 per 1M tokens (default estimate)
        output_price_per_1k = 0.015  # $15 per 1M tokens (default estimate)

        # Claude pricing overrides
        if "claude" in model_name.lower():
            if "opus" in model_name.lower():
                input_price_per_1k = 0.015
                output_price_per_1k = 0.075
            elif "sonnet" in model_name.lower():
                input_price_per_1k = 0.003
                output_price_per_1k = 0.015
            elif "haiku" in model_name.lower():
                input_price_per_1k = 0.00025
                output_price_per_1k = 0.00125

        # Calculate costs
        min_cost = (total_input_min / 1000) * input_price_per_1k + (
            total_output_min / 1000
        ) * output_price_per_1k
        expected_cost = (total_input_expected / 1000) * input_price_per_1k + (
            total_output_expected / 1000
        ) * output_price_per_1k
        max_cost = (total_input_max / 1000) * input_price_per_1k + (
            total_output_max / 1000
        ) * output_price_per_1k

        return CostEstimate(
            min_cost=round(min_cost, 4),
            max_cost=round(max_cost, 4),
            expected_cost=round(expected_cost, 4),
            model=model_name,
            context_tokens=context_tokens,
            estimated_output_tokens=total_output_expected,
            metadata={
                "mode": mode,
                "expected_iterations": expected_iterations,
                "min_iterations": min_iterations,
                "max_iterations": max_iterations,
            },
        )

    def _determine_mode(self, context_tokens: int) -> str:
        """
        Determine RLM mode based on context size.

        Args:
            context_tokens: Number of tokens in context

        Returns:
            Mode string ("basic" or "full")
        """
        if self._mode == "auto":
            if context_tokens < 100000:
                return "basic"
            return "full"
        return self._mode


# =============================================================================
# Factory Registration
# =============================================================================


def register_rlm_strategy() -> None:
    """Register RLM strategy with the StrategyFactory."""
    from contextflow.strategies.base import StrategyFactory

    StrategyFactory.register("rlm", RLMStrategy)
    StrategyFactory.register("rlm_basic", RLMStrategy)
    StrategyFactory.register("rlm_full", RLMStrategy)


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_rlm(
    provider: BaseProvider,
    task: str,
    context: str,
    constraints: list[str] | None = None,
    mode: str = "auto",
    **kwargs: Any,
) -> StrategyResult:
    """
    Execute RLM strategy with default settings.

    Convenience function for quick RLM execution.

    Args:
        provider: LLM provider
        task: Task to process
        context: Context to analyze
        constraints: Optional constraints
        mode: Execution mode
        **kwargs: Additional RLMStrategy parameters

    Returns:
        StrategyResult with answer
    """
    strategy = RLMStrategy(provider=provider, mode=mode, **kwargs)
    return await strategy.execute(task=task, context=context, constraints=constraints)
