"""
REPL Environment for ContextFlow RLM Strategy.

Provides safe code execution environment for LLM-generated code.

Features:
- Variable storage and retrieval
- Function registration (sync and async)
- Sandboxed execution with restricted builtins
- Output capture
- Timeout handling
- Async support for llm_query() calls
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from contextflow.utils.errors import RLMCodeExecutionError
from contextflow.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ExecutionMode(Enum):
    """Code execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    AUTO = "auto"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class REPLVariable:
    """Variable stored in REPL environment."""

    name: str
    value: Any
    var_type: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    read_count: int = 0
    is_function: bool = False

    def __repr__(self) -> str:
        """String representation with truncated value."""
        value_str = str(self.value)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        return f"REPLVariable({self.name}: {self.var_type} = {value_str})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "var_type": self.var_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "read_count": self.read_count,
            "is_function": self.is_function,
        }


@dataclass
class CodeExecutionResult:
    """Result from code execution."""

    success: bool
    output: str
    return_value: Any = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    execution_time: float = 0.0
    variables_created: list[str] = field(default_factory=list)
    variables_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "success": self.success,
            "output": self.output[:1000] if len(self.output) > 1000 else self.output,
            "return_value": (
                str(self.return_value)[:200] if self.return_value is not None else None
            ),
            "error": self.error,
            "error_type": self.error_type,
            "execution_time": round(self.execution_time, 4),
            "variables_created": self.variables_created,
            "variables_modified": self.variables_modified,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.success:
            return f"[SUCCESS] Output: {self.output[:200]}..."
        return f"[ERROR] {self.error_type}: {self.error}"


# =============================================================================
# Restricted Builtins
# =============================================================================

# Safe builtins for sandboxed execution
SAFE_BUILTINS: set[str] = {
    # Types
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    "frozenset",
    "bytes",
    "bytearray",
    "complex",
    # Functions
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "pow",
    "any",
    "all",
    "isinstance",
    "issubclass",
    "hasattr",
    "getattr",
    "setattr",
    "callable",
    "repr",
    "format",
    "hash",
    "id",
    "type",
    # String methods
    "ord",
    "chr",
    "ascii",
    "bin",
    "hex",
    "oct",
    # Iteration
    "iter",
    "next",
    # Math
    "divmod",
    # Errors
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "StopIteration",
    # Boolean
    "True",
    "False",
    "None",
    # Print (will be captured)
    "print",
}

# Builtins that are explicitly blocked (dangerous)
BLOCKED_BUILTINS: set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "exit",
    "quit",
    "memoryview",
    "delattr",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "object",
    "__build_class__",
}


# =============================================================================
# REPL Environment
# =============================================================================


class REPLEnvironment:
    """
    Safe code execution environment for RLM.

    Provides a sandboxed Python execution environment where LLM-generated
    code can be safely executed. Variables persist across executions,
    and both sync and async functions can be registered.

    Usage:
        repl = REPLEnvironment()

        # Set variables
        repl.set_variable("context", large_text)
        repl.set_variable("task", "Find all API endpoints")

        # Register functions
        repl.set_function("llm_query", llm_query_fn, is_async=True)

        # Execute code
        result = await repl.execute_async('''
        endpoints = []
        for line in context.split("\\n"):
            if "def " in line and "route" in line:
                endpoints.append(line.strip())
        RESULT = endpoints
        ''')

        # Get result
        if result.success:
            endpoints = repl.get_variable("RESULT")

    Attributes:
        sandbox_mode: Whether to restrict builtins for safety
        max_output_length: Maximum characters in output capture
        timeout: Execution timeout in seconds
        allowed_imports: List of allowed module imports (if any)
    """

    def __init__(
        self,
        sandbox_mode: bool = True,
        max_output_length: int = 10000,
        timeout: float = 30.0,
        allowed_imports: list[str] | None = None,
    ) -> None:
        """
        Initialize REPL Environment.

        Args:
            sandbox_mode: Restrict builtins for safety
            max_output_length: Max chars in output capture
            timeout: Execution timeout in seconds
            allowed_imports: List of allowed module imports
        """
        self._variables: dict[str, REPLVariable] = {}
        self._functions: dict[str, Callable[..., Any]] = {}
        self._async_functions: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

        self._sandbox_mode = sandbox_mode
        self._max_output_length = max_output_length
        self._timeout = timeout
        self._allowed_imports = set(allowed_imports or [])

        # Internal state
        self._output_buffer: list[str] = []
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_error: str | None = None

        logger.debug(
            "REPLEnvironment initialized",
            sandbox_mode=sandbox_mode,
            max_output_length=max_output_length,
            timeout=timeout,
            allowed_imports=list(self._allowed_imports),
        )

    # =========================================================================
    # Variable Management
    # =========================================================================

    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the environment.

        Args:
            name: Variable name (must be valid Python identifier)
            value: Variable value (any type)

        Raises:
            ValueError: If name is not a valid identifier
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid variable name: '{name}'")

        var_type = type(value).__name__
        is_function = callable(value)

        if name in self._variables:
            # Update existing variable
            existing = self._variables[name]
            existing.value = value
            existing.var_type = var_type
            existing.updated_at = datetime.now()
            existing.is_function = is_function
            logger.debug("Variable updated", name=name, type=var_type)
        else:
            # Create new variable
            self._variables[name] = REPLVariable(
                name=name,
                value=value,
                var_type=var_type,
                is_function=is_function,
            )
            logger.debug("Variable created", name=name, type=var_type)

    def get_variable(self, name: str, default: Any = None) -> Any:
        """
        Get a variable from the environment.

        Args:
            name: Variable name
            default: Default value if not found

        Returns:
            Variable value or default
        """
        if name not in self._variables:
            return default

        var = self._variables[name]
        var.read_count += 1
        return var.value

    def has_variable(self, name: str) -> bool:
        """
        Check if variable exists.

        Args:
            name: Variable name

        Returns:
            True if variable exists
        """
        return name in self._variables

    def delete_variable(self, name: str) -> bool:
        """
        Delete a variable.

        Args:
            name: Variable name

        Returns:
            True if deleted, False if not found
        """
        if name in self._variables:
            del self._variables[name]
            logger.debug("Variable deleted", name=name)
            return True
        return False

    def list_variables(self) -> list[str]:
        """
        List all variable names.

        Returns:
            List of variable names
        """
        return list(self._variables.keys())

    def get_variable_info(self, name: str) -> REPLVariable | None:
        """
        Get detailed variable info.

        Args:
            name: Variable name

        Returns:
            REPLVariable object or None if not found
        """
        return self._variables.get(name)

    def get_all_variables(self) -> dict[str, Any]:
        """
        Get all variables as a dictionary.

        Returns:
            Dictionary of variable name to value
        """
        return {name: var.value for name, var in self._variables.items()}

    # =========================================================================
    # Function Management
    # =========================================================================

    def set_function(
        self,
        name: str,
        func: Callable[..., Any],
        is_async: bool = False,
    ) -> None:
        """
        Register a function in the environment.

        Args:
            name: Function name accessible in code
            func: The function to register
            is_async: Whether function is async

        Raises:
            ValueError: If name is not a valid identifier
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid function name: '{name}'")

        if is_async:
            self._async_functions[name] = func
            logger.debug("Async function registered", name=name)
        else:
            self._functions[name] = func
            logger.debug("Sync function registered", name=name)

    def remove_function(self, name: str) -> bool:
        """
        Remove a registered function.

        Args:
            name: Function name

        Returns:
            True if removed, False if not found
        """
        if name in self._functions:
            del self._functions[name]
            return True
        if name in self._async_functions:
            del self._async_functions[name]
            return True
        return False

    def list_functions(self) -> dict[str, bool]:
        """
        List all registered functions.

        Returns:
            Dictionary of function name to is_async flag
        """
        result: dict[str, bool] = {}
        for name in self._functions:
            result[name] = False
        for name in self._async_functions:
            result[name] = True
        return result

    # =========================================================================
    # Code Execution
    # =========================================================================

    def execute(self, code: str) -> CodeExecutionResult:
        """
        Execute code synchronously.

        Args:
            code: Python code to execute

        Returns:
            CodeExecutionResult with output and status

        Note:
            Use execute_async() for code that calls async functions.
        """
        try:
            # Run async execute in a new event loop
            return asyncio.run(self.execute_async(code))
        except RuntimeError:
            # Already in an event loop - use different approach
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.execute_async(code))
            finally:
                loop.close()

    async def execute_async(self, code: str) -> CodeExecutionResult:
        """
        Execute code with async support.

        Allows calling registered async functions like llm_query().

        Args:
            code: Python code to execute

        Returns:
            CodeExecutionResult with output and status
        """
        start_time = time.time()
        self._execution_count += 1
        self._output_buffer = []

        logger.debug(
            "Executing code",
            execution_count=self._execution_count,
            code_length=len(code),
        )

        # Track variables before execution
        vars_before = set(self._variables.keys())
        var_values_before = {name: id(var.value) for name, var in self._variables.items()}

        # Build execution globals
        globals_dict = self._build_globals()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_code_internal(code, globals_dict),
                timeout=self._timeout,
            )

            # Track new and modified variables
            vars_created: list[str] = []
            vars_modified: list[str] = []

            # Check namespace for new variables
            for name, value in globals_dict.items():
                if name.startswith("_") or name in self._build_safe_builtins():
                    continue
                if callable(value) and name in self._functions:
                    continue
                if callable(value) and name in self._async_functions:
                    continue

                if name not in vars_before:
                    # New variable
                    self.set_variable(name, value)
                    vars_created.append(name)
                elif name in self._variables and id(value) != var_values_before.get(name):
                    # Modified variable
                    self.set_variable(name, value)
                    vars_modified.append(name)

            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            output = self._truncate_output("".join(self._output_buffer))

            logger.debug(
                "Code executed successfully",
                execution_time=round(execution_time, 4),
                output_length=len(output),
                variables_created=vars_created,
                variables_modified=vars_modified,
            )

            return CodeExecutionResult(
                success=True,
                output=output if output else str(result) if result else "(no output)",
                return_value=result,
                execution_time=execution_time,
                variables_created=vars_created,
                variables_modified=vars_modified,
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Code execution timed out after {self._timeout}s"
            self._last_error = error_msg
            logger.warning("Code execution timeout", timeout=self._timeout)

            return CodeExecutionResult(
                success=False,
                output="".join(self._output_buffer),
                error=error_msg,
                error_type="TimeoutError",
                execution_time=execution_time,
            )

        except SyntaxError as e:
            execution_time = time.time() - start_time
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            self._last_error = error_msg

            logger.warning(
                "Code syntax error",
                line=e.lineno,
                error=e.msg,
            )

            return CodeExecutionResult(
                success=False,
                output="".join(self._output_buffer),
                error=error_msg,
                error_type="SyntaxError",
                traceback=traceback.format_exc(),
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            error_type = type(e).__name__
            self._last_error = f"{error_type}: {error_msg}"

            logger.warning(
                "Code execution failed",
                error_type=error_type,
                error=error_msg,
                execution_time=round(execution_time, 4),
            )

            return CodeExecutionResult(
                success=False,
                output="".join(self._output_buffer),
                error=error_msg,
                error_type=error_type,
                traceback=traceback.format_exc(),
                execution_time=execution_time,
            )

    async def _execute_code_internal(
        self,
        code: str,
        globals_dict: dict[str, Any],
    ) -> Any:
        """
        Internal code execution with async support.

        Args:
            code: Code to execute
            globals_dict: Execution namespace

        Returns:
            Execution result (if any)

        Raises:
            RLMCodeExecutionError: If code has syntax errors
        """
        # Compile code
        try:
            compiled = compile(code, "<repl>", "exec")
        except SyntaxError as e:
            raise RLMCodeExecutionError(
                code=code,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
            ) from e

        # Execute compiled code
        exec(compiled, globals_dict)

        # Return special _result variable if set
        return globals_dict.get("_result")

    # =========================================================================
    # Globals Building
    # =========================================================================

    def _build_globals(self) -> dict[str, Any]:
        """
        Build globals dict for execution.

        Returns:
            Namespace dictionary with builtins, variables, and functions
        """
        # Start with safe builtins
        if self._sandbox_mode:
            namespace: dict[str, Any] = {
                "__builtins__": self._build_safe_builtins(),
            }
        else:
            namespace = {
                "__builtins__": __builtins__,
            }

        # Override print with capture function
        namespace["print"] = self._capture_print

        # Add variables
        for name, var in self._variables.items():
            namespace[name] = var.value

        # Add sync functions
        for name, func in self._functions.items():
            namespace[name] = func

        # Add async functions with sync wrapper
        for name, async_func in self._async_functions.items():
            namespace[name] = self._create_sync_wrapper(async_func)

        # Add allowed imports as a restricted __import__
        if self._allowed_imports:
            namespace["__builtins__"]["__import__"] = self._restricted_import

        return namespace

    def _build_safe_builtins(self) -> dict[str, Any]:
        """
        Build restricted builtins dict.

        Returns:
            Dictionary of safe builtin functions
        """
        import builtins

        safe_builtins: dict[str, Any] = {}

        for name in SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Special handling for None, True, False
        safe_builtins["None"] = None
        safe_builtins["True"] = True
        safe_builtins["False"] = False

        return safe_builtins

    def _restricted_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """
        Restricted import that only allows specific modules.

        Args:
            name: Module name to import
            globals: Globals dict
            locals: Locals dict
            fromlist: Names to import from module
            level: Relative import level

        Returns:
            Imported module

        Raises:
            ImportError: If module is not allowed
        """
        if name not in self._allowed_imports:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox mode")

        return __builtins__["__import__"](name, globals, locals, fromlist, level)

    # =========================================================================
    # Output Capture
    # =========================================================================

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

    def _truncate_output(self, output: str) -> str:
        """
        Truncate output if too long.

        Args:
            output: Output string

        Returns:
            Truncated output
        """
        if len(output) > self._max_output_length:
            truncated = output[: self._max_output_length]
            truncated += (
                f"\n...(output truncated, {len(output) - self._max_output_length} chars omitted)"
            )
            return truncated
        return output

    # =========================================================================
    # Async Function Wrapper
    # =========================================================================

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
                # We're in an async context, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result(timeout=self._timeout)
            except RuntimeError:
                # No running loop, run directly
                return asyncio.run(async_func(*args, **kwargs))

        # Preserve function name for better error messages
        sync_wrapper.__name__ = async_func.__name__
        sync_wrapper.__doc__ = async_func.__doc__

        return sync_wrapper

    # =========================================================================
    # State Management
    # =========================================================================

    def clear(self) -> None:
        """Clear all variables and reset state."""
        self._variables.clear()
        self._output_buffer.clear()
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_error = None
        logger.debug("REPL environment cleared")

    def reset(self) -> None:
        """Reset to initial state (keep functions)."""
        self._variables.clear()
        self._output_buffer.clear()
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_error = None
        logger.debug("REPL environment reset (functions preserved)")

    def get_state_snapshot(self) -> dict[str, Any]:
        """
        Get snapshot of current state.

        Returns:
            Dictionary with state information
        """
        return {
            "variables": {
                name: {
                    "value": var.value,
                    "type": var.var_type,
                    "created_at": var.created_at.isoformat(),
                    "updated_at": var.updated_at.isoformat(),
                    "read_count": var.read_count,
                }
                for name, var in self._variables.items()
            },
            "functions": list(self._functions.keys()),
            "async_functions": list(self._async_functions.keys()),
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "last_error": self._last_error,
            "sandbox_mode": self._sandbox_mode,
            "timeout": self._timeout,
            "max_output_length": self._max_output_length,
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """
        Restore from snapshot.

        Args:
            snapshot: State snapshot from get_state_snapshot()

        Note:
            Functions are not restored from snapshots for security reasons.
        """
        self._variables.clear()

        for name, var_data in snapshot.get("variables", {}).items():
            self._variables[name] = REPLVariable(
                name=name,
                value=var_data["value"],
                var_type=var_data["type"],
                created_at=datetime.fromisoformat(var_data["created_at"]),
                updated_at=datetime.fromisoformat(var_data["updated_at"]),
                read_count=var_data["read_count"],
            )

        self._execution_count = snapshot.get("execution_count", 0)
        self._total_execution_time = snapshot.get("total_execution_time", 0.0)
        self._last_error = snapshot.get("last_error")

        logger.debug(
            "REPL state restored",
            variables=list(self._variables.keys()),
            execution_count=self._execution_count,
        )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "execution_count": self._execution_count,
            "total_execution_time": round(self._total_execution_time, 4),
            "average_execution_time": (
                round(self._total_execution_time / self._execution_count, 4)
                if self._execution_count > 0
                else 0.0
            ),
            "variable_count": len(self._variables),
            "function_count": len(self._functions) + len(self._async_functions),
            "last_error": self._last_error,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_rlm_repl(
    context: str,
    llm_query_fn: Callable[..., Coroutine[Any, Any, str]],
    task: str | None = None,
    timeout: float = 30.0,
    sandbox_mode: bool = True,
) -> REPLEnvironment:
    """
    Create REPL pre-configured for RLM.

    Args:
        context: The context/document content
        llm_query_fn: Async function for LLM sub-queries
        task: Optional task description
        timeout: Execution timeout in seconds
        sandbox_mode: Whether to enable sandbox mode

    Returns:
        Configured REPLEnvironment
    """
    repl = REPLEnvironment(
        sandbox_mode=sandbox_mode,
        timeout=timeout,
    )

    # Set core variables
    repl.set_variable("context", context)
    if task:
        repl.set_variable("task", task)

    # Register llm_query as async function
    repl.set_function("llm_query", llm_query_fn, is_async=True)

    logger.debug(
        "RLM REPL created",
        context_length=len(context),
        has_task=task is not None,
        timeout=timeout,
    )

    return repl


async def execute_code_safely(
    code: str,
    variables: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> CodeExecutionResult:
    """
    Execute code with default safety settings.

    Args:
        code: Python code to execute
        variables: Optional dictionary of variables to set
        timeout: Execution timeout in seconds

    Returns:
        CodeExecutionResult with output and status
    """
    repl = REPLEnvironment(
        sandbox_mode=True,
        timeout=timeout,
    )

    # Set provided variables
    if variables:
        for name, value in variables.items():
            repl.set_variable(name, value)

    return await repl.execute_async(code)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ExecutionMode",
    # Data Classes
    "REPLVariable",
    "CodeExecutionResult",
    # Constants
    "SAFE_BUILTINS",
    "BLOCKED_BUILTINS",
    # Main Class
    "REPLEnvironment",
    # Convenience Functions
    "create_rlm_repl",
    "execute_code_safely",
]
