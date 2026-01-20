"""
Structured logging setup for ContextFlow.

Provides consistent logging across all modules with support for
JSON formatting (production) and pretty printing (development).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, output JSON logs (for production)
        log_file: Optional file path to write logs to

    Example:
        # Development (pretty console output)
        setup_logging(level="DEBUG", json_format=False)

        # Production (JSON for log aggregation)
        setup_logging(level="INFO", json_format=True)
    """
    # Convert level string to int
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        logging.getLogger().addHandler(file_handler)

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        # JSON output for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        logger = get_logger(__name__)
        logger.info("Task started", task_id="123", tokens=5000)
        logger.error("Task failed", error=str(e), exc_info=True)
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding contextual information to logs.

    Example:
        with LogContext(task_id="123", user="john"):
            logger.info("Processing started")  # Includes task_id and user
            do_something()
            logger.info("Processing completed")  # Also includes context
    """

    def __init__(self, **context: Any):
        self.context = context
        self._token: object | None = None

    def __enter__(self) -> LogContext:
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_execution(
    logger: structlog.BoundLogger,
    operation: str,
    **extra: Any,
) -> None:
    """
    Log an operation with standard format.

    Args:
        logger: Logger instance
        operation: Operation name
        **extra: Additional context
    """
    logger.info(
        f"Executing: {operation}",
        operation=operation,
        **extra,
    )


def log_completion(
    logger: structlog.BoundLogger,
    operation: str,
    duration_ms: float,
    **extra: Any,
) -> None:
    """
    Log operation completion with duration.

    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **extra: Additional context
    """
    logger.info(
        f"Completed: {operation}",
        operation=operation,
        duration_ms=round(duration_ms, 2),
        **extra,
    )


def log_error(
    logger: structlog.BoundLogger,
    operation: str,
    error: Exception,
    **extra: Any,
) -> None:
    """
    Log an error with standard format.

    Args:
        logger: Logger instance
        operation: Operation name
        error: Exception that occurred
        **extra: Additional context
    """
    logger.error(
        f"Failed: {operation}",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        exc_info=True,
        **extra,
    )


# =============================================================================
# Specialized Loggers
# =============================================================================


class ProviderLogger:
    """Logger specifically for provider operations."""

    def __init__(self, provider_name: str):
        self.logger = get_logger(f"contextflow.providers.{provider_name}")
        self.provider_name = provider_name

    # Standard logging methods (proxy to underlying logger)
    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, provider=self.provider_name, **extra)

    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        self.logger.info(message, provider=self.provider_name, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, provider=self.provider_name, **extra)

    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        self.logger.error(message, provider=self.provider_name, **extra)

    def log_request(
        self,
        model: str,
        input_tokens: int,
        **extra: Any,
    ) -> None:
        """Log an API request."""
        self.logger.debug(
            "API request",
            provider=self.provider_name,
            model=model,
            input_tokens=input_tokens,
            **extra,
        )

    def log_response(
        self,
        model: str,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        **extra: Any,
    ) -> None:
        """Log an API response."""
        self.logger.debug(
            "API response",
            provider=self.provider_name,
            model=model,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(cost_usd, 6),
            **extra,
        )

    def log_error(self, error: Exception, **extra: Any) -> None:
        """Log a provider error."""
        self.logger.error(
            "Provider error",
            provider=self.provider_name,
            error_type=type(error).__name__,
            error_message=str(error),
            exc_info=True,
            **extra,
        )


class StrategyLogger:
    """Logger specifically for strategy operations."""

    def __init__(self, strategy_name: str):
        self.logger = get_logger(f"contextflow.strategies.{strategy_name}")
        self.strategy_name = strategy_name

    def log_start(self, token_count: int, **extra: Any) -> None:
        """Log strategy execution start."""
        self.logger.info(
            "Strategy execution started",
            strategy_name=self.strategy_name,
            token_count=token_count,
            **extra,
        )

    def log_iteration(self, iteration: int, **extra: Any) -> None:
        """Log strategy iteration."""
        self.logger.debug(
            "Strategy iteration",
            strategy_name=self.strategy_name,
            iteration=iteration,
            **extra,
        )

    def log_complete(
        self,
        total_tokens: int,
        total_cost: float,
        duration_seconds: float,
        **extra: Any,
    ) -> None:
        """Log strategy completion."""
        self.logger.info(
            "Strategy execution completed",
            strategy_name=self.strategy_name,
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            duration_seconds=round(duration_seconds, 2),
            **extra,
        )


# Initialize default logging on import
setup_logging()
