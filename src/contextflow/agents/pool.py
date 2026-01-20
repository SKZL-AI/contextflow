"""
Agent Pool for ContextFlow.

Manages multiple SubAgents for parallel task execution.

Features:
- Parallel task execution with configurable concurrency
- Load balancing across agents
- Resource management (rate limiting)
- Agent lifecycle management

Based on Boris' Best Practices:
- Step 8: Subagents for parallel processing
- Step 13: Verification feedback loop
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextflow.agents.sub_agent import (
    AgentConfig,
    AgentResult,
    AgentRole,
    SubAgent,
)
from contextflow.utils.errors import PoolError
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider
    from contextflow.rag.temp_rag import TemporaryRAG

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class PoolStrategy(Enum):
    """
    Task distribution strategies for the agent pool.

    Each strategy determines how tasks are assigned to available agents.
    """

    ROUND_ROBIN = "round_robin"  # Distribute evenly in order
    LEAST_BUSY = "least_busy"  # Send to agent with fewest executions
    ROLE_MATCH = "role_match"  # Match task role to agent role
    RANDOM = "random"  # Random distribution


class PoolStatus(Enum):
    """Current status of the pool."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    SCALING = "scaling"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PoolConfig:
    """
    Configuration for Agent Pool.

    Controls pool behavior, scaling, and resource limits.

    Attributes:
        max_agents: Maximum number of agents in the pool
        max_concurrent_tasks: Maximum tasks running simultaneously
        default_timeout: Default task timeout in seconds
        strategy: Task distribution strategy
        auto_scale: Whether to automatically scale agents
        min_agents: Minimum agents to maintain
        agent_config: Default configuration for agents
        rate_limit_rpm: Rate limit (requests per minute), 0 for unlimited
        rate_limit_tpm: Token limit (tokens per minute), 0 for unlimited
    """

    max_agents: int = 10
    max_concurrent_tasks: int = 5
    default_timeout: float = 60.0
    strategy: PoolStrategy = PoolStrategy.LEAST_BUSY
    auto_scale: bool = True
    min_agents: int = 1
    agent_config: AgentConfig | None = None
    rate_limit_rpm: int = 0  # 0 = unlimited
    rate_limit_tpm: int = 0  # 0 = unlimited


@dataclass
class PoolTask:
    """
    Task to be executed by pool.

    Encapsulates all information needed to execute a task.

    Attributes:
        id: Unique task identifier
        task: Task description
        context: Optional context to include
        role: Agent role for execution
        constraints: Optional verification constraints
        priority: Task priority (higher = more important)
        created_at: Task creation timestamp
        timeout: Task-specific timeout override
    """

    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    task: str = ""
    context: str | None = None
    role: AgentRole = AgentRole.ANALYZER
    constraints: list[str] | None = None
    priority: int = 0  # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)
    timeout: float | None = None

    def __lt__(self, other: PoolTask) -> bool:
        """Enable priority queue sorting (higher priority first)."""
        return self.priority > other.priority


@dataclass
class PoolStats:
    """
    Pool statistics.

    Provides insight into pool performance and utilization.

    Attributes:
        total_agents: Total number of agents in pool
        active_agents: Number of agents currently executing
        idle_agents: Number of agents available
        tasks_queued: Number of tasks waiting
        tasks_completed: Total completed tasks
        tasks_failed: Total failed tasks
        average_task_time: Average execution time in seconds
        uptime: Pool uptime in seconds
        total_tokens: Total tokens used
        requests_per_minute: Current request rate
    """

    total_agents: int
    active_agents: int
    idle_agents: int
    tasks_queued: int
    tasks_completed: int
    tasks_failed: int
    average_task_time: float
    uptime: float
    total_tokens: int = 0
    requests_per_minute: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_agents": self.total_agents,
            "active_agents": self.active_agents,
            "idle_agents": self.idle_agents,
            "tasks_queued": self.tasks_queued,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "average_task_time": round(self.average_task_time, 3),
            "uptime": round(self.uptime, 2),
            "total_tokens": self.total_tokens,
            "requests_per_minute": round(self.requests_per_minute, 2),
        }


# =============================================================================
# Agent Pool Class
# =============================================================================


class AgentPool:
    """
    Pool of SubAgents for parallel task execution.

    Manages a pool of SubAgents that can execute tasks in parallel,
    with load balancing, auto-scaling, and resource management.

    Features:
    - Parallel task execution with configurable concurrency
    - Multiple load balancing strategies
    - Auto-scaling based on workload
    - Rate limiting for API protection
    - Agent lifecycle management

    Usage:
        # Basic usage
        pool = AgentPool(provider, max_agents=5)

        # Submit single task
        result = await pool.submit(task="Analyze this", context=content)

        # Submit multiple tasks
        results = await pool.map(
            tasks=["Task 1", "Task 2", "Task 3"],
            contexts=[ctx1, ctx2, ctx3]
        )

        # Parallel execution with different roles
        results = await pool.execute_parallel([
            PoolTask(task="Summarize", role=AgentRole.SUMMARIZER),
            PoolTask(task="Extract APIs", role=AgentRole.EXTRACTOR),
            PoolTask(task="Review code", role=AgentRole.CODE_REVIEWER)
        ])

        # Context manager usage
        async with AgentPool(provider) as pool:
            result = await pool.submit(task="Analyze")

    Attributes:
        provider: LLM provider for all agents
        config: Pool configuration
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: PoolConfig | None = None,
        rag: TemporaryRAG | None = None,
    ) -> None:
        """
        Initialize Agent Pool.

        Args:
            provider: LLM provider for all agents
            config: Pool configuration (uses defaults if None)
            rag: Optional RAG for context retrieval
        """
        self._provider = provider
        self._config = config or PoolConfig()
        self._rag = rag

        # Agent management
        self._agents: dict[str, SubAgent] = {}  # agent_id -> agent
        self._agent_roles: dict[AgentRole, list[str]] = {}  # role -> [agent_ids]
        self._active_agents: set[str] = set()  # Currently executing
        self._agent_lock = asyncio.Lock()

        # Task management
        self._task_queue: asyncio.PriorityQueue[PoolTask] = asyncio.PriorityQueue()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)

        # Statistics
        self._created_at = datetime.now()
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_execution_time = 0.0
        self._total_tokens = 0
        self._request_times: list[float] = []  # For RPM calculation

        # Rate limiting
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = 0.0

        # Round-robin tracking
        self._round_robin_index = 0

        # Pool status
        self._status = PoolStatus.INITIALIZING

        logger.info(
            "AgentPool initialized",
            max_agents=self._config.max_agents,
            max_concurrent=self._config.max_concurrent_tasks,
            strategy=self._config.strategy.value,
            auto_scale=self._config.auto_scale,
        )

        # Initialize minimum agents
        self._initialize_agents()
        self._status = PoolStatus.RUNNING

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def status(self) -> PoolStatus:
        """Current pool status."""
        return self._status

    @property
    def agent_count(self) -> int:
        """Current number of agents in pool."""
        return len(self._agents)

    @property
    def active_count(self) -> int:
        """Number of currently active agents."""
        return len(self._active_agents)

    @property
    def idle_count(self) -> int:
        """Number of idle agents."""
        return len(self._agents) - len(self._active_agents)

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    async def submit(
        self,
        task: str,
        context: str | None = None,
        role: AgentRole = AgentRole.ANALYZER,
        constraints: list[str] | None = None,
        timeout: float | None = None,
        priority: int = 0,
    ) -> AgentResult:
        """
        Submit a single task to the pool.

        Acquires an available agent, executes the task, and returns
        the result. Respects concurrency limits and rate limiting.

        Args:
            task: Task description
            context: Optional context to include
            role: Agent role for execution
            constraints: Optional verification constraints
            timeout: Task timeout override
            priority: Task priority (higher = more important)

        Returns:
            AgentResult with output and verification

        Raises:
            PoolError: If pool is not running
            AgentError: If execution fails
        """
        if self._status != PoolStatus.RUNNING:
            raise PoolError(message=f"Pool is not running (status: {self._status.value})")

        pool_task = PoolTask(
            task=task,
            context=context,
            role=role,
            constraints=constraints,
            priority=priority,
            timeout=timeout or self._config.default_timeout,
        )

        return await self._execute_task(pool_task)

    async def map(
        self,
        tasks: list[str],
        contexts: list[str] | None = None,
        role: AgentRole = AgentRole.ANALYZER,
        constraints: list[str] | None = None,
    ) -> list[AgentResult]:
        """
        Map tasks to agents and collect results.

        Distributes tasks across available agents and executes them
        in parallel, respecting concurrency limits.

        Args:
            tasks: List of task descriptions
            contexts: Optional list of contexts (one per task)
            role: Agent role for all tasks
            constraints: Optional constraints for all tasks

        Returns:
            List of AgentResult objects in same order as tasks

        Raises:
            PoolError: If pool is not running
            ValueError: If contexts list length doesn't match tasks
        """
        if self._status != PoolStatus.RUNNING:
            raise PoolError(message=f"Pool is not running (status: {self._status.value})")

        if contexts is not None and len(contexts) != len(tasks):
            raise ValueError(
                f"Contexts length ({len(contexts)}) must match tasks length ({len(tasks)})"
            )

        # Create pool tasks
        pool_tasks = [
            PoolTask(
                task=task,
                context=contexts[i] if contexts else None,
                role=role,
                constraints=constraints,
                timeout=self._config.default_timeout,
            )
            for i, task in enumerate(tasks)
        ]

        return await self.execute_parallel(pool_tasks)

    async def execute_parallel(
        self,
        pool_tasks: list[PoolTask],
    ) -> list[AgentResult]:
        """
        Execute multiple tasks in parallel.

        Executes all tasks concurrently, respecting the pool's
        concurrency limits and rate limiting.

        Args:
            pool_tasks: List of PoolTask objects to execute

        Returns:
            List of AgentResult objects in same order as tasks

        Raises:
            PoolError: If pool is not running
        """
        if self._status != PoolStatus.RUNNING:
            raise PoolError(message=f"Pool is not running (status: {self._status.value})")

        if not pool_tasks:
            return []

        logger.info(
            "Executing parallel tasks",
            count=len(pool_tasks),
            strategy=self._config.strategy.value,
        )

        # Sort by priority (higher first)
        sorted_tasks = sorted(pool_tasks, key=lambda t: t.priority, reverse=True)

        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[self._execute_task(task) for task in sorted_tasks],
            return_exceptions=True,
        )

        # Process results, converting exceptions to failed results
        final_results: list[AgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                final_results.append(
                    AgentResult(
                        success=False,
                        output="",
                        task=sorted_tasks[i].task,
                        agent_id="pool-error",
                        role=sorted_tasks[i].role,
                        execution_time=0.0,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        # Reorder to match original task order
        task_id_to_result = {sorted_tasks[i].id: final_results[i] for i in range(len(sorted_tasks))}
        ordered_results = [task_id_to_result[task.id] for task in pool_tasks]

        return ordered_results

    # =========================================================================
    # Internal Execution Methods
    # =========================================================================

    async def _execute_task(self, pool_task: PoolTask) -> AgentResult:
        """
        Execute a single task with concurrency control.

        Args:
            pool_task: Task to execute

        Returns:
            AgentResult from execution
        """
        async with self._semaphore:
            # Apply rate limiting
            await self._apply_rate_limit()

            # Get agent
            agent = await self._get_agent(pool_task.role)

            try:
                # Mark agent as active
                self._active_agents.add(agent.agent_id)

                # Execute task
                start_time = time.time()
                result = await asyncio.wait_for(
                    agent.execute(
                        task=pool_task.task,
                        context=pool_task.context,
                        constraints=pool_task.constraints,
                    ),
                    timeout=pool_task.timeout or self._config.default_timeout,
                )
                execution_time = time.time() - start_time

                # Update statistics
                self._total_execution_time += execution_time
                if result.success:
                    self._tasks_completed += 1
                else:
                    self._tasks_failed += 1

                if result.token_usage:
                    self._total_tokens += result.token_usage.get("total_tokens", 0)

                # Record request time for RPM calculation
                self._request_times.append(time.time())
                self._cleanup_request_times()

                return result

            except TimeoutError:
                self._tasks_failed += 1
                logger.warning(
                    "Task timed out",
                    task_id=pool_task.id,
                    agent_id=agent.agent_id,
                    timeout=pool_task.timeout,
                )
                return AgentResult(
                    success=False,
                    output="",
                    task=pool_task.task,
                    agent_id=agent.agent_id,
                    role=pool_task.role,
                    execution_time=pool_task.timeout or self._config.default_timeout,
                    error=f"Task timed out after {pool_task.timeout}s",
                )

            except Exception as e:
                self._tasks_failed += 1
                logger.error(
                    "Task execution failed",
                    task_id=pool_task.id,
                    error=str(e),
                )
                return AgentResult(
                    success=False,
                    output="",
                    task=pool_task.task,
                    agent_id=agent.agent_id,
                    role=pool_task.role,
                    execution_time=0.0,
                    error=str(e),
                )

            finally:
                # Release agent
                await self._release_agent(agent)

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self._config.rate_limit_rpm <= 0:
            return

        async with self._rate_limit_lock:
            min_interval = 60.0 / self._config.rate_limit_rpm
            elapsed = time.time() - self._last_request_time

            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                logger.debug(
                    "Rate limiting",
                    wait_time=round(wait_time, 3),
                )
                await asyncio.sleep(wait_time)

            self._last_request_time = time.time()

    def _cleanup_request_times(self) -> None:
        """Remove request times older than 1 minute."""
        cutoff = time.time() - 60.0
        self._request_times = [t for t in self._request_times if t > cutoff]

    # =========================================================================
    # Agent Management
    # =========================================================================

    def _initialize_agents(self) -> None:
        """Initialize minimum number of agents."""
        for _ in range(self._config.min_agents):
            self._create_agent(AgentRole.ANALYZER)

    async def _get_agent(self, role: AgentRole) -> SubAgent:
        """
        Get an available agent based on strategy.

        Args:
            role: Requested agent role

        Returns:
            Available SubAgent

        Raises:
            PoolError: If no agent can be obtained
        """
        async with self._agent_lock:
            # Check if we need to scale up
            if self._config.auto_scale and self.idle_count == 0:
                if self.agent_count < self._config.max_agents:
                    await self._scale_up(role)

            # Get agent based on strategy
            agent = self._select_agent(role)

            if agent is None:
                # Create new agent if possible
                if self.agent_count < self._config.max_agents:
                    agent = self._create_agent(role)
                else:
                    raise PoolError(message="No available agents and max agents reached")

            return agent

    def _select_agent(self, role: AgentRole) -> SubAgent | None:
        """
        Select an agent based on the configured strategy.

        Args:
            role: Requested agent role

        Returns:
            Selected agent or None if no suitable agent found
        """
        # Get idle agents
        idle_agents = [
            agent for agent_id, agent in self._agents.items() if agent_id not in self._active_agents
        ]

        if not idle_agents:
            return None

        strategy = self._config.strategy

        if strategy == PoolStrategy.ROUND_ROBIN:
            agent = idle_agents[self._round_robin_index % len(idle_agents)]
            self._round_robin_index += 1
            return agent

        elif strategy == PoolStrategy.LEAST_BUSY:
            # Select agent with fewest executions
            return min(
                idle_agents,
                key=lambda a: a.get_stats()["execution_count"],
            )

        elif strategy == PoolStrategy.ROLE_MATCH:
            # Prefer agents with matching role
            matching = [a for a in idle_agents if a.role == role]
            if matching:
                return matching[0]
            # Fall back to any idle agent
            return idle_agents[0]

        elif strategy == PoolStrategy.RANDOM:
            return random.choice(idle_agents)

        else:
            # Default to first available
            return idle_agents[0]

    async def _release_agent(self, agent: SubAgent) -> None:
        """
        Release agent back to pool.

        Args:
            agent: Agent to release
        """
        async with self._agent_lock:
            self._active_agents.discard(agent.agent_id)

            # Check if we should scale down
            if self._config.auto_scale:
                await self._scale_down()

    def _create_agent(self, role: AgentRole) -> SubAgent:
        """
        Create a new agent.

        Args:
            role: Role for the new agent

        Returns:
            Newly created SubAgent
        """
        # Use pool config or create default
        agent_config = self._config.agent_config or AgentConfig(
            role=role,
            enable_verification=True,
            timeout=self._config.default_timeout,
        )

        # Ensure role matches
        agent_config.role = role

        agent = SubAgent(
            provider=self._provider,
            role=role,
            config=agent_config,
            rag=self._rag,
        )

        # Register agent
        self._agents[agent.agent_id] = agent

        # Track by role
        if role not in self._agent_roles:
            self._agent_roles[role] = []
        self._agent_roles[role].append(agent.agent_id)

        logger.debug(
            "Agent created",
            agent_id=agent.agent_id,
            role=role.value,
            total_agents=len(self._agents),
        )

        return agent

    async def _scale_up(self, role: AgentRole | None = None) -> None:
        """
        Add more agents if needed.

        Args:
            role: Optional role for the new agent
        """
        if self.agent_count >= self._config.max_agents:
            return

        self._status = PoolStatus.SCALING
        target_role = role or AgentRole.ANALYZER
        self._create_agent(target_role)
        self._status = PoolStatus.RUNNING

        logger.info(
            "Scaled up",
            new_agent_count=self.agent_count,
            role=target_role.value,
        )

    async def _scale_down(self) -> None:
        """Remove idle agents if we have more than minimum."""
        if self.agent_count <= self._config.min_agents:
            return

        # Only scale down if many agents are idle
        if self.idle_count <= self._config.min_agents:
            return

        # Find an idle agent to remove
        idle_agents = [agent_id for agent_id in self._agents if agent_id not in self._active_agents]

        if len(idle_agents) > self._config.min_agents:
            self._status = PoolStatus.SCALING
            agent_id_to_remove = idle_agents[-1]
            agent = self._agents.pop(agent_id_to_remove)

            # Remove from role tracking
            role = agent.role
            if role in self._agent_roles:
                self._agent_roles[role] = [
                    aid for aid in self._agent_roles[role] if aid != agent_id_to_remove
                ]

            self._status = PoolStatus.RUNNING

            logger.debug(
                "Scaled down",
                removed_agent=agent_id_to_remove,
                remaining_agents=self.agent_count,
            )

    # =========================================================================
    # Statistics and Status
    # =========================================================================

    def get_stats(self) -> PoolStats:
        """
        Get pool statistics.

        Returns:
            PoolStats with current pool metrics
        """
        uptime = (datetime.now() - self._created_at).total_seconds()
        total_tasks = self._tasks_completed + self._tasks_failed
        avg_task_time = self._total_execution_time / total_tasks if total_tasks > 0 else 0.0

        # Calculate RPM
        rpm = len(self._request_times)  # Requests in last minute

        return PoolStats(
            total_agents=self.agent_count,
            active_agents=self.active_count,
            idle_agents=self.idle_count,
            tasks_queued=self._task_queue.qsize(),
            tasks_completed=self._tasks_completed,
            tasks_failed=self._tasks_failed,
            average_task_time=avg_task_time,
            uptime=uptime,
            total_tokens=self._total_tokens,
            requests_per_minute=float(rpm),
        )

    def get_agent_stats(self) -> list[dict[str, Any]]:
        """
        Get statistics for all agents in the pool.

        Returns:
            List of agent statistics dictionaries
        """
        return [agent.get_stats() for agent in self._agents.values()]

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def shutdown(self) -> None:
        """
        Gracefully shutdown pool.

        Waits for active tasks to complete and releases all agents.
        """
        if self._status == PoolStatus.STOPPED:
            return

        logger.info(
            "Shutting down pool",
            active_agents=self.active_count,
            pending_tasks=self._task_queue.qsize(),
        )

        self._status = PoolStatus.SHUTTING_DOWN

        # Wait for active tasks to complete (with timeout)
        wait_start = time.time()
        while self._active_agents and (time.time() - wait_start) < 30:
            await asyncio.sleep(0.1)

        # Clear agents
        async with self._agent_lock:
            self._agents.clear()
            self._agent_roles.clear()
            self._active_agents.clear()

        self._status = PoolStatus.STOPPED

        logger.info(
            "Pool shutdown complete",
            tasks_completed=self._tasks_completed,
            tasks_failed=self._tasks_failed,
        )

    async def __aenter__(self) -> AgentPool:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __repr__(self) -> str:
        return (
            f"AgentPool("
            f"agents={self.agent_count}, "
            f"active={self.active_count}, "
            f"status={self._status.value})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def parallel_execute(
    provider: BaseProvider,
    tasks: list[str],
    contexts: list[str] | None = None,
    max_concurrent: int = 5,
    role: AgentRole = AgentRole.ANALYZER,
) -> list[AgentResult]:
    """
    Quick parallel execution without pool setup.

    Creates a temporary pool, executes tasks, and shuts down.

    Args:
        provider: LLM provider
        tasks: List of task descriptions
        contexts: Optional list of contexts
        max_concurrent: Maximum concurrent tasks
        role: Agent role for all tasks

    Returns:
        List of AgentResult objects

    Example:
        results = await parallel_execute(
            provider=my_provider,
            tasks=["Summarize doc 1", "Summarize doc 2"],
            contexts=[doc1, doc2],
            max_concurrent=3
        )
    """
    config = PoolConfig(
        max_agents=max_concurrent,
        max_concurrent_tasks=max_concurrent,
        min_agents=1,
        auto_scale=True,
    )

    async with AgentPool(provider, config) as pool:
        return await pool.map(
            tasks=tasks,
            contexts=contexts,
            role=role,
        )


async def execute_with_roles(
    provider: BaseProvider,
    tasks_with_roles: list[dict[str, Any]],
    max_concurrent: int = 5,
) -> list[AgentResult]:
    """
    Execute tasks with specific roles.

    Args:
        provider: LLM provider
        tasks_with_roles: List of dicts with 'task', 'context', 'role' keys
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of AgentResult objects

    Example:
        results = await execute_with_roles(
            provider=my_provider,
            tasks_with_roles=[
                {"task": "Summarize", "context": doc1, "role": AgentRole.SUMMARIZER},
                {"task": "Review code", "context": code, "role": AgentRole.CODE_REVIEWER},
            ]
        )
    """
    config = PoolConfig(
        max_agents=max_concurrent,
        max_concurrent_tasks=max_concurrent,
        min_agents=1,
        strategy=PoolStrategy.ROLE_MATCH,
    )

    pool_tasks = [
        PoolTask(
            task=t.get("task", ""),
            context=t.get("context"),
            role=t.get("role", AgentRole.ANALYZER),
            constraints=t.get("constraints"),
            priority=t.get("priority", 0),
        )
        for t in tasks_with_roles
    ]

    async with AgentPool(provider, config) as pool:
        return await pool.execute_parallel(pool_tasks)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "PoolStrategy",
    "PoolStatus",
    # Data Classes
    "PoolConfig",
    "PoolTask",
    "PoolStats",
    # Main Class
    "AgentPool",
    # Convenience Functions
    "parallel_execute",
    "execute_with_roles",
]
