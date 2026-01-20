"""
SubAgent for ContextFlow RLM Strategy.

SubAgents are specialized workers that handle sub-tasks:
- Fresh context window per task
- Specialized system prompts
- Verification capability (Boris Step 13)
- RAG access for context retrieval

Based on Boris' Best Practices:
- Step 8: Subagents (code-simplifier, verify-app)
- Step 13: Verification feedback loop
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextflow.core.types import Message
from contextflow.strategies.verification import VerificationProtocol, VerificationResult
from contextflow.utils.errors import AgentError, AgentTimeoutError
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from contextflow.providers.base import BaseProvider
    from contextflow.rag.temp_rag import TemporaryRAG

logger = get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class AgentRole(Enum):
    """
    Pre-defined agent roles with specialized prompts.

    Each role comes with an optimized system prompt designed for
    specific types of tasks in the RLM pipeline.
    """

    ANALYZER = "analyzer"  # Analyzes content
    SUMMARIZER = "summarizer"  # Creates summaries
    EXTRACTOR = "extractor"  # Extracts specific info
    VERIFIER = "verifier"  # Verifies outputs
    CODE_REVIEWER = "code_reviewer"  # Reviews code
    RESEARCHER = "researcher"  # Researches topics
    SYNTHESIZER = "synthesizer"  # Synthesizes multiple sources
    CUSTOM = "custom"  # Custom role


class AgentStatus(Enum):
    """Current status of the agent."""

    IDLE = "idle"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentConfig:
    """
    Configuration for a SubAgent.

    Controls behavior, verification, and resource limits.

    Attributes:
        role: Agent role (determines default system prompt)
        system_prompt: Custom system prompt (overrides role default)
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature for LLM
        enable_verification: Whether to verify outputs
        verification_threshold: Minimum confidence for pass
        max_retries: Maximum retry attempts with feedback
        timeout: Execution timeout in seconds
        use_rag: Whether to use RAG for context retrieval
        rag_k: Number of RAG results to include in context
    """

    role: AgentRole = AgentRole.CUSTOM
    system_prompt: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    enable_verification: bool = True
    verification_threshold: float = 0.7
    max_retries: int = 3
    timeout: float = 60.0
    use_rag: bool = False
    rag_k: int = 5


@dataclass
class AgentResult:
    """
    Result from agent execution.

    Contains the output, verification results, and execution metadata.

    Attributes:
        success: Whether execution was successful
        output: Generated output text
        task: Original task description
        agent_id: Unique agent identifier
        role: Agent role used
        execution_time: Time taken in seconds
        token_usage: Token usage breakdown
        verification_result: Verification result if enabled
        metadata: Additional metadata
        error: Error message if failed
    """

    success: bool
    output: str
    task: str
    agent_id: str
    role: AgentRole
    execution_time: float
    token_usage: dict[str, int] = field(default_factory=dict)
    verification_result: VerificationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "task": self.task,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "execution_time": round(self.execution_time, 4),
            "token_usage": self.token_usage,
            "verification_passed": (
                self.verification_result.passed
                if self.verification_result
                else None
            ),
            "metadata": self.metadata,
            "error": self.error,
        }


# =============================================================================
# Role-Based System Prompts
# =============================================================================

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ANALYZER: """You are an expert analyst. Your task is to:
- Carefully analyze the provided content
- Identify key patterns, themes, and insights
- Structure your analysis clearly with sections
- Support conclusions with evidence from the content
- Be thorough but concise

Format your analysis with clear headings and bullet points where appropriate.""",

    AgentRole.SUMMARIZER: """You are an expert summarizer. Your task is to:
- Create concise, accurate summaries
- Capture all key points without losing important details
- Maintain the original meaning and tone
- Use clear, straightforward language
- Structure summaries logically

Prioritize clarity and completeness over brevity.""",

    AgentRole.EXTRACTOR: """You are an information extraction expert. Your task is to:
- Extract specific information as requested
- Be precise and complete in your extraction
- Format output clearly and consistently
- Only include information that appears in the source
- Note when requested information is not found

Use structured formats (lists, tables) when appropriate.""",

    AgentRole.VERIFIER: """You are a verification expert. Your task is to:
- Verify that outputs match requirements
- Check for accuracy and completeness
- Identify any issues, gaps, or inconsistencies
- Provide clear pass/fail assessment with reasoning
- Suggest specific improvements if verification fails

Be strict but fair in your assessments.""",

    AgentRole.CODE_REVIEWER: """You are a code review expert. Your task is to:
- Review code for correctness and best practices
- Identify potential bugs, issues, or vulnerabilities
- Check for proper error handling
- Suggest improvements and optimizations
- Consider security concerns

Be specific in your feedback with line references when possible.""",

    AgentRole.RESEARCHER: """You are a research expert. Your task is to:
- Research the topic thoroughly using provided context
- Gather and organize relevant information
- Synthesize findings into coherent insights
- Identify gaps in information
- Cite sources when available

Structure your research with clear sections and evidence.""",

    AgentRole.SYNTHESIZER: """You are a synthesis expert. Your task is to:
- Combine information from multiple sources
- Identify common themes and important differences
- Create coherent, unified output
- Resolve contradictions when possible
- Note unresolved conflicts or ambiguities

Present a balanced view that integrates all perspectives.""",

    AgentRole.CUSTOM: """You are a helpful assistant. Complete the task as requested.
Be thorough, accurate, and format your response clearly.""",
}


# =============================================================================
# SubAgent Class
# =============================================================================


class SubAgent:
    """
    Specialized agent for handling sub-tasks in the RLM pipeline.

    Features:
    - Fresh context per task (no history bleed between tasks)
    - Role-based system prompts for specialized behavior
    - Integrated verification with feedback loop
    - Optional RAG access for context retrieval

    Usage:
        # Create agent with a specific role
        agent = SubAgent(
            provider=provider,
            role=AgentRole.ANALYZER,
            name="content-analyzer"
        )

        # Execute a task
        result = await agent.execute(
            task="Analyze the API structure",
            context=code_content
        )

        # Check verification result
        if result.verification_result and result.verification_result.passed:
            print("Verified output:", result.output)

        # Execute with RAG
        agent_with_rag = SubAgent(
            provider=provider,
            role=AgentRole.RESEARCHER,
            rag=temp_rag
        )
        result = await agent_with_rag.execute_with_rag(
            task="Find information about authentication",
            query="authentication implementation"
        )

    Attributes:
        provider: LLM provider for completions
        role: Agent role (determines system prompt)
        name: Agent name
        config: Agent configuration
        rag: Optional RAG for context retrieval
    """

    def __init__(
        self,
        provider: BaseProvider,
        role: AgentRole = AgentRole.CUSTOM,
        name: str | None = None,
        config: AgentConfig | None = None,
        rag: TemporaryRAG | None = None,
    ) -> None:
        """
        Initialize SubAgent.

        Args:
            provider: LLM provider for completions
            role: Agent role (determines system prompt)
            name: Agent name (auto-generated if None)
            config: Agent configuration (uses defaults if None)
            rag: Optional RAG for context retrieval
        """
        self._provider = provider
        self._role = role
        self._name = name or f"agent-{role.value}-{uuid.uuid4().hex[:8]}"
        self._config = config or AgentConfig(role=role)
        self._rag = rag

        # Ensure config role matches constructor role
        if self._config.role != role:
            self._config.role = role

        # Internal state
        self._status = AgentStatus.IDLE
        self._created_at = datetime.now()
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._total_tokens = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._verification_passes = 0
        self._verification_failures = 0

        # Verifier instance (lazy initialized)
        self._verifier: VerificationProtocol | None = None

        logger.info(
            "SubAgent initialized",
            agent_id=self._name,
            role=role.value,
            enable_verification=self._config.enable_verification,
            use_rag=self._config.use_rag,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._name

    @property
    def role(self) -> AgentRole:
        """Agent role."""
        return self._role

    @property
    def status(self) -> AgentStatus:
        """Current agent status."""
        return self._status

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    async def execute(
        self,
        task: str,
        context: str | None = None,
        constraints: list[str] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Execute a task with fresh context.

        This is the primary method for task execution. Each call
        gets a fresh context window with no history from previous tasks.

        Args:
            task: Task description
            context: Optional context to include
            constraints: Optional constraints for verification
            **kwargs: Additional provider arguments

        Returns:
            AgentResult with output and verification

        Raises:
            AgentError: If execution fails
            AgentTimeoutError: If execution times out
        """
        start_time = time.time()
        self._execution_count += 1
        self._status = AgentStatus.EXECUTING

        logger.info(
            "Executing task",
            agent_id=self._name,
            task_length=len(task),
            has_context=context is not None,
            has_constraints=constraints is not None,
        )

        try:
            # Build messages for LLM call
            messages = await self._build_prompt(task, context)

            # Execute with timeout
            try:
                output = await asyncio.wait_for(
                    self._call_llm(messages, **kwargs),
                    timeout=self._config.timeout,
                )
            except TimeoutError:
                self._status = AgentStatus.FAILED
                self._failed_tasks += 1
                raise AgentTimeoutError(
                    agent_id=self._name,
                    timeout=int(self._config.timeout),
                )

            # Calculate initial token usage
            token_usage = {
                "input_tokens": self._estimate_tokens(
                    self.get_system_prompt() + task + (context or "")
                ),
                "output_tokens": self._estimate_tokens(output),
            }
            token_usage["total_tokens"] = (
                token_usage["input_tokens"] + token_usage["output_tokens"]
            )

            # Verification if enabled
            verification_result: VerificationResult | None = None
            if self._config.enable_verification:
                self._status = AgentStatus.VERIFYING
                verification_result = await self._verify_output(
                    task, output, constraints
                )

                # Retry loop if verification fails
                if not verification_result.passed and self._config.max_retries > 0:
                    output, verification_result, retry_tokens = await self._retry_loop(
                        task=task,
                        context=context,
                        output=output,
                        verification_result=verification_result,
                        constraints=constraints,
                        **kwargs,
                    )
                    # Add retry tokens to usage
                    token_usage["total_tokens"] += retry_tokens

                # Track verification stats
                if verification_result.passed:
                    self._verification_passes += 1
                else:
                    self._verification_failures += 1

            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            self._total_tokens += token_usage["total_tokens"]
            self._successful_tasks += 1
            self._status = AgentStatus.COMPLETED

            logger.info(
                "Task completed",
                agent_id=self._name,
                execution_time=round(execution_time, 3),
                tokens_used=token_usage["total_tokens"],
                verification_passed=(
                    verification_result.passed if verification_result else None
                ),
            )

            return AgentResult(
                success=True,
                output=output,
                task=task,
                agent_id=self._name,
                role=self._role,
                execution_time=execution_time,
                token_usage=token_usage,
                verification_result=verification_result,
                metadata={
                    "execution_count": self._execution_count,
                    "had_context": context is not None,
                    "had_constraints": constraints is not None,
                },
            )

        except AgentTimeoutError:
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            self._failed_tasks += 1
            self._status = AgentStatus.FAILED

            logger.error(
                "Task failed",
                agent_id=self._name,
                error=str(e),
                execution_time=round(execution_time, 3),
            )

            return AgentResult(
                success=False,
                output="",
                task=task,
                agent_id=self._name,
                role=self._role,
                execution_time=execution_time,
                error=str(e),
            )

    async def execute_with_rag(
        self,
        task: str,
        query: str | None = None,
        constraints: list[str] | None = None,
    ) -> AgentResult:
        """
        Execute task with RAG-retrieved context.

        Automatically retrieves relevant context from the RAG system
        before executing the task.

        Args:
            task: Task description
            query: RAG query (uses task if None)
            constraints: Optional constraints

        Returns:
            AgentResult with RAG-enhanced output

        Raises:
            AgentError: If RAG is not configured
        """
        if self._rag is None:
            raise AgentError(
                message="RAG not configured for this agent. "
                "Initialize SubAgent with rag parameter.",
            )

        # Use task as query if not specified
        rag_query = query or task

        logger.debug(
            "Retrieving RAG context",
            agent_id=self._name,
            query_length=len(rag_query),
            k=self._config.rag_k,
        )

        # Retrieve context from RAG
        context = await self._retrieve_context(rag_query)

        # Add RAG metadata to result
        result = await self.execute(
            task=task,
            context=context,
            constraints=constraints,
        )
        result.metadata["rag_query"] = rag_query
        result.metadata["rag_context_length"] = len(context)

        return result

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _retrieve_context(self, query: str) -> str:
        """
        Retrieve context from RAG.

        Uses 3-layer progressive disclosure for token efficiency.

        Args:
            query: Search query

        Returns:
            Retrieved context as formatted string
        """
        if self._rag is None:
            return ""

        # Layer 1: Get IDs
        doc_ids = await self._rag.search_compact(query, k=self._config.rag_k)

        if not doc_ids:
            logger.debug("No RAG results found", query=query[:50])
            return ""

        # Layer 3: Get full documents (skip Layer 2 for agents)
        documents = await self._rag.get_full_documents(doc_ids)

        # Format context
        context_parts: list[str] = []
        for i, doc in enumerate(documents):
            context_parts.append(
                f"--- Document {i + 1} ---\n{doc.content}\n"
            )

        context = "\n".join(context_parts)
        logger.debug(
            "RAG context retrieved",
            num_docs=len(documents),
            context_length=len(context),
        )

        return context

    async def _build_prompt(
        self,
        task: str,
        context: str | None,
    ) -> list[Message]:
        """
        Build messages for LLM call.

        Constructs the prompt with system prompt, optional context,
        and the task itself.

        Args:
            task: Task description
            context: Optional context

        Returns:
            List of Message objects for the LLM call
        """
        # Build user message
        user_content_parts: list[str] = []

        if context:
            user_content_parts.append(
                f"## Context\n\n{context}\n\n"
            )

        user_content_parts.append(
            f"## Task\n\n{task}"
        )

        user_content = "".join(user_content_parts)

        return [Message(role="user", content=user_content)]

    async def _call_llm(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call the LLM provider.

        Args:
            messages: Messages for the completion
            **kwargs: Additional provider arguments

        Returns:
            Generated output text
        """
        response = await self._provider.complete(
            messages=messages,
            system=self.get_system_prompt(),
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            **kwargs,
        )
        return response.content

    async def _verify_output(
        self,
        task: str,
        output: str,
        constraints: list[str] | None,
    ) -> VerificationResult:
        """
        Verify output using VerificationProtocol.

        Args:
            task: Original task
            output: Generated output
            constraints: Optional constraints

        Returns:
            VerificationResult with assessment
        """
        # Lazy initialize verifier
        if self._verifier is None:
            self._verifier = VerificationProtocol(
                provider=self._provider,
                min_confidence=self._config.verification_threshold,
            )

        return await self._verifier.verify(
            task=task,
            output=output,
            constraints=constraints,
        )

    async def _retry_loop(
        self,
        task: str,
        context: str | None,
        output: str,
        verification_result: VerificationResult,
        constraints: list[str] | None,
        **kwargs: Any,
    ) -> tuple[str, VerificationResult, int]:
        """
        Retry with verification feedback until pass or max retries.

        Args:
            task: Original task
            context: Original context
            output: Current output
            verification_result: Current verification result
            constraints: Optional constraints
            **kwargs: Additional provider arguments

        Returns:
            Tuple of (final_output, final_verification, total_retry_tokens)
        """
        self._status = AgentStatus.RETRYING
        current_output = output
        current_verification = verification_result
        total_retry_tokens = 0
        retry_count = 0

        while (
            not current_verification.passed
            and retry_count < self._config.max_retries
        ):
            retry_count += 1

            logger.debug(
                "Retrying with feedback",
                agent_id=self._name,
                retry=retry_count,
                issues=current_verification.issues[:3],
            )

            # Generate improved output
            current_output = await self._retry_with_feedback(
                task=task,
                context=context,
                previous_output=current_output,
                verification_result=current_verification,
            )

            # Estimate retry tokens
            total_retry_tokens += self._estimate_tokens(current_output)

            # Re-verify
            current_verification = await self._verify_output(
                task=task,
                output=current_output,
                constraints=constraints,
            )
            current_verification.iteration = retry_count

        return current_output, current_verification, total_retry_tokens

    async def _retry_with_feedback(
        self,
        task: str,
        context: str | None,
        previous_output: str,
        verification_result: VerificationResult,
    ) -> str:
        """
        Retry with verification feedback.

        Constructs a prompt that includes the previous output
        and verification issues to guide improvement.

        Args:
            task: Original task
            context: Original context
            previous_output: Previous attempt output
            verification_result: Previous verification result

        Returns:
            Improved output
        """
        # Format issues and suggestions
        issues_text = "\n".join(
            f"- {issue}" for issue in verification_result.issues
        ) or "No specific issues identified"

        suggestions_text = "\n".join(
            f"- {s}" for s in verification_result.suggestions
        ) or "No specific suggestions"

        # Build retry prompt
        retry_prompt_parts = [
            f"## Original Task\n\n{task}\n",
        ]

        if context:
            retry_prompt_parts.append(f"## Context\n\n{context}\n")

        retry_prompt_parts.extend([
            f"## Previous Output\n\n{previous_output}\n",
            f"## Issues to Address\n\n{issues_text}\n",
            f"## Suggestions\n\n{suggestions_text}\n",
            "## Instructions\n\n"
            "Please provide an improved version that addresses the issues above. "
            "Maintain the good aspects of the original while fixing the problems.",
        ])

        retry_prompt = "\n".join(retry_prompt_parts)
        messages = [Message(role="user", content=retry_prompt)]

        return await self._call_llm(messages)

    # =========================================================================
    # Configuration and Status
    # =========================================================================

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns the custom prompt if configured, otherwise
        returns the default prompt for the agent's role.

        Returns:
            System prompt string
        """
        if self._config.system_prompt:
            return self._config.system_prompt
        return ROLE_PROMPTS.get(self._role, ROLE_PROMPTS[AgentRole.CUSTOM])

    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration.

        Args:
            **kwargs: Configuration attributes to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug(
                    "Config updated",
                    agent_id=self._name,
                    key=key,
                    value=value,
                )

    def set_rag(self, rag: TemporaryRAG) -> None:
        """
        Set or update the RAG instance.

        Args:
            rag: TemporaryRAG instance
        """
        self._rag = rag
        self._config.use_rag = True
        logger.debug("RAG set for agent", agent_id=self._name)

    def get_stats(self) -> dict[str, Any]:
        """
        Get agent execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )

        return {
            "agent_id": self._name,
            "role": self._role.value,
            "status": self._status.value,
            "created_at": self._created_at.isoformat(),
            "execution_count": self._execution_count,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "total_execution_time": round(self._total_execution_time, 3),
            "avg_execution_time": round(avg_execution_time, 3),
            "total_tokens": self._total_tokens,
            "verification_passes": self._verification_passes,
            "verification_failures": self._verification_failures,
            "verification_success_rate": (
                self._verification_passes
                / (self._verification_passes + self._verification_failures)
                if (self._verification_passes + self._verification_failures) > 0
                else 0.0
            ),
            "config": {
                "enable_verification": self._config.enable_verification,
                "verification_threshold": self._config.verification_threshold,
                "max_retries": self._config.max_retries,
                "timeout": self._config.timeout,
                "use_rag": self._config.use_rag,
            },
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._total_tokens = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._verification_passes = 0
        self._verification_failures = 0
        logger.debug("Stats reset", agent_id=self._name)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses provider's count_tokens if available, otherwise
        uses rough character estimation.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        try:
            return self._provider.count_tokens(text)
        except Exception:
            # Fallback: ~4 chars per token
            return len(text) // 4

    def __repr__(self) -> str:
        return (
            f"SubAgent("
            f"id={self._name!r}, "
            f"role={self._role.value!r}, "
            f"status={self._status.value!r})"
        )


# =============================================================================
# Pre-configured Agent Factories
# =============================================================================


def create_analyzer_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create an analyzer agent.

    Optimized for analyzing content and identifying patterns.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    return SubAgent(
        provider=provider,
        role=AgentRole.ANALYZER,
        name=name,
        config=config,
        **kwargs,
    )


def create_summarizer_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create a summarizer agent.

    Optimized for creating concise summaries.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    return SubAgent(
        provider=provider,
        role=AgentRole.SUMMARIZER,
        name=name,
        config=config,
        **kwargs,
    )


def create_extractor_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create an extractor agent.

    Optimized for extracting specific information.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    return SubAgent(
        provider=provider,
        role=AgentRole.EXTRACTOR,
        name=name,
        config=config,
        **kwargs,
    )


def create_verifier_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create a verifier agent.

    Optimized for verifying outputs and checking correctness.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    # Verifier typically doesn't need self-verification
    if config is None:
        config = AgentConfig(
            role=AgentRole.VERIFIER,
            enable_verification=False,  # Avoid recursive verification
            temperature=0.3,  # Lower temperature for consistent checks
        )

    return SubAgent(
        provider=provider,
        role=AgentRole.VERIFIER,
        name=name,
        config=config,
        **kwargs,
    )


def create_code_reviewer_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create a code reviewer agent.

    Optimized for reviewing code and identifying issues.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    return SubAgent(
        provider=provider,
        role=AgentRole.CODE_REVIEWER,
        name=name,
        config=config,
        **kwargs,
    )


def create_researcher_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    rag: TemporaryRAG | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create a researcher agent.

    Optimized for researching topics, typically with RAG access.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        rag: Optional RAG for context retrieval
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    if config is None and rag is not None:
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            use_rag=True,
            rag_k=10,  # More results for research
        )

    return SubAgent(
        provider=provider,
        role=AgentRole.RESEARCHER,
        name=name,
        config=config,
        rag=rag,
        **kwargs,
    )


def create_synthesizer_agent(
    provider: BaseProvider,
    name: str | None = None,
    config: AgentConfig | None = None,
    **kwargs: Any,
) -> SubAgent:
    """
    Create a synthesizer agent.

    Optimized for combining information from multiple sources.

    Args:
        provider: LLM provider
        name: Optional agent name
        config: Optional custom config
        **kwargs: Additional SubAgent arguments

    Returns:
        Configured SubAgent
    """
    if config is None:
        config = AgentConfig(
            role=AgentRole.SYNTHESIZER,
            max_tokens=8192,  # Larger output for synthesis
        )

    return SubAgent(
        provider=provider,
        role=AgentRole.SYNTHESIZER,
        name=name,
        config=config,
        **kwargs,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_agent_task(
    provider: BaseProvider,
    task: str,
    context: str,
    role: AgentRole = AgentRole.ANALYZER,
    verify: bool = True,
) -> str:
    """
    Quick one-off agent task.

    Creates a temporary agent, executes the task, and returns the output.

    Args:
        provider: LLM provider
        task: Task description
        context: Context to include
        role: Agent role to use
        verify: Whether to verify output

    Returns:
        Generated output string

    Raises:
        AgentError: If task execution fails
    """
    config = AgentConfig(
        role=role,
        enable_verification=verify,
        max_retries=2 if verify else 0,
    )

    agent = SubAgent(
        provider=provider,
        role=role,
        config=config,
    )

    result = await agent.execute(task=task, context=context)

    if not result.success:
        raise AgentError(
            message=f"Quick task failed: {result.error}",
            details={"task": task, "role": role.value},
        )

    return result.output


async def parallel_agent_tasks(
    provider: BaseProvider,
    tasks: list[dict[str, Any]],
    role: AgentRole = AgentRole.ANALYZER,
    max_concurrent: int = 5,
) -> list[AgentResult]:
    """
    Execute multiple agent tasks in parallel.

    Args:
        provider: LLM provider
        tasks: List of task dicts with 'task' and optional 'context' keys
        role: Agent role to use for all tasks
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of AgentResult objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_task(task_dict: dict[str, Any], index: int) -> AgentResult:
        async with semaphore:
            agent = SubAgent(
                provider=provider,
                role=role,
                name=f"parallel-agent-{index}",
            )
            return await agent.execute(
                task=task_dict.get("task", ""),
                context=task_dict.get("context"),
                constraints=task_dict.get("constraints"),
            )

    results = await asyncio.gather(
        *[run_task(t, i) for i, t in enumerate(tasks)],
        return_exceptions=True,
    )

    # Convert exceptions to failed results
    final_results: list[AgentResult] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            final_results.append(
                AgentResult(
                    success=False,
                    output="",
                    task=tasks[i].get("task", ""),
                    agent_id=f"parallel-agent-{i}",
                    role=role,
                    execution_time=0.0,
                    error=str(result),
                )
            )
        else:
            final_results.append(result)

    return final_results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentRole",
    "AgentStatus",
    # Data Classes
    "AgentConfig",
    "AgentResult",
    # Constants
    "ROLE_PROMPTS",
    # Main Class
    "SubAgent",
    # Factory Functions
    "create_analyzer_agent",
    "create_summarizer_agent",
    "create_extractor_agent",
    "create_verifier_agent",
    "create_code_reviewer_agent",
    "create_researcher_agent",
    "create_synthesizer_agent",
    # Convenience Functions
    "quick_agent_task",
    "parallel_agent_tasks",
]
