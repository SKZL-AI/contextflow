# API Reference

Complete API documentation for ContextFlow.

## Table of Contents

- [ContextFlow Class](#contextflow-class)
- [Configuration](#configuration)
- [Methods](#methods)
  - [process()](#process)
  - [stream()](#stream)
  - [analyze()](#analyze)
  - [configure()](#configure)
  - [get_provider()](#get_provider)
- [Types](#types)
- [Errors](#errors)

---

## ContextFlow Class

The main class for interacting with ContextFlow.

### Constructor

```python
def __init__(self, config: Optional[ContextFlowConfig] = None) -> None
```

Creates a new ContextFlow instance.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `ContextFlowConfig` | No | Configuration options |

#### Example

```python
from contextflow import ContextFlow
import os

# With configuration
flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-20250514"
)

# With defaults (reads from environment)
flow = ContextFlow()
```

---

## Configuration

### ContextFlowConfig

```python
from dataclasses import dataclass
from typing import Optional, List, Literal

@dataclass
class ContextFlowConfig:
    """Configuration for ContextFlow instance."""

    # Provider Settings
    provider: Optional[Provider] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None

    # Strategy Settings
    default_strategy: Optional[Strategy] = None
    strategy_config: Optional[StrategyConfig] = None

    # Verification Settings
    verification: Optional[VerificationConfig] = None

    # Runtime Settings
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    retry_delay: Optional[int] = None
    log_level: Optional[LogLevel] = None
```

### Provider

```python
from typing import Literal

Provider = Literal["claude", "openai", "ollama", "custom"]
```

### Strategy

```python
from typing import Literal

Strategy = Literal["auto", "gsd", "ralph", "rlm"]
```

### StrategyConfig

```python
from dataclasses import dataclass
from typing import Optional, List, Literal

@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    gsd: Optional[GSDConfig] = None
    ralph: Optional[RALPHConfig] = None
    rlm: Optional[RLMConfig] = None

@dataclass
class GSDConfig:
    """GSD strategy configuration."""
    max_tokens: Optional[int] = 1000        # Default: 1000
    temperature: Optional[float] = 0.3      # Default: 0.3
    direct_response: Optional[bool] = True  # Default: True
    skip_verification: Optional[bool] = False  # Default: False

@dataclass
class RALPHConfig:
    """RALPH strategy configuration."""
    max_iterations: Optional[int] = 5       # Default: 5
    temperature: Optional[float] = 0.5      # Default: 0.5
    phases: Optional[List[RALPHPhase]] = None  # Default: all phases
    enable_phase_logging: Optional[bool] = False  # Default: False
    min_confidence: Optional[float] = 0.7   # Default: 0.7

@dataclass
class RLMConfig:
    """RLM strategy configuration."""
    max_depth: Optional[int] = 10           # Default: 10
    temperature: Optional[float] = 0.7      # Default: 0.7
    reasoning_style: Optional[Literal["chain-of-thought", "tree-of-thought"]] = None
    enable_backtracking: Optional[bool] = True  # Default: True
    evaluation_threshold: Optional[float] = 0.8  # Default: 0.8
    max_iterations: Optional[int] = 15      # Default: 15
```

### VerificationConfig

```python
from dataclasses import dataclass
from typing import Optional, List, Callable

@dataclass
class VerificationConfig:
    """Verification configuration."""
    enabled: Optional[bool] = True          # Default: True
    checks: Optional[List[VerificationCheck]] = None
    strict_mode: Optional[bool] = False     # Default: False
    custom_checks: Optional[List[CustomCheck]] = None

VerificationCheck = Literal[
    "format",
    "completeness",
    "accuracy",
    "consistency",
    "safety"
]
```

### LogLevel

```python
from typing import Literal

LogLevel = Literal["debug", "info", "warn", "error", "silent"]
```

---

## Methods

### process()

Processes an input request and returns a result.

```python
async def process(self, input: ProcessInput) -> ProcessResult
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `ProcessInput` | Yes | The input to process |

#### ProcessInput

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class ProcessInput:
    """Input for process() method."""
    input: str
    context: Optional[Dict[str, Any]] = None
    strategy: Optional[Strategy] = None
    options: Optional[ProcessOptions] = None

@dataclass
class ProcessOptions:
    """Options for process() method."""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    verify_output: Optional[bool] = None
    include_metadata: Optional[bool] = None
    timeout: Optional[int] = None
```

#### ProcessResult

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ProcessResult:
    """Result from process() method."""
    output: str
    strategy: Strategy
    usage: TokenUsage
    metadata: Optional[ProcessMetadata] = None
    verification: Optional[VerificationResult] = None

@dataclass
class TokenUsage:
    """Token usage information."""
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class ProcessMetadata:
    """Metadata from process() method."""
    duration: int
    provider: str
    model: str
    phases: Optional[List[PhaseInfo]] = None
    reasoning_trace: Optional[List[ReasoningStep]] = None
```

#### Example

```python
result = await flow.process(
    ProcessInput(
        input="Explain machine learning",
        context={
            "audience": "beginners",
            "format": "bullet points"
        },
        strategy="ralph",
        options=ProcessOptions(
            max_tokens=2000,
            verify_output=True,
            include_metadata=True
        )
    )
)

print(result.output)
print(f"Strategy: {result.strategy}")
print(f"Tokens: {result.usage.total_tokens}")
print(f"Duration: {result.metadata.duration if result.metadata else 'N/A'}ms")
```

---

### stream()

Streams a response in real-time.

```python
def stream(self, input: ProcessInput) -> AsyncIterator[StreamChunk]
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `ProcessInput` | Yes | The input to process |

#### StreamChunk

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class StreamChunk:
    """A chunk from the stream() method."""
    content: str
    type: Literal["content", "metadata", "done"]
    metadata: Optional[ProcessMetadata] = None
    usage: Optional[TokenUsage] = None
```

#### Example

```python
import sys

stream = flow.stream(
    ProcessInput(
        input="Write a short story about space exploration",
        strategy="ralph"
    )
)

full_content = ""

async for chunk in stream:
    if chunk.type == "content":
        sys.stdout.write(chunk.content)
        sys.stdout.flush()
        full_content += chunk.content
    elif chunk.type == "metadata":
        print(f"\nMetadata: {chunk.metadata}")
    elif chunk.type == "done":
        print(f"\nTotal tokens: {chunk.usage.total_tokens if chunk.usage else 'N/A'}")
```

#### Stream with Cancellation

```python
import asyncio

# In Python, use asyncio.CancelledError for cancellation instead of AbortController.
# Create a task and cancel it after a timeout.

async def stream_with_timeout():
    try:
        stream = flow.stream(
            ProcessInput(input="Write a very long essay")
        )

        async for chunk in stream:
            sys.stdout.write(chunk.content)
            sys.stdout.flush()
    except asyncio.CancelledError:
        print("Stream cancelled")

# Run with timeout
async def main():
    task = asyncio.create_task(stream_with_timeout())

    # Cancel after 5 seconds
    await asyncio.sleep(5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
```

---

### analyze()

Analyzes an input without processing, returning routing information.

```python
async def analyze(self, input: AnalyzeInput) -> AnalyzeResult
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `AnalyzeInput` | Yes | The input to analyze |

#### AnalyzeInput

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AnalyzeInput:
    """Input for analyze() method."""
    input: str
    context: Optional[Dict[str, Any]] = None
```

#### AnalyzeResult

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AnalyzeResult:
    """Result from analyze() method."""
    recommended_strategy: Strategy
    confidence: float
    scores: Dict[Strategy, float]
    signals: List[RoutingSignal]
    reasoning: str
    estimated_tokens: int
    estimated_duration: int

@dataclass
class RoutingSignal:
    """A signal that influenced routing decision."""
    signal: str
    weight: float
    strategy: Strategy
```

#### Example

```python
analysis = await flow.analyze(
    AnalyzeInput(
        input="Design a real-time chat application architecture",
        context={
            "requirements": ["scalability", "low latency", "message persistence"]
        }
    )
)

print(f"Recommended: {analysis.recommended_strategy}")
print(f"Confidence: {analysis.confidence}")
print(f"Reasoning: {analysis.reasoning}")
print(f"Scores: {analysis.scores}")
# {'gsd': 0.12, 'ralph': 0.45, 'rlm': 0.89}
```

---

### configure()

Updates configuration at runtime.

```python
def configure(self, **config) -> None
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | `ContextFlowConfig` (partial) | Yes | Configuration to update |

#### Example

```python
import os

# Update provider
flow.configure(
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4"
)

# Update strategy defaults
flow.configure(
    strategy_config=StrategyConfig(
        ralph=RALPHConfig(max_iterations=7)
    )
)

# Update verification
flow.configure(
    verification=VerificationConfig(
        enabled=True,
        strict_mode=True
    )
)
```

---

### get_provider()

Returns the current provider instance.

```python
def get_provider(self) -> ProviderInstance
```

#### ProviderInstance

```python
from typing import Protocol, List

class ProviderInstance(Protocol):
    """Protocol for provider instances."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def is_available(self) -> bool: ...

    async def get_models(self) -> List[str]: ...

    def get_usage(self) -> TokenUsage: ...
```

#### Example

```python
provider = flow.get_provider()
print(f"Current provider: {provider.name}")
print(f"Model: {provider.model}")

available = await provider.is_available()
print(f"Available: {available}")

models = await provider.get_models()
print(f"Available models: {models}")
```

---

## Types

### Complete Type Definitions

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal, Protocol

# Core Types
Provider = Literal["claude", "openai", "ollama", "custom"]
Strategy = Literal["auto", "gsd", "ralph", "rlm"]
LogLevel = Literal["debug", "info", "warn", "error", "silent"]

# RALPH Phases
RALPHPhase = Literal["research", "analyze", "learn", "plan", "help"]

# Verification Checks
VerificationCheck = Literal[
    "format",
    "completeness",
    "accuracy",
    "consistency",
    "safety"
]

# Token Usage
@dataclass
class TokenUsage:
    """Token usage information."""
    input_tokens: int
    output_tokens: int
    total_tokens: int

# Process Input/Output
@dataclass
class ProcessInput:
    """Input for process() method."""
    input: str
    context: Optional[Dict[str, Any]] = None
    strategy: Optional[Strategy] = None
    options: Optional[ProcessOptions] = None

@dataclass
class ProcessResult:
    """Result from process() method."""
    output: str
    strategy: Strategy
    usage: TokenUsage
    metadata: Optional[ProcessMetadata] = None
    verification: Optional[VerificationResult] = None

# Stream Types
@dataclass
class StreamChunk:
    """A chunk from the stream() method."""
    content: str
    type: Literal["content", "metadata", "done"]
    metadata: Optional[ProcessMetadata] = None
    usage: Optional[TokenUsage] = None

# Verification Types
@dataclass
class VerificationResult:
    """Result from verification."""
    passed: bool
    checks: List[CheckResult]
    score: float
    issues: Optional[List[VerificationIssue]] = None

@dataclass
class CheckResult:
    """Result of a single verification check."""
    check: VerificationCheck
    passed: bool
    score: float
    message: Optional[str] = None

@dataclass
class VerificationIssue:
    """An issue found during verification."""
    severity: Literal["error", "warning", "info"]
    check: VerificationCheck
    message: str
    suggestion: Optional[str] = None

# Metadata Types
@dataclass
class ProcessMetadata:
    """Metadata from process() method."""
    duration: int
    provider: str
    model: str
    phases: Optional[List[PhaseInfo]] = None
    reasoning_trace: Optional[List[ReasoningStep]] = None

@dataclass
class PhaseInfo:
    """Information about a RALPH phase."""
    name: RALPHPhase
    duration: int
    tokens: int

@dataclass
class ReasoningStep:
    """A step in the reasoning trace."""
    index: int
    description: str
    confidence: float
    tokens: int
    outcome: Optional[str] = None
```

---

## Errors

### Error Classes

```python
class ContextFlowError(Exception):
    """Base error class for ContextFlow."""

    def __init__(self, message: str, code: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.code = code
        self.cause = cause


class ProviderError(ContextFlowError):
    """Provider-specific errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        retryable: bool = False
    ):
        super().__init__(message, code="PROVIDER_ERROR")
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ValidationError(ContextFlowError):
    """Validation errors."""

    def __init__(self, message: str, field: str, value: Any):
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field
        self.value = value


class StrategyError(ContextFlowError):
    """Strategy errors."""

    def __init__(self, message: str, strategy: Strategy, phase: Optional[str] = None):
        super().__init__(message, code="STRATEGY_ERROR")
        self.strategy = strategy
        self.phase = phase


class VerificationError(ContextFlowError):
    """Verification errors."""

    def __init__(self, message: str, result: VerificationResult):
        super().__init__(message, code="VERIFICATION_FAILED")
        self.result = result


class TimeoutError(ContextFlowError):
    """Timeout errors."""

    def __init__(self, message: str, timeout: int):
        super().__init__(message, code="TIMEOUT")
        self.timeout = timeout
```

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_CONFIG` | Configuration validation failed |
| `PROVIDER_ERROR` | Provider API error |
| `PROVIDER_UNAVAILABLE` | Provider not available |
| `RATE_LIMITED` | Rate limit exceeded |
| `TIMEOUT` | Request timed out |
| `STRATEGY_ERROR` | Strategy execution failed |
| `VERIFICATION_FAILED` | Verification checks failed |
| `INVALID_INPUT` | Invalid input provided |
| `CONTEXT_TOO_LARGE` | Context exceeds limits |

### Error Handling Example

```python
from contextflow import (
    ContextFlow,
    ContextFlowError,
    ProviderError,
    VerificationError,
    TimeoutError,
)

try:
    result = await flow.process(ProcessInput(input="..."))
except TimeoutError as error:
    print(f"Timeout after {error.timeout}ms")
except ProviderError as error:
    print(f"Provider {error.provider} error: {error}")
    if error.retryable:
        # Implement retry logic
        pass
except VerificationError as error:
    print(f"Verification failed: {error.result.issues}")
except ContextFlowError as error:
    print(f"ContextFlow error [{error.code}]: {error}")
except Exception:
    raise
```

---

## See Also

- [Getting Started](./getting-started.md)
- [Strategies](./strategies.md)
- [Providers](./providers.md)
- [Verification](./verification.md)
