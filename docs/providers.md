# Providers

ContextFlow supports multiple AI providers, allowing you to switch between them based on your needs.

## Supported Providers

| Provider | Status | Models | Streaming | Notes |
|----------|--------|--------|-----------|-------|
| **Claude** | Full Support | claude-sonnet-4-20250514, claude-3-5-sonnet, etc. | Yes | Recommended |
| **OpenAI** | Full Support | gpt-4, gpt-4-turbo, gpt-3.5-turbo | Yes | |
| **Ollama** | Full Support | llama2, mistral, codellama, etc. | Yes | Local/self-hosted |
| **Custom** | Full Support | Any | Configurable | For custom endpoints |

---

## Claude (Anthropic)

### Configuration

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-20250514"
)
```

### Available Models

| Model | Description | Max Tokens |
|-------|-------------|------------|
| `claude-opus-4-20250514` | Most capable, best for complex tasks | 200K |
| `claude-sonnet-4-20250514` | Balanced performance and speed | 200K |
| `claude-3-5-sonnet-20241022` | Previous generation Sonnet | 200K |
| `claude-3-5-haiku-20241022` | Fast, efficient for simple tasks | 200K |

### Environment Variables

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
ANTHROPIC_BASE_URL=https://api.anthropic.com  # Optional
```

### Advanced Configuration

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-20250514",
    provider_config={
        "claude": {
            "max_tokens": 4096,
            "anthropic_version": "2024-01-01",
            "default_headers": {
                "anthropic-beta": "some-feature"
            }
        }
    }
)
```

---

## OpenAI

### Configuration

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4-turbo"
)
```

### Available Models

| Model | Description | Max Tokens |
|-------|-------------|------------|
| `gpt-4-turbo` | Latest GPT-4, best performance | 128K |
| `gpt-4` | Standard GPT-4 | 8K |
| `gpt-4-32k` | Extended context GPT-4 | 32K |
| `gpt-3.5-turbo` | Fast, cost-effective | 16K |

### Environment Variables

```env
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...  # Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

### Advanced Configuration

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4-turbo",
    provider_config={
        "openai": {
            "organization": os.environ.get("OPENAI_ORG_ID"),
            "max_retries": 3,
            "timeout": 60000,
            "default_headers": {
                "OpenAI-Beta": "assistants=v2"
            }
        }
    }
)
```

### Azure OpenAI

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="openai",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    model="gpt-4",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    provider_config={
        "openai": {
            "api_version": "2024-02-01",
            "azure_deployment": "your-deployment-name"
        }
    }
)
```

---

## Ollama

### Configuration

```python
from contextflow import ContextFlow

flow = ContextFlow(
    provider="ollama",
    model="llama2",
    base_url="http://localhost:11434"
)
```

### Available Models

Ollama supports any model available in its library:

| Model | Description | Recommended For |
|-------|-------------|-----------------|
| `llama2` | Meta's Llama 2 | General use |
| `llama2:70b` | Llama 2 70B | Complex tasks |
| `mistral` | Mistral 7B | Balanced |
| `codellama` | Code-specialized | Coding tasks |
| `mixtral` | Mixtral 8x7B | High quality |

### Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Advanced Configuration

```python
from contextflow import ContextFlow

flow = ContextFlow(
    provider="ollama",
    model="mixtral",
    base_url="http://localhost:11434",
    provider_config={
        "ollama": {
            "num_ctx": 4096,        # Context window
            "num_gpu": 1,           # GPU layers
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
)
```

### Pulling Models

Before using a model, ensure it's pulled:

```bash
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

---

## Custom Provider

Implement your own provider for custom endpoints or unsupported services.

### Basic Custom Provider

```python
import os
from typing import AsyncIterator
import httpx
from contextflow import ContextFlow
from contextflow.providers.base import BaseProvider, CompletionResponse, StreamChunk

class MyCustomProvider(BaseProvider):
    """Custom provider implementation."""

    name = "my-provider"

    async def complete(self, request) -> CompletionResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://my-api.com/complete",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('MY_API_KEY')}"
                },
                json={
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResponse(
                content=data["text"],
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"]
                }
            )

    async def stream(self, request) -> AsyncIterator[StreamChunk]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://my-api.com/stream",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('MY_API_KEY')}"
                },
                json={
                    "prompt": request.prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield StreamChunk(content=chunk)


# Usage
custom_provider = MyCustomProvider()
flow = ContextFlow(
    provider="custom",
    custom_provider=custom_provider
)
```

### Full Custom Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass


@dataclass
class CompletionRequest:
    """Request for completion."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """Response from completion."""
    content: str
    usage: TokenUsage
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamChunk:
    """Chunk from streaming response."""
    content: str


class CustomProvider(ABC):
    """
    Abstract base class for custom providers.

    Implement this class to create your own provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Required: Completion method."""
        pass

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Optional: Streaming method."""
        raise NotImplementedError("Streaming not supported")

    async def is_available(self) -> bool:
        """Optional: Check availability."""
        return True

    async def get_models(self) -> List[str]:
        """Optional: List available models."""
        return []

    def count_tokens(self, text: str) -> int:
        """Optional: Token counting."""
        # Simple approximation: ~4 chars per token
        return len(text) // 4
```

### Custom Provider with All Methods

```python
import os
from typing import List, AsyncIterator
import httpx
from contextflow.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    TokenUsage
)


class FullCustomProvider(BaseProvider):
    """Full custom provider with all methods implemented."""

    name = "full-custom"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Implementation
        return CompletionResponse(
            content="...",
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        # Streaming implementation
        yield StreamChunk(content="chunk1")
        yield StreamChunk(content="chunk2")

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://my-api.com/health")
                return response.is_success
        except Exception:
            return False

    async def get_models(self) -> List[str]:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://my-api.com/models")
            response.raise_for_status()
            data = response.json()
            return data["models"]

    def count_tokens(self, text: str) -> int:
        # Simple approximation: ~4 chars per token
        return (len(text) + 3) // 4
```

---

## Provider Switching

### Runtime Provider Switching

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Switch to OpenAI at runtime
flow.configure(
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4-turbo"
)

# Switch to Ollama for local processing
flow.configure(
    provider="ollama",
    model="llama2",
    base_url="http://localhost:11434"
)
```

### Fallback Providers

```python
import os
from contextflow import ContextFlow, with_fallback

flow = with_fallback([
    {
        "provider": "claude",
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "model": "claude-sonnet-4-20250514"
    },
    {
        "provider": "openai",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4-turbo"
    },
    {
        "provider": "ollama",
        "model": "llama2"
    }
])

# Will try providers in order if one fails
result = await flow.process(
    input="Process this request"
)
```

---

## Provider Comparison

### Feature Matrix

| Feature | Claude | OpenAI | Ollama |
|---------|--------|--------|--------|
| Streaming | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Limited |
| Vision | Yes | Yes | Limited |
| Max Context | 200K | 128K | Varies |
| Local Deployment | No | No | Yes |
| Rate Limits | Yes | Yes | No |

### Performance Tips

1. **Claude**: Best for complex reasoning and long contexts
2. **OpenAI**: Best for general tasks and function calling
3. **Ollama**: Best for privacy-sensitive data and offline use

---

## See Also

- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
- [Configuration](./getting-started.md#configuration)
