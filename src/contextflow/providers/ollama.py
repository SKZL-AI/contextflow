"""
Ollama provider implementation for local LLM execution.

Supports local LLM inference through Ollama with models like Llama, Mistral,
Mixtral, CodeLlama, and others running on localhost.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextflow.core.types import (
    CompletionResponse,
    Message,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
)
from contextflow.providers.base import BaseProvider
from contextflow.utils.errors import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from contextflow.utils.logging import ProviderLogger


class OllamaProvider(BaseProvider):
    """
    Ollama provider for local LLM inference.

    Supports:
    - Llama 2, Llama 3, Llama 3.1, Llama 3.2
    - Mixtral, Mistral
    - CodeLlama, DeepSeek-Coder-v2
    - Qwen 2.5
    - Streaming responses
    - System prompts
    - No API key required (local service)

    Example:
        provider = OllamaProvider(model="llama3.2")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)

    Note:
        Requires Ollama to be running locally on localhost:11434.
        Install Ollama from https://ollama.ai and pull models with:
        `ollama pull llama3.2`
    """

    # Model configurations with context and output limits
    MODELS: dict[str, dict[str, int]] = {
        # Llama family
        "llama2": {"context": 4_096, "output": 4_096},
        "llama2:7b": {"context": 4_096, "output": 4_096},
        "llama2:13b": {"context": 4_096, "output": 4_096},
        "llama2:70b": {"context": 4_096, "output": 4_096},
        "llama3": {"context": 8_192, "output": 8_192},
        "llama3:8b": {"context": 8_192, "output": 8_192},
        "llama3:70b": {"context": 8_192, "output": 8_192},
        "llama3.1": {"context": 128_000, "output": 8_192},
        "llama3.1:8b": {"context": 128_000, "output": 8_192},
        "llama3.1:70b": {"context": 128_000, "output": 8_192},
        "llama3.1:405b": {"context": 128_000, "output": 8_192},
        "llama3.2": {"context": 128_000, "output": 8_192},
        "llama3.2:1b": {"context": 128_000, "output": 8_192},
        "llama3.2:3b": {"context": 128_000, "output": 8_192},
        # Mixtral family
        "mixtral": {"context": 32_768, "output": 4_096},
        "mixtral:8x7b": {"context": 32_768, "output": 4_096},
        "mixtral:8x22b": {"context": 65_536, "output": 4_096},
        # Mistral family
        "mistral": {"context": 32_768, "output": 4_096},
        "mistral:7b": {"context": 32_768, "output": 4_096},
        "mistral-nemo": {"context": 128_000, "output": 4_096},
        # CodeLlama family
        "codellama": {"context": 16_384, "output": 4_096},
        "codellama:7b": {"context": 16_384, "output": 4_096},
        "codellama:13b": {"context": 16_384, "output": 4_096},
        "codellama:34b": {"context": 16_384, "output": 4_096},
        "codellama:70b": {"context": 16_384, "output": 4_096},
        # Qwen family
        "qwen2.5": {"context": 128_000, "output": 8_192},
        "qwen2.5:0.5b": {"context": 32_768, "output": 8_192},
        "qwen2.5:1.5b": {"context": 32_768, "output": 8_192},
        "qwen2.5:3b": {"context": 32_768, "output": 8_192},
        "qwen2.5:7b": {"context": 128_000, "output": 8_192},
        "qwen2.5:14b": {"context": 128_000, "output": 8_192},
        "qwen2.5:32b": {"context": 128_000, "output": 8_192},
        "qwen2.5:72b": {"context": 128_000, "output": 8_192},
        "qwen2.5-coder": {"context": 128_000, "output": 8_192},
        # DeepSeek family
        "deepseek-coder-v2": {"context": 128_000, "output": 8_192},
        "deepseek-coder-v2:16b": {"context": 128_000, "output": 8_192},
        "deepseek-coder-v2:236b": {"context": 128_000, "output": 8_192},
    }

    # Default model configuration for unknown models
    DEFAULT_MODEL_CONFIG: dict[str, int] = {"context": 4_096, "output": 4_096}

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: object,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model to use (e.g., "llama3.2", "mixtral")
            api_key: Not used for Ollama (local service)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
        """
        # Ollama doesn't require API key
        super().__init__(
            model=model,
            api_key=None,
            base_url=base_url or "http://localhost:11434",
            timeout=timeout,
            max_retries=max_retries,
        )

        self.logger = ProviderLogger("ollama")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ollama"

    @property
    def provider_type(self) -> ProviderType:
        """Provider category."""
        return ProviderType.LOCAL

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and limits."""
        model_info = self.MODELS.get(self.model, self.DEFAULT_MODEL_CONFIG)
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=False,  # Ollama has limited tool support
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=None,  # No rate limits for local
            rate_limit_tpm=None,
            supports_batch_processing=False,
            supports_vision=False,  # Model-dependent, conservative default
            latency_p50_ms=2000.0,  # Local inference is typically slower
            latency_p99_ms=10000.0,
        )

    def _build_chat_payload(
        self,
        messages: list[dict[str, str]],
        model: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs: object,
    ) -> dict[str, Any]:
        """
        Build the payload for Ollama /api/chat endpoint.

        Args:
            messages: Converted message list
            model: Model name
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Dictionary payload for the API request.
        """
        # Prepend system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        return payload

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError,)),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: object,
    ) -> CompletionResponse:
        """
        Execute Ollama chat completion.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Ollama-specific parameters

        Returns:
            CompletionResponse with generated text and metadata.

        Raises:
            ProviderUnavailableError: When Ollama service is not running
            ProviderError: On model not found or other API errors
            ProviderTimeoutError: When request times out
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Log request
        input_text = " ".join(m["content"] for m in conv_messages)
        input_tokens = self.count_tokens(input_text, model)
        self.logger.log_request(model, input_tokens)

        start_time = time.time()

        try:
            client = await self._get_client()
            payload = self._build_chat_payload(
                messages=conv_messages,
                model=model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )

            # Add stop sequences if provided
            if stop_sequences:
                payload["options"]["stop"] = stop_sequences

            response = await client.post("/api/chat", json=payload)

            # Handle errors
            if response.status_code == 404:
                error_msg = (
                    f"Model '{model}' not found. " f"Pull it first with: ollama pull {model}"
                )
                raise ProviderError("ollama", error_msg, status_code=404)

            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000
            content = data.get("message", {}).get("content", "")

            # Ollama returns token counts in response
            output_tokens = data.get("eval_count", self.count_tokens(content, model))
            actual_input_tokens = data.get("prompt_eval_count", input_tokens)

            # Log response (cost is 0 for local execution)
            self.logger.log_response(model, output_tokens, latency_ms, 0.0)

            return CompletionResponse(
                content=content,
                tokens_used=actual_input_tokens + output_tokens,
                input_tokens=actual_input_tokens,
                output_tokens=output_tokens,
                model=model,
                finish_reason=data.get("done_reason", "stop"),
                cost_usd=0.0,  # Local execution is free
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.ConnectError as e:
            self.logger.log_error(e)
            raise ProviderUnavailableError(
                "ollama",
                details={
                    "message": (
                        "Cannot connect to Ollama. " "Ensure Ollama is running with: ollama serve"
                    ),
                    "base_url": self.base_url,
                },
            ) from e

        except httpx.TimeoutException as e:
            self.logger.log_error(e)
            raise ProviderTimeoutError("ollama", self.timeout) from e

        except httpx.HTTPStatusError as e:
            self.logger.log_error(e)
            raise ProviderError(
                "ollama",
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs: object,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream Ollama chat completion.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional Ollama-specific parameters

        Yields:
            StreamChunk objects as they arrive from Ollama.

        Raises:
            ProviderUnavailableError: When Ollama service is not running
            ProviderError: On model not found or other API errors
            ProviderTimeoutError: When request times out
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        try:
            client = await self._get_client()
            payload = self._build_chat_payload(
                messages=conv_messages,
                model=model,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            async with client.stream("POST", "/api/chat", json=payload) as response:
                # Handle errors
                if response.status_code == 404:
                    error_msg = (
                        f"Model '{model}' not found. " f"Pull it first with: ollama pull {model}"
                    )
                    raise ProviderError("ollama", error_msg, status_code=404)

                response.raise_for_status()

                chunk_index = 0
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract content from message
                    message = data.get("message", {})
                    content = message.get("content", "")

                    if content:
                        yield StreamChunk(
                            content=content,
                            is_final=False,
                            chunk_index=chunk_index,
                        )
                        chunk_index += 1

                    # Check if done
                    if data.get("done", False):
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            chunk_index=chunk_index,
                        )
                        break

        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                "ollama",
                details={
                    "message": (
                        "Cannot connect to Ollama. " "Ensure Ollama is running with: ollama serve"
                    ),
                    "base_url": self.base_url,
                },
            ) from e

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError("ollama", self.timeout) from e

        except httpx.HTTPStatusError as e:
            raise ProviderError(
                "ollama",
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count for text.

        Uses character-based estimation (~4 characters per token on average).
        This is an approximation as Ollama models use different tokenizers.

        Args:
            text: Text to count tokens for
            model: Optional model (not used, estimation is model-agnostic)

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Average ~4 characters per token for most LLMs
        return max(1, len(text) // 4)

    async def validate_credentials(self) -> bool:
        """
        Validate that Ollama service is available.

        Since Ollama is a local service, this checks connectivity
        rather than credentials.

        Returns:
            True if Ollama is reachable and responsive.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """
        List locally available Ollama models.

        Returns:
            List of model names available on the local Ollama instance.

        Raises:
            ProviderUnavailableError: When Ollama service is not running
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                "ollama",
                details={"message": "Cannot connect to Ollama service"},
            ) from e

    async def pull_model(self, model: str) -> AsyncIterator[dict[str, Any]]:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name to pull (e.g., "llama3.2")

        Yields:
            Progress updates during model download.

        Raises:
            ProviderUnavailableError: When Ollama service is not running
            ProviderError: On pull failure
        """
        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                "ollama",
                details={"message": "Cannot connect to Ollama service"},
            ) from e

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"OllamaProvider(model={self.model!r}, base_url={self.base_url!r})"

    async def __aenter__(self) -> OllamaProvider:
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
