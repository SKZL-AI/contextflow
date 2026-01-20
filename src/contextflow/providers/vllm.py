"""
vLLM provider implementation for high-throughput inference.

vLLM provides an OpenAI-compatible API for serving Hugging Face models
with optimizations like PagedAttention and continuous batching.
Ideal for self-hosted deployments requiring high throughput.
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


class VLLMProvider(BaseProvider):
    """
    vLLM provider for high-throughput LLM inference.

    vLLM offers an OpenAI-compatible API with optimizations for
    serving large language models efficiently. It supports:
    - Hugging Face model IDs (e.g., meta-llama/Llama-2-70b-chat-hf)
    - Continuous batching for high throughput
    - PagedAttention for efficient memory usage
    - Tensor parallelism for multi-GPU inference
    - SSE streaming responses

    Since vLLM is self-hosted, no API key is required and all
    inference is free (cost_usd = 0.0).

    Example:
        provider = VLLMProvider(
            model="meta-llama/Llama-2-70b-chat-hf",
            base_url="http://localhost:8000"
        )
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)

    Attributes:
        batch_size: Batch size for throughput optimization (info only).
        tensor_parallel_size: Number of GPUs for tensor parallelism (info only).
    """

    # Common vLLM-served models with their context limits
    MODELS = {
        "meta-llama/Llama-2-7b-chat-hf": {"context": 4_096, "output": 4_096},
        "meta-llama/Llama-2-13b-chat-hf": {"context": 4_096, "output": 4_096},
        "meta-llama/Llama-2-70b-chat-hf": {"context": 4_096, "output": 4_096},
        "meta-llama/Meta-Llama-3-8B-Instruct": {"context": 8_192, "output": 8_192},
        "meta-llama/Meta-Llama-3-70B-Instruct": {"context": 8_192, "output": 8_192},
        "mistralai/Mistral-7B-Instruct-v0.2": {"context": 32_768, "output": 32_768},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"context": 32_768, "output": 32_768},
        "codellama/CodeLlama-34b-Instruct-hf": {"context": 16_384, "output": 16_384},
        "default": {"context": 8_192, "output": 4_096},
    }

    # Token estimation: ~4 characters per token (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        batch_size: int = 1,
        tensor_parallel_size: int = 1,
        **kwargs: object,
    ):
        """
        Initialize vLLM provider.

        Args:
            model: Hugging Face model ID (e.g., meta-llama/Llama-2-70b-chat-hf).
            api_key: Optional API key (not required for self-hosted vLLM).
            base_url: vLLM server URL (default: http://localhost:8000).
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
            retry_backoff: Exponential backoff multiplier for retries.
            batch_size: Batch size for throughput (informational, server-side config).
            tensor_parallel_size: Number of GPUs for tensor parallelism (informational).
        """
        base_url = base_url or "http://localhost:8000"
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.logger = ProviderLogger("vllm")

        # HTTP client for API requests
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the HTTP client.

        Returns:
            Configured httpx AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "vllm"

    @property
    def provider_type(self) -> ProviderType:
        """Provider category."""
        return ProviderType.OPEN_SOURCE

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and limits."""
        model_info = self.MODELS.get(self.model, self.MODELS["default"])
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=False,  # Basic vLLM doesn't support function calling
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=None,  # Self-hosted, no rate limits
            rate_limit_tpm=None,
            supports_batch_processing=True,  # vLLM excels at batching
            supports_vision=False,  # Depends on model
            latency_p50_ms=500,  # Typically faster than cloud providers
            latency_p99_ms=2000,
        )

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
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        **kwargs: object,
    ) -> CompletionResponse:
        """
        Execute a completion request against vLLM server.

        Args:
            messages: List of conversation messages.
            system: Optional system prompt.
            model: Optional model override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 - 2.0).
            top_p: Nucleus sampling parameter.
            stop_sequences: Sequences that stop generation.
            top_k: Top-k sampling parameter (-1 to disable).
            repetition_penalty: Penalty for token repetition (1.0 = no penalty).
            **kwargs: Additional vLLM-specific parameters.

        Returns:
            CompletionResponse with generated text and metadata.

        Raises:
            ProviderUnavailableError: If vLLM server is not reachable.
            ProviderTimeoutError: If request times out.
            ProviderError: For other API errors.
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        # Log request
        input_text = " ".join(m["content"] for m in conv_messages)
        input_tokens = self.count_tokens(input_text, model)
        self.logger.log_request(model, input_tokens)

        start_time = time.time()

        # Build request payload (OpenAI-compatible format)
        payload: dict[str, Any] = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        # Add optional parameters
        if stop_sequences:
            payload["stop"] = stop_sequences
        if top_k > 0:
            payload["top_k"] = top_k
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty

        try:
            client = await self._get_client()
            response = await client.post("/v1/chat/completions", json=payload)

            # Handle HTTP errors
            if response.status_code >= 500:
                raise ProviderUnavailableError(
                    "vllm",
                    details={"status_code": response.status_code, "body": response.text},
                )

            if response.status_code >= 400:
                error_detail = response.text
                if "not found" in error_detail.lower() or "model" in error_detail.lower():
                    raise ProviderError(
                        "vllm",
                        f"Model not loaded: {model}",
                        status_code=response.status_code,
                        details={"model": model, "error": error_detail},
                    )
                raise ProviderError(
                    "vllm",
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract response data
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "stop")

            # Token usage from response or estimate
            usage = data.get("usage", {})
            usage_input = usage.get("prompt_tokens", input_tokens)
            usage_output = usage.get("completion_tokens", self.count_tokens(content, model))

            # Log response (cost is 0 for self-hosted)
            self.logger.log_response(model, usage_output, latency_ms, 0.0)

            return CompletionResponse(
                content=content,
                tokens_used=usage_input + usage_output,
                input_tokens=usage_input,
                output_tokens=usage_output,
                model=model,
                finish_reason=finish_reason,
                cost_usd=0.0,  # Self-hosted, no cost
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.ConnectError as e:
            self.logger.log_error(e)
            raise ProviderUnavailableError(
                "vllm",
                details={"base_url": self.base_url, "error": str(e)},
            ) from e

        except httpx.TimeoutException as e:
            self.logger.log_error(e)
            raise ProviderTimeoutError("vllm", self.timeout) from e

        except httpx.HTTPError as e:
            self.logger.log_error(e)
            raise ProviderError("vllm", f"HTTP error: {str(e)}") from e

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        **kwargs: object,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion response from vLLM server using SSE.

        Args:
            messages: List of conversation messages.
            system: Optional system prompt.
            model: Optional model override.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter (-1 to disable).
            repetition_penalty: Penalty for token repetition.
            **kwargs: Additional vLLM-specific parameters.

        Yields:
            StreamChunk objects as they arrive from the server.

        Raises:
            ProviderUnavailableError: If vLLM server is not reachable.
            ProviderError: For other API errors.
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        # Add optional parameters
        if top_k > 0:
            payload["top_k"] = top_k
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty

        try:
            client = await self._get_client()

            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    error_body = await response.aread()
                    if response.status_code >= 500:
                        raise ProviderUnavailableError(
                            "vllm",
                            details={"status_code": response.status_code},
                        )
                    raise ProviderError(
                        "vllm",
                        f"Stream error: {error_body.decode()}",
                        status_code=response.status_code,
                    )

                chunk_index = 0
                async for line in response.aiter_lines():
                    # SSE format: "data: {...}"
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    # Check for stream end
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            chunk_index=chunk_index,
                        )
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            yield StreamChunk(
                                content=content,
                                is_final=False,
                                chunk_index=chunk_index,
                            )
                            chunk_index += 1

                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

        except httpx.ConnectError as e:
            raise ProviderUnavailableError(
                "vllm",
                details={"base_url": self.base_url, "error": str(e)},
            ) from e

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError("vllm", self.timeout) from e

        except httpx.HTTPError as e:
            raise ProviderError("vllm", f"Stream error: {str(e)}") from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count using character-based approximation.

        Since vLLM serves various Hugging Face models, we use a conservative
        character-based estimation (~4 chars per token) for simplicity.
        For production use, consider loading the model's tokenizer.

        Args:
            text: Text to count tokens for.
            model: Optional model (not used in estimation).

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    async def validate_credentials(self) -> bool:
        """
        Validate connection to vLLM server.

        Since vLLM is self-hosted and typically doesn't require authentication,
        this method checks if the server is reachable and the model is loaded.

        Returns:
            True if server is reachable and model is available.
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> dict[str, object]:
        """
        Get information about the current model and vLLM configuration.

        Returns:
            Dictionary with model and server configuration details.
        """
        base_info = super().get_model_info()
        base_info.update({
            "batch_size": self.batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "base_url": self.base_url,
            "self_hosted": True,
        })
        return base_info

    async def __aenter__(self) -> VLLMProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"VLLMProvider(model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"tensor_parallel_size={self.tensor_parallel_size})"
        )
