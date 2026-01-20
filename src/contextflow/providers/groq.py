"""
Groq LPU provider implementation for ultra-fast inference.

Provides access to Groq's Language Processing Unit (LPU) hardware,
offering extremely low latency inference for open-source models
including Llama 3.1, Llama 3, Mixtral, and Gemma families.
"""

from __future__ import annotations

import json
import os
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
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from contextflow.utils.logging import ProviderLogger


class GroqProvider(BaseProvider):
    """
    Groq LPU provider for ultra-fast inference.

    Groq's Language Processing Unit (LPU) hardware provides extremely low
    latency inference, typically 100-300ms for most requests - significantly
    faster than traditional GPU-based inference.

    Supports:
    - Llama 3.1 (70B, 8B) with 128K context
    - Llama 3 (70B, 8B) with 8K context
    - Mixtral 8x7B with 32K context
    - Gemma (7B, 9B) with 8K context
    - OpenAI-compatible API
    - SSE Streaming responses
    - System prompts

    Example:
        provider = GroqProvider(model="llama-3.1-70b-versatile")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)
        print(f"Latency: {response.latency_ms:.0f}ms")  # Ultra-fast!

    Note:
        Groq API is OpenAI-compatible, using POST /chat/completions endpoint.
        Rate limits vary by model and account tier.
    """

    # Groq API base URL (OpenAI-compatible)
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    # Model configurations with context windows
    MODELS: dict[str, dict[str, int]] = {
        # Llama 3.1 series (128K context)
        "llama-3.1-70b-versatile": {"context": 128_000, "output": 8_192},
        "llama-3.1-8b-instant": {"context": 128_000, "output": 8_192},
        # Llama 3 series (8K context)
        "llama3-70b-8192": {"context": 8_192, "output": 8_192},
        "llama3-8b-8192": {"context": 8_192, "output": 8_192},
        # Mixtral series (32K context)
        "mixtral-8x7b-32768": {"context": 32_768, "output": 32_768},
        # Gemma series (8K context)
        "gemma-7b-it": {"context": 8_192, "output": 8_192},
        "gemma2-9b-it": {"context": 8_192, "output": 8_192},
    }

    # Pricing per 1M tokens (USD)
    PRICING: dict[str, dict[str, float]] = {
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama3-70b-8192": {"input": 0.59, "output": 0.79},
        "llama3-8b-8192": {"input": 0.05, "output": 0.08},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "gemma-7b-it": {"input": 0.07, "output": 0.07},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    }

    # Rate limits by model (requests per minute)
    RATE_LIMITS: dict[str, int] = {
        "llama-3.1-70b-versatile": 30,
        "llama-3.1-8b-instant": 30,
        "llama3-70b-8192": 30,
        "llama3-8b-8192": 30,
        "mixtral-8x7b-32768": 30,
        "gemma-7b-it": 30,
        "gemma2-9b-it": 30,
    }

    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        **kwargs: object,
    ) -> None:
        """
        Initialize Groq LPU provider.

        Args:
            model: Groq model to use (default: llama-3.1-70b-versatile)
            api_key: Groq API key (or use GROQ_API_KEY env var)
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum retry attempts (default: 3)
            retry_backoff: Exponential backoff multiplier (default: 1.5)
            **kwargs: Additional provider-specific options

        Raises:
            ProviderAuthenticationError: If API key is not provided and
                GROQ_API_KEY environment variable is not set.
        """
        api_key = api_key or os.getenv("GROQ_API_KEY")

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        self.logger = ProviderLogger("groq")

        # Initialize async HTTP client
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
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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
        return "groq"

    @property
    def provider_type(self) -> ProviderType:
        """Provider category (external API)."""
        return ProviderType.EXTERNAL_API

    @property
    def capabilities(self) -> ProviderCapabilities:
        """
        Provider capabilities and limits.

        Groq LPU provides ultra-low latency inference with typical
        response times of 100-300ms (p50).
        """
        model_info = self.MODELS.get(self.model, self.MODELS["llama-3.1-70b-versatile"])
        rate_limit = self.RATE_LIMITS.get(self.model, 30)

        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=False,  # Limited tool support currently
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=rate_limit,
            rate_limit_tpm=None,  # Varies by account tier
            supports_batch_processing=False,
            supports_vision=False,
            latency_p50_ms=200.0,  # Ultra-fast LPU inference
            latency_p99_ms=500.0,  # Still very fast at p99
        )

    def _build_request_payload(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Build the request payload for Groq API.

        Args:
            messages: Converted messages list
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            stream: Whether to stream the response

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        return payload

    def _handle_error_response(self, status_code: int, response_body: dict[str, Any]) -> None:
        """
        Handle error responses from Groq API.

        Args:
            status_code: HTTP status code
            response_body: Parsed response body

        Raises:
            ProviderAuthenticationError: On 401 status
            ProviderRateLimitError: On 429 status
            ProviderTimeoutError: On timeout-related errors
            ProviderError: On other API errors
        """
        error_message = response_body.get("error", {}).get("message", "Unknown error")

        if status_code == 401:
            self.logger.log_error(ProviderAuthenticationError("groq"), status_code=status_code)
            raise ProviderAuthenticationError("groq", details={"message": error_message})

        if status_code == 429:
            retry_after = response_body.get("error", {}).get("retry_after")
            self.logger.log_error(ProviderRateLimitError("groq"), status_code=status_code)
            raise ProviderRateLimitError(
                "groq", retry_after=retry_after, details={"message": error_message}
            )

        if status_code == 408 or "timeout" in error_message.lower():
            self.logger.log_error(
                ProviderTimeoutError("groq", self.timeout), status_code=status_code
            )
            raise ProviderTimeoutError("groq", self.timeout, details={"message": error_message})

        error = ProviderError("groq", error_message, status_code=status_code, details=response_body)
        self.logger.log_error(error, status_code=status_code)
        raise error

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a completion.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD.
        """
        pricing = self.PRICING.get(model, self.PRICING["llama-3.1-70b-versatile"])
        cost = (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]
        return cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
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
        Execute a completion request via Groq LPU.

        Sends a synchronous (non-streaming) request to the Groq API
        and returns the complete response.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override (defaults to instance model)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            top_p: Nucleus sampling parameter (default: 1.0)
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated text and metadata.

        Raises:
            ProviderAuthenticationError: If API key is invalid
            ProviderRateLimitError: If rate limit is exceeded
            ProviderTimeoutError: If request times out
            ProviderError: On other API errors
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided (OpenAI-compatible format)
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        # Log request
        input_text = " ".join(m["content"] for m in conv_messages)
        input_tokens_est = self.count_tokens(input_text, model)
        self.logger.log_request(model, input_tokens_est)

        # Build payload
        payload = self._build_request_payload(
            messages=conv_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stream=False,
        )

        start_time = time.time()
        client = await self._get_client()

        try:
            response = await client.post("/chat/completions", json=payload)
            latency_ms = (time.time() - start_time) * 1000

            # Handle errors
            if response.status_code != 200:
                self._handle_error_response(response.status_code, response.json())

            # Parse response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "stop")

            # Extract token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", input_tokens_est)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

            # Calculate cost
            cost = self._calculate_cost(model, input_tokens, output_tokens)

            # Log response
            self.logger.log_response(model, output_tokens, latency_ms, cost)

            return CompletionResponse(
                content=content,
                tokens_used=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                finish_reason=finish_reason,
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.TimeoutException as e:
            self.logger.log_error(e)
            raise ProviderTimeoutError("groq", self.timeout) from e

        except httpx.HTTPStatusError as e:
            self.logger.log_error(e)
            raise ProviderError("groq", str(e), status_code=e.response.status_code) from e

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
        Stream a completion response via Groq LPU.

        Uses Server-Sent Events (SSE) to stream tokens as they are
        generated by the Groq LPU for minimal time-to-first-token.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override (defaults to instance model)
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            top_p: Nucleus sampling parameter (default: 1.0)
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects as tokens arrive.

        Raises:
            ProviderAuthenticationError: If API key is invalid
            ProviderRateLimitError: If rate limit is exceeded
            ProviderError: On other API errors
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        # Build payload with streaming enabled
        payload = self._build_request_payload(
            messages=conv_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        client = await self._get_client()

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                # Check for error status
                if response.status_code != 200:
                    body = await response.aread()
                    self._handle_error_response(response.status_code, json.loads(body))

                chunk_index = 0

                # Process SSE stream
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Remove "data: " prefix
                    if line.startswith("data: "):
                        line = line[6:]

                    # Check for stream end
                    if line == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        delta = data["choices"][0].get("delta", {})
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

                # Yield final chunk
                yield StreamChunk(
                    content="",
                    is_final=True,
                    chunk_index=chunk_index,
                )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError("groq", self.timeout) from e

        except httpx.HTTPStatusError as e:
            raise ProviderError("groq", str(e), status_code=e.response.status_code) from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count for text.

        Groq does not provide a public tokenizer, so we use a character-based
        estimation of approximately 4 characters per token, which is a
        reasonable approximation for most LLM tokenizers.

        Args:
            text: Text to count tokens for
            model: Optional model identifier (unused, for interface compatibility)

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Average ~4 characters per token for LLM tokenizers
        return max(1, len(text) // 4)

    async def validate_credentials(self) -> bool:
        """
        Validate Groq API credentials.

        Sends a minimal completion request to verify the API key is valid.

        Returns:
            True if credentials are valid, False otherwise.
        """
        try:
            await self.complete(
                messages=[Message(role="user", content="Hi")],
                max_tokens=5,
            )
            return True
        except ProviderAuthenticationError:
            return False
        except Exception:
            # Other errors (rate limit, etc.) still indicate valid credentials
            return True

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"GroqProvider(model={self.model!r})"

    async def __aenter__(self) -> GroqProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit - closes HTTP client."""
        await self.close()
