"""
Mistral AI provider implementation.

Supports Mistral AI models with OpenAI-compatible API, streaming,
JSON mode, and EU-based data processing.
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


class MistralProvider(BaseProvider):
    """
    Mistral AI provider.

    Supports:
    - Mistral Large, Medium (deprecated), Small
    - Open Mixtral models (8x7B, 8x22B)
    - Codestral (code-optimized)
    - Ministral models (3B, 8B) with 128K context
    - Streaming responses via SSE
    - JSON mode (response_format)
    - EU-based data processing
    - Safe mode option

    The Mistral API is OpenAI-compatible, using the /chat/completions endpoint.

    Example:
        provider = MistralProvider(model="mistral-large-latest")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)

        # With JSON mode
        response = await provider.complete(
            messages=[Message(role="user", content="Return a JSON object with name and age.")],
            json_mode=True
        )
    """

    # API Configuration
    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    # Model configurations with context windows
    MODELS: dict[str, dict[str, int]] = {
        "mistral-large-latest": {"context": 32_000, "output": 8_192},
        "mistral-medium-latest": {"context": 32_000, "output": 8_192},  # deprecated
        "mistral-small-latest": {"context": 32_000, "output": 8_192},
        "open-mixtral-8x7b": {"context": 32_000, "output": 8_192},
        "open-mixtral-8x22b": {"context": 64_000, "output": 8_192},
        "codestral-latest": {"context": 32_000, "output": 8_192},
        "ministral-8b-latest": {"context": 128_000, "output": 8_192},
        "ministral-3b-latest": {"context": 128_000, "output": 8_192},
    }

    # Pricing per 1M tokens (input, output)
    PRICING: dict[str, dict[str, float]] = {
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-medium-latest": {"input": 2.70, "output": 8.10},  # estimated
        "mistral-small-latest": {"input": 0.20, "output": 0.60},
        "open-mixtral-8x7b": {"input": 0.70, "output": 0.70},
        "open-mixtral-8x22b": {"input": 2.00, "output": 6.00},
        "codestral-latest": {"input": 0.30, "output": 0.90},
        "ministral-8b-latest": {"input": 0.10, "output": 0.10},
        "ministral-3b-latest": {"input": 0.04, "output": 0.04},
    }

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        safe_mode: bool = False,
        eu_data_processing: bool = True,
        **kwargs: object,
    ):
        """
        Initialize Mistral AI provider.

        Args:
            model: Mistral model to use (e.g., "mistral-large-latest")
            api_key: Mistral API key (or use MISTRAL_API_KEY env var)
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier for retries
            safe_mode: Enable Mistral's safe mode for content filtering
            eu_data_processing: Flag for EU-based data processing (metadata only)
        """
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        base_url = base_url or self.DEFAULT_BASE_URL

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        self.safe_mode = safe_mode
        self.eu_data_processing = eu_data_processing
        self.logger = ProviderLogger("mistral")

        # Initialize async HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "mistral"

    @property
    def provider_type(self) -> ProviderType:
        """Provider category."""
        return ProviderType.EXTERNAL_API

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and limits."""
        model_info = self.MODELS.get(self.model, self.MODELS["mistral-large-latest"])
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=True,
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=None,  # Depends on subscription tier
            rate_limit_tpm=None,
            supports_batch_processing=False,
            supports_vision=False,  # Mistral doesn't support vision yet
            latency_p50_ms=900,
            latency_p99_ms=2500,
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
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Build the API request payload.

        Args:
            messages: Converted messages list
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Stop sequences
            stream: Whether to stream the response
            json_mode: Whether to enable JSON mode

        Returns:
            Request payload dictionary
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

        if self.safe_mode:
            payload["safe_prompt"] = True

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def _handle_error_response(
        self,
        status_code: int,
        response_body: dict[str, Any],
    ) -> None:
        """
        Handle error responses from the API.

        Args:
            status_code: HTTP status code
            response_body: Parsed response body

        Raises:
            ProviderAuthenticationError: For 401 errors
            ProviderRateLimitError: For 429 errors
            ProviderTimeoutError: For timeout-related errors
            ProviderError: For other errors
        """
        error_message = response_body.get("message", "Unknown error")
        error_details = response_body.get("error", {})

        if isinstance(error_details, dict):
            error_message = error_details.get("message", error_message)

        if status_code == 401:
            raise ProviderAuthenticationError("mistral", details={"response": response_body})

        if status_code == 429:
            retry_after = None
            if "retry_after" in response_body:
                retry_after = int(response_body["retry_after"])
            raise ProviderRateLimitError(
                "mistral",
                retry_after=retry_after,
                details={"response": response_body},
            )

        if status_code == 408 or "timeout" in error_message.lower():
            raise ProviderTimeoutError(
                "mistral",
                self.timeout,
                details={"response": response_body},
            )

        raise ProviderError(
            "mistral",
            f"API error ({status_code}): {error_message}",
            status_code=status_code,
            details={"response": response_body},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
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
        json_mode: bool = False,
        **kwargs: object,
    ) -> CompletionResponse:
        """
        Execute a Mistral AI completion request.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            json_mode: Enable JSON mode for structured output
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated text and metadata

        Raises:
            ProviderAuthenticationError: On auth failures
            ProviderRateLimitError: On rate limit exceeded
            ProviderTimeoutError: On request timeout
            ProviderError: On other API errors
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
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
            json_mode=json_mode,
        )

        start_time = time.time()

        try:
            client = await self._get_client()
            response = await client.post("/chat/completions", json=payload)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                try:
                    error_body = response.json()
                except json.JSONDecodeError:
                    error_body = {"message": response.text}
                self._handle_error_response(response.status_code, error_body)

            data = response.json()

            # Extract response data
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "stop")
            usage = data.get("usage", {})
            usage_input = usage.get("prompt_tokens", input_tokens_est)
            usage_output = usage.get("completion_tokens", self.count_tokens(content))

            # Calculate cost
            pricing = self.PRICING.get(model, self.PRICING["mistral-large-latest"])
            cost = (usage_input / 1_000_000) * pricing["input"] + (
                usage_output / 1_000_000
            ) * pricing["output"]

            # Log response
            self.logger.log_response(model, usage_output, latency_ms, cost)

            return CompletionResponse(
                content=content,
                tokens_used=usage_input + usage_output,
                input_tokens=usage_input,
                output_tokens=usage_output,
                model=model,
                finish_reason=finish_reason,
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.TimeoutException as e:
            self.logger.log_error(e)
            raise ProviderTimeoutError("mistral", self.timeout) from e

        except httpx.TransportError as e:
            self.logger.log_error(e)
            raise ProviderError("mistral", f"Network error: {str(e)}") from e

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        json_mode: bool = False,
        **kwargs: object,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a Mistral AI completion response via SSE.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            json_mode: Enable JSON mode for structured output
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects as they arrive

        Raises:
            ProviderAuthenticationError: On auth failures
            ProviderRateLimitError: On rate limit exceeded
            ProviderError: On other API errors
        """
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        # Build payload with stream=True
        payload = self._build_request_payload(
            messages=conv_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            json_mode=json_mode,
        )

        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    try:
                        error_body = json.loads(error_text)
                    except json.JSONDecodeError:
                        error_body = {"message": error_text.decode()}
                    self._handle_error_response(response.status_code, error_body)

                chunk_index = 0
                async for line in response.aiter_lines():
                    # Skip empty lines and SSE comments
                    if not line or line.startswith(":"):
                        continue

                    # Parse SSE data
                    if line.startswith("data: "):
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
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield StreamChunk(
                                        content=content,
                                        is_final=False,
                                        chunk_index=chunk_index,
                                    )
                                    chunk_index += 1
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks
                            continue

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError("mistral", self.timeout) from e

        except httpx.TransportError as e:
            raise ProviderError("mistral", f"Network error: {str(e)}") from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count for Mistral models.

        Mistral doesn't expose a public tokenizer, so we use a character-based
        estimation of approximately 4 characters per token on average.

        Args:
            text: Text to count tokens for
            model: Optional model (unused, kept for interface compatibility)

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Average ~4 characters per token for Mistral models
        return max(1, len(text) // 4)

    async def validate_credentials(self) -> bool:
        """
        Validate Mistral API credentials.

        Returns:
            True if credentials are valid, False otherwise
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
            # Other errors (rate limit, network issues) still mean credentials may be valid
            return True

    def get_model_info(self) -> dict[str, object]:
        """
        Get detailed information about the current model.

        Returns:
            Dictionary with model details including Mistral-specific info
        """
        base_info = super().get_model_info()
        base_info["mistral_specific"] = {
            "safe_mode": self.safe_mode,
            "eu_data_processing": self.eu_data_processing,
            "supports_json_mode": True,
            "is_code_optimized": "codestral" in self.model.lower(),
        }
        return base_info

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"MistralProvider(model={self.model!r}, "
            f"safe_mode={self.safe_mode}, "
            f"eu_data_processing={self.eu_data_processing})"
        )

    async def __aenter__(self) -> MistralProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit - close the client."""
        await self.close()
