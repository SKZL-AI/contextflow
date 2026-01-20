"""
OpenAI GPT provider implementation.

Supports GPT-4, GPT-4 Turbo, GPT-4o, and GPT-3.5 models with streaming,
system prompts, and accurate token counting via tiktoken.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

import tiktoken
from openai import APIError, AsyncOpenAI, AuthenticationError, RateLimitError
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


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT provider.

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini
    - GPT-3.5 Turbo
    - Streaming responses
    - System prompts
    - Function/tool calling
    - Accurate token counting via tiktoken

    Example:
        provider = OpenAIProvider(model="gpt-4o")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)
    """

    # Model configurations
    MODELS = {
        "gpt-4-turbo": {"context": 128_000, "output": 4_096},
        "gpt-4-turbo-preview": {"context": 128_000, "output": 4_096},
        "gpt-4o": {"context": 128_000, "output": 4_096},
        "gpt-4o-mini": {"context": 128_000, "output": 16_384},
        "gpt-4": {"context": 8_192, "output": 4_096},
        "gpt-3.5-turbo": {"context": 16_385, "output": 4_096},
    }

    # Pricing per 1M tokens
    PRICING = {
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: object,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            base_url: Optional custom API base URL (for Azure, etc.)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.logger = ProviderLogger("openai")

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.PROPRIETARY

    @property
    def capabilities(self) -> ProviderCapabilities:
        model_info = self.MODELS.get(self.model, self.MODELS["gpt-4o"])
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=True,
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=3_500,
            rate_limit_tpm=90_000,
            supports_batch_processing=True,
            supports_vision=True,
            latency_p50_ms=800,
            latency_p99_ms=2000,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIError,)),
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
        """Execute OpenAI completion."""
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

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=conv_messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            usage_input = response.usage.prompt_tokens if response.usage else 0
            usage_output = response.usage.completion_tokens if response.usage else 0

            # Calculate cost
            pricing = self.PRICING.get(model, self.PRICING["gpt-4o"])
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
                finish_reason=response.choices[0].finish_reason or "stop",
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response=response.model_dump(),
            )

        except AuthenticationError as e:
            self.logger.log_error(e)
            raise ProviderAuthenticationError("openai") from e

        except RateLimitError as e:
            self.logger.log_error(e)
            raise ProviderRateLimitError("openai") from e

        except APIError as e:
            self.logger.log_error(e)
            if "timeout" in str(e).lower():
                raise ProviderTimeoutError("openai", self.timeout) from e
            raise ProviderError("openai", str(e)) from e

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
        """Stream OpenAI completion."""
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Prepend system message if provided
        if system:
            conv_messages.insert(0, {"role": "system", "content": system})

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=conv_messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            chunk_index = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1

            # Final chunk
            yield StreamChunk(
                content="",
                is_final=True,
                chunk_index=chunk_index,
            )

        except AuthenticationError as e:
            raise ProviderAuthenticationError("openai") from e

        except RateLimitError as e:
            raise ProviderRateLimitError("openai") from e

        except APIError as e:
            raise ProviderError("openai", str(e)) from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Text to count tokens for
            model: Optional model (uses instance tokenizer if not provided)

        Returns:
            Exact token count
        """
        if not text:
            return 0

        if model and model != self.model:
            try:
                encoder = tiktoken.encoding_for_model(model)
                return len(encoder.encode(text))
            except KeyError:
                pass

        return len(self.tokenizer.encode(text))

    async def validate_credentials(self) -> bool:
        """Validate OpenAI API credentials."""
        try:
            await self.complete(
                messages=[Message(role="user", content="Hi")],
                max_tokens=5,
            )
            return True
        except ProviderAuthenticationError:
            return False
        except Exception:
            return True
