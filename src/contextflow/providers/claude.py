"""
Anthropic Claude provider implementation.

Supports Claude 3 family models (Opus, Sonnet, Haiku) with streaming,
system prompts, and accurate token counting.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from anthropic import APIError, AsyncAnthropic, AuthenticationError, RateLimitError
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


class ClaudeProvider(BaseProvider):
    """
    Anthropic Claude provider.

    Supports:
    - Claude 3 Opus, Sonnet, Haiku
    - Claude 3.5 Sonnet
    - Streaming responses
    - System prompts
    - Long context (200K tokens)

    Example:
        provider = ClaudeProvider(model="claude-3-5-sonnet-20241022")
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            system="You are a helpful assistant."
        )
        print(response.content)
    """

    # Model configurations
    MODELS = {
        "claude-3-opus-20240229": {"context": 200_000, "output": 4_096},
        "claude-3-sonnet-20240229": {"context": 200_000, "output": 4_096},
        "claude-3-5-sonnet-20241022": {"context": 200_000, "output": 8_192},
        "claude-3-haiku-20240307": {"context": 200_000, "output": 4_096},
    }

    # Pricing per 1M tokens
    PRICING = {
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: object,
    ):
        """
        Initialize Claude provider.

        Args:
            model: Claude model to use
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.logger = ProviderLogger("claude")

    @property
    def name(self) -> str:
        return "claude"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.PROPRIETARY

    @property
    def capabilities(self) -> ProviderCapabilities:
        model_info = self.MODELS.get(self.model, self.MODELS["claude-3-5-sonnet-20241022"])
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=True,
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=1000,
            rate_limit_tpm=100_000,
            supports_batch_processing=True,
            supports_vision=True,
            latency_p50_ms=1200,
            latency_p99_ms=3000,
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
        """Execute Claude completion."""
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        # Log request
        input_text = " ".join(m["content"] for m in conv_messages)
        input_tokens = self.count_tokens(input_text, model)
        self.logger.log_request(model, input_tokens)

        start_time = time.time()

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences or [],
                system=system or "",
                messages=conv_messages,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text
            usage_input = response.usage.input_tokens
            usage_output = response.usage.output_tokens

            # Calculate cost
            pricing = self.PRICING.get(model, self.PRICING["claude-3-5-sonnet-20241022"])
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
                finish_reason=response.stop_reason or "end_turn",
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response=response.model_dump(),
            )

        except AuthenticationError as e:
            self.logger.log_error(e)
            raise ProviderAuthenticationError("claude") from e

        except RateLimitError as e:
            self.logger.log_error(e)
            raise ProviderRateLimitError("claude") from e

        except APIError as e:
            self.logger.log_error(e)
            if "timeout" in str(e).lower():
                raise ProviderTimeoutError("claude", self.timeout) from e
            raise ProviderError("claude", str(e)) from e

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
        """Stream Claude completion."""
        model = model or self.model
        conv_messages = self._convert_messages(messages)

        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system=system or "",
                messages=conv_messages,
            ) as stream:
                chunk_index = 0
                async for text in stream.text_stream:
                    yield StreamChunk(
                        content=text,
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
            raise ProviderAuthenticationError("claude") from e

        except RateLimitError as e:
            raise ProviderRateLimitError("claude") from e

        except APIError as e:
            raise ProviderError("claude", str(e)) from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens for Claude.

        Note: Anthropic doesn't expose a public tokenizer, so we use
        a character-based estimation (~4 chars per token on average).
        """
        if not text:
            return 0
        # Claude averages ~4 characters per token
        return max(1, len(text) // 4)

    async def validate_credentials(self) -> bool:
        """Validate Anthropic API credentials."""
        try:
            await self.complete(
                messages=[Message(role="user", content="Hi")],
                max_tokens=5,
            )
            return True
        except ProviderAuthenticationError:
            return False
        except Exception:
            # Other errors (rate limit, etc.) still mean credentials are valid
            return True
