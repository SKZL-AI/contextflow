"""
Google Gemini provider implementation.

Supports Gemini models with massive context windows (up to 1M tokens),
making it ideal for processing large documents and complex contexts.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import HarmBlockThreshold, HarmCategory
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


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider.

    Supports:
    - Gemini 1.5 Pro/Flash (1M token context)
    - Gemini 2.0 Flash Experimental
    - Gemini Pro (legacy, 32K context)
    - Streaming responses
    - System prompts via system_instruction
    - Native token counting

    Key Advantage: Massive context window (1M tokens) for processing
    entire codebases, long documents, or extensive conversation histories.

    Example:
        provider = GeminiProvider(model="gemini-1.5-pro")
        response = await provider.complete(
            messages=[Message(role="user", content="Analyze this codebase...")],
            system="You are a code review expert."
        )
        print(response.content)
    """

    # Model configurations with context and output limits
    MODELS: dict[str, dict[str, int]] = {
        "gemini-1.5-pro": {"context": 1_000_000, "output": 8_192},
        "gemini-1.5-pro-latest": {"context": 1_000_000, "output": 8_192},
        "gemini-1.5-flash": {"context": 1_000_000, "output": 8_192},
        "gemini-1.5-flash-latest": {"context": 1_000_000, "output": 8_192},
        "gemini-2.0-flash-exp": {"context": 1_000_000, "output": 8_192},
        "gemini-pro": {"context": 32_000, "output": 2_048},  # Legacy
    }

    # Pricing per 1M tokens (tiered for 1.5-pro based on context usage)
    # Standard pricing for <=128K context, premium for >128K
    PRICING: dict[str, dict[str, dict[str, float]]] = {
        "gemini-1.5-pro": {
            "standard": {"input": 1.25, "output": 5.00},  # <=128K context
            "premium": {"input": 2.50, "output": 10.00},  # >128K context
        },
        "gemini-1.5-pro-latest": {
            "standard": {"input": 1.25, "output": 5.00},
            "premium": {"input": 2.50, "output": 10.00},
        },
        "gemini-1.5-flash": {
            "standard": {"input": 0.075, "output": 0.30},
            "premium": {"input": 0.15, "output": 0.60},
        },
        "gemini-1.5-flash-latest": {
            "standard": {"input": 0.075, "output": 0.30},
            "premium": {"input": 0.15, "output": 0.60},
        },
        "gemini-2.0-flash-exp": {
            "standard": {"input": 0.0, "output": 0.0},  # Free during preview
            "premium": {"input": 0.0, "output": 0.0},
        },
        "gemini-pro": {
            "standard": {"input": 0.50, "output": 1.50},
            "premium": {"input": 0.50, "output": 1.50},
        },
    }

    # Context threshold for tiered pricing (128K tokens)
    CONTEXT_TIER_THRESHOLD: int = 128_000

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs: object,
    ):
        """
        Initialize Gemini provider.

        Args:
            model: Gemini model to use (default: gemini-1.5-pro)
            api_key: Google API key (or use GOOGLE_API_KEY env var)
            base_url: Optional custom API base URL (not commonly used)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration options
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Configure the Gemini SDK
        genai.configure(api_key=self.api_key)

        # Initialize the generative model with safety settings disabled
        self._safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self._model: genai.GenerativeModel | None = None
        self._current_system: str | None = None
        self.logger = ProviderLogger("gemini")

    def _get_model(self, system: str | None = None) -> genai.GenerativeModel:
        """
        Get or create a GenerativeModel instance.

        Args:
            system: Optional system instruction for the model

        Returns:
            Configured GenerativeModel instance
        """
        # Recreate model if system prompt changed
        if self._model is None or self._current_system != system:
            self._current_system = system
            self._model = genai.GenerativeModel(
                model_name=self.model,
                safety_settings=self._safety_settings,
                system_instruction=system if system else None,
            )
        return self._model

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "gemini"

    @property
    def provider_type(self) -> ProviderType:
        """Provider category."""
        return ProviderType.PROPRIETARY

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Provider capabilities and limits."""
        model_info = self.MODELS.get(self.model, self.MODELS["gemini-1.5-pro"])
        return ProviderCapabilities(
            max_context_tokens=model_info["context"],
            max_output_tokens=model_info["output"],
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=True,
            supported_models=list(self.MODELS.keys()),
            rate_limit_rpm=60,  # Varies by tier
            rate_limit_tpm=4_000_000,  # Up to 4M TPM for paid tier
            supports_batch_processing=False,
            supports_vision=True,
            latency_p50_ms=800,
            latency_p99_ms=2500,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert Message objects to Gemini format.

        Gemini uses:
        - "user" role for user messages
        - "model" role for assistant messages
        - System messages are passed via system_instruction parameter

        Args:
            messages: List of Message objects

        Returns:
            List of Gemini-formatted message dictionaries
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # System messages are handled via system_instruction
                continue
            elif msg.role == "assistant":
                converted.append({
                    "role": "model",
                    "parts": [{"text": msg.content}],
                })
            else:
                # user and any other roles
                converted.append({
                    "role": "user",
                    "parts": [{"text": msg.content}],
                })
        return converted

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """
        Calculate cost based on token usage and model.

        Uses tiered pricing for models that support it (1.5-pro/flash).

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(model, self.PRICING["gemini-1.5-pro"])

        # Determine pricing tier based on context usage
        if input_tokens > self.CONTEXT_TIER_THRESHOLD:
            tier = "premium"
        else:
            tier = "standard"

        tier_pricing = pricing[tier]
        cost = (
            (input_tokens / 1_000_000) * tier_pricing["input"]
            + (output_tokens / 1_000_000) * tier_pricing["output"]
        )
        return cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((google_exceptions.ServiceUnavailable,)),
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
        Execute Gemini completion.

        Args:
            messages: List of conversation messages
            system: Optional system prompt (passed as system_instruction)
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Gemini-specific parameters

        Returns:
            CompletionResponse with generated text and metadata

        Raises:
            ProviderAuthenticationError: On authentication failure
            ProviderRateLimitError: On rate limit exceeded
            ProviderTimeoutError: On request timeout
            ProviderError: On other API errors
        """
        use_model = model or self.model
        gemini_model = self._get_model(system)

        # Convert messages to Gemini format
        conv_messages = self._convert_messages(messages)

        # Estimate input tokens for logging
        input_text = " ".join(
            part.get("text", "")
            for msg in conv_messages
            for part in msg.get("parts", [])
        )
        input_tokens_estimate = self.count_tokens(input_text, use_model)
        self.logger.log_request(use_model, input_tokens_estimate)

        start_time = time.time()

        try:
            # Configure generation settings
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences or [],
            )

            # Execute async completion
            response = await gemini_model.generate_content_async(
                contents=conv_messages,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response content
            content = response.text if response.text else ""

            # Get usage metadata
            usage_metadata = response.usage_metadata
            usage_input = (
                usage_metadata.prompt_token_count if usage_metadata else input_tokens_estimate
            )
            usage_output = (
                usage_metadata.candidates_token_count if usage_metadata else self.count_tokens(content, use_model)
            )

            # Determine finish reason
            finish_reason = "stop"
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason_raw = str(response.candidates[0].finish_reason)
                if "MAX_TOKENS" in finish_reason_raw.upper():
                    finish_reason = "max_tokens"
                elif "SAFETY" in finish_reason_raw.upper():
                    finish_reason = "safety"

            # Calculate cost
            cost = self._calculate_cost(usage_input, usage_output, use_model)

            # Log response
            self.logger.log_response(use_model, usage_output, latency_ms, cost)

            return CompletionResponse(
                content=content,
                tokens_used=usage_input + usage_output,
                input_tokens=usage_input,
                output_tokens=usage_output,
                model=use_model,
                finish_reason=finish_reason,
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response={"text": content, "usage": {
                    "prompt_token_count": usage_input,
                    "candidates_token_count": usage_output,
                }},
            )

        except google_exceptions.PermissionDenied as e:
            self.logger.log_error(e)
            raise ProviderAuthenticationError("gemini") from e

        except google_exceptions.InvalidArgument as e:
            self.logger.log_error(e)
            if "api key" in str(e).lower():
                raise ProviderAuthenticationError("gemini") from e
            raise ProviderError("gemini", str(e)) from e

        except google_exceptions.ResourceExhausted as e:
            self.logger.log_error(e)
            raise ProviderRateLimitError("gemini") from e

        except google_exceptions.DeadlineExceeded as e:
            self.logger.log_error(e)
            raise ProviderTimeoutError("gemini", self.timeout) from e

        except google_exceptions.ServiceUnavailable as e:
            self.logger.log_error(e)
            raise ProviderError("gemini", "Service temporarily unavailable") from e

        except Exception as e:
            self.logger.log_error(e)
            raise ProviderError("gemini", str(e)) from e

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
        Stream Gemini completion.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects as they arrive

        Raises:
            ProviderAuthenticationError: On authentication failure
            ProviderRateLimitError: On rate limit exceeded
            ProviderError: On other API errors
        """
        use_model = model or self.model
        gemini_model = self._get_model(system)

        # Convert messages to Gemini format
        conv_messages = self._convert_messages(messages)

        try:
            # Configure generation settings
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Execute async streaming completion
            response = await gemini_model.generate_content_async(
                contents=conv_messages,
                generation_config=generation_config,
                stream=True,
            )

            chunk_index = 0
            async for chunk in response:
                if chunk.text:
                    yield StreamChunk(
                        content=chunk.text,
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

        except google_exceptions.PermissionDenied as e:
            raise ProviderAuthenticationError("gemini") from e

        except google_exceptions.InvalidArgument as e:
            if "api key" in str(e).lower():
                raise ProviderAuthenticationError("gemini") from e
            raise ProviderError("gemini", str(e)) from e

        except google_exceptions.ResourceExhausted as e:
            raise ProviderRateLimitError("gemini") from e

        except Exception as e:
            raise ProviderError("gemini", str(e)) from e

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens using Gemini's native token counter.

        Args:
            text: Text to count tokens for
            model: Optional model for model-specific tokenization

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            # Use Gemini's native token counter
            gemini_model = self._get_model()
            count_result = gemini_model.count_tokens(text)
            return count_result.total_tokens
        except Exception:
            # Fallback to estimation (~4 chars per token)
            return max(1, len(text) // 4)

    async def validate_credentials(self) -> bool:
        """
        Validate Google API credentials.

        Returns:
            True if credentials are valid

        Raises:
            ProviderAuthenticationError: If validation fails
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
            # Other errors (rate limit, etc.) still mean credentials are valid
            return True

    def get_model_info(self) -> dict[str, object]:
        """
        Get detailed information about the current model.

        Returns:
            Dictionary with model details including context window size
        """
        model_info = self.MODELS.get(self.model, self.MODELS["gemini-1.5-pro"])
        pricing = self.PRICING.get(self.model, self.PRICING["gemini-1.5-pro"])

        return {
            "provider": self.name,
            "model": self.model,
            "context_window": model_info["context"],
            "max_output_tokens": model_info["output"],
            "pricing": pricing,
            "capabilities": {
                "max_context_tokens": self.capabilities.max_context_tokens,
                "max_output_tokens": self.capabilities.max_output_tokens,
                "supports_streaming": self.capabilities.supports_streaming,
                "supports_system_prompt": self.capabilities.supports_system_prompt,
                "supports_vision": self.capabilities.supports_vision,
            },
            "notes": "1M token context window ideal for large documents and codebases",
        }
