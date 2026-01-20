"""
Token counting and cost estimation utilities.

Supports multiple tokenizers for different providers and includes
pricing information for accurate cost estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

# =============================================================================
# Pricing Data (USD per 1M tokens)
# =============================================================================

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    # Google Gemini
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
    "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    # Groq
    "mixtral-8x7b-32768": {"input": 0.27, "output": 0.27},
    "llama2-70b-4096": {"input": 0.70, "output": 0.80},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    # Mistral
    "mistral-large-latest": {"input": 4.0, "output": 12.0},
    "mistral-medium-latest": {"input": 2.7, "output": 8.1},
    "mistral-small-latest": {"input": 1.0, "output": 3.0},
    # Local models (free)
    "llama2": {"input": 0.0, "output": 0.0},
    "llama3": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
}

# Model context limits
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Claude
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000,
    # OpenAI
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Gemini
    "gemini-pro": 32_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    # Groq
    "mixtral-8x7b-32768": 32_768,
    "llama-3.1-70b-versatile": 131_072,
    # Mistral
    "mistral-large-latest": 32_000,
}


@dataclass
class TokenAnalysis:
    """Result of token analysis."""

    total_tokens: int
    estimated_cost_input: float
    estimated_cost_output: float
    estimated_cost_total: float
    recommended_strategy: str
    context_usage_percent: float
    warnings: list[str]


class TokenEstimator:
    """
    Token counting and cost estimation.

    Supports multiple tokenizer backends:
    - tiktoken (for OpenAI models)
    - Character-based estimation (fallback)

    Example:
        estimator = TokenEstimator()
        count = estimator.count_tokens("Hello, world!")
        cost = estimator.estimate_cost(count, "claude-3-sonnet-20240229")
    """

    # Tiktoken encoding for different model families
    ENCODING_MAP: dict[str, str] = {
        "gpt-4": "cl100k_base",
        "gpt-3.5": "cl100k_base",
        "text-embedding": "cl100k_base",
        "claude": "cl100k_base",  # Approximation
        "gemini": "cl100k_base",  # Approximation
        "mistral": "cl100k_base",  # Approximation
    }

    def __init__(self, default_encoding: str = "cl100k_base"):
        """
        Initialize token estimator.

        Args:
            default_encoding: Default tiktoken encoding to use
        """
        self._encoders: dict[str, tiktoken.Encoding] = {}
        self._default_encoding = default_encoding

    def _get_encoder(self, model: str | None = None) -> tiktoken.Encoding:
        """Get appropriate encoder for model."""
        encoding_name = self._default_encoding

        if model:
            # Try to get encoding for specific model
            try:
                return tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to family-based encoding
                for family, encoding in self.ENCODING_MAP.items():
                    if family in model.lower():
                        encoding_name = encoding
                        break

        # Cache encoders
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Optional model name for model-specific tokenization

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            encoder = self._get_encoder(model)
            return len(encoder.encode(text))
        except Exception:
            # Fallback: estimate ~4 characters per token
            return max(1, len(text) // 4)

    def count_tokens_batch(
        self, texts: list[str], model: str | None = None
    ) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(text, model) for text in texts]

    def count_messages_tokens(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> int:
        """
        Count tokens in a list of messages.

        Accounts for message formatting overhead.
        """
        total = 0
        encoder = self._get_encoder(model)

        for message in messages:
            # Add tokens for message structure (role, content markers)
            total += 4  # Overhead per message
            for key, value in message.items():
                total += len(encoder.encode(str(value)))

        total += 2  # Overhead for start/end
        return total

    def estimate_cost(
        self,
        tokens: int,
        model: str,
        is_input: bool = True,
    ) -> float:
        """
        Estimate cost in USD for given tokens.

        Args:
            tokens: Number of tokens
            model: Model name
            is_input: True for input tokens, False for output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0

        price_key = "input" if is_input else "output"
        price_per_million = pricing.get(price_key, 0.0)
        return (tokens / 1_000_000) * price_per_million

    def estimate_total_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Estimate total cost for input and output tokens."""
        input_cost = self.estimate_cost(input_tokens, model, is_input=True)
        output_cost = self.estimate_cost(output_tokens, model, is_input=False)
        return input_cost + output_cost

    def analyze(
        self,
        texts: list[str],
        model: str = "claude-3-5-sonnet-20241022",
        expected_output_ratio: float = 0.25,
    ) -> TokenAnalysis:
        """
        Comprehensive token analysis.

        Args:
            texts: List of texts to analyze
            model: Model to use for analysis
            expected_output_ratio: Expected output tokens as ratio of input

        Returns:
            TokenAnalysis with counts, costs, and recommendations
        """
        total_tokens = sum(self.count_tokens_batch(texts, model))
        expected_output = int(total_tokens * expected_output_ratio)

        input_cost = self.estimate_cost(total_tokens, model, is_input=True)
        output_cost = self.estimate_cost(expected_output, model, is_input=False)

        # Get context limit
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 100_000)
        context_usage = (total_tokens / context_limit) * 100

        # Determine recommended strategy
        if total_tokens < 10_000:
            strategy = "gsd_direct"
        elif total_tokens < 100_000:
            strategy = "ralph_structured"
        else:
            strategy = "rlm_full"

        # Generate warnings
        warnings = []
        if context_usage > 90:
            warnings.append(f"Context usage critical: {context_usage:.1f}%")
        elif context_usage > 75:
            warnings.append(f"Context usage high: {context_usage:.1f}%")

        if input_cost > 1.0:
            warnings.append(f"High estimated cost: ${input_cost + output_cost:.2f}")

        return TokenAnalysis(
            total_tokens=total_tokens,
            estimated_cost_input=input_cost,
            estimated_cost_output=output_cost,
            estimated_cost_total=input_cost + output_cost,
            recommended_strategy=strategy,
            context_usage_percent=context_usage,
            warnings=warnings,
        )

    def get_context_limit(self, model: str) -> int:
        """Get context limit for a model."""
        return MODEL_CONTEXT_LIMITS.get(model, 100_000)

    def get_pricing(self, model: str) -> dict[str, float] | None:
        """Get pricing for a model."""
        return MODEL_PRICING.get(model)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_estimator: TokenEstimator | None = None


def _get_estimator() -> TokenEstimator:
    """Get default estimator instance."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = TokenEstimator()
    return _default_estimator


def estimate_tokens(text: str, model: str | None = None) -> int:
    """
    Quick token estimation.

    Args:
        text: Text to estimate
        model: Optional model name

    Returns:
        Estimated token count
    """
    return _get_estimator().count_tokens(text, model)


def estimate_cost(tokens: int, model: str, is_input: bool = True) -> float:
    """
    Quick cost estimation.

    Args:
        tokens: Token count
        model: Model name
        is_input: Whether these are input tokens

    Returns:
        Estimated cost in USD
    """
    return _get_estimator().estimate_cost(tokens, model, is_input)
