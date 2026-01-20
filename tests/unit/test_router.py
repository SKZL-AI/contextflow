"""
Unit tests for StrategyRouter.

Tests the strategy routing decision matrix including:
- Strategy selection for different token counts
- Density and complexity estimation
- Fallback strategies
- Cost estimation
- Router configuration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from contextflow.core.router import (
    StrategyRouter,
    RouterConfig,
    ContextAnalysis,
    ComplexityLevel,
    auto_route,
    analyze_context,
    get_recommended_strategy,
)
from contextflow.strategies.base import StrategyType, StrategyResult

# Note: StrategyType is imported from strategies.base as that's where
# the router module imports it from (it has the GUIDED variants)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider for router tests."""
    provider = MagicMock()
    provider.name = "mock-router"
    provider.model = "mock-model"
    provider.complete = AsyncMock()
    provider.count_tokens = MagicMock(return_value=1000)
    return provider


@pytest.fixture
def default_router(mock_provider: MagicMock) -> StrategyRouter:
    """Create a StrategyRouter with default configuration."""
    return StrategyRouter(provider=mock_provider)


@pytest.fixture
def custom_config() -> RouterConfig:
    """Create custom router configuration."""
    return RouterConfig(
        gsd_max_tokens=15_000,
        ralph_min_tokens=15_000,
        ralph_max_tokens=80_000,
        rlm_min_tokens=80_000,
        density_threshold_low=0.4,
        density_threshold_high=0.8,
    )


@pytest.fixture
def small_context() -> str:
    """Create small context text (~1000 tokens)."""
    return "This is a small context. " * 250  # ~1000 tokens


@pytest.fixture
def medium_context() -> str:
    """Create medium context text (~20K tokens)."""
    return "This is medium-sized context with various content. " * 5000  # ~20K tokens


@pytest.fixture
def large_context() -> str:
    """Create large context text (~75K tokens)."""
    return "This is a large context with substantial content. " * 20000  # ~75K tokens


@pytest.fixture
def very_large_context() -> str:
    """Create very large context text (~150K tokens)."""
    return "This is a very large context requiring RLM processing. " * 40000  # ~150K tokens


# =============================================================================
# Router Configuration Tests
# =============================================================================


class TestRouterConfig:
    """Tests for RouterConfig."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = RouterConfig()

        assert config.gsd_max_tokens == 10_000
        assert config.ralph_min_tokens == 10_000
        assert config.ralph_max_tokens == 100_000
        assert config.rlm_min_tokens == 100_000
        assert config.density_threshold_low == 0.5
        assert config.density_threshold_high == 0.7
        assert config.enable_fallback is True
        assert config.enable_cost_estimation is True

    def test_custom_config_values(self) -> None:
        """Test custom configuration values."""
        config = RouterConfig(
            gsd_max_tokens=15_000,
            density_threshold_high=0.8,
            enable_fallback=False
        )

        assert config.gsd_max_tokens == 15_000
        assert config.density_threshold_high == 0.8
        assert config.enable_fallback is False

    def test_config_to_dict(self) -> None:
        """Test configuration serialization to dictionary."""
        config = RouterConfig()
        config_dict = config.to_dict()

        assert "gsd_max_tokens" in config_dict
        assert "ralph_max_tokens" in config_dict
        assert "density_threshold_low" in config_dict
        assert isinstance(config_dict["gsd_max_tokens"], int)


# =============================================================================
# Router Initialization Tests
# =============================================================================


class TestRouterInitialization:
    """Tests for StrategyRouter initialization."""

    def test_init_with_provider(self, mock_provider: MagicMock) -> None:
        """Test router initialization with provider."""
        router = StrategyRouter(provider=mock_provider)

        assert router.provider == mock_provider
        assert router.config is not None

    def test_init_with_custom_config(
        self,
        mock_provider: MagicMock,
        custom_config: RouterConfig
    ) -> None:
        """Test router initialization with custom configuration."""
        router = StrategyRouter(
            provider=mock_provider,
            config=custom_config
        )

        assert router.config == custom_config
        assert router.config.gsd_max_tokens == 15_000

    def test_router_repr(self, default_router: StrategyRouter) -> None:
        """Test router string representation."""
        repr_str = repr(default_router)

        assert "StrategyRouter" in repr_str
        assert "mock-router" in repr_str


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for strategy selection based on token count."""

    def test_small_context_selects_gsd(
        self,
        mock_provider: MagicMock,
        small_context: str
    ) -> None:
        """Test that small context (<10K) selects GSD strategy."""
        # Configure mock to return small token count
        mock_provider.count_tokens = MagicMock(return_value=5000)

        router = StrategyRouter(provider=mock_provider)
        analysis = router.analyze(
            task="Summarize this text",
            context=small_context
        )

        assert analysis.recommended_strategy in [
            StrategyType.GSD_DIRECT,
            StrategyType.GSD_GUIDED
        ]

    def test_medium_context_selects_ralph(
        self,
        mock_provider: MagicMock,
        medium_context: str
    ) -> None:
        """Test that medium context (10K-100K) selects RALPH strategy."""
        mock_provider.count_tokens = MagicMock(return_value=30000)

        router = StrategyRouter(provider=mock_provider)
        analysis = router.analyze(
            task="Analyze this document",
            context=medium_context
        )

        assert analysis.recommended_strategy in [
            StrategyType.RALPH_ITERATIVE,
            StrategyType.RALPH_STRUCTURED
        ]

    def test_large_context_selects_rlm(
        self,
        mock_provider: MagicMock,
        very_large_context: str
    ) -> None:
        """Test that large context (>100K) selects RLM strategy."""
        mock_provider.count_tokens = MagicMock(return_value=150000)

        router = StrategyRouter(provider=mock_provider)
        analysis = router.analyze(
            task="Process this large document",
            context=very_large_context
        )

        assert analysis.recommended_strategy in [
            StrategyType.RLM_BASIC,
            StrategyType.RLM_FULL
        ]


# =============================================================================
# Density Estimation Tests
# =============================================================================


class TestDensityEstimation:
    """Tests for information density estimation."""

    def test_sparse_content_low_density(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that sparse content has low density score."""
        # Lots of whitespace and repetition
        sparse_content = """


        This   is   very   sparse   content.



        With   lots   of   whitespace.



        """

        router = StrategyRouter(provider=mock_provider)
        density = router._estimate_density(sparse_content)

        assert density < 0.5

    def test_dense_code_high_density(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that code content has higher density."""
        code_content = """
def process_data(items: List[Dict]) -> List[str]:
    results = []
    for item in items:
        if item.get("valid") and item["value"] > 0:
            processed = transform(item["value"])
            results.append(f"Result: {processed}")
    return results
"""

        router = StrategyRouter(provider=mock_provider)
        density = router._estimate_density(code_content)

        # Code should be moderately to highly dense
        assert density > 0.3

    def test_structured_content_moderate_density(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that structured content (tables, lists) has moderate density."""
        structured_content = """
| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

- Item one with description
- Item two with description
- Item three with description
"""

        router = StrategyRouter(provider=mock_provider)
        density = router._estimate_density(structured_content)

        assert 0.3 < density < 0.9

    def test_empty_content_zero_density(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that empty content has zero density."""
        router = StrategyRouter(provider=mock_provider)
        density = router._estimate_density("")

        assert density == 0.0


# =============================================================================
# Complexity Assessment Tests
# =============================================================================


class TestComplexityAssessment:
    """Tests for task complexity assessment."""

    def test_simple_task_low_complexity(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that simple tasks have low complexity."""
        simple_tasks = [
            "What is the capital of France?",
            "List the items in this document",
            "Find the date mentioned",
            "Define the term 'photosynthesis'",
        ]

        router = StrategyRouter(provider=mock_provider)

        for task in simple_tasks:
            complexity = router._assess_complexity(task)
            assert complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]

    def test_complex_task_high_complexity(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that complex tasks have high complexity."""
        complex_tasks = [
            "Analyze and compare the different approaches, evaluating their pros and cons",
            "Synthesize the findings and explain the implications for future research",
            "Examine the patterns and demonstrate the relationship between causes and effects",
        ]

        router = StrategyRouter(provider=mock_provider)

        for task in complex_tasks:
            complexity = router._assess_complexity(task)
            assert complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.EXHAUSTIVE]

    def test_exhaustive_task_keywords(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that exhaustive keywords trigger high complexity."""
        exhaustive_task = "Provide a comprehensive and exhaustive analysis covering all aspects, leaving nothing out"

        router = StrategyRouter(provider=mock_provider)
        complexity = router._assess_complexity(exhaustive_task)

        assert complexity == ComplexityLevel.EXHAUSTIVE

    def test_multipart_question_increases_complexity(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that multiple questions increase complexity."""
        multipart = "What is the main topic? How does it relate to the secondary theme? Why is this significant?"

        router = StrategyRouter(provider=mock_provider)
        complexity = router._assess_complexity(multipart)

        assert complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]


# =============================================================================
# Alternative Strategies Tests
# =============================================================================


class TestAlternativeStrategies:
    """Tests for alternative strategy generation."""

    def test_alternatives_for_small_context(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test alternative strategies for small context."""
        mock_provider.count_tokens = MagicMock(return_value=5000)

        router = StrategyRouter(provider=mock_provider)
        alternatives = router._get_alternative_strategies(
            StrategyType.GSD_DIRECT,
            5000
        )

        # Should include other viable strategies
        assert len(alternatives) <= 3
        # GSD_DIRECT should not be in alternatives (it's the primary)
        assert StrategyType.GSD_DIRECT not in alternatives

    def test_alternatives_limited_to_three(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that alternatives are limited to 3."""
        router = StrategyRouter(provider=mock_provider)
        alternatives = router._get_alternative_strategies(
            StrategyType.GSD_DIRECT,
            5000
        )

        assert len(alternatives) <= 3


# =============================================================================
# Fallback Strategy Tests
# =============================================================================


class TestFallbackStrategies:
    """Tests for fallback strategy selection."""

    def test_gsd_fallback_to_guided(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that GSD_DIRECT falls back to GSD_GUIDED."""
        router = StrategyRouter(provider=mock_provider)
        fallback = router._get_fallback_strategy(
            StrategyType.GSD_DIRECT,
            5000
        )

        assert fallback == StrategyType.GSD_GUIDED

    def test_ralph_fallback_hierarchy(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test RALPH fallback hierarchy."""
        router = StrategyRouter(provider=mock_provider)

        fallback_iterative = router._get_fallback_strategy(
            StrategyType.RALPH_ITERATIVE,
            30000
        )
        assert fallback_iterative == StrategyType.RALPH_STRUCTURED

        fallback_structured = router._get_fallback_strategy(
            StrategyType.RALPH_STRUCTURED,
            50000
        )
        assert fallback_structured == StrategyType.RLM_BASIC

    def test_rlm_full_no_fallback(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that RLM_FULL has no fallback."""
        router = StrategyRouter(provider=mock_provider)
        fallback = router._get_fallback_strategy(
            StrategyType.RLM_FULL,
            200000
        )

        assert fallback is None


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostEstimation:
    """Tests for strategy cost estimation."""

    def test_cost_estimation_enabled(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that cost estimation returns values when enabled."""
        router = StrategyRouter(provider=mock_provider)

        costs = router._estimate_costs(
            token_count=10000,
            strategies=[StrategyType.GSD_DIRECT, StrategyType.RALPH_STRUCTURED]
        )

        assert "gsd_direct" in costs
        assert "ralph_structured" in costs
        assert costs["gsd_direct"] >= 0
        assert costs["ralph_structured"] >= 0

    def test_cost_estimation_disabled(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that cost estimation returns empty when disabled."""
        config = RouterConfig(enable_cost_estimation=False)
        router = StrategyRouter(provider=mock_provider, config=config)

        costs = router._estimate_costs(
            token_count=10000,
            strategies=[StrategyType.GSD_DIRECT]
        )

        assert costs == {}

    def test_rlm_more_expensive_than_gsd(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that RLM strategies cost more than GSD."""
        router = StrategyRouter(provider=mock_provider)

        costs = router._estimate_costs(
            token_count=50000,
            strategies=[StrategyType.GSD_GUIDED, StrategyType.RLM_FULL]
        )

        # RLM should be more expensive due to multiple passes
        if "gsd_guided" in costs and "rlm_full" in costs:
            assert costs["rlm_full"] > costs["gsd_guided"]


# =============================================================================
# Context Analysis Tests
# =============================================================================


class TestContextAnalysis:
    """Tests for ContextAnalysis dataclass."""

    def test_analysis_structure(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that analysis returns correct structure."""
        mock_provider.count_tokens = MagicMock(return_value=5000)

        router = StrategyRouter(provider=mock_provider)
        analysis = router.analyze(
            task="Test task",
            context="Test context content"
        )

        assert isinstance(analysis, ContextAnalysis)
        assert isinstance(analysis.token_count, int)
        assert isinstance(analysis.estimated_density, float)
        assert isinstance(analysis.complexity, ComplexityLevel)
        assert isinstance(analysis.recommended_strategy, StrategyType)
        assert isinstance(analysis.reasoning, str)
        assert isinstance(analysis.alternative_strategies, list)
        assert isinstance(analysis.estimated_cost, dict)

    def test_analysis_to_dict(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test analysis serialization to dictionary."""
        mock_provider.count_tokens = MagicMock(return_value=5000)

        router = StrategyRouter(provider=mock_provider)
        analysis = router.analyze(
            task="Test task",
            context="Test context"
        )
        analysis_dict = analysis.to_dict()

        assert "token_count" in analysis_dict
        assert "recommended_strategy" in analysis_dict
        assert "reasoning" in analysis_dict
        assert isinstance(analysis_dict["complexity"], str)


# =============================================================================
# Preferred Strategy Override Tests
# =============================================================================


class TestPreferredStrategy:
    """Tests for preferred strategy override."""

    def test_preferred_strategy_used(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that preferred strategy overrides recommendation."""
        mock_provider.count_tokens = MagicMock(return_value=5000)

        config = RouterConfig(preferred_strategy=StrategyType.GSD_GUIDED)
        router = StrategyRouter(provider=mock_provider, config=config)

        analysis = router.analyze(
            task="Simple task",
            context="Small context"
        )

        assert analysis.recommended_strategy == StrategyType.GSD_GUIDED

    def test_preferred_strategy_rejected_if_too_large(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that preferred strategy is rejected if context too large."""
        mock_provider.count_tokens = MagicMock(return_value=50000)

        config = RouterConfig(preferred_strategy=StrategyType.GSD_DIRECT)
        router = StrategyRouter(provider=mock_provider, config=config)

        analysis = router.analyze(
            task="Task",
            context="Large context" * 10000
        )

        # Should NOT be GSD_DIRECT since context is too large
        assert analysis.recommended_strategy != StrategyType.GSD_DIRECT


# =============================================================================
# Strategy Info Tests
# =============================================================================


class TestStrategyInfo:
    """Tests for strategy information methods."""

    def test_get_strategy_info(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test getting information about a strategy."""
        router = StrategyRouter(provider=mock_provider)

        info = router.get_strategy_info(StrategyType.GSD_DIRECT)

        assert "name" in info
        assert "description" in info
        assert "optimal_for" in info
        assert "max_tokens" in info

    def test_list_strategies(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test listing all available strategies."""
        router = StrategyRouter(provider=mock_provider)

        strategies = router.list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 4  # At least GSD, RALPH, RLM variants


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_context_function(self) -> None:
        """Test standalone analyze_context function."""
        analysis = analyze_context(
            task="Summarize this",
            context="This is some content to analyze. " * 100
        )

        assert isinstance(analysis, ContextAnalysis)
        assert analysis.recommended_strategy is not None

    def test_get_recommended_strategy_function(self) -> None:
        """Test standalone get_recommended_strategy function."""
        # Small token count
        strategy = get_recommended_strategy(
            token_count=5000,
            density=0.3,
            complexity="low"
        )
        assert strategy in [StrategyType.GSD_DIRECT, StrategyType.GSD_GUIDED]

        # Large token count
        strategy = get_recommended_strategy(
            token_count=150000,
            density=0.7,
            complexity="high"
        )
        assert strategy in [StrategyType.RLM_BASIC, StrategyType.RLM_FULL]


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_token_count_uses_estimator(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that token counting uses the token estimator."""
        router = StrategyRouter(provider=mock_provider)

        # Mock the token estimator
        router._token_estimator.count_tokens = MagicMock(return_value=12345)

        count = router._count_tokens("Test text")

        assert count == 12345
        router._token_estimator.count_tokens.assert_called_once()


# =============================================================================
# Can Handle Context Tests
# =============================================================================


class TestCanHandleContext:
    """Tests for context size capability checking."""

    def test_gsd_can_handle_small(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that GSD can handle small contexts."""
        router = StrategyRouter(provider=mock_provider)

        assert router._can_handle_context(StrategyType.GSD_DIRECT, 5000) is True
        assert router._can_handle_context(StrategyType.GSD_DIRECT, 15000) is False

    def test_ralph_can_handle_medium(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that RALPH can handle medium contexts."""
        router = StrategyRouter(provider=mock_provider)

        assert router._can_handle_context(StrategyType.RALPH_STRUCTURED, 50000) is True
        assert router._can_handle_context(StrategyType.RALPH_STRUCTURED, 150000) is False

    def test_rlm_can_handle_large(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test that RLM can handle large contexts."""
        router = StrategyRouter(provider=mock_provider)

        assert router._can_handle_context(StrategyType.RLM_FULL, 500000) is True
        assert router._can_handle_context(StrategyType.RLM_FULL, 1000000) is True


# =============================================================================
# Decision Matrix Tests
# =============================================================================


class TestDecisionMatrix:
    """Tests for the decision matrix implementation."""

    def test_decision_matrix_small_simple(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: <10K tokens, low complexity -> GSD_DIRECT."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=5000,
            density=0.3,
            complexity=ComplexityLevel.LOW
        )

        assert strategy == StrategyType.GSD_DIRECT

    def test_decision_matrix_small_complex(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: <10K tokens, high complexity -> GSD_GUIDED."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=5000,
            density=0.5,
            complexity=ComplexityLevel.HIGH
        )

        assert strategy == StrategyType.GSD_GUIDED

    def test_decision_matrix_medium_sparse(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: 10K-50K tokens, sparse -> RALPH_ITERATIVE."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=25000,
            density=0.3,
            complexity=ComplexityLevel.MEDIUM
        )

        assert strategy == StrategyType.RALPH_ITERATIVE

    def test_decision_matrix_medium_dense(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: 10K-50K tokens, dense -> RALPH_STRUCTURED."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=25000,
            density=0.6,
            complexity=ComplexityLevel.MEDIUM
        )

        assert strategy == StrategyType.RALPH_STRUCTURED

    def test_decision_matrix_large_dense(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: 50K-100K tokens, high density -> RLM_BASIC."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=75000,
            density=0.8,
            complexity=ComplexityLevel.HIGH
        )

        assert strategy == StrategyType.RLM_BASIC

    def test_decision_matrix_very_large(
        self,
        mock_provider: MagicMock
    ) -> None:
        """Test: >100K tokens -> RLM_FULL."""
        router = StrategyRouter(provider=mock_provider)

        strategy, reasoning = router._select_strategy(
            token_count=150000,
            density=0.5,
            complexity=ComplexityLevel.MEDIUM
        )

        assert strategy == StrategyType.RLM_FULL
