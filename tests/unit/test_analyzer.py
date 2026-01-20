"""
Unit tests for ContextAnalyzer.

Tests context analysis functionality including:
- Token counting
- Density estimation
- Complexity assessment
- Strategy recommendations
- Content type detection
- Caching behavior
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from contextflow.core.analyzer import (
    COMPLEXITY_INDICATORS,
    AnalyzerConfig,
    ChunkSuggestion,
    ComplexityLevel,
    ContentType,
    ContextAnalysis,
    ContextAnalyzer,
    CostEstimate,
    DensityLevel,
    analyze_context,
    analyze_context_async,
    estimate_analysis_cost,
    get_recommended_strategy,
)
from contextflow.core.types import StrategyType
from contextflow.utils.errors import ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider for analyzer tests."""
    provider = MagicMock()
    provider.name = "mock-analyzer"
    provider.model = "mock-model"
    provider.complete = AsyncMock()
    return provider


@pytest.fixture
def default_analyzer() -> ContextAnalyzer:
    """Create analyzer with default configuration."""
    return ContextAnalyzer()


@pytest.fixture
def analyzer_with_provider(mock_provider: MagicMock) -> ContextAnalyzer:
    """Create analyzer with mock provider."""
    return ContextAnalyzer(provider=mock_provider)


@pytest.fixture
def custom_config() -> AnalyzerConfig:
    """Create custom analyzer configuration."""
    return AnalyzerConfig(
        gsd_max_tokens=15_000,
        ralph_max_tokens=80_000,
        density_threshold_low=0.25,
        density_threshold_high=0.75,
        chunk_target_size=5000,
    )


@pytest.fixture
def code_content() -> str:
    """Sample code content for testing."""
    return '''
import asyncio
from typing import List, Dict, Optional

class DataProcessor:
    """Process data efficiently."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._cache: Dict[str, Any] = {}

    async def process(self, items: List[str]) -> List[Dict]:
        results = []
        for item in items:
            if item in self._cache:
                results.append(self._cache[item])
            else:
                processed = await self._transform(item)
                self._cache[item] = processed
                results.append(processed)
        return results

    async def _transform(self, item: str) -> Dict:
        await asyncio.sleep(0.1)
        return {"value": item, "processed": True}
'''


@pytest.fixture
def prose_content() -> str:
    """Sample prose content for testing."""
    return """
    The quick brown fox jumps over the lazy dog. This sentence contains every
    letter of the English alphabet, making it useful for typography testing.
    The story continues with more narrative elements that flow naturally from
    one idea to the next, creating a cohesive piece of writing that readers
    can follow easily. Good prose should be clear, engaging, and meaningful.
    """


@pytest.fixture
def data_content() -> str:
    """Sample structured data content for testing."""
    return """
{
    "users": [
        {"id": 1, "name": "Alice", "active": true, "score": 95.5},
        {"id": 2, "name": "Bob", "active": false, "score": 87.3},
        {"id": 3, "name": "Charlie", "active": true, "score": 92.1}
    ],
    "metadata": {
        "version": "1.0",
        "created": "2024-01-15",
        "records": 3
    }
}
"""


# =============================================================================
# Analyzer Configuration Tests
# =============================================================================


class TestAnalyzerConfig:
    """Tests for AnalyzerConfig."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = AnalyzerConfig()

        assert config.gsd_max_tokens == 10_000
        assert config.ralph_min_tokens == 10_000
        assert config.ralph_max_tokens == 100_000
        assert config.rlm_min_tokens == 100_000
        assert config.density_threshold_low == 0.3
        assert config.density_threshold_high == 0.7
        assert config.enable_llm_analysis is False
        assert config.chunk_target_size == 4000
        assert config.chunk_overlap_ratio == 0.125

    def test_config_to_dict(self) -> None:
        """Test configuration serialization."""
        config = AnalyzerConfig()
        config_dict = config.to_dict()

        assert "gsd_max_tokens" in config_dict
        assert "density_threshold_high" in config_dict
        assert "chunk_target_size" in config_dict


# =============================================================================
# Analyzer Initialization Tests
# =============================================================================


class TestAnalyzerInitialization:
    """Tests for ContextAnalyzer initialization."""

    def test_init_without_provider(self) -> None:
        """Test initialization without provider."""
        analyzer = ContextAnalyzer()

        assert analyzer._provider is None
        assert analyzer.has_provider is False
        assert analyzer._config is not None

    def test_init_with_provider(self, mock_provider: MagicMock) -> None:
        """Test initialization with provider."""
        analyzer = ContextAnalyzer(provider=mock_provider)

        assert analyzer._provider == mock_provider
        assert analyzer.has_provider is True

    def test_init_with_custom_config(self, custom_config: AnalyzerConfig) -> None:
        """Test initialization with custom configuration."""
        analyzer = ContextAnalyzer(config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config.gsd_max_tokens == 15_000

    def test_analyzer_repr(self, default_analyzer: ContextAnalyzer) -> None:
        """Test analyzer string representation."""
        repr_str = repr(default_analyzer)

        assert "ContextAnalyzer" in repr_str
        assert "has_provider=" in repr_str


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_basic(self, default_analyzer: ContextAnalyzer) -> None:
        """Test basic token counting."""
        text = "This is a simple test sentence for token counting."
        count = default_analyzer._count_tokens(text)

        # Should return a reasonable token count
        assert count > 0
        assert count < len(text)  # Tokens should be fewer than characters

    def test_count_tokens_empty_string(self, default_analyzer: ContextAnalyzer) -> None:
        """Test token counting with empty string."""
        count = default_analyzer._count_tokens("")

        assert count == 0

    def test_count_tokens_long_text(self, default_analyzer: ContextAnalyzer) -> None:
        """Test token counting with longer text."""
        long_text = "This is a test sentence. " * 1000
        count = default_analyzer._count_tokens(long_text)

        # Roughly 4 chars per token
        expected_min = len(long_text) // 6
        expected_max = len(long_text) // 3
        assert expected_min < count < expected_max


# =============================================================================
# Density Estimation Tests
# =============================================================================


class TestDensityEstimation:
    """Tests for information density estimation."""

    def test_empty_text_zero_density(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that empty text has zero density."""
        density = default_analyzer._estimate_density("")

        assert density == 0.0

    def test_short_text_zero_density(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that very short text has zero density."""
        density = default_analyzer._estimate_density("Hi")

        assert density == 0.0

    def test_whitespace_heavy_low_density(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that whitespace-heavy content has lower density."""
        sparse_text = "Word   " * 50  # Lots of whitespace

        density = default_analyzer._estimate_density(sparse_text)

        assert density < 0.5

    def test_code_higher_density(
        self, default_analyzer: ContextAnalyzer, code_content: str
    ) -> None:
        """Test that code content has higher density."""
        density = default_analyzer._estimate_density(code_content)

        # Code should be moderately dense
        assert density > 0.3

    def test_density_in_valid_range(
        self, default_analyzer: ContextAnalyzer, prose_content: str
    ) -> None:
        """Test that density is always in [0, 1] range."""
        density = default_analyzer._estimate_density(prose_content)

        assert 0.0 <= density <= 1.0


# =============================================================================
# Density Classification Tests
# =============================================================================


class TestDensityClassification:
    """Tests for density level classification."""

    def test_classify_sparse(self, default_analyzer: ContextAnalyzer) -> None:
        """Test classification of sparse density."""
        level = default_analyzer._classify_density(0.2)

        assert level == DensityLevel.SPARSE

    def test_classify_medium(self, default_analyzer: ContextAnalyzer) -> None:
        """Test classification of medium density."""
        level = default_analyzer._classify_density(0.5)

        assert level == DensityLevel.MEDIUM

    def test_classify_dense(self, default_analyzer: ContextAnalyzer) -> None:
        """Test classification of dense content."""
        level = default_analyzer._classify_density(0.8)

        assert level == DensityLevel.DENSE


# =============================================================================
# Complexity Assessment Tests
# =============================================================================


class TestComplexityAssessment:
    """Tests for task complexity assessment."""

    def test_simple_task(self, default_analyzer: ContextAnalyzer) -> None:
        """Test simple task complexity."""
        simple_tasks = [
            "What is the name?",
            "List the items",
            "Find the date",
        ]

        for task in simple_tasks:
            complexity = default_analyzer._assess_complexity(task)
            assert complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]

    def test_complex_task(self, default_analyzer: ContextAnalyzer) -> None:
        """Test complex task complexity."""
        complex_tasks = [
            "Analyze the data and explain why the patterns emerge",
            "Compare and contrast the approaches, evaluating their implications",
        ]

        for task in complex_tasks:
            complexity = default_analyzer._assess_complexity(task)
            assert complexity in [
                ComplexityLevel.MODERATE,
                ComplexityLevel.COMPLEX,
                ComplexityLevel.EXHAUSTIVE,
            ]

    def test_exhaustive_task(self, default_analyzer: ContextAnalyzer) -> None:
        """Test exhaustive task complexity."""
        exhaustive_task = "Provide an exhaustive and comprehensive analysis of everything, covering all aspects thoroughly"

        complexity = default_analyzer._assess_complexity(exhaustive_task)

        assert complexity == ComplexityLevel.EXHAUSTIVE

    def test_constraints_increase_complexity(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that constraints increase complexity assessment."""
        task = "Summarize the document"

        complexity_no_constraints = default_analyzer._assess_complexity(task, None)
        complexity_with_constraints = default_analyzer._assess_complexity(
            task,
            [
                "Must be under 500 words",
                "Include all key points",
                "Use bullet points",
                "Be technical",
            ],
        )

        # With many constraints, complexity should increase
        complexity_values = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 2,
            ComplexityLevel.COMPLEX: 3,
            ComplexityLevel.EXHAUSTIVE: 4,
        }

        assert (
            complexity_values[complexity_with_constraints]
            >= complexity_values[complexity_no_constraints]
        )


# =============================================================================
# Content Type Detection Tests
# =============================================================================


class TestContentTypeDetection:
    """Tests for content type detection."""

    def test_detect_code(self, default_analyzer: ContextAnalyzer, code_content: str) -> None:
        """Test detection of code content."""
        content_type = default_analyzer._detect_content_type(code_content)

        assert content_type in [ContentType.CODE, ContentType.MIXED]

    def test_detect_data(self, default_analyzer: ContextAnalyzer, data_content: str) -> None:
        """Test detection of data content."""
        content_type = default_analyzer._detect_content_type(data_content)

        assert content_type in [ContentType.DATA, ContentType.MIXED]

    def test_detect_prose(self, default_analyzer: ContextAnalyzer, prose_content: str) -> None:
        """Test detection of prose content."""
        content_type = default_analyzer._detect_content_type(prose_content)

        assert content_type in [ContentType.PROSE, ContentType.DOCUMENTATION]

    def test_detect_short_unknown(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that very short content is unknown."""
        content_type = default_analyzer._detect_content_type("Hi")

        assert content_type == ContentType.UNKNOWN


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for strategy selection logic."""

    def test_small_context_gsd(self, default_analyzer: ContextAnalyzer) -> None:
        """Test small context selects GSD."""
        strategy, reasoning = default_analyzer._select_strategy(
            token_count=5000, density=0.5, complexity=ComplexityLevel.SIMPLE
        )

        assert strategy == StrategyType.GSD_DIRECT

    def test_medium_context_ralph(self, default_analyzer: ContextAnalyzer) -> None:
        """Test medium context selects RALPH."""
        strategy, reasoning = default_analyzer._select_strategy(
            token_count=30000, density=0.5, complexity=ComplexityLevel.MODERATE
        )

        assert strategy == StrategyType.RALPH_STRUCTURED

    def test_large_context_rlm(self, default_analyzer: ContextAnalyzer) -> None:
        """Test large context selects RLM."""
        strategy, reasoning = default_analyzer._select_strategy(
            token_count=150000, density=0.5, complexity=ComplexityLevel.COMPLEX
        )

        assert strategy in [StrategyType.RLM_FULL, StrategyType.RLM_DENSE]

    def test_reasoning_included(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that reasoning string is included."""
        strategy, reasoning = default_analyzer._select_strategy(
            token_count=5000, density=0.5, complexity=ComplexityLevel.SIMPLE
        )

        assert len(reasoning) > 0
        assert "token" in reasoning.lower()


# =============================================================================
# Full Analysis Tests
# =============================================================================


class TestFullAnalysis:
    """Tests for full context analysis."""

    def test_analyze_returns_context_analysis(
        self, default_analyzer: ContextAnalyzer, prose_content: str
    ) -> None:
        """Test that analyze returns ContextAnalysis object."""
        analysis = default_analyzer.analyze(task="Summarize this text", context=prose_content)

        assert isinstance(analysis, ContextAnalysis)
        assert analysis.token_count > 0
        assert 0.0 <= analysis.density <= 1.0
        assert analysis.recommended_strategy is not None
        assert len(analysis.reasoning) > 0

    def test_analyze_validates_empty_task(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that empty task raises ValidationError."""
        with pytest.raises(ValidationError):
            default_analyzer.analyze(task="", context="Some context")

    def test_analyze_validates_empty_context(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that empty context raises ValidationError."""
        with pytest.raises(ValidationError):
            default_analyzer.analyze(task="Do something", context="")

    def test_analysis_to_dict(self, default_analyzer: ContextAnalyzer, prose_content: str) -> None:
        """Test analysis serialization to dictionary."""
        analysis = default_analyzer.analyze(task="Summarize", context=prose_content)
        analysis_dict = analysis.to_dict()

        assert "token_count" in analysis_dict
        assert "density" in analysis_dict
        assert "density_level" in analysis_dict
        assert "complexity" in analysis_dict
        assert "content_type" in analysis_dict
        assert "recommended_strategy" in analysis_dict
        assert "reasoning" in analysis_dict


# =============================================================================
# Chunk Suggestion Tests
# =============================================================================


class TestChunkSuggestion:
    """Tests for chunking suggestions."""

    def test_chunk_suggestion_structure(
        self, default_analyzer: ContextAnalyzer, code_content: str
    ) -> None:
        """Test chunk suggestion has correct structure."""
        suggestion = default_analyzer._suggest_chunking(
            token_count=10000, content_type=ContentType.CODE
        )

        assert isinstance(suggestion, ChunkSuggestion)
        assert suggestion.size > 0
        assert suggestion.overlap >= 0
        assert suggestion.total_chunks >= 1

    def test_code_smaller_chunks(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that code gets smaller chunk sizes."""
        code_suggestion = default_analyzer._suggest_chunking(10000, ContentType.CODE)
        prose_suggestion = default_analyzer._suggest_chunking(10000, ContentType.PROSE)

        assert code_suggestion.size < prose_suggestion.size

    def test_data_larger_chunks(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that data gets larger chunk sizes."""
        data_suggestion = default_analyzer._suggest_chunking(10000, ContentType.DATA)
        prose_suggestion = default_analyzer._suggest_chunking(10000, ContentType.PROSE)

        assert data_suggestion.size > prose_suggestion.size


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_cost_estimate_structure(self, default_analyzer: ContextAnalyzer) -> None:
        """Test cost estimate has correct structure."""
        costs = default_analyzer._estimate_costs(
            token_count=10000, strategy=StrategyType.GSD_DIRECT
        )

        # Should have at least one model estimated
        assert len(costs) > 0

        for model, estimate in costs.items():
            assert isinstance(estimate, CostEstimate)
            assert estimate.input_cost >= 0
            assert estimate.output_cost >= 0
            assert estimate.total_cost >= 0


# =============================================================================
# Warnings Generation Tests
# =============================================================================


class TestWarningsGeneration:
    """Tests for warning generation."""

    def test_large_context_warning(self, default_analyzer: ContextAnalyzer) -> None:
        """Test warning generated for very large context."""
        warnings = default_analyzer._generate_warnings(
            token_count=600000, density=0.5, complexity=ComplexityLevel.MODERATE
        )

        assert len(warnings) > 0
        assert any("large" in w.lower() for w in warnings)

    def test_high_density_complex_warning(self, default_analyzer: ContextAnalyzer) -> None:
        """Test warning for high density + complex task combination."""
        warnings = default_analyzer._generate_warnings(
            token_count=50000, density=0.85, complexity=ComplexityLevel.COMPLEX
        )

        assert len(warnings) > 0


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for analysis caching."""

    def test_cache_hit(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that same input returns cached result."""
        task = "Summarize this"
        context = "Some context content here."

        # First call
        analysis1 = default_analyzer.analyze(task, context, use_cache=True)

        # Second call should use cache
        analysis2 = default_analyzer.analyze(task, context, use_cache=True)

        # Should be same object from cache
        assert analysis1 is analysis2

    def test_cache_bypass(self, default_analyzer: ContextAnalyzer) -> None:
        """Test that cache can be bypassed."""
        task = "Summarize this"
        context = "Some context content here."

        analysis1 = default_analyzer.analyze(task, context, use_cache=False)
        analysis2 = default_analyzer.analyze(task, context, use_cache=False)

        # Should NOT be same object
        assert analysis1 is not analysis2

    def test_clear_cache(self, default_analyzer: ContextAnalyzer) -> None:
        """Test cache clearing."""
        default_analyzer.analyze("Task", "Context" * 10, use_cache=True)

        default_analyzer.clear_cache()

        stats = default_analyzer.get_cache_stats()
        assert stats["size"] == 0

    def test_cache_stats(self, default_analyzer: ContextAnalyzer) -> None:
        """Test cache statistics."""
        default_analyzer.clear_cache()
        default_analyzer.analyze("Task", "Context" * 10, use_cache=True)

        stats = default_analyzer.get_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "utilization" in stats
        assert stats["size"] == 1


# =============================================================================
# Configuration Update Tests
# =============================================================================


class TestConfigurationUpdate:
    """Tests for configuration updates."""

    def test_update_config(self, default_analyzer: ContextAnalyzer) -> None:
        """Test configuration update."""
        original_max = default_analyzer.config.gsd_max_tokens

        default_analyzer.update_config(gsd_max_tokens=20000)

        assert default_analyzer.config.gsd_max_tokens == 20000
        assert default_analyzer.config.gsd_max_tokens != original_max


# =============================================================================
# Async Analysis Tests
# =============================================================================


class TestAsyncAnalysis:
    """Tests for async analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_async_without_llm(self, default_analyzer: ContextAnalyzer) -> None:
        """Test async analysis without LLM."""
        analysis = await default_analyzer.analyze_async(
            task="Summarize", context="Content to analyze" * 50, use_llm=False
        )

        assert isinstance(analysis, ContextAnalysis)
        assert analysis.recommended_strategy is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_context_function(self) -> None:
        """Test standalone analyze_context function."""
        analysis = analyze_context(task="Summarize", context="Some content to analyze. " * 100)

        assert isinstance(analysis, ContextAnalysis)

    def test_get_recommended_strategy_function(self) -> None:
        """Test standalone get_recommended_strategy function."""
        strategy = get_recommended_strategy(token_count=5000, density=0.5, complexity="moderate")

        assert strategy == StrategyType.GSD_DIRECT

    def test_estimate_analysis_cost_function(self) -> None:
        """Test standalone cost estimation function."""
        estimate = estimate_analysis_cost(token_count=10000, model="claude-3-5-sonnet-20241022")

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost >= 0

    @pytest.mark.asyncio
    async def test_analyze_context_async_function(self) -> None:
        """Test standalone async analysis function."""
        analysis = await analyze_context_async(
            task="Summarize", context="Content here. " * 100, use_llm=False
        )

        assert isinstance(analysis, ContextAnalysis)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_complexity_indicators_structure(self) -> None:
        """Test COMPLEXITY_INDICATORS structure."""
        assert "exhaustive" in COMPLEXITY_INDICATORS
        assert "complex" in COMPLEXITY_INDICATORS
        assert "moderate" in COMPLEXITY_INDICATORS
        assert "simple" in COMPLEXITY_INDICATORS

        for level, keywords in COMPLEXITY_INDICATORS.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0
