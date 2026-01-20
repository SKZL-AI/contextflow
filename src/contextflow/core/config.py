"""
ContextFlow Configuration System.

Supports loading from environment variables, YAML files, and programmatic configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from contextflow.core.types import ChunkingStrategy, StrategyType


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    api_key: str | None = None
    model: str = ""
    base_url: str | None = None
    timeout: int = 120
    max_retries: int = 3


@dataclass
class StrategyConfig:
    """Configuration for strategy thresholds."""

    gsd_max_tokens: int = 10_000
    ralph_min_tokens: int = 5_000
    ralph_max_tokens: int = 100_000
    rlm_min_tokens: int = 50_000
    rlm_max_parallel_agents: int = 10
    rlm_max_iterations: int = 50
    rlm_max_depth: int = 3


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    chunk_size: int = 4_000
    chunk_overlap: int = 500
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SMART
    similarity_top_k: int = 5
    use_gpu: bool = False


@dataclass
class APIConfig:
    """Configuration for API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    json_format: bool = False
    include_timestamp: bool = True
    log_file: str | None = None


@dataclass
class ContextFlowConfig:
    """
    Master configuration for ContextFlow.

    Can be created from:
    - Environment variables (load with from_env())
    - YAML file (load with from_file())
    - Programmatically (direct instantiation)

    Example:
        # From environment
        config = ContextFlowConfig.from_env()

        # From file
        config = ContextFlowConfig.from_file("contextflow.yaml")

        # Programmatic
        config = ContextFlowConfig(
            default_provider="claude",
            strategy=StrategyConfig(rlm_max_parallel_agents=20)
        )
    """

    # Default provider
    default_provider: str = "claude"
    default_strategy: StrategyType = StrategyType.AUTO

    # Provider configurations
    claude: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(model="claude-3-5-sonnet-20241022")
    )
    openai: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(model="gpt-4o")
    )
    ollama: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            model="llama2", base_url="http://localhost:11434"
        )
    )
    vllm: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(base_url="http://localhost:8000")
    )
    groq: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(model="mixtral-8x7b-32768")
    )
    gemini: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(model="gemini-pro")
    )
    mistral: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(model="mistral-large-latest")
    )

    # Component configurations
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Feature flags
    enable_caching: bool = True
    enable_cost_tracking: bool = True
    enable_telemetry: bool = False

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> ContextFlowConfig:
        """
        Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to .env file

        Returns:
            ContextFlowConfig instance
        """
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        def get_env(key: str, default: Any = None) -> Any:
            return os.getenv(key, default)

        def get_env_int(key: str, default: int) -> int:
            val = os.getenv(key)
            return int(val) if val else default

        def get_env_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            if val in ("false", "0", "no"):
                return False
            return default

        return cls(
            default_provider=get_env("CONTEXTFLOW_DEFAULT_PROVIDER", "claude"),
            # Provider configs
            claude=ProviderConfig(
                api_key=get_env("ANTHROPIC_API_KEY"),
                model=get_env("CONTEXTFLOW_CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            ),
            openai=ProviderConfig(
                api_key=get_env("OPENAI_API_KEY"),
                model=get_env("CONTEXTFLOW_OPENAI_MODEL", "gpt-4o"),
            ),
            ollama=ProviderConfig(
                model=get_env("CONTEXTFLOW_OLLAMA_MODEL", "llama2"),
                base_url=get_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            vllm=ProviderConfig(
                base_url=get_env("VLLM_BASE_URL", "http://localhost:8000"),
            ),
            groq=ProviderConfig(
                api_key=get_env("GROQ_API_KEY"),
                model=get_env("CONTEXTFLOW_GROQ_MODEL", "mixtral-8x7b-32768"),
            ),
            gemini=ProviderConfig(
                api_key=get_env("GOOGLE_API_KEY"),
                model=get_env("CONTEXTFLOW_GEMINI_MODEL", "gemini-pro"),
            ),
            mistral=ProviderConfig(
                api_key=get_env("MISTRAL_API_KEY"),
                model=get_env("CONTEXTFLOW_MISTRAL_MODEL", "mistral-large-latest"),
            ),
            # Strategy config
            strategy=StrategyConfig(
                gsd_max_tokens=get_env_int("CONTEXTFLOW_GSD_MAX_TOKENS", 10_000),
                ralph_max_tokens=get_env_int("CONTEXTFLOW_RALPH_MAX_TOKENS", 100_000),
                rlm_max_parallel_agents=get_env_int(
                    "CONTEXTFLOW_RLM_MAX_PARALLEL_AGENTS", 10
                ),
                rlm_max_iterations=get_env_int("CONTEXTFLOW_RLM_MAX_ITERATIONS", 50),
            ),
            # RAG config
            rag=RAGConfig(
                embedding_provider=get_env("CONTEXTFLOW_EMBEDDING_PROVIDER", "openai"),
                embedding_model=get_env(
                    "CONTEXTFLOW_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                chunk_size=get_env_int("CONTEXTFLOW_CHUNK_SIZE", 4_000),
                chunk_overlap=get_env_int("CONTEXTFLOW_CHUNK_OVERLAP", 500),
            ),
            # API config
            api=APIConfig(
                host=get_env("CONTEXTFLOW_API_HOST", "0.0.0.0"),
                port=get_env_int("CONTEXTFLOW_API_PORT", 8000),
                api_key=get_env("CONTEXTFLOW_API_KEY"),
                cors_origins=get_env("CONTEXTFLOW_CORS_ORIGINS", "*").split(","),
            ),
            # Logging config
            logging=LoggingConfig(
                level=get_env("LOG_LEVEL", "INFO"),
                json_format=get_env_bool("LOG_JSON", False),
            ),
            # Features
            enable_telemetry=get_env_bool("CONTEXTFLOW_TELEMETRY_ENABLED", False),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ContextFlowConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ContextFlowConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ContextFlowConfig:
        """Create config from dictionary."""
        # Parse nested configs
        strategy_data = data.get("strategy", {})
        rag_data = data.get("rag", {})
        api_data = data.get("api", {})
        logging_data = data.get("logging", {})

        # Parse provider configs
        providers = {}
        for provider_name in [
            "claude",
            "openai",
            "ollama",
            "vllm",
            "groq",
            "gemini",
            "mistral",
        ]:
            provider_data = data.get(provider_name, {})
            if provider_data:
                providers[provider_name] = ProviderConfig(**provider_data)

        return cls(
            default_provider=data.get("default_provider", "claude"),
            strategy=StrategyConfig(**strategy_data) if strategy_data else StrategyConfig(),
            rag=RAGConfig(**rag_data) if rag_data else RAGConfig(),
            api=APIConfig(**api_data) if api_data else APIConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
            **providers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "default_provider": self.default_provider,
            "default_strategy": self.default_strategy.value,
            "claude": {
                "model": self.claude.model,
                "base_url": self.claude.base_url,
            },
            "openai": {
                "model": self.openai.model,
            },
            "ollama": {
                "model": self.ollama.model,
                "base_url": self.ollama.base_url,
            },
            "strategy": {
                "gsd_max_tokens": self.strategy.gsd_max_tokens,
                "ralph_max_tokens": self.strategy.ralph_max_tokens,
                "rlm_max_parallel_agents": self.strategy.rlm_max_parallel_agents,
            },
            "rag": {
                "embedding_provider": self.rag.embedding_provider,
                "chunk_size": self.rag.chunk_size,
            },
        }

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        provider_map = {
            "claude": self.claude,
            "openai": self.openai,
            "ollama": self.ollama,
            "vllm": self.vllm,
            "groq": self.groq,
            "gemini": self.gemini,
            "mistral": self.mistral,
        }
        if provider_name not in provider_map:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_map[provider_name]


# Global config instance (can be overridden)
_global_config: ContextFlowConfig | None = None


def get_config() -> ContextFlowConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ContextFlowConfig.from_env()
    return _global_config


def set_config(config: ContextFlowConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
