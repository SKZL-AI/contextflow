# Contributing to ContextFlow

Thank you for your interest in contributing to ContextFlow! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project, you agree to maintain a respectful and inclusive
environment. Please be considerate in your interactions with other contributors.

## License

By contributing to ContextFlow, you agree that your contributions will be licensed
under the **MIT License**. All contributions become part of the project and are
subject to the same license terms as the rest of the codebase.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry (dependency management)
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/contextflow.git
   cd contextflow
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/contextflow.git
   ```

---

## Development Setup

### Install Dependencies

```bash
# Install all dependencies including dev and test
poetry install --with dev,test

# Activate the virtual environment
poetry shell
```

### Verify Installation

```bash
# Run quick verification
python -c "from contextflow.core import ContextFlow; print('OK')"

# Check providers
python -c "from contextflow.providers import list_providers; print(list_providers())"
```

---

## Code Style

We use strict code formatting and linting tools.

### Formatting

```bash
# Auto-format code
make format
# or
poetry run black src/ tests/
poetry run isort src/ tests/
```

### Linting

```bash
# Check for issues
make lint
# or
poetry run ruff check src/ tests/
```

### Type Checking

```bash
# Run type checker
make typecheck
# or
poetry run mypy src/
```

### Style Guidelines

1. **Type Hints Required** - All functions must have type annotations
   ```python
   # Good
   async def process(self, task: str, context: str) -> ProcessResult:
       ...

   # Bad
   async def process(self, task, context):
       ...
   ```

2. **Docstrings Required** - All public methods need docstrings
   ```python
   async def complete(
       self,
       messages: list[Message],
       system: str | None = None,
   ) -> CompletionResponse:
       """
       Send completion request to provider.

       Args:
           messages: List of conversation messages
           system: Optional system prompt

       Returns:
           CompletionResponse with text and token usage

       Raises:
           ProviderError: If API call fails
       """
   ```

3. **No Print Statements** - Use structured logging
   ```python
   # Good
   from contextflow.utils.logging import ProviderLogger
   logger = ProviderLogger("my_module")
   logger.info("Processing", count=10)

   # Bad
   print(f"Processing {count} items")
   ```

4. **Async/Await** - Use async throughout
   ```python
   # Good
   await asyncio.sleep(1)
   async with httpx.AsyncClient() as client:
       await client.get(url)

   # Bad
   time.sleep(1)
   requests.get(url)
   ```

---

## Testing

### Running Tests

```bash
# Run all tests (WARNING: may use significant memory)
make test

# Run unit tests only (recommended)
poetry run pytest tests/unit/ -v

# Run specific test file
poetry run pytest tests/unit/test_analyzer.py -v

# Run with coverage
make test-cov
```

### Writing Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use pytest-asyncio for async tests
4. Mock external services (providers, APIs)

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_my_feature():
    # Arrange
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=...)

    # Act
    result = await my_function(mock_provider)

    # Assert
    assert result.status == "success"
```

---

## Pull Request Process

### Before Submitting

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** with clear, atomic commits

3. **Run quality checks**:
   ```bash
   make check  # format + lint + typecheck
   ```

4. **Run tests**:
   ```bash
   poetry run pytest tests/unit/ -v
   ```

5. **Update documentation** if needed

### Submitting

1. Push your branch:
   ```bash
   git push origin feature/my-new-feature
   ```

2. Create a Pull Request on GitHub

3. Fill in the PR template with:
   - Summary of changes
   - Related issue (if any)
   - Test plan
   - Screenshots (if UI changes)

### PR Requirements

- All CI checks must pass
- At least one approval from maintainers
- No merge conflicts
- Documentation updated (if applicable)

---

## Issue Guidelines

### Bug Reports

Please include:
- Python version
- ContextFlow version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Please include:
- Use case description
- Proposed solution
- Alternatives considered
- Willingness to contribute

### Questions

For questions, please use:
- GitHub Discussions (preferred)
- Issue with "question" label

---

## Project Structure

```
contextflow/
├── src/contextflow/
│   ├── core/           # Orchestrator, Analyzer, Router
│   ├── providers/      # LLM Providers (Claude, OpenAI, etc.)
│   ├── strategies/     # GSD, RALPH, RLM
│   ├── rag/            # FAISS RAG, Chunker
│   ├── api/            # FastAPI REST API
│   ├── cli/            # Typer CLI
│   └── mcp/            # Model Context Protocol
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation
└── examples/           # Usage examples
```

---

## Getting Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `examples/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions

---

## Recognition

Contributors will be recognized in:
- Release notes
- Contributors list
- Special thanks section

Thank you for contributing to ContextFlow!
