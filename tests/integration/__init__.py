"""
Integration tests for ContextFlow.

This package contains end-to-end tests that verify the full processing pipelines
with mock providers and verification loops.

Test Modules:
    - test_orchestrator_pipeline: Full ContextFlow orchestrator pipeline tests
    - test_strategy_routing: Automatic strategy selection end-to-end tests
    - test_verification_integration: Verification across strategies tests
    - test_rag_integration: RAG with strategies tests
    - test_api_integration: API endpoints with mock processing tests
"""

__all__ = [
    "test_orchestrator_pipeline",
    "test_strategy_routing",
    "test_verification_integration",
    "test_rag_integration",
    "test_api_integration",
]
