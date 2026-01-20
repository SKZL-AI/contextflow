#!/usr/bin/env python
"""
RAG Processing Example.

This example demonstrates ContextFlow's RAG (Retrieval-Augmented Generation)
capabilities for processing large document collections:
- Document indexing with FAISS
- 3-layer search (compact -> document -> full)
- Semantic retrieval
- Processing with RAG-enhanced context

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable
    - Install contextflow: pip install -e .
    - Install optional: pip install sentence-transformers faiss-cpu

Run:
    python examples/05_rag_processing.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow, StrategyType


# =============================================================================
# Sample Document Collection
# =============================================================================

# Simulating a collection of documents about a software system
DOCUMENTS = {
    "architecture.md": """
# System Architecture

## Overview

The platform follows a microservices architecture with event-driven
communication. Services are deployed in Kubernetes clusters across
three regions for high availability.

## Core Services

### User Service
Handles authentication, authorization, and user profile management.
- Database: PostgreSQL with read replicas
- Cache: Redis for session storage
- Auth: OAuth 2.0 with JWT tokens

### Order Service
Manages the complete order lifecycle from creation to fulfillment.
- Database: PostgreSQL with sharding by region
- Queue: RabbitMQ for async processing
- Events: Publishes order state changes

### Inventory Service
Tracks product inventory across warehouses.
- Database: Cassandra for high write throughput
- Sync: Real-time inventory updates via Kafka
- Alerts: Low stock notifications

### Payment Service
Processes payments and handles financial transactions.
- PCI-DSS compliant infrastructure
- Integrates with Stripe, PayPal, and bank APIs
- Fraud detection with ML models
""",

    "api_docs.md": """
# API Documentation

## REST API v2

Base URL: https://api.example.com/v2

### Authentication

All endpoints require Bearer token authentication.

```
Authorization: Bearer <access_token>
```

### Users Endpoints

#### GET /users/{id}
Retrieve user profile by ID.

Response:
```json
{
  "id": "usr_123",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### POST /users
Create new user account.

Request:
```json
{
  "email": "newuser@example.com",
  "password": "securepass123",
  "name": "Jane Doe"
}
```

### Orders Endpoints

#### GET /orders
List orders for authenticated user.

Query Parameters:
- status: Filter by order status
- limit: Max results (default 20)
- offset: Pagination offset

#### POST /orders
Create new order.

Request:
```json
{
  "items": [
    {"product_id": "prod_456", "quantity": 2}
  ],
  "shipping_address": {...}
}
```

### Inventory Endpoints

#### GET /inventory/{product_id}
Check product availability.

Response:
```json
{
  "product_id": "prod_456",
  "available": 150,
  "reserved": 12,
  "warehouses": [...]
}
```
""",

    "deployment.md": """
# Deployment Guide

## Environment Setup

### Development
- Local Kubernetes with Kind or Minikube
- Local databases via Docker Compose
- Mock payment service

### Staging
- Shared GKE cluster
- Cloud SQL databases
- Stripe test mode

### Production
- Multi-region GKE clusters (us-west, us-east, eu-west)
- Cloud SQL with regional failover
- Full payment integration

## CI/CD Pipeline

### Build Stage
1. Run unit tests
2. Run integration tests
3. Build Docker images
4. Push to Container Registry
5. Security scan with Trivy

### Deploy Stage
1. Apply Kubernetes manifests
2. Run database migrations
3. Health check verification
4. Traffic shift (canary/blue-green)
5. Smoke tests

## Scaling Configuration

### Horizontal Pod Autoscaler
```yaml
minReplicas: 3
maxReplicas: 50
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling
- Read replicas auto-scale based on query load
- Write scaling via sharding configuration
- Connection pooling with PgBouncer
""",

    "troubleshooting.md": """
# Troubleshooting Guide

## Common Issues

### 401 Unauthorized Errors
**Symptom**: API returns 401 for valid requests.

**Causes**:
1. Expired access token
2. Invalid token format
3. Token revoked

**Solutions**:
- Refresh the access token
- Verify token is correctly formatted in Authorization header
- Re-authenticate if token was revoked

### Order Processing Delays
**Symptom**: Orders stuck in "pending" state.

**Causes**:
1. RabbitMQ queue backlog
2. Payment service timeout
3. Inventory sync lag

**Solutions**:
- Check RabbitMQ management console
- Verify payment service health
- Force inventory sync via admin API

### Database Connection Errors
**Symptom**: "Connection refused" or timeout errors.

**Causes**:
1. Connection pool exhaustion
2. Database overloaded
3. Network issues

**Solutions**:
- Increase connection pool size
- Add read replicas
- Check Kubernetes network policies

### High Memory Usage
**Symptom**: Pods getting OOMKilled.

**Causes**:
1. Memory leak in application
2. Large payload processing
3. Insufficient resource limits

**Solutions**:
- Profile application memory
- Implement streaming for large payloads
- Adjust resource requests/limits

## Debugging Commands

```bash
# Check pod logs
kubectl logs -f deployment/user-service

# Get pod events
kubectl describe pod <pod-name>

# Check service health
curl http://localhost:8080/health

# Database connections
SELECT count(*) FROM pg_stat_activity;
```
"""
}


# =============================================================================
# Example Functions
# =============================================================================


async def basic_rag_example() -> None:
    """
    Basic RAG processing with document collection.

    Shows how to use process_with_rag for document-aware queries.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic RAG Processing")
    print("=" * 60)

    # Combine documents for this example
    full_context = "\n\n---\n\n".join([
        f"# {name}\n{content}"
        for name, content in DOCUMENTS.items()
    ])

    async with ContextFlow() as cf:
        # Use RAG-enhanced processing
        result = await cf.process_with_rag(
            task="What databases are used by each service and why?",
            context=full_context,
            k=5,  # Retrieve top 5 relevant chunks
        )

        print(f"\nTask: What databases are used by each service?")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Tokens Used: {result.total_tokens}")
        print(f"\n--- Answer ---")
        print(result.answer)


async def semantic_search_example() -> None:
    """
    Semantic search across document collection.

    Demonstrates how RAG finds relevant information across documents.
    """
    print("\n" + "=" * 60)
    print("Example 2: Semantic Search Across Documents")
    print("=" * 60)

    full_context = "\n\n---\n\n".join([
        f"# {name}\n{content}"
        for name, content in DOCUMENTS.items()
    ])

    queries = [
        "How do I fix authentication errors?",
        "What is the deployment process for production?",
        "How does the order service communicate with inventory?",
    ]

    async with ContextFlow() as cf:
        for query in queries:
            print(f"\n--- Query: {query} ---")

            result = await cf.process_with_rag(
                task=query,
                context=full_context,
                k=3,  # Top 3 relevant chunks
            )

            # Show a snippet of the answer
            answer_preview = result.answer[:300] + "..." if len(result.answer) > 300 else result.answer
            print(f"Answer: {answer_preview}")


async def multi_document_analysis() -> None:
    """
    Analyze information spread across multiple documents.

    Shows RAG's ability to synthesize from multiple sources.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-Document Analysis")
    print("=" * 60)

    full_context = "\n\n---\n\n".join([
        f"# {name}\n{content}"
        for name, content in DOCUMENTS.items()
    ])

    async with ContextFlow() as cf:
        result = await cf.process_with_rag(
            task="""
            Create a comprehensive checklist for deploying a new feature
            that affects the User Service. Include:
            1. Pre-deployment checks
            2. Deployment steps
            3. Post-deployment verification
            4. Common issues to watch for
            """,
            context=full_context,
            k=10,  # Get more context for comprehensive answer
            constraints=[
                "Reference specific services mentioned in docs",
                "Include actual commands where applicable",
                "Organize as a numbered checklist",
            ],
        )

        print(f"\nTask: Create deployment checklist for User Service")
        print(f"Tokens: {result.total_tokens}")
        print(f"Time: {result.execution_time:.2f}s")
        print(f"\n--- Deployment Checklist ---")
        print(result.answer)


async def targeted_retrieval_example() -> None:
    """
    Targeted retrieval with specific context needs.

    Shows how to optimize retrieval for specific query types.
    """
    print("\n" + "=" * 60)
    print("Example 4: Targeted Retrieval"  )
    print("=" * 60)

    full_context = "\n\n---\n\n".join([
        f"# {name}\n{content}"
        for name, content in DOCUMENTS.items()
    ])

    async with ContextFlow() as cf:
        # Technical deep-dive query
        print("\n--- Technical Query (high k for depth) ---")
        result_technical = await cf.process_with_rag(
            task="Explain the complete payment processing flow including security measures",
            context=full_context,
            k=8,  # More chunks for comprehensive answer
        )
        print(f"Retrieved chunks for technical query")
        print(f"Answer preview: {result_technical.answer[:400]}...")

        # Quick lookup query
        print("\n--- Quick Lookup Query (low k for speed) ---")
        result_quick = await cf.process_with_rag(
            task="What is the API base URL?",
            context=full_context,
            k=2,  # Fewer chunks for quick answer
        )
        print(f"Answer: {result_quick.answer}")


async def comparison_rag_vs_direct() -> None:
    """
    Compare RAG-enhanced vs direct processing.

    Shows the benefit of RAG for large document collections.
    """
    print("\n" + "=" * 60)
    print("Example 5: RAG vs Direct Processing Comparison")
    print("=" * 60)

    full_context = "\n\n---\n\n".join([
        f"# {name}\n{content}"
        for name, content in DOCUMENTS.items()
    ])

    task = "What are all the debugging commands mentioned in the documentation?"

    async with ContextFlow() as cf:
        # Process with RAG
        print("\n--- With RAG ---")
        result_rag = await cf.process_with_rag(
            task=task,
            context=full_context,
            k=5,
        )
        print(f"Tokens used: {result_rag.total_tokens}")
        print(f"Time: {result_rag.execution_time:.2f}s")
        print(f"Answer: {result_rag.answer[:300]}...")

        # Process directly (all context at once)
        print("\n--- Without RAG (Direct) ---")
        result_direct = await cf.process(
            task=task,
            context=full_context,
            strategy=StrategyType.AUTO,
        )
        print(f"Tokens used: {result_direct.total_tokens}")
        print(f"Time: {result_direct.execution_time:.2f}s")
        print(f"Answer: {result_direct.answer[:300]}...")

        # Compare
        print("\n--- Comparison ---")
        print(f"RAG tokens: {result_rag.total_tokens} vs Direct tokens: {result_direct.total_tokens}")
        token_savings = ((result_direct.total_tokens - result_rag.total_tokens) /
                        result_direct.total_tokens * 100) if result_direct.total_tokens > 0 else 0
        print(f"Token savings with RAG: {token_savings:.1f}%")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow RAG Processing Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples.")
        return

    # Check for optional dependencies
    try:
        import sentence_transformers
        print("sentence-transformers: Available")
    except ImportError:
        print("\nNote: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        print("Some RAG features may use fallback embeddings.\n")

    try:
        await basic_rag_example()
        await semantic_search_example()
        await multi_document_analysis()
        await targeted_retrieval_example()
        await comparison_rag_vs_direct()

        print("\n" + "=" * 60)
        print("All RAG processing examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
