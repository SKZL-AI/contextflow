# Context Analyzer Agent

Specialized subagent for analyzing document context and recommending strategies.

## Purpose

Analyze input documents to determine:
1. Total token count
2. Information density
3. Complexity level
4. Optimal processing strategy

## Invocation

```python
from contextflow.agents import SubAgent

analyzer = SubAgent(
    name="context-analyzer",
    system_prompt="""
    You are a context analysis expert. Analyze documents to determine:
    1. Token count estimation
    2. Information density (sparse vs dense)
    3. Content complexity (simple, moderate, complex)
    4. Recommended strategy (GSD, RALPH, RLM)

    Provide clear reasoning for your recommendations.
    """
)

result = await analyzer.execute(
    task="Analyze this codebase for processing",
    context=document_content
)
```

## Analysis Factors

### Token Metrics
- Raw character count
- Estimated token count (chars / 4)
- Chunk distribution

### Information Density
- **Sparse (< 0.3):** Lots of whitespace, comments, boilerplate
- **Medium (0.3-0.7):** Normal code/documentation
- **Dense (> 0.7):** Compressed data, minified code, logs

### Complexity Indicators
- **Simple:** Single topic, linear flow
- **Moderate:** Multiple topics, some branching
- **Complex:** Interconnected concepts, requires deep analysis

## Strategy Recommendation Matrix

| Tokens | Density | Complexity | Strategy |
|--------|---------|------------|----------|
| <10K | Any | Simple | GSD_DIRECT |
| <10K | Any | Complex | GSD_GUIDED |
| 10K-50K | Sparse | Any | RALPH_ITERATIVE |
| 10K-50K | Dense | Any | RALPH_STRUCTURED |
| 50K-100K | Sparse | Any | RALPH_STRUCTURED |
| 50K-100K | Dense | Any | RLM_BASIC |
| >100K | Any | Any | RLM_FULL |

## Output Format

```json
{
  "token_count": 45000,
  "density": 0.65,
  "complexity": "moderate",
  "recommended_strategy": "RALPH_STRUCTURED",
  "reasoning": "Document has moderate density with interconnected code modules. RALPH's iterative approach will handle the 45K tokens efficiently.",
  "estimated_cost": {
    "claude-3-sonnet": 0.15,
    "gpt-4o": 0.12
  },
  "chunk_suggestion": {
    "size": 4000,
    "overlap": 500,
    "total_chunks": 12
  }
}
```

## Integration with ContextFlow

```python
from contextflow.core.analyzer import ContextAnalyzer

# Uses this agent internally
analyzer = ContextAnalyzer()
result = await analyzer.analyze(documents)

# Access strategy recommendation
print(result.recommended_strategy)
print(result.reasoning)
```
