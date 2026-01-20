# /analyze - Analyze Context Complexity

Analyze documents to determine optimal processing strategy.

## Usage
```
/analyze <file_or_directory>
```

## Workflow

1. **Token Count**
   ```bash
   python -c "
   from contextflow.utils.tokens import TokenEstimator
   estimator = TokenEstimator()
   count = estimator.count_file('$ARGUMENTS')
   print(f'Token count: {count}')
   "
   ```

2. **Complexity Analysis**
   ```bash
   python -c "
   from contextflow.core.analyzer import ContextAnalyzer
   analyzer = ContextAnalyzer()
   result = analyzer.analyze('$ARGUMENTS')
   print(f'Complexity: {result.complexity}')
   print(f'Information Density: {result.density}')
   print(f'Recommended Strategy: {result.recommended_strategy}')
   "
   ```

3. **Report**
   - Token count and estimated cost
   - Recommended strategy with reasoning
   - Chunk breakdown (if applicable)

## Strategy Selection Matrix

| Tokens | Complexity | Strategy |
|--------|------------|----------|
| <10K | Low | GSD (Direct) |
| 10K-50K | Medium | RALPH (Iterative) |
| 50K-100K | High | RALPH or RLM |
| >100K | Any | RLM (Recursive) |

## Examples

```bash
# Analyze a single file
/analyze src/main.py

# Analyze entire directory
/analyze src/

# Analyze with specific model pricing
/analyze --model claude-3-opus src/
```
