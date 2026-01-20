# ContextFlow Strategies

ContextFlow uses three distinct strategies optimized for different types of tasks. Understanding when and how each strategy works helps you get the best results.

## Strategy Overview

| Strategy | Purpose | Best For | Complexity |
|----------|---------|----------|------------|
| **GSD** | Get Stuff Done | Quick tasks, simple queries | Low |
| **RALPH** | Research-Analyze-Learn-Plan-Help | Multi-step analysis, research | Medium |
| **RLM** | Reasoning Language Model | Complex reasoning, planning | High |

## GSD Strategy

### What is GSD?

GSD (Get Stuff Done) is the lightweight, fast-execution strategy designed for straightforward tasks that don't require extensive reasoning or multiple steps.

### When GSD is Used

GSD is automatically selected when:
- The task is clearly defined and simple
- No complex analysis is required
- The expected output is straightforward
- Speed is prioritized over depth

### GSD Characteristics

```
┌─────────────────────────────────┐
│           GSD Flow              │
├─────────────────────────────────┤
│  Input → Process → Output       │
│                                 │
│  • Single-pass execution        │
│  • Minimal context expansion    │
│  • Fast response time           │
│  • Lower token usage            │
└─────────────────────────────────┘
```

### Example Use Cases

- Simple Q&A: "What is the capital of Japan?"
- Basic transformations: "Convert this text to uppercase"
- Code formatting: "Format this JSON"
- Quick calculations: "What is 15% of 250?"

### Configuration

```python
flow = ContextFlow(
    strategy_config={
        "gsd": {
            "max_tokens": 1000,
            "temperature": 0.3,
            "direct_response": True,
            "skip_verification": False
        }
    }
)
```

### Forcing GSD

```python
result = await flow.process(
    input="Translate 'hello' to Spanish",
    strategy="gsd"
)
```

---

## RALPH Strategy

### What is RALPH?

RALPH (Research-Analyze-Learn-Plan-Help) is a structured approach for tasks requiring systematic analysis and multi-step reasoning.

### The RALPH Phases

```
┌─────────────────────────────────────────────────────────┐
│                    RALPH Process                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Research │ → │ Analyze  │ → │  Learn   │            │
│  └──────────┘   └──────────┘   └──────────┘            │
│       ↓                                                  │
│  ┌──────────┐   ┌──────────┐                            │
│  │   Plan   │ → │   Help   │                            │
│  └──────────┘   └──────────┘                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 1. Research Phase
- Gathers relevant information from context
- Identifies key concepts and entities
- Expands understanding of the domain

#### 2. Analyze Phase
- Breaks down the problem into components
- Identifies relationships and patterns
- Evaluates different perspectives

#### 3. Learn Phase
- Synthesizes insights from analysis
- Updates understanding based on findings
- Identifies knowledge gaps

#### 4. Plan Phase
- Develops structured approach to response
- Prioritizes information
- Organizes logical flow

#### 5. Help Phase
- Generates the final response
- Ensures completeness
- Validates against original request

### When RALPH is Used

RALPH is automatically selected when:
- The task requires analysis or research
- Multiple perspectives need consideration
- The answer benefits from structured thinking
- Moderate complexity is detected

### Example Use Cases

- Research tasks: "Compare React and Vue frameworks"
- Analysis: "Analyze the sentiment of these reviews"
- Summarization: "Summarize key points from this document"
- Explanations: "Explain how blockchain consensus works"

### Configuration

```python
flow = ContextFlow(
    strategy_config={
        "ralph": {
            "max_iterations": 5,
            "temperature": 0.5,
            "phases": ["research", "analyze", "learn", "plan", "help"],
            "enable_phase_logging": True,
            "min_confidence": 0.7
        }
    }
)
```

### RALPH with Phase Control

```python
result = await flow.process(
    input="Analyze customer feedback trends",
    strategy="ralph",
    options={
        "phases": ["research", "analyze", "help"],  # Skip learn and plan
        "max_iterations": 3
    }
)

# Access phase details
print(result.metadata.phases)
# [
#   {"name": "research", "duration": 1200, "tokens": 450},
#   {"name": "analyze", "duration": 800, "tokens": 380},
#   {"name": "help", "duration": 600, "tokens": 520}
# ]
```

---

## RLM Strategy

### What is RLM?

RLM (Reasoning Language Model) is the most sophisticated strategy, designed for complex tasks requiring deep reasoning, iterative refinement, and multi-step problem solving.

### RLM Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      RLM Process                               │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐                                              │
│  │   Input     │                                              │
│  └──────┬──────┘                                              │
│         ▼                                                      │
│  ┌─────────────┐     ┌─────────────┐                          │
│  │ Decompose   │ ──→ │  Reason     │ ←──┐                     │
│  │  Problem    │     │   Step      │    │                     │
│  └─────────────┘     └──────┬──────┘    │                     │
│                             │           │  Iterate            │
│                             ▼           │                     │
│                      ┌─────────────┐    │                     │
│                      │  Evaluate   │ ───┘                     │
│                      │  Progress   │                          │
│                      └──────┬──────┘                          │
│                             │ Complete                        │
│                             ▼                                 │
│                      ┌─────────────┐                          │
│                      │  Synthesize │                          │
│                      │   Output    │                          │
│                      └─────────────┘                          │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### RLM Capabilities

1. **Problem Decomposition** - Breaks complex problems into manageable sub-problems
2. **Iterative Reasoning** - Refines understanding through multiple passes
3. **Self-Evaluation** - Continuously assesses progress and quality
4. **Backtracking** - Can revisit and revise earlier conclusions
5. **Synthesis** - Combines insights into coherent output

### When RLM is Used

RLM is automatically selected when:
- The task involves complex reasoning
- Multiple interdependent steps are required
- Problem-solving or planning is needed
- High-quality, thorough output is expected

### Example Use Cases

- System design: "Design a distributed caching system"
- Complex coding: "Implement a red-black tree with balancing"
- Strategic planning: "Create a go-to-market strategy"
- Problem solving: "Debug this complex race condition"

### Configuration

```python
flow = ContextFlow(
    strategy_config={
        "rlm": {
            "max_depth": 10,
            "temperature": 0.7,
            "reasoning_style": "chain-of-thought",
            "enable_backtracking": True,
            "evaluation_threshold": 0.8,
            "max_iterations": 15
        }
    }
)
```

### RLM with Reasoning Trace

```python
result = await flow.process(
    input="Design a fault-tolerant message queue",
    strategy="rlm",
    options={
        "include_reasoning_trace": True
    }
)

# Access reasoning trace
for step in result.metadata.reasoning_trace:
    print(f"Step {step.index}: {step.description}")
    print(f"  Confidence: {step.confidence}")
    print(f"  Tokens: {step.tokens}")
```

---

## Auto-Routing

### How Auto-Routing Works

When `strategy="auto"` (default), ContextFlow analyzes the input to determine the best strategy:

```
┌─────────────────────────────────────────────────────────┐
│                  Auto-Routing Process                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐                                           │
│  │  Input   │                                           │
│  └────┬─────┘                                           │
│       ▼                                                  │
│  ┌──────────────────────┐                               │
│  │  Complexity Analysis │                               │
│  │  ─────────────────── │                               │
│  │  • Token count       │                               │
│  │  • Question type     │                               │
│  │  • Keywords          │                               │
│  │  • Context size      │                               │
│  └────────┬─────────────┘                               │
│           ▼                                              │
│  ┌──────────────────────┐                               │
│  │   Strategy Scoring   │                               │
│  │  ─────────────────── │                               │
│  │  GSD:   0.2          │                               │
│  │  RALPH: 0.7  ←────── │  Selected                    │
│  │  RLM:   0.5          │                               │
│  └──────────────────────┘                               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Signal | GSD | RALPH | RLM |
|--------|-----|-------|-----|
| Simple question words (what, when) | +0.3 | 0 | 0 |
| Analysis keywords (analyze, compare) | 0 | +0.4 | +0.1 |
| Design/architecture keywords | 0 | +0.1 | +0.4 |
| Short input (<50 tokens) | +0.2 | 0 | -0.1 |
| Long input (>500 tokens) | -0.1 | +0.2 | +0.2 |
| Code/technical context | 0 | +0.2 | +0.3 |
| Multi-step indicators | -0.2 | +0.3 | +0.3 |

### Analyzing Routing Decisions

```python
analysis = await flow.analyze(
    input="Create a recommendation engine using collaborative filtering"
)

print(analysis)
# {
#   "recommended_strategy": "rlm",
#   "confidence": 0.82,
#   "scores": {
#     "gsd": 0.15,
#     "ralph": 0.58,
#     "rlm": 0.82
#   },
#   "signals": [
#     {"signal": "design_keyword", "weight": 0.4, "strategy": "rlm"},
#     {"signal": "technical_context", "weight": 0.3, "strategy": "rlm"},
#     {"signal": "multi_step", "weight": 0.3, "strategy": "rlm"}
#   ],
#   "reasoning": "Complex system design requiring iterative development"
# }
```

---

## Strategy Comparison

### Performance Characteristics

| Metric | GSD | RALPH | RLM |
|--------|-----|-------|-----|
| Typical Latency | <2s | 3-8s | 5-30s |
| Token Usage | Low | Medium | High |
| Output Depth | Surface | Moderate | Deep |
| Iteration Count | 1 | 3-5 | 5-15 |

### Choosing the Right Strategy

```python
# Use GSD for quick, simple tasks
await flow.process(
    input="Format this date: 2024-01-15",
    strategy="gsd"
)

# Use RALPH for analysis and research
await flow.process(
    input="Analyze the pros and cons of microservices",
    strategy="ralph"
)

# Use RLM for complex reasoning
await flow.process(
    input="Design a real-time bidding system for ad tech",
    strategy="rlm"
)

# Use auto when unsure (recommended)
await flow.process(
    input="Your task here",
    strategy="auto"
)
```

---

## Custom Strategy Configuration

### Strategy Presets

```python
from contextflow import ContextFlow, StrategyPresets

flow = ContextFlow(
    strategy_config={
        **StrategyPresets.balanced,  # Balanced defaults
        # Override specific settings
        "ralph": {
            **StrategyPresets.balanced["ralph"],
            "max_iterations": 7
        }
    }
)
```

### Available Presets

- `StrategyPresets.fast` - Optimized for speed
- `StrategyPresets.balanced` - Default balanced settings
- `StrategyPresets.thorough` - Optimized for quality
- `StrategyPresets.economical` - Optimized for token efficiency

---

## Next Steps

- [**API Reference**](./api-reference.md) - Detailed API documentation
- [**Verification**](./verification.md) - Ensure output quality
- [**Providers**](./providers.md) - Configure AI providers
