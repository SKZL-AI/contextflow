# /verify - Verify ContextFlow Output

Verify that processing output meets task requirements.

## Usage
```
/verify <task> <output>
```

## Workflow (Boris Step 13)

1. **Self-Check**
   - Does output address the original task?
   - Is the format correct?

2. **Constraint-Check**
   - All requirements fulfilled?
   - No hallucinations or fabrications?

3. **Quality-Check**
   - Output completeness
   - Accuracy assessment

## Verification Protocol

```python
from contextflow.strategies.verification import VerificationProtocol

verifier = VerificationProtocol(provider='claude')

result = await verifier.verify(
    task="Find all API endpoints",
    output=processing_output,
    constraints=[
        "List HTTP method for each endpoint",
        "Include route path",
        "Note any authentication requirements"
    ]
)

if result.passed:
    print("Verification PASSED")
else:
    print(f"Verification FAILED: {result.issues}")
```

## Verification Result

```python
@dataclass
class VerificationResult:
    passed: bool
    confidence: float  # 0.0 - 1.0
    issues: List[str]
    suggestions: List[str]
    iterations: int
```

## Examples

```bash
# Verify summarization output
/verify "Summarize main features" "The application provides..."

# Verify with specific constraints
/verify --constraints "Must include error handling" "The code implements..."
```
