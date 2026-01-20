# Verification Protocol

ContextFlow includes a built-in verification system to ensure output quality and reliability. This documentation covers the verification protocol, check types, configuration, and custom verification implementation.

## Overview

The verification protocol validates AI outputs against configurable criteria before returning results. This helps ensure:

- Output completeness
- Format compliance
- Factual consistency
- Safety standards

## Boris Step 13 Protocol

ContextFlow implements the "Boris Step 13" verification protocol, a systematic approach to output validation.

### Protocol Flow

```
┌─────────────────────────────────────────────────────────┐
│              Boris Step 13 Verification                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                                       │
│  │  AI Output   │                                       │
│  └──────┬───────┘                                       │
│         ▼                                                │
│  ┌──────────────────────────────────────┐                   │
│  │     Verification Checks          │                   │
│  │  ┌────────────────────────────┐  │                   │
│  │  │ 1. Format Validation       │  │                   │
│  │  │ 2. Completeness Check      │  │                   │
│  │  │ 3. Accuracy Assessment     │  │                   │
│  │  │ 4. Consistency Validation  │  │                   │
│  │  │ 5. Safety Check            │  │                   │
│  │  └────────────────────────────┘  │                   │
│  └──────────────┬───────────────────┘                   │
│                 ▼                                        │
│  ┌──────────────────────────────────┐                   │
│  │      Aggregate Results           │                   │
│  │  Score: 0.87 / 1.0               │                   │
│  │  Passed: 4/5 checks              │                   │
│  └──────────────┬───────────────────┘                   │
│                 ▼                                        │
│         ┌──────────────┐                                │
│         │   Pass?      │                                │
│         └──────┬───────┘                                │
│           Yes  │   No                                   │
│           ┌────┴────┐                                   │
│           ▼         ▼                                   │
│     ┌─────────┐ ┌─────────┐                            │
│     │ Return  │ │  Retry  │                            │
│     │ Output  │ │ or Warn │                            │
│     └─────────┘ └─────────┘                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Check Types

### Format Check

Validates that the output matches expected formatting requirements.

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["format"],
        "format_rules": {
            "expected_format": "json",  # "json" | "markdown" | "text" | "code"
            "schema": my_json_schema,    # Optional JSON schema
            "code_language": "python"    # For code format
        }
    }
)
```

**What it checks:**
- JSON validity (if expected)
- Markdown structure
- Code syntax
- Required sections

### Completeness Check

Ensures the output addresses all aspects of the request.

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["completeness"],
        "completeness_rules": {
            "required_sections": ["introduction", "body", "conclusion"],
            "min_length": 500,
            "max_length": 5000,
            "required_elements": ["example", "explanation"]
        }
    }
)
```

**What it checks:**
- All required sections present
- Minimum content length
- Required elements included
- Questions answered

### Accuracy Check

Validates factual consistency and logical soundness.

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["accuracy"],
        "accuracy_rules": {
            "cross_reference": True,
            "fact_check_context": True,
            "logical_consistency": True
        }
    }
)
```

**What it checks:**
- Internal consistency
- Alignment with provided context
- Logical soundness
- No contradictions

### Consistency Check

Ensures the output maintains consistency throughout.

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["consistency"],
        "consistency_rules": {
            "terminology_consistent": True,
            "tone_consistent": True,
            "style_guide": "technical"
        }
    }
)
```

**What it checks:**
- Consistent terminology
- Consistent tone
- Style guide adherence
- No conflicting statements

### Safety Check

Validates that output meets safety standards.

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["safety"],
        "safety_rules": {
            "no_harmful_content": True,
            "no_pii": True,
            "no_sensitive_data": True,
            "content_policy": "strict"
        }
    }
)
```

**What it checks:**
- No harmful content
- No PII exposure
- No sensitive data leakage
- Policy compliance

## Configuration

### Basic Configuration

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["format", "completeness", "accuracy"],
        "strict_mode": False
    }
)
```

### Full Configuration

```python
flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["format", "completeness", "accuracy", "consistency", "safety"],
        "strict_mode": True,
        "threshold": 0.8,
        "max_retries": 2,
        "retry_on_failure": True,

        # Check-specific rules
        "format_rules": { ... },
        "completeness_rules": { ... },
        "accuracy_rules": { ... },
        "consistency_rules": { ... },
        "safety_rules": { ... },

        # Custom checks
        "custom_checks": [...]
    }
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | True | Enable verification |
| `checks` | list[str] | all | Checks to run |
| `strict_mode` | bool | False | Fail on any issue |
| `threshold` | float | 0.7 | Minimum score to pass |
| `max_retries` | int | 1 | Retry count on failure |
| `retry_on_failure` | bool | True | Auto-retry on failure |

## Custom Verification

### Creating Custom Checks

```python
from typing import Any, Protocol
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a verification check."""
    check: str
    passed: bool
    score: float  # 0-1
    message: str | None = None
    details: dict[str, Any] | None = None
    suggestions: list[str] | None = None


class CustomCheck(Protocol):
    """Protocol for custom verification checks."""
    name: str
    description: str | None

    async def check(
        self,
        output: str,
        context: dict[str, Any] | None = None
    ) -> CheckResult:
        ...


@dataclass
class BrandComplianceCheck:
    """Ensures output follows brand guidelines."""
    name: str = "brand-compliance"
    description: str = "Ensures output follows brand guidelines"

    async def check(
        self,
        output: str,
        context: dict[str, Any] | None = None
    ) -> CheckResult:
        brand_terms = ["Acme Corp", "ACME", "Acme"]
        forbidden_terms = ["competitor", "cheap", "basic"]

        issues: list[str] = []
        score = 1.0

        # Check for forbidden terms
        for term in forbidden_terms:
            if term.lower() in output.lower():
                issues.append(f'Contains forbidden term: "{term}"')
                score -= 0.2

        # Check brand term usage
        has_brand_mention = any(term in output for term in brand_terms)
        if not has_brand_mention and context and context.get("require_brand_mention"):
            issues.append("Missing brand mention")
            score -= 0.1

        return CheckResult(
            check="brand-compliance",
            passed=score >= 0.7,
            score=max(0, score),
            message="; ".join(issues) if issues else "Brand compliant"
        )


custom_brand_check = BrandComplianceCheck()

flow = ContextFlow(
    verification={
        "enabled": True,
        "checks": ["format", "completeness"],
        "custom_checks": [custom_brand_check]
    }
)
```

### Custom Check Interface

```python
from typing import Any, Literal, Protocol
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a verification check."""
    check: str
    passed: bool
    score: float  # 0-1
    message: str | None = None
    details: dict[str, Any] | None = None
    suggestions: list[str] | None = None


@dataclass
class ProcessMetadata:
    """Metadata about the current processing context."""
    ...


class CustomCheck(Protocol):
    """
    Protocol for custom verification checks.

    Attributes:
        name: Unique identifier for this check
        description: Optional human-readable description
        priority: Higher values run first (optional)
    """
    name: str
    description: str | None
    priority: int | None  # Higher runs first

    async def check(
        self,
        output: str,
        context: dict[str, Any] | None = None,
        metadata: ProcessMetadata | None = None
    ) -> CheckResult:
        """Execute the verification check."""
        ...

    def on_failure(
        self,
        result: CheckResult,
        attempt: int
    ) -> Literal["retry", "fail", "warn"]:
        """Optional: Custom retry logic on check failure."""
        ...
```

### Advanced Custom Check

```python
from typing import Any, Literal
from dataclasses import dataclass, field


@dataclass
class CodeQualityCheck:
    """Validates code output quality."""
    name: str = "code-quality"
    description: str = "Validates code output quality"
    priority: int = 10

    async def check(
        self,
        output: str,
        context: dict[str, Any] | None = None,
        metadata: Any | None = None
    ) -> CheckResult:
        issues: list[str] = []
        suggestions: list[str] = []
        score = 1.0

        # Check for common issues
        if "TODO" in output:
            issues.append("Contains TODO comments")
            suggestions.append("Complete all TODO items before submission")
            score -= 0.1

        if "print(" in output:
            issues.append("Contains debug logging")
            suggestions.append("Remove print statements")
            score -= 0.05

        # Check for documentation
        if '"""' not in output and "#" not in output:
            issues.append("Missing code comments")
            suggestions.append("Add documentation comments")
            score -= 0.15

        return CheckResult(
            check="code-quality",
            passed=score >= 0.7,
            score=score,
            message="; ".join(issues) if issues else "Code quality passed",
            details={"issue_count": len(issues)},
            suggestions=suggestions
        )

    def on_failure(
        self,
        result: CheckResult,
        attempt: int
    ) -> Literal["retry", "fail", "warn"]:
        if attempt < 2 and result.score > 0.5:
            return "retry"
        return "warn" if result.score > 0.4 else "fail"


code_quality_check = CodeQualityCheck()
```

## Verification Results

### Accessing Results

```python
result = await flow.process(
    input="Generate a product description",
    options={"verify_output": True}
)

if result.verification:
    print("Passed:", result.verification.passed)
    print("Score:", result.verification.score)

    for check in result.verification.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  {check.check}: {status} ({check.score})")
        if check.message:
            print(f"    {check.message}")

    if result.verification.issues:
        print("Issues:")
        for issue in result.verification.issues:
            print(f"  [{issue.severity}] {issue.message}")
```

### VerificationResult Type

```python
from typing import Literal
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a single verification check."""
    check: str
    passed: bool
    score: float  # 0-1
    message: str | None = None
    details: dict[str, Any] | None = None
    suggestions: list[str] | None = None


@dataclass
class VerificationIssue:
    """An issue found during verification."""
    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    suggestion: str | None = None


@dataclass
class VerificationResult:
    """Complete verification results."""
    passed: bool
    score: float  # Aggregate score 0-1
    checks: list[CheckResult]
    issues: list[VerificationIssue] | None = None
    retry_count: int | None = None
    duration: float | None = None
```

## Best Practices

1. **Start with basic checks** - Enable format and completeness first
2. **Use strict mode in production** - Ensure quality before deployment
3. **Create domain-specific checks** - Custom checks for your use case
4. **Monitor verification metrics** - Track pass rates and common issues
5. **Balance strictness and usability** - Too strict can cause excessive retries

---

## See Also

- [API Reference](./api-reference.md)
- [Strategies](./strategies.md)
- [Getting Started](./getting-started.md)
