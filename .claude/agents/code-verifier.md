# Code Verifier Agent

Specialized subagent for verifying code quality and correctness.

## Purpose

Verify that generated or modified code:
1. Meets the original task requirements
2. Follows project coding standards (CLAUDE.md)
3. Has no obvious bugs or issues
4. Includes proper error handling

## Invocation

```python
from contextflow.agents import SubAgent

verifier = SubAgent(
    name="code-verifier",
    system_prompt="""
    You are a code verification expert. Your job is to verify that code:
    1. Accomplishes the stated task
    2. Follows Python best practices
    3. Has proper type hints
    4. Includes error handling
    5. Is well-documented

    Be thorough but concise. Report issues clearly.
    """
)

result = await verifier.execute(
    task="Verify this provider implementation",
    context=code_to_verify
)
```

## Verification Checklist

### Structural
- [ ] Proper imports
- [ ] Class/function organization
- [ ] No circular dependencies

### Type Safety
- [ ] All parameters typed
- [ ] Return types specified
- [ ] Optional types handled

### Error Handling
- [ ] Custom exceptions used
- [ ] Proper try/except blocks
- [ ] No bare except clauses

### Documentation
- [ ] Docstrings present
- [ ] Args documented
- [ ] Returns documented
- [ ] Raises documented

### Project Standards
- [ ] Uses ProviderLogger (not print)
- [ ] Async functions use await
- [ ] No hardcoded credentials

## Output Format

```json
{
  "passed": true/false,
  "score": 0.0-1.0,
  "issues": [
    {
      "severity": "error|warning|info",
      "location": "line 42",
      "message": "Missing type hint for parameter 'data'"
    }
  ],
  "suggestions": [
    "Consider adding retry logic for API calls"
  ]
}
```
