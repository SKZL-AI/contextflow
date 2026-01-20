# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. Do NOT Open a Public Issue

Security vulnerabilities should not be disclosed publicly until they have been addressed.

### 2. Report Privately

Please report security vulnerabilities by emailing:

**security@contextflow.ai** (or create a private security advisory on GitHub)

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (critical: 24-72 hours, high: 1-2 weeks, medium: 1 month)

### 4. Disclosure

Once a fix is available:
1. We will release a patched version
2. We will publish a security advisory
3. We will credit the reporter (unless they prefer anonymity)

---

## Security Best Practices

### API Keys

```python
# NEVER hardcode API keys
# Bad
client = Anthropic(api_key="sk-ant-...")

# Good - Use environment variables
import os
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

### Environment Variables

- Use `.env` files for local development
- NEVER commit `.env` to version control
- Use secrets management in production (AWS Secrets Manager, HashiCorp Vault, etc.)

### Network Security

- Always use HTTPS in production
- Configure CORS appropriately
- Use rate limiting for API endpoints

### Input Validation

- All user inputs are validated via Pydantic models
- File paths are sanitized
- SQL injection prevented via parameterized queries (SQLite)

### Dependencies

- Dependencies are pinned in `poetry.lock`
- Regular security audits via `safety` and `bandit`
- Dependabot alerts enabled

---

## Security Features

### Built-in Protections

1. **Input Validation** - Pydantic models validate all inputs
2. **Error Handling** - Sensitive information not leaked in errors
3. **Logging** - Structured logging without sensitive data
4. **Rate Limiting** - Configurable rate limits on API
5. **CORS** - Configurable cross-origin policies

### LLM-Specific Security

1. **Prompt Injection** - Users should implement prompt guards for production
2. **Output Filtering** - Verification protocol can filter harmful outputs
3. **Token Limits** - Configurable limits prevent runaway costs
4. **Provider Isolation** - Each provider is isolated

---

## Security Checklist for Production

```
[ ] API keys stored in secure secrets manager
[ ] HTTPS enabled with valid certificate
[ ] CORS configured for specific origins only
[ ] Rate limiting enabled
[ ] Logging configured (without sensitive data)
[ ] Regular dependency updates
[ ] Input validation enabled
[ ] Error messages sanitized
[ ] Session management secured
[ ] Network policies configured
```

---

## Known Security Considerations

### 1. RLM Code Execution

The RLM strategy includes a REPL environment for code execution. While sandboxed:
- Only safe builtins are available
- External libraries are restricted
- Timeout limits prevent infinite loops

**Recommendation**: In high-security environments, consider disabling RLM or adding additional sandboxing.

### 2. RAG Document Handling

Documents added to RAG are stored in memory:
- No encryption at rest (memory-based)
- Documents are not persisted by default
- Consider sensitivity of indexed content

### 3. Session Data

Session data is stored in SQLite:
- Not encrypted by default
- Consider database encryption for sensitive use cases

---

## Contact

For security-related inquiries:
- Email: security@contextflow.ai
- GitHub Security Advisories: [Create Advisory](https://github.com/contextflow/contextflow/security/advisories/new)

Thank you for helping keep ContextFlow secure!
