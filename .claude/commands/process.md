# /process - Process Documents with ContextFlow

Process documents using automatic strategy selection.

## Usage
```
/process <task> <file_or_directory>
```

## Workflow

1. **Analyze Context**
   ```bash
   python -c "
   from contextflow.core.analyzer import ContextAnalyzer
   analyzer = ContextAnalyzer()
   result = analyzer.analyze('$ARGUMENTS')
   print(f'Tokens: {result.token_count}')
   print(f'Strategy: {result.recommended_strategy}')
   "
   ```

2. **Execute Strategy**
   ```bash
   python -c "
   import asyncio
   from contextflow import ContextFlow

   async def main():
       cf = ContextFlow(provider='claude')
       result = await cf.process(
           task='$TASK',
           documents=['$FILE'],
           strategy='auto'
       )
       print(result.answer)

   asyncio.run(main())
   "
   ```

3. **Verify Result**
   - Check output matches task requirements
   - Verify no errors in execution
   - Confirm strategy was appropriate

## Examples

```bash
# Summarize a large codebase
/process "Summarize the main functionality" src/

# Find all API endpoints
/process "List all API endpoints with their HTTP methods" api/routes.py

# Analyze dependencies
/process "What external dependencies does this project use?" pyproject.toml
```
