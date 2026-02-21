# AGENTS.md
## Absolute Rules
1. Instruction budget rule: keep this AGENTS.md in 20-30 lines.
2. No backward compatibility: never preserve legacy behavior/interfaces.
3. Commit and push regularly in small, reviewable increments.
4. Refactor periodically to reduce complexity and technical debt.
5. Always use the latest versions for libraries/dependencies, and research their usage thoroughly before introducing them.
## Development Policy
1. Write tests for core logic (config parsing, result aggregation, adapter interfaces).
2. External integrations (GPU cloud APIs, Chrome MCP) are tested via manual verification.
3. Keep provider adapters behind clean interfaces so core logic remains testable.
4. Never merge code that breaks existing tests.
## Required Flow Per Change
1. Implement the change.
2. Add or update tests for any core logic touched.
3. Run the verification loop below.
4. Confirm all checks pass before committing.
## Python Verification Loop
1. `pytest`
2. `ruff check .`
3. `ruff format --check .`
