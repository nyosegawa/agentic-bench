# AGENTS.md
## Absolute Rules
1. Instruction budget rule: keep this AGENTS.md in 20-30 lines.
2. No backward compatibility: never preserve legacy behavior/interfaces.
3. Commit and push regularly in small, reviewable increments.
4. Refactor periodically to reduce complexity and technical debt.
5. Always use the latest versions for libraries/dependencies, and research their usage thoroughly before introducing them.
## Development Policy
1. This repository is TDD-first by default.
2. Always execute Red -> Green -> Refactor.
3. Start with a failing test before production code.
4. If tests are hard to write, create seams (split function, trait, adapter) and test through them.
5. GUI features start with domain/backend logic (TDD), then reflect in GUI. Never GUI-first.
## Required Flow Per Change
1. Define expected behavior as a test.
2. Run tests and confirm failure (Red).
3. Implement the minimal fix (Green).
4. Refactor safely with tests green.
5. Re-run full relevant scope before finishing.
## Python Verification Loop
1. `pytest`
2. `ruff check .`
3. `ruff format --check .`
