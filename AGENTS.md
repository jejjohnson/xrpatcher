# Copilot Agent Guidelines

This file contains standing instructions for the Copilot coding agent working on this repository.

## Before Every Commit

Always verify all of the following pass before creating a commit or reporting progress:

1. **Tests** – `uv run pytest tests/ -q` must have 0 failures.
2. **Lint** – `uv run --group lint ruff check xrpatcher/` must report no issues.
3. **Format** – `uv run --group lint ruff format --check xrpatcher/` must report no files to reformat.
4. **Type checks** – if a `mypy` or `pyright` configuration exists, run it and fix any errors in changed files.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>: <description>
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add ...`
- `fix: ruff format violations in ...py`
- `chore: add AGENTS.md with workflow rules`

## Pull Request Title

PR titles must also follow Conventional Commits format (same rule as commit messages).

**Do not change** the PR title or description between sessions except to:
- Correct a conventional-commits format violation in the title.
- Append new items to the description checklist.

Never rewrite the existing description; only add to it.
