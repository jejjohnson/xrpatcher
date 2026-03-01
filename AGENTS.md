# Copilot Agent Guidelines

This file contains standing instructions for the Copilot coding agent working on this repository.

## Before Every Commit

Always verify all of the following pass before creating a commit or reporting progress:

1. **Tests** – `uv run pytest tests/ -q` must have 0 failures.
2. **Lint** – `uv run --group lint ruff check xrpatcher/` must report no issues.
3. **Format** – `uv run --group lint ruff format --check xrpatcher/` must report no files to reformat.
4. **Type checks** – run `uv run --group typecheck ty check xrpatcher` (or `make typecheck`) and fix any type errors in changed files.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[(<scope>)][!]: <description>
```

- `(scope)` is optional and describes the affected area (e.g., `feat(api): …`).
- `!` is optional and denotes a breaking change (e.g., `feat!: …`).
- `<description>` **must start with a lowercase letter** (enforced by CI).

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add ...`
- `feat(api): add new endpoint`
- `fix!: change default behaviour`
- `fix: ruff format violations in ...py`
- `chore: add AGENTS.md with workflow rules`

## Pull Request Title

PR titles must also follow Conventional Commits format (same rule as commit messages).

**Do not change** the PR title or description between sessions except to:
- Correct a conventional-commits format violation in the title.
- Append new items to the description checklist.

Never rewrite the existing description; only add to it.
