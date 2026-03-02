# Agent Guidelines

This file contains standing instructions for **all** coding agents working on this repository (Copilot, Claude, etc.).

## Before Every Commit

**All agents must** verify that every one of the following passes before creating a commit or reporting progress. No exceptions.

1. **Tests** – `uv run pytest -q` (or `make test`) must have 0 failures.
2. **Lint** – `uv run --group lint ruff check xrpatcher/` (or `make lint`) must report no issues.
3. **Format** – `uv run --group lint ruff format --check xrpatcher/` must report no files to reformat.
4. **Type checks** – `uv run --group typecheck ty check xrpatcher` (or `make typecheck`) must report no errors in changed files.

## Pull Request Descriptions

**Never replace or remove an existing PR title or description.** When reporting progress on a PR that already has a title and description, only append new checklist items or update the status of existing ones. The original content must be preserved in full.

This is a common failure mode: an agent called to make a small follow-up change will supply a fresh description scoped only to its own work, silently discarding all prior context. Always read the existing description first and treat it as the base.

## Documentation

This repo uses **MkDocs + Material + mkdocstrings + mkdocs-jupyter** for documentation.

- **Build locally**: `make docs-serve` (or `uv run --group docs mkdocs serve`)
- **Build static site**: `make docs` (or `uv run --group docs mkdocs build`)
- **Deploy to GitHub Pages**: `make docs-deploy` (or `uv run --group docs mkdocs gh-deploy --force`)
- **Auto-deploy**: the `pages.yml` workflow deploys automatically on every push to `main`

When writing docstrings, use **Google style** (enforced by `mkdocstrings` config).

Notebooks in `docs/` may be stored as `.ipynb` files or as `jupytext`-paired `.py` files.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[(<scope>)][!]: <description>
[optional body]
[optional footer(s)]
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
