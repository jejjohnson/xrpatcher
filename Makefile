.PHONY: help install lint format typecheck test precommit docs docs-serve docs-deploy
.DEFAULT_GOAL = help

PKGROOT = xrpatcher

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup
install:  ## Install all dependencies
	uv sync --all-groups

##@ Quality
lint:  ## Lint code using ruff
	uv run --group lint ruff check ${PKGROOT}/

format:  ## Format code using ruff
	uv run --group lint ruff format ${PKGROOT}/
	uv run --group lint ruff check --fix ${PKGROOT}/

typecheck:  ## Type check code using ty
	uv run --group typecheck ty check ${PKGROOT}

##@ Testing
test:  ## Run tests with pytest
	uv run pytest -v

##@ Pre-commit
precommit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

##@ Docs
docs:  ## Build documentation
	@if [ ! -f mkdocs.yml ] || [ ! -d docs ]; then \
		echo "Error: MkDocs configuration (mkdocs.yml) or docs/ directory not found. Documentation tooling is not yet set up in this repository."; \
		exit 1; \
	fi
	uv run --group docs mkdocs build

docs-serve:  ## Serve documentation locally
	@if [ ! -f mkdocs.yml ] || [ ! -d docs ]; then \
		echo "Error: MkDocs configuration (mkdocs.yml) or docs/ directory not found. Documentation tooling is not yet set up in this repository."; \
		exit 1; \
	fi
	uv run --group docs mkdocs serve

docs-deploy:  ## Deploy documentation to GitHub Pages
	@if [ ! -f mkdocs.yml ] || [ ! -d docs ]; then \
		echo "Error: MkDocs configuration (mkdocs.yml) or docs/ directory not found. Documentation tooling is not yet set up in this repository."; \
		exit 1; \
	fi
	uv run --group docs mkdocs gh-deploy --force
