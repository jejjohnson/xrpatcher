.PHONY: help install_mamba install update_macos update_linux
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.10
NAME = xrpatcher
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
PKGROOT = xrpatcher
TESTS = ${PKGROOT}/tests

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Installation
install_mamba: ## Install mamba in base environment
	conda env create -f environment.yaml

install_conda: ## Install conda env in Linux
	mamba env create -f environment.yaml

update_conda: ## Update conda env in MACOS
	mamba env update -f environment.yaml

update_mamba: ## Update conda env in linux
	mamba env update -f environment.yaml

install_precommit: ## Install precommit tools
	mamba install pre-commit -c conda-forge
	pre-commit install --all-files

##@ Formatting
black:  ## Format code in-place using black.
	black ${PKGROOT}/ -l 79 .

isort:  ## Format imports in-place using isort.
	isort ${PKGROOT}/

format: ## Code styling - black, isort
		make black
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		make isort
		@printf "\033[1;34misort passes!\033[0m\n\n"

##@ Testing
test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v ${PKGROOT}
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"

precommit: ## Run recommit
		@printf "\033[1;34mRunning precommit routine...\033[0m\n\n"
		pre-commit run --all-files
		@printf "\033[1;34mFinished!\033[0m\n\n"

##@ JupyterBook
	
jb_build: ## Build Jupyterbook
	rm -rf jbook/_build/
	jupyter-book build jbook --all

jb_clean: ## Clean JupyterBook
	jupyter-book clean jbook
