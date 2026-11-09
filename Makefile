GOINFRE ?= /goinfre/$(shell id -un)
export UV_CACHE_DIR ?= $(GOINFRE)/uv-cache
export UV_PROJECT_ENVIRONMENT ?= $(GOINFRE)/venvs/call_me_maybe
export HF_HOME ?= $(GOINFRE)/hf

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint
