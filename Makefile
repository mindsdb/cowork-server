#!make
.PHONY: help test test/unit test/unit/coverage coverage/html

.DEFAULT_GOAL := help

PYTEST := uv run python -m pytest
TESTS := tests/

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  \033[36mtest/unit\033[0m              Run unit tests"
	@echo "  \033[36mtest\033[0m                   Run unit tests (alias)"
	@echo "  \033[36mtest/unit/coverage\033[0m     Run unit tests with coverage"
	@echo "  \033[36mcoverage/html\033[0m          Generate HTML coverage report"

test/unit: ## Run unit tests
	$(PYTEST) $(TESTS)

test: test/unit ## Run unit tests (alias)

test/unit/coverage: ## Run unit tests with coverage
	$(PYTEST) --cov=cowork $(TESTS)

coverage/html: ## Generate HTML coverage report
	$(PYTEST) --cov=cowork $(TESTS) --cov-report=html
