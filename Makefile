#!make
.PHONY: help test test/unit test/integration test/integration-production-read-only test/unit/coverage coverage/html

.DEFAULT_GOAL := help

PYTEST := uv run python -m pytest
TESTS := tests/ --ignore=tests/integration
INTEGRATION_TESTS := tests/integration

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  \033[36mtest/unit\033[0m              Run unit tests"
	@echo "  \033[36mtest\033[0m                   Run unit tests (alias)"
	@echo "  \033[36mtest/integration\033[0m       Run integration + post-deploy tests"
	@echo "  \033[36mtest/integration-production-read-only\033[0m  Run the production GET-only smoke"
	@echo "  \033[36mtest/unit/coverage\033[0m     Run unit tests with coverage"
	@echo "  \033[36mcoverage/html\033[0m          Generate HTML coverage report"

test/unit: ## Run unit tests
	$(PYTEST) $(TESTS)

test: test/unit ## Run unit tests (alias)

test/integration: ## Run integration + post-deploy tests (skip themselves without a target)
	$(PYTEST) -v $(INTEGRATION_TESTS) -m "not production_read_only"

test/integration-production-read-only: ## Run only the production GET-only smoke
	$(PYTEST) -v tests/integration/test_production_read_only.py

test/unit/coverage: ## Run unit tests with coverage
	$(PYTEST) --cov=cowork $(TESTS)

coverage/html: ## Generate HTML coverage report
	$(PYTEST) --cov=cowork $(TESTS) --cov-report=html
