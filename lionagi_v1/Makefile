setup: ## install dev deps
	uv sync
test:
	pytest -q
lint:
	ruff check .
type:
	pyright
fmt:
	isort .
	black .
	ruff check . --fix
	deno fmt .