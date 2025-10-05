# Repository Guidelines

## Project Structure & Module Organization
The `lionagi/` package houses the agent runtime (flows, adapters, safety layers), while `libs/` captures experimental helpers intended for promotion into the core package. Tests live in `tests/` and mirror package paths one-to-one; use them as a map when exploring unfamiliar modules. Reference implementations and walkthroughs sit in `cookbooks/` and `notebooks/`, with production-ready assets in `docs/` and UI elements in `assets/`. CLI entry points (`main.py`, utilities under `scripts/`) provide runnable examples; keep new executables inside `scripts/`.

## Build, Test, and Development Commands
Synchronize dependencies with `uv sync` (includes optional dev tools) before editing. Key workflows:

```bash
uv run ruff check lionagi tests          # static analysis + lint
uv run black . --check                    # formatting guard (79 char lines)
uv run pytest                             # run full suite with xdist defaults
uv run pytest -m "not slow"               # quick feedback path
uv run python scripts/concat.py --help    # inspect packaged utilities
```

## Coding Style & Naming Conventions
Use 4-space indentation and keep lines under 79 characters (enforced by Black and Ruff). Import ordering follows `isort`'s Black profile; prefer explicit relative imports inside `lionagi/`. Name modules and functions with `snake_case`, classes with `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Document public APIs with concise docstrings and type hints; align new configuration with `pydantic` models already present in the package.

## Testing Guidelines
Pytest is configured in `pyproject.toml` with strict markers and timeouts—matching `tests/` naming `test_<feature>.py` ensures discovery. Tag long-running scenarios with `@pytest.mark.slow` or `@pytest.mark.integration`, and isolate async flows with `@pytest.mark.asyncio`. Maintain or improve coverage reported in `coverage.xml`; new features should add unit tests beside the implementation and, when applicable, cookbook or docs snippets demonstrating usage.

## Commit & Pull Request Guidelines
Follow the Conventional Commit style visible in recent history (`docs:`, `refactor:`, etc.), using imperative, <=72-character subjects and descriptive bodies. Each PR should include: a short summary of changes, linked issues or discussions, validation evidence (command output, screenshots for docs), and call out any new dependencies or environment variables. Keep branches focused; rebase on `main` before requesting review and confirm lint/tests pass locally.
