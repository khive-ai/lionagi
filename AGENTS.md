# Repository Guidelines

## Project Structure & Module Organization
- Core runtime lives in `lionagi/` with subpackages for adapters, operations, session orchestration, protocol definitions, and tool integrations.
- Reusable assets, demos, and reference notebooks sit under `assets/`, `cookbooks/`, and `notebooks/`; automation scripts are in `scripts/`.
- Tests mirror the package layout in `tests/` (for example `tests/session/`, `tests/operations/`) with shared fixtures in `tests/fixtures/`.
- Documentation sources are maintained in `docs/` (MkDocs) and long-form guides in `CLAUDE.md`; published artifacts land in `site/` after builds.

## Build, Test, and Development Commands
- Install or update the environment with `uv sync --dev`; this pulls project and tooling dependencies from `pyproject.toml` / `uv.lock`.
- Run the full suite via `uv run pytest`; pass `-m "not slow"` or `-k session` to target subsets while retaining pytest defaults (parallel, strict markers).
- Format and lint before committing: `uv run black .`, `uv run ruff check lionagi tests`, and `pre-commit run --all-files` for the configured hook chain.

## Coding Style & Naming Conventions
- Follow Black’s 79-character line length and isort’s Black profile; keep imports grouped standard/third-party/local.
- Prefer explicit type hints and Pydantic models for structured data; name modules and directories in snake_case, classes in PascalCase, async coroutines with `_async` suffix when clarity helps.
- Avoid introducing new global state; prefer dependency injection through service or session layers within `lionagi/`.

## Testing Guidelines
- Place unit tests adjacent to the relevant module subtree (`tests/operations/test_branch.py`, etc.) using filenames that start with `test_` and classes prefixed `Test`.
- Leverage pytest markers declared in `pyproject.toml` (`unit`, `integration`, `slow`, `performance`); mark long-running or network-dependent flows to keep CI lean.
- Maintain coverage by extending fixtures and factories in `tests/fixtures/`; include regression cases for bugs and concurrency edge scenarios.

## Commit & Pull Request Guidelines
- Match the existing history: concise, sentence-case summaries in the imperative voice (e.g., `Improve session retry logic`).
- Each PR should describe scope, testing evidence (`uv run pytest` output, linting), linked GitHub issues, and screenshots or logs for UX-facing changes.
- Keep changesets focused, update `docs/` or `mkdocs.yml` alongside API changes, and request reviews once CI is green.

## Documentation & Configuration Notes
- Keep `.env.example` aligned with new configuration keys; avoid committing real secrets or API tokens.
- When altering public interfaces, update `docs/` and regenerate the MkDocs site with `uv run mkdocs build` to verify navigation.
