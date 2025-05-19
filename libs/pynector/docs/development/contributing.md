# Contributing to Pynector

We love your input! We want to make contributing to Pynector as easy and
transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as
accept pull requests.

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Submit your pull request!

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/pynector.git
cd pynector

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies with development extras
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/pynector

# Run specific test files
uv run pytest tests/test_client.py
```

## Coding Standards

- We follow [PEP 8](https://pep8.org/) for Python code style.
- Use [Black](https://black.readthedocs.io/) for code formatting.
- Sort imports using [isort](https://pycqa.github.io/isort/).

## Documentation

- We use MkDocs with Material theme for documentation.
- Please update documentation for any changes that affect public APIs.
- To build and preview the documentation:

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
```

## Golden Path Workflow

Pynector follows a standardized development workflow:

1. **Research**: For significant features, start with a research report
2. **Spec**: Create a technical design specification
3. **Plan + Tests**: Create an implementation plan and tests
4. **Code + Green tests**: Implement according to the plan
5. **Commit**: Using conventional commits
6. **PR/CI**: Submit PR and ensure CI checks pass
7. **Review**: Code review process
8. **Documentation**: Update documentation
9. **Merge & clean**: Merge PR and clean up branch

For more details, see our
[Golden Path documentation](https://github.com/ohdearquant/pynector/blob/main/.roo/rules/002_golden_path.md).

## License

By contributing, you agree that your contributions will be licensed under the
project's
[MIT License](https://github.com/ohdearquant/pynector/blob/main/LICENSE).
