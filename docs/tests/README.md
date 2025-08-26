# Documentation Testing

This directory contains test scripts for all code examples in the LionAGI documentation.

## Purpose

- Ensure all documentation code examples actually work
- Catch API changes that break examples
- Provide runnable versions of documentation snippets
- Serve as additional examples for users

## Structure

```
tests/
├── README.md                    # This file
├── test_quickstart.py          # Tests for quickstart examples
├── test_patterns.py            # Tests for orchestration patterns
├── test_core_concepts.py       # Tests for core concept examples
├── test_orchestration_guide.py # Tests for AI agent orchestration
├── test_cookbook/              # Tests for cookbook examples
│   ├── test_claim_extraction.py
│   ├── test_hr_automation.py
│   └── ...
└── run_all_tests.py           # Run all documentation tests
```

## Running Tests

### Test All Documentation
```bash
cd docs/tests
python run_all_tests.py
```

### Test Specific Section
```bash
python test_patterns.py
python test_quickstart.py
```

### Test Individual Example
```bash
python -m pytest test_quickstart.py::test_first_agent
```

## Adding New Tests

When adding new code examples to documentation:

1. Create corresponding test in this directory
2. Use same structure as documentation
3. Add assertions to verify expected behavior
4. Update run_all_tests.py to include new test

## Requirements

Tests require:
- Python 3.10+
- lionagi package installed
- API keys configured (OPENAI_API_KEY, etc.)
- pytest for test runner

## CI Integration

These tests should run on:
- Every PR that modifies documentation
- Nightly to catch API drift
- Before releases to ensure examples work