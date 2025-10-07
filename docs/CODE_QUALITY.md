# Code Quality and Linting Setup

This document describes the code quality tools and linting setup for the Performance Dashboard project.

## Overview

The project uses a comprehensive set of tools to ensure code quality, consistency, and security:

- **Ruff**: Fast Python linter (replaces flake8, pylint, and more)
- **Black**: Code formatter
- **isort**: Import sorting
- **mypy**: Static type checking
- **pytest**: Testing framework with coverage
- **pre-commit**: Git hooks for automated checks

## Quick Start

### 1. Set up development environment

```bash

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### 2. Run code quality checks

```bash
# Format code automatically
make format

# Run all linting checks
make lint

# Run type checking
make type-check

# Run all CI checks locally
make ci-local
```

## Tools Configuration

### Ruff Configuration

Ruff is configured in `pyproject.toml` and includes:

- **Enabled rules**: E, W, F, I, B, C4, UP, ARG, C90, T20, SIM, ICN
- **Line length**: 88 characters (matches Black)
- **Max complexity**: 12

Common rules:

- `E501`: Line too long (handled by Black)
- `F401`: Unused imports
- `B008`: Function calls in argument defaults
- `I001`: Import sorting

### Black Configuration

Black is the primary code formatter with:

- **Line length**: 88 characters
- **Target versions**: Python 3.9+
- **Style**: PEP 8 compliant

### mypy Configuration

Type checking configuration:

- **Python version**: 3.9+
- **Missing imports**: Ignored for external libraries
- **Untyped defs**: Allowed (can be made stricter)

### Pre-commit Hooks

The following hooks run on every commit:

1. **Trailing whitespace removal**
2. **End-of-file fixer**
3. **YAML/JSON validation**
4. **Ruff linting and formatting**
5. **mypy type checking**
6. **Bandit security scanning**
7. **pydocstyle docstring checking**
8. **Prettier formatting** (YAML/Markdown)

## GitHub Actions CI/CD Workflow

The CI/CD pipeline is defined in `.github/workflows/ci.yml` and runs automatically on:

- **Push** to `main` or `develop` branches
- **Pull requests** targeting `main` or `develop`
- **Manual trigger** via workflow_dispatch

### Workflow Jobs

The workflow includes the following parallel jobs:

#### 1. Lint & Format Check (`lint`)

Runs all pre-commit hooks to validate code quality:

- **Trailing whitespace** removal
- **End of file** fixes
- **YAML/TOML validation**
- **Ruff linting** with auto-fixes
- **Ruff formatting** (Python code formatter)
- **mypy** type checking
- **Bandit** security scanning
- **pydocstyle** docstring validation (Google convention)
- **Prettier** for YAML/Markdown formatting

**Status**: Required - PR cannot merge if this fails

#### 2. Type Check (`type-check`)

Runs mypy static type checking:

- Generates JUnit XML report
- Uploads `mypy-report.xml` as artifact
- **Status**: Advisory only (continues on error)

#### 3. Test & Coverage (`test`)

Runs the test suite with coverage reporting:

- Executes pytest with coverage
- Generates coverage reports (XML, HTML, terminal)
- Uploads coverage to Codecov (if configured)
- Uploads coverage artifacts for review
- **Status**: Required - PR cannot merge if tests fail

#### 4. Documentation Check (`docs-check`)

Validates Python docstrings:

- Checks Google-style docstring format
- Ensures all public modules/functions documented
- **Status**: Advisory only (continues on error)

### Viewing CI Results

1. Navigate to the **Actions** tab in the GitHub repository
2. Select the workflow run to view job details
3. Click on individual jobs to see logs and artifacts
4. Download artifacts (test reports, coverage) from the job summary page

## Customizing Rules

### Disabling Specific Rules

For specific lines:

```python
# ruff: noqa: E501
very_long_line_that_exceeds_the_limit_but_is_necessary_for_some_reason()

# mypy: ignore
problematic_type_annotation = some_complex_function()
```

For entire files:

```python
# ruff: noqa
```

### Project-wide Exceptions

Edit `pyproject.toml`:

```toml
[tool.ruff]
ignore = [
    "E501",  # line too long
    "T201",  # print statements
]

[tool.ruff.per-file-ignores]
"tests/*" = ["T201"]  # Allow prints in tests
```

### Getting Help

1. Check tool documentation:
   - [Ruff](https://beta.ruff.rs/docs/)
   - [Black](https://black.readthedocs.io/)
   - [mypy](https://mypy.readthedocs.io/)

2. Run with verbose flags:

   ```bash
   ruff check . --verbose
   mypy . --verbose
   ```

3. Check GitHub Actions logs for specific error details:
   - Go to the **Actions** tab in the repository
   - Click on the failed workflow run
   - Review individual job logs for detailed error messages
