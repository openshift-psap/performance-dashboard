# Test Suite Documentation

## Overview

Comprehensive test suite for the Performance Dashboard covering data processing, import scripts, styling, and integration workflows.

## Test Structure

```
tests/
├── __init__.py                      # Test package marker
├── conftest.py                      # Shared fixtures and configuration
├── test_data_processing.py          # Data processing functions (71 tests)
├── test_import_script.py            # Import script functions (46 tests)
└── test_integration.py              # Integration tests (20 tests)
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

### Run Specific Test File

```bash
pytest tests/test_data_processing.py
```

### Run Specific Test Class

```bash
pytest tests/test_data_processing.py::TestAssignProfile
```

### Run Specific Test

```bash
pytest tests/test_data_processing.py::TestAssignProfile::test_profile_a_balanced
```

### Run Tests by Marker

```bash
# Run only integration tests
pytest -m integration

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

---

## Test Markers

Tests are organized using pytest markers:

### Built-in Markers:

- `@pytest.mark.unit` - Unit tests (auto-applied)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

### Usage:

```python
@pytest.mark.integration
@pytest.mark.slow
def test_performance_analysis():
    # Test code here
    pass
```

---

## CI/CD Integration

Tests are automatically run in CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    name: Test & Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests with coverage
        run: |
          pytest --cov=. --cov-report=xml --cov-report=term
```

Tests must pass before merge!

---

## Writing New Tests

### 1. Use Existing Fixtures

```python
def test_my_feature(sample_csv_data):
    # Use fixture
    result = my_function(sample_csv_data)
    assert result is not None
```

### 2. Group Related Tests

```python
class TestMyFeature:
    def test_case_1(self):
        assert True

    def test_case_2(self):
        assert True
```

### 3. Use Descriptive Names

```python
def test_profile_assignment_with_balanced_tokens():
    # Clear what this tests
    pass
```

### 4. Add Docstrings

```python
def test_example():
    """Test that example function returns expected value."""
    pass
```

### 5. Use Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("input,expected", [
    (1000, "Profile A"),
    (512, "Profile B"),
])
def test_profiles(input, expected):
    assert assign_profile(input) == expected
```

---

### Common Commands:

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific marker
pytest -m integration

# Run and show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Show slowest tests
pytest --durations=10
```

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---
