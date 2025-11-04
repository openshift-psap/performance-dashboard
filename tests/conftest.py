"""Pytest configuration and shared fixtures for dashboard tests."""

import json

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_data():
    """Provide sample CSV data for testing."""
    data = {
        "run": ["run1", "run2", "run3", "run4", "run5"],
        "accelerator": ["H200", "H200", "MI300X", "MI300X", "TPU"],
        "model": [
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B",
            "mistralai/Mistral-7B",
            "meta-llama/Llama-3.3-70B",
        ],
        "version": [
            "vLLM-0.10.0",
            "vLLM-0.10.1",
            "vLLM-0.10.0",
            "vLLM-0.10.1",
            "sglang-0.5.2",
        ],
        "TP": [1, 2, 4, 8, 1],
        "prompt toks": [1000, 1000, 512, 512, 2048],
        "output toks": [1000, 1000, 2048, 2048, 128],
        "intended concurrency": [10, 10, 20, 20, 30],
        "measured_concurrency": [10, 10, 20, 20, 30],
        "output_tok/sec": [150.5, 155.3, 200.3, 195.8, 180.7],
        "total_tok/sec": [300.5, 310.6, 400.6, 391.6, 360.0],
        "ttft_p50": [0.03, 0.029, 0.025, 0.026, 0.035],
        "ttft_p95": [0.05, 0.048, 0.04, 0.042, 0.055],
        "ttft_p99": [0.06, 0.058, 0.05, 0.052, 0.065],
        "itl_p50": [0.01, 0.009, 0.008, 0.009, 0.012],
        "itl_p95": [0.02, 0.019, 0.015, 0.016, 0.022],
        "itl_p99": [0.025, 0.024, 0.020, 0.021, 0.028],
        "request_latency_median": [0.5, 0.48, 0.4, 0.42, 0.6],
        "request_latency_max": [1.2, 1.1, 0.9, 0.95, 1.4],
        "successful_requests": [95, 98, 97, 99, 96],
        "errored_requests": [5, 2, 3, 1, 4],
        "error_rate": [5.0, 2.0, 3.0, 1.0, 4.0],
        "runtime_args": [
            "tensor-parallel-size: 1",
            "tensor-parallel-size: 2",
            "tensor-parallel-size: 4",
            "tensor-parallel-size: 8",
            "tensor-parallel-size: 1",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def version_comparison_data():
    """Provide sample data for testing version comparison functionality."""
    data = {
        "run": [
            "run1",
            "run2",
            "run3",
            "run4",
            "run5",
            "run6",
            "run7",
            "run8",
            "run9",
            "run10",
        ],
        "accelerator": [
            "H200",
            "H200",
            "H200",
            "H200",
            "MI300X",
            "MI300X",
            "MI300X",
            "MI300X",
            "H200",
            "H200",
        ],
        "model": [
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "mistralai/Mistral-7B",
            "mistralai/Mistral-7B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
        ],
        "version": [
            "vLLM-0.10.0",
            "vLLM-0.10.0",
            "vLLM-0.10.1",
            "vLLM-0.10.1",
            "vLLM-0.10.0",
            "vLLM-0.10.1",
            "vLLM-0.10.0",
            "vLLM-0.10.1",
            "vLLM-0.10.0",
            "vLLM-0.10.1",
        ],
        "TP": [1, 1, 1, 1, 2, 2, 1, 1, 2, 2],
        "intended concurrency": [10, 25, 10, 25, 10, 10, 10, 10, 10, 10],
        "measured_concurrency": [10, 25, 10, 25, 10, 10, 10, 10, 10, 10],
        "output_tok/sec": [
            150.0,
            200.0,
            165.0,
            210.0,
            180.0,
            190.0,
            140.0,
            145.0,
            170.0,
            180.0,
        ],
        "ttft_p95": [50.0, 60.0, 48.0, 58.0, 55.0, 52.0, 60.0, 59.0, 53.0, 50.0],
        "ttft_p95_s": [
            0.05,
            0.06,
            0.048,
            0.058,
            0.055,
            0.052,
            0.06,
            0.059,
            0.053,
            0.05,
        ],
        "itl_p95": [20.0, 25.0, 19.0, 24.0, 22.0, 21.0, 25.0, 24.5, 21.0, 20.0],
        "prompt toks": [1000] * 10,
        "output toks": [1000] * 10,
        "successful_requests": [95] * 10,
        "errored_requests": [5] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_benchmark_json():
    """Provide sample benchmark JSON data for testing."""
    return {
        "benchmarks": [
            {
                "run_id": "test-uuid-123",
                "args": {
                    "profile": {
                        "streams": [10, 20, 30],
                        "measured_rates": [100, 200, 300],
                        "measured_concurrencies": [10, 20, 30],
                    }
                },
                "request_loader": {
                    "data": json.dumps({"prompt_tokens": 1000, "output_tokens": 1000})
                },
                "run_stats": {
                    "requests_made": {"successful": 95, "errored": 5},
                    "total_time": 60.0,
                },
                "metrics": {
                    "output_tokens_per_second": {
                        "successful": {
                            "mean": 150.5,
                            "std": 10.2,
                            "median": 148.0,
                            "min": 130.0,
                            "max": 170.0,
                        }
                    },
                    "inter_token_latency": {
                        "successful": {
                            "p50": 0.01,
                            "p95": 0.02,
                            "p99": 0.03,
                            "mean": 0.015,
                        }
                    },
                    "time_to_first_token": {
                        "successful": {
                            "p50": 0.03,
                            "p95": 0.05,
                            "p99": 0.07,
                            "mean": 0.04,
                        }
                    },
                    "request_latency": {
                        "successful": {
                            "p50": 0.5,
                            "p95": 0.8,
                            "p99": 1.0,
                            "median": 0.5,
                            "max": 1.2,
                        }
                    },
                    "total_tokens_per_second": {
                        "successful": {"mean": 300.0, "median": 295.0}
                    },
                },
            }
        ]
    }


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def temp_json_file(tmp_path, sample_benchmark_json):
    """Create a temporary JSON file with sample benchmark data."""
    json_file = tmp_path / "test_benchmark.json"
    with open(json_file, "w") as f:
        json.dump(sample_benchmark_json, f, indent=2)
    return json_file


@pytest.fixture
def accelerator_color_map():
    """Provide accelerator color mapping for tests."""
    return {
        "H200": "#1f77b4",
        "MI300X": "#ff7f0e",
        "TPU": "#2ca02c",
        "A100": "#d62728",
    }


@pytest.fixture
def valid_accelerators():
    """Provide list of valid accelerator types."""
    return ["H200", "MI300X", "TPU", "A100", "H100"]


@pytest.fixture
def valid_tp_sizes():
    """Provide list of valid TP sizes."""
    return [1, 2, 4, 8, 16, 32]


@pytest.fixture
def sample_runtime_args():
    """Provide sample runtime arguments."""
    return {
        "basic": "tensor-parallel-size: 8",
        "detailed": "tensor-parallel-size: 8; max-model-len: 8192; dtype: float16",
        "complex": "tensor-parallel-size: 8; max-model-len: 8192; dtype: float16; gpu-memory-utilization: 0.9",
    }


@pytest.fixture
def mock_session_state():
    """Provide mock session state for testing."""
    return {
        "theme_mode": "auto",
        "performance_plots_expanded": False,
        "cost_analysis_expanded": False,
        "filters_applied": False,
        "selected_accelerators": [],
        "selected_models": [],
        "selected_versions": [],
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "ui: marks tests that test UI components")


# Pytest collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests based on file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark slow tests based on name patterns
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
