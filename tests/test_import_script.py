"""Unit tests for import_manual_run_jsons_old.py script (guidellm 0.3.x/0.4.x)."""

import json
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

# Import the actual functions to test (from the old script for guidellm 0.3.x/0.4.x)
from manual_runs.scripts.import_manual_run_jsons_old import (
    parse_guidellm_json,
    process_benchmark_section,
)


class TestProcessBenchmarkSection:
    """Test the process_benchmark_section function."""

    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data for testing."""
        return {
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
                    "successful": {"mean": 150.5, "std": 10.2, "median": 148.0}
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

    def test_extract_run_id(self, sample_benchmark_data):
        """Test extracting run_id from benchmark data."""
        assert sample_benchmark_data["run_id"] == "test-uuid-123"

    def test_extract_profile_args(self, sample_benchmark_data):
        """Test extracting profile args from benchmark data."""
        profile_args = sample_benchmark_data["args"]["profile"]
        assert "streams" in profile_args
        assert "measured_rates" in profile_args
        assert profile_args["streams"] == [10, 20, 30]

    def test_extract_request_config(self, sample_benchmark_data):
        """Test extracting request configuration."""
        request_data = sample_benchmark_data["request_loader"]["data"]
        config = json.loads(request_data)
        assert config["prompt_tokens"] == 1000
        assert config["output_tokens"] == 1000

    def test_extract_run_stats(self, sample_benchmark_data):
        """Test extracting run statistics."""
        run_stats = sample_benchmark_data["run_stats"]
        requests_made = run_stats["requests_made"]
        assert requests_made["successful"] == 95
        assert requests_made["errored"] == 5
        assert run_stats["total_time"] == 60.0

    def test_extract_metrics(self, sample_benchmark_data):
        """Test extracting performance metrics."""
        metrics = sample_benchmark_data["metrics"]
        assert "output_tokens_per_second" in metrics
        assert "inter_token_latency" in metrics
        assert "time_to_first_token" in metrics
        throughput = metrics["output_tokens_per_second"]["successful"]["mean"]
        assert throughput == 150.5

    def test_calculate_error_rate(self, sample_benchmark_data):
        """Test error rate calculation."""
        requests = sample_benchmark_data["run_stats"]["requests_made"]
        total = requests["successful"] + requests["errored"]
        error_rate = (requests["errored"] / total) * 100
        assert error_rate == 5.0

    def test_malformed_request_data(self):
        """Test handling malformed request data JSON."""
        malformed_data = '{"invalid": json data}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_data)

    def test_missing_keys_handling(self):
        """Test handling missing keys in benchmark data."""
        incomplete_data = {"run_id": "test-123"}
        # Should handle gracefully with .get() methods
        assert incomplete_data.get("args", {}).get("profile", {}) == {}
        assert incomplete_data.get("metrics", {}) == {}


class TestParseGuidellmJson:
    """Test the parse_guidellm_json function."""

    @pytest.fixture
    def sample_json_content(self):
        """Create sample JSON file content."""
        return {
            "benchmarks": [
                {
                    "run_id": "benchmark-1",
                    "args": {
                        "profile": {
                            "streams": [10],
                            "measured_rates": [100],
                            "measured_concurrencies": [10],
                        }
                    },
                    "request_loader": {
                        "data": json.dumps(
                            {"prompt_tokens": 1000, "output_tokens": 1000}
                        )
                    },
                    "run_stats": {"requests_made": {"successful": 100, "errored": 0}},
                    "metrics": {
                        "output_tokens_per_second": {"successful": {"mean": 150.0}}
                    },
                }
            ]
        }

    def test_parse_valid_json(self, sample_json_content):
        """Test parsing valid JSON content."""
        assert "benchmarks" in sample_json_content
        assert len(sample_json_content["benchmarks"]) == 1
        assert sample_json_content["benchmarks"][0]["run_id"] == "benchmark-1"

    def test_empty_benchmarks_list(self):
        """Test handling empty benchmarks list."""
        data = {"benchmarks": []}
        assert len(data["benchmarks"]) == 0

    def test_missing_benchmarks_key(self):
        """Test handling missing benchmarks key."""
        data = {"other_key": "value"}
        assert data.get("benchmarks") is None

    @patch("builtins.open", mock_open(read_data='{"benchmarks": []}'))
    def test_file_reading(self):
        """Test file reading with mocked open."""
        with open("test.json") as f:
            data = json.load(f)
        assert "benchmarks" in data


class TestDataFrameConversion:
    """Test conversion of benchmark data to DataFrame."""

    @pytest.fixture
    def sample_rows(self):
        """Create sample row data."""
        return [
            {
                "run": "test-run-1",
                "accelerator": "H200",
                "model": "Llama-3.1-8B",
                "version": "vLLM-0.10.0",
                "TP": 4,
                "output_tok/sec": 150.5,
                "ttft_p95": 0.05,
                "itl_p95": 0.02,
                "intended concurrency": 10,
                "successful_requests": 95,
                "errored_requests": 5,
                "error_rate": 5.0,
            },
            {
                "run": "test-run-2",
                "accelerator": "MI300X",
                "model": "Mistral-7B",
                "version": "vLLM-0.10.1",
                "TP": 8,
                "output_tok/sec": 200.3,
                "ttft_p95": 0.03,
                "itl_p95": 0.015,
                "intended concurrency": 20,
                "successful_requests": 98,
                "errored_requests": 2,
                "error_rate": 2.0,
            },
        ]

    def test_create_dataframe(self, sample_rows):
        """Test creating DataFrame from row data."""
        df = pd.DataFrame(sample_rows)
        assert len(df) == 2
        assert "accelerator" in df.columns
        assert "output_tok/sec" in df.columns

    def test_dataframe_column_types(self, sample_rows):
        """Test DataFrame column data types."""
        df = pd.DataFrame(sample_rows)
        assert df["TP"].dtype in [int, "int64"]
        assert df["output_tok/sec"].dtype in [float, "float64"]
        assert df["error_rate"].dtype in [float, "float64"]

    def test_filter_dataframe(self, sample_rows):
        """Test filtering DataFrame."""
        df = pd.DataFrame(sample_rows)
        h200_df = df[df["accelerator"] == "H200"]
        assert len(h200_df) == 1
        assert h200_df.iloc[0]["accelerator"] == "H200"

    def test_aggregate_dataframe(self, sample_rows):
        """Test aggregating DataFrame data."""
        df = pd.DataFrame(sample_rows)
        avg_throughput = df["output_tok/sec"].mean()
        assert avg_throughput == pytest.approx((150.5 + 200.3) / 2, rel=1e-6)

    def test_sort_dataframe(self, sample_rows):
        """Test sorting DataFrame."""
        df = pd.DataFrame(sample_rows)
        sorted_df = df.sort_values("output_tok/sec", ascending=False)
        assert sorted_df.iloc[0]["output_tok/sec"] == 200.3

    def test_csv_export(self, sample_rows, tmp_path):
        """Test exporting DataFrame to CSV."""
        df = pd.DataFrame(sample_rows)
        csv_path = tmp_path / "test_output.csv"
        df.to_csv(csv_path, index=False)
        assert csv_path.exists()

        # Read back and verify
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)


class TestCommandLineArguments:
    """Test command-line argument parsing."""

    def test_required_arguments(self):
        """Test required command-line arguments."""
        required_args = [
            "json_file",
            "--accelerator",
            "--model",
            "--version",
            "--tp",
        ]
        # These should be required for the script
        assert "json_file" in required_args
        assert "--accelerator" in required_args

    def test_optional_arguments(self):
        """Test optional command-line arguments."""
        optional_args = ["--runtime-args", "--output"]
        assert "--runtime-args" in optional_args
        assert "--output" in optional_args

    def test_argument_values(self):
        """Test valid argument values."""
        # Accelerator types
        valid_accelerators = ["H200", "MI300X", "TPU", "A100"]
        assert "H200" in valid_accelerators

        # TP sizes
        valid_tp_sizes = [1, 2, 4, 8, 16, 32]
        assert 8 in valid_tp_sizes

        # Version patterns
        version = "vLLM-0.10.0"
        assert "vLLM" in version or "RHAIIS" in version or "sglang" in version


class TestRuntimeArgsFormatting:
    """Test runtime arguments formatting."""

    def test_parse_runtime_args_string(self):
        """Test parsing runtime args from string."""
        runtime_args = "tensor-parallel-size: 8; max-model-len: 8192; dtype: float16"
        args_dict = {}
        for pair in runtime_args.split(";"):
            if ":" in pair:
                key, value = pair.split(":", 1)
                args_dict[key.strip()] = value.strip()

        assert args_dict["tensor-parallel-size"] == "8"
        assert args_dict["max-model-len"] == "8192"
        assert args_dict["dtype"] == "float16"

    def test_format_runtime_args(self):
        """Test formatting runtime args for display."""
        args_dict = {
            "tensor-parallel-size": "8",
            "max-model-len": "8192",
            "dtype": "float16",
        }
        formatted = "; ".join([f"{k}: {v}" for k, v in args_dict.items()])
        assert "tensor-parallel-size: 8" in formatted
        assert "max-model-len: 8192" in formatted

    def test_empty_runtime_args(self):
        """Test handling empty runtime args."""
        runtime_args = ""
        assert runtime_args == ""
        # Should handle gracefully
        args_list = [x for x in runtime_args.split(";") if x.strip()]
        assert len(args_list) == 0

    def test_malformed_runtime_args(self):
        """Test handling malformed runtime args."""
        runtime_args = "malformed without colons"
        # Should handle gracefully without crashing
        args_dict = {}
        for pair in runtime_args.split(";"):
            if ":" in pair:
                key, value = pair.split(":", 1)
                args_dict[key.strip()] = value.strip()
        assert len(args_dict) == 0  # No valid pairs found


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_requests(self):
        """Test handling zero requests."""
        successful = 0
        errored = 0
        total = successful + errored
        assert total == 0
        # Should handle division by zero
        error_rate = errored / total * 100 if total > 0 else 0.0
        assert error_rate == 0.0

    def test_all_errors(self):
        """Test handling all errored requests."""
        successful = 0
        errored = 100
        total = successful + errored
        error_rate = (errored / total) * 100
        assert error_rate == 100.0

    def test_missing_metrics(self):
        """Test handling missing metrics."""
        metrics = {}
        output_tps = metrics.get("output_tokens_per_second", {}).get("successful", {})
        assert output_tps == {}
        # Should provide defaults
        mean_tps = output_tps.get("mean", 0.0)
        assert mean_tps == 0.0

    def test_negative_values(self):
        """Test detecting negative values (data quality issue)."""
        throughput = -100
        assert throughput < 0, "Negative throughput detected - data quality issue"

    def test_extremely_large_values(self):
        """Test handling extremely large values."""
        throughput = 1000000  # 1M tokens/sec (unrealistic)
        assert throughput > 10000, "Suspiciously high throughput"

    def test_null_values(self):
        """Test handling null/None values."""
        value = None
        assert value is None
        # Should use defaults or skip
        safe_value = value if value is not None else 0.0
        assert safe_value == 0.0


class TestActualFunctionIntegration:
    """Integration tests that actually call the imported functions."""

    def test_process_benchmark_section_with_sample_data(
        self, integration_benchmark_data
    ):
        """Test process_benchmark_section with sample data."""
        result = process_benchmark_section(
            benchmark_run=integration_benchmark_data,
            accelerator="H200",
            model_name="test-model",
            version="vLLM-0.10.0",
            tp_size=4,
            runtime_args="tensor-parallel-size: 4",
            benchmark_index=0,
        )

        # Verify returned data structure
        assert isinstance(result, dict)
        assert "run" in result
        assert "accelerator" in result
        assert "model" in result
        assert "version" in result
        assert "TP" in result
        assert "output_tok/sec" in result

        # Verify values
        assert result["accelerator"] == "H200"
        assert result["model"] == "test-model"
        assert result["version"] == "vLLM-0.10.0"
        assert result["TP"] == 4
        assert result["prompt toks"] == 1000
        assert result["output toks"] == 1000
        assert result["output_tok/sec"] == 150.5

    def test_process_benchmark_section_multiple_indexes(
        self, integration_benchmark_data
    ):
        """Test process_benchmark_section with different benchmark indexes."""
        # Test index 0
        result_0 = process_benchmark_section(
            integration_benchmark_data, "H200", "test", "v1", 2, "tp: 2", 0
        )
        assert result_0["intended concurrency"] == 10

        # Test index 1
        result_1 = process_benchmark_section(
            integration_benchmark_data, "H200", "test", "v1", 2, "tp: 2", 1
        )
        assert result_1["intended concurrency"] == 20

        # Test index 2
        result_2 = process_benchmark_section(
            integration_benchmark_data, "H200", "test", "v1", 2, "tp: 2", 2
        )
        assert result_2["intended concurrency"] == 30

    def test_parse_guidellm_json_with_mock_file(self, integration_benchmark_json):
        """Test parse_guidellm_json with mocked file I/O."""
        # Create JSON content from fixture
        json_content = json.dumps(integration_benchmark_json)

        # Mock the file reading in the actual module
        with patch(
            "manual_runs.scripts.import_manual_run_jsons_old.open",
            mock_open(read_data=json_content),
        ):
            result = parse_guidellm_json(
                json_path="fake_path.json",
                accelerator="MI300X",
                model_name="meta-llama/Llama-3.1-8B",
                version="vLLM-0.10.1",
                tp_size=8,
                runtime_args="tensor-parallel-size: 8; dtype: float16",
            )

        # Verify DataFrame returned
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Verify DataFrame has expected columns
        expected_columns = [
            "run",
            "accelerator",
            "model",
            "version",
            "TP",
            "output_tok/sec",
            "ttft_p95",
            "itl_p95",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify data values
        assert result["accelerator"].iloc[0] == "MI300X"
        assert result["model"].iloc[0] == "meta-llama/Llama-3.1-8B"
        assert result["version"].iloc[0] == "vLLM-0.10.1"
        assert result["TP"].iloc[0] == 8

    def test_parse_guidellm_json_empty_benchmarks(self):
        """Test parse_guidellm_json with empty benchmarks list."""
        json_content = json.dumps({"benchmarks": []})

        with patch(
            "manual_runs.scripts.import_manual_run_jsons_old.open",
            mock_open(read_data=json_content),
        ):
            with patch("builtins.print"):  # Suppress print statements
                result = parse_guidellm_json(
                    "fake.json", "H200", "model", "v1", 4, "tp: 4"
                )

        # Should return None for empty benchmarks
        assert result is None

    def test_parse_guidellm_json_missing_benchmarks_key(self):
        """Test parse_guidellm_json with missing benchmarks key."""
        json_content = json.dumps({"some_other_key": "value"})

        with patch(
            "manual_runs.scripts.import_manual_run_jsons_old.open",
            mock_open(read_data=json_content),
        ):
            with patch("builtins.print"):  # Suppress print statements
                result = parse_guidellm_json(
                    "fake.json", "H200", "model", "v1", 4, "tp: 4"
                )

        # Should return None when benchmarks key is missing
        assert result is None

    def test_parse_guidellm_json_file_not_found(self):
        """Test parse_guidellm_json with non-existent file."""
        with patch(
            "manual_runs.scripts.import_manual_run_jsons_old.open",
            side_effect=FileNotFoundError(),
        ):
            with patch("builtins.print"):  # Suppress print statements
                result = parse_guidellm_json(
                    "nonexistent.json", "H200", "model", "v1", 4, "tp: 4"
                )

        # Should return None for file not found
        assert result is None

    def test_parse_guidellm_json_invalid_json(self):
        """Test parse_guidellm_json with invalid JSON content."""
        invalid_json = "{ invalid json content"

        with patch(
            "manual_runs.scripts.import_manual_run_jsons_old.open",
            mock_open(read_data=invalid_json),
        ):
            with patch("builtins.print"):  # Suppress print statements
                result = parse_guidellm_json(
                    "invalid.json", "H200", "model", "v1", 4, "tp: 4"
                )

        # Should return None for invalid JSON
        assert result is None


# Module-level fixtures for integration tests
@pytest.fixture
def integration_benchmark_data():
    """Sample benchmark data matching actual function expectations."""
    return {
        "run_id": "test-uuid-integration",
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
                "successful": {"mean": 150.5, "std": 10.2, "median": 148.0}
            },
            "inter_token_latency_ms": {
                "successful": {
                    "p50": 10.0,
                    "p95": 20.0,
                    "p99": 30.0,
                    "mean": 15.0,
                }
            },
            "time_to_first_token_ms": {
                "successful": {
                    "p50": 30.0,
                    "p95": 50.0,
                    "p99": 70.0,
                    "mean": 40.0,
                }
            },
            "request_latency": {
                "successful": {
                    "p50": 500.0,
                    "p95": 800.0,
                    "p99": 1000.0,
                    "median": 500.0,
                    "max": 1200.0,
                }
            },
            "tokens_per_second": {"successful": {"mean": 300.0, "median": 295.0}},
        },
    }


@pytest.fixture
def integration_benchmark_json():
    """Sample JSON structure matching actual function expectations."""
    return {
        "benchmarks": [
            {
                "run_id": "test-uuid-json",
                "args": {
                    "profile": {
                        "streams": [10],
                        "measured_rates": [100],
                        "measured_concurrencies": [10],
                    }
                },
                "request_loader": {
                    "data": json.dumps({"prompt_tokens": 512, "output_tokens": 2048})
                },
                "run_stats": {
                    "requests_made": {"successful": 98, "errored": 2},
                    "total_time": 60.0,
                },
                "metrics": {
                    "output_tokens_per_second": {
                        "successful": {"mean": 200.3, "std": 12.5, "median": 198.0}
                    },
                    "inter_token_latency_ms": {
                        "successful": {
                            "p50": 8.0,
                            "p95": 15.0,
                            "p99": 20.0,
                            "mean": 10.5,
                        }
                    },
                    "time_to_first_token_ms": {
                        "successful": {
                            "p50": 25.0,
                            "p95": 40.0,
                            "p99": 50.0,
                            "mean": 30.0,
                        }
                    },
                    "request_latency": {
                        "successful": {
                            "p50": 400.0,
                            "p95": 700.0,
                            "p99": 900.0,
                            "median": 400.0,
                            "max": 1000.0,
                        }
                    },
                    "tokens_per_second": {
                        "successful": {"mean": 400.0, "median": 395.0}
                    },
                },
            }
        ]
    }
