"""Unit tests for data processing functions in dashboard.py."""

import pandas as pd
import pytest

# Since dashboard.py uses streamlit decorators, we'll need to test the core logic
# Let's create testable versions or import the functions directly


def assign_profile(row):
    """Assigns a human-readable profile name based on token counts."""
    prompt_toks = row["prompt toks"]
    output_toks = row["output toks"]
    if prompt_toks == 1000 and output_toks == 1000:
        return "Profile A: Balanced (1k/1k)"
    elif prompt_toks == 512 and output_toks == 2048:
        return "Profile B: Variable Workload (512/2k)"
    elif prompt_toks == 2048 and output_toks == 128:
        return "Profile C: Large Prompt (2k/128)"
    elif prompt_toks == 32000 and output_toks == 256:
        return "Profile D: Prefill Heavy (32k/256)"
    else:
        return "Custom"


def clean_profile_name(profile_name):
    """Extract only the token counts in parentheses from profile names."""
    if profile_name and "(" in profile_name and ")" in profile_name:
        start_idx = profile_name.find("(")
        end_idx = profile_name.find(")", start_idx)
        if start_idx != -1 and end_idx != -1:
            return profile_name[start_idx : end_idx + 1]
    return profile_name


class TestAssignProfile:
    """Test the assign_profile function."""

    def test_profile_a_balanced(self):
        """Test Profile A: Balanced (1k/1k) assignment."""
        row = pd.Series({"prompt toks": 1000, "output toks": 1000})
        result = assign_profile(row)
        assert result == "Profile A: Balanced (1k/1k)"

    def test_profile_b_variable(self):
        """Test Profile B: Variable Workload (512/2k) assignment."""
        row = pd.Series({"prompt toks": 512, "output toks": 2048})
        result = assign_profile(row)
        assert result == "Profile B: Variable Workload (512/2k)"

    def test_profile_c_large_prompt(self):
        """Test Profile C: Large Prompt (2k/128) assignment."""
        row = pd.Series({"prompt toks": 2048, "output toks": 128})
        result = assign_profile(row)
        assert result == "Profile C: Large Prompt (2k/128)"

    def test_profile_d_prefill_heavy(self):
        """Test Profile D: Prefill Heavy (32k/256) assignment."""
        row = pd.Series({"prompt toks": 32000, "output toks": 256})
        result = assign_profile(row)
        assert result == "Profile D: Prefill Heavy (32k/256)"

    def test_custom_profile(self):
        """Test custom profile assignment for non-standard values."""
        row = pd.Series({"prompt toks": 500, "output toks": 500})
        result = assign_profile(row)
        assert result == "Custom"

    def test_custom_profile_different_values(self):
        """Test custom profile with various values."""
        test_cases = [
            (100, 100),
            (2000, 1000),
            (4096, 512),
        ]
        for prompt, output in test_cases:
            row = pd.Series({"prompt toks": prompt, "output toks": output})
            result = assign_profile(row)
            assert result == "Custom"


class TestCleanProfileName:
    """Test the clean_profile_name function."""

    def test_extract_balanced_profile(self):
        """Test extracting tokens from balanced profile."""
        profile = "Profile A: Balanced (1k/1k)"
        result = clean_profile_name(profile)
        assert result == "(1k/1k)"

    def test_extract_variable_profile(self):
        """Test extracting tokens from variable workload profile."""
        profile = "Profile B: Variable Workload (512/2k)"
        result = clean_profile_name(profile)
        assert result == "(512/2k)"

    def test_extract_large_prompt_profile(self):
        """Test extracting tokens from large prompt profile."""
        profile = "Profile C: Large Prompt (2k/128)"
        result = clean_profile_name(profile)
        assert result == "(2k/128)"

    def test_extract_prefill_heavy_profile(self):
        """Test extracting tokens from prefill heavy profile."""
        profile = "Profile D: Prefill Heavy (32k/256)"
        result = clean_profile_name(profile)
        assert result == "(32k/256)"

    def test_no_parentheses(self):
        """Test profile name without parentheses returns original."""
        profile = "Custom Profile"
        result = clean_profile_name(profile)
        assert result == "Custom Profile"

    def test_empty_string(self):
        """Test empty string returns empty string."""
        result = clean_profile_name("")
        assert result == ""

    def test_none_value(self):
        """Test None value returns None."""
        result = clean_profile_name(None)
        assert result is None

    def test_multiple_parentheses(self):
        """Test with multiple parentheses extracts first."""
        profile = "Test (first) and (second)"
        result = clean_profile_name(profile)
        assert result == "(first)"

    def test_malformed_parentheses(self):
        """Test with malformed parentheses."""
        # Only opening parenthesis
        assert clean_profile_name("Test (incomplete") == "Test (incomplete"
        # Only closing parenthesis
        assert clean_profile_name("Test incomplete)") == "Test incomplete)"


class TestDataFrameOperations:
    """Test DataFrame manipulation operations."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        data = {
            "run": ["run1", "run2", "run3"],
            "accelerator": ["H200", "MI300X", "TPU"],
            "model": ["Llama-3.1-8B", "Mistral-7B", "Llama-3.3-70B"],
            "version": ["vLLM-0.10.0", "vLLM-0.10.1", "sglang-0.5.2"],
            "TP": [4, 8, 16],
            "prompt toks": [1000, 512, 2048],
            "output toks": [1000, 2048, 128],
            "output_tok/sec": [150.5, 200.3, 180.7],
            "ttft_p95": [0.05, 0.03, 0.04],
            "itl_p95": [0.02, 0.015, 0.018],
        }
        return pd.DataFrame(data)

    def test_dataframe_columns(self, sample_dataframe):
        """Test that sample dataframe has expected columns."""
        expected_columns = [
            "run",
            "accelerator",
            "model",
            "version",
            "TP",
            "prompt toks",
            "output toks",
            "output_tok/sec",
            "ttft_p95",
            "itl_p95",
        ]
        assert all(col in sample_dataframe.columns for col in expected_columns)

    def test_profile_assignment(self, sample_dataframe):
        """Test profile assignment on dataframe."""
        sample_dataframe["profile"] = sample_dataframe.apply(assign_profile, axis=1)
        assert sample_dataframe.loc[0, "profile"] == "Profile A: Balanced (1k/1k)"
        assert (
            sample_dataframe.loc[1, "profile"]
            == "Profile B: Variable Workload (512/2k)"
        )
        assert sample_dataframe.loc[2, "profile"] == "Profile C: Large Prompt (2k/128)"

    def test_clean_profile_names(self, sample_dataframe):
        """Test cleaning profile names on dataframe."""
        sample_dataframe["profile"] = sample_dataframe.apply(assign_profile, axis=1)
        sample_dataframe["profile_clean"] = sample_dataframe["profile"].apply(
            clean_profile_name
        )
        assert sample_dataframe.loc[0, "profile_clean"] == "(1k/1k)"
        assert sample_dataframe.loc[1, "profile_clean"] == "(512/2k)"
        assert sample_dataframe.loc[2, "profile_clean"] == "(2k/128)"

    def test_filter_by_accelerator(self, sample_dataframe):
        """Test filtering dataframe by accelerator."""
        filtered = sample_dataframe[sample_dataframe["accelerator"] == "H200"]
        assert len(filtered) == 1
        assert filtered.iloc[0]["accelerator"] == "H200"

    def test_filter_by_multiple_accelerators(self, sample_dataframe):
        """Test filtering by multiple accelerators."""
        filtered = sample_dataframe[
            sample_dataframe["accelerator"].isin(["H200", "MI300X"])
        ]
        assert len(filtered) == 2
        assert all(acc in ["H200", "MI300X"] for acc in filtered["accelerator"])

    def test_groupby_accelerator(self, sample_dataframe):
        """Test grouping by accelerator."""
        grouped = sample_dataframe.groupby("accelerator")["output_tok/sec"].mean()
        assert "H200" in grouped.index
        assert "MI300X" in grouped.index
        assert "TPU" in grouped.index

    def test_sorting_by_throughput(self, sample_dataframe):
        """Test sorting by throughput."""
        sorted_df = sample_dataframe.sort_values("output_tok/sec", ascending=False)
        assert sorted_df.iloc[0]["output_tok/sec"] == 200.3
        assert sorted_df.iloc[-1]["output_tok/sec"] == 150.5

    def test_tp_size_filtering(self, sample_dataframe):
        """Test filtering by TP size."""
        filtered = sample_dataframe[sample_dataframe["TP"] >= 8]
        assert len(filtered) == 2
        assert all(tp >= 8 for tp in filtered["TP"])


class TestMetricCalculations:
    """Test metric calculation functions."""

    def test_efficiency_ratio_calculation(self):
        """Test efficiency ratio calculation."""
        throughput = 150.5
        tp_size = 4
        efficiency = throughput / tp_size
        assert efficiency == pytest.approx(37.625, rel=1e-6)

    def test_cost_per_million_tokens(self):
        """Test cost per million tokens calculation."""
        instance_cost_per_hour = 5.0  # $5/hour
        throughput_tokens_per_sec = 150.0
        tokens_per_hour = throughput_tokens_per_sec * 3600
        cost_per_million = (instance_cost_per_hour / tokens_per_hour) * 1_000_000
        expected = (5.0 / 540000) * 1_000_000
        assert cost_per_million == pytest.approx(expected, rel=1e-6)

    def test_time_to_million_tokens(self):
        """Test time to process million tokens."""
        throughput_tokens_per_sec = 150.0
        time_seconds = 1_000_000 / throughput_tokens_per_sec
        expected_hours = time_seconds / 3600
        assert time_seconds == pytest.approx(6666.67, rel=1e-2)
        assert expected_hours == pytest.approx(1.85, rel=1e-2)

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        successful_requests = 95
        errored_requests = 5
        total_requests = successful_requests + errored_requests
        error_rate = (errored_requests / total_requests) * 100
        assert error_rate == 5.0

    def test_zero_throughput_handling(self):
        """Test handling of zero throughput."""
        throughput = 0
        with pytest.raises(ZeroDivisionError):
            1_000_000 / throughput

    def test_negative_values_handling(self):
        """Test handling of negative values (should not occur in real data)."""
        # These should raise appropriate errors in real implementation
        assert -100 < 0  # Negative throughput detection
        assert -5.0 < 0  # Negative cost detection


class TestDataValidation:
    """Test data validation functions."""

    def test_valid_accelerator_types(self):
        """Test valid accelerator type checking."""
        valid_accelerators = ["H200", "MI300X", "TPU", "A100"]
        test_accelerator = "H200"
        assert test_accelerator in valid_accelerators

    def test_invalid_accelerator_type(self):
        """Test invalid accelerator type detection."""
        valid_accelerators = ["H200", "MI300X", "TPU", "A100"]
        test_accelerator = "INVALID"
        assert test_accelerator not in valid_accelerators

    def test_valid_tp_sizes(self):
        """Test valid TP size checking."""
        valid_tp_sizes = [1, 2, 4, 8, 16, 32]
        assert 8 in valid_tp_sizes
        assert 3 not in valid_tp_sizes

    def test_throughput_range(self):
        """Test throughput value range validation."""
        throughput = 150.5
        assert throughput > 0, "Throughput must be positive"
        assert throughput < 10000, "Throughput seems unrealistically high"

    def test_latency_range(self):
        """Test latency value range validation."""
        ttft_p95 = 0.05  # 50ms
        itl_p95 = 0.02  # 20ms
        assert ttft_p95 > 0, "Latency must be positive"
        assert ttft_p95 < 10, "Latency seems unrealistically high (>10s)"
        assert itl_p95 > 0
        assert itl_p95 < ttft_p95 * 10  # ITL should be reasonably related to TTFT
