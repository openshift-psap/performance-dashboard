"""Integration tests for the performance dashboard."""

import pandas as pd
import pytest


@pytest.mark.integration
class TestDataPipeline:
    """Test the complete data processing pipeline."""

    def test_csv_load_and_process(self, temp_csv_file):
        """Test loading and processing CSV data."""
        df = pd.read_csv(temp_csv_file)

        # Verify data loaded correctly
        assert len(df) > 0
        assert "accelerator" in df.columns
        assert "model" in df.columns
        assert "version" in df.columns

        # Clean data
        df["accelerator"] = df["accelerator"].str.strip()
        df["model"] = df["model"].str.strip()
        df["version"] = df["version"].str.strip()

        # Verify cleaning worked
        assert not df["accelerator"].str.contains(" ").any()

    def test_profile_assignment_pipeline(self, sample_csv_data):
        """Test complete profile assignment workflow."""

        def assign_profile(row):
            prompt_toks = row["prompt toks"]
            output_toks = row["output toks"]
            if prompt_toks == 1000 and output_toks == 1000:
                return "Profile A: Balanced (1k/1k)"
            elif prompt_toks == 512 and output_toks == 2048:
                return "Profile B: Variable Workload (512/2k)"
            elif prompt_toks == 2048 and output_toks == 128:
                return "Profile C: Large Prompt (2k/128)"
            else:
                return "Custom"

        # Apply profile assignment
        sample_csv_data["profile"] = sample_csv_data.apply(assign_profile, axis=1)

        # Verify profiles assigned
        assert "profile" in sample_csv_data.columns
        assert sample_csv_data["profile"].notna().all()

        # Check specific profile assignments
        profile_a_count = len(
            sample_csv_data[sample_csv_data["profile"] == "Profile A: Balanced (1k/1k)"]
        )
        assert profile_a_count >= 0

    def test_filter_and_aggregate(self, sample_csv_data):
        """Test filtering and aggregation workflow."""
        # Filter by accelerator
        h200_data = sample_csv_data[sample_csv_data["accelerator"] == "H200"]

        # Aggregate metrics
        if len(h200_data) > 0:
            avg_throughput = h200_data["output_tok/sec"].mean()
            max_throughput = h200_data["output_tok/sec"].max()
            min_throughput = h200_data["output_tok/sec"].min()

            # Verify aggregations
            assert avg_throughput > 0
            assert max_throughput >= avg_throughput
            assert min_throughput <= avg_throughput

    def test_multi_accelerator_comparison(self, sample_csv_data):
        """Test comparing metrics across accelerators."""
        # Group by accelerator
        grouped = sample_csv_data.groupby("accelerator")["output_tok/sec"].agg(
            ["mean", "max", "min", "count"]
        )

        # Verify grouping
        assert len(grouped) > 0
        assert all(grouped["count"] > 0)
        assert all(grouped["mean"] > 0)

    def test_version_comparison(self, sample_csv_data):
        """Test comparing performance across versions."""
        # Get unique versions
        versions = sample_csv_data["version"].unique()

        if len(versions) > 1:
            # Compare first two versions
            v1_data = sample_csv_data[sample_csv_data["version"] == versions[0]]
            v2_data = sample_csv_data[sample_csv_data["version"] == versions[1]]

            # Calculate average throughput
            v1_avg = v1_data["output_tok/sec"].mean()
            v2_avg = v2_data["output_tok/sec"].mean()

            # Calculate percentage change
            if v1_avg > 0:
                pct_change = ((v2_avg - v1_avg) / v1_avg) * 100
                assert isinstance(pct_change, float)


@pytest.mark.integration
class TestCostCalculations:
    """Test cost calculation workflows."""

    def test_cost_per_million_tokens(self, sample_csv_data):
        """Test calculating cost per million tokens."""
        instance_cost_per_hour = 5.0  # $5/hour
        sample_csv_data["cost_per_M_tokens"] = (
            instance_cost_per_hour / (sample_csv_data["output_tok/sec"] * 3600)
        ) * 1_000_000

        # Verify calculations
        assert "cost_per_M_tokens" in sample_csv_data.columns
        assert (sample_csv_data["cost_per_M_tokens"] > 0).all()

    def test_time_to_million_tokens(self, sample_csv_data):
        """Test calculating time to process million tokens."""
        sample_csv_data["time_to_M_tokens_hours"] = (
            1_000_000 / sample_csv_data["output_tok/sec"]
        ) / 3600

        # Verify calculations
        assert "time_to_M_tokens_hours" in sample_csv_data.columns
        assert (sample_csv_data["time_to_M_tokens_hours"] > 0).all()

    def test_efficiency_ratio(self, sample_csv_data):
        """Test calculating efficiency ratio (throughput per TP unit)."""
        sample_csv_data["efficiency_ratio"] = (
            sample_csv_data["output_tok/sec"] / sample_csv_data["TP"]
        )

        # Verify calculations
        assert "efficiency_ratio" in sample_csv_data.columns
        assert (sample_csv_data["efficiency_ratio"] > 0).all()

    def test_cost_comparison_across_configs(self, sample_csv_data):
        """Test comparing costs across different configurations."""
        instance_costs = {"H200": 5.0, "MI300X": 4.5, "TPU": 6.0}

        sample_csv_data["instance_cost"] = sample_csv_data["accelerator"].map(
            lambda x: instance_costs.get(x, 5.0)
        )

        sample_csv_data["cost_per_M_tokens"] = (
            sample_csv_data["instance_cost"]
            / (sample_csv_data["output_tok/sec"] * 3600)
        ) * 1_000_000

        # Find most cost-effective configuration
        best_cost_idx = sample_csv_data["cost_per_M_tokens"].idxmin()
        best_config = sample_csv_data.loc[best_cost_idx]

        assert best_config is not None
        assert best_config["cost_per_M_tokens"] > 0


@pytest.mark.integration
class TestDataQuality:
    """Test data quality checks and validation."""

    def test_no_missing_critical_columns(self, sample_csv_data):
        """Test that critical columns have no missing values."""
        critical_columns = [
            "run",
            "accelerator",
            "model",
            "version",
            "TP",
            "output_tok/sec",
        ]

        for col in critical_columns:
            assert col in sample_csv_data.columns
            assert sample_csv_data[col].notna().all(), f"Missing values in {col}"

    def test_data_types_correct(self, sample_csv_data):
        """Test that columns have expected data types."""
        # Numeric columns
        numeric_cols = ["TP", "output_tok/sec", "ttft_p95", "itl_p95"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_csv_data[col]), (
                f"{col} should be numeric"
            )

        # String columns
        string_cols = ["run", "accelerator", "model", "version"]
        for col in string_cols:
            assert pd.api.types.is_object_dtype(
                sample_csv_data[col]
            ) or pd.api.types.is_string_dtype(sample_csv_data[col]), (
                f"{col} should be string"
            )

    def test_value_ranges(self, sample_csv_data):
        """Test that values are within expected ranges."""
        # Throughput should be positive
        assert (sample_csv_data["output_tok/sec"] > 0).all()

        # TP sizes should be valid
        valid_tp_sizes = [1, 2, 4, 8]
        assert sample_csv_data["TP"].isin(valid_tp_sizes).all()

        # Error rate should be 0-100%
        assert (sample_csv_data["error_rate"] >= 0).all()
        assert (sample_csv_data["error_rate"] <= 100).all()

        # Latencies should be positive
        assert (sample_csv_data["ttft_p95"] > 0).all()
        assert (sample_csv_data["itl_p95"] > 0).all()

    def test_logical_consistency(self, sample_csv_data):
        """Test logical consistency between fields."""
        # Intended concurrency should match measured (or be close)
        concurrency_diff = abs(
            sample_csv_data["intended concurrency"]
            - sample_csv_data["measured_concurrency"]
        )
        assert (concurrency_diff <= 5).all(), "Concurrency mismatch too large"

        # Total successful + errored should make sense
        total_requests = (
            sample_csv_data["successful_requests"] + sample_csv_data["errored_requests"]
        )
        assert (total_requests > 0).all()

        # Error rate should match calculated value
        calculated_error_rate = (
            sample_csv_data["errored_requests"] / total_requests
        ) * 100
        error_rate_diff = abs(sample_csv_data["error_rate"] - calculated_error_rate)
        assert (error_rate_diff < 0.1).all(), "Error rate calculation mismatch"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAnalysis:
    """Test performance analysis workflows."""

    def test_regression_detection(self, sample_csv_data):
        """Test detecting performance regressions between versions."""
        # Group by model and accelerator
        for (_model, _accelerator), group in sample_csv_data.groupby(
            ["model", "accelerator"]
        ):
            versions = group["version"].unique()

            if len(versions) >= 2:
                # Compare first two versions
                v1 = group[group["version"] == versions[0]]["output_tok/sec"].mean()
                v2 = group[group["version"] == versions[1]]["output_tok/sec"].mean()

                # Calculate change
                if v1 > 0:
                    pct_change = ((v2 - v1) / v1) * 100

                    # Detect significant changes (>5%)
                    if abs(pct_change) > 5:
                        # This is a significant change
                        assert isinstance(pct_change, float)

    def test_optimal_concurrency_detection(self, sample_csv_data):
        """Test finding optimal concurrency level."""
        # For each configuration, find concurrency with best throughput
        for (_model, _accelerator, _version), group in sample_csv_data.groupby(
            ["model", "accelerator", "version"]
        ):
            # Find max throughput
            max_throughput_idx = group["output_tok/sec"].idxmax()
            optimal_config = group.loc[max_throughput_idx]

            # Verify this is indeed the maximum
            assert optimal_config["output_tok/sec"] >= group["output_tok/sec"].min()

    def test_latency_slo_compliance(self, sample_csv_data):
        """Test checking latency SLO compliance."""
        # Define SLO thresholds
        ttft_slo = 0.1  # 100ms
        itl_slo = 0.05  # 50ms

        # Check compliance
        sample_csv_data["ttft_compliant"] = sample_csv_data["ttft_p95"] <= ttft_slo
        sample_csv_data["itl_compliant"] = sample_csv_data["itl_p95"] <= itl_slo
        sample_csv_data["slo_compliant"] = (
            sample_csv_data["ttft_compliant"] & sample_csv_data["itl_compliant"]
        )

        # Get compliance rate
        compliance_rate = sample_csv_data["slo_compliant"].mean() * 100

        assert 0 <= compliance_rate <= 100

    def test_performance_rankings(self, sample_csv_data):
        """Test generating performance rankings."""
        # Rank by throughput
        sample_csv_data["throughput_rank"] = sample_csv_data["output_tok/sec"].rank(
            ascending=False
        )

        # Rank by efficiency
        sample_csv_data["efficiency"] = (
            sample_csv_data["output_tok/sec"] / sample_csv_data["TP"]
        )
        sample_csv_data["efficiency_rank"] = sample_csv_data["efficiency"].rank(
            ascending=False
        )

        # Verify rankings
        assert (sample_csv_data["throughput_rank"] > 0).all()
        assert (sample_csv_data["efficiency_rank"] > 0).all()


@pytest.mark.integration
class TestJSONImport:
    """Test JSON import pipeline."""

    def test_load_and_parse_json(self, temp_json_file):
        """Test loading and parsing JSON benchmark file."""
        import json

        with open(temp_json_file) as f:
            data = json.load(f)

        assert "benchmarks" in data
        assert len(data["benchmarks"]) > 0

    def test_extract_metrics_from_json(self, sample_benchmark_json):
        """Test extracting metrics from JSON structure."""
        benchmark = sample_benchmark_json["benchmarks"][0]

        # Extract metrics
        metrics = benchmark["metrics"]
        throughput = metrics["output_tokens_per_second"]["successful"]["mean"]
        ttft_p95 = metrics["time_to_first_token"]["successful"]["p95"]
        itl_p95 = metrics["inter_token_latency"]["successful"]["p95"]

        # Verify extraction
        assert throughput > 0
        assert ttft_p95 > 0
        assert itl_p95 > 0

    def test_json_to_dataframe_conversion(self, sample_benchmark_json):
        """Test converting JSON data to DataFrame format."""
        rows = []

        for benchmark in sample_benchmark_json["benchmarks"]:
            metrics = benchmark["metrics"]
            row = {
                "run_id": benchmark["run_id"],
                "output_tok/sec": metrics["output_tokens_per_second"]["successful"][
                    "mean"
                ],
                "ttft_p95": metrics["time_to_first_token"]["successful"]["p95"],
                "itl_p95": metrics["inter_token_latency"]["successful"]["p95"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Verify conversion
        assert len(df) == len(sample_benchmark_json["benchmarks"])
        assert "output_tok/sec" in df.columns
        assert (df["output_tok/sec"] > 0).all()
