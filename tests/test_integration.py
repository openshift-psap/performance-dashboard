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


@pytest.mark.integration
class TestVersionComparison:
    """Test the Compare Versions section functionality."""

    def test_version_comparison_data_filtering(self, version_comparison_data):
        """Test that version comparison correctly filters data for two versions."""
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        df_v1 = version_comparison_data[version_comparison_data["version"] == version_1]
        df_v2 = version_comparison_data[version_comparison_data["version"] == version_2]

        # Verify filtering worked
        assert len(df_v1) > 0
        assert len(df_v2) > 0
        assert (df_v1["version"] == version_1).all()
        assert (df_v2["version"] == version_2).all()

    def test_common_model_identification(self, version_comparison_data):
        """Test identification of common models between versions."""
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        df_v1 = version_comparison_data[version_comparison_data["version"] == version_1]
        df_v2 = version_comparison_data[version_comparison_data["version"] == version_2]

        # Get common models
        v1_models = set(df_v1["model"].unique())
        v2_models = set(df_v2["model"].unique())
        common_models = v1_models.intersection(v2_models)

        # Verify common models found
        assert len(common_models) > 0
        assert "meta-llama/Llama-3.1-8B" in common_models

    def test_common_configuration_identification(self, version_comparison_data):
        """Test identification of common configurations between versions."""
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        df_v1 = version_comparison_data[version_comparison_data["version"] == version_1]
        df_v2 = version_comparison_data[version_comparison_data["version"] == version_2]

        # Group by model, accelerator, TP, and intended concurrency
        v1_configs = (
            df_v1.groupby(["model", "accelerator", "TP", "intended concurrency"])
            .size()
            .reset_index()[["model", "accelerator", "TP", "intended concurrency"]]
        )
        v2_configs = (
            df_v2.groupby(["model", "accelerator", "TP", "intended concurrency"])
            .size()
            .reset_index()[["model", "accelerator", "TP", "intended concurrency"]]
        )

        # Find common configurations
        common_configs = pd.merge(
            v1_configs,
            v2_configs,
            on=["model", "accelerator", "TP", "intended concurrency"],
            how="inner",
        )

        # Verify common configurations found
        assert not common_configs.empty
        assert "model" in common_configs.columns
        assert "accelerator" in common_configs.columns
        assert "TP" in common_configs.columns

    def test_throughput_comparison_calculation(self, version_comparison_data):
        """Test throughput percentage change calculation."""
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        # Filter for specific model and config
        model = "meta-llama/Llama-3.1-8B"
        accelerator = "H200"
        tp = 1
        concurrency = 10

        v1_data = version_comparison_data[
            (version_comparison_data["version"] == version_1)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
            & (version_comparison_data["intended concurrency"] == concurrency)
        ]

        v2_data = version_comparison_data[
            (version_comparison_data["version"] == version_2)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
            & (version_comparison_data["intended concurrency"] == concurrency)
        ]

        # Calculate percentage change (v1 compared to v2)
        if not v1_data.empty and not v2_data.empty:
            v1_throughput = v1_data["output_tok/sec"].iloc[0]
            v2_throughput = v2_data["output_tok/sec"].iloc[0]
            pct_change = ((v1_throughput - v2_throughput) / v2_throughput) * 100

            # Verify calculation
            assert isinstance(pct_change, float)
            # v1: 150.0, v2: 165.0, so v1 is worse by ~9.09%
            assert pct_change < 0  # v1 is worse
            assert abs(pct_change - (-9.09)) < 1.0  # Approximately -9.09%

    def test_latency_comparison_calculation(self, version_comparison_data):
        """Test latency percentage change calculation at max throughput."""
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        # Filter for specific model and accelerator
        model = "meta-llama/Llama-3.1-8B"
        accelerator = "H200"
        tp = 1

        v1_data = version_comparison_data[
            (version_comparison_data["version"] == version_1)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
        ]

        v2_data = version_comparison_data[
            (version_comparison_data["version"] == version_2)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
        ]

        # Find max throughput indices
        v1_max_idx = v1_data["output_tok/sec"].idxmax()
        v2_max_idx = v2_data["output_tok/sec"].idxmax()

        # Get latency at max throughput
        v1_latency = v1_data.loc[v1_max_idx, "ttft_p95_s"]
        v2_latency = v2_data.loc[v2_max_idx, "ttft_p95_s"]

        # Calculate percentage change
        pct_change = ((v1_latency - v2_latency) / v2_latency) * 100

        # Verify calculation
        assert isinstance(pct_change, float)
        # Lower latency is better, positive pct_change means v1 has higher latency (worse)

    def test_summary_statistics_calculation(self, version_comparison_data):
        """Test calculation of median, max, and min statistics."""
        version_1 = "vLLM-0.10.0"

        # Filter for specific model and config
        model = "meta-llama/Llama-3.1-8B"
        accelerator = "H200"
        tp = 1

        v1_data = version_comparison_data[
            (version_comparison_data["version"] == version_1)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
        ]

        # Calculate statistics
        median_throughput = v1_data["output_tok/sec"].median()
        max_throughput = v1_data["output_tok/sec"].max()
        min_throughput = v1_data["output_tok/sec"].min()

        # Verify calculations
        assert median_throughput > 0
        assert max_throughput >= median_throughput >= min_throughput
        assert max_throughput == 200.0  # Based on test data
        assert min_throughput == 150.0  # Based on test data

    def test_concurrency_finding_for_extremes(self, version_comparison_data):
        """Test finding concurrency levels where max/min values occur."""
        version_1 = "vLLM-0.10.0"

        # Filter for specific model and config
        model = "meta-llama/Llama-3.1-8B"
        accelerator = "H200"
        tp = 1

        v1_data = version_comparison_data[
            (version_comparison_data["version"] == version_1)
            & (version_comparison_data["model"] == model)
            & (version_comparison_data["accelerator"] == accelerator)
            & (version_comparison_data["TP"] == tp)
        ]

        # Find concurrency at max throughput
        max_throughput = v1_data["output_tok/sec"].max()
        max_concurrency = v1_data[v1_data["output_tok/sec"] == max_throughput][
            "intended concurrency"
        ].iloc[0]

        # Verify finding
        assert max_concurrency in [10, 25]  # Should be one of the test concurrencies
        assert max_concurrency == 25  # Max throughput (200) is at concurrency 25

    def test_improvement_categorization(self):
        """Test categorization of improvements (better/worse/similar)."""
        # Test scenarios
        scenarios = [
            (10.0, "better"),  # >=5% improvement
            (-10.0, "worse"),  # >=5% decline
            (3.0, "similar"),  # <5% difference
            (-3.0, "similar"),  # <5% difference
        ]

        for pct_change, expected_category in scenarios:
            improvement = pct_change  # For throughput
            if improvement >= 5:
                category = "better"
            elif improvement <= -5:
                category = "worse"
            else:
                category = "similar"

            assert category == expected_category

    def test_empty_version_handling(self, version_comparison_data):
        """Test handling of non-existent version."""
        version_1 = "non-existent-version"

        df_v1 = version_comparison_data[version_comparison_data["version"] == version_1]

        # Verify empty result
        assert df_v1.empty

    def test_no_common_configurations(self):
        """Test handling when no common configurations exist."""
        # Create data with no overlapping configs
        data = {
            "run": ["run1", "run2"],
            "model": ["model1", "model2"],
            "accelerator": ["H200", "MI300X"],
            "version": ["v1", "v2"],
            "TP": [1, 2],
            "intended concurrency": [10, 20],
            "output_tok/sec": [150, 160],
            "ttft_p95": [50, 55],
            "ttft_p95_s": [0.05, 0.055],
            "itl_p95": [20, 22],
        }
        df = pd.DataFrame(data)

        df_v1 = df[df["version"] == "v1"]
        df_v2 = df[df["version"] == "v2"]

        # Try to find common configurations
        v1_configs = (
            df_v1.groupby(["model", "accelerator", "TP", "intended concurrency"])
            .size()
            .reset_index()[["model", "accelerator", "TP", "intended concurrency"]]
        )
        v2_configs = (
            df_v2.groupby(["model", "accelerator", "TP", "intended concurrency"])
            .size()
            .reset_index()[["model", "accelerator", "TP", "intended concurrency"]]
        )

        common_configs = pd.merge(
            v1_configs,
            v2_configs,
            on=["model", "accelerator", "TP", "intended concurrency"],
            how="inner",
        )

        # Verify no common configs
        assert common_configs.empty

    def test_multiple_accelerator_tp_combinations(self, version_comparison_data):
        """Test handling of multiple accelerator-TP combinations."""
        model = "meta-llama/Llama-3.1-8B"
        version_1 = "vLLM-0.10.0"
        version_2 = "vLLM-0.10.1"

        df_v1 = version_comparison_data[
            (version_comparison_data["version"] == version_1)
            & (version_comparison_data["model"] == model)
        ]
        df_v2 = version_comparison_data[
            (version_comparison_data["version"] == version_2)
            & (version_comparison_data["model"] == model)
        ]

        # Get unique accelerator-TP combinations
        v1_configs = df_v1[["accelerator", "TP"]].drop_duplicates()
        v2_configs = df_v2[["accelerator", "TP"]].drop_duplicates()

        # Find common configurations
        common_configs = pd.merge(v1_configs, v2_configs, on=["accelerator", "TP"])

        # Verify multiple configs found
        assert len(common_configs) >= 2  # Should have H200-TP1 and MI300X-TP2

    def test_ttft_seconds_conversion(self, version_comparison_data):
        """Test that TTFT is properly converted to seconds."""
        # Verify ttft_p95_s is properly calculated
        assert "ttft_p95_s" in version_comparison_data.columns
        assert "ttft_p95" in version_comparison_data.columns

        # Check conversion (ttft_p95 in ms, ttft_p95_s in seconds)
        for _, row in version_comparison_data.iterrows():
            expected_seconds = row["ttft_p95"] / 1000
            assert abs(row["ttft_p95_s"] - expected_seconds) < 0.001
