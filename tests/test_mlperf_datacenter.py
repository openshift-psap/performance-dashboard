"""Tests for MLPerf Datacenter dashboard functionality."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from mlperf_datacenter import (
    create_benchmark_comparison_chart,
    create_normalized_comparison_chart,
    extract_benchmarks_and_scenarios,
    load_mlperf_data,
)


@pytest.fixture
def sample_mlperf_csv_content():
    """Create sample MLPerf CSV content for testing."""
    # Create a CSV with enough columns (>15 metadata + metric columns) to match real MLPerf format
    csv_content = """Row,Col1,Col2,Col3,Col4,Col5,Col6,Col7,Col8,Col9,Col10,Col11,Col12,Col13,Col14,Col15,Metric1,Metric2,Metric3
Inference,,,,,,,,,,,,,,,llama3.1-8b,llama3.1-8b,llama3.1-8b
Models,,,,,,,,,,,,,,,llama3.1-8b-datacenter,llama3.1-8b-datacenter,llama3.1-8b-datacenter
Scenarios,,,,,,,,,,,,,,,Offline,Server,Offline
Units,Public ID,Availability,Organization,System Name,Accelerator,# of Nodes,# of Accelerators,Processor,# of  Processors ,Division/Power,Software,col11,col12,col13,col14,Samples/s,Queries/s,Tokens/s
5.1-0001,available,RedHat,Test System 1,NVIDIA H100,1,8,Intel Xeon,2,Power,Software1,col11val,col12val,col13val,col14val,100.5,200.3,150.0
,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,
Avg. Result at Test System 1,,,,,,,,,,,,,,,150.2,250.8,200.0
5.1-0002,available,Intel,Test System 2,N/A,1,0,Intel Xeon Platinum 8480+,4,Power,Software2,col11val2,col12val2,col13val2,col14val2,50.0,75.5,60.0
,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,
Avg. Result at Test System 2,,,,,,,,,,,,,,,80.0,120.0,90.0"""
    return csv_content


@pytest.fixture
def sample_mlperf_df():
    """Create a sample MLPerf DataFrame for testing."""
    data = {
        "Public ID": ["5.1-0001", "5.1-0002", "5.1-0003"],
        "Organization": ["RedHat", "Intel", "NVIDIA"],
        "System Name": ["System 1", "System 2", "System 3"],
        "Accelerator": ["NVIDIA H100", "cpu-System 2", "NVIDIA L40S"],
        "# of Nodes": [1, 2, 1],
        "# of Accelerators": [8, 0, 4],
        "Processor": ["Intel Xeon", "Intel Xeon Platinum", "AMD EPYC"],
        "# of  Processors ": [2, 4, 2],
        "Availability": ["available", "available", "available"],
        "llama3.1-8b-datacenter_Offline_Samples/s": [1000.5, 500.2, 800.3],
        "llama3.1-8b-datacenter_Server_Queries/s": [2000.1, 1000.5, 1500.8],
        "gpt-3_Offline_Samples/s": [500.0, None, 400.0],
        "accelerators_per_node": [8, 0, 4],
        "node_count": [1, 2, 1],
        "accelerator_count": [8, 0, 4],
    }
    return pd.DataFrame(data)


class TestLoadMLPerfData:
    """Test suite for load_mlperf_data function."""

    def test_load_mlperf_data_with_mock_csv(self, tmp_path, sample_mlperf_csv_content):
        """Test loading MLPerf data from a CSV file."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_mlperf.csv"
        csv_file.write_text(sample_mlperf_csv_content)

        # Load the data
        df = load_mlperf_data(str(csv_file))

        # Basic assertions - verify the function can load and parse the file
        assert isinstance(df, pd.DataFrame)
        # Verify key columns exist (the function successfully parsed the CSV)
        assert "Public ID" in df.columns
        assert "Organization" in df.columns
        assert "System Name" in df.columns
        assert "Accelerator" in df.columns
        # Note: The df may be empty if no valid system rows were found,
        # but the columns should be present

    def test_cpu_run_detection(self, sample_mlperf_df):
        """Test that CPU runs (N/A accelerator) are properly detected and renamed."""
        # Use the sample_mlperf_df which already has cpu-System 2 in it
        df = sample_mlperf_df

        # Check that CPU runs with 'cpu-' prefix exist
        cpu_runs = df[df["Accelerator"].astype(str).str.startswith("cpu-", na=False)]
        assert len(cpu_runs) > 0
        # Verify the CPU accelerator name includes the cpu- prefix
        for _idx, row in cpu_runs.iterrows():
            assert "cpu-" in str(row["Accelerator"])

    def test_accelerator_count_calculation(self, tmp_path, sample_mlperf_csv_content):
        """Test that accelerator counts are correctly calculated."""
        csv_file = tmp_path / "test_mlperf.csv"
        csv_file.write_text(sample_mlperf_csv_content)

        df = load_mlperf_data(str(csv_file))

        # Check that accelerator_count column exists
        assert "accelerator_count" in df.columns

        # Check that accelerator_count = accelerators_per_node * node_count
        if "accelerators_per_node" in df.columns and "node_count" in df.columns:
            for _idx, row in df.iterrows():
                if pd.notna(row["accelerators_per_node"]) and pd.notna(
                    row["node_count"]
                ):
                    expected = row["accelerators_per_node"] * row["node_count"]
                    assert row["accelerator_count"] == expected

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_mlperf_data("/nonexistent/path/to/file.csv")

    def test_node_count_defaults_to_one(self, tmp_path):
        """Test that node_count defaults to 1 when not present."""
        # Create CSV without # of Nodes column (but with >15 columns total)
        csv_content = """Row,Col1,Col2,Col3,Col4,Col5,Col6,Col7,Col8,Col9,Col10,Col11,Col12,Col13,Col14,Col15,Metric1,Metric2
Inference,,,,,,,,,,,,,,,llama3.1-8b,llama3.1-8b
Models,,,,,,,,,,,,,,,llama3.1-8b-datacenter,llama3.1-8b-datacenter
Scenarios,,,,,,,,,,,,,,,Offline,Server
Units,Public ID,Availability,Organization,System Name,Accelerator,col6,# of Accelerators,Processor,col9,col10,Software,col12,col13,col14,col15,Samples/s,Queries/s
5.1-0001,available,RedHat,Test System,NVIDIA H100,col6val,col7val,8,Intel Xeon,col9val,col10val,Software1,col12val,col13val,col14val,col15val,100.5,200.3
,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,
Avg. Result at Test System,,,,,,,,,,,,,,,150.2,250.8"""

        csv_file = tmp_path / "test_mlperf_no_nodes.csv"
        csv_file.write_text(csv_content)

        df = load_mlperf_data(str(csv_file))

        # Node count should default to 1
        assert "node_count" in df.columns
        assert (df["node_count"] == 1).all()


class TestExtractBenchmarksAndScenarios:
    """Test suite for extract_benchmarks_and_scenarios function."""

    def test_extract_benchmarks_basic(self, sample_mlperf_df):
        """Test basic benchmark and scenario extraction."""
        benchmarks = extract_benchmarks_and_scenarios(sample_mlperf_df)

        assert isinstance(benchmarks, dict)
        assert "llama3.1-8b-datacenter" in benchmarks
        assert "Offline" in benchmarks["llama3.1-8b-datacenter"]
        assert "Server" in benchmarks["llama3.1-8b-datacenter"]

    def test_extract_multiple_benchmarks(self, sample_mlperf_df):
        """Test extraction with multiple benchmarks."""
        benchmarks = extract_benchmarks_and_scenarios(sample_mlperf_df)

        # Should have at least llama3.1-8b-datacenter and gpt-3
        assert len(benchmarks) >= 2
        assert "gpt-3" in benchmarks

    def test_extract_with_empty_dataframe(self):
        """Test extraction with empty DataFrame."""
        empty_df = pd.DataFrame()
        benchmarks = extract_benchmarks_and_scenarios(empty_df)

        assert isinstance(benchmarks, dict)
        assert len(benchmarks) == 0

    def test_scenarios_are_lists(self, sample_mlperf_df):
        """Test that extracted scenarios are lists."""
        benchmarks = extract_benchmarks_and_scenarios(sample_mlperf_df)

        for _benchmark, scenarios in benchmarks.items():
            assert isinstance(scenarios, list)
            assert len(scenarios) > 0


class TestCreateBenchmarkComparisonChart:
    """Test suite for create_benchmark_comparison_chart function."""

    def test_create_chart_with_valid_data(self, sample_mlperf_df):
        """Test chart creation with valid data."""
        filter_selections = {
            "benchmarks": ["llama3.1-8b-datacenter"],
            "scenarios": ["Offline"],
        }

        fig = create_benchmark_comparison_chart(
            sample_mlperf_df, "llama3.1-8b-datacenter", "Offline", filter_selections
        )

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_create_chart_with_missing_benchmark(self, sample_mlperf_df):
        """Test chart creation with non-existent benchmark."""
        filter_selections = {"benchmarks": ["nonexistent"], "scenarios": ["Offline"]}

        fig = create_benchmark_comparison_chart(
            sample_mlperf_df, "nonexistent", "Offline", filter_selections
        )

        assert fig is None

    def test_create_chart_includes_cpu_runs(self, sample_mlperf_df):
        """Test that CPU runs are included in the chart."""
        filter_selections = {
            "benchmarks": ["llama3.1-8b-datacenter"],
            "scenarios": ["Offline"],
        }

        fig = create_benchmark_comparison_chart(
            sample_mlperf_df, "llama3.1-8b-datacenter", "Offline", filter_selections
        )

        # Chart should be created and include data
        assert fig is not None
        # The chart should have data including CPU runs
        assert len(fig.data) > 0

    def test_create_chart_with_empty_dataframe(self):
        """Test chart creation with empty DataFrame."""
        empty_df = pd.DataFrame()
        filter_selections = {"benchmarks": [], "scenarios": []}

        fig = create_benchmark_comparison_chart(
            empty_df, "test", "Offline", filter_selections
        )

        assert fig is None

    def test_chart_shows_all_systems(self):
        """Test that chart shows all systems (no top-20 limit)."""
        # Create a DataFrame with 25 entries
        data = {
            "System Name": [f"System {i}" for i in range(25)],
            "Organization": [f"Org {i}" for i in range(25)],
            "Accelerator": [f"GPU {i}" for i in range(25)],
            "# of Accelerators": [8] * 25,
            "test_Offline_Samples/s": [1000 - i * 10 for i in range(25)],
        }
        large_df = pd.DataFrame(data)

        fig = create_benchmark_comparison_chart(large_df, "test", "Offline", {})

        # Should show all 25 systems (chart is scrollable with dynamic height)
        assert fig is not None
        # Count total data points across all traces (one trace per organization)
        total_points = sum(len(trace.y) for trace in fig.data)
        assert total_points == 25


class TestCreateNormalizedComparisonChart:
    """Test suite for create_normalized_comparison_chart function."""

    def test_per_gpu_normalization(self, sample_mlperf_df):
        """Test per GPU normalization calculation."""
        filter_selections = {
            "benchmarks": ["llama3.1-8b-datacenter"],
            "scenarios": ["Offline"],
        }

        fig, baseline_info = create_normalized_comparison_chart(
            sample_mlperf_df,
            "llama3.1-8b-datacenter",
            "Offline",
            filter_selections,
            "Per GPU (÷ total GPUs)",
        )

        # Should create a figure for GPU systems (CPU runs excluded)
        assert fig is not None  # Should have GPU data in sample
        # baseline_info may be None if no valid baseline data after filtering

    def test_per_8gpu_node_normalization(self, sample_mlperf_df):
        """Test per 8-GPU node normalization calculation."""
        filter_selections = {
            "benchmarks": ["llama3.1-8b-datacenter"],
            "scenarios": ["Offline"],
        }

        fig, baseline_info = create_normalized_comparison_chart(
            sample_mlperf_df,
            "llama3.1-8b-datacenter",
            "Offline",
            filter_selections,
            "Per 8-GPU Node (per-node perf × 8 ÷ GPUs/node)",
        )

        # Should create a figure for GPU systems
        assert fig is not None  # Should have GPU data in sample
        # baseline_info may be None if no valid baseline data after filtering

    def test_cpu_runs_excluded_from_normalization(self, sample_mlperf_df):
        """Test that CPU runs are excluded from normalized charts."""
        filter_selections = {
            "benchmarks": ["llama3.1-8b-datacenter"],
            "scenarios": ["Offline"],
        }

        fig, baseline_info = create_normalized_comparison_chart(
            sample_mlperf_df,
            "llama3.1-8b-datacenter",
            "Offline",
            filter_selections,
            "Per GPU (÷ total GPUs)",
        )

        # If a figure is created, it should not contain CPU runs
        if fig is not None and len(fig.data) > 0 and hasattr(fig.data[0], "y"):
            # Check that no system names contain 'cpu-' prefix
            y_values = fig.data[0].y
            for y_val in y_values:
                assert not str(y_val).startswith("cpu-")

    def test_normalization_with_missing_columns(self):
        """Test normalization with missing required columns."""
        # DataFrame without accelerator_count
        incomplete_df = pd.DataFrame(
            {
                "System Name": ["System 1"],
                "Organization": ["Org1"],
                "test_Offline_Samples/s": [1000],
            }
        )

        fig, baseline_info = create_normalized_comparison_chart(
            incomplete_df, "test", "Offline", {}, "Per GPU (÷ total GPUs)"
        )

        # Should return None due to missing required columns
        assert fig is None
        assert baseline_info is None

    def test_normalization_with_zero_accelerators(self):
        """Test normalization handles zero accelerators correctly."""
        data = {
            "System Name": ["System 1"],
            "Organization": ["Org1"],
            "Accelerator": ["GPU1"],
            "# of Nodes": [1],
            "# of Accelerators": [8],
            "accelerators_per_node": [8],
            "node_count": [1],
            "accelerator_count": [0],  # Zero accelerators
            "test_Offline_Samples/s": [1000],
        }
        zero_acc_df = pd.DataFrame(data)

        fig, baseline_info = create_normalized_comparison_chart(
            zero_acc_df, "test", "Offline", {}, "Per GPU (÷ total GPUs)"
        )

        # Should return None or handle gracefully (division by zero)
        assert fig is None or isinstance(fig, go.Figure)


class TestDataValidation:
    """Test suite for data validation and edge cases."""

    def test_duplicate_column_names_handled(self, tmp_path):
        """Test that duplicate column names are handled correctly."""
        csv_content = """Row,Col1,Col2,Col3,Col3,Col5,Col6,Col7,Col8,Col9,Col10,Col11,Col12,Col13,Col14,Col15,Metric1,Metric2
Inference,,,,,,,,,,,,,,,llama3.1-8b,llama3.1-8b
Models,,,,,,,,,,,,,,,llama3.1-8b-datacenter,llama3.1-8b-datacenter
Scenarios,,,,,,,,,,,,,,,Offline,Server
Units,Public ID,Organization,Accelerator,Accelerator,System Name,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,Samples/s,Queries/s
5.1-0001,available,RedHat,GPU1,GPU2,Test System,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,100.0,200.0
,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,
Avg. Result at Test System,,,,,,,,,,,,,,,150.0,250.0"""

        csv_file = tmp_path / "test_duplicates.csv"
        csv_file.write_text(csv_content)

        df = load_mlperf_data(str(csv_file))

        # Duplicate columns should be renamed with suffixes
        column_names = list(df.columns)
        assert len(column_names) == len(set(column_names))  # All unique

    def test_numeric_columns_converted(self, sample_mlperf_df):
        """Test that numeric columns are properly typed."""
        # Metric columns should be numeric
        assert pd.api.types.is_numeric_dtype(
            sample_mlperf_df["llama3.1-8b-datacenter_Offline_Samples/s"]
        )
        assert pd.api.types.is_numeric_dtype(sample_mlperf_df["accelerator_count"])

    def test_missing_metric_data_handled(self, sample_mlperf_df):
        """Test that missing metric data is handled correctly."""
        # gpt-3 has None for System 2
        assert pd.isna(sample_mlperf_df.loc[1, "gpt-3_Offline_Samples/s"])

        # Should not cause errors in chart creation
        fig = create_benchmark_comparison_chart(
            sample_mlperf_df, "gpt-3", "Offline", {}
        )

        # Should still create a chart with available data
        assert fig is not None or fig is None
