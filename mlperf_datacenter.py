"""MLPerf Inference Datacenter Results Dashboard.

This module provides functionality to load, process, and visualize
MLPerf Inference v5.1 Datacenter benchmark results.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_mlperf_data(file_path: str) -> pd.DataFrame:
    """Load and parse MLPerf Inference Datacenter results CSV.

    The CSV has a complex multi-row header structure that needs special handling.
    Supports both the new standard CSV format and legacy UTF-16 TSV format.

    Args:
        file_path: Path to the MLPerf CSV file

    Returns:
        Parsed DataFrame with flattened column names
    """
    # Try different encodings and delimiters to handle various CSV/TSV exports
    # New format: UTF-8 with comma delimiters (standard CSV)
    # Legacy format: UTF-16 with tab delimiters (TSV)
    encodings = [
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "latin-1",
        "iso-8859-1",
        "cp1252",
    ]
    delimiters = [",", "\t"]  # Try comma first (standard CSV)

    raw_df = None
    successful_encoding = None
    successful_delimiter = None

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                raw_df = pd.read_csv(
                    file_path, header=None, encoding=encoding, sep=delimiter
                )
                # Verify it actually parsed correctly (should have multiple columns)
                if (
                    raw_df.shape[1] > 15
                ):  # Should have at least 15 metadata columns + metric columns
                    successful_encoding = encoding
                    successful_delimiter = delimiter
                    break  # Success!
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                continue  # Try next combination
        if successful_encoding is not None:
            break  # Found a working combination

    if raw_df is None or successful_encoding is None:
        raise ValueError("Could not read file with any encoding/delimiter combination")

    # Extract header rows (rows 0-4 contain header information)
    # Row 0: Benchmark/Model/Scenario/Units labels (not useful)
    # Row 1: "Inference" repeated
    # Row 2: Model names
    # Row 3: Scenarios
    # Row 4: Units

    # Extract header rows for building column names
    header_row = raw_df.iloc[4].fillna("")  # Row 4 has metadata column names

    # Row 2 has model names but with merged cells (NaN for subsequent columns)
    # We need to forward-fill to propagate model names across their scenarios
    model_row = raw_df.iloc[2].copy()
    model_row = model_row.ffill()  # Forward fill NaN values

    scenario_row = raw_df.iloc[3].fillna("")
    units_row = raw_df.iloc[4].fillna("")  # Row 4 also has units for metric columns

    # Read the actual data starting from row 5 (0-indexed) using the same encoding and delimiter
    # Use header=None to avoid treating first data row as column names
    df = pd.read_csv(
        file_path,
        skiprows=5,
        header=None,
        encoding=successful_encoding,
        sep=successful_delimiter,
    )

    # Create meaningful column names by combining model + scenario + units
    new_columns = []
    for i in range(len(df.columns)):
        if i < 15:  # First 15 columns are metadata - use names from header row
            col_name = header_row.iloc[i]
            # Handle NaN or empty column names
            if pd.isna(col_name) or col_name == "":
                new_columns.append(f"col_{i}")
            else:
                new_columns.append(str(col_name))
        else:
            # For metric columns, combine model + scenario + unit
            model = str(model_row.iloc[i]) if pd.notna(model_row.iloc[i]) else ""
            scenario = (
                str(scenario_row.iloc[i]) if pd.notna(scenario_row.iloc[i]) else ""
            )
            unit = str(units_row.iloc[i]) if pd.notna(units_row.iloc[i]) else ""

            # Only create combined name if we have both model and scenario
            if model and scenario and model != "nan" and scenario != "nan":
                new_columns.append(f"{model}_{scenario}_{unit}")
            else:
                new_columns.append(f"col_{i}")

    df.columns = new_columns

    # Make duplicate column names unique by adding suffixes
    # This is necessary because some metadata columns appear multiple times (e.g., Accelerator, Availability)
    new_col_names = list(df.columns)
    seen: dict[str, int] = {}
    for i, col in enumerate(new_col_names):
        if col in seen:
            # This is a duplicate, add suffix
            seen[col] += 1
            new_col_names[i] = f"{col}.{seen[col]}"
        else:
            seen[col] = 0
    df.columns = new_col_names

    # Clean up the dataframe
    # The CSV structure has rows in groups of 4 for each system:
    # Row 1: System metadata (Public ID, Organization, etc.)
    # Row 2: # of Processors
    # Row 3: # of Accelerators
    # Row 4: Avg. Result at System Name (contains actual benchmark values)

    if "Public ID" in df.columns:
        # DON'T remove NaN rows yet - we need them to identify result rows

        # Identify system rows (have actual Public IDs like "5.1-0001")
        system_rows_mask = df["Public ID"].str.match(r"^\d+\.\d+-\d+$", na=False)

        # Identify result rows (contain "Avg. Result" in ANY column)
        # Check all non-metric columns for "Avg. Result" text
        result_rows_mask = pd.Series([False] * len(df), index=df.index, dtype=bool)
        for col in df.columns[:15]:  # Only check metadata columns
            if df[col].dtype == "object":
                mask = df[col].astype(str).str.contains("Avg. Result", na=False)
                result_rows_mask = (result_rows_mask | mask).astype(bool)  # type: ignore[assignment]

        # Get metric columns
        metric_cols = [
            col
            for col in df.columns
            if "_Samples/s" in col or "_Queries/s" in col or "_Tokens/s" in col
        ]

        # Copy metric values from result rows to their corresponding system rows
        # Each result row is 3 rows after its system row
        system_indices = df[system_rows_mask].index.tolist()
        result_indices = df[result_rows_mask].index.tolist()

        for sys_idx, res_idx in zip(system_indices, result_indices):
            if res_idx == sys_idx + 3:  # Verify they're in the expected position
                # Copy metric values from result row to system row
                for col in metric_cols:
                    df.loc[sys_idx, col] = df.loc[res_idx, col]

        # NOW filter to only keep system rows
        df = df[system_rows_mask].copy()

    # Convert numeric columns
    # First remove commas from number strings (e.g., "1,642.22" -> "1642.22")
    for col in df.columns:
        if "_Samples/s" in col or "_Queries/s" in col or "_Tokens/s" in col:
            # Remove commas if the column is string type
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract accelerator count and node count
    # NOTE: '# of Accelerators' in MLPerf data is PER NODE, not total
    if "# of Accelerators" in df.columns:
        df["accelerators_per_node"] = pd.to_numeric(
            df["# of Accelerators"], errors="coerce"
        )

    if "# of Nodes" in df.columns:
        df["node_count"] = pd.to_numeric(df["# of Nodes"], errors="coerce")
        # Default to 1 node if missing
        df["node_count"] = df["node_count"].fillna(1)
    else:
        df["node_count"] = 1

    # Calculate total accelerators across all nodes
    if "accelerators_per_node" in df.columns and "node_count" in df.columns:
        df["accelerator_count"] = df["accelerators_per_node"] * df["node_count"]

    # Rename column for clarity (remove the "(click + for details)" part)
    if "System Name (click + for details)" in df.columns:
        df = df.rename(columns={"System Name (click + for details)": "System Name"})

    # Handle CPU runs where Accelerator is N/A or missing
    # For CPU runs, use system name as the accelerator with "cpu-" prefix
    if "Accelerator" in df.columns and "System Name" in df.columns:
        cpu_mask = (
            df["Accelerator"].isna()
            | (df["Accelerator"] == "N/A")
            | (df["Accelerator"] == "")
            | (df["Accelerator"].astype(str).str.strip() == "")
        )

        if cpu_mask.any():
            # Create CPU accelerator names based on system name
            df.loc[cpu_mask, "Accelerator"] = "cpu-" + df.loc[
                cpu_mask, "System Name"
            ].astype(str)

    return df


def extract_benchmarks_and_scenarios(df: pd.DataFrame) -> dict[str, list[str]]:
    """Extract available benchmarks and their scenarios from column names.

    Args:
        df: MLPerf DataFrame

    Returns:
        Dictionary mapping benchmark names to list of available scenarios
    """
    benchmarks: dict[str, list[str]] = {}

    for col in df.columns:
        if "_Offline_" in col or "_Server_" in col or "_Interactive_" in col:
            # Extract model name and scenario
            parts = col.split("_")
            if len(parts) >= 2:
                # Model name might have dashes or numbers
                model_parts = []
                for part in parts:
                    if part in ["Offline", "Server", "Interactive"]:
                        break
                    model_parts.append(part)

                model = "-".join(model_parts)

                if "Offline" in col:
                    scenario = "Offline"
                elif "Server" in col:
                    scenario = "Server"
                elif "Interactive" in col:
                    scenario = "Interactive"
                else:
                    continue

                if model not in benchmarks:
                    benchmarks[model] = []
                if scenario not in benchmarks[model]:
                    benchmarks[model].append(scenario)

    return benchmarks


def render_mlperf_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Render smart cascading filter UI and return filtered dataframe.

    Filters are hierarchical:
    1. Benchmark (first layer)
    2. Scenario (based on selected benchmarks)
    3. Organization (based on benchmark-scenario)
    4. Accelerator (based on org)
    5. # of Accelerators (based on accelerator)

    Only "available" entries are shown (preview data excluded).

    Args:
        df: MLPerf DataFrame

    Returns:
        Tuple of (filtered_df, filter_selections)
    """
    st.markdown("### Filter your data")
    st.caption("Each filter updates based on your previous selections")

    # Initialize session state for filter management
    if "mlperf_filters_initialized" not in st.session_state:
        st.session_state.mlperf_filters_initialized = True
        st.session_state.mlperf_filter_change_key = 0
        st.session_state.mlperf_filters_were_cleared = False

    # Filter out preview data - only show available entries
    # df_available = df[df['Availability'] == 'available'].copy()
    df_available = df.copy()
    # Create columns for filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns([1, 1, 1])

    # Get benchmarks and scenarios from column names
    benchmarks_dict = extract_benchmarks_and_scenarios(df_available)
    all_benchmarks = sorted(benchmarks_dict.keys())

    # FILTER 1: MLC Model (First layer - no dependencies)
    with filter_col1:
        # Determine baseline defaults
        baseline_models = []
        preferred_models = ["llama3.1-8b-datacenter"]
        for model in preferred_models:
            if model in all_benchmarks:
                baseline_models.append(model)

        if not baseline_models:
            baseline_models = (
                all_benchmarks[:2] if len(all_benchmarks) > 2 else all_benchmarks
            )

        # Store baseline in session state on first run
        if "mlperf_baseline_models" not in st.session_state:
            st.session_state.mlperf_baseline_models = baseline_models

        # Determine current defaults based on state
        if st.session_state.get(
            "mlperf_clear_all_filters", False
        ) or st.session_state.get("mlperf_filters_were_cleared", False):
            default_models = []
        elif st.session_state.get("mlperf_reset_to_defaults", False):
            default_models = st.session_state.mlperf_baseline_models
        else:
            default_models = baseline_models

        selected_benchmarks = st.multiselect(
            "1️⃣ Select MLC Model(s)",
            options=all_benchmarks,
            default=default_models,
            key=f"mlperf_bench_filter_{st.session_state.mlperf_filter_change_key}",
            help="Select which MLC (MLCommons) models to analyze",
        )

    # Determine available scenarios based on selected benchmarks
    available_scenarios: list[str]
    if selected_benchmarks:
        scenarios_set = set()
        for bench in selected_benchmarks:
            if bench in benchmarks_dict:
                scenarios_set.update(benchmarks_dict[bench])
        available_scenarios = sorted(scenarios_set)
    else:
        available_scenarios = []

    # FILTER 2: Scenario (Second layer - depends on benchmarks)
    with filter_col2:
        if available_scenarios:
            # Determine baseline defaults
            baseline_scenarios = []
            preferred_scenarios = ["Offline", "Server"]
            for scenario in preferred_scenarios:
                if scenario in available_scenarios:
                    baseline_scenarios.append(scenario)

            if not baseline_scenarios:
                baseline_scenarios = (
                    available_scenarios[:1] if available_scenarios else []
                )

            # Store baseline in session state on first run
            if "mlperf_baseline_scenarios" not in st.session_state:
                st.session_state.mlperf_baseline_scenarios = baseline_scenarios

            # Determine current defaults based on state
            if st.session_state.get(
                "mlperf_clear_all_filters", False
            ) or st.session_state.get("mlperf_filters_were_cleared", False):
                default_scenarios = []
            elif st.session_state.get("mlperf_reset_to_defaults", False):
                default_scenarios = [
                    s
                    for s in st.session_state.mlperf_baseline_scenarios
                    if s in available_scenarios
                ]
            else:
                default_scenarios = baseline_scenarios

            selected_scenarios = st.multiselect(
                "2️⃣ Select Scenario(s)",
                options=available_scenarios,
                default=default_scenarios,
                key=f"mlperf_scenario_filter_{st.session_state.mlperf_filter_change_key}",
                help="Offline=batch, Server=online, Interactive=single-stream",
            )
        else:
            st.multiselect(
                "2️⃣ Select Scenario(s)",
                options=[],
                default=[],
                key=f"mlperf_scenario_filter_{st.session_state.mlperf_filter_change_key}",
                disabled=True,
                help="Select MLC models first",
            )
            selected_scenarios = []

    # Determine which systems have data for selected benchmark-scenario combinations
    systems_with_data: set[str] = set()
    if selected_benchmarks and selected_scenarios:
        for bench in selected_benchmarks:
            for scenario in selected_scenarios:
                # Find column for this benchmark-scenario
                matching_cols = [
                    col
                    for col in df_available.columns
                    if bench in col and scenario in col
                ]
                if matching_cols:
                    metric_col = matching_cols[0]
                    # Get systems that have non-null data for this metric
                    systems_with_data.update(
                        df_available[df_available[metric_col].notna()][
                            "Organization"
                        ].unique()
                    )

    available_orgs = sorted([org for org in systems_with_data if pd.notna(org)])

    # FILTER 3: Organization (Third layer - depends on benchmark + scenario)
    with filter_col3:
        if available_orgs:
            # Determine baseline defaults
            baseline_orgs = []
            preferred_orgs = ["RedHat"]
            for org in preferred_orgs:
                if org in available_orgs:
                    baseline_orgs.append(org)

            if not baseline_orgs:
                baseline_orgs = (
                    available_orgs[:5] if len(available_orgs) > 5 else available_orgs
                )

            # Store baseline in session state on first run
            if "mlperf_baseline_orgs" not in st.session_state:
                st.session_state.mlperf_baseline_orgs = baseline_orgs

            # Determine current defaults based on state
            if st.session_state.get(
                "mlperf_clear_all_filters", False
            ) or st.session_state.get("mlperf_filters_were_cleared", False):
                default_orgs = []
            elif st.session_state.get("mlperf_reset_to_defaults", False):
                default_orgs = [
                    o
                    for o in st.session_state.mlperf_baseline_orgs
                    if o in available_orgs
                ]
            elif st.session_state.get("mlperf_select_all_orgs", False):
                default_orgs = available_orgs  # Select all organizations for current model-scenario
            else:
                default_orgs = baseline_orgs

            selected_orgs = st.multiselect(
                "3️⃣ Select Organization(s)",
                options=available_orgs,
                default=default_orgs,
                key=f"mlperf_org_filter_{st.session_state.mlperf_filter_change_key}",
                help="Vendors with data for selected models",
            )
        else:
            st.multiselect(
                "3️⃣ Select Organization(s)",
                options=[],
                default=[],
                key=f"mlperf_org_filter_{st.session_state.mlperf_filter_change_key}",
                disabled=True,
                help="Select MLC models and scenarios first",
            )
            selected_orgs = []

    # Apply filters so far to get available accelerators
    filtered_so_far = df_available.copy()
    if selected_orgs:
        filtered_so_far = filtered_so_far[
            filtered_so_far["Organization"].isin(selected_orgs)
        ]

    # Get accelerators that have data for the selected MLC models and scenarios
    accelerators_with_data: set[str] = set()
    if selected_benchmarks and selected_scenarios:
        for bench in selected_benchmarks:
            for scenario in selected_scenarios:
                # Find column for this benchmark-scenario
                matching_cols = [
                    col
                    for col in df_available.columns
                    if bench in col and scenario in col
                ]
                if matching_cols:
                    metric_col = matching_cols[0]
                    # Get accelerators from filtered data that have non-null data for this metric
                    valid_rows = filtered_so_far[filtered_so_far[metric_col].notna()]
                    accelerators_with_data.update(valid_rows["Accelerator"].unique())

        available_accelerators = sorted(
            [acc for acc in accelerators_with_data if pd.notna(acc) and acc != ""]
        )
    else:
        # Fallback to all accelerators if no models/scenarios selected
        available_accelerators = sorted(
            [
                acc
                for acc in filtered_so_far["Accelerator"].unique()
                if pd.notna(acc) and acc != ""
            ]
        )

    # FILTER 4: Accelerator (Fourth layer - depends on organization)
    with filter_col4:
        if available_accelerators:
            # Determine baseline defaults
            baseline_accelerators = []
            preferred_accelerators = ["NVIDIA L40S", "NVIDIA H100-SXM-80GB"]
            for acc in preferred_accelerators:
                if acc in available_accelerators:
                    baseline_accelerators.append(acc)

            if not baseline_accelerators:
                baseline_accelerators = (
                    available_accelerators[:5]
                    if len(available_accelerators) > 5
                    else available_accelerators
                )

            # Store baseline in session state on first run
            if "mlperf_baseline_accelerators" not in st.session_state:
                st.session_state.mlperf_baseline_accelerators = baseline_accelerators

            # Determine current defaults based on state
            if st.session_state.get(
                "mlperf_clear_all_filters", False
            ) or st.session_state.get("mlperf_filters_were_cleared", False):
                default_accelerators = []
            elif st.session_state.get("mlperf_reset_to_defaults", False):
                default_accelerators = [
                    a
                    for a in st.session_state.mlperf_baseline_accelerators
                    if a in available_accelerators
                ]
            elif selected_orgs:
                # Auto-select all available accelerators when organization(s) are selected
                default_accelerators = available_accelerators
            else:
                default_accelerators = baseline_accelerators

            selected_accelerators = st.multiselect(
                "4️⃣ Select Accelerator(s)",
                options=available_accelerators,
                default=default_accelerators,
                key=f"mlperf_acc_filter_{st.session_state.mlperf_filter_change_key}",
                help="GPU/accelerator types",
            )
        else:
            st.multiselect(
                "4️⃣ Select Accelerator(s)",
                options=[],
                default=[],
                key=f"mlperf_acc_filter_{st.session_state.mlperf_filter_change_key}",
                disabled=True,
                help="Select organizations first",
            )
            selected_accelerators = []

    # Apply accelerator filter
    if selected_accelerators:
        filtered_so_far = filtered_so_far[
            filtered_so_far["Accelerator"].isin(selected_accelerators)
        ]

    # Get available accelerator counts that have data for selected MLC models and scenarios
    acc_counts_with_data: set[float] = set()
    if (
        selected_benchmarks
        and selected_scenarios
        and "accelerator_count" in filtered_so_far.columns
    ):
        for bench in selected_benchmarks:
            for scenario in selected_scenarios:
                # Find column for this benchmark-scenario
                matching_cols = [
                    col
                    for col in df_available.columns
                    if bench in col and scenario in col
                ]
                if matching_cols:
                    metric_col = matching_cols[0]
                    # Get accelerator counts from filtered data that have non-null data for this metric
                    valid_rows = filtered_so_far[filtered_so_far[metric_col].notna()]
                    acc_counts_with_data.update(
                        valid_rows["accelerator_count"].dropna().unique()
                    )

        available_acc_counts = sorted([int(c) for c in acc_counts_with_data if c > 0])
    elif "accelerator_count" in filtered_so_far.columns:
        # Fallback to all counts if no models/scenarios selected
        available_acc_counts = sorted(
            [
                int(c)
                for c in filtered_so_far["accelerator_count"].dropna().unique()
                if c > 0
            ]
        )
    else:
        available_acc_counts = []

    # FILTER 5: # of Accelerators (Fifth layer - depends on accelerator)
    with filter_col5:
        if available_acc_counts:
            # Determine baseline defaults
            baseline_acc_counts = []
            preferred_counts = [1]
            for count in preferred_counts:
                if count in available_acc_counts:
                    baseline_acc_counts.append(count)

            if not baseline_acc_counts:
                baseline_acc_counts = available_acc_counts

            # Store baseline in session state on first run
            if "mlperf_baseline_acc_counts" not in st.session_state:
                st.session_state.mlperf_baseline_acc_counts = baseline_acc_counts

            # Determine current defaults based on state
            if st.session_state.get(
                "mlperf_clear_all_filters", False
            ) or st.session_state.get("mlperf_filters_were_cleared", False):
                default_acc_counts = []
            elif st.session_state.get("mlperf_reset_to_defaults", False):
                default_acc_counts = [
                    c
                    for c in st.session_state.mlperf_baseline_acc_counts
                    if c in available_acc_counts
                ]
            elif selected_orgs:
                # Auto-select all available accelerator counts when organization(s) are selected
                default_acc_counts = available_acc_counts
            else:
                default_acc_counts = baseline_acc_counts

            selected_acc_counts = st.multiselect(
                "5️⃣ Select Total # of Accelerators",
                options=available_acc_counts,
                default=default_acc_counts,
                key=f"mlperf_acc_count_filter_{st.session_state.mlperf_filter_change_key}",
                help="Total GPUs across all nodes (# of Nodes × Accelerators per node)",
            )
        else:
            st.multiselect(
                "5️⃣ Select Total # of Accelerators",
                options=[],
                default=[],
                key=f"mlperf_acc_count_filter_{st.session_state.mlperf_filter_change_key}",
                disabled=True,
                help="Select accelerators first",
            )
            selected_acc_counts = []

    # FILTER CONTROL BUTTONS
    with filter_col6:
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button(
                "🌐 Select All Orgs",
                help="Select all organizations for the current model-scenario combination",
                key="mlperf_select_all_orgs_btn",
            ):
                st.session_state.mlperf_clear_all_filters = False
                st.session_state.mlperf_filters_were_cleared = False
                st.session_state.mlperf_reset_to_defaults = False
                st.session_state.mlperf_select_all_orgs = True
                st.session_state.mlperf_filter_change_key += 1
                st.rerun()

        with btn_col2:
            if st.button(
                "🔄 Reset to defaults",
                help="Reset filters to system defaults",
                key="mlperf_reset_btn",
            ):
                st.session_state.mlperf_clear_all_filters = False
                st.session_state.mlperf_filters_were_cleared = False
                st.session_state.mlperf_reset_to_defaults = True
                st.session_state.mlperf_select_all_orgs = False
                st.session_state.mlperf_filter_change_key += 1
                st.rerun()

        with btn_col3:
            if st.button(
                "🧹 Clear all filters",
                help="Clear all filter selections",
                key="mlperf_clear_btn",
            ):
                st.session_state.mlperf_clear_all_filters = True
                st.session_state.mlperf_filters_were_cleared = True
                st.session_state.mlperf_select_all_orgs = False
                st.session_state.mlperf_filter_change_key += 1
                st.rerun()

    # Apply all filters
    filtered_df = df_available.copy()

    if selected_orgs:
        filtered_df = filtered_df[filtered_df["Organization"].isin(selected_orgs)]

    if selected_accelerators:
        filtered_df = filtered_df[
            filtered_df["Accelerator"].isin(selected_accelerators)
        ]

    # For accelerator count filtering, exclude CPU runs (they don't have accelerator counts)
    if selected_acc_counts and "accelerator_count" in filtered_df.columns:
        # Identify CPU runs
        cpu_mask = (
            filtered_df["Accelerator"].astype(str).str.startswith("cpu-", na=False)
        )
        # Apply count filter only to non-CPU runs
        non_cpu_filtered = filtered_df[
            ~cpu_mask & filtered_df["accelerator_count"].isin(selected_acc_counts)
        ]
        # Include all CPU runs regardless of count filter
        cpu_runs = filtered_df[cpu_mask]
        # Combine them
        filtered_df = pd.concat([non_cpu_filtered, cpu_runs], ignore_index=False)

    filter_selections = {
        "benchmarks": selected_benchmarks,
        "scenarios": selected_scenarios,
        "organizations": selected_orgs,
        "accelerators": selected_accelerators,
        "acc_counts": selected_acc_counts,
        "availability": ["available"],  # Always filter to available only
    }

    # Reset state flags after processing
    if st.session_state.get("mlperf_clear_all_filters", False):
        st.session_state.mlperf_clear_all_filters = False
    if st.session_state.get("mlperf_reset_to_defaults", False):
        st.session_state.mlperf_reset_to_defaults = False
    if st.session_state.get("mlperf_select_all_orgs", False):
        st.session_state.mlperf_select_all_orgs = False

    return filtered_df, filter_selections


def create_benchmark_comparison_chart(
    df: pd.DataFrame, benchmark: str, scenario: str, filter_selections: dict
) -> Optional[go.Figure]:
    """Create bar chart comparing systems on a specific benchmark and scenario.

    Args:
        df: Filtered MLPerf DataFrame
        benchmark: Benchmark name
        scenario: Scenario name
        filter_selections: Dictionary of filter selections

    Returns:
        Plotly figure or None if no data
    """
    # Find the column for this benchmark + scenario
    matching_cols = [col for col in df.columns if benchmark in col and scenario in col]

    if not matching_cols:
        return None

    # Use the first matching column
    metric_col = matching_cols[0]

    # Extract unit from column name
    unit = metric_col.split("_")[-1] if "_" in metric_col else "Performance"

    # Prepare data for plotting
    system_col = "System Name"

    # Select columns, handling cases where '# of Accelerators' might not exist
    if "# of Accelerators" in df.columns:
        plot_df = df[
            [system_col, "Accelerator", "Organization", "# of Accelerators", metric_col]
        ].copy()
    else:
        plot_df = df[[system_col, "Accelerator", "Organization", metric_col]].copy()
        plot_df["# of Accelerators"] = None

    # Only drop rows where the actual metric is missing
    plot_df = plot_df.dropna(subset=[metric_col])
    plot_df = plot_df[plot_df[metric_col] > 0]

    # For CPU runs, fill in '# of Accelerators' with a display value
    cpu_mask = plot_df["Accelerator"].astype(str).str.startswith("cpu-", na=False)
    if cpu_mask.any():
        plot_df.loc[cpu_mask, "# of Accelerators"] = "N/A (CPU)"

    if plot_df.empty:
        return None

    # Create unique identifier for each system-organization combination
    # This prevents stacking when multiple orgs use the same system name
    plot_df["System_Display"] = (
        plot_df[system_col] + " (" + plot_df["Organization"] + ")"
    )

    # Sort by performance
    plot_df = plot_df.sort_values(metric_col, ascending=False)

    # Take top 20 for readability
    if len(plot_df) > 20:
        plot_df = plot_df.head(20)

    # Create bar chart
    fig = px.bar(
        plot_df,
        x=metric_col,
        y="System_Display",
        color="Organization",
        hover_data=["Organization", "Accelerator", "# of Accelerators"],
        orientation="h",
        title=f"{benchmark} - {scenario} Scenario<br><sub>Higher is Better ↑</sub>",
        labels={metric_col: f"Performance ({unit})", "System_Display": "System Name"},
        height=max(400, len(plot_df) * 25),
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=True,
        legend={
            "title": "Organization",
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    return fig


def create_normalized_comparison_chart(
    df: pd.DataFrame,
    benchmark: str,
    scenario: str,
    filter_selections: dict,
    norm_method: str,
) -> Optional[go.Figure]:
    """Create normalized bar chart comparing systems on a specific benchmark and scenario.

    Args:
        df: Filtered MLPerf DataFrame
        benchmark: Benchmark name
        scenario: Scenario name
        filter_selections: Dictionary of filter selections
        norm_method: Normalization method string

    Returns:
        Plotly figure or None if no data
    """
    # Find the column for this benchmark + scenario
    matching_cols = [col for col in df.columns if benchmark in col and scenario in col]

    if not matching_cols:
        return None

    # Use the first matching column
    metric_col = matching_cols[0]

    # Extract unit from column name
    unit = metric_col.split("_")[-1] if "_" in metric_col else "Performance"

    # Check if required columns are available
    if "accelerator_count" not in df.columns:
        st.warning("⚠️ Accelerator count data not available for normalization")
        return None

    if "node_count" not in df.columns:
        st.warning("⚠️ Node count data not available for normalization")
        return None

    # Prepare data for plotting
    system_col = "System Name"
    required_cols = [
        system_col,
        "Accelerator",
        "Organization",
        metric_col,
        "accelerator_count",
        "node_count",
        "accelerators_per_node",
    ]
    plot_df = df[required_cols].copy()

    # Exclude CPU runs from normalized comparisons (they don't have GPU counts)
    plot_df = plot_df[
        ~plot_df["Accelerator"].astype(str).str.startswith("cpu-", na=False)
    ]

    plot_df = plot_df.dropna(
        subset=[metric_col, "accelerator_count", "node_count", "accelerators_per_node"]
    )
    plot_df = plot_df[plot_df[metric_col] > 0]
    plot_df = plot_df[plot_df["accelerator_count"] > 0]
    plot_df = plot_df[plot_df["node_count"] > 0]

    if plot_df.empty:
        return None

    # Store original values before normalization
    plot_df["Total_Accelerators"] = plot_df["accelerator_count"].copy()
    plot_df["Num_Nodes"] = plot_df["node_count"].copy()
    plot_df["GPUs_Per_Node"] = plot_df["accelerators_per_node"].copy()
    plot_df["Total_Performance"] = plot_df[metric_col].copy()

    # Calculate per-node performance
    plot_df["Performance_Per_Node"] = plot_df[metric_col] / plot_df["node_count"]

    # Perform normalization
    if "Per GPU" in norm_method:
        # Normalize to per-GPU performance
        plot_df["Normalized_Value"] = plot_df[metric_col] / plot_df["accelerator_count"]
        norm_label = f"Per GPU ({unit}/GPU)"
        title_suffix = "Normalized per GPU"
    else:
        # Normalize to 8-GPU node: (Performance per node) × (8 / GPUs per node)
        plot_df["Normalized_Value"] = plot_df["Performance_Per_Node"] * (
            8 / plot_df["GPUs_Per_Node"]
        )
        norm_label = f"Per 8-GPU Node ({unit}/8-GPU Node)"
        title_suffix = "Normalized per 8-GPU Node"

    # Create unique identifier for each system-organization combination
    # This prevents stacking when multiple orgs use the same system name
    plot_df["System_Display"] = (
        plot_df[system_col] + " (" + plot_df["Organization"] + ")"
    )

    # Sort by normalized performance
    plot_df = plot_df.sort_values("Normalized_Value", ascending=False)

    # Take top 20 for readability
    if len(plot_df) > 20:
        plot_df = plot_df.head(20)

    # Create bar chart with explicit hover template
    fig = px.bar(
        plot_df,
        x="Normalized_Value",
        y="System_Display",
        color="Organization",
        hover_data={
            "Organization": True,
            "Accelerator": True,
            "Num_Nodes": ":.0f",
            "GPUs_Per_Node": ":.0f",
            "Total_Accelerators": ":.0f",
            "Total_Performance": ":,.2f",
            "Performance_Per_Node": ":,.2f",
            "Normalized_Value": ":,.3f",
            "System_Display": False,  # Hide y-axis from hover since it's already visible
        },
        orientation="h",
        title=f"{benchmark} - {scenario} Scenario ({title_suffix})<br><sub>Higher is Better ↑</sub>",
        labels={
            "Normalized_Value": norm_label,
            "System_Display": "System Name",
            "Num_Nodes": "# of Nodes",
            "GPUs_Per_Node": "GPUs per Node",
            "Total_Accelerators": "Total Accelerators",
            "Total_Performance": f"Total {unit}",
            "Performance_Per_Node": f"{unit} per Node",
        },
        height=max(400, len(plot_df) * 25),
    )

    # Add custom hover template
    # customdata order: Organization, Accelerator, Num_Nodes, GPUs_Per_Node, Total_Accelerators, Total_Performance, Performance_Per_Node, Normalized_Value
    if "Per GPU" in norm_method:
        hover_template = (
            "<b>%{y}</b><br>"
            "Organization: %{customdata[0]}<br>"
            "Accelerator: %{customdata[1]}<br>"
            "# of Nodes: %{customdata[2]:.0f}<br>"
            "GPUs per Node: %{customdata[3]:.0f}<br>"
            "Total Accelerators: %{customdata[4]:.0f}<br>"
            "Total Performance: %{customdata[5]:,.2f}<br>"
            "<b>Per GPU: %{x:,.3f}</b>"
            "<extra></extra>"
        )
    else:
        hover_template = (
            "<b>%{y}</b><br>"
            "Organization: %{customdata[0]}<br>"
            "Accelerator: %{customdata[1]}<br>"
            "# of Nodes: %{customdata[2]:.0f}<br>"
            "GPUs per Node: %{customdata[3]:.0f}<br>"
            "Total Accelerators: %{customdata[4]:.0f}<br>"
            "Total Performance: %{customdata[5]:,.2f}<br>"
            "Performance per Node: %{customdata[6]:,.2f}<br>"
            "<b>Per 8-GPU Node: %{x:,.3f}</b>"
            "<extra></extra>"
        )

    fig.update_traces(hovertemplate=hover_template)

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=True,
        legend={
            "title": "Organization",
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    return fig


def render_mlperf_results_table(df: pd.DataFrame, filter_selections: dict):
    """Render interactive results table.

    Args:
        df: Filtered MLPerf DataFrame
        filter_selections: Dictionary of filter selections
    """
    st.markdown("### 📋 Detailed Results")

    # Select columns to display
    metadata_cols = [
        "Public ID",
        "Organization",
        "System Name",
        "Accelerator",
        "# of Nodes",
        "# of Accelerators",
        "Processor",
        "# of  Processors ",
        "Availability",
    ]

    # Add metric columns based on selected benchmarks and scenarios
    metric_cols = []
    benchmarks = filter_selections.get("benchmarks", [])
    scenarios = filter_selections.get("scenarios", [])

    for benchmark in benchmarks:
        for scenario in scenarios:
            matching_cols = [
                col for col in df.columns if benchmark in col and scenario in col
            ]
            metric_cols.extend(matching_cols)

    # Combine columns
    display_cols = [col for col in metadata_cols if col in df.columns] + metric_cols
    display_df = df[display_cols].copy()

    # Remove rows with all NaN metrics
    display_df = display_df.dropna(subset=metric_cols, how="all")

    if display_df.empty:
        st.warning("No results match the selected filters.")
        return

    st.info(
        f"💡 **Tip**: Click on column headers to sort. Hover over column headers for descriptions. Showing {len(display_df)} results."
    )

    # Configure columns with help text
    column_config = {}

    # Define specific help text for key columns
    help_texts = {
        "# of Nodes": "Number of nodes in the system",
        "# of Accelerators": "Number of accelerators (GPUs) per node",
        "Accelerator": "Type of accelerator/GPU used",
        "Organization": "Organization that submitted the results",
        "System Name": "Name of the system configuration",
        "Public ID": "MLPerf submission identifier",
        "Processor": "CPU processor model",
        "# of  Processors ": "Number of CPU processors",
        "Availability": "Results availability status (available/preview)",
    }

    for col in display_cols:
        if col in metadata_cols:
            help_text = help_texts.get(col, f"System {col.lower()}")
            column_config[col] = st.column_config.TextColumn(col, help=help_text)
        else:
            # Metric column - show performance metric description
            help_text = f"Performance metric for {col}"
            column_config[col] = st.column_config.NumberColumn(
                col, help=help_text, format="%.2f"
            )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        column_config=column_config,
        hide_index=True,
    )


def render_mlperf_dashboard(mlperf_csv_path: str):
    """Main function to render the MLPerf Datacenter dashboard.

    Args:
        mlperf_csv_path: Path to MLPerf CSV file
    """
    st.markdown("##  MLPerf Inference - Datacenter")
    st.markdown(
        "MLPerf Inference Datacenter benchmark measures how fast systems can process "
        "inputs and produce results using trained models. This dashboard shows results from industry-wide submissions."
    )

    # Load data
    try:
        with st.spinner("Loading MLPerf data..."):
            df = load_mlperf_data(mlperf_csv_path)
        # st.success(f"✅ Loaded {len(df)} MLPerf system submissions")
    except FileNotFoundError:
        st.error(f"❌ Error: MLPerf CSV file not found at '{mlperf_csv_path}'")
        st.info(
            "Please ensure the file 'Table - Inference.csv' is in the dashboard directory."
        )
        return
    except Exception as e:
        st.error(f"❌ Error loading MLPerf data: {e}")
        st.info(
            "💡 **Troubleshooting tips:**\n"
            "- Check that the CSV file is not corrupted\n"
            "- Ensure the file has the correct multi-row header structure\n"
            "- Try re-downloading the CSV from MLPerf results"
        )
        return

    # Render filters
    filtered_df, filter_selections = render_mlperf_filters(df)

    if filtered_df.empty:
        st.warning("⚠️ No systems match the selected filters.")
        return

    st.markdown(f"**{len(filtered_df)} submissions match your filters**")

    # Add custom CSS to make tabs bigger
    st.markdown(
        """
        <style>
        /* Make MLPerf tabs bigger */
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 2rem;
            padding: 1.5rem 3rem;
            font-weight: 700;
            height: auto;
            min-height: 70px;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            font-size: 2.1rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 MLC Model Comparisons",
            "⚖️ Normalized Result Comparisons",
            "📋 Detailed Results",
        ]
    )

    with tab1:
        st.markdown("### 📊 Model-by-Model Comparisons")

        benchmarks = filter_selections.get("benchmarks", [])
        scenarios = filter_selections.get("scenarios", [])

        if not benchmarks or not scenarios:
            st.info(
                "Please select at least one MLC model and one scenario in the filters above."
            )
        else:
            for benchmark in benchmarks:
                for scenario in scenarios:
                    with st.expander(f"🔍 {benchmark} - {scenario}", expanded=False):
                        fig = create_benchmark_comparison_chart(
                            filtered_df, benchmark, scenario, filter_selections
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data available for {benchmark} - {scenario}")

    with tab2:
        st.markdown("### ⚖️ Normalized Result Comparisons")
        st.markdown(
            "Compare hardware performance normalized by GPU count for fair comparisons."
        )
        st.info(
            "ℹ️ **Note**: CPU runs are excluded from this section as normalization by GPU count is not applicable to CPU-only systems."
        )

        with st.expander("ℹ️ How normalization works", expanded=False):
            st.markdown(
                "**Normalization helps compare different hardware configurations fairly:**\n\n"
                "• **Per GPU**: Shows performance efficiency per single GPU\n"
                "  - Calculation: Total ÷ (# of Nodes × GPUs per node)\n\n"
                "• **Per 8-GPU Node**: Normalizes all results to a single 8-GPU node\n"
                "  - Step 1: Per-node performance = Total ÷ # of Nodes\n"
                "  - Step 2: Normalize to 8 GPUs = ((Per-node performance) ÷ GPUs per node) × 8"
            )

        # Normalization selector
        norm_method = st.radio(
            "Normalize by:",
            options=[
                "Per GPU (÷ total GPUs)",
                "Per 8-GPU Node (per-node perf × 8 ÷ GPUs/node)",
            ],
            index=0,
            horizontal=True,
            help="Normalize performance to compare hardware on equal GPU count basis. Note: MLPerf '# of Accelerators' is per node.",
        )

        benchmarks = filter_selections.get("benchmarks", [])
        scenarios = filter_selections.get("scenarios", [])

        if not benchmarks or not scenarios:
            st.info(
                "Please select at least one MLC model and one scenario in the filters above."
            )
        else:
            for benchmark in benchmarks:
                for scenario in scenarios:
                    with st.expander(f"🔍 {benchmark} - {scenario}", expanded=False):
                        fig = create_normalized_comparison_chart(
                            filtered_df,
                            benchmark,
                            scenario,
                            filter_selections,
                            norm_method,
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data available for {benchmark} - {scenario}")

    with tab3:
        render_mlperf_results_table(filtered_df, filter_selections)
