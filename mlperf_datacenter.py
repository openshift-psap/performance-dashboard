"""MLPerf Inference Datacenter Results Dashboard.

This module provides functionality to load, process, and visualize
MLPerf Inference v5.1 Datacenter benchmark results.
"""

import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_styles import (
    generate_color_palette,
    get_mlperf_dashboard_css,
    get_mlperf_table_tooltip_css,
)


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
        for col in df.columns[
            :17
        ]:  # Check metadata columns (includes column 15/16 where "Avg. Result" appears)
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
        # v5.0: result row is 2 rows after system row
        # v5.1: result row is 3 rows after system row
        system_indices = df[system_rows_mask].index.tolist()
        result_indices = df[result_rows_mask].index.tolist()

        for sys_idx, res_idx in zip(system_indices, result_indices):
            # Check if result row is at expected offset (+2 or +3 rows)
            if res_idx in [sys_idx + 2, sys_idx + 3]:
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

    # Normalize column names across different MLPerf versions
    # Remove "(click + for details)" suffixes and standardize column names
    column_mappings = {
        "System Name (click + for details)": "System Name",
        "Accelerator (click + for details)": "Accelerator",
        "Processor (click + for details)": "Processor",
        "Sum of # of  Processors ": "# of  Processors ",
        "Sum of # of Processors": "# of  Processors ",
    }

    df = df.rename(columns=column_mappings)

    # Handle duplicate column names (can occur after renaming)
    # Keep the first occurrence and drop subsequent duplicates
    if df.columns.duplicated().any():
        # Find duplicate columns
        duplicate_mask = df.columns.duplicated(keep="first")
        df.columns[duplicate_mask].tolist()

        # Drop duplicate columns
        df = df.loc[:, ~duplicate_mask]

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


def render_mlperf_filters(
    df: pd.DataFrame, mlperf_versions: dict, selected_version: str
) -> tuple[pd.DataFrame, dict]:
    """Render smart cascading filter UI and return filtered dataframe.

    Filters are hierarchical:
    0. Version (independent)
    1. Benchmark (first layer)
    2. Scenario (based on selected benchmarks)
    3. Organization (based on benchmark-scenario)
    4. Accelerator (based on org)
    5. # of Accelerators (based on accelerator)

    Only "available" entries are shown (preview data excluded).

    Args:
        df: MLPerf DataFrame
        mlperf_versions: Dict mapping version labels to CSV file paths
        selected_version: Currently selected version

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

    # Create columns for filters - add version filter as first column
    filter_row0_col1, filter_row0_col2, filter_row0_col3 = st.columns(3)
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns([1, 1, 1])

    # FILTER 0: MLPerf Version (independent filter)
    with filter_row0_col1:
        version_selector = st.selectbox(
            "0️⃣ Select MLPerf Version",
            options=list(mlperf_versions.keys()),
            index=list(mlperf_versions.keys()).index(selected_version),
            key="mlperf_version_filter",
            help="Choose which MLPerf Inference version results to display",
        )

        # If version changed, trigger reload
        if version_selector != selected_version:
            st.session_state.mlperf_version = version_selector
            st.rerun()

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
            default_model = all_benchmarks[0] if all_benchmarks else None
        elif st.session_state.get("mlperf_reset_to_defaults", False):
            default_model = (
                st.session_state.mlperf_baseline_models[0]
                if st.session_state.mlperf_baseline_models
                else all_benchmarks[0]
            )
        elif st.session_state.get("mlperf_select_all_orgs", False):
            # Preserve current selection when "Select All Orgs" is clicked
            preserved = st.session_state.get("mlperf_preserved_models", baseline_models)
            default_model = preserved[0] if preserved else baseline_models[0]
        elif st.session_state.get("theme_change_only", False):
            # Preserve current selection during theme changes
            preserved = st.session_state.get("mlperf_preserved_models", baseline_models)
            default_model = preserved[0] if preserved else baseline_models[0]
        else:
            default_model = baseline_models[0] if baseline_models else all_benchmarks[0]

        # Get the index of the default model
        default_idx = (
            all_benchmarks.index(default_model)
            if default_model in all_benchmarks
            else 0
        )

        selected_benchmark = st.selectbox(
            "1️⃣ Select MLC Model",
            options=all_benchmarks,
            index=default_idx,
            key=f"mlperf_bench_filter_{st.session_state.mlperf_filter_change_key}",
            help="Select which MLC (MLCommons) model to analyze",
        )

        # Convert to list for compatibility with rest of code
        selected_benchmarks = [selected_benchmark] if selected_benchmark else []

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
            elif st.session_state.get("mlperf_select_all_orgs", False):
                # Preserve current selection when "Select All Orgs" is clicked
                preserved = st.session_state.get(
                    "mlperf_preserved_scenarios", baseline_scenarios
                )
                default_scenarios = [s for s in preserved if s in available_scenarios]
            elif st.session_state.get("theme_change_only", False):
                # Preserve current selection during theme changes
                preserved = st.session_state.get(
                    "mlperf_preserved_scenarios", baseline_scenarios
                )
                default_scenarios = [s for s in preserved if s in available_scenarios]
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
            elif st.session_state.get("theme_change_only", False):
                # Preserve current selection during theme changes
                preserved = st.session_state.get("mlperf_preserved_orgs", baseline_orgs)
                default_orgs = [o for o in preserved if o in available_orgs]
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
            elif st.session_state.get("theme_change_only", False):
                # Preserve current selection during theme changes
                preserved = st.session_state.get(
                    "mlperf_preserved_accelerators", baseline_accelerators
                )
                default_accelerators = [
                    a for a in preserved if a in available_accelerators
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
            elif st.session_state.get("theme_change_only", False):
                # Preserve current selection during theme changes
                preserved = st.session_state.get(
                    "mlperf_preserved_acc_counts", baseline_acc_counts
                )
                default_acc_counts = [c for c in preserved if c in available_acc_counts]
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

    # Store current selections (to preserve them for "Select All Orgs" and theme changes)
    st.session_state.mlperf_preserved_models = selected_benchmarks
    st.session_state.mlperf_preserved_scenarios = selected_scenarios
    st.session_state.mlperf_preserved_orgs = selected_orgs
    st.session_state.mlperf_preserved_accelerators = selected_accelerators
    st.session_state.mlperf_preserved_acc_counts = selected_acc_counts

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
    if st.session_state.get("theme_change_only", False):
        st.session_state.theme_change_only = False

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

    # Show all systems (removed top 20 limit - chart is scrollable)
    # Chart height dynamically adjusts: max(400, len(plot_df) * 25)

    # Generate unique colors for all organizations
    unique_orgs = sorted(plot_df["Organization"].unique())
    colors = generate_color_palette(len(unique_orgs))
    color_map = dict(zip(unique_orgs, colors))

    # Build customdata array explicitly
    customdata = plot_df[["Organization", "Accelerator", "# of Accelerators"]].values

    # Create bar chart
    fig = px.bar(
        plot_df,
        x=metric_col,
        y="System_Display",
        color="Organization",
        color_discrete_map=color_map,
        orientation="h",
        title=f"{benchmark} - {scenario} Scenario<br><sub>Higher is Better ↑</sub>",
        labels={metric_col: f"Performance ({unit})", "System_Display": "System Name"},
        height=max(400, len(plot_df) * 25),
    )

    # Set the customdata explicitly
    fig.update_traces(customdata=customdata)

    # Add custom hover template with bolded important metrics
    # customdata order: Organization, Accelerator, # of Accelerators
    hover_template = (
        "<b>%{y}</b><br>"
        "Organization: %{customdata[0]}<br>"
        "Accelerator: %{customdata[1]}<br>"
        "# of Accelerators: %{customdata[2]:.0f}<br>"
        f"<b>Performance: %{{x:,.2f}} {unit}</b>"
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
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
            "font_family": "sans-serif",
            "font_color": "black",
            "bordercolor": "lightgray",
        },
    )

    return fig


def create_normalized_comparison_chart(
    df: pd.DataFrame,
    benchmark: str,
    scenario: str,
    filter_selections: dict,
    norm_method: str,
    df_unfiltered: Optional[pd.DataFrame] = None,
) -> tuple[Optional[go.Figure], Optional[dict]]:
    """Create normalized bar chart comparing systems on a specific benchmark and scenario.

    Args:
        df: Filtered MLPerf DataFrame
        benchmark: Benchmark name
        scenario: Scenario name
        filter_selections: Dictionary of filter selections
        norm_method: Normalization method string
        df_unfiltered: Unfiltered MLPerf DataFrame for calculating global minimum (optional)

    Returns:
        Tuple of (Plotly figure or None if no data, baseline system info dict or None)
    """
    # Find the column for this benchmark + scenario
    matching_cols = [col for col in df.columns if benchmark in col and scenario in col]

    if not matching_cols:
        return None, None

    # Use the first matching column
    metric_col = matching_cols[0]

    # Extract unit from column name
    unit = metric_col.split("_")[-1] if "_" in metric_col else "Performance"

    # Check if required columns are available
    if "accelerator_count" not in df.columns:
        st.warning("⚠️ Accelerator count data not available for normalization")
        return None, None

    if "node_count" not in df.columns:
        st.warning("⚠️ Node count data not available for normalization")
        return None, None

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
        return None, None

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

    # Calculate global minimum from unfiltered data (if provided) to ensure consistent baseline
    if df_unfiltered is not None and all(
        col in df_unfiltered.columns for col in required_cols
    ):
        try:
            # Process unfiltered data the same way to get global minimum
            global_df = df_unfiltered[required_cols].copy()
            global_df = global_df[
                ~global_df["Accelerator"].astype(str).str.startswith("cpu-", na=False)
            ]
            global_df = global_df.dropna(
                subset=[
                    metric_col,
                    "accelerator_count",
                    "node_count",
                    "accelerators_per_node",
                ]
            )
            global_df = global_df[global_df[metric_col] > 0]
            global_df = global_df[global_df["accelerator_count"] > 0]
            global_df = global_df[global_df["node_count"] > 0]

            if not global_df.empty:
                # Apply same normalization
                if "Per GPU" in norm_method:
                    global_df["Normalized_Value"] = (
                        global_df[metric_col] / global_df["accelerator_count"]
                    )
                else:
                    global_df["Performance_Per_Node"] = (
                        global_df[metric_col] / global_df["node_count"]
                    )
                    global_df["Normalized_Value"] = global_df[
                        "Performance_Per_Node"
                    ] * (8 / global_df["accelerators_per_node"])

                min_normalized = global_df["Normalized_Value"].min()
                # Find the baseline system (the one with minimum normalized value)
                baseline_system = global_df.loc[global_df["Normalized_Value"].idxmin()]
                baseline_info = {
                    "system_name": baseline_system[system_col],
                    "organization": baseline_system["Organization"],
                    "accelerator": baseline_system["Accelerator"],
                    "value": min_normalized,
                }
            else:
                min_normalized = plot_df["Normalized_Value"].min()
                baseline_info = None
        except Exception:
            # Fallback to filtered data minimum if any error
            min_normalized = plot_df["Normalized_Value"].min()
            baseline_info = None
    else:
        # Fallback to filtered data minimum
        min_normalized = plot_df["Normalized_Value"].min()
        baseline_info = None

    # Calculate benefit percentage compared to global minimum
    if min_normalized > 0:
        plot_df["Benefit_%"] = (
            (plot_df["Normalized_Value"] - min_normalized) / min_normalized
        ) * 100
    else:
        plot_df["Benefit_%"] = 0

    # Sort by normalized performance
    plot_df = plot_df.sort_values("Normalized_Value", ascending=False)

    # Show all systems (removed top 20 limit - chart is scrollable)
    # Chart height dynamically adjusts: max(400, len(plot_df) * 25)
    # Note: Benefit % is calculated relative to global minimum across entire dataset

    # Generate unique colors for all organizations
    unique_orgs = sorted(plot_df["Organization"].unique())
    colors = generate_color_palette(len(unique_orgs))
    color_map = dict(zip(unique_orgs, colors))

    # Create bar chart WITH hover_data so Plotly handles the data ordering automatically
    fig = px.bar(
        plot_df,
        x="Normalized_Value",
        y="System_Display",
        color="Organization",
        color_discrete_map=color_map,
        hover_data={
            "Organization": True,
            "Accelerator": True,
            "Num_Nodes": ":.0f",
            "GPUs_Per_Node": ":.0f",
            "Total_Accelerators": ":.0f",
            "Total_Performance": ":,.2f",
            "Performance_Per_Node": ":,.2f",
            "Normalized_Value": ":,.3f",
            "Benefit_%": ":.1f",
            "System_Display": False,
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
            "Benefit_%": "Performance Benefit %",
        },
        height=max(400, len(plot_df) * 25),
    )

    # Let Plotly Express handle the hover template automatically from hover_data
    # This ensures proper data alignment across all traces/organizations

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
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
            "font_family": "sans-serif",
            "font_color": "black",
            "bordercolor": "lightgray",
        },
    )

    return fig, baseline_info


def load_dataset_for_model(model_name: str) -> Optional[pd.DataFrame]:
    """Load the dataset file (pickle or JSON) for a specific MLPerf model.

    Args:
        model_name: Name of the MLPerf model (e.g., 'deepseek-r1', 'llama-3.1-8b')

    Returns:
        DataFrame with dataset information including input_length and output_length columns,
        or None if not available
    """
    # Map model names to dataset summary CSV files (keys should match normalized model names)
    # Note: Dots get REPLACED with hyphens during normalization (not removed)
    # CSV files should have columns: input_length, output_length
    dataset_map = {
        "deepseek-r1": "mlperf-data/summaries/deepseek-r1.csv",
        "llama3-1-8b-datacenter": "mlperf-data/summaries/llama3-1-8b-datacenter.csv",
        "llama2-70b-99": "mlperf-data/summaries/llama2-70b-99.csv",
        "llama2-70b-99-9": "mlperf-data/summaries/llama2-70b-99.csv",  # Same dataset as llama2-70b-99
    }

    # Normalize model name for matching (converts "llama3.1-8b-datacenter" → "llama3-1-8b-datacenter")
    model_key = model_name.lower().replace(" ", "-").replace(".", "-")

    if model_key not in dataset_map:
        return None

    csv_path = dataset_map[model_key]

    if not os.path.exists(csv_path):
        return None

    try:
        # Load the pre-generated CSV summary file
        # Expected columns: input_length, output_length
        data = pd.read_csv(csv_path)

        # Validate required columns exist
        if "input_length" not in data.columns or "output_length" not in data.columns:
            st.error(
                f"Dataset CSV must contain 'input_length' and 'output_length' columns. Found: {list(data.columns)}"
            )
            return None

        return data
    except Exception as e:
        st.error(f"Error loading dataset summary: {e}")
        return None


def create_dataset_histograms(
    data: pd.DataFrame,
) -> tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Create histograms for input and output token lengths.

    Args:
        data: DataFrame containing 'input_length' and 'output_length' columns

    Returns:
        Tuple of (input histogram figure, output histogram figure)
    """
    if data is None or data.empty:
        return None, None

    # Check for required columns
    input_col = None
    output_col = None

    # Try different possible column names
    for col in data.columns:
        col_lower = col.lower()
        if "input" in col_lower and (
            "length" in col_lower or "len" in col_lower or "token" in col_lower
        ):
            input_col = col
        elif "output" in col_lower and (
            "length" in col_lower or "len" in col_lower or "token" in col_lower
        ):
            output_col = col

    if input_col is None or output_col is None:
        st.warning(
            f"Could not find input/output columns. Available columns: {list(data.columns)}"
        )
        return None, None

    # Calculate statistics for input tokens
    input_data = data[input_col].dropna()
    input_mean = input_data.mean()
    input_median = input_data.median()
    input_min = input_data.min()
    input_max = input_data.max()

    # Calculate statistics for output tokens
    output_data = data[output_col].dropna()
    output_mean = output_data.mean()
    output_median = output_data.median()
    output_min = output_data.min()
    output_max = output_data.max()

    # Create input token histogram
    fig_input = go.Figure()
    fig_input.add_trace(
        go.Histogram(
            x=input_data,
            nbinsx=50,
            marker_color="#1f77b4",  # Blue
            marker_line={"color": "#0d3d5c", "width": 1},  # Add border to bars
            name="Input Tokens",
        )
    )

    # Add mean line
    fig_input.add_vline(
        x=input_mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {input_mean:.2f}",
        annotation_position="top left",
    )

    # Add median line
    fig_input.add_vline(
        x=input_median,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Median: {input_median:.2f}",
        annotation_position="top",
    )

    # Add max line
    fig_input.add_vline(
        x=input_max,
        line_dash="dashdot",
        line_color="green",
        annotation_text=f"Max: {int(input_max)}",
        annotation_position="top right",
    )

    fig_input.update_layout(
        title=f"Histogram of Input Token Length<br><sub>Mean: {input_mean:.2f}, Median: {input_median:.2f}, Min: {int(input_min)}, Max: {int(input_max)}</sub>",
        xaxis_title="Input Token Length",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    # Create output token histogram
    fig_output = go.Figure()
    fig_output.add_trace(
        go.Histogram(
            x=output_data,
            nbinsx=50,
            marker_color="#8B4513",  # Brown/red
            marker_line={"color": "#5c2a0a", "width": 1},  # Add border to bars
            name="Output Tokens",
        )
    )

    # Add mean line
    fig_output.add_vline(
        x=output_mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {output_mean:.2f}",
        annotation_position="top left",
    )

    # Add median line
    fig_output.add_vline(
        x=output_median,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Median: {output_median:.2f}",
        annotation_position="top",
    )

    # Add max line
    fig_output.add_vline(
        x=output_max,
        line_dash="dashdot",
        line_color="green",
        annotation_text=f"Max: {int(output_max)}",
        annotation_position="top right",
    )

    fig_output.update_layout(
        title=f"Histogram of Output Token Length<br><sub>Mean: {output_mean:.2f}, Median: {output_median:.2f}, Min: {int(output_min)}, Max: {int(output_max)}</sub>",
        xaxis_title="Output Token Length",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    return fig_input, fig_output


def create_offline_vs_server_comparison(df: pd.DataFrame, filter_selections: dict):
    """Create comparison of Offline vs Server performance degradation.

    Args:
        df: Filtered MLPerf DataFrame
        filter_selections: Dictionary of filter selections

    Returns:
        DataFrame with degradation comparison or None if no common systems
    """
    benchmarks = filter_selections.get("benchmarks", [])

    if not benchmarks:
        return None

    comparison_data = []

    for benchmark in benchmarks:
        # Find Offline and Server columns for this benchmark
        offline_cols = [
            col for col in df.columns if benchmark in col and "Offline" in col
        ]
        server_cols = [
            col for col in df.columns if benchmark in col and "Server" in col
        ]

        if not offline_cols or not server_cols:
            continue

        offline_col = offline_cols[0]
        server_col = server_cols[0]

        # Get systems that have data for both scenarios
        systems_with_both = df[df[offline_col].notna() & df[server_col].notna()].copy()

        for _, row in systems_with_both.iterrows():
            offline_perf = row[offline_col]
            server_perf = row[server_col]

            if offline_perf > 0:
                # Calculate degradation: ((offline - server) / offline) * 100
                degradation_pct = ((offline_perf - server_perf) / offline_perf) * 100

                comparison_data.append(
                    {
                        "Benchmark": benchmark,
                        "Organization": row.get("Organization", "Unknown"),
                        "System Name": row.get("System Name", "Unknown"),
                        "Accelerator": row.get("Accelerator", "Unknown"),
                        "# of Accelerators": row.get("# of Accelerators", "N/A"),
                        "Offline Performance": offline_perf,
                        "Server Performance": server_perf,
                        "Degradation %": degradation_pct,
                        "Server/Offline Ratio": (server_perf / offline_perf) * 100
                        if offline_perf > 0
                        else 0,
                    }
                )

    if not comparison_data:
        return None

    return pd.DataFrame(comparison_data)


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

    # Reorder columns: metadata, then all metrics
    final_cols = [col for col in metadata_cols if col in df.columns]

    # Add all metric columns
    final_cols.extend(metric_cols)

    display_df = display_df[final_cols]

    st.info(
        f"💡 **Tip**: Click on column headers to sort. Showing {len(display_df)} results."
    )

    # Apply tooltip CSS for table column headers in light mode
    st.markdown(get_mlperf_table_tooltip_css(), unsafe_allow_html=True)

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

    for col in final_cols:
        if col in metadata_cols:
            help_text = help_texts.get(col, f"System {col.lower()}")
            # Pin Organization and System Name columns
            if col == "Organization" or col == "System Name":
                column_config[col] = st.column_config.TextColumn(
                    col, help=help_text, pinned=True
                )
            else:
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


def create_version_comparison(
    version_data: dict, model_name: str, scenarios: list, filter_selections: dict
) -> Optional[pd.DataFrame]:
    """Create a comparison dataframe across multiple MLPerf versions.

    Args:
        version_data: Dict mapping version labels to DataFrames
        model_name: Selected MLC model name
        scenarios: List of selected scenarios
        filter_selections: Current filter selections

    Returns:
        DataFrame with version comparison data, or None if no data
    """
    if not version_data or not scenarios:
        return None

    comparison_rows = []

    # For each version, extract relevant data
    for version_label, df in version_data.items():
        # Apply filters (organizations, accelerators, etc.)
        filtered_df = df.copy()

        orgs = filter_selections.get("organizations", [])
        if orgs:
            filtered_df = filtered_df[filtered_df["Organization"].isin(orgs)]

        accelerators = filter_selections.get("accelerators", [])
        if accelerators:
            filtered_df = filtered_df[filtered_df["Accelerator"].isin(accelerators)]

        # Extract metrics for each scenario
        for scenario in scenarios:
            # Find metric column
            matching_cols = [
                col
                for col in filtered_df.columns
                if model_name in col and scenario in col
            ]
            if not matching_cols:
                continue

            metric_col = matching_cols[0]
            unit = metric_col.split("_")[-1] if "_" in metric_col else "Performance"

            # Get systems with data for this metric
            systems_with_data = filtered_df[filtered_df[metric_col].notna()].copy()

            for _, row in systems_with_data.iterrows():
                system_display = f"{row.get('System Name', 'Unknown')} ({row['Organization']}) - {row['Accelerator']}"
                comparison_rows.append(
                    {
                        "Version": version_label,
                        "Organization": row["Organization"],
                        "System Name": row["System Name"],
                        "Accelerator": row["Accelerator"],
                        "# of Accelerators": row.get("# of Accelerators", "N/A"),
                        "Scenario": scenario,
                        "Performance": row[metric_col],
                        "Unit": unit,
                        "System_Display": system_display,
                        "_System_Key": f"{row['Organization']}_{row.get('System Name', 'Unknown')}_{row['Accelerator']}_{scenario}",
                    }
                )

    if not comparison_rows:
        return None

    comparison_df = pd.DataFrame(comparison_rows)

    # Only keep systems that appear in multiple versions (using internal _System_Key)
    system_counts = comparison_df.groupby("_System_Key")["Version"].nunique()
    multi_version_systems = system_counts[system_counts > 1].index

    comparison_df = comparison_df[
        comparison_df["_System_Key"].isin(multi_version_systems)
    ]

    if comparison_df.empty:
        return None

    # Drop the internal key column before returning
    comparison_df = comparison_df.drop(columns=["_System_Key"])

    return comparison_df


def create_version_comparison_chart(
    comparison_df: pd.DataFrame, model_name: str, scenarios: list
) -> Optional[go.Figure]:
    """Create a grouped bar chart comparing performance across versions.

    Args:
        comparison_df: DataFrame with version comparison data
        model_name: Selected MLC model name
        scenarios: List of scenarios

    Returns:
        Plotly figure or None
    """
    if comparison_df is None or comparison_df.empty:
        return None

    # System_Display is already created in create_version_comparison function

    # Create grouped bar chart (horizontal)
    fig = px.bar(
        comparison_df,
        x="Performance",
        y="System_Display",
        color="Version",
        barmode="group",
        orientation="h",
        facet_col="Scenario" if len(scenarios) > 1 else None,
        title=f"{model_name} Performance Across Versions<br><sub>Grouped by System (Higher is Better →)</sub>",
        labels={
            "Performance": f"Performance ({comparison_df['Unit'].iloc[0]})",
            "System_Display": "System",
            "Version": "MLPerf Version",
        },
        height=max(500, len(comparison_df["System_Display"].unique()) * 40),
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=True,
        legend={
            "title": "MLPerf Version",
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
        hoverlabel={
            "bgcolor": "white",
            "font_size": 13,
            "font_family": "sans-serif",
            "font_color": "black",
            "bordercolor": "lightgray",
        },
    )

    # Add improvement percentage annotations
    # Calculate % change from earliest to latest version
    for system in comparison_df["System_Display"].unique():
        system_data = comparison_df[
            comparison_df["System_Display"] == system
        ].sort_values("Version")
        if len(system_data) >= 2:
            versions_sorted = sorted(system_data["Version"].unique())
            first_version = versions_sorted[0]
            last_version = versions_sorted[-1]

            first_perf = system_data[system_data["Version"] == first_version][
                "Performance"
            ].iloc[0]
            last_perf = system_data[system_data["Version"] == last_version][
                "Performance"
            ].iloc[0]

            if first_perf > 0:
                ((last_perf - first_perf) / first_perf) * 100
                # Note: We could add annotations here but it might clutter the chart

    return fig


def render_mlperf_dashboard(mlperf_versions: dict):
    """Main function to render the MLPerf Datacenter dashboard.

    Args:
        mlperf_versions: Dict mapping version labels to CSV file paths
    """
    st.markdown("##  MLPerf Inference - Datacenter")
    st.markdown(
        "MLPerf Inference Datacenter benchmark measures how fast systems can process "
        "inputs and produce results using trained models. This dashboard shows results from industry-wide submissions."
    )

    # Initialize session state for version
    if "mlperf_version" not in st.session_state:
        st.session_state.mlperf_version = "v5.1"  # Default to latest version

    # Get the selected version from session state
    selected_version = st.session_state.mlperf_version

    # Get the CSV file path for the selected version
    mlperf_csv_path = mlperf_versions[selected_version]

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

    # Render filters (including version selector)
    filtered_df, filter_selections = render_mlperf_filters(
        df, mlperf_versions, selected_version
    )

    if filtered_df.empty:
        st.warning("⚠️ No systems match the selected filters.")
        return

    st.markdown(f"**{len(filtered_df)} submissions match your filters**")

    # Apply MLPerf dashboard CSS
    st.markdown(get_mlperf_dashboard_css(), unsafe_allow_html=True)

    # Store unfiltered dataframe for global minimum calculation
    df_unfiltered = df

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 MLC Model Comparisons",
            "⚖️ Normalized Result Comparisons",
            "🔄 Offline vs Server Comparison",
            "📋 Detailed Results",
            "📈 Dataset Representation",
            "🔀 Compare Across Versions",
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
            "ℹ️ **Note**: These calculations assume linear scaling, which is an approximation. "
            "**When upscaling** (e.g., 1 GPU → 8 GPUs), results tend to **over-estimate** actual performance. **When downscaling** (e.g., 16 GPUs → 8 GPUs), results tend to **under-estimate** actual performance. \
            CPU runs are excluded from this section as normalization by GPU count is not applicable to CPU-only systems."
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

        st.success(
            "💡 **Performance Benefit**: The charts display a **Performance Benefit %** in the hover tooltip, "
            "which shows the percentage improvement of each system compared to the **lowest performing system for this model** "
            "(regardless of filter selections). The baseline system and it's corresponding value is displayed above each chart for reference. "
            "For example, a 50% benefit means the system performs 50% better than the baseline."
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
                        fig, baseline_info = create_normalized_comparison_chart(
                            filtered_df,
                            benchmark,
                            scenario,
                            filter_selections,
                            norm_method,
                            df_unfiltered,
                        )
                        if fig:
                            # Display baseline information
                            if baseline_info:
                                st.info(
                                    f"📊 **Baseline System** (0% benefit): "
                                    f"**{baseline_info['system_name']}** ({baseline_info['organization']}) - "
                                    f"{baseline_info['accelerator']} - "
                                    f"Normalized Value: {baseline_info['value']:,.2f}"
                                )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data available for {benchmark} - {scenario}")

    with tab3:
        st.markdown("### 🔄 Offline vs Server Performance Comparison")
        st.markdown(
            "Compare performance degradation from Offline (batch) to Server (online) scenarios for systems that have data in both scenarios."
        )

        comparison_df = create_offline_vs_server_comparison(
            filtered_df, filter_selections
        )

        if comparison_df is None or comparison_df.empty:
            st.info(
                "ℹ️ No systems found with data in both Offline and Server scenarios. "
                "Please select benchmarks and ensure your filters include systems tested in both scenarios."
            )
        else:
            st.success(
                f"✅ Found {len(comparison_df)} systems with both Offline and Server results"
            )

            # Display summary statistics
            avg_degradation = comparison_df["Degradation %"].mean()
            min_degradation = comparison_df["Degradation %"].min()
            max_degradation = comparison_df["Degradation %"].max()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Degradation",
                    f"{avg_degradation:.1f}%",
                    help="Average performance drop from Offline to Server mode",
                )
            with col2:
                st.metric(
                    "Minimum Degradation",
                    f"{min_degradation:.1f}%",
                    help="Best case - smallest performance drop",
                )
            with col3:
                st.metric(
                    "Maximum Degradation",
                    f"{max_degradation:.1f}%",
                    help="Worst case - largest performance drop",
                )

            st.markdown("---")

            # Create unique identifier for each system-organization combination
            # This prevents stacking when multiple orgs use the same system name
            comparison_df["System_Display"] = (
                comparison_df["System Name"]
                + " ("
                + comparison_df["Organization"]
                + ")"
            )

            # Generate unique colors for organizations
            unique_orgs = sorted(comparison_df["Organization"].unique())
            colors = generate_color_palette(len(unique_orgs))
            color_map = dict(zip(unique_orgs, colors))

            # Sort comparison_df for plotting
            plot_df = comparison_df.sort_values("Degradation %", ascending=False).copy()

            # Create visualization with hover_data (let Plotly handle data alignment)
            fig = px.bar(
                plot_df,
                x="Degradation %",
                y="System_Display",
                color="Organization",
                color_discrete_map=color_map,
                hover_data={
                    "Organization": True,
                    "Benchmark": True,
                    "Accelerator": True,
                    "# of Accelerators": ":.0f",
                    "Offline Performance": ":,.2f",
                    "Server Performance": ":,.2f",
                    "Degradation %": ":.1f",
                    "Server/Offline Ratio": ":.1f",
                    "System_Display": False,
                },
                orientation="h",
                title="Performance Degradation: Offline → Server (Higher = More Degradation)",
                labels={
                    "Degradation %": "Performance Degradation (%)",
                    "System_Display": "System",
                    "Offline Performance": "Offline Performance",
                    "Server Performance": "Server Performance",
                    "Server/Offline Ratio": "Server/Offline Ratio (%)",
                },
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
                hoverlabel={
                    "bgcolor": "white",
                    "font_size": 13,
                    "font_family": "sans-serif",
                    "font_color": "black",
                    "bordercolor": "lightgray",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display detailed table
            with st.expander("📊 Detailed Comparison Table", expanded=False):
                st.info(
                    "💡 **Note**: Degradation % shows how much performance drops from Offline to Server. "
                    "Server/Offline Ratio shows what percentage of offline performance is retained in server mode."
                )

                # Format the dataframe for display
                display_df = comparison_df.copy()
                display_df["Offline Performance"] = display_df[
                    "Offline Performance"
                ].round(2)
                display_df["Server Performance"] = display_df[
                    "Server Performance"
                ].round(2)
                display_df["Degradation %"] = display_df["Degradation %"].round(1)
                display_df["Server/Offline Ratio"] = display_df[
                    "Server/Offline Ratio"
                ].round(1)

                st.dataframe(
                    display_df.sort_values("Degradation %", ascending=False),
                    use_container_width=True,
                    column_config={
                        "Benchmark": st.column_config.TextColumn(
                            "Benchmark", help="MLC model benchmark"
                        ),
                        "Organization": st.column_config.TextColumn(
                            "Organization", help="Submitting organization", pinned=True
                        ),
                        "System Name": st.column_config.TextColumn(
                            "System Name", help="System configuration", pinned=True
                        ),
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator", help="GPU/accelerator type"
                        ),
                        "# of Accelerators": "# of Accelerators",
                        "Offline Performance": st.column_config.NumberColumn(
                            "Offline Performance",
                            help="Performance in Offline (batch) scenario",
                            format="%.2f",
                        ),
                        "Server Performance": st.column_config.NumberColumn(
                            "Server Performance",
                            help="Performance in Server (online) scenario",
                            format="%.2f",
                        ),
                        "Degradation %": st.column_config.NumberColumn(
                            "Degradation %",
                            help="Performance drop percentage: ((Offline - Server) / Offline) × 100",
                            format="%.1f%%",
                        ),
                        "Server/Offline Ratio": st.column_config.NumberColumn(
                            "Server/Offline Ratio",
                            help="Percentage of offline performance retained in server mode",
                            format="%.1f%%",
                        ),
                    },
                    hide_index=True,
                )

    with tab4:
        render_mlperf_results_table(filtered_df, filter_selections)

    with tab5:
        st.markdown("### 📈 Dataset Representation")
        st.markdown(
            "View token length distribution statistics for the evaluation dataset used for the selected MLC model. "
            "These histograms show the distribution of input (prompt) and output (completion) token lengths in the dataset."
        )

        benchmarks = filter_selections.get("benchmarks", [])

        if not benchmarks:
            st.info(
                "Please select at least one MLC model in the filters above to view dataset statistics."
            )
        elif len(benchmarks) > 1:
            st.info("Please select only one MLC model to view dataset statistics.")
        else:
            model_name = benchmarks[0]
            st.markdown(f"#### Dataset for: **{model_name}**")

            # Load dataset for the selected model
            with st.spinner(f"Loading dataset for {model_name}..."):
                dataset = load_dataset_for_model(model_name)

            if dataset is None:
                st.info(
                    f"📊 Dataset data not available for **{model_name}**.\n\n"
                    "Dataset statistics are currently only available for selected models. "
                    "More datasets will be added in future updates."
                )
            else:
                st.success(
                    f"✅ Loaded {len(dataset)} samples from the {model_name} evaluation dataset"
                )

                # Check if this is a JSON-based dataset (output tokens are estimated)
                if model_name.lower().replace(" ", "-").replace(".", "-") in [
                    "llama3-1-8b-datacenter"
                ]:
                    st.info(
                        "ℹ️ **Note**: Output token lengths for this dataset are estimated based on character count "
                        "(~4 characters per token). Input token lengths are exact from the tokenized data."
                    )

                # Create histograms
                fig_input, fig_output = create_dataset_histograms(dataset)

                if fig_input and fig_output:
                    # Display histograms side by side
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(fig_input, use_container_width=True)

                    with col2:
                        st.plotly_chart(fig_output, use_container_width=True)

                    # Show additional statistics
                    with st.expander("📊 Detailed Statistics", expanded=False):
                        # Find input and output columns
                        input_col = None
                        output_col = None
                        for col in dataset.columns:
                            col_lower = col.lower()
                            if "input" in col_lower and (
                                "length" in col_lower
                                or "len" in col_lower
                                or "token" in col_lower
                            ):
                                input_col = col
                            elif "output" in col_lower and (
                                "length" in col_lower
                                or "len" in col_lower
                                or "token" in col_lower
                            ):
                                output_col = col

                        if input_col and output_col:
                            input_stats = dataset[input_col].describe()
                            output_stats = dataset[output_col].describe()

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Input Token Statistics:**")
                                st.dataframe(
                                    input_stats.to_frame(name="Input Tokens"),
                                    use_container_width=True,
                                )

                            with col2:
                                st.markdown("**Output Token Statistics:**")
                                st.dataframe(
                                    output_stats.to_frame(name="Output Tokens"),
                                    use_container_width=True,
                                )
                else:
                    st.error("Could not generate histograms from the dataset.")

    with tab6:
        st.markdown("### 🔀 Compare Performance Across Versions")
        st.markdown(
            "Compare how systems perform across different MLPerf versions. "
            "This helps track performance improvements and identify optimization opportunities."
        )

        benchmarks = filter_selections.get("benchmarks", [])
        scenarios = filter_selections.get("scenarios", [])

        if not benchmarks:
            st.info(
                "Please select at least one MLC model in the filters above to compare across versions."
            )
        elif len(benchmarks) > 1:
            st.info("Please select only one MLC model to compare across versions.")
        elif not scenarios:
            st.info("Please select at least one scenario to compare across versions.")
        else:
            model_name = benchmarks[0]
            st.markdown(f"#### Comparing **{model_name}** across versions")

            # Load data from all available versions
            with st.spinner("Loading data from all versions..."):
                version_data = {}
                for version_label, csv_path in mlperf_versions.items():
                    try:
                        version_df = load_mlperf_data(csv_path)
                        # Check if this model exists in this version
                        version_benchmarks = extract_benchmarks_and_scenarios(
                            version_df
                        )
                        if model_name in version_benchmarks:
                            version_data[version_label] = version_df
                    except Exception as e:
                        st.warning(f"Could not load {version_label}: {e}")

            if len(version_data) < 2:
                st.info(
                    f"📊 **{model_name}** is only available in {len(version_data)} version(s).\n\n"
                    "Cross-version comparison requires the model to be present in at least 2 versions."
                )
            else:
                st.success(
                    f"✅ Found **{model_name}** in {len(version_data)} versions: {', '.join(version_data.keys())}"
                )

                # Create comparison visualization
                comparison_df = create_version_comparison(
                    version_data, model_name, scenarios, filter_selections
                )

                if comparison_df is not None and not comparison_df.empty:
                    # Display comparison chart
                    fig = create_version_comparison_chart(
                        comparison_df, model_name, scenarios
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Display detailed comparison table
                    with st.expander(
                        "📊 Detailed Version Comparison Table", expanded=True
                    ):
                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                            height=600,
                            column_config={
                                "Organization": st.column_config.TextColumn(
                                    "Organization", pinned=True
                                ),
                                "System Name": st.column_config.TextColumn(
                                    "System Name", pinned=True
                                ),
                            },
                            hide_index=True,
                        )
                else:
                    st.info(
                        "No common systems found across the selected versions and scenarios."
                    )
