"""vLLM CPU Dashboard Module.

This module provides functionality to load, process, and visualize
vLLM CPU inference benchmark results across different CPU platforms,
models, and configurations.
"""

import contextlib
import io
import logging
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Set global Plotly template if not already set by main dashboard
if "plotly_white_light" not in pio.templates:
    _light_hover = go.layout.Template(
        layout=go.Layout(
            hoverlabel={
                "bgcolor": "white",
                "font_color": "#262730",
                "bordercolor": "#d1d5db",
            },
        ),
    )
    pio.templates["plotly_white_light"] = pio.templates["plotly_white"]
    pio.templates["plotly_white_light"].layout.update(_light_hover.layout)
    pio.templates.default = "plotly_white_light"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# S3 Configuration from environment variables
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY_CPU = os.environ.get("S3_KEY_CPU", "cpu_dashboard.csv")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def _read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    """Read a CSV file from S3 bucket."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 access. Install with: pip install boto3"
        )

    session_kwargs = {"region_name": region}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        session_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY

    session = boto3.Session(**session_kwargs)
    s3_client = session.client("s3")

    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(io.StringIO(csv_content))


def _keep_expander_open(expander_key):
    """Helper to keep an expander open after widget interaction."""
    st.session_state[expander_key] = True


def assign_profile(row):
    """Assigns a profile label from the actual ISL/OSL values in the data."""
    prompt_toks = int(row["prompt toks"])
    output_toks = int(row["output toks"])
    return f"{prompt_toks}/{output_toks}"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_cpu_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess vLLM CPU benchmark data from CSV file or S3.

    Args:
        file_path: Path to the CSV file to load (fallback).

    Returns:
        DataFrame with loaded and processed data, or None if error occurs.
    """
    try:
        if S3_BUCKET:
            try:
                df = _read_csv_from_s3(S3_BUCKET, S3_KEY_CPU, S3_REGION)
                logger.info(
                    f"Successfully loaded CPU data from S3: s3://{S3_BUCKET}/{S3_KEY_CPU}"
                )
            except Exception as s3_error:
                logger.warning(
                    f"S3 load failed ({s3_error}), falling back to local file"
                )
                df = pd.read_csv(file_path)
        else:
            logger.info(f"Loading CPU data from local file: {file_path}")
            df = pd.read_csv(file_path)

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Convert numeric columns
        numeric_cols = ["TP", "core_count", "omp_num_threads", "cpuset_mems"]
        for col_name in numeric_cols:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # Assign profile based on prompt/output tokens
        if "prompt toks" in df.columns and "output toks" in df.columns:
            df["profile"] = df.apply(assign_profile, axis=1)

        # Calculate efficiency (output tokens/sec per core) where core_count > 0
        if "output_tok/sec" in df.columns and "core_count" in df.columns:
            cores = pd.to_numeric(df["core_count"], errors="coerce")
            df["efficiency"] = np.where(cores > 0, df["output_tok/sec"] / cores, np.nan)

        # Convert TTFT metrics from milliseconds to seconds
        ttft_cols = [
            "ttft_median",
            "ttft_p95",
            "ttft_p1",
            "ttft_p99",
            "ttft_p999",
            "ttft_mean",
        ]
        for ttft_col in ttft_cols:
            if ttft_col in df.columns:
                df[f"{ttft_col}_s"] = df[ttft_col] / 1000

        # Calculate error rate
        if "successful_requests" in df.columns and "errored_requests" in df.columns:
            total_requests = df["successful_requests"] + df["errored_requests"]
            df["error_rate"] = (df["errored_requests"] / total_requests * 100).fillna(0)

        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {str(e)}")
        return None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def render_cpu_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Render cascading filter UI for vLLM CPU dashboard.

    Args:
        df: CPU benchmark DataFrame

    Returns:
        Tuple of (filtered_df, filter_selections)
    """
    st.markdown("### Filter your data")

    # Row 1: Platform, Workload (ISL/OSL), Version
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        all_accelerators = sorted(df["accelerator"].unique().tolist())
        default_accelerators = st.session_state.get(
            "cpu_baseline_accelerators", all_accelerators
        )
        default_accelerators = [
            a for a in default_accelerators if a in all_accelerators
        ]
        if not default_accelerators:
            default_accelerators = all_accelerators

        selected_accelerators = st.multiselect(
            "1️⃣ Platform (CPU)",
            options=all_accelerators,
            default=default_accelerators,
            key="cpu_filter_accelerator",
        )
        if not selected_accelerators:
            selected_accelerators = all_accelerators

    # Cascade: filter available workloads by selected platforms
    acc_filtered = df[df["accelerator"].isin(selected_accelerators)]

    with filter_col2:
        all_profiles = sorted(acc_filtered["profile"].unique().tolist())
        selected_profile = st.selectbox(
            "2️⃣ Workload (ISL/OSL)",
            options=all_profiles,
            index=0,
            key="cpu_filter_profile",
        )

    # Cascade: filter available versions by platform + workload
    profile_filtered = acc_filtered[acc_filtered["profile"] == selected_profile]

    with filter_col3:
        all_versions = sorted(profile_filtered["version"].unique().tolist())
        default_versions = st.session_state.get("cpu_baseline_versions", all_versions)
        default_versions = [v for v in default_versions if v in all_versions]
        if not default_versions:
            default_versions = all_versions

        selected_versions = st.multiselect(
            "3️⃣ Version",
            options=all_versions,
            default=default_versions,
            key="cpu_filter_version",
        )
        if not selected_versions:
            selected_versions = all_versions

    # Row 2: Model, Core Count, OMP Num Threads
    ver_filtered = profile_filtered[profile_filtered["version"].isin(selected_versions)]

    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col4:
        all_models = sorted(ver_filtered["model"].unique().tolist())
        selected_model = st.selectbox(
            "4️⃣ Model",
            options=all_models,
            index=0,
            key="cpu_filter_model",
        )

    model_filtered = ver_filtered[ver_filtered["model"] == selected_model]

    with filter_col5:
        if "core_count" in model_filtered.columns:
            all_cores = sorted(
                [int(c) for c in model_filtered["core_count"].dropna().unique()]
            )
            if all_cores:
                selected_cores = st.multiselect(
                    "5️⃣ Core Count",
                    options=all_cores,
                    default=all_cores,
                    key="cpu_filter_cores",
                )
                if not selected_cores:
                    selected_cores = all_cores
            else:
                selected_cores = None
        else:
            selected_cores = None

    with filter_col6:
        if "omp_num_threads" in model_filtered.columns:
            all_omp = sorted(
                [int(t) for t in model_filtered["omp_num_threads"].dropna().unique()]
            )
            if all_omp:
                selected_omp = st.multiselect(
                    "6️⃣ OMP Num Threads",
                    options=all_omp,
                    default=all_omp,
                    key="cpu_filter_omp",
                )
                if not selected_omp:
                    selected_omp = all_omp
            else:
                selected_omp = None
        else:
            selected_omp = None

    # Apply all filters
    mask = (
        df["accelerator"].isin(selected_accelerators)
        & (df["model"] == selected_model)
        & df["version"].isin(selected_versions)
        & (df["profile"] == selected_profile)
    )

    if selected_cores is not None:
        mask = mask & (df["core_count"].isin(selected_cores) | df["core_count"].isna())

    if selected_omp is not None:
        mask = mask & (
            df["omp_num_threads"].isin(selected_omp) | df["omp_num_threads"].isna()
        )

    filtered_df = df[mask].copy()

    filter_selections = {
        "accelerators": selected_accelerators,
        "model": selected_model,
        "versions": selected_versions,
        "profile": selected_profile,
        "cores": selected_cores,
        "omp_threads": selected_omp,
    }

    return filtered_df, filter_selections


# ---------------------------------------------------------------------------
# Section: Performance Plots
# ---------------------------------------------------------------------------


def render_performance_plots_section(filtered_df, use_expander=True):
    """Render performance plots section for vLLM CPU dashboard."""
    if use_expander:
        ctx = st.expander("📈 Performance Plots", expanded=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("📈 Performance Plots")
        st.markdown(
            "💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view."
        )

        if filtered_df.empty:
            st.warning("No data available for selected filters.")
            return

        # Build run identifier for legend
        filtered_df = filtered_df.copy()
        filtered_df["model_short"] = filtered_df["model"].apply(
            lambda x: x.split("/")[-1] if pd.notna(x) else "Unknown"
        )

        core_str = filtered_df["core_count"].apply(
            lambda v: f"{int(v)}c" if pd.notna(v) and v > 0 else ""
        )

        # Include short uuid to distinguish multiple runs of the same config
        uuid_short = filtered_df["uuid"].apply(lambda u: u[:8] if pd.notna(u) else "")
        filtered_df["run_identifier"] = (
            filtered_df["accelerator"]
            + " | "
            + filtered_df["model_short"]
            + " | "
            + filtered_df["version"]
            + " | "
            + core_str
            + " | "
            + uuid_short
        )

        col1, col2 = st.columns(2)
        with col1:
            x_axis_options = {
                "Concurrency": "intended concurrency",
                "Throughput (Output Tok/s)": "output_tok/sec",
            }
            x_axis_label = st.selectbox(
                "Select X-Axis",
                options=list(x_axis_options.keys()),
                key="cpu_perf_plots_x_axis",
            )
            x_axis = x_axis_options[x_axis_label]

        with col2:
            y_axis_options = {
                "Throughput (Output tokens/second generated)": "output_tok/sec",
                "Efficiency (Output tokens/sec per core)": "efficiency",
                "Inter-Token Latency P95 (Time between tokens, ms)": "itl_p95",
                "Inter-Token Latency Mean (ms)": "itl_mean",
                "Time to First Token P95 (Response start delay, s)": "ttft_p95_s",
                "Time to First Token Mean (s)": "ttft_mean_s",
                "Request Latency Median (Total request processing time, s)": "request_latency_median",
                "Request Latency Max (s)": "request_latency_max",
                "Time Per Output Token P95 (ms)": "tpot_p95",
                "Time Per Output Token Mean (ms)": "tpot_mean",
                "Total Throughput (Total tokens/second processed)": "total_tok/sec",
                "Successful Requests": "successful_requests",
                "Error Rate (% Failed requests)": "error_rate",
            }
            y_axis_label = st.selectbox(
                "Select Y-Axis",
                options=list(y_axis_options.keys()),
                index=0,
                key="cpu_perf_plots_y_axis",
            )
            y_axis = y_axis_options[y_axis_label]

        # Build chart with explicit traces per config group
        fig = go.Figure()
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        grouped = filtered_df.groupby("run_identifier", sort=False)

        # Sort groups by model then platform for consistent ordering
        sorted_groups = sorted(grouped, key=lambda g: g[0])

        for idx, (run_id, group_df) in enumerate(sorted_groups):
            group_df = group_df.sort_values(x_axis)
            fig.add_trace(
                go.Scatter(
                    x=group_df[x_axis],
                    y=group_df[y_axis],
                    name=run_id,
                    mode="lines+markers",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    marker={"size": 7},
                    hovertemplate=(
                        f"<b>{run_id}</b><br>"
                        f"{x_axis_label}: %{{x:.2f}}<br>"
                        f"{y_axis_label}: %{{y:.2f}}<br>"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"{x_axis_label} vs. {y_axis_label}",
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            template="plotly_white_light",
            height=600,
            hovermode="closest",
            legend_title_text="Run Details (Platform | Model | Version | Cores | UUID)",
            legend={"font": {"size": 11}},
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        caption_col1, caption_col2 = st.columns([3, 1])
        with caption_col2:
            st.caption("📜 **Tip**: Scroll within the legend box to see all runs")


# ---------------------------------------------------------------------------
# Section: Compare Versions
# ---------------------------------------------------------------------------


def _compare_metric(data_v1, data_v2, metric_col, higher_is_better):
    """Compare a metric between two version datasets at common concurrency points.

    Returns:
        (pct_diff, v1_better, v1_peak_conc, v2_peak_conc, is_similar)
    """
    v1_concs = set(data_v1["intended concurrency"].dropna().unique())
    v2_concs = set(data_v2["intended concurrency"].dropna().unique())
    common = v1_concs.intersection(v2_concs)

    if not common:
        return None, None, None, None, None

    v1_common = data_v1[data_v1["intended concurrency"].isin(common)]
    v2_common = data_v2[data_v2["intended concurrency"].isin(common)]

    v1_vals = v1_common[metric_col].dropna()
    v2_vals = v2_common[metric_col].dropna()

    if v1_vals.empty or v2_vals.empty:
        return None, None, None, None, None

    if higher_is_better:
        v1_val = v1_vals.max()
        v2_val = v2_vals.max()
        v1_peak_conc = float(v1_common.loc[v1_vals.idxmax(), "intended concurrency"])
        v2_peak_conc = float(v2_common.loc[v2_vals.idxmax(), "intended concurrency"])
    else:
        v1_val = v1_vals.min()
        v2_val = v2_vals.min()
        v1_peak_conc = float(v1_common.loc[v1_vals.idxmin(), "intended concurrency"])
        v2_peak_conc = float(v2_common.loc[v2_vals.idxmin(), "intended concurrency"])

    if v2_val == 0:
        return None, None, None, None, None

    pct_diff = ((v1_val - v2_val) / v2_val) * 100
    v1_better = pct_diff > 0 if higher_is_better else pct_diff < 0

    return pct_diff, v1_better, v1_peak_conc, v2_peak_conc, abs(pct_diff) < 5


@st.fragment
def render_compare_versions_section(df, use_expander=True):
    """Compare two vLLM CPU versions across multiple metrics."""
    if use_expander:
        ctx = st.expander("⚖️ Compare Versions", expanded=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("⚖️ Compare Versions")
        st.markdown(
            "💡 **Compare performance between two vLLM versions across all models and metrics.**"
        )

        available_versions = sorted(df["version"].unique().tolist())

        if len(available_versions) < 2:
            st.warning("⚠️ Need at least 2 versions in the data to compare.")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            version_1 = st.selectbox(
                "Version 1 (Baseline)",
                options=available_versions,
                index=0,
                key="cpu_compare_v1",
            )

        with col2:
            version_2_options = [v for v in available_versions if v != version_1]
            version_2 = (
                st.selectbox(
                    "Version 2 (Comparison)",
                    options=version_2_options,
                    index=0 if version_2_options else None,
                    key="cpu_compare_v2",
                )
                if version_2_options
                else None
            )

        if not version_2:
            st.warning("⚠️ Please select a second version to compare.")
            return

        # Only show platforms that exist in BOTH selected versions
        v1_accelerators = set(df[df["version"] == version_1]["accelerator"].unique())
        v2_accelerators = set(df[df["version"] == version_2]["accelerator"].unique())
        common_accelerators = sorted(v1_accelerators.intersection(v2_accelerators))

        if not common_accelerators:
            st.warning(
                f"⚠️ No common platforms found between {version_1} and {version_2}."
            )
            return

        with col3:
            selected_accelerator = st.selectbox(
                "Platform (CPU)",
                options=common_accelerators,
                index=0,
                key="cpu_compare_accelerator",
            )

        # Only show profiles that exist for the selected platform in BOTH versions
        v1_profiles = set(
            df[
                (df["version"] == version_1)
                & (df["accelerator"] == selected_accelerator)
            ]["profile"].unique()
        )
        v2_profiles = set(
            df[
                (df["version"] == version_2)
                & (df["accelerator"] == selected_accelerator)
            ]["profile"].unique()
        )
        common_profiles = sorted(v1_profiles.intersection(v2_profiles))

        if not common_profiles:
            st.warning(
                f"⚠️ No common workloads found for {selected_accelerator} "
                f"between {version_1} and {version_2}."
            )
            return

        with col4:
            selected_profile = st.selectbox(
                "Workload (ISL/OSL)",
                options=common_profiles,
                index=0,
                key="cpu_compare_profile",
            )

        # Filter data for each version
        mask_v1 = (
            (df["version"] == version_1)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        )
        mask_v2 = (
            (df["version"] == version_2)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        )
        df_v1 = df[mask_v1].copy()
        df_v2 = df[mask_v2].copy()

        if df_v1.empty or df_v2.empty:
            st.warning(
                "⚠️ No data available for the selected combination. "
                "Try different platform or workload settings."
            )
            return

        # Find common models
        v1_models = set(df_v1["model"].unique())
        v2_models = set(df_v2["model"].unique())
        common_models = sorted(v1_models.intersection(v2_models))

        if not common_models:
            st.warning(
                f"⚠️ No common models found between {version_1} and {version_2} "
                f"for {selected_accelerator} with workload {selected_profile}."
            )
            return

        # Metrics to compare
        metrics = [
            ("output_tok/sec", "Output Tok/s (Peak)", True),
            ("ttft_p95", "TTFT P95 (ms)", False),
            ("tpot_p95", "TPOT P95 (ms)", False),
            ("itl_p95", "ITL P95 (ms)", False),
            ("request_latency_median", "Request Latency Median (s)", False),
        ]

        results = []
        for model in common_models:
            model_v1 = df_v1[df_v1["model"] == model]
            model_v2 = df_v2[df_v2["model"] == model]

            model_short = model.split("/")[-1] if "/" in model else model
            row = {"Model": model_short}

            for metric_col, metric_label, higher_is_better in metrics:
                if metric_col not in model_v1.columns:
                    continue
                result = _compare_metric(
                    model_v1, model_v2, metric_col, higher_is_better
                )
                pct_diff, v1_better, _, _, is_similar = result

                if pct_diff is not None:
                    if is_similar:
                        row[metric_label] = f"🟡 ~{abs(pct_diff):.1f}% (similar)"
                    elif v1_better:
                        row[metric_label] = (
                            f"🟢 +{abs(pct_diff):.1f}% ({version_1} better)"
                        )
                    else:
                        row[metric_label] = (
                            f"🔴 -{abs(pct_diff):.1f}% ({version_2} better)"
                        )
                else:
                    row[metric_label] = "N/A"

            results.append(row)

        if results:
            st.markdown(
                f"**Comparison: {version_1} vs {version_2}** on "
                f"**{selected_accelerator}** with workload **{selected_profile}**"
            )
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            st.caption(
                f"Legend: 🟢 {version_1} performs better than {version_2} | "
                f"🔴 {version_1} performs worse than {version_2} | "
                f"🟡 Similar Performance (< 5% difference)"
            )


# ---------------------------------------------------------------------------
# Section: Runtime Server Configs
# ---------------------------------------------------------------------------


def render_runtime_configs_section(filtered_df, use_expander=True):
    """Render runtime server configs section for vLLM CPU dashboard."""
    if use_expander:
        ctx = st.expander("⚙️ Runtime Server Configs", expanded=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("⚙️ Runtime Server Configs")

        if "runtime_args" not in filtered_df.columns:
            st.info("No runtime configuration data available.")
            return

        st.markdown("**Runtime configurations for your current filter selections:**")
        st.info(
            "📊 **Column Legend**: Shows the server runtime arguments used for each "
            "Platform + Model + Version + Core configuration that matches your current filters."
        )

        subset_cols = ["model", "accelerator", "version", "profile"]
        extra_cols = []
        if "core_count" in filtered_df.columns:
            subset_cols.append("core_count")
            extra_cols.append("core_count")
        if "omp_num_threads" in filtered_df.columns:
            subset_cols.append("omp_num_threads")
            extra_cols.append("omp_num_threads")
        if "cpuset_cpus" in filtered_df.columns:
            subset_cols.append("cpuset_cpus")
            extra_cols.append("cpuset_cpus")

        unique_configs = filtered_df.drop_duplicates(subset=subset_cols)

        if unique_configs.empty:
            st.info("No configurations found for current filters.")
            return

        display_cols = ["model", "accelerator", "version"]
        if "runtime_args" in unique_configs.columns:
            display_cols.append("runtime_args")
        display_cols.append("profile")
        display_cols += extra_cols
        if "image_tag" in unique_configs.columns:
            display_cols.append("image_tag")
        if "guidellm_version" in unique_configs.columns:
            display_cols.append("guidellm_version")

        configs_df = unique_configs[display_cols].copy()
        configs_df.reset_index(drop=True, inplace=True)
        configs_df.insert(0, "Config #", range(1, len(configs_df) + 1))

        rename_map = {
            "model": "Model",
            "accelerator": "Platform",
            "version": "Version",
            "profile": "Workload (ISL/OSL)",
            "core_count": "Core Count",
            "omp_num_threads": "OMP Threads",
            "cpuset_cpus": "CPU Set",
            "runtime_args": "Runtime Arguments",
            "image_tag": "Image Tag",
            "guidellm_version": "GuideLLM Version",
        }
        configs_df.rename(columns=rename_map, inplace=True)

        # Format numeric columns
        for col in ["Core Count", "OMP Threads"]:
            if col in configs_df.columns:
                configs_df[col] = configs_df[col].apply(
                    lambda v: "N/A" if pd.isna(v) else str(int(v))
                )

        row_height = 40
        header_height = 40
        padding = 20
        dynamic_height = min(
            max(len(configs_df) * row_height + header_height + padding, 150),
            600,
        )

        st.dataframe(
            configs_df,
            use_container_width=True,
            hide_index=True,
            height=dynamic_height,
        )


# ---------------------------------------------------------------------------
# Section: Filtered Data
# ---------------------------------------------------------------------------


def render_filtered_data_section(filtered_df, use_expander=True):
    """Render filtered data table section for vLLM CPU dashboard."""
    if use_expander:
        ctx = st.expander("📄 Filtered Data", expanded=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("📄 Filtered Data")

        if filtered_df.empty:
            st.warning("No data available for selected filters.")
            return

        st.info(
            "💡 **Tips**: Hover over column headers to see detailed descriptions of each field."
        )
        display_df = filtered_df.copy()
        display_df.reset_index(drop=True, inplace=True)
        display_df.insert(0, "Row #", range(1, len(display_df) + 1))

        column_config = {
            "Row #": st.column_config.NumberColumn("Row #", pinned=True),
            "run": st.column_config.TextColumn(
                "run",
                help="Unique identifier combining platform, model, and configuration",
                pinned=True,
            ),
            "accelerator": st.column_config.TextColumn(
                "accelerator",
                help="CPU platform type (e.g., EPYC-NO-SMT, Xeon-NO-SMT)",
            ),
            "model": st.column_config.TextColumn(
                "model", help="Full path/name of the LLM model being benchmarked"
            ),
            "version": st.column_config.TextColumn(
                "version",
                help="vLLM version (e.g., vLLM-0.18.0+rhaiv.5)",
                pinned=True,
            ),
            "prompt toks": st.column_config.NumberColumn(
                "prompt toks",
                help="Target number of prompt (input) tokens in the benchmark",
            ),
            "output toks": st.column_config.NumberColumn(
                "output toks",
                help="Target number of output tokens to generate in the benchmark",
            ),
            "intended concurrency": st.column_config.NumberColumn(
                "intended concurrency",
                help="Target concurrency level - number of parallel requests sent to the server",
                pinned=True,
            ),
            "measured concurrency": st.column_config.NumberColumn(
                "measured concurrency",
                help="Actual concurrency level achieved during the benchmark run",
                format="%.2f",
            ),
            "output_tok/sec": st.column_config.NumberColumn(
                "output_tok/sec",
                help="Output tokens per second - key throughput metric (higher is better)",
                format="%.2f",
            ),
            "total_tok/sec": st.column_config.NumberColumn(
                "total_tok/sec",
                help="Total tokens per second (input + output)",
                format="%.2f",
            ),
            "ttft_median": st.column_config.NumberColumn(
                "ttft_median",
                help="Time to First Token - Median (ms)",
                format="%.2f",
            ),
            "ttft_p95": st.column_config.NumberColumn(
                "ttft_p95",
                help="Time to First Token - 95th percentile (ms)",
                format="%.2f",
            ),
            "tpot_median": st.column_config.NumberColumn(
                "tpot_median",
                help="Time Per Output Token - Median (ms)",
                format="%.2f",
            ),
            "tpot_p95": st.column_config.NumberColumn(
                "tpot_p95",
                help="Time Per Output Token - 95th percentile (ms)",
                format="%.2f",
            ),
            "itl_median": st.column_config.NumberColumn(
                "itl_median",
                help="Inter-Token Latency - Median (ms)",
                format="%.2f",
            ),
            "itl_p95": st.column_config.NumberColumn(
                "itl_p95",
                help="Inter-Token Latency - 95th percentile (ms)",
                format="%.2f",
            ),
            "core_count": st.column_config.NumberColumn(
                "core_count",
                help="Number of CPU cores used for inference",
            ),
            "cpuset_cpus": st.column_config.TextColumn(
                "cpuset_cpus",
                help="CPU core pinning configuration",
            ),
            "omp_num_threads": st.column_config.NumberColumn(
                "omp_num_threads",
                help="OpenMP thread count configuration",
            ),
        }

        row_height = 40
        header_height = 40
        padding = 20
        dynamic_height = min(
            max(len(display_df) * row_height + header_height + padding, 200),
            800,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=dynamic_height,
            column_config=column_config,
        )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def render_cpu_dashboard(cpu_csv_path: str):
    """Render the vLLM CPU benchmark dashboard.

    Args:
        cpu_csv_path: Path to the CPU dashboard CSV data file
    """
    # Load data
    df = load_cpu_data(cpu_csv_path)

    if df is None or df.empty:
        st.error("No data available. Please check the data file.")
        return

    # Section navigation
    section_list = [
        "📈 Performance Plots",
        "⚖️ Compare Versions",
        "⚙️ Runtime Server Configs",
        "📄 Filtered Data",
    ]

    SECTION_GROUPS = [
        (
            "Performance Analysis",
            [
                "📈 Performance Plots",
                "⚖️ Compare Versions",
            ],
        ),
        (
            "Tools",
            [
                "⚙️ Runtime Server Configs",
                "📄 Filtered Data",
            ],
        ),
    ]

    CPU_SECTION_SLUG_MAP = {
        "📈 Performance Plots": "performance_plots",
        "⚖️ Compare Versions": "compare_versions",
        "⚙️ Runtime Server Configs": "runtime_configs",
        "📄 Filtered Data": "filtered_data",
    }
    CPU_SLUG_TO_SECTION = {v: k for k, v in CPU_SECTION_SLUG_MAP.items()}

    # Restore section from URL on first load
    if "cpu_url_loaded" not in st.session_state:
        st.session_state.cpu_url_loaded = True
        if "section" in st.query_params:
            slug = st.query_params["section"]
            if slug in CPU_SLUG_TO_SECTION:
                st.session_state.cpu_active_section = CPU_SLUG_TO_SECTION[slug]

    current_section = st.session_state.get("cpu_active_section", section_list[0])
    if current_section not in section_list:
        current_section = section_list[0]
    st.session_state.cpu_active_section = current_section

    CPU_SECTIONS_WITHOUT_GLOBAL_FILTERS = {
        "⚖️ Compare Versions",
    }
    _show_global_filters = current_section not in CPU_SECTIONS_WITHOUT_GLOBAL_FILTERS

    if _show_global_filters:
        filtered_df, filter_selections = render_cpu_filters(df)
        st.markdown("---")
    else:
        filtered_df = df.copy()
        filter_selections = {}

    # Sidebar navigation
    with st.sidebar:
        for group_name, group_sections in SECTION_GROUPS:
            visible = [s for s in group_sections if s in section_list]
            if not visible:
                continue
            st.markdown(
                f'<p class="nav-group-header">{group_name}</p>',
                unsafe_allow_html=True,
            )
            for section_name in visible:
                is_active = section_name == current_section
                btn_type: Literal["primary", "secondary"] = (
                    "primary" if is_active else "secondary"
                )
                if st.button(
                    section_name,
                    key=f"cpu_nav_{section_name}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.cpu_active_section = section_name
                    st.rerun()

    # Render selected section
    if not filtered_df.empty:

        def _render_selected_section(sel):
            if sel == "📈 Performance Plots":
                render_performance_plots_section(filtered_df, use_expander=False)
            elif sel == "⚖️ Compare Versions":
                render_compare_versions_section(df, use_expander=False)
            elif sel == "⚙️ Runtime Server Configs":
                render_runtime_configs_section(filtered_df, use_expander=False)
            elif sel == "📄 Filtered Data":
                render_filtered_data_section(filtered_df, use_expander=False)

        _render_selected_section(current_section)

    else:
        st.warning(
            "❌ **No data matches your current filter settings.** Please adjust the filters."
        )

    # Sync section to URL
    with contextlib.suppress(Exception):
        desired_params: dict[str, str] = {}
        if "view" in st.query_params:
            desired_params["view"] = st.query_params["view"]

        section_slug = CPU_SECTION_SLUG_MAP.get(current_section)
        if section_slug:
            desired_params["section"] = section_slug

        if _show_global_filters:
            sel_acc = filter_selections.get("accelerators", [])
            sel_mod = filter_selections.get("model", "")
            sel_ver = filter_selections.get("versions", [])

            if sel_acc:
                desired_params["accelerators"] = ",".join(sel_acc)
            if sel_mod:
                desired_params["model"] = sel_mod
            if sel_ver:
                desired_params["versions"] = ",".join(sel_ver)

        st.query_params.update(desired_params)

    # Click anywhere on main area to collapse sidebar + scroll to top + hamburger icon
    import streamlit.components.v1 as _stc

    _stc.html(
        f"""
<script>
(function() {{
    var doc = parent.document;

    // --- Scroll to top on section change ---
    var currentSection = "{current_section}";
    if (doc._lastSection && doc._lastSection !== currentSection) {{
        var main = doc.querySelector('[data-testid="stMain"]');
        if (main) main.scrollTop = 0;
        var sc = doc.querySelector('.main');
        if (sc) sc.scrollTop = 0;
        parent.window.scrollTo(0, 0);
    }}
    doc._lastSection = currentSection;

    // --- Click-to-close sidebar ---
    if (doc._sidebarClickClose) {{
        doc.removeEventListener('click', doc._sidebarClickClose);
    }}
    if (doc._clickCloseTimeout) {{
        clearTimeout(doc._clickCloseTimeout);
    }}

    doc._sidebarClickClose = function(e) {{
        var sb = doc.querySelector('[data-testid="stSidebar"]');
        if (!sb || sb.getAttribute('aria-expanded') !== 'true') return;
        var main = doc.querySelector('[data-testid="stMain"]');
        if (!main || !main.contains(e.target)) return;
        setTimeout(function() {{
            var sb2 = doc.querySelector('[data-testid="stSidebar"]');
            if (!sb2 || sb2.getAttribute('aria-expanded') !== 'true') return;
            var closeBtn = sb2.querySelector('[data-testid="stSidebarHeader"] button')
                        || sb2.querySelector('button[kind="headerNoPadding"]')
                        || sb2.querySelector('button[kind="header"]');
            if (closeBtn) closeBtn.click();
        }}, 0);
    }};

    var clickDelay = doc._clickCloseInitialized ? 0 : 1500;
    doc._clickCloseInitialized = true;
    doc._clickCloseTimeout = setTimeout(function() {{
        doc.addEventListener('click', doc._sidebarClickClose);
    }}, clickDelay);

    // --- Hamburger icon replacement ---
    if (doc._hamburgerInterval) clearInterval(doc._hamburgerInterval);

    function scan() {{
        var sb = doc.querySelector('[data-testid="stSidebar"]');
        if (sb) {{
            var hdr = sb.querySelector('[data-testid="stSidebarHeader"] button')
                   || sb.querySelector('button[kind="headerNoPadding"]')
                   || sb.querySelector('button[kind="header"]');
            if (hdr) {{
                hdr.classList.add('hamburger-btn');
                hdr.setAttribute('data-tooltip', 'Collapse sidebar');
            }}
        }}
        var sidebarOpen = sb && sb.getAttribute('aria-expanded') === 'true';
        if (!sidebarOpen) {{
            var header = doc.querySelector('[data-testid="stHeader"]');
            if (header) {{
                var firstBtn = header.querySelector('button');
                if (firstBtn) {{
                    firstBtn.classList.add('hamburger-btn');
                    firstBtn.setAttribute('data-tooltip', 'Expand sidebar');
                    if (!firstBtn.classList.contains('hamburger-pulse')) {{
                        firstBtn.classList.add('hamburger-pulse');
                    }}
                }}
            }}
        }}
        // Remove pulse when sidebar is open
        if (sidebarOpen && sb) {{
            var hdrBtn = sb.querySelector('[data-testid="stSidebarHeader"] button')
                      || sb.querySelector('button[kind="headerNoPadding"]')
                      || sb.querySelector('button[kind="header"]');
            if (hdrBtn) hdrBtn.classList.remove('hamburger-pulse');
        }}
    }}

    scan();
    doc._hamburgerInterval = setInterval(scan, 500);
}})();
</script>
""",
        height=0,
    )
