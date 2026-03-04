"""LLM Inference Performance Dashboard.

A comprehensive dashboard for analyzing and comparing LLM inference performance
across different models, versions, and hardware configurations.
"""

import base64
import contextlib
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import styling functions
from dashboard_styles import (
    apply_theme_css,
    get_app_css,
    initialize_session_state,
    initialize_streamlit_config,
)

# Import MLPerf dashboard
try:
    from mlperf_datacenter import render_mlperf_dashboard

    MLPERF_AVAILABLE = True
except ImportError:
    MLPERF_AVAILABLE = False
    print("Warning: mlperf_datacenter module not found. MLPerf view will be disabled.")

# Import LLM-D dashboard
try:
    from llmd_dashboard import render_llmd_dashboard

    LLMD_AVAILABLE = True
except ImportError:
    LLMD_AVAILABLE = False
    print("Warning: llmd_dashboard module not found. LLM-D view will be disabled.")

# Configure logging to stdout for container logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# S3 Configuration from environment variables
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY = os.environ.get("S3_KEY", "consolidated_dashboard.csv")
S3_KEY_LLMD = os.environ.get("S3_KEY_LLMD", "llmd-dashboard.csv")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    """Read a CSV file from S3 bucket.

    Args:
        bucket: S3 bucket name.
        key: S3 object key (path to file in bucket).
        region: AWS region name.

    Returns:
        DataFrame with the CSV data.

    Raises:
        Exception: If unable to read from S3.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        # Check if credentials are provided
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            # Use provided credentials
            s3_client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
        else:
            # Try anonymous access for public buckets, or use IAM role if on AWS
            try:
                # First try with default credentials (IAM role)
                s3_client = boto3.client("s3", region_name=region)
                # Test if we can access the object
                s3_client.head_object(Bucket=bucket, Key=key)
            except Exception:
                # Fall back to anonymous access for public buckets
                s3_client = boto3.client(
                    "s3",
                    region_name=region,
                    config=Config(signature_version=UNSIGNED),
                )

        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        return pd.read_csv(io.StringIO(csv_content))

    except ImportError:
        raise ImportError(
            "boto3 is required for S3 access. Install with: pip install boto3"
        )
    except Exception as e:
        raise Exception(
            f"Failed to read from S3 bucket '{bucket}', key '{key}': {str(e)}"
        )


def get_csv_source() -> str:
    """Determine the CSV data source (S3 or local).

    Returns:
        'S3' if S3_BUCKET is configured, otherwise 'local'.
    """
    return "S3" if S3_BUCKET else "local"


def get_logo_base64():
    """Load and encode the Red Hat logo as base64.

    Returns:
        Base64 encoded string of the logo image, or None if file not found.
    """
    logo_path = Path(__file__).parent / "assets" / "RedHat-logo.png"
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes max
def load_data(file_path, cache_key=None):
    """Load and preprocess performance data from CSV file or S3.

    If S3_BUCKET environment variable is set, data is loaded from S3.
    Otherwise, falls back to local file system.

    Args:
        file_path: Path to the CSV file to load (used as fallback or S3 key).
        cache_key: Optional cache key for cache invalidation.

    Returns:
        DataFrame with loaded and processed data, or None if error occurs.
    """
    try:
        # Try S3 first if configured
        if S3_BUCKET:
            try:
                df = read_csv_from_s3(S3_BUCKET, S3_KEY, S3_REGION)
                logger.info(
                    f"Successfully loaded data from S3: s3://{S3_BUCKET}/{S3_KEY}"
                )
            except Exception as s3_error:
                logger.warning(
                    f"S3 load failed ({s3_error}), falling back to local file"
                )
                df = pd.read_csv(file_path)
        else:
            logger.info(f"Loading data from local file: {file_path}")
            df = pd.read_csv(file_path)

        df["run"] = df["run"].str.strip()
        df["accelerator"] = df["accelerator"].str.strip()
        df["model"] = df["model"].str.strip()
        df["version"] = df["version"].str.strip()
        df["TP"] = pd.to_numeric(df["TP"], errors="coerce")
        return df
    except FileNotFoundError:
        st.error(
            f"Error: The data file was not found at '{file_path}'. Please make sure the file exists."
        )
        return None
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {str(e)}")
        return None


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
    elif prompt_toks == 8000 and output_toks == 1000:
        return "Profile D: Prefill Heavy (8k/1k)"
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


def create_kpi_card(title, value, subtitle="", format_func=None):
    """Create a styled KPI card."""
    formatted_value = format_func(value) if format_func else str(value)

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{formatted_value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """
    return card_html


def keep_expander_open(expander_key):
    """Helper function to keep an expander open after widget interaction."""
    st.session_state[expander_key] = True


@st.cache_data(ttl=600)
def load_rhaiis_dataset(dataset_name):
    """Load a pre-generated CSV summary for a RHAIIS benchmark dataset.

    Args:
        dataset_name: Friendly name of the dataset (e.g., 'DeepSeek-R1')

    Returns:
        DataFrame with at least 'input_length' column (and optionally
        'output_length'), or None if not available.
    """
    dataset_map = {
        "DeepSeek-R1": "datasets/summaries/deepseek-r1.csv",
        "GPT-OSS Perf Eval": "datasets/summaries/gpt-oss.csv",
        "ShareGPT Vicuna": "datasets/summaries/sharegpt-vicuna.csv",
    }

    if dataset_name not in dataset_map:
        return None

    csv_path = dataset_map[dataset_name]

    if not os.path.exists(csv_path):
        return None

    try:
        data = pd.read_csv(csv_path)
        if "input_length" not in data.columns:
            st.error(
                f"Dataset CSV must contain an 'input_length' column. Found: {list(data.columns)}"
            )
            return None
        return data
    except Exception as e:
        st.error(f"Error loading dataset summary: {e}")
        return None


def create_rhaiis_dataset_histograms(data):
    """Create histograms for input and output token lengths.

    Args:
        data: DataFrame containing at least 'input_length' column,
              and optionally 'output_length'.

    Returns:
        Tuple of (input histogram figure, output histogram figure or None)
    """
    if data is None or data.empty:
        return None, None

    input_col = None
    output_col = None

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

    if input_col is None:
        st.warning(
            f"Could not find input length column. Available columns: {list(data.columns)}"
        )
        return None, None

    # --- Input token histogram ---
    input_data = data[input_col].dropna()
    input_mean = input_data.mean()
    input_median = input_data.median()
    input_min = input_data.min()
    input_max = input_data.max()

    fig_input = go.Figure()
    fig_input.add_trace(
        go.Histogram(
            x=input_data,
            nbinsx=50,
            marker_color="#1f77b4",
            marker_line={"color": "#0d3d5c", "width": 1},
            name="Input Tokens",
        )
    )
    fig_input.add_vline(
        x=input_mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {input_mean:.2f}",
        annotation_position="top left",
    )
    fig_input.add_vline(
        x=input_median,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Median: {input_median:.2f}",
        annotation_position="top",
    )
    fig_input.add_vline(
        x=input_max,
        line_dash="dashdot",
        line_color="green",
        annotation_text=f"Max: {int(input_max)}",
        annotation_position="top right",
    )
    fig_input.update_layout(
        title=(
            f"Histogram of Input Token Length<br>"
            f"<sub>Mean: {input_mean:.2f}, Median: {input_median:.2f}, "
            f"Min: {int(input_min)}, Max: {int(input_max)}</sub>"
        ),
        xaxis_title="Input Token Length",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )

    # --- Output token histogram (if available) ---
    fig_output = None
    if output_col is not None:
        output_data = data[output_col].dropna()
        if not output_data.empty:
            output_mean = output_data.mean()
            output_median = output_data.median()
            output_min = output_data.min()
            output_max = output_data.max()

            fig_output = go.Figure()
            fig_output.add_trace(
                go.Histogram(
                    x=output_data,
                    nbinsx=50,
                    marker_color="#8B4513",
                    marker_line={"color": "#5c2a0a", "width": 1},
                    name="Output Tokens",
                )
            )
            fig_output.add_vline(
                x=output_mean,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Mean: {output_mean:.2f}",
                annotation_position="top left",
            )
            fig_output.add_vline(
                x=output_median,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Median: {output_median:.2f}",
                annotation_position="top",
            )
            fig_output.add_vline(
                x=output_max,
                line_dash="dashdot",
                line_color="green",
                annotation_text=f"Max: {int(output_max)}",
                annotation_position="top right",
            )
            fig_output.update_layout(
                title=(
                    f"Histogram of Output Token Length<br>"
                    f"<sub>Mean: {output_mean:.2f}, Median: {output_median:.2f}, "
                    f"Min: {int(output_min)}, Max: {int(output_max)}</sub>"
                ),
                xaxis_title="Output Token Length",
                yaxis_title="Frequency",
                showlegend=False,
                height=400,
            )

    return fig_input, fig_output


def render_dataset_representation_section(selected_profile):
    """Render the Dataset Representation section (visible only for Custom ISL/OSL).

    Shows token length distribution histograms for real benchmark datasets
    when the user has selected a 'Custom' ISL/OSL profile.

    Args:
        selected_profile: The currently selected ISL/OSL profile string.
    """
    if selected_profile != "Custom":
        return

    with st.expander("📈 Dataset Representation", expanded=False):
        st.markdown(
            "View token length distribution statistics for the evaluation dataset "
            "used with custom ISL/OSL configurations. These histograms show the "
            "distribution of input (prompt) and output (completion) token lengths "
            "in the dataset."
        )

        available_datasets = ["DeepSeek-R1", "GPT-OSS Perf Eval", "ShareGPT Vicuna"]
        selected_dataset = st.selectbox(
            "Select Dataset",
            available_datasets,
            key="rhaiis_dataset_selector",
        )

        if selected_dataset:
            with st.spinner(f"Loading dataset: {selected_dataset}..."):
                dataset = load_rhaiis_dataset(selected_dataset)

            if dataset is None:
                st.info(
                    f"Dataset not available for **{selected_dataset}**.\n\n"
                    "Please ensure the summary CSV files have been generated. "
                    "Run `python datasets/generate_summaries.py` from the project root."
                )
            else:
                has_output = "output_length" in dataset.columns
                sample_info = f"{len(dataset):,} samples"
                if has_output:
                    sample_info += " (input + output token lengths)"
                else:
                    sample_info += " (input token lengths only)"
                st.success(f"Loaded {sample_info} from the {selected_dataset} dataset")

                if selected_dataset == "ShareGPT Vicuna":
                    st.info(
                        "Note: Statistical outliers have been removed from this dataset "
                        "using the IQR method (values beyond Q3 + 1.5 × IQR) to improve "
                        "histogram readability."
                    )

                if not has_output:
                    st.info(
                        "Output token lengths are not available for this dataset. "
                        "Only input token length distribution is shown."
                    )

                fig_input, fig_output = create_rhaiis_dataset_histograms(dataset)

                if fig_input:
                    if fig_output:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_input, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_output, use_container_width=True)
                    else:
                        st.plotly_chart(fig_input, use_container_width=True)

                    # Detailed statistics expander
                    with st.expander("Detailed Statistics", expanded=False):
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
                        elif input_col:
                            input_stats = dataset[input_col].describe()
                            st.markdown("**Input Token Statistics:**")
                            st.dataframe(
                                input_stats.to_frame(name="Input Tokens"),
                                use_container_width=True,
                            )
                else:
                    st.error("Could not generate histograms from the dataset.")


def render_performance_plots_section(filtered_df):
    """📊 Performance Plots Section - Complete functionality from original."""
    if "performance_plots_expanded" not in st.session_state:
        st.session_state.performance_plots_expanded = False

    with st.expander(
        "📊 Performance Plots", expanded=st.session_state.performance_plots_expanded
    ):
        st.markdown(
            "💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view."
        )

        filtered_df["model_short"] = filtered_df["model"].apply(
            lambda x: x.split("/")[-1] if pd.notna(x) else "Unknown"
        )
        filtered_df["run_identifier"] = (
            filtered_df["accelerator"]
            + " | "
            + filtered_df["model"]
            + " | "
            + filtered_df["version"]
            + " | TP="
            + filtered_df["TP"].astype(str)
        )

        filtered_df_sorted = filtered_df.sort_values(
            ["model_short", "accelerator", "version", "TP"]
        ).copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis_options = {
                "Concurrency": "intended concurrency",
                "Throughput (Output Tok/s)": "output_tok/sec",
            }
            x_axis_label = st.selectbox(
                "Select X-Axis",
                options=list(x_axis_options.keys()),
                key="perf_plots_x_axis",
                on_change=keep_expander_open,
                args=("performance_plots_expanded",),
            )
            x_axis = x_axis_options[x_axis_label]

        with col2:
            y_axis_options = {
                "Throughput (Output tokens/second generated)": "output_tok/sec",
                "Efficiency (Output tokens/sec per TP unit)": "efficiency_ratio",
                "Inter-Token Latency P95 (Time between tokens)": "itl_p95",
                "Time to First Token P95 (Response start delay)": "ttft_p95_s",
                "Request Latency Median (Total request processing time)": "request_latency_median",
                "Request Latency Max (Maximum request processing time)": "request_latency_max",
                "Time Per Output Token P95 (Token generation time)": "tpot_p95",
                "Total Throughput (Total tokens/second processed)": "total_tok/sec",
                "Request Count (Successful completions)": "successful_requests",
                "Error Rate (% Failed requests)": "error_rate",
            }
            y_axis_label = st.selectbox(
                "Select Y-Axis",
                options=list(y_axis_options.keys()),
                key="perf_plots_y_axis",
                on_change=keep_expander_open,
                args=("performance_plots_expanded",),
            )
            y_axis = y_axis_options[y_axis_label]

        with col3:
            if x_axis == "intended concurrency":
                concurrency_values = sorted(
                    filtered_df_sorted["intended concurrency"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                if concurrency_values:
                    max_conc = st.selectbox(
                        "Show concurrency up to",
                        options=concurrency_values,
                        index=len(concurrency_values) - 1,
                        key="perf_plots_max_concurrency",
                        on_change=keep_expander_open,
                        args=("performance_plots_expanded",),
                    )
                    filtered_df_sorted = filtered_df_sorted[
                        filtered_df_sorted["intended concurrency"] <= max_conc
                    ]

        # Add units to y-axis label for certain metrics
        y_axis_display_label = y_axis_label
        if y_axis == "ttft_p95_s":
            y_axis_display_label = f"{y_axis_label} (s)"
        elif y_axis == "itl_p95" or y_axis == "tpot_p95":
            y_axis_display_label = f"{y_axis_label} (ms)"
        elif y_axis == "request_latency_median" or y_axis == "request_latency_max":
            y_axis_display_label = f"{y_axis_label} (s)"

        fig = px.line(
            filtered_df_sorted.sort_values(by=x_axis),
            x=x_axis,
            y=y_axis,
            color="run_identifier",
            markers=True,
            title=f"{x_axis_label} vs. {y_axis_label}",
            labels={
                x_axis: x_axis_label,
                y_axis: y_axis_display_label,
                "run_identifier": "Run",
            },
            template="plotly_white",
            category_orders={
                "run_identifier": filtered_df_sorted["run_identifier"].unique().tolist()
            },
        )
        fig.update_layout(
            legend_title_text="Run Details (Accelerator | Model | Version | TP)",
            legend={"font": {"size": 14}},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Right-align the legend caption
        caption_col1, caption_col2 = st.columns([3, 1])
        with caption_col2:
            st.caption("📜 **Tip**: Scroll within the legend box to see all runs")


def load_pareto_data(csv_file_path, preloaded_df=None):
    """Load benchmark results from CSV file or S3 for Pareto analysis.

    If a preloaded DataFrame is provided, uses it directly (avoiding a
    duplicate S3/disk read).  Otherwise, falls back to S3 or local file.

    Args:
        csv_file_path: Path to the CSV file to load (used as fallback).
        preloaded_df: Optional pre-loaded DataFrame to reuse.

    Returns:
        List of result dictionaries for Pareto tradeoff analysis.
    """
    try:
        if preloaded_df is not None:
            df = preloaded_df.copy()
        elif S3_BUCKET:
            try:
                df = read_csv_from_s3(S3_BUCKET, S3_KEY, S3_REGION)
                logger.info(f"Pareto data loaded from S3: s3://{S3_BUCKET}/{S3_KEY}")
            except Exception as s3_error:
                logger.warning(
                    f"S3 load failed ({s3_error}), falling back to local file"
                )
                df = pd.read_csv(csv_file_path)
        else:
            df = pd.read_csv(csv_file_path)

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        results = []

        for _, row in df.iterrows():
            # Map accelerator to hardware label
            hw = row.get("accelerator", "")

            # Calculate throughput per GPU
            tp = row.get("TP", 1)
            if pd.isna(tp) or tp == 0:
                tp = 1

            total_throughput = row.get("total_tok/sec", 0)
            tput_per_gpu = total_throughput / tp if tp > 0 else 0

            output_throughput = row.get("output_tok/sec", 0)
            output_tput_per_gpu = output_throughput / tp if tp > 0 else 0

            # Calculate interactivity from tpot_median (tokens per output token)
            # tpot is in milliseconds, interactivity is tok/s/user
            tpot_median = row.get("tpot_median", None)
            median_intvty = 0
            if pd.notna(tpot_median) and tpot_median > 0:
                # Convert ms to seconds and take reciprocal: 1000 / tpot_ms = tok/s/user
                median_intvty = 1000.0 / tpot_median

            version = str(row.get("version", ""))
            str(row.get("model", ""))

            # Get TTFT (Time to First Token) - convert ms to seconds
            ttft_p95 = row.get("ttft_p95", 0)
            ttft_p95_s = ttft_p95 / 1000.0 if pd.notna(ttft_p95) else 0

            # Get ISL (Input Sequence Length) and OSL (Output Sequence Length)
            prompt_toks = row.get("prompt toks", 0)
            output_toks = row.get("output toks", 0)
            isl_osl = (
                f"{int(prompt_toks)}/{int(output_toks)}"
                if pd.notna(prompt_toks) and pd.notna(output_toks)
                else "Unknown"
            )

            result = {
                "hw": hw,
                "tp": int(tp),
                "conc": row.get("intended concurrency", 0),
                "model": row.get("model", "Unknown"),
                "version": version,
                "tput_per_gpu": tput_per_gpu,
                "output_tput_per_gpu": output_tput_per_gpu,
                "median_e2el": row.get("request_latency_median", 0),
                "median_intvty": median_intvty,
                "output_tok_per_sec": row.get("output_tok/sec", 0),
                "ttft_p95_s": ttft_p95_s,
                "isl": int(prompt_toks) if pd.notna(prompt_toks) else 0,
                "osl": int(output_toks) if pd.notna(output_toks) else 0,
                "isl_osl": isl_osl,
            }

            results.append(result)

        return results

    except FileNotFoundError:
        st.error(f"CSV file not found: {csv_file_path}")
        return []
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        return []


def render_pareto_plots_section(preloaded_df=None):
    """🔄 Pareto Tradeoff Analysis Section - Interactive plots showing performance vs latency tradeoffs."""
    if "pareto_expanded" not in st.session_state:
        st.session_state.pareto_expanded = False

    with st.expander(
        "🔄 Pareto Tradeoff Analysis", expanded=st.session_state.pareto_expanded
    ):
        # Load data (reuses preloaded_df when available to avoid duplicate S3 fetch)
        results = load_pareto_data(
            "consolidated_dashboard.csv", preloaded_df=preloaded_df
        )

        if not results:
            st.warning(
                "⚠️ No results found in 'consolidated_dashboard.csv'. "
                "Please ensure the CSV file exists and contains valid data."
            )
            return

        # Model and Version filters
        st.markdown(
            """
            These Pareto curves help you understand the **performance vs. latency tradeoff** across different hardware
            and tensor parallelism configurations. Use them to identify optimal concurrency levels, compare accelerator
            efficiency, and find the best configuration for your workload requirements.
            """
        )
        filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

        with filter_col1:
            # Get unique models from results
            unique_models = sorted({r.get("model", "Unknown") for r in results})

            # Set default model
            default_model = "openai/gpt-oss-120b"
            if default_model not in unique_models and unique_models:
                default_model = unique_models[0]

            default_idx = (
                unique_models.index(default_model)
                if default_model in unique_models
                else 0
            )

            selected_model = st.selectbox(
                "Select Model",
                options=unique_models,
                index=default_idx,
                key="pareto_model_select",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
            )

        # Filter by selected model
        results = [r for r in results if r.get("model", "Unknown") == selected_model]
        if not results:
            st.warning(f"No results found for model: '{selected_model}'")
            return

        with filter_col2:
            # Get unique versions from filtered results
            unique_versions = sorted({r.get("version", "Unknown") for r in results})

            # Set default versions - prefer these if available
            preferred_versions = ["RHAIIS-3.2.3", "RHAIIS-3.2.5"]
            default_versions = [v for v in preferred_versions if v in unique_versions]
            if not default_versions and unique_versions:
                default_versions = [unique_versions[0]]

            selected_versions = st.multiselect(
                "Select Version(s)",
                options=unique_versions,
                default=default_versions,
                key="pareto_version_select",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
            )

        # Filter by selected versions
        if not selected_versions:
            st.warning("Please select at least one version")
            return
        results = [
            r for r in results if r.get("version", "Unknown") in selected_versions
        ]
        if not results:
            st.warning("No results found for selected versions")
            return

        with filter_col3:
            # Get unique ISL/OSL combinations from filtered results
            unique_isl_osl = sorted({r.get("isl_osl", "Unknown") for r in results})

            # Set default ISL/OSL
            default_isl_osl = "1000/1000"
            if default_isl_osl not in unique_isl_osl and unique_isl_osl:
                default_isl_osl = unique_isl_osl[0]

            default_idx = (
                unique_isl_osl.index(default_isl_osl)
                if default_isl_osl in unique_isl_osl
                else 0
            )

            selected_isl_osl = st.selectbox(
                "Select ISL/OSL",
                options=unique_isl_osl,
                index=default_idx,
                key="pareto_isl_osl_select",
                help="ISL = Input Sequence Length (prompt tokens), OSL = Output Sequence Length (output tokens)",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
            )

        # Filter by selected ISL/OSL
        results = [
            r for r in results if r.get("isl_osl", "Unknown") == selected_isl_osl
        ]
        if not results:
            st.warning(f"No results found for ISL/OSL: '{selected_isl_osl}'")
            return

        with filter_col4:
            # Get unique accelerators from filtered results
            unique_hw = sorted({r.get("hw", "unknown").upper() for r in results})
            selected_hw = st.selectbox(
                "Select Accelerator",
                options=["All Accelerators"] + unique_hw,
                key="pareto_hw_select",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
            )

        with filter_col5:
            # Throughput metric selector
            throughput_options = {
                "Total Tokens/sec/GPU": "tput_per_gpu",
                "Output Tokens/sec/GPU": "output_tput_per_gpu",
            }
            selected_throughput_label = st.selectbox(
                "Throughput Metric",
                options=list(throughput_options.keys()),
                key="pareto_throughput_metric",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
                help="Total = prompt + output tokens, Output = output tokens only",
            )
            selected_throughput_key = throughput_options[selected_throughput_label]

        # Filter by selected hardware
        if selected_hw != "All Accelerators":
            results = [r for r in results if r.get("hw", "").upper() == selected_hw]
            if not results:
                st.warning(f"No results found for accelerator: '{selected_hw}'")
                return

        # Get unique accelerators, TP sizes, and versions
        unique_hw = sorted({r.get("hw", "unknown") for r in results})
        unique_tps = sorted({r.get("tp", 1) for r in results})
        unique_versions_in_results = sorted(
            {r.get("version", "Unknown") for r in results}
        )

        # Create a comprehensive color palette
        color_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d2",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
            "#90EE90",
            "#008000",
            "#000000",
            "#FF0000",
            "#800080",
            "#FFA500",
            "#4285F4",
            "#00CED1",
            "#FF1493",
            "#32CD32",
        ]

        # Create unique color mapping for each version+accelerator+TP combination
        hw_tp_version_color_map = {}
        color_idx = 0
        for version in sorted(unique_versions_in_results):
            for hw in sorted(unique_hw):
                for tp in sorted(unique_tps):
                    hw_tp_version_key = f"{version}_{hw.lower()}_{tp}"
                    hw_tp_version_color_map[hw_tp_version_key] = color_palette[
                        color_idx % len(color_palette)
                    ]
                    color_idx += 1

        # Create tabs for different plot types with larger buttons
        st.markdown(
            """
            <style>
            div[data-testid="stTabs"] button[data-baseweb="tab"] {
                font-size: 1.2rem;
                padding: 12px 24px;
                font-weight: 600;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        tab1, tab2 = st.tabs(
            ["📊 Throughput vs. End-to-End Latency", "📈 Throughput vs. Interactivity"]
        )

        with tab1:
            st.markdown("### Token Throughput per GPU vs. End-to-end Latency")
            st.markdown(
                """
            💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view.
            """
            )

            # Use all filtered results (no precision filter)
            filtered_results = results

            if not filtered_results:
                st.warning("No results found")
            else:
                # Create the plot
                import plotly.graph_objects as go

                fig = go.Figure()

                # Group by version, then by accelerator, then by TP size
                for version in sorted(unique_versions_in_results):
                    for hw in sorted(unique_hw):
                        for tp_size in sorted(unique_tps):
                            # Filter results for this version, accelerator and TP combination
                            hw_tp_version_results = [
                                r
                                for r in filtered_results
                                if r.get("version", "Unknown") == version
                                and r.get("hw", "unknown").lower() == hw.lower()
                                and r.get("tp", 1) == tp_size
                            ]

                            if hw_tp_version_results:
                                # Sort by concurrency for proper line drawing
                                hw_tp_version_results_sorted = sorted(
                                    hw_tp_version_results,
                                    key=lambda x: x.get("conc", 0),
                                )

                                xs = [
                                    r.get("median_e2el", 0)
                                    for r in hw_tp_version_results_sorted
                                ]
                                ys = [
                                    r.get(selected_throughput_key, 0)
                                    for r in hw_tp_version_results_sorted
                                ]
                                models = [
                                    r.get("model", "Unknown")
                                    for r in hw_tp_version_results_sorted
                                ]
                                concs = [
                                    r.get("conc", "N/A")
                                    for r in hw_tp_version_results_sorted
                                ]
                                isl_osls = [
                                    r.get("isl_osl", "N/A")
                                    for r in hw_tp_version_results_sorted
                                ]

                                # Get unique color for this version+accelerator+TP combination
                                hw_tp_version_key = f"{version}_{hw.lower()}_{tp_size}"
                                color = hw_tp_version_color_map.get(
                                    hw_tp_version_key, "#999999"
                                )

                                # Determine metric label for hover text
                                metric_hover_label = (
                                    "Total Throughput"
                                    if selected_throughput_key == "tput_per_gpu"
                                    else "Output Throughput"
                                )

                                hover_text = [
                                    f"Version: {version}<br>"
                                    f"Accelerator: {hw.upper()}<br>"
                                    f"TP Size: {tp_size}<br>"
                                    f"Concurrent Requests: {conc} Users<br>"
                                    f"Latency: {x:.2f}s<br>"
                                    f"{metric_hover_label}: {y:.2f} tok/s/gpu"
                                    for conc, isl_osl, model, x, y in zip(
                                        concs, isl_osls, models, xs, ys
                                    )
                                ]

                                fig.add_trace(
                                    go.Scatter(
                                        x=xs,
                                        y=ys,
                                        mode="markers+lines",
                                        name=f"{version} | {hw.upper()} (TP={tp_size})",
                                        marker={
                                            "size": 10,
                                            "color": color,
                                            "line": {"width": 1, "color": "white"},
                                        },
                                        line={"color": color, "width": 2},
                                        hovertext=hover_text,
                                        hoverinfo="text",
                                        legendgroup=version,  # Group by version in legend
                                    )
                                )

                # Dynamic title and y-axis label based on selected metric
                if selected_throughput_key == "tput_per_gpu":
                    plot_title = "Note: Throughput is Total Tokens per second (prompt + output tokens combined)"
                    y_axis_label = "Total Token Throughput per GPU (tok/s/gpu)"
                else:
                    plot_title = "Note: Throughput is Output Tokens per second only"
                    y_axis_label = "Output Token Throughput per GPU (tok/s/gpu)"

                fig.update_layout(
                    title=plot_title,
                    xaxis_title="End-to-end Latency (s)",
                    yaxis_title=y_axis_label,
                    template="plotly_dark",
                    hovermode="closest",
                    showlegend=True,
                    legend={
                        "title": "Version | Accelerator (TP Size)",
                        "font": {"size": 12},
                    },
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Token Throughput per GPU vs. Interactivity")
            st.markdown(
                """
            💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view.
            """
            )

            # Use all filtered results (no precision filter)
            filtered_results = results

            if not filtered_results:
                st.warning("No results found")
            else:
                # Create the plot
                import plotly.graph_objects as go

                fig = go.Figure()

                # Group by version, then by accelerator, then by TP size
                for version in sorted(unique_versions_in_results):
                    for hw in sorted(unique_hw):
                        for tp_size in sorted(unique_tps):
                            # Filter results for this version, accelerator and TP combination
                            hw_tp_version_results = [
                                r
                                for r in filtered_results
                                if r.get("version", "Unknown") == version
                                and r.get("hw", "unknown").lower() == hw.lower()
                                and r.get("tp", 1) == tp_size
                            ]

                            if hw_tp_version_results:
                                # Sort by concurrency for proper line drawing
                                hw_tp_version_results_sorted = sorted(
                                    hw_tp_version_results,
                                    key=lambda x: x.get("conc", 0),
                                )

                                xs = [
                                    r.get("median_intvty", 0)
                                    for r in hw_tp_version_results_sorted
                                ]
                                ys = [
                                    r.get(selected_throughput_key, 0)
                                    for r in hw_tp_version_results_sorted
                                ]
                                models = [
                                    r.get("model", "Unknown")
                                    for r in hw_tp_version_results_sorted
                                ]
                                concs = [
                                    r.get("conc", "N/A")
                                    for r in hw_tp_version_results_sorted
                                ]
                                isl_osls = [
                                    r.get("isl_osl", "N/A")
                                    for r in hw_tp_version_results_sorted
                                ]

                                # Get unique color for this version+accelerator+TP combination
                                hw_tp_version_key = f"{version}_{hw.lower()}_{tp_size}"
                                color = hw_tp_version_color_map.get(
                                    hw_tp_version_key, "#999999"
                                )

                                # Determine metric label for hover text
                                metric_hover_label = (
                                    "Total Throughput"
                                    if selected_throughput_key == "tput_per_gpu"
                                    else "Output Throughput"
                                )

                                hover_text = [
                                    f"Version: {version}<br>"
                                    f"Accelerator: {hw.upper()}<br>"
                                    f"TP Size: {tp_size}<br>"
                                    f"Concurrent Requests: {conc} Users<br>"
                                    f"Interactivity: {x:.2f} tok/s/user<br>"
                                    f"{metric_hover_label}: {y:.2f} tok/s/gpu"
                                    for conc, isl_osl, model, x, y in zip(
                                        concs, isl_osls, models, xs, ys
                                    )
                                ]

                                fig.add_trace(
                                    go.Scatter(
                                        x=xs,
                                        y=ys,
                                        mode="markers+lines",
                                        name=f"{version} | {hw.upper()} (TP={tp_size})",
                                        marker={
                                            "size": 10,
                                            "color": color,
                                            "line": {"width": 1, "color": "white"},
                                        },
                                        line={"color": color, "width": 2},
                                        hovertext=hover_text,
                                        hoverinfo="text",
                                        legendgroup=version,  # Group by version in legend
                                    )
                                )

                # Dynamic title and y-axis label based on selected metric
                if selected_throughput_key == "tput_per_gpu":
                    plot_title = "Note: Throughput is Total Tokens per second (prompt + output tokens combined)"
                    y_axis_label = "Total Token Throughput per GPU (tok/s/gpu)"
                else:
                    plot_title = "Note: Throughput is Output Tokens per second only"
                    y_axis_label = "Output Token Throughput per GPU (tok/s/gpu)"

                fig.update_layout(
                    title=plot_title,
                    xaxis_title="Interactivity (tok/s/user)",
                    yaxis_title=y_axis_label,
                    template="plotly_dark",
                    hovermode="closest",
                    showlegend=True,
                    legend={
                        "title": "Version | Accelerator (TP Size)",
                        "font": {"size": 12},
                    },
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        with st.expander("📋 Summary Statistics"):
            df_results = pd.DataFrame(results)

            if not df_results.empty:
                # Display key columns
                display_cols = [
                    "hw",
                    "model",
                    "version",
                    "isl",
                    "osl",
                    "tp",
                    "conc",
                    "tput_per_gpu",
                    "output_tput_per_gpu",
                    "median_e2el",
                    "median_intvty",
                ]
                available_cols = [
                    col for col in display_cols if col in df_results.columns
                ]

                # Format numeric columns for better readability
                df_display = df_results[available_cols].copy()
                if "tput_per_gpu" in df_display.columns:
                    df_display["tput_per_gpu"] = df_display["tput_per_gpu"].round(2)
                if "output_tput_per_gpu" in df_display.columns:
                    df_display["output_tput_per_gpu"] = df_display[
                        "output_tput_per_gpu"
                    ].round(2)
                if "median_e2el" in df_display.columns:
                    df_display["median_e2el"] = df_display["median_e2el"].round(3)
                if "median_intvty" in df_display.columns:
                    df_display["median_intvty"] = df_display["median_intvty"].round(2)

                # Rename columns for better readability
                df_display = df_display.rename(
                    columns={
                        "hw": "Accelerator",
                        "isl": "ISL",
                        "osl": "OSL",
                        "tp": "TP",
                        "conc": "Concurrency",
                    }
                )

                # Sort by the selected throughput metric
                sort_col = (
                    "tput_per_gpu"
                    if selected_throughput_key == "tput_per_gpu"
                    else "output_tput_per_gpu"
                )

                st.dataframe(
                    df_display.sort_values(by=sort_col, ascending=False).reset_index(
                        drop=True
                    ),
                    use_container_width=True,
                    column_config={
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator", help="Hardware/Accelerator type"
                        ),
                        "model": st.column_config.TextColumn(
                            "model", help="Model name"
                        ),
                        "version": st.column_config.TextColumn(
                            "version", help="Software version"
                        ),
                        "ISL": st.column_config.NumberColumn(
                            "ISL", help="Input Sequence Length (prompt tokens)"
                        ),
                        "OSL": st.column_config.NumberColumn(
                            "OSL", help="Output Sequence Length (output tokens)"
                        ),
                        "TP": st.column_config.NumberColumn(
                            "TP", help="Tensor Parallelism size"
                        ),
                        "Concurrency": st.column_config.NumberColumn(
                            "Concurrency", help="Number of concurrent requests"
                        ),
                        "tput_per_gpu": st.column_config.NumberColumn(
                            "tput_per_gpu",
                            help="Total Throughput per GPU: Total tokens/second (prompt + output) divided by TP size. Higher is better.",
                        ),
                        "output_tput_per_gpu": st.column_config.NumberColumn(
                            "output_tput_per_gpu",
                            help="Output Throughput per GPU: Output tokens/second divided by TP size. Higher is better.",
                        ),
                        "median_e2el": st.column_config.NumberColumn(
                            "median_e2el",
                            help="Median End-to-End Latency: Time from request start to completion. From CSV column 'request_latency_median'. Lower is better.",
                        ),
                        "median_intvty": st.column_config.NumberColumn(
                            "median_intvty",
                            help="Median Interactivity: Output tokens per second per user. Calculated as (1000 ÷ tpot_median). Higher means faster token generation.",
                        ),
                    },
                )


def render_custom_pareto_tradeoff_section(filtered_df):
    """🔄 Pareto Tradeoff Graphs — multi-model view for Custom ISL/OSL profiles.

    Plots all models present in *filtered_df* on Pareto-style throughput-vs-latency
    and throughput-vs-interactivity charts for a user-selected version.

    Args:
        filtered_df: DataFrame already filtered by the sidebar (accelerator, models,
                     Custom ISL/OSL profile, TP).
    """
    if "custom_pareto_expanded" not in st.session_state:
        st.session_state.custom_pareto_expanded = False

    with st.expander(
        "🔄 Pareto Tradeoff Graphs",
        expanded=st.session_state.custom_pareto_expanded,
    ):
        if filtered_df.empty:
            st.warning("No data available for the current filter selection.")
            return

        st.markdown(
            "These Pareto curves compare **all selected models** on the same chart "
            "for a chosen version, showing the **throughput vs. latency / interactivity "
            "tradeoff** across hardware and TP configurations."
        )

        filter_c1, filter_c2, filter_c3 = st.columns(3)

        with filter_c1:
            unique_versions = sorted(filtered_df["version"].dropna().unique())
            if not unique_versions:
                st.warning("No versions available.")
                return
            selected_version = st.selectbox(
                "Select Version",
                options=unique_versions,
                key="custom_pareto_version",
                on_change=keep_expander_open,
                args=("custom_pareto_expanded",),
            )

        with filter_c2:
            unique_accel = sorted(filtered_df["accelerator"].dropna().unique())
            selected_accel = st.selectbox(
                "Select Accelerator",
                options=["All Accelerators"] + list(unique_accel),
                key="custom_pareto_accel",
                on_change=keep_expander_open,
                args=("custom_pareto_expanded",),
            )

        with filter_c3:
            throughput_options = {
                "Total Tokens/sec/GPU": "total",
                "Output Tokens/sec/GPU": "output",
            }
            selected_tput_label = st.selectbox(
                "Throughput Metric",
                options=list(throughput_options.keys()),
                key="custom_pareto_tput",
                on_change=keep_expander_open,
                args=("custom_pareto_expanded",),
                help="Total = prompt + output tokens, Output = output tokens only",
            )
            tput_mode = throughput_options[selected_tput_label]

        vdf = filtered_df[filtered_df["version"] == selected_version].copy()
        if selected_accel != "All Accelerators":
            vdf = vdf[vdf["accelerator"] == selected_accel]

        if vdf.empty:
            st.warning("No data for the selected version / accelerator combination.")
            return

        vdf["tp_safe"] = vdf["TP"].fillna(1).replace(0, 1).astype(int)
        if tput_mode == "total":
            vdf["tput_per_gpu"] = vdf["total_tok/sec"] / vdf["tp_safe"]
            y_col, y_label = (
                "tput_per_gpu",
                "Total Token Throughput per GPU (tok/s/gpu)",
            )
            metric_hover = "Total Throughput"
        else:
            vdf["tput_per_gpu"] = vdf["output_tok/sec"] / vdf["tp_safe"]
            y_col, y_label = (
                "tput_per_gpu",
                "Output Token Throughput per GPU (tok/s/gpu)",
            )
            metric_hover = "Output Throughput"

        vdf["median_intvty"] = vdf["tpot_median"].apply(
            lambda t: 1000.0 / t if pd.notna(t) and t > 0 else 0
        )

        color_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d2",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
            "#90EE90",
            "#008000",
            "#000000",
            "#FF0000",
            "#800080",
            "#FFA500",
            "#4285F4",
            "#00CED1",
            "#FF1493",
            "#32CD32",
        ]

        groups = (
            vdf.groupby(["model", "accelerator", "tp_safe"], sort=True)
            .size()
            .reset_index()
            .drop(columns=0)
        )

        color_map = {}
        for idx, row in groups.iterrows():
            key = (row["model"], row["accelerator"], row["tp_safe"])
            color_map[key] = color_palette[idx % len(color_palette)]

        st.markdown(
            "<style>div[data-testid='stTabs'] button[data-baseweb='tab'] "
            "{font-size:1.2rem;padding:12px 24px;font-weight:600;}</style>",
            unsafe_allow_html=True,
        )
        tab1, tab2 = st.tabs(
            ["📊 Throughput vs. End-to-End Latency", "📈 Throughput vs. Interactivity"]
        )

        def _build_traces(fig, x_col, x_hover_label, x_fmt):
            for (model, accel, tp), color in color_map.items():
                subset = vdf[
                    (vdf["model"] == model)
                    & (vdf["accelerator"] == accel)
                    & (vdf["tp_safe"] == tp)
                ].sort_values("intended concurrency")

                if subset.empty:
                    continue

                model_short = model.split("/")[-1] if "/" in model else model
                trace_name = f"{model_short} | {accel.upper()} (TP={tp})"

                hover_text = [
                    f"Model: {model_short}<br>"
                    f"Accelerator: {accel.upper()}<br>"
                    f"TP Size: {tp}<br>"
                    f"Concurrent Requests: {int(r['intended concurrency'])} Users<br>"
                    f"{x_hover_label}: {r[x_col]:{x_fmt}}<br>"
                    f"{metric_hover}: {r[y_col]:.2f} tok/s/gpu"
                    for _, r in subset.iterrows()
                ]

                fig.add_trace(
                    go.Scatter(
                        x=subset[x_col].tolist(),
                        y=subset[y_col].tolist(),
                        mode="markers+lines",
                        name=trace_name,
                        marker={
                            "size": 10,
                            "color": color,
                            "line": {"width": 1, "color": "white"},
                        },
                        line={"color": color, "width": 2},
                        hovertext=hover_text,
                        hoverinfo="text",
                        legendgroup=model_short,
                    )
                )

        import plotly.graph_objects as go

        with tab1:
            st.markdown("### Token Throughput per GPU vs. End-to-end Latency")
            fig1 = go.Figure()
            _build_traces(fig1, "request_latency_median", "Latency", ".2f")
            fig1.update_layout(
                title="Note: Throughput is "
                + (
                    "Total Tokens per second"
                    if tput_mode == "total"
                    else "Output Tokens per second only"
                ),
                xaxis_title="End-to-end Latency (s)",
                yaxis_title=y_label,
                template="plotly_dark",
                hovermode="closest",
                showlegend=True,
                legend={"title": "Model | Accelerator (TP)", "font": {"size": 12}},
                height=600,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            st.markdown("### Token Throughput per GPU vs. Interactivity")
            fig2 = go.Figure()
            _build_traces(fig2, "median_intvty", "Interactivity", ".2f")
            fig2.update_layout(
                title="Note: Throughput is "
                + (
                    "Total Tokens per second"
                    if tput_mode == "total"
                    else "Output Tokens per second only"
                ),
                xaxis_title="Interactivity (tok/s/user)",
                yaxis_title=y_label,
                template="plotly_dark",
                hovermode="closest",
                showlegend=True,
                legend={"title": "Model | Accelerator (TP)", "font": {"size": 12}},
                height=600,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Summary Statistics"):
            display_df = vdf[
                [
                    "accelerator",
                    "model",
                    "version",
                    "TP",
                    "intended concurrency",
                    "tput_per_gpu",
                    "request_latency_median",
                    "median_intvty",
                ]
            ].copy()
            display_df = display_df.rename(
                columns={
                    "accelerator": "Accelerator",
                    "model": "Model",
                    "version": "Version",
                    "intended concurrency": "Concurrency",
                    "tput_per_gpu": "Throughput/GPU",
                    "request_latency_median": "E2E Latency (s)",
                    "median_intvty": "Interactivity",
                }
            )
            display_df["Throughput/GPU"] = display_df["Throughput/GPU"].round(2)
            display_df["E2E Latency (s)"] = display_df["E2E Latency (s)"].round(3)
            display_df["Interactivity"] = display_df["Interactivity"].round(2)
            st.dataframe(
                display_df.sort_values("Throughput/GPU", ascending=False).reset_index(
                    drop=True
                ),
                use_container_width=True,
            )


def render_performance_trends_section(df: pd.DataFrame) -> None:
    """📈 Performance Trends Section - Show performance evolution across releases.

    Uses geometric mean across all concurrency levels to provide a robust
    aggregate metric for each version/configuration combination.
    This section uses the full (unfiltered) DataFrame so it has its own
    independent inference server and version filters.

    Args:
        df: The full (unfiltered) DataFrame containing all benchmark data.
    """
    import re

    if "performance_trends_expanded" not in st.session_state:
        st.session_state.performance_trends_expanded = False

    with st.expander(
        "📈 Performance Trends Across Releases",
        expanded=st.session_state.performance_trends_expanded,
    ):
        st.markdown(
            "**Track how performance metrics have evolved across different releases** for your selected models and configurations."
        )
        st.info(
            "📊 **Note**: Values shown are **geometric means** across **common concurrency levels** shared by all selected versions, "
            "ensuring fair apples-to-apples comparison even when different versions were benchmarked at different concurrency ranges. "
            "This section has its own independent filters."
        )

        if df.empty:
            st.warning("No data available.")
            return

        # Extract version prefix (e.g., RHAIIS, vLLM, sglang) for grouping
        def get_version_prefix(version: str) -> str:
            if version.startswith("RHAIIS"):
                return "RHAIIS"
            elif version.startswith("vLLM"):
                return "vLLM"
            elif version.startswith("sglang"):
                return "sglang"
            elif version.startswith("TRT-LLM"):
                return "TRT-LLM"
            elif version.startswith("NIM"):
                return "NIM"
            else:
                return "Other"

        full_df = df.copy()
        full_df["version_prefix"] = full_df["version"].apply(get_version_prefix)

        # Version sorting function for proper chronological ordering
        def version_sort_key(version: str) -> tuple:
            """Sort versions chronologically (e.g., RHAIIS-3.1 < RHAIIS-3.2 < RHAIIS-3.2.1)."""
            # Extract numeric parts from version string
            parts = re.findall(r"(\d+)", version)
            # Pad with zeros for consistent sorting
            return tuple(int(p) for p in parts) if parts else (0,)

        def is_clean_version(version: str) -> bool:
            """Check if version has no postfix suffix.

            Clean versions: RHAIIS-3.2.3, vLLM-0.10.0, sglang-0.5.5, TRT-LLM-1.0.0rc5
            Postfix versions: RHAIIS-3.2.3-async, vLLM-0.11.0-gm3, sglang-0.5.5-rerun
            """
            return bool(
                re.match(
                    r"^[A-Za-z]+(?:-[A-Za-z]+)*-(\d+(?:\.\d+)*(?:rc\d+)?)$", version
                )
            )

        # Filter controls - Row 1: Inference Server, Accelerator, Model
        # Accelerator comes before Model so that changing models does NOT
        # reset the accelerator selection.
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Select inference server family - default to RHAIIS
            version_prefixes = sorted(full_df["version_prefix"].unique().tolist())
            # Ensure RHAIIS is first if available
            if "RHAIIS" in version_prefixes:
                version_prefixes.remove("RHAIIS")
                version_prefixes = ["RHAIIS"] + version_prefixes

            prefix_key = "trends_version_prefix"
            if prefix_key not in st.session_state and version_prefixes:
                st.session_state[prefix_key] = version_prefixes[0]

            selected_prefix = st.selectbox(
                "Select Inference Server",
                options=version_prefixes,
                key=prefix_key,
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

        # Filter to selected inference server family
        prefix_df = full_df[full_df["version_prefix"] == selected_prefix].copy()

        with filter_col2:
            # Select accelerator (scoped by inference server only, NOT by model)
            accelerators = sorted(prefix_df["accelerator"].unique().tolist())
            if not accelerators:
                st.warning(f"No accelerators found for {selected_prefix}.")
                return

            # Default to H200 if available
            accel_key = "trends_accelerator"
            if (
                accel_key not in st.session_state
                or st.session_state.get(accel_key) not in accelerators
            ):
                st.session_state[accel_key] = (
                    "H200" if "H200" in accelerators else accelerators[0]
                )

            selected_accelerator = st.selectbox(
                "Select Accelerator",
                options=accelerators,
                key=accel_key,
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

        accel_df = prefix_df[prefix_df["accelerator"] == selected_accelerator].copy()

        with filter_col3:
            # Select model (scoped by inference server + accelerator)
            models = sorted(accel_df["model"].unique().tolist())
            if not models:
                st.warning(
                    f"No models found for {selected_prefix} on {selected_accelerator}."
                )
                return

            # Default to Llama-3.3-70B-Instruct-FP8-dynamic if available,
            # with fallback to the non-FP8 variant
            preferred_models = [
                "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]
            model_key = "trends_model"

            def _best_default_model(model_list: list[str]) -> str:
                for pm in preferred_models:
                    if pm in model_list:
                        return pm
                return model_list[0]

            if (
                model_key not in st.session_state
                or st.session_state.get(model_key) not in models
            ):
                st.session_state[model_key] = _best_default_model(models)

            selected_model = st.selectbox(
                "Select Model",
                options=models,
                format_func=lambda x: x.split("/")[-1] if "/" in x else x,
                key=model_key,
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

        model_df = accel_df[accel_df["model"] == selected_model].copy()

        # Filter controls - Row 2: Profile, Versions
        filter_col4, filter_col5 = st.columns(2)

        with filter_col4:
            # Select ISL/OSL profile (scoped by model + accelerator)
            profiles = sorted(model_df["profile"].unique().tolist())
            if not profiles:
                st.warning("No profiles found for this configuration.")
                return

            # Default to "Profile A: Balanced (1k/1k)" if available
            default_profile = "Profile A: Balanced (1k/1k)"
            profile_key = "trends_profile"
            if (
                profile_key not in st.session_state
                or st.session_state.get(profile_key) not in profiles
            ):
                st.session_state[profile_key] = (
                    default_profile if default_profile in profiles else profiles[0]
                )

            selected_profile = st.selectbox(
                "Select ISL/OSL Profile",
                options=profiles,
                key=profile_key,
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

        profile_df = model_df[model_df["profile"] == selected_profile].copy()

        with filter_col5:
            # Show versions that exist for the selected model + accelerator + profile
            all_versions_for_config = sorted(
                profile_df["version"].unique().tolist(), key=version_sort_key
            )
            if not all_versions_for_config:
                st.warning("No versions found for this configuration.")
                return

            # Default to only clean versions (no postfix like -async, -sanity, -gm3)
            # Also exclude RHAIIS-3.1 by default (significantly slower baseline
            # that skews the visual comparison; users can still select it manually)
            default_versions = [
                v
                for v in all_versions_for_config
                if is_clean_version(v) and v != "RHAIIS-3.1"
            ]
            # Fall back to all if no clean versions exist
            if not default_versions:
                default_versions = all_versions_for_config

            selected_versions = st.multiselect(
                "Select Version(s)",
                options=all_versions_for_config,
                default=default_versions,
                key="trends_versions_multi",
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

            if not selected_versions:
                st.warning("Please select at least one version.")
                return

        # Filter to selected versions
        version_df = profile_df[profile_df["version"].isin(selected_versions)].copy()

        # Filter controls - Row 3: TP sizes, Metric
        filter_col6, filter_col7 = st.columns(2)

        with filter_col6:
            # Multi-select TP sizes - default to all
            tp_sizes = sorted(version_df["TP"].unique().tolist())
            if not tp_sizes:
                st.warning("No TP configurations found.")
                return

            selected_tps = st.multiselect(
                "Select TP Size(s)",
                options=tp_sizes,
                default=tp_sizes,  # Select all by default
                key="trends_tp_multi",
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )

            if not selected_tps:
                st.warning("Please select at least one TP size.")
                return

        with filter_col7:
            # Select metric to visualize
            metric_options = {
                "Throughput (Output tok/sec)": "output_tok/sec",
                "TTFT P95 (ms)": "ttft_p95",
                "ITL P95 (ms)": "itl_p95",
                "Request Latency Median (s)": "request_latency_median",
                "Total Throughput (tok/sec)": "total_tok/sec",
            }
            selected_metric_label = st.selectbox(
                "Select Metric",
                options=list(metric_options.keys()),
                key="trends_metric",
                on_change=keep_expander_open,
                args=("performance_trends_expanded",),
            )
            selected_metric = metric_options[selected_metric_label]

        # Filter to selected TP sizes
        trends_df = version_df[version_df["TP"].isin(selected_tps)].copy()

        if trends_df.empty:
            st.warning("No data found for the selected configuration.")
            return

        # --- Find common concurrency levels across all versions for each TP ---
        # This ensures fair apples-to-apples comparison (e.g., if v3.1 ran up to
        # concurrency 500 but v3.2+ ran up to 650, we only compare on the shared set)
        common_conc_per_tp = {}
        excluded_conc_per_tp = {}
        for tp in selected_tps:
            tp_subset = trends_df[trends_df["TP"] == tp]
            versions_in_tp = tp_subset["version"].unique()
            if len(versions_in_tp) == 0:
                continue
            # Get concurrency levels for each version
            conc_sets = []
            for v in versions_in_tp:
                conc_for_v = set(
                    tp_subset[tp_subset["version"] == v][
                        "intended concurrency"
                    ].unique()
                )
                conc_sets.append(conc_for_v)
            # Intersection = concurrency levels present in ALL versions
            common = conc_sets[0]
            all_conc = conc_sets[0].copy()
            for s in conc_sets[1:]:
                common = common & s
                all_conc = all_conc | s
            # Exclude concurrency=1 — single-request throughput is not
            # representative of production workloads and disproportionately
            # skews the geometric mean (especially for older releases).
            common = {c for c in common if c > 1}
            common_conc_per_tp[tp] = sorted(common)
            excluded_conc_per_tp[tp] = sorted(all_conc - common)

        # Filter trends_df to only common concurrency levels
        filtered_rows = []
        for tp in selected_tps:
            if tp in common_conc_per_tp and common_conc_per_tp[tp]:
                mask = (trends_df["TP"] == tp) & (
                    trends_df["intended concurrency"].isin(common_conc_per_tp[tp])
                )
                filtered_rows.append(trends_df[mask])
        if not filtered_rows:
            st.warning("No common concurrency levels found across selected versions.")
            return
        trends_df_common = pd.concat(filtered_rows, ignore_index=True)

        # Show info about common concurrency filtering
        conc_info_parts = []
        for tp in sorted(selected_tps):
            if tp in common_conc_per_tp:
                common_str = ", ".join(str(c) for c in common_conc_per_tp[tp])
                conc_info_parts.append(
                    f"**TP={tp}**: {len(common_conc_per_tp[tp])} common levels ({common_str})"
                )
                if excluded_conc_per_tp.get(tp):
                    excluded_str = ", ".join(str(c) for c in excluded_conc_per_tp[tp])
                    conc_info_parts[-1] += f" — excluded: {excluded_str}"
        if conc_info_parts:
            st.info(
                "**Fair comparison mode**: Geometric means are computed only over concurrency levels "
                "common to **all** selected versions (excluding concurrency=1, which is not representative "
                "of production workloads), ensuring apples-to-apples comparison.\n\n"
                + "\n\n".join(conc_info_parts)
            )

        # Calculate geometric mean for each version + TP combination across common concurrency levels
        def calc_geometric_mean(series: pd.Series) -> float:
            """Calculate geometric mean of positive values using numpy.

            Uses the same approach as other sections in the dashboard:
            np.exp(np.mean(np.log(positive_values)))
            """
            positive_values = series[series > 0].values
            if len(positive_values) == 0:
                return 0.0
            return float(np.exp(np.mean(np.log(positive_values))))

        # Group by version and TP, calculate geometric mean across COMMON concurrency levels
        agg_df = (
            trends_df_common.groupby(["version", "TP"], as_index=False)
            .agg(
                {
                    "output_tok/sec": calc_geometric_mean,
                    "ttft_p95": calc_geometric_mean,
                    "itl_p95": calc_geometric_mean,
                    "request_latency_median": calc_geometric_mean,
                    "total_tok/sec": calc_geometric_mean,
                    "successful_requests": "sum",
                    "errored_requests": "sum",
                    "intended concurrency": lambda x: sorted(
                        x.unique()
                    ),  # Track concurrency levels used
                }
            )
            .rename(columns={"intended concurrency": "concurrency_levels"})
        )

        # Also compute peak throughput (max output_tok/sec across ALL concurrency levels, not just common)
        peak_df = (
            trends_df.groupby(["version", "TP"], as_index=False)
            .agg({"output_tok/sec": "max"})
            .rename(columns={"output_tok/sec": "peak_output_tok_sec"})
        )
        agg_df = agg_df.merge(peak_df, on=["version", "TP"], how="left")

        # Get available versions and sort them chronologically
        available_versions = agg_df["version"].unique().tolist()
        available_versions = sorted(available_versions, key=version_sort_key)

        if len(available_versions) < 2:
            st.info(
                f"Only {len(available_versions)} version(s) available for this configuration. "
                "Need at least 2 versions to show trends."
            )
            if len(available_versions) == 1:
                st.write(f"Available version: **{available_versions[0]}**")
            return

        # Create ordered categorical for proper x-axis ordering
        agg_df["version"] = pd.Categorical(
            agg_df["version"], categories=available_versions, ordered=True
        )
        agg_df = agg_df.sort_values(["version", "TP"])

        # Create TP label for legend
        agg_df["TP_label"] = "TP=" + agg_df["TP"].astype(str)

        # Display configuration summary
        model_short = (
            selected_model.split("/")[-1] if "/" in selected_model else selected_model
        )
        tp_display = ", ".join([f"TP={tp}" for tp in sorted(selected_tps)])
        # Get short profile name for display
        profile_short = clean_profile_name(selected_profile)
        st.markdown(f"### 📊 {selected_prefix} Performance Trends (Geometric Mean)")
        st.markdown(
            f"**Model:** {model_short} | **Accelerator:** {selected_accelerator} | "
            f"**Profile:** {profile_short} | **{tp_display}**"
        )

        # Create the trend visualization with multiple TP lines
        is_latency_metric = selected_metric in [
            "ttft_p95",
            "itl_p95",
            "request_latency_median",
        ]

        # Generate colors for different TP sizes
        tp_colors = dict(
            zip(
                sorted(selected_tps),
                [
                    "#2ecc71",
                    "#3498db",
                    "#9b59b6",
                    "#e74c3c",
                    "#f39c12",
                    "#1abc9c",
                    "#e67e22",
                    "#34495e",
                ],
            )
        )

        fig = go.Figure()

        for tp in sorted(selected_tps):
            tp_data = agg_df[agg_df["TP"] == tp].copy()
            if tp_data.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=tp_data["version"].astype(str),
                    y=tp_data[selected_metric],
                    mode="lines+markers",
                    name=f"TP={tp}",
                    line={"color": tp_colors.get(tp, "#333"), "width": 3},
                    marker={"size": 10, "line": {"width": 2, "color": "white"}},
                    hovertemplate=(
                        f"<b>TP={tp}</b><br>"
                        + "Version: %{x}<br>"
                        + f"{selected_metric_label}: %{{y:.2f}}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Explicitly set x-axis category order to our chronologically sorted versions
        sorted_version_strings = [str(v) for v in available_versions]

        fig.update_layout(
            title=f"{selected_metric_label} Across {selected_prefix} Releases (Geometric Mean — Common Concurrency Levels)",
            xaxis_title="Release Version",
            yaxis_title=f"{selected_metric_label} (Geometric Mean)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            xaxis={
                "categoryorder": "array",
                "categoryarray": sorted_version_strings,
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )

        st.plotly_chart(fig, use_container_width=True, key="trends_main_chart")

        # Show summary table for each TP
        st.markdown("### 📋 Version-by-Version Comparison")

        for tp in sorted(selected_tps):
            tp_data = agg_df[agg_df["TP"] == tp].sort_values("version")
            if tp_data.empty:
                continue

            st.markdown(f"#### TP = {tp}")

            # Calculate changes between versions
            summary_data = []

            # Find the best value across all versions for this TP
            if is_latency_metric:
                best_value = tp_data[selected_metric].min()
                best_version = tp_data.loc[tp_data[selected_metric].idxmin(), "version"]
            else:
                best_value = tp_data[selected_metric].max()
                best_version = tp_data.loc[tp_data[selected_metric].idxmax(), "version"]

            for idx, (_, row) in enumerate(tp_data.iterrows()):
                version = row["version"]
                value = row[selected_metric]
                concurrency_count = (
                    len(row["concurrency_levels"])
                    if isinstance(row["concurrency_levels"], list)
                    else 1
                )

                entry = {
                    "Version": str(version),
                    f"{selected_metric_label} (Geom Mean)": f"{value:.2f}",
                    "Concurrency Levels": concurrency_count,
                }

                # Change vs previous version
                if idx > 0:
                    prev_value = tp_data[selected_metric].iloc[idx - 1]
                    if prev_value > 0:
                        change = ((value - prev_value) / prev_value) * 100
                        if is_latency_metric:
                            icon = "🟢" if change < 0 else "🔴" if change > 0 else "🟡"
                        else:
                            icon = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                        entry["Change vs Previous Version"] = f"{icon} {change:+.1f}%"
                    else:
                        entry["Change vs Previous Version"] = "N/A"
                else:
                    entry["Change vs Previous Version"] = "—"

                # Change vs best version
                if best_value > 0 and version != best_version:
                    best_change = ((value - best_value) / best_value) * 100
                    if is_latency_metric:
                        icon = (
                            "🟢"
                            if best_change < 0
                            else "🔴"
                            if best_change > 0
                            else "🟡"
                        )
                    else:
                        icon = (
                            "🟢"
                            if best_change > 0
                            else "🔴"
                            if best_change < 0
                            else "🟡"
                        )
                    entry[f"Change vs Best ({best_version})"] = (
                        f"{icon} {best_change:+.1f}%"
                    )
                elif version == best_version:
                    entry[f"Change vs Best ({best_version})"] = "⭐ Best"
                else:
                    entry[f"Change vs Best ({best_version})"] = "—"

                summary_data.append(entry)

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Additional multi-metric view
        with st.expander("📊 Multi-Metric Comparison", expanded=False):
            st.markdown(
                f"Compare multiple metrics side by side across versions for "
                f"**{model_short}** on **{selected_accelerator}** — Profile: **{profile_short}**"
            )
            st.caption(
                "**[Geo Mean]** = Geometric mean across common concurrency levels (excluding C=1). "
                "**[Max]** = Maximum value across all concurrency levels (not a geometric mean)."
            )

            for tp in sorted(selected_tps):
                tp_data = agg_df[agg_df["TP"] == tp].sort_values("version")
                if tp_data.empty or len(tp_data) < 2:
                    continue

                st.markdown(f"**TP = {tp}**")

                # Define all metrics for the 2x3 grid: (column_name, display_title, color, higher_is_better)
                all_multi_metrics = [
                    (
                        "output_tok/sec",
                        "Output Throughput (tok/s) [Geo Mean]",
                        "#27ae60",
                        True,
                    ),
                    (
                        "total_tok/sec",
                        "Total Throughput (tok/s) [Geo Mean]",
                        "#2ecc71",
                        True,
                    ),
                    (
                        "peak_output_tok_sec",
                        "Peak Output Throughput (tok/s) [Max]",
                        "#1abc9c",
                        True,
                    ),
                    ("ttft_p95", "TTFT P95 (ms) [Geo Mean]", "#e74c3c", False),
                    ("itl_p95", "ITL P95 (ms) [Geo Mean]", "#c0392b", False),
                    (
                        "request_latency_median",
                        "Request Latency Median (s) [Geo Mean]",
                        "#e67e22",
                        False,
                    ),
                ]

                available_multi = [
                    (col, title, color, hib)
                    for col, title, color, hib in all_multi_metrics
                    if col in tp_data.columns and tp_data[col].notna().any()
                ]

                if available_multi:
                    n_metrics = len(available_multi)
                    n_cols = min(n_metrics, 3)
                    n_rows = (n_metrics + n_cols - 1) // n_cols

                    fig_multi = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=[title for _, title, _, _ in available_multi],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.08,
                    )

                    for idx, (col, title, color, _higher_is_better) in enumerate(
                        available_multi
                    ):
                        row = idx // n_cols + 1
                        col_num = idx % n_cols + 1

                        fig_multi.add_trace(
                            go.Scatter(
                                x=tp_data["version"].astype(str),
                                y=tp_data[col],
                                mode="lines+markers",
                                name=title,
                                line={"color": color, "width": 2},
                                marker={"size": 8},
                            ),
                            row=row,
                            col=col_num,
                        )

                    fig_multi.update_layout(
                        height=300 * n_rows,
                        showlegend=False,
                        template="plotly_white",
                    )
                    # Set x-axis category order for each subplot
                    for i in range(n_metrics):
                        axis_key = "xaxis" if i == 0 else f"xaxis{i + 1}"
                        fig_multi.update_layout(
                            **{
                                axis_key: {
                                    "categoryorder": "array",
                                    "categoryarray": sorted_version_strings,
                                }
                            }
                        )

                    st.plotly_chart(
                        fig_multi,
                        use_container_width=True,
                        key=f"trends_multi_metric_tp_{tp}",
                    )

        # Show raw data
        with st.expander("📄 Raw Data (Geometric Mean Values)", expanded=False):
            st.caption(
                "Values below are **geometric means** across **common concurrency levels** "
                "(shared by all selected versions, excluding concurrency=1) for each version and TP combination. "
                "**Peak Output Throughput** is the maximum output_tok/sec across all concurrency levels (not a geometric mean)."
            )
            display_cols = [
                "version",
                "TP",
                "output_tok/sec",
                "total_tok/sec",
                "peak_output_tok_sec",
                "ttft_p95",
                "itl_p95",
                "request_latency_median",
                "successful_requests",
                "errored_requests",
            ]
            display_cols = [c for c in display_cols if c in agg_df.columns]
            st.dataframe(
                agg_df[display_cols].sort_values(["version", "TP"]).round(2),
                use_container_width=True,
                hide_index=True,
            )


@st.fragment
def render_compare_versions_summary_section(df):
    """⚖️ Compare Versions Section - Generate a summary table comparing two versions across multiple metrics."""
    if "compare_versions_summary_expanded" not in st.session_state:
        st.session_state.compare_versions_summary_expanded = False

    with st.expander(
        "⚖️ Compare Versions",
        expanded=st.session_state.compare_versions_summary_expanded,
    ):
        st.markdown(
            "💡 **Generate a comprehensive summary table comparing performance between two versions across all models and metrics.**"
        )

        # Get available versions, accelerators, and profiles from full data
        available_versions = sorted(df["version"].unique().tolist())
        available_accelerators = sorted(df["accelerator"].unique().tolist())
        available_profiles = sorted(df["profile"].unique().tolist())

        if len(available_versions) < 2:
            st.warning(
                "⚠️ Need at least 2 versions in the data to compare. Please check your data."
            )
            return

        # Filters row
        col1, col2, col3, col4 = st.columns(4)

        # Set default versions
        default_v1 = "vLLM-0.13.0-competitive"
        default_v2 = "TRT-LLM-1.2.0rc2"

        # Find index for default version 1
        v1_default_index = 0
        if default_v1 in available_versions:
            v1_default_index = available_versions.index(default_v1)

        with col1:
            version_1 = st.selectbox(
                "Select Version 1 (Baseline)",
                options=available_versions,
                index=v1_default_index,
                key="compare_summary_v1",
                on_change=keep_expander_open,
                args=("compare_versions_summary_expanded",),
            )

        with col2:
            version_2_options = [v for v in available_versions if v != version_1]
            # Find index for default version 2
            v2_default_index = 0
            if default_v2 in version_2_options:
                v2_default_index = version_2_options.index(default_v2)

            version_2 = (
                st.selectbox(
                    "Select Version 2 (Comparison)",
                    options=version_2_options,
                    index=v2_default_index if version_2_options else None,
                    key="compare_summary_v2",
                    on_change=keep_expander_open,
                    args=("compare_versions_summary_expanded",),
                )
                if version_2_options
                else None
            )

        with col3:
            # Default to H200 if available
            accel_default_index = 0
            if "H200" in available_accelerators:
                accel_default_index = available_accelerators.index("H200")

            selected_accelerator = st.selectbox(
                "Select GPU",
                options=available_accelerators,
                index=accel_default_index,
                key="compare_summary_accelerator",
                on_change=keep_expander_open,
                args=("compare_versions_summary_expanded",),
            )

        with col4:
            # Set default profile to "Profile A: Balanced (1k/1k)"
            default_profile = "Profile A: Balanced (1k/1k)"
            profile_default_index = 0
            if default_profile in available_profiles:
                profile_default_index = available_profiles.index(default_profile)

            selected_profile = st.selectbox(
                "Select ISL/OSL Profile",
                options=available_profiles,
                index=profile_default_index,
                key="compare_summary_profile",
                on_change=keep_expander_open,
                args=("compare_versions_summary_expanded",),
            )

        if not version_2:
            st.warning("⚠️ Please select a second version to compare.")
            return

        # Filter data for each version based on selected accelerator and profile
        df_v1 = df[
            (df["version"] == version_1)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        ].copy()

        df_v2 = df[
            (df["version"] == version_2)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        ].copy()

        if df_v1.empty or df_v2.empty:
            st.warning(
                "⚠️ No data available for the selected combination. "
                "Try different accelerator or profile settings."
            )
            return

        # Find common model+TP combinations between both versions
        # This ensures we compare the same model with the same TP value
        v1_model_tp = set(zip(df_v1["model"].tolist(), df_v1["TP"].tolist()))
        v2_model_tp = set(zip(df_v2["model"].tolist(), df_v2["TP"].tolist()))
        common_model_tp = sorted(v1_model_tp.intersection(v2_model_tp))

        if not common_model_tp:
            st.warning(
                f"⚠️ No common model+TP combinations found between {version_1} and {version_2} "
                f"for {selected_accelerator} with profile {selected_profile}."
            )
            return

        # Collect the union of all common concurrency levels across model+TP combos
        all_common_concurrencies: set = set()
        for model, tp in common_model_tp:
            v1_conc = set(
                df_v1[(df_v1["model"] == model) & (df_v1["TP"] == tp)][
                    "intended concurrency"
                ]
                .dropna()
                .unique()
            )
            v2_conc = set(
                df_v2[(df_v2["model"] == model) & (df_v2["TP"] == tp)][
                    "intended concurrency"
                ]
                .dropna()
                .unique()
            )
            all_common_concurrencies.update(v1_conc.intersection(v2_conc))

        all_common_concurrencies_sorted = sorted(
            int(c) for c in all_common_concurrencies
        )

        if all_common_concurrencies_sorted:
            # Key includes filter selections so the widget resets when filters change
            conc_key = f"compare_summary_conc_{version_1}_{version_2}_{selected_accelerator}_{selected_profile}"
            selected_concurrencies = st.multiselect(
                "Select Concurrency Level(s) for Geometric Mean",
                options=all_common_concurrencies_sorted,
                default=all_common_concurrencies_sorted,
                key=conc_key,
                on_change=keep_expander_open,
                args=("compare_versions_summary_expanded",),
                help=(
                    "Choose which concurrency levels to include in geometric mean calculations. "
                    "Only concurrency levels common to both versions are shown. "
                    "Peak throughput always uses all available concurrency levels."
                ),
            )
            if not selected_concurrencies:
                st.warning("⚠️ Please select at least one concurrency level.")
                return
            selected_conc_set = set(selected_concurrencies)
            st.caption(
                f"ℹ️ Geometric mean metrics use concurrency levels: "
                f"{', '.join(str(c) for c in sorted(selected_concurrencies))}. "
                f"Peak throughput uses all common concurrency levels."
            )
        else:
            selected_conc_set = set()

        # Extract ISL/OSL from profile for display
        profile_short = selected_profile
        if "(" in selected_profile and ")" in selected_profile:
            profile_short = selected_profile.split("(")[-1].replace(")", "")

        # Display title with GPU and ISL/OSL info + "How are these calculated?" popover
        title_col, popover_col = st.columns([5, 1])
        with title_col:
            st.markdown(f"### {selected_accelerator} GPU, ISL/OSL: {profile_short}")
        with popover_col:
            with st.popover("ℹ️ How are these calculated?"):
                st.markdown("""
                **Mean Change Calculation:**
                - Calculated by taking the percentage change at each common concurrency level, then taking the arithmetic mean (average) of all those changes
                - Shows the average performance difference across all concurrency levels
                - Can be affected by outliers (extreme values)
                - Formula: `mean([(v1 - v2) / v2 × 100 for each concurrency level])`

                **Median Change Calculation:**
                - Calculated by taking the percentage change at each common concurrency level, then taking the median of all those changes
                - Shows the typical performance difference across all concurrency levels
                - More robust to outliers than mean - better represents typical performance
                - Formula: `median([(v1 - v2) / v2 × 100 for each concurrency level])`

                **Geometric Mean Change Calculation:**

                *Step 1: Convert % changes to Growth Factors*
                - A **growth factor** is a multiplier that represents the ratio between V1 and V2
                - Formula: `growth_factor = 1 + (% change / 100)`
                - Examples:
                  - +10% → `1 + (10/100)` = **1.10** → means V1 is 110% of V2 (10% larger)
                  - -20% → `1 + (-20/100)` = **0.80** → means V1 is 80% of V2 (20% smaller)
                  - 0% → `1 + (0/100)` = **1.00** → means V1 equals V2 (no change)

                *Step 2: Compute Geometric Mean of Growth Factors*
                - Multiply all growth factors together, then take the nth root
                - Formula: `geom_mean_factor = (∏ growth_factors)^(1/n)`
                - Example: For [1.10, 0.90], geom_mean = (1.10 × 0.90)^0.5 = 0.99^0.5 ≈ 0.995

                *Step 3: Convert back to % change*
                - Formula: `geom_mean_% = (geom_mean_factor - 1) × 100`
                - Example: 1.10 - 1 = 0.10 → 0.10 × 100 = +10%
                - Example: 0.95 - 1 = -0.05 → -0.05 × 100 = -5%
                - **Why subtract 1?** Because a growth factor of 1.00 means "no change" (0%). The "1" represents the original value, so we subtract it to isolate just the change portion.

                *Why use Geometric Mean?*
                - Arithmetic mean of +100% and -50% = +25% ❌ (misleading!)
                - Geometric mean: (2.0 × 0.5)^0.5 - 1 = 1.0 - 1 = 0% ✅ (correct: doubling then halving = no net change)
                - Better for ratios/percentages because it respects multiplicative relationships

                **Peak Change Calculation:**
                - **For Throughput**: `((Version 1 Max - Version 2 Max) / Version 2 Max) × 100`
                  - Compares maximum throughput values (best = highest performance)
                  - Higher is better
                - **For Latency (TTFT/ITL)**: `((Version 1 Latency @ Max Throughput - Version 2 Latency @ Max Throughput) / Version 2 Latency @ Max Throughput) × 100`
                  - Compares latency values at the concurrency where max throughput occurs for each version
                  - This shows latency characteristics at peak performance
                  - Lower is better

                **How to Interpret the Percentage Values:**

                The percentage shows how much higher or lower V1's values are compared to V2:
                - **+X%** means V1's metric value is X% **higher** than V2's
                - **-X%** means V1's metric value is X% **lower** than V2's

                | Metric Type | +X% means | -X% means |
                |-------------|-----------|-----------|
                | **Throughput** | V1 is X% faster ✅ | V1 is X% slower ❌ |
                | **Latency (TTFT/ITL)** | V1 is X% slower ❌ | V1 is X% faster ✅ |

                *Example*: If TTFT shows +10%, it means V1's time-to-first-token is 10% higher (slower) than V2's.

                **Status Classification:**
                The status emoji is determined by consensus across all four metrics (Mean, Median, Geometric Mean, and Peak change):
                - 🟢 **Better**: At least 3 out of 4 metrics show ≥5% improvement
                - 🟡 **Similar**: Mixed signals (some metrics up, some down) or all metrics show <5% difference
                - 🔴 **Worse**: At least 3 out of 4 metrics show ≥5% decline

                This consensus approach provides a more robust assessment by requiring multiple metrics to agree before declaring a clear winner or loser.

                **Note**: Each accelerator-TP combination is compared independently across all common concurrency levels.
                    """)
        st.markdown(f"**Comparing:** {version_1} vs {version_2}")

        # Define metrics to compare
        metrics_config = {
            "Peak Output Throughput": {
                "column": "output_tok/sec",
                "aggregation": "peak",
                "higher_is_better": True,
                "show_concurrency": True,
            },
            "Output Throughput (Geometric Mean)": {
                "column": "output_tok/sec",
                "aggregation": "geom_mean",
                "higher_is_better": True,
                "show_concurrency": False,
            },
            "Total Throughput (Geometric Mean)": {
                "column": "total_tok/sec",
                "aggregation": "geom_mean",
                "higher_is_better": True,
                "show_concurrency": False,
            },
            "End-to-End Latency (Geometric Mean)": {
                "column": "request_latency_median",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
            "TTFT P95 (Geometric Mean)": {
                "column": "ttft_p95",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
            "ITL P95 (Geometric Mean)": {
                "column": "itl_p95",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
        }

        def calculate_geom_mean(values):
            """Calculate geometric mean of positive values."""
            positive_values = [v for v in values if v > 0]
            if not positive_values:
                return None
            return np.exp(np.mean(np.log(positive_values)))

        def get_comparison_result(v1_data, v2_data, metric_config, user_conc_set):
            """Calculate comparison between two versions for a specific metric."""
            column = metric_config["column"]
            aggregation = metric_config["aggregation"]
            higher_is_better = metric_config["higher_is_better"]

            # Get common concurrencies for this model+TP, intersected with user selection
            v1_concurrencies = set(v1_data["intended concurrency"].dropna().unique())
            v2_concurrencies = set(v2_data["intended concurrency"].dropna().unique())
            common_concurrencies = v1_concurrencies.intersection(v2_concurrencies)

            # For peak metrics use all common concurrencies;
            # for geom_mean metrics restrict to user-selected concurrencies
            if aggregation == "geom_mean":
                common_concurrencies = common_concurrencies.intersection(user_conc_set)

            if not common_concurrencies:
                return None, None, None, None, None

            v1_common = v1_data[
                v1_data["intended concurrency"].isin(common_concurrencies)
            ]
            v2_common = v2_data[
                v2_data["intended concurrency"].isin(common_concurrencies)
            ]

            v1_values = v1_common[column].dropna().tolist()
            v2_values = v2_common[column].dropna().tolist()

            if not v1_values or not v2_values:
                return None, None, None, None, None

            if aggregation == "peak":
                if higher_is_better:
                    v1_val = max(v1_values)
                    v2_val = max(v2_values)
                    v1_peak_idx = v1_common[column].idxmax()
                    v1_peak_conc = int(
                        v1_common.loc[v1_peak_idx, "intended concurrency"]
                    )
                    v2_peak_idx = v2_common[column].idxmax()
                    v2_peak_conc = int(
                        v2_common.loc[v2_peak_idx, "intended concurrency"]
                    )
                else:
                    v1_val = min(v1_values)
                    v2_val = min(v2_values)
                    v1_peak_idx = v1_common[column].idxmin()
                    v1_peak_conc = int(
                        v1_common.loc[v1_peak_idx, "intended concurrency"]
                    )
                    v2_peak_idx = v2_common[column].idxmin()
                    v2_peak_conc = int(
                        v2_common.loc[v2_peak_idx, "intended concurrency"]
                    )
            else:  # geom_mean
                v1_val = calculate_geom_mean(v1_values)
                v2_val = calculate_geom_mean(v2_values)
                v1_peak_conc = None
                v2_peak_conc = None

            if v1_val is None or v2_val is None or v2_val == 0:
                return None, None, None, None, None

            pct_diff = ((v1_val - v2_val) / v2_val) * 100
            v1_better = pct_diff > 0 if higher_is_better else pct_diff < 0

            return pct_diff, v1_better, v1_peak_conc, v2_peak_conc, abs(pct_diff) < 5

        # Check for duplicate rows (same version/model/TP/concurrency)
        dup_warnings = []
        for _label, df_check, ver_name in [
            ("Version 1", df_v1, version_1),
            ("Version 2", df_v2, version_2),
        ]:
            for model, tp in common_model_tp:
                subset = df_check[(df_check["model"] == model) & (df_check["TP"] == tp)]
                conc_counts = subset["intended concurrency"].value_counts()
                dups = conc_counts[conc_counts > 1]
                if not dups.empty:
                    m_short = model.split("/")[-1] if "/" in model else model
                    tp_s = f"TP={int(tp)}" if pd.notna(tp) else ""
                    conc_list = ", ".join(str(int(c)) for c in sorted(dups.index))
                    dup_warnings.append(
                        f"**{ver_name}** — {m_short} ({tp_s}): duplicate rows at "
                        f"concurrency {conc_list}"
                    )
        if dup_warnings:
            st.warning(
                "⚠️ **Duplicate data rows detected** — geometric mean results may be "
                "skewed. Consider removing duplicates from the CSV.\n\n"
                + "\n".join(f"- {w}" for w in dup_warnings)
            )

        # Build summary table data
        summary_data = []

        for model, tp in common_model_tp:
            model_short = model.split("/")[-1] if "/" in model else model

            # Get data for this specific model+TP combination
            v1_model_data = df_v1[(df_v1["model"] == model) & (df_v1["TP"] == tp)]
            v2_model_data = df_v2[(df_v2["model"] == model) & (df_v2["TP"] == tp)]

            # Format TP for display
            tp_str = f"(TP={int(tp)})" if pd.notna(tp) else ""

            row_data = {"Model": f"{model_short} {tp_str}"}

            for metric_name, metric_config in metrics_config.items():
                pct_diff, v1_better, v1_peak, v2_peak, is_similar = (
                    get_comparison_result(
                        v1_model_data, v2_model_data, metric_config, selected_conc_set
                    )
                )

                if pct_diff is None:
                    row_data[metric_name] = "N/A"
                else:
                    sign = "+" if pct_diff > 0 else ""
                    if metric_config["show_concurrency"] and v1_peak is not None:
                        cell_text = (
                            f"{version_1} ({sign}{pct_diff:.1f}%) "
                            f"peak@{v1_peak} vs {v2_peak}"
                        )
                    else:
                        cell_text = f"{version_1} ({sign}{pct_diff:.1f}%)"

                    if is_similar:
                        color = "🟡"
                    elif v1_better:
                        color = "🟢"
                    else:
                        color = "🔴"

                    row_data[metric_name] = f"{color} {cell_text}"

            summary_data.append(row_data)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # --- Metric comparison dialog (popup) ---
            @st.dialog("Version Comparison — Metric Details", width="large")
            def _show_metric_dialog(metric_name):
                """Render a popup with interactive line graphs."""
                mcfg = metrics_config[metric_name]
                col = mcfg["column"]
                agg = mcfg["aggregation"]
                mcfg["higher_is_better"]

                # Clean title: strip aggregation suffix, add "vs Concurrency"
                display_title = metric_name.replace(" (Geometric Mean)", "").replace(
                    " (Peak)", ""
                )
                st.markdown(f"#### {display_title} vs Concurrency")
                st.markdown(
                    f"**{version_1}** vs **{version_2}** &nbsp;|&nbsp; "
                    f"**{selected_accelerator}** &nbsp;|&nbsp; ISL/OSL: **{profile_short}**"
                )

                # Paired color palettes: warm tones for v1, cool tones for v2
                _palette_v1 = [
                    "#EF553B",
                    "#FF7F0E",
                    "#D62728",
                    "#E377C2",
                    "#FF6692",
                    "#FFA15A",
                    "#FECB52",
                    "#F0027F",
                    "#BF5B17",
                    "#E6550D",
                    "#FD8D3C",
                    "#FDAE6B",
                    "#FC4E2A",
                    "#FB6A4A",
                    "#CB181D",
                    "#EF3B2C",
                ]
                _palette_v2 = [
                    "#636EFA",
                    "#1F77B4",
                    "#00CC96",
                    "#19D3F3",
                    "#AB63FA",
                    "#17BECF",
                    "#2CA02C",
                    "#7F7F7F",
                    "#386CB0",
                    "#3690C0",
                    "#74C476",
                    "#9E9AC8",
                    "#6A51A3",
                    "#807DBA",
                    "#0570B0",
                    "#4292C6",
                ]

                # Collect per-concurrency data for all models
                per_model = []
                for m, tp in common_model_tp:
                    m_short = m.split("/")[-1] if "/" in m else m
                    tp_s = f" (TP={int(tp)})" if pd.notna(tp) else ""
                    lbl = f"{m_short}{tp_s}"

                    d1 = df_v1[(df_v1["model"] == m) & (df_v1["TP"] == tp)]
                    d2 = df_v2[(df_v2["model"] == m) & (df_v2["TP"] == tp)]

                    c1 = set(d1["intended concurrency"].dropna().unique())
                    c2 = set(d2["intended concurrency"].dropna().unique())
                    cc = c1.intersection(c2)
                    if agg == "geom_mean":
                        cc = cc.intersection(selected_conc_set)
                    if not cc:
                        continue

                    d1c = d1[d1["intended concurrency"].isin(cc)]
                    d2c = d2[d2["intended concurrency"].isin(cc)]

                    cc_sorted = sorted(cc)
                    v1_by_c, v2_by_c = [], []
                    for c in cc_sorted:
                        r1 = d1c[d1c["intended concurrency"] == c][col].values
                        r2 = d2c[d2c["intended concurrency"] == c][col].values
                        v1_by_c.append(float(r1[0]) if len(r1) > 0 else None)
                        v2_by_c.append(float(r2[0]) if len(r2) > 0 else None)

                    if not any(v is not None for v in v1_by_c) and not any(
                        v is not None for v in v2_by_c
                    ):
                        continue

                    per_model.append(
                        {
                            "label": lbl,
                            "conc": cc_sorted,
                            "v1": v1_by_c,
                            "v2": v2_by_c,
                        }
                    )

                if not per_model:
                    st.warning("No data available for this metric.")
                    return

                # Convert TTFT P95 from ms → seconds
                if col == "ttft_p95":
                    for md in per_model:
                        md["v1"] = [
                            v / 1000 if v is not None else None for v in md["v1"]
                        ]
                        md["v2"] = [
                            v / 1000 if v is not None else None for v in md["v2"]
                        ]

                # Build a single interactive line chart with all models
                fig = go.Figure()
                for idx, md in enumerate(per_model):
                    c_v1 = _palette_v1[idx % len(_palette_v1)]
                    c_v2 = _palette_v2[idx % len(_palette_v2)]
                    x_vals = [int(c) for c in md["conc"]]

                    # Version 1 — solid line, warm color
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=md["v1"],
                            mode="lines+markers",
                            name=f"{md['label']} ({version_1})",
                            line={"color": c_v1, "width": 2.5},
                            marker={"size": 8},
                            legendgroup=md["label"],
                            hovertemplate=(
                                f"<b>{md['label']}</b> — {version_1}<br>"
                                "Concurrency: %{x}<br>"
                                "Value: %{y:,.2f}<extra></extra>"
                            ),
                        )
                    )
                    # Version 2 — solid line, cool color
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=md["v2"],
                            mode="lines+markers",
                            name=f"{md['label']} ({version_2})",
                            line={"color": c_v2, "width": 2.5},
                            marker={"size": 8},
                            legendgroup=md["label"],
                            hovertemplate=(
                                f"<b>{md['label']}</b> — {version_2}<br>"
                                "Concurrency: %{x}<br>"
                                "Value: %{y:,.2f}<extra></extra>"
                            ),
                        )
                    )

                # Y-axis unit
                if "tok/sec" in col:
                    y_title = "Tokens / sec"
                elif "latency" in col.lower() or col == "ttft_p95":
                    y_title = "Seconds"
                else:
                    y_title = "Milliseconds"

                fig.update_layout(
                    height=600,
                    xaxis_title="Concurrency",
                    yaxis_title=y_title,
                    margin={"t": 30, "b": 60},
                    hovermode="x unified",
                    legend={
                        "orientation": "v",
                        "yanchor": "top",
                        "y": 1,
                        "xanchor": "left",
                        "x": 1.02,
                        "font": {"size": 11},
                        "itemclick": "toggle",
                        "itemdoubleclick": "toggleothers",
                    },
                    xaxis={
                        "type": "category",
                        "categoryorder": "array",
                        "categoryarray": sorted(
                            {int(c) for md in per_model for c in md["conc"]}
                        ),
                    },
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"dlg_line_{metric_name}",
                )

                st.caption(
                    "💡 **Tip:** Click a legend entry to toggle it. "
                    "Double-click to isolate a single trace. "
                    f"Warm colors (reds/oranges) = **{version_1}**, "
                    f"cool colors (blues/greens) = **{version_2}**."
                )

                if agg == "geom_mean":
                    conc_str = ", ".join(str(int(c)) for c in sorted(selected_conc_set))
                    st.caption(
                        f"ℹ️ Showing data at concurrency levels: {conc_str} "
                        "(filtered by geometric mean concurrency selection)."
                    )
                else:
                    st.caption(
                        "ℹ️ Showing data across all common concurrency "
                        "levels between the two versions."
                    )

            # --- Metric comparison buttons ---
            st.markdown(
                "**📊 Click a metric below to open a detailed comparison popup:**"
            )
            # Exclude "Peak Output Throughput" (same underlying graph as
            # Output Throughput since both use output_tok/sec vs concurrency)
            btn_metrics = [m for m in metrics_config if m != "Peak Output Throughput"]
            btn_cols = st.columns(len(btn_metrics))
            for i, m_name in enumerate(btn_metrics):
                with btn_cols[i]:
                    short = m_name.replace(" (Geometric Mean)", "").replace(
                        "Throughput", "Tput"
                    )
                    if st.button(
                        f"📊 {short}",
                        key=f"cmp_btn_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.compare_versions_summary_expanded = True
                        _show_metric_dialog(m_name)

            st.markdown("")

            # Add hover tip note above table, aligned right
            st.markdown(
                "<div style='text-align: right;'>"
                "<span style='font-size: 0.85em; color: gray;'>"
                "💡 <b>Tip:</b> Hover over column headers to see detailed descriptions."
                "</span></div>",
                unsafe_allow_html=True,
            )

            # Define column config with help tooltips
            column_config = {
                "Model": st.column_config.TextColumn(
                    "Model",
                    help="Model name with tensor parallelism (TP) configuration",
                ),
                "Peak Output Throughput": st.column_config.TextColumn(
                    "Peak Output Throughput",
                    help="Maximum output tokens/sec achieved. Shows peak concurrency for V1 vs V2 (e.g. peak@200 vs 100).",
                ),
                "Output Throughput (Geometric Mean)": st.column_config.TextColumn(
                    "Output Throughput (Geometric Mean)",
                    help="Geometric mean of output tok/sec across selected concurrency levels",
                ),
                "Total Throughput (Geometric Mean)": st.column_config.TextColumn(
                    "Total Throughput (Geometric Mean)",
                    help="Geometric mean of total (input + output) tok/sec across selected concurrency levels",
                ),
                "End-to-End Latency (Geometric Mean)": st.column_config.TextColumn(
                    "End-to-End Latency (Geometric Mean)",
                    help="Geometric mean of request latency median across selected concurrency levels",
                ),
                "TTFT P95 (Geometric Mean)": st.column_config.TextColumn(
                    "TTFT P95 (Geometric Mean)",
                    help="Geometric mean of Time-to-First-Token (P95) across all concurrency levels",
                ),
                "ITL P95 (Geometric Mean)": st.column_config.TextColumn(
                    "ITL P95 (Geometric Mean)",
                    help="Geometric mean of Inter-Token Latency (P95) across all concurrency levels",
                ),
            }

            # Display the table with column tooltips
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

            # Legend
            st.markdown("---")
            st.markdown(
                f"**Legend:** "
                f"🟢 {version_1} performs better than {version_2} | "
                f"🔴 {version_1} performs worse than {version_2} | "
                f"🟡 Similar Performance (< 5% difference)"
            )

            # Detailed model comparison sections
            st.markdown("---")
            st.markdown("### 📋 Detailed Model Comparisons")
            st.markdown("*Click on a model to see detailed metrics comparison*")

            for idx, (model, tp) in enumerate(common_model_tp, 1):
                model_short = model.split("/")[-1] if "/" in model else model

                # Get data for this specific model+TP combination
                v1_model_data = df_v1[(df_v1["model"] == model) & (df_v1["TP"] == tp)]
                v2_model_data = df_v2[(df_v2["model"] == model) & (df_v2["TP"] == tp)]

                # Get TP value for display
                tp_val = int(tp) if pd.notna(tp) else "N/A"

                # Get common concurrencies
                v1_concurrencies = set(
                    v1_model_data["intended concurrency"].dropna().unique()
                )
                v2_concurrencies = set(
                    v2_model_data["intended concurrency"].dropna().unique()
                )
                common_conc = v1_concurrencies.intersection(v2_concurrencies)

                if not common_conc:
                    continue

                v1_common = v1_model_data[
                    v1_model_data["intended concurrency"].isin(common_conc)
                ]
                v2_common = v2_model_data[
                    v2_model_data["intended concurrency"].isin(common_conc)
                ]

                # Find peak throughput info for each version
                v1_peak_idx = v1_common["output_tok/sec"].idxmax()
                v2_peak_idx = v2_common["output_tok/sec"].idxmax()

                v1_peak_throughput = v1_common.loc[v1_peak_idx, "output_tok/sec"]
                v2_peak_throughput = v2_common.loc[v2_peak_idx, "output_tok/sec"]
                v1_peak_conc = int(v1_common.loc[v1_peak_idx, "intended concurrency"])
                v2_peak_conc = int(v2_common.loc[v2_peak_idx, "intended concurrency"])

                # Get total throughput at peak
                v1_total_throughput = v1_common.loc[v1_peak_idx, "total_tok/sec"]
                v2_total_throughput = v2_common.loc[v2_peak_idx, "total_tok/sec"]

                # Get latency metrics at peak throughput
                v1_e2e_latency = v1_common.loc[v1_peak_idx, "request_latency_median"]
                v2_e2e_latency = v2_common.loc[v2_peak_idx, "request_latency_median"]

                v1_ttft = v1_common.loc[v1_peak_idx, "ttft_p95"]
                v2_ttft = v2_common.loc[v2_peak_idx, "ttft_p95"]

                v1_itl = v1_common.loc[v1_peak_idx, "itl_p95"]
                v2_itl = v2_common.loc[v2_peak_idx, "itl_p95"]

                def format_value(val, unit="", decimals=0, round_up=False):
                    """Format a numeric value with optional unit."""
                    if pd.isna(val):
                        return "N/A"
                    if round_up:
                        import math

                        if decimals == 0:
                            return f"~{int(math.ceil(val)):,}{unit}"
                        else:
                            factor = 10**decimals
                            rounded_val = math.ceil(val * factor) / factor
                            return f"~{rounded_val:,.{decimals}f}{unit}"
                    if decimals == 0:
                        return f"~{int(val):,}{unit}"
                    return f"~{val:,.{decimals}f}{unit}"

                def get_winner_text(v1_val, v2_val, higher_is_better, metric_name):
                    """Generate winner text for a metric."""
                    if pd.isna(v1_val) or pd.isna(v2_val) or v2_val == 0:
                        return "N/A"

                    pct_diff = ((v1_val - v2_val) / v2_val) * 100

                    if higher_is_better:
                        if pct_diff > 5:
                            return f"{version_1} has +{abs(pct_diff):.1f}% higher {metric_name}"
                        elif pct_diff < -5:
                            return f"{version_2} has +{abs(pct_diff):.1f}% higher {metric_name}"
                        else:
                            return f"Similar (~{abs(pct_diff):.1f}% difference)"
                    else:
                        if pct_diff < -5:
                            return f"{version_1} has {abs(pct_diff):.1f}% lower {metric_name}"
                        elif pct_diff > 5:
                            return f"{version_2} has {abs(pct_diff):.1f}% lower {metric_name}"
                        else:
                            return f"Similar (~{abs(pct_diff):.1f}% difference)"

                with st.expander(f"{idx}. {model_short} (TP={tp_val})"):
                    detail_rows = [
                        {
                            "Metric": "Peak Output Throughput (output tok/s)",
                            version_1: f"{format_value(v1_peak_throughput)} tok/s at {v1_peak_conc} concurrent users",
                            version_2: f"{format_value(v2_peak_throughput)} tok/s at {v2_peak_conc} concurrent users",
                            "Difference/Winner": get_winner_text(
                                v1_peak_throughput,
                                v2_peak_throughput,
                                True,
                                "peak output throughput",
                            ),
                        },
                        {
                            "Metric": "Total Throughput (input + output tok/s)",
                            version_1: f"{format_value(v1_total_throughput)} tok/s at {v1_peak_conc} concurrent users",
                            version_2: f"{format_value(v2_total_throughput)} tok/s at {v2_peak_conc} concurrent users",
                            "Difference/Winner": get_winner_text(
                                v1_total_throughput,
                                v2_total_throughput,
                                True,
                                "total throughput",
                            ),
                        },
                        {
                            "Metric": "Median E2E Latency at Peak Throughput",
                            version_1: f"{format_value(v1_e2e_latency, 's', 0, round_up=True)}",
                            version_2: f"{format_value(v2_e2e_latency, 's', 0, round_up=True)}",
                            "Difference/Winner": get_winner_text(
                                v1_e2e_latency, v2_e2e_latency, False, "E2E latency"
                            ),
                        },
                        {
                            "Metric": "TTFT P95 at Peak Throughput",
                            version_1: f"{format_value(v1_ttft / 1000, 's', 2, round_up=True)}"
                            if pd.notna(v1_ttft)
                            else "N/A",
                            version_2: f"{format_value(v2_ttft / 1000, 's', 2, round_up=True)}"
                            if pd.notna(v2_ttft)
                            else "N/A",
                            "Difference/Winner": get_winner_text(
                                v1_ttft, v2_ttft, False, "P95 TTFT"
                            ),
                        },
                        {
                            "Metric": "ITL P95 at Peak Throughput",
                            version_1: f"{format_value(v1_itl, 'ms', 0, round_up=True)}",
                            version_2: f"{format_value(v2_itl, 'ms', 0, round_up=True)}",
                            "Difference/Winner": get_winner_text(
                                v1_itl, v2_itl, False, "P95 ITL"
                            ),
                        },
                    ]

                    detail_df = pd.DataFrame(detail_rows)
                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("No comparison data available for the selected filters.")


def render_compare_models_section(filtered_df, selected_profile):
    """⚖️ Compare Models Section - Compare two models across multiple metrics (Custom ISL/OSL only)."""
    if selected_profile != "Custom":
        return

    if "compare_models_expanded" not in st.session_state:
        st.session_state.compare_models_expanded = False

    with st.expander(
        "⚖️ Compare Models",
        expanded=st.session_state.compare_models_expanded,
    ):
        st.markdown(
            "💡 **Compare performance between two models across all versions and metrics.**"
        )

        available_models = sorted(filtered_df["model"].unique().tolist())

        if len(available_models) < 2:
            st.warning(
                "⚠️ Need at least 2 models in the filtered data to compare. "
                "Please adjust your filters."
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            model_1 = st.selectbox(
                "Select Model 1 (Baseline)",
                options=available_models,
                index=0,
                key="compare_models_m1",
                on_change=keep_expander_open,
                args=("compare_models_expanded",),
            )

        with col2:
            model_2_options = [m for m in available_models if m != model_1]
            model_2 = (
                st.selectbox(
                    "Select Model 2 (Comparison)",
                    options=model_2_options,
                    index=0 if model_2_options else None,
                    key="compare_models_m2",
                    on_change=keep_expander_open,
                    args=("compare_models_expanded",),
                )
                if model_2_options
                else None
            )

        if not model_2:
            st.warning("⚠️ Please select a second model to compare.")
            return

        df_m1 = filtered_df[filtered_df["model"] == model_1].copy()
        df_m2 = filtered_df[filtered_df["model"] == model_2].copy()

        if df_m1.empty or df_m2.empty:
            st.warning(
                "⚠️ No data available for one of the selected models with the current filters."
            )
            return

        # Find common version+TP combinations between both models
        m1_version_tp = set(zip(df_m1["version"].tolist(), df_m1["TP"].tolist()))
        m2_version_tp = set(zip(df_m2["version"].tolist(), df_m2["TP"].tolist()))
        common_version_tp = sorted(m1_version_tp.intersection(m2_version_tp))

        if not common_version_tp:
            st.warning(
                "⚠️ No common version+TP combinations found between the two models "
                "with the current filters."
            )
            return

        # Collect the union of all common concurrency levels across version+TP combos
        all_common_concurrencies: set = set()
        for version, tp in common_version_tp:
            m1_conc = set(
                df_m1[(df_m1["version"] == version) & (df_m1["TP"] == tp)][
                    "intended concurrency"
                ]
                .dropna()
                .unique()
            )
            m2_conc = set(
                df_m2[(df_m2["version"] == version) & (df_m2["TP"] == tp)][
                    "intended concurrency"
                ]
                .dropna()
                .unique()
            )
            all_common_concurrencies.update(m1_conc.intersection(m2_conc))

        all_common_concurrencies_sorted = sorted(
            int(c) for c in all_common_concurrencies
        )

        if all_common_concurrencies_sorted:
            conc_key = f"compare_models_conc_{model_1}_{model_2}"
            selected_concurrencies = st.multiselect(
                "Select Concurrency Level(s) for Geometric Mean",
                options=all_common_concurrencies_sorted,
                default=all_common_concurrencies_sorted,
                key=conc_key,
                on_change=keep_expander_open,
                args=("compare_models_expanded",),
                help=(
                    "Choose which concurrency levels to include in geometric mean calculations. "
                    "Only concurrency levels common to both models are shown. "
                    "Peak throughput always uses all available concurrency levels."
                ),
            )
            if not selected_concurrencies:
                st.warning("⚠️ Please select at least one concurrency level.")
                return
            selected_conc_set = set(selected_concurrencies)
            st.caption(
                f"ℹ️ Geometric mean metrics use concurrency levels: "
                f"{', '.join(str(c) for c in sorted(selected_concurrencies))}. "
                f"Peak throughput uses all common concurrency levels."
            )
        else:
            selected_conc_set = set()

        model_1_short = model_1.split("/")[-1] if "/" in model_1 else model_1
        model_2_short = model_2.split("/")[-1] if "/" in model_2 else model_2

        title_col, popover_col = st.columns([5, 1])
        with title_col:
            st.markdown(f"### Comparing: {model_1_short} vs {model_2_short}")
        with popover_col:
            with st.popover("ℹ️ How are these calculated?"):
                st.markdown("""
                **Geometric Mean Change Calculation:**

                *Step 1: Convert % changes to Growth Factors*
                - Formula: `growth_factor = 1 + (% change / 100)`

                *Step 2: Compute Geometric Mean of Growth Factors*
                - Multiply all growth factors together, then take the nth root

                *Step 3: Convert back to % change*
                - Formula: `geom_mean_% = (geom_mean_factor - 1) × 100`

                *Why use Geometric Mean?*
                - Arithmetic mean of +100% and -50% = +25% (misleading!)
                - Geometric mean: (2.0 x 0.5)^0.5 - 1 = 0% (correct)
                - Better for ratios/percentages because it respects multiplicative relationships

                **Peak Change Calculation:**
                - **For Throughput**: `((Model 1 Max - Model 2 Max) / Model 2 Max) x 100`
                - **For Latency**: Compares latency values at the concurrency where max throughput occurs

                **How to Interpret:**
                - **+X%** means Model 1's metric value is X% **higher** than Model 2's
                - **-X%** means Model 1's metric value is X% **lower** than Model 2's

                | Metric Type | +X% means | -X% means |
                |-------------|-----------|-----------|
                | **Throughput** | M1 is X% faster | M1 is X% slower |
                | **Latency** | M1 is X% slower | M1 is X% faster |

                **Status:** 🟢 Better (>=5% improvement) | 🟡 Similar (<5%) | 🔴 Worse (>=5% decline)
                    """)

        st.markdown(f"**Comparing:** {model_1_short} vs {model_2_short}")

        metrics_config = {
            "Peak Output Throughput": {
                "column": "output_tok/sec",
                "aggregation": "peak",
                "higher_is_better": True,
                "show_concurrency": True,
            },
            "Output Throughput (Geometric Mean)": {
                "column": "output_tok/sec",
                "aggregation": "geom_mean",
                "higher_is_better": True,
                "show_concurrency": False,
            },
            "Total Throughput (Geometric Mean)": {
                "column": "total_tok/sec",
                "aggregation": "geom_mean",
                "higher_is_better": True,
                "show_concurrency": False,
            },
            "End-to-End Latency (Geometric Mean)": {
                "column": "request_latency_median",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
            "TTFT P95 (Geometric Mean)": {
                "column": "ttft_p95",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
            "ITL P95 (Geometric Mean)": {
                "column": "itl_p95",
                "aggregation": "geom_mean",
                "higher_is_better": False,
                "show_concurrency": False,
            },
        }

        def calculate_geom_mean(values):
            """Calculate geometric mean of positive values."""
            positive_values = [v for v in values if v > 0]
            if not positive_values:
                return None
            return np.exp(np.mean(np.log(positive_values)))

        def get_comparison_result(m1_data, m2_data, metric_config, user_conc_set):
            """Calculate comparison between two models for a specific metric."""
            column = metric_config["column"]
            aggregation = metric_config["aggregation"]
            higher_is_better = metric_config["higher_is_better"]

            m1_concurrencies = set(m1_data["intended concurrency"].dropna().unique())
            m2_concurrencies = set(m2_data["intended concurrency"].dropna().unique())
            common_concurrencies = m1_concurrencies.intersection(m2_concurrencies)

            if aggregation == "geom_mean":
                common_concurrencies = common_concurrencies.intersection(user_conc_set)

            if not common_concurrencies:
                return None, None, None, None, None

            m1_common = m1_data[
                m1_data["intended concurrency"].isin(common_concurrencies)
            ]
            m2_common = m2_data[
                m2_data["intended concurrency"].isin(common_concurrencies)
            ]

            m1_values = m1_common[column].dropna().tolist()
            m2_values = m2_common[column].dropna().tolist()

            if not m1_values or not m2_values:
                return None, None, None, None, None

            if aggregation == "peak":
                if higher_is_better:
                    m1_val = max(m1_values)
                    m2_val = max(m2_values)
                    m1_peak_idx = m1_common[column].idxmax()
                    m1_peak_conc = int(
                        m1_common.loc[m1_peak_idx, "intended concurrency"]
                    )
                    m2_peak_idx = m2_common[column].idxmax()
                    m2_peak_conc = int(
                        m2_common.loc[m2_peak_idx, "intended concurrency"]
                    )
                else:
                    m1_val = min(m1_values)
                    m2_val = min(m2_values)
                    m1_peak_idx = m1_common[column].idxmin()
                    m1_peak_conc = int(
                        m1_common.loc[m1_peak_idx, "intended concurrency"]
                    )
                    m2_peak_idx = m2_common[column].idxmin()
                    m2_peak_conc = int(
                        m2_common.loc[m2_peak_idx, "intended concurrency"]
                    )
            else:
                m1_val = calculate_geom_mean(m1_values)
                m2_val = calculate_geom_mean(m2_values)
                m1_peak_conc = None
                m2_peak_conc = None

            if m1_val is None or m2_val is None or m2_val == 0:
                return None, None, None, None, None

            pct_diff = ((m1_val - m2_val) / m2_val) * 100
            m1_better = pct_diff > 0 if higher_is_better else pct_diff < 0

            return pct_diff, m1_better, m1_peak_conc, m2_peak_conc, abs(pct_diff) < 5

        # Check for duplicate rows (same model/version/TP/concurrency)
        dup_warnings = []
        for _label, df_check, name in [
            ("Model 1", df_m1, model_1_short),
            ("Model 2", df_m2, model_2_short),
        ]:
            for version, tp in common_version_tp:
                subset = df_check[
                    (df_check["version"] == version) & (df_check["TP"] == tp)
                ]
                conc_counts = subset["intended concurrency"].value_counts()
                dups = conc_counts[conc_counts > 1]
                if not dups.empty:
                    tp_s = f"TP={int(tp)}" if pd.notna(tp) else ""
                    conc_list = ", ".join(str(int(c)) for c in sorted(dups.index))
                    dup_warnings.append(
                        f"**{name}** ({version} {tp_s}): duplicate rows at "
                        f"concurrency {conc_list}"
                    )
        if dup_warnings:
            st.warning(
                "⚠️ **Duplicate data rows detected** — geometric mean results may be "
                "skewed. Consider removing duplicates from the CSV.\n\n"
                + "\n".join(f"- {w}" for w in dup_warnings)
            )

        # Build summary table data
        summary_data = []

        for version, tp in common_version_tp:
            v_m1_data = df_m1[(df_m1["version"] == version) & (df_m1["TP"] == tp)]
            v_m2_data = df_m2[(df_m2["version"] == version) & (df_m2["TP"] == tp)]

            tp_str = f"(TP={int(tp)})" if pd.notna(tp) else ""
            row_data = {"Version": f"{version} {tp_str}"}

            for metric_name, metric_config in metrics_config.items():
                pct_diff, m1_better, m1_peak, m2_peak, is_similar = (
                    get_comparison_result(
                        v_m1_data, v_m2_data, metric_config, selected_conc_set
                    )
                )

                if pct_diff is None:
                    row_data[metric_name] = "N/A"
                else:
                    sign = "+" if pct_diff > 0 else ""
                    if metric_config["show_concurrency"] and m1_peak is not None:
                        cell_text = (
                            f"{model_1_short} ({sign}{pct_diff:.1f}%) "
                            f"peak@{m1_peak} vs {m2_peak}"
                        )
                    else:
                        cell_text = f"{model_1_short} ({sign}{pct_diff:.1f}%)"

                    if is_similar:
                        color = "🟡"
                    elif m1_better:
                        color = "🟢"
                    else:
                        color = "🔴"

                    row_data[metric_name] = f"{color} {cell_text}"

            summary_data.append(row_data)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # --- Metric comparison dialog (popup) ---
            @st.dialog("Model Comparison — Metric Details", width="large")
            def _show_model_metric_dialog(metric_name):
                """Render a popup with interactive line graphs."""
                mcfg = metrics_config[metric_name]
                col = mcfg["column"]
                agg = mcfg["aggregation"]

                display_title = metric_name.replace(" (Geometric Mean)", "").replace(
                    " (Peak)", ""
                )
                st.markdown(f"#### {display_title} vs Concurrency")
                st.markdown(f"**{model_1_short}** vs **{model_2_short}**")

                _palette_m1 = [
                    "#EF553B",
                    "#FF7F0E",
                    "#D62728",
                    "#E377C2",
                    "#FF6692",
                    "#FFA15A",
                    "#FECB52",
                    "#F0027F",
                    "#BF5B17",
                    "#E6550D",
                    "#FD8D3C",
                    "#FDAE6B",
                    "#FC4E2A",
                    "#FB6A4A",
                    "#CB181D",
                    "#EF3B2C",
                ]
                _palette_m2 = [
                    "#636EFA",
                    "#1F77B4",
                    "#00CC96",
                    "#19D3F3",
                    "#AB63FA",
                    "#17BECF",
                    "#2CA02C",
                    "#7F7F7F",
                    "#386CB0",
                    "#3690C0",
                    "#74C476",
                    "#9E9AC8",
                    "#6A51A3",
                    "#807DBA",
                    "#0570B0",
                    "#4292C6",
                ]

                per_version = []
                for v, tp in common_version_tp:
                    tp_s = f" (TP={int(tp)})" if pd.notna(tp) else ""
                    lbl = f"{v}{tp_s}"

                    d1 = df_m1[(df_m1["version"] == v) & (df_m1["TP"] == tp)]
                    d2 = df_m2[(df_m2["version"] == v) & (df_m2["TP"] == tp)]

                    c1 = set(d1["intended concurrency"].dropna().unique())
                    c2 = set(d2["intended concurrency"].dropna().unique())
                    cc = c1.intersection(c2)
                    if agg == "geom_mean":
                        cc = cc.intersection(selected_conc_set)
                    if not cc:
                        continue

                    d1c = d1[d1["intended concurrency"].isin(cc)]
                    d2c = d2[d2["intended concurrency"].isin(cc)]

                    cc_sorted = sorted(cc)
                    m1_by_c, m2_by_c = [], []
                    for c in cc_sorted:
                        r1 = d1c[d1c["intended concurrency"] == c][col].values
                        r2 = d2c[d2c["intended concurrency"] == c][col].values
                        m1_by_c.append(float(r1[0]) if len(r1) > 0 else None)
                        m2_by_c.append(float(r2[0]) if len(r2) > 0 else None)

                    if not any(v is not None for v in m1_by_c) and not any(
                        v is not None for v in m2_by_c
                    ):
                        continue

                    per_version.append(
                        {
                            "label": lbl,
                            "conc": cc_sorted,
                            "m1": m1_by_c,
                            "m2": m2_by_c,
                        }
                    )

                if not per_version:
                    st.warning("No data available for this metric.")
                    return

                # Convert TTFT P95 from ms → seconds
                if col == "ttft_p95":
                    for md in per_version:
                        md["m1"] = [
                            v / 1000 if v is not None else None for v in md["m1"]
                        ]
                        md["m2"] = [
                            v / 1000 if v is not None else None for v in md["m2"]
                        ]

                fig = go.Figure()
                for idx, md in enumerate(per_version):
                    c_m1 = _palette_m1[idx % len(_palette_m1)]
                    c_m2 = _palette_m2[idx % len(_palette_m2)]
                    x_vals = [int(c) for c in md["conc"]]

                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=md["m1"],
                            mode="lines+markers",
                            name=f"{md['label']} ({model_1_short})",
                            line={"color": c_m1, "width": 2.5},
                            marker={"size": 8},
                            legendgroup=md["label"],
                            hovertemplate=(
                                f"<b>{md['label']}</b> — {model_1_short}<br>"
                                "Concurrency: %{x}<br>"
                                "Value: %{y:,.2f}<extra></extra>"
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=md["m2"],
                            mode="lines+markers",
                            name=f"{md['label']} ({model_2_short})",
                            line={"color": c_m2, "width": 2.5},
                            marker={"size": 8},
                            legendgroup=md["label"],
                            hovertemplate=(
                                f"<b>{md['label']}</b> — {model_2_short}<br>"
                                "Concurrency: %{x}<br>"
                                "Value: %{y:,.2f}<extra></extra>"
                            ),
                        )
                    )

                if "tok/sec" in col:
                    y_title = "Tokens / sec"
                elif "latency" in col.lower() or col == "ttft_p95":
                    y_title = "Seconds"
                else:
                    y_title = "Milliseconds"

                fig.update_layout(
                    height=600,
                    xaxis_title="Concurrency",
                    yaxis_title=y_title,
                    margin={"t": 30, "b": 60},
                    hovermode="x unified",
                    legend={
                        "orientation": "v",
                        "yanchor": "top",
                        "y": 1,
                        "xanchor": "left",
                        "x": 1.02,
                        "font": {"size": 11},
                        "itemclick": "toggle",
                        "itemdoubleclick": "toggleothers",
                    },
                    xaxis={
                        "type": "category",
                        "categoryorder": "array",
                        "categoryarray": sorted(
                            {int(c) for md in per_version for c in md["conc"]}
                        ),
                    },
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"cmp_models_dlg_line_{metric_name}",
                )

                st.caption(
                    "💡 **Tip:** Click a legend entry to toggle it. "
                    "Double-click to isolate a single trace. "
                    f"Warm colors (reds/oranges) = **{model_1_short}**, "
                    f"cool colors (blues/greens) = **{model_2_short}**."
                )

                if agg == "geom_mean":
                    conc_str = ", ".join(str(int(c)) for c in sorted(selected_conc_set))
                    st.caption(
                        f"ℹ️ Showing data at concurrency levels: {conc_str} "
                        "(filtered by geometric mean concurrency selection)."
                    )
                else:
                    st.caption(
                        "ℹ️ Showing data across all common concurrency "
                        "levels between the two models."
                    )

            # --- Metric comparison buttons ---
            st.markdown(
                "**📊 Click a metric below to open a detailed comparison popup:**"
            )
            btn_metrics = [m for m in metrics_config if m != "Peak Output Throughput"]
            btn_cols = st.columns(len(btn_metrics))
            for i, m_name in enumerate(btn_metrics):
                with btn_cols[i]:
                    short = m_name.replace(" (Geometric Mean)", "").replace(
                        "Throughput", "Tput"
                    )
                    if st.button(
                        f"📊 {short}",
                        key=f"cmp_models_btn_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.compare_models_expanded = True
                        _show_model_metric_dialog(m_name)

            st.markdown("")

            st.markdown(
                "<div style='text-align: right;'>"
                "<span style='font-size: 0.85em; color: gray;'>"
                "💡 <b>Tip:</b> Hover over column headers to see detailed descriptions."
                "</span></div>",
                unsafe_allow_html=True,
            )

            column_config = {
                "Version": st.column_config.TextColumn(
                    "Version",
                    help="Version with tensor parallelism (TP) configuration",
                ),
                "Peak Output Throughput": st.column_config.TextColumn(
                    "Peak Output Throughput",
                    help="Maximum output tokens/sec achieved. Shows peak concurrency for M1 vs M2 (e.g. peak@200 vs 100).",
                ),
                "Output Throughput (Geometric Mean)": st.column_config.TextColumn(
                    "Output Throughput (Geometric Mean)",
                    help="Geometric mean of output tok/sec across selected concurrency levels",
                ),
                "Total Throughput (Geometric Mean)": st.column_config.TextColumn(
                    "Total Throughput (Geometric Mean)",
                    help="Geometric mean of total (input + output) tok/sec across selected concurrency levels",
                ),
                "End-to-End Latency (Geometric Mean)": st.column_config.TextColumn(
                    "End-to-End Latency (Geometric Mean)",
                    help="Geometric mean of request latency median across selected concurrency levels",
                ),
                "TTFT P95 (Geometric Mean)": st.column_config.TextColumn(
                    "TTFT P95 (Geometric Mean)",
                    help="Geometric mean of Time-to-First-Token (P95) across selected concurrency levels",
                ),
                "ITL P95 (Geometric Mean)": st.column_config.TextColumn(
                    "ITL P95 (Geometric Mean)",
                    help="Geometric mean of Inter-Token Latency (P95) across selected concurrency levels",
                ),
            }

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

            st.markdown("---")
            st.markdown(
                f"**Legend:** "
                f"🟢 {model_1_short} performs better than {model_2_short} | "
                f"🔴 {model_1_short} performs worse than {model_2_short} | "
                f"🟡 Similar Performance (< 5% difference)"
            )

            # Detailed version+TP comparison sections
            st.markdown("---")
            st.markdown("### 📋 Detailed Version Comparisons")
            st.markdown("*Click on a version to see detailed metrics comparison*")

            for idx, (version, tp) in enumerate(common_version_tp, 1):
                v_m1_data = df_m1[(df_m1["version"] == version) & (df_m1["TP"] == tp)]
                v_m2_data = df_m2[(df_m2["version"] == version) & (df_m2["TP"] == tp)]

                tp_val = int(tp) if pd.notna(tp) else "N/A"

                m1_concurrencies = set(
                    v_m1_data["intended concurrency"].dropna().unique()
                )
                m2_concurrencies = set(
                    v_m2_data["intended concurrency"].dropna().unique()
                )
                common_conc = m1_concurrencies.intersection(m2_concurrencies)

                if not common_conc:
                    continue

                m1_common = v_m1_data[
                    v_m1_data["intended concurrency"].isin(common_conc)
                ]
                m2_common = v_m2_data[
                    v_m2_data["intended concurrency"].isin(common_conc)
                ]

                m1_peak_idx = m1_common["output_tok/sec"].idxmax()
                m2_peak_idx = m2_common["output_tok/sec"].idxmax()

                m1_peak_throughput = m1_common.loc[m1_peak_idx, "output_tok/sec"]
                m2_peak_throughput = m2_common.loc[m2_peak_idx, "output_tok/sec"]
                m1_peak_conc = int(m1_common.loc[m1_peak_idx, "intended concurrency"])
                m2_peak_conc = int(m2_common.loc[m2_peak_idx, "intended concurrency"])

                m1_total_throughput = m1_common.loc[m1_peak_idx, "total_tok/sec"]
                m2_total_throughput = m2_common.loc[m2_peak_idx, "total_tok/sec"]

                m1_e2e_latency = m1_common.loc[m1_peak_idx, "request_latency_median"]
                m2_e2e_latency = m2_common.loc[m2_peak_idx, "request_latency_median"]

                m1_ttft = m1_common.loc[m1_peak_idx, "ttft_p95"]
                m2_ttft = m2_common.loc[m2_peak_idx, "ttft_p95"]

                m1_itl = m1_common.loc[m1_peak_idx, "itl_p95"]
                m2_itl = m2_common.loc[m2_peak_idx, "itl_p95"]

                def format_value(val, unit="", decimals=0, round_up=False):
                    """Format a numeric value with optional unit."""
                    if pd.isna(val):
                        return "N/A"
                    if round_up:
                        import math

                        if decimals == 0:
                            return f"~{int(math.ceil(val)):,}{unit}"
                        else:
                            factor = 10**decimals
                            rounded_val = math.ceil(val * factor) / factor
                            return f"~{rounded_val:,.{decimals}f}{unit}"
                    if decimals == 0:
                        return f"~{int(val):,}{unit}"
                    return f"~{val:,.{decimals}f}{unit}"

                def get_winner_text(m1_val, m2_val, higher_is_better, metric_name):
                    """Generate winner text for a metric."""
                    if pd.isna(m1_val) or pd.isna(m2_val) or m2_val == 0:
                        return "N/A"

                    pct_diff = ((m1_val - m2_val) / m2_val) * 100

                    if higher_is_better:
                        if pct_diff > 5:
                            return f"{model_1_short} has +{abs(pct_diff):.1f}% higher {metric_name}"
                        elif pct_diff < -5:
                            return f"{model_2_short} has +{abs(pct_diff):.1f}% higher {metric_name}"
                        else:
                            return f"Similar (~{abs(pct_diff):.1f}% difference)"
                    else:
                        if pct_diff < -5:
                            return f"{model_1_short} has {abs(pct_diff):.1f}% lower {metric_name}"
                        elif pct_diff > 5:
                            return f"{model_2_short} has {abs(pct_diff):.1f}% lower {metric_name}"
                        else:
                            return f"Similar (~{abs(pct_diff):.1f}% difference)"

                with st.expander(f"{idx}. {version} (TP={tp_val})"):
                    detail_rows = [
                        {
                            "Metric": "Peak Output Throughput (output tok/s)",
                            model_1_short: f"{format_value(m1_peak_throughput)} tok/s at {m1_peak_conc} concurrent users",
                            model_2_short: f"{format_value(m2_peak_throughput)} tok/s at {m2_peak_conc} concurrent users",
                            "Difference/Winner": get_winner_text(
                                m1_peak_throughput,
                                m2_peak_throughput,
                                True,
                                "peak output throughput",
                            ),
                        },
                        {
                            "Metric": "Total Throughput (input + output tok/s)",
                            model_1_short: f"{format_value(m1_total_throughput)} tok/s at {m1_peak_conc} concurrent users",
                            model_2_short: f"{format_value(m2_total_throughput)} tok/s at {m2_peak_conc} concurrent users",
                            "Difference/Winner": get_winner_text(
                                m1_total_throughput,
                                m2_total_throughput,
                                True,
                                "total throughput",
                            ),
                        },
                        {
                            "Metric": "Median E2E Latency at Peak Throughput",
                            model_1_short: f"{format_value(m1_e2e_latency, 's', 0, round_up=True)}",
                            model_2_short: f"{format_value(m2_e2e_latency, 's', 0, round_up=True)}",
                            "Difference/Winner": get_winner_text(
                                m1_e2e_latency, m2_e2e_latency, False, "E2E latency"
                            ),
                        },
                        {
                            "Metric": "TTFT P95 at Peak Throughput",
                            model_1_short: f"{format_value(m1_ttft / 1000, 's', 2, round_up=True)}"
                            if pd.notna(m1_ttft)
                            else "N/A",
                            model_2_short: f"{format_value(m2_ttft / 1000, 's', 2, round_up=True)}"
                            if pd.notna(m2_ttft)
                            else "N/A",
                            "Difference/Winner": get_winner_text(
                                m1_ttft, m2_ttft, False, "P95 TTFT"
                            ),
                        },
                        {
                            "Metric": "ITL P95 at Peak Throughput",
                            model_1_short: f"{format_value(m1_itl, 'ms', 0, round_up=True)}",
                            model_2_short: f"{format_value(m2_itl, 'ms', 0, round_up=True)}",
                            "Difference/Winner": get_winner_text(
                                m1_itl, m2_itl, False, "P95 ITL"
                            ),
                        },
                    ]

                    detail_df = pd.DataFrame(detail_rows)
                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("No comparison data available for the selected filters.")


def render_model_performance_comparison_section(filtered_df, accelerator_color_map):
    """🏆 Model Performance Comparison Section - Complete functionality with SLO analysis from original."""
    if "model_comparison_expanded" not in st.session_state:
        st.session_state.model_comparison_expanded = False

    with st.expander(
        "🏆 Model Performance Comparison",
        expanded=st.session_state.model_comparison_expanded,
    ):
        st.markdown(
            "💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view."
        )
        st.markdown("")

        # Get available concurrency levels from the data
        available_concurrencies = sorted(
            filtered_df["intended concurrency"].dropna().unique().tolist()
        )

        if not available_concurrencies:
            st.warning("⚠️ No concurrency data available in the selected filters.")
            return

        # Let user select concurrency level
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                "**Select Concurrency Level**: Compare models at the same concurrency for fair comparison."
            )
        with col2:
            selected_concurrency = st.selectbox(
                "Concurrency",
                options=available_concurrencies,
                index=(
                    available_concurrencies.index(100)
                    if 100 in available_concurrencies
                    else 0
                ),
                key="model_comparison_concurrency",
                on_change=keep_expander_open,
                args=("model_comparison_expanded",),
            )

        st.markdown("")

        def get_performance_at_fixed_concurrency(group, target_concurrency):
            """Get performance metrics at a fixed concurrency level for fair comparison."""
            # Filter to only the target concurrency level
            concurrency_filtered = group[
                group["intended concurrency"] == target_concurrency
            ]

            if concurrency_filtered.empty:
                return pd.Series(
                    {
                        "output_tok/sec": np.nan,
                        "throughput_version": "No Data",
                        "throughput_tp": np.nan,
                        "ttft_p95": np.nan,
                        "ttft_version": "No Data",
                        "ttft_tp": np.nan,
                        "itl_p95": np.nan,
                        "itl_version": "No Data",
                        "itl_tp": np.nan,
                        "efficiency_ratio": np.nan,
                        "efficiency_version": "No Data",
                        "efficiency_tp": np.nan,
                        "error_rate": np.nan,
                        "concurrency_level": target_concurrency,
                    }
                )

            # Find best performance at the target concurrency level
            max_throughput_idx = concurrency_filtered["output_tok/sec"].idxmax()
            max_throughput_row = concurrency_filtered.loc[max_throughput_idx]

            min_ttft_idx = concurrency_filtered["ttft_p95"].idxmin()
            min_ttft_row = concurrency_filtered.loc[min_ttft_idx]

            min_itl_idx = concurrency_filtered["itl_p95"].idxmin()
            min_itl_row = concurrency_filtered.loc[min_itl_idx]

            max_efficiency_idx = concurrency_filtered["efficiency_ratio"].idxmax()
            max_efficiency_row = concurrency_filtered.loc[max_efficiency_idx]

            return pd.Series(
                {
                    "output_tok/sec": max_throughput_row["output_tok/sec"],
                    "throughput_version": max_throughput_row["version"],
                    "throughput_tp": max_throughput_row["TP"],
                    "ttft_p95": min_ttft_row["ttft_p95"],
                    "ttft_version": min_ttft_row["version"],
                    "ttft_tp": min_ttft_row["TP"],
                    "itl_p95": min_itl_row["itl_p95"],
                    "itl_version": min_itl_row["version"],
                    "itl_tp": min_itl_row["TP"],
                    "efficiency_ratio": max_efficiency_row["efficiency_ratio"],
                    "efficiency_version": max_efficiency_row["version"],
                    "efficiency_tp": max_efficiency_row["TP"],
                    "error_rate": concurrency_filtered["error_rate"].mean(),
                    "concurrency_level": target_concurrency,
                }
            )

        def get_optimal_concurrency_performance(
            group,
            itl_threshold=50,
            ttft_threshold=2000,
            percentile_suffix="p95",
            debug_info=None,
        ):
            """Find the best concurrency level that meets PSAP latency SLOs and return performance at that level."""
            itl_col = f"itl_{percentile_suffix}"
            ttft_col = f"ttft_{percentile_suffix}"

            # Filter data that meets latency constraints
            if itl_col in group.columns and ttft_col in group.columns:
                itl_compliant = group[group[itl_col] <= itl_threshold]
                ttft_compliant = group[group[ttft_col] <= ttft_threshold]
                slo_compliant = group[
                    (group[itl_col] <= itl_threshold)
                    & (group[ttft_col] <= ttft_threshold)
                ]

                if debug_info is not None:
                    model_name = group["model"].iloc[0] if len(group) > 0 else "Unknown"
                    accelerator_name = (
                        group["accelerator"].iloc[0] if len(group) > 0 else "Unknown"
                    )
                    debug_info.append(
                        {
                            "model": model_name,
                            "accelerator": accelerator_name,
                            "total_configs": len(group),
                            "itl_compliant_configs": len(itl_compliant),
                            "ttft_compliant_configs": len(ttft_compliant),
                            "both_compliant_configs": len(slo_compliant),
                            f"min_{itl_col}": (
                                group[itl_col].min() if len(group) > 0 else np.nan
                            ),
                            f"min_{ttft_col}": (
                                group[ttft_col].min() if len(group) > 0 else np.nan
                            ),
                        }
                    )
            else:
                return pd.Series(
                    {
                        "output_tok/sec": np.nan,
                        "throughput_version": f"No {percentile_suffix} data available",
                        "throughput_tp": np.nan,
                        "ttft_p95": np.nan,
                        "ttft_version": f"No {percentile_suffix} data available",
                        "ttft_tp": np.nan,
                        "itl_p95": np.nan,
                        "itl_version": f"No {percentile_suffix} data available",
                        "itl_tp": np.nan,
                        "efficiency_ratio": np.nan,
                        "efficiency_version": f"No {percentile_suffix} data available",
                        "efficiency_tp": np.nan,
                        "error_rate": np.nan,
                        "optimal_concurrency": np.nan,
                    }
                )

            if slo_compliant.empty:
                return pd.Series(
                    {
                        "output_tok/sec": np.nan,
                        "throughput_version": "No SLO-compliant data",
                        "throughput_tp": np.nan,
                        "ttft_p95": np.nan,
                        "ttft_version": "No SLO-compliant data",
                        "ttft_tp": np.nan,
                        "itl_p95": np.nan,
                        "itl_version": "No SLO-compliant data",
                        "itl_tp": np.nan,
                        "efficiency_ratio": np.nan,
                        "efficiency_version": "No SLO-compliant data",
                        "efficiency_tp": np.nan,
                        "error_rate": np.nan,
                        "optimal_concurrency": np.nan,
                    }
                )

            # Among SLO-compliant data, find the configuration with highest throughput
            max_throughput_idx = slo_compliant["output_tok/sec"].idxmax()
            best_row = slo_compliant.loc[max_throughput_idx]
            optimal_concurrency = best_row["intended concurrency"]

            # Get all data at this optimal concurrency level for comprehensive metrics
            optimal_concurrency_data = group[
                group["intended concurrency"] == optimal_concurrency
            ]

            # Find best metrics at this concurrency level
            if not optimal_concurrency_data.empty:
                max_throughput_idx = optimal_concurrency_data["output_tok/sec"].idxmax()
                max_throughput_row = optimal_concurrency_data.loc[max_throughput_idx]
                min_ttft_idx = optimal_concurrency_data["ttft_p95"].idxmin()
                min_ttft_row = optimal_concurrency_data.loc[min_ttft_idx]
                min_itl_idx = optimal_concurrency_data["itl_p95"].idxmin()
                min_itl_row = optimal_concurrency_data.loc[min_itl_idx]
                max_efficiency_idx = optimal_concurrency_data[
                    "efficiency_ratio"
                ].idxmax()
                max_efficiency_row = optimal_concurrency_data.loc[max_efficiency_idx]
            else:
                # Fallback to using the best_row if optimal_concurrency_data is empty
                max_throughput_row = best_row
                min_ttft_row = best_row
                min_itl_row = best_row
                max_efficiency_row = best_row

            ttft_percentile = (
                best_row[ttft_col]
                if ttft_col in best_row.index and pd.notna(best_row[ttft_col])
                else np.nan
            )
            itl_percentile = (
                best_row[itl_col]
                if itl_col in best_row.index and pd.notna(best_row[itl_col])
                else np.nan
            )

            throughput_value = (
                max_throughput_row["output_tok/sec"]
                if "output_tok/sec" in max_throughput_row.index
                else np.nan
            )
            tp_value = (
                max_throughput_row["TP"] if "TP" in max_throughput_row.index else np.nan
            )
            version_value = (
                max_throughput_row["version"]
                if "version" in max_throughput_row.index
                else "Unknown"
            )

            return pd.Series(
                {
                    "output_tok/sec": throughput_value,
                    "throughput_version": version_value,
                    "throughput_tp": tp_value,
                    "ttft_p95": (
                        min_ttft_row["ttft_p95"]
                        if "ttft_p95" in min_ttft_row.index
                        else np.nan
                    ),
                    "ttft_version": (
                        min_ttft_row["version"]
                        if "version" in min_ttft_row.index
                        else version_value
                    ),
                    "ttft_tp": (
                        min_ttft_row["TP"] if "TP" in min_ttft_row.index else tp_value
                    ),
                    f"ttft_{percentile_suffix}": ttft_percentile,
                    "itl_p95": (
                        min_itl_row["itl_p95"]
                        if "itl_p95" in min_itl_row.index
                        else np.nan
                    ),
                    "itl_version": (
                        min_itl_row["version"]
                        if "version" in min_itl_row.index
                        else version_value
                    ),
                    "itl_tp": (
                        min_itl_row["TP"] if "TP" in min_itl_row.index else tp_value
                    ),
                    f"itl_{percentile_suffix}": itl_percentile,
                    "efficiency_ratio": (
                        max_efficiency_row["efficiency_ratio"]
                        if "efficiency_ratio" in max_efficiency_row.index
                        else np.nan
                    ),
                    "efficiency_version": (
                        max_efficiency_row["version"]
                        if "version" in max_efficiency_row.index
                        else version_value
                    ),
                    "efficiency_tp": (
                        max_efficiency_row["TP"]
                        if "TP" in max_efficiency_row.index
                        else tp_value
                    ),
                    "error_rate": (
                        optimal_concurrency_data["error_rate"].mean()
                        if not optimal_concurrency_data.empty
                        else np.nan
                    ),
                    "optimal_concurrency": optimal_concurrency,
                }
            )

        model_comparison = (
            filtered_df.groupby(["model", "accelerator"])
            .apply(
                lambda group: get_performance_at_fixed_concurrency(
                    group, selected_concurrency
                )
            )
            .reset_index()
        )

        # Convert TTFT from ms to seconds
        if "ttft_p95" in model_comparison.columns:
            model_comparison["ttft_p95_s"] = model_comparison["ttft_p95"] / 1000

        model_comparison["model_short"] = model_comparison["model"].apply(
            lambda x: x.split("/")[-1] if pd.notna(x) else "Unknown"
        )
        model_comparison["accelerator"] = model_comparison["accelerator"].fillna(
            "Unknown"
        )
        model_comparison["model_accelerator"] = (
            model_comparison["model_short"]
            + " ("
            + model_comparison["accelerator"]
            + ")"
        )

        model_comparison["model_accelerator_version"] = (
            model_comparison["model_short"]
            + " ("
            + model_comparison["accelerator"]
            + ")"
            + " ["
            + model_comparison["throughput_version"]
            + "]"
        )

        if not model_comparison.empty:
            required_cols = [
                "output_tok/sec",
                "ttft_p95_s",
                "itl_p95",
                "efficiency_ratio",
            ]
            existing_cols = [
                col for col in required_cols if col in model_comparison.columns
            ]
            if existing_cols:
                model_comparison = model_comparison.dropna(subset=existing_cols)

        if not model_comparison.empty:
            # Count how many models have actual data (not all NaN)
            models_with_data = model_comparison[
                model_comparison["output_tok/sec"].notna()
            ]
            if len(models_with_data) > 0:
                st.info(
                    f"📊 **Fair Comparison**: All metrics shown are at **Concurrency Level {selected_concurrency}** for apples-to-apples comparison across models and accelerators. "
                    f"When multiple versions are available, the **best performance** across the selected version filters is displayed."
                )
            else:
                st.warning(
                    f"⚠️ No data available at concurrency level {selected_concurrency} for the selected filters. Try a different concurrency level."
                )

        model_comparison = model_comparison.drop_duplicates(
            subset=["model_accelerator"]
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Peak Throughput comparison at fixed concurrency
            fig_throughput = px.bar(
                model_comparison.sort_values("output_tok/sec", ascending=True),
                x="output_tok/sec",
                y="model_accelerator_version",
                color="accelerator",
                color_discrete_map=accelerator_color_map,
                orientation="h",
                title=f"Peak Throughput by Model & Accelerator (at Concurrency {selected_concurrency})<br><sub>Higher is Better ↑</sub>",
                labels={
                    "output_tok/sec": "Peak Output Tokens/sec",
                    "model_accelerator_version": "Model (Accelerator) [Version]",
                },
                template="plotly_white",
                hover_data={"throughput_version": True},
            )
            fig_throughput.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_throughput, use_container_width=True)

        with chart_col2:
            # Best TTFT Latency comparison at fixed concurrency
            fig_latency = px.bar(
                model_comparison.sort_values("ttft_p95_s", ascending=False),
                x="ttft_p95_s",
                y="model_accelerator",
                color="accelerator",
                color_discrete_map=accelerator_color_map,
                orientation="h",
                title=f"Best TTFT P95 Latency by Model & Accelerator (at Concurrency {selected_concurrency})<br><sub>Lower is Better ↓</sub>",
                labels={
                    "ttft_p95_s": "Best TTFT P95 (s)",
                    "model_accelerator": "Model (Accelerator)",
                },
                template="plotly_white",
                hover_data={"ttft_version": True},
            )
            fig_latency.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_latency, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            # Peak Efficiency comparison at fixed concurrency
            fig_efficiency = px.bar(
                model_comparison.sort_values("efficiency_ratio", ascending=True),
                x="efficiency_ratio",
                y="model_accelerator",
                color="accelerator",
                color_discrete_map=accelerator_color_map,
                orientation="h",
                title=f"Peak Efficiency Ratio by Model & Accelerator (at Concurrency {selected_concurrency})<br><sub>Higher is Better ↑</sub>",
                labels={
                    "efficiency_ratio": "Peak Efficiency (Tokens/sec per TP)",
                    "model_accelerator": "Model (Accelerator)",
                },
                template="plotly_white",
                hover_data={"efficiency_version": True},
            )
            fig_efficiency.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_efficiency, use_container_width=True)

        with chart_col4:
            # Best Inter-token latency comparison at fixed concurrency
            fig_itl = px.bar(
                model_comparison.sort_values("itl_p95", ascending=False),
                x="itl_p95",
                y="model_accelerator",
                color="accelerator",
                color_discrete_map=accelerator_color_map,
                orientation="h",
                title=f"Best Inter-Token Latency P95 by Model & Accelerator (at Concurrency {selected_concurrency})<br><sub>Lower is Better ↓</sub>",
                labels={
                    "itl_p95": "Best Inter-Token Latency P95 (ms)",
                    "model_accelerator": "Model (Accelerator)",
                },
                template="plotly_white",
                hover_data={"itl_version": True},
            )
            fig_itl.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_itl, use_container_width=True)


def render_cost_analysis_section(filtered_df, accelerator_color_map):
    """💰 Cost Analysis Section - Complete functionality with cloud pricing calculations from original."""
    with st.expander("💰 Cost Analysis - Cost per Million Tokens", expanded=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                "💡 **Cost Methodology**: Based on PSAP AI Costs Dashboard methodology - throughput performance at optimal concurrency that meets PSAP latency SLOs."
            )
        with col2, st.popover("ℹ️ Formulas"):
            st.markdown(
                """
                **Cost Calculation Formulas:**

                📊 **Time to Million Tokens (TTMT)**
                ```
                TTMT = 1,000,000 tokens ÷ Effective Throughput (tokens/sec)
                ```
                - H200/MI300X: Uses adjusted throughput
                - TPU: Uses raw throughput

                💰 **Cost per Million Tokens (CPMT)**
                ```
                CPMT = (Instance Cost/hour × TTMT) ÷ 3600 seconds/hour
                ```

                **Where:**
                - **Instance Cost/hour**: Cloud provider pricing
                  - H200/MI300X: Pay for full 8-GPU instance regardless of TP
                  - TPU: Per-core pricing, multiplied by TP count
                - **Throughput**:
                  - H200/MI300X: Adjusted throughput (Raw Throughput × 8 GPUs / TP)
                  - TPU: Raw throughput (you pay per core used)
                - **Optimal Concurrency**: Best concurrency meeting PSAP SLOs
                """
            )

        st.info(
            "💡 **Tip**: For the most accurate cost calculations, use the **(512/2k)** ISL/OSL filter, "
            "as it provides more data points and better represents typical workload patterns."
        )

        with st.expander(
            "💰 Cloud Instance Pricing (as of October 20th, 2025)", expanded=False
        ):
            price_col1, price_col2, price_col3 = st.columns(3)

            with price_col1:
                st.markdown(
                    """
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 5px;
                ">
                    <h5 style="margin: 0; color: white; font-size: 18px;">🔷 H200 (NVIDIA)</h5>
                    <div style="font-size: 20px; font-weight: bold; margin: 5px 0;">$63.30/hour</div>
                    <div style="font-size: 15px; opacity: 0.9;">Instance: AWS p5en.48xlarge</div>
                    <div style="font-size: 15px; opacity: 0.8;">Configuration: 8×NVIDIA-H200-144GB</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with price_col2:
                st.markdown(
                    """
                <div style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 5px;
                ">
                    <h5 style="margin: 0; color: white; font-size: 18px;">🔶 MI300X (AMD)</h5>
                    <div style="font-size: 20px; font-weight: bold; margin: 5px 0;">$48.00/hour</div>
                    <div style="font-size: 15px; opacity: 0.9;">Instance: Azure ND96isr MI300X v5</div>
                    <div style="font-size: 15px; opacity: 0.8;">Configuration: 8×AMD-MI300X-192GB</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with price_col3:
                st.markdown(
                    """
                <div style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 5px;
                ">
                    <h5 style="margin: 0; color: white; font-size: 18px;">🔵 TPU Trillium (GCP)</h5>
                    <div style="font-size: 20px; font-weight: bold; margin: 5px 0;">$2.70/hour</div>
                    <div style="font-size: 15px; opacity: 0.9;">TPU Trillium</div>
                    <div style="font-size: 15px; opacity: 0.8;">Per core pricing</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("")
            st.info(
                "💡 **Pricing Note**: Costs shown are on-demand rates and may vary with reserved instances, spot pricing, or volume discounts."
            )

        st.subheader("🎯 Latency Constraints (PSAP Standard)")

        # Percentile selection
        percentile_choice = st.selectbox(
            "Latency Percentile",
            options=["P95", "P99", "P50 (Median)"],
            index=0,
            help="Choose which percentile to use for latency constraints",
        )

        percentile_map = {
            "P95": {"suffix": "p95", "itl_default": 65.0, "ttft_default": 3400.0},
            "P99": {"suffix": "p99", "itl_default": 65.0, "ttft_default": 3400.0},
            "P50 (Median)": {
                "suffix": "median",
                "itl_default": 65.0,
                "ttft_default": 3400.0,
            },
        }

        percentile_info = percentile_map[percentile_choice]
        percentile_label = percentile_choice.split(" ")[0]

        latency_col1, latency_col2 = st.columns(2)

        with latency_col1:
            itl_threshold = st.number_input(
                f"Max ITL {percentile_label} (ms)",
                min_value=10.0,
                value=percentile_info["itl_default"],
                step=5.0,
                help=f"Inter-Token Latency {percentile_label} threshold",
            )

        with latency_col2:
            ttft_threshold = st.number_input(
                f"Max TTFT {percentile_label} (ms)",
                min_value=500.0,
                value=percentile_info["ttft_default"],
                step=100.0,
                help=f"Time To First Token {percentile_label} threshold",
            )

        st.markdown("")

        accelerator_pricing = {
            "H200": {
                "instance_cost_per_hour": 63.30,
                "total_gpus": 8,
                "description": "H200 - AWS p5en.48xlarge ($63.30/hour)",
                "instance_details": "AWS p5en.48xlarge - ondemand (8xNVIDIA-H200-144GB)",
            },
            "MI300X": {
                "instance_cost_per_hour": 48.00,
                "total_gpus": 8,
                "description": "MI300X - Azure ND96isr MI300X v5 ($48.00/hour)",
                "instance_details": "Azure ND96isr MI300X v5 - ondemand (8xAMD-MI300X-192GB)",
            },
            "TPU": {
                "instance_cost_per_hour": 2.70,
                "total_gpus": 1,
                "description": "TPU - Trillium ($2.70/core/hour)",
                "instance_details": "TPU Trillium",
            },
        }

        def calculate_cost_metrics(
            throughput_tokens_per_sec,
            instance_cost_per_hour,
            tp_count,
            model_name,
            accelerator_type,
        ):
            """Calculate cost metrics based on inference costs only - uses full instance cost."""
            if pd.isna(throughput_tokens_per_sec) or throughput_tokens_per_sec <= 0:
                return {
                    "ttmt_seconds": np.nan,
                    "ttmt_minutes": np.nan,
                    "cpmt_inference": np.nan,
                    "cpmt_total": np.nan,
                    "total_instance_cost_per_hour": np.nan,
                }

            # TTMT: Time to Million Tokens (seconds)
            million_tokens = 1_000_000
            ttmt_seconds = million_tokens / throughput_tokens_per_sec
            ttmt_minutes = ttmt_seconds / 60

            # Calculate instance cost based on accelerator type
            if accelerator_type == "TPU":
                # TPU pricing is per core, so multiply by TP count
                total_instance_cost_per_hour = instance_cost_per_hour * tp_count
            else:
                # H200 and MI300X: You pay for the full instance regardless of TP used
                total_instance_cost_per_hour = instance_cost_per_hour

            # Base inference cost
            seconds_per_hour = 3_600
            cpmt_inference = (
                ttmt_seconds * total_instance_cost_per_hour / seconds_per_hour
            )

            cpmt_total = cpmt_inference

            return {
                "ttmt_seconds": ttmt_seconds,
                "ttmt_minutes": ttmt_minutes,
                "cpmt_inference": cpmt_inference,
                "cpmt_total": cpmt_total,
                "total_instance_cost_per_hour": total_instance_cost_per_hour,
            }

        def get_optimal_concurrency_performance(
            group,
            itl_threshold=50,
            ttft_threshold=2000,
            percentile_suffix="p95",
            debug_info=None,
        ):
            """Find the best concurrency level that meets PSAP latency SLOs and return performance at that level."""
            itl_col = f"itl_{percentile_suffix}"
            ttft_col = f"ttft_{percentile_suffix}"

            # Filter data that meets latency constraints
            if itl_col in group.columns and ttft_col in group.columns:
                # Check individual constraints for debugging
                itl_compliant = group[group[itl_col] <= itl_threshold]
                ttft_compliant = group[group[ttft_col] <= ttft_threshold]
                slo_compliant = group[
                    (group[itl_col] <= itl_threshold)
                    & (group[ttft_col] <= ttft_threshold)
                ]

                if debug_info is not None:
                    model_name = group["model"].iloc[0] if len(group) > 0 else "Unknown"
                    accelerator_name = (
                        group["accelerator"].iloc[0] if len(group) > 0 else "Unknown"
                    )
                    debug_info.append(
                        {
                            "model": model_name,
                            "accelerator": accelerator_name,
                            "total_configs": len(group),
                            "itl_compliant_configs": len(itl_compliant),
                            "ttft_compliant_configs": len(ttft_compliant),
                            "both_compliant_configs": len(slo_compliant),
                            f"min_{itl_col}": (
                                group[itl_col].min() if len(group) > 0 else np.nan
                            ),
                            f"min_{ttft_col}": (
                                group[ttft_col].min() if len(group) > 0 else np.nan
                            ),
                        }
                    )
            else:
                return pd.Series(
                    {
                        "output_tok/sec": np.nan,
                        "throughput_version": f"No {percentile_suffix} data available",
                        "throughput_tp": np.nan,
                        "ttft_p95": np.nan,
                        "ttft_version": f"No {percentile_suffix} data available",
                        "ttft_tp": np.nan,
                        "itl_p95": np.nan,
                        "itl_version": f"No {percentile_suffix} data available",
                        "itl_tp": np.nan,
                        "efficiency_ratio": np.nan,
                        "efficiency_version": f"No {percentile_suffix} data available",
                        "efficiency_tp": np.nan,
                        "error_rate": np.nan,
                        "optimal_concurrency": np.nan,
                    }
                )

            if slo_compliant.empty:
                return pd.Series(
                    {
                        "output_tok/sec": np.nan,
                        "throughput_version": "No SLO-compliant data",
                        "throughput_tp": np.nan,
                        "ttft_p95": np.nan,
                        "ttft_version": "No SLO-compliant data",
                        "ttft_tp": np.nan,
                        "itl_p95": np.nan,
                        "itl_version": "No SLO-compliant data",
                        "itl_tp": np.nan,
                        "efficiency_ratio": np.nan,
                        "efficiency_version": "No SLO-compliant data",
                        "efficiency_tp": np.nan,
                        "error_rate": np.nan,
                        "optimal_concurrency": np.nan,
                    }
                )

            # Among SLO-compliant data, find the configuration with highest throughput
            max_throughput_idx = slo_compliant["output_tok/sec"].idxmax()
            best_row = slo_compliant.loc[max_throughput_idx]
            optimal_concurrency = best_row["intended concurrency"]

            optimal_concurrency_data = group[
                group["intended concurrency"] == optimal_concurrency
            ]

            # Find best metrics at this concurrency level
            if not optimal_concurrency_data.empty:
                max_throughput_idx = optimal_concurrency_data["output_tok/sec"].idxmax()
                max_throughput_row = optimal_concurrency_data.loc[max_throughput_idx]
                min_ttft_idx = optimal_concurrency_data["ttft_p95"].idxmin()
                min_ttft_row = optimal_concurrency_data.loc[min_ttft_idx]
                min_itl_idx = optimal_concurrency_data["itl_p95"].idxmin()
                min_itl_row = optimal_concurrency_data.loc[min_itl_idx]
                max_efficiency_idx = optimal_concurrency_data[
                    "efficiency_ratio"
                ].idxmax()
                max_efficiency_row = optimal_concurrency_data.loc[max_efficiency_idx]
            else:
                # Fallback to using the best_row if optimal_concurrency_data is empty
                max_throughput_row = best_row
                min_ttft_row = best_row
                min_itl_row = best_row
                max_efficiency_row = best_row

            ttft_percentile = (
                best_row[ttft_col]
                if ttft_col in best_row.index and pd.notna(best_row[ttft_col])
                else np.nan
            )
            itl_percentile = (
                best_row[itl_col]
                if itl_col in best_row.index and pd.notna(best_row[itl_col])
                else np.nan
            )

            throughput_value = (
                max_throughput_row["output_tok/sec"]
                if "output_tok/sec" in max_throughput_row.index
                else np.nan
            )
            tp_value = (
                max_throughput_row["TP"] if "TP" in max_throughput_row.index else np.nan
            )
            version_value = (
                max_throughput_row["version"]
                if "version" in max_throughput_row.index
                else "Unknown"
            )

            return pd.Series(
                {
                    "output_tok/sec": throughput_value,
                    "throughput_version": version_value,
                    "throughput_tp": tp_value,
                    "ttft_p95": (
                        min_ttft_row["ttft_p95"]
                        if "ttft_p95" in min_ttft_row.index
                        else np.nan
                    ),
                    "ttft_version": (
                        min_ttft_row["version"]
                        if "version" in min_ttft_row.index
                        else version_value
                    ),
                    "ttft_tp": (
                        min_ttft_row["TP"] if "TP" in min_ttft_row.index else tp_value
                    ),
                    f"ttft_{percentile_suffix}": ttft_percentile,
                    "itl_p95": (
                        min_itl_row["itl_p95"]
                        if "itl_p95" in min_itl_row.index
                        else np.nan
                    ),
                    "itl_version": (
                        min_itl_row["version"]
                        if "version" in min_itl_row.index
                        else version_value
                    ),
                    "itl_tp": (
                        min_itl_row["TP"] if "TP" in min_itl_row.index else tp_value
                    ),
                    f"itl_{percentile_suffix}": itl_percentile,
                    "efficiency_ratio": (
                        max_efficiency_row["efficiency_ratio"]
                        if "efficiency_ratio" in max_efficiency_row.index
                        else np.nan
                    ),
                    "efficiency_version": (
                        max_efficiency_row["version"]
                        if "version" in max_efficiency_row.index
                        else version_value
                    ),
                    "efficiency_tp": (
                        max_efficiency_row["TP"]
                        if "TP" in max_efficiency_row.index
                        else tp_value
                    ),
                    "error_rate": (
                        optimal_concurrency_data["error_rate"].mean()
                        if not optimal_concurrency_data.empty
                        else np.nan
                    ),
                    "optimal_concurrency": optimal_concurrency,
                }
            )

        st.info(
            f"🎯 **Optimal Concurrency Analysis**: Finding best concurrency levels that meet PSAP SLOs (ITL {percentile_label} ≤ {itl_threshold}ms, TTFT {percentile_label} ≤ {ttft_threshold}ms)"
        )

        debug_info = []

        # FIRST PASS: Identify which model/accelerator/TP combinations have SLO-compliant configurations
        slo_analysis_data = []
        model_accelerator_tp_groups = filtered_df.groupby(
            ["model", "accelerator", "TP"]
        )

        for (model, accelerator, tp), group in model_accelerator_tp_groups:
            result_series = get_optimal_concurrency_performance(
                group,
                itl_threshold,
                ttft_threshold,
                percentile_info["suffix"],
                debug_info,
            )

            result_dict = result_series.to_dict()
            result_dict["model"] = model
            result_dict["accelerator"] = accelerator
            result_dict["TP"] = tp

            slo_analysis_data.append(result_dict)

        slo_analysis = pd.DataFrame(slo_analysis_data)

        # Convert TTFT from ms to seconds
        if "ttft_p95" in slo_analysis.columns:
            slo_analysis["ttft_p95_s"] = slo_analysis["ttft_p95"] / 1000

        if "optimal_concurrency" in slo_analysis.columns:
            slo_compliant_models = slo_analysis[
                slo_analysis["optimal_concurrency"].notna()
            ].copy()
        else:
            slo_compliant_models = pd.DataFrame()  # Empty DataFrame as fallback

        # Now apply the cost calculation ONLY to SLO-compliant models
        cost_model_comparison = slo_compliant_models.copy()

        if "model" in cost_model_comparison.columns:
            cost_model_comparison["model_short"] = cost_model_comparison["model"].apply(
                lambda x: x.split("/")[-1] if pd.notna(x) else "Unknown"
            )

        if slo_compliant_models.empty:
            st.error(
                "❌ **No SLO-compliant models found** - Cannot proceed with cost analysis"
            )
            return

        if not cost_model_comparison.empty:
            required_cols_for_cost = ["output_tok/sec"]
            if "optimal_concurrency" in cost_model_comparison.columns:
                required_cols_for_cost.append("optimal_concurrency")
            existing_cols = [
                col
                for col in required_cols_for_cost
                if col in cost_model_comparison.columns
            ]
            if existing_cols:
                cost_model_comparison = cost_model_comparison.dropna(
                    subset=existing_cols
                )

        if not cost_model_comparison.empty:
            cost_analysis_data = []

            for _, row in cost_model_comparison.iterrows():
                accelerator = row["accelerator"]
                model_name = row.get("model_short", "Unknown")

                if accelerator in accelerator_pricing:
                    pricing_info = accelerator_pricing[accelerator]
                    tp_count = row.get("throughput_tp", 1)

                    # Calculate adjusted throughput = throughput × (total GPUs / TP)
                    raw_throughput = row.get("output_tok/sec", 0)
                    total_gpus = pricing_info.get("total_gpus", 1)

                    if accelerator == "TPU":
                        # TPU: Pay per core used, so use raw throughput
                        throughput_for_calc = raw_throughput
                        adjusted_throughput = raw_throughput
                    else:
                        # H200/MI300X: Pay for full instance, so use adjusted throughput
                        adjusted_throughput = (
                            raw_throughput * (total_gpus / tp_count)
                            if tp_count > 0
                            else 0
                        )
                        throughput_for_calc = adjusted_throughput

                    cost_metrics = calculate_cost_metrics(
                        throughput_for_calc,
                        pricing_info["instance_cost_per_hour"],
                        tp_count,
                        model_name,
                        accelerator,
                    )

                    if "optimal_concurrency" in row.index and pd.notna(
                        row.get("optimal_concurrency")
                    ):
                        concurrency_used = row["optimal_concurrency"]
                    else:
                        concurrency_used = 100

                    ttft_percentile_value = row.get(
                        f"ttft_{percentile_info['suffix']}", np.nan
                    )
                    itl_percentile_value = row.get(
                        f"itl_{percentile_info['suffix']}", np.nan
                    )

                    cost_analysis_data.append(
                        {
                            "model": model_name,
                            "accelerator": accelerator,
                            "accelerator_desc": pricing_info["description"],
                            "version": row.get("throughput_version", "Unknown"),
                            "tp": tp_count,
                            "throughput": raw_throughput,
                            "adjusted_throughput": adjusted_throughput,
                            "concurrency_used": concurrency_used,
                            f"ttft_{percentile_info['suffix']}": ttft_percentile_value,
                            f"itl_{percentile_info['suffix']}": itl_percentile_value,
                            "ttmt_minutes": cost_metrics["ttmt_minutes"],
                            "cpmt_inference": cost_metrics["cpmt_inference"],
                            "cpmt_total": cost_metrics["cpmt_total"],
                            "instance_cost_per_hour": pricing_info[
                                "instance_cost_per_hour"
                            ],
                            "total_instance_cost_per_hour": cost_metrics[
                                "total_instance_cost_per_hour"
                            ],
                        }
                    )

            if cost_analysis_data:
                cost_df = pd.DataFrame(cost_analysis_data)
                cost_df = cost_df.dropna(subset=["cpmt_total"])

                if not cost_df.empty:
                    # Create a combined label showing model and TP for better visualization
                    cost_df["model_tp_label"] = cost_df.apply(
                        lambda row: f"{row['model']} (TP={int(row['tp'])})", axis=1
                    )

                    cost_col1, cost_col2 = st.columns(2)

                    with cost_col1:
                        # Time to Million Tokens chart
                        fig_time = px.bar(
                            cost_df.sort_values("ttmt_minutes", ascending=True),
                            x="ttmt_minutes",
                            y="model_tp_label",
                            color="accelerator",
                            color_discrete_map=accelerator_color_map,
                            orientation="h",
                            title="Time to Million Tokens (minutes) - Lower is Better",
                            labels={
                                "ttmt_minutes": "Time to Million Tokens (minutes)",
                                "model_tp_label": "Model (TP Configuration)",
                            },
                            template="plotly_white",
                            hover_data={
                                "version": True,
                                "throughput": ":.1f",
                                "tp": True,
                                "cpmt_total": ":.3f",
                            },
                        )
                        fig_time.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig_time, use_container_width=True)
                        st.caption(
                            "📊 Multiple accelerator types used for results, see 'Formulas' for calculation details. Click legend items to show/hide accelerator types."
                        )

                    with cost_col2:
                        # Cost per Million Tokens chart
                        fig_cost = px.bar(
                            cost_df.sort_values("cpmt_total", ascending=True),
                            x="cpmt_total",
                            y="model_tp_label",
                            color="accelerator",
                            color_discrete_map=accelerator_color_map,
                            orientation="h",
                            title="Cost per Million Tokens (USD) - Lower is Better",
                            labels={
                                "cpmt_total": "Cost per Million Tokens (USD)",
                                "model_tp_label": "Model (TP Configuration)",
                            },
                            template="plotly_white",
                            hover_data={
                                "version": True,
                                "throughput": ":.1f",
                                "tp": True,
                                "ttmt_minutes": ":.1f",
                            },
                        )
                        fig_cost.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig_cost, use_container_width=True)

                    # Cost efficiency ranking table
                    st.info(
                        "💡 **Tip**: Hover over column headers in the table below to see detailed descriptions of each field."
                    )
                    ranking_col1, ranking_col2 = st.columns([3, 2])
                    with ranking_col1:
                        st.subheader(
                            "📊 Cost Efficiency Ranking (at Optimal Concurrency meeting PSAP SLOs)"
                        )
                    with ranking_col2:
                        help_col1, help_col2 = st.columns(2)
                        with help_col1:
                            with st.popover("ℹ️ Column Help"):
                                st.markdown(
                                    f"""
                                **Column Explanations:**

                                 **Rank**: Sorted by lowest cost (best value)

                                 **Model**: AI model name

                                 **Accelerator**: Hardware and cloud instance details

                                 **Version**: Inference server version used

                                 **TP**: Tensor Parallelism (number of GPUs)

                                 **Concurrency**: Optimal concurrent requests

                                 **Throughput**: Output tokens generated per second

                                 **Adjusted Throughput**:
                                 - H200/MI300X: Raw Throughput × (8 GPUs / TP)
                                 - TPU: Raw throughput (pay per core used)

                                 **TTFT {percentile_label}**: Time to First Token ({percentile_choice.lower()})

                                 **ITL {percentile_label}**: Inter-Token Latency ({percentile_choice.lower()})

                                 **Instance Cost**: Full cloud instance hourly cost

                                ⏱ **Time to 1M Tokens**: Minutes to generate 1 million tokens

                                 **Total Cost per 1M Tokens**: Final cost comparison metric
                                """
                                )
                        with help_col2:
                            with st.popover("ℹ️ Formulas"):
                                st.markdown(
                                    """
                                **Cost Calculation Formulas:**

                                📊 **Time to Million Tokens (TTMT)**
                                ```
                                TTMT = 1,000,000 tokens ÷ Effective Throughput (tokens/sec)
                                ```
                                - H200/MI300X: Uses adjusted throughput
                                - TPU: Uses raw throughput

                                💰 **Cost per Million Tokens (CPMT)**
                                ```
                                CPMT = (Instance Cost/hour × TTMT) ÷ 3600 seconds/hour
                                ```

                                **Where:**
                                - **Instance Cost/hour**: Cloud provider pricing
                                  - H200/MI300X: Pay for full 8-GPU instance regardless of TP
                                  - TPU: Per-core pricing, multiplied by TP count
                                - **Throughput**:
                                  - H200/MI300X: Adjusted throughput (Raw Throughput × 8 GPUs / TP)
                                  - TPU: Raw throughput (you pay per core used)
                                - **Optimal Concurrency**: Best concurrency meeting PSAP SLOs
                                """
                                )

                    cost_display_df = cost_df.copy()
                    cost_display_df = cost_display_df.sort_values(
                        "cpmt_total", ascending=True
                    )
                    cost_display_df.reset_index(drop=True, inplace=True)
                    cost_display_df.insert(
                        0, "Rank", range(1, len(cost_display_df) + 1)
                    )

                    cost_display_df["Model"] = cost_display_df["model"]
                    cost_display_df["Accelerator"] = cost_display_df["accelerator_desc"]
                    cost_display_df["Version"] = cost_display_df["version"]
                    cost_display_df["TP"] = cost_display_df["tp"].astype(int)
                    cost_display_df["Concurrency"] = cost_display_df[
                        "concurrency_used"
                    ].apply(
                        lambda x: f"{int(x)}" if pd.notna(x) and x != "N/A" else "N/A"
                    )
                    cost_display_df["Throughput (tok/s)"] = cost_display_df[
                        "throughput"
                    ].round(1)
                    cost_display_df["Adjusted Throughput (tok/s)"] = cost_display_df[
                        "adjusted_throughput"
                    ].round(1)

                    ttft_col_name = f"TTFT {percentile_label} (ms)"
                    itl_col_name = f"ITL {percentile_label} (ms)"
                    ttft_data_col = f"ttft_{percentile_info['suffix']}"
                    itl_data_col = f"itl_{percentile_info['suffix']}"

                    cost_display_df[ttft_col_name] = cost_display_df[
                        ttft_data_col
                    ].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                    cost_display_df[itl_col_name] = cost_display_df[itl_data_col].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                    )
                    cost_display_df["Instance Cost ($/hour)"] = cost_display_df[
                        "total_instance_cost_per_hour"
                    ].round(1)
                    cost_display_df["Time to 1M Tokens (min)"] = cost_display_df[
                        "ttmt_minutes"
                    ].round(1)
                    cost_display_df["Total Cost per 1M Tokens ($)"] = cost_display_df[
                        "cpmt_total"
                    ].round(3)

                    display_cols = [
                        "Rank",
                        "Model",
                        "Accelerator",
                        "Version",
                        "TP",
                        "Concurrency",
                        "Throughput (tok/s)",
                        "Adjusted Throughput (tok/s)",
                        ttft_col_name,
                        itl_col_name,
                        "Instance Cost ($/hour)",
                        "Time to 1M Tokens (min)",
                        "Total Cost per 1M Tokens ($)",
                    ]

                    # Define column configurations with help text
                    cost_column_config = {
                        "Rank": st.column_config.NumberColumn(
                            "Rank",
                            help="Cost efficiency ranking - lower rank means better value (sorted by lowest cost per 1M tokens)",
                        ),
                        "Model": st.column_config.TextColumn(
                            "Model", help="AI model name being benchmarked"
                        ),
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator",
                            help="Hardware accelerator type and cloud instance details (e.g., H200, MI300X, TPU)",
                        ),
                        "Version": st.column_config.TextColumn(
                            "Version",
                            help="Inference server version used (e.g., RHAIIS-3.2.1, vLLM-0.10.0)",
                        ),
                        "TP": st.column_config.NumberColumn(
                            "TP",
                            help="Tensor Parallelism size - number of GPUs/cores used to split the model",
                        ),
                        "Concurrency": st.column_config.TextColumn(
                            "Concurrency",
                            help="Optimal concurrency level - number of parallel requests that achieves best throughput while meeting PSAP latency SLOs",
                        ),
                        "Throughput (tok/s)": st.column_config.NumberColumn(
                            "Throughput (tok/s)",
                            help="Raw output tokens per second generated at optimal concurrency (higher is better)",
                            format="%.1f",
                        ),
                        "Adjusted Throughput (tok/s)": st.column_config.NumberColumn(
                            "Adjusted Throughput (tok/s)",
                            help="Effective throughput for cost calculations: H200/MI300X = Raw × (8 GPUs / TP) since you pay for full instance; TPU = Raw throughput since you pay per core used",
                            format="%.1f",
                        ),
                        ttft_col_name: st.column_config.TextColumn(
                            ttft_col_name,
                            help=f"Time to First Token {percentile_label} - latency until first token is generated at optimal concurrency (lower is better). PSAP SLO: ≤ {ttft_threshold}ms",
                        ),
                        itl_col_name: st.column_config.TextColumn(
                            itl_col_name,
                            help=f"Inter-Token Latency {percentile_label} - time between consecutive tokens at optimal concurrency (lower is better). PSAP SLO: ≤ {itl_threshold}ms",
                        ),
                        "Instance Cost ($/hour)": st.column_config.NumberColumn(
                            "Instance Cost ($/hour)",
                            help="Cloud instance hourly cost: H200/MI300X pay for full 8-GPU instance regardless of TP; TPU pays per core used (TP × per-core cost)",
                            format="%.1f",
                        ),
                        "Time to 1M Tokens (min)": st.column_config.NumberColumn(
                            "Time to 1M Tokens (min)",
                            help="Time required to generate 1 million tokens = 1,000,000 ÷ Adjusted Throughput ÷ 60 (lower is faster)",
                            format="%.1f",
                        ),
                        "Total Cost per 1M Tokens ($)": st.column_config.NumberColumn(
                            "Total Cost per 1M Tokens ($)",
                            help="Final cost efficiency metric = (Instance Cost/hour × Time to 1M Tokens in hours). Lower is more cost-efficient ⭐",
                            format="%.3f",
                        ),
                    }

                    st.dataframe(
                        cost_display_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config=cost_column_config,
                    )

                    st.info(
                        f"💡 **Performance Details**: **Concurrency** shows the optimal concurrency level where each model achieves best throughput while meeting PSAP SLOs. **TTFT {percentile_label}** and **ITL {percentile_label}** show the actual latency values achieved at this optimal concurrency, confirming SLO compliance (ITL {percentile_label} ≤ {itl_threshold}ms, TTFT {percentile_label} ≤ {ttft_threshold}ms)."
                    )

                    st.subheader("💡 Cost Insights")

                    insight_col1, insight_col2, insight_col3 = st.columns(3)

                    with insight_col1:
                        most_cost_efficient = cost_df.loc[
                            cost_df["cpmt_total"].idxmin()
                        ]
                        st.markdown(
                            create_kpi_card(
                                "🏆 Most Cost Efficient",
                                most_cost_efficient["cpmt_total"],
                                f"{most_cost_efficient['model']} ({most_cost_efficient['accelerator']})",
                                lambda x: f"${x:.3f}/1M tokens",
                            ),
                            unsafe_allow_html=True,
                        )

                    with insight_col2:
                        fastest_generation = cost_df.loc[
                            cost_df["ttmt_minutes"].idxmin()
                        ]
                        st.markdown(
                            create_kpi_card(
                                "⚡ Fastest Generation",
                                fastest_generation["ttmt_minutes"],
                                f"{fastest_generation['model']} ({fastest_generation['accelerator']})",
                                lambda x: f"{x:.1f} min/1M tokens",
                            ),
                            unsafe_allow_html=True,
                        )

                    with insight_col3:
                        overall_max_cost_row = cost_df.loc[
                            cost_df["cpmt_total"].idxmax()
                        ]
                        overall_min_cost_row = cost_df.loc[
                            cost_df["cpmt_total"].idxmin()
                        ]
                        overall_range = (
                            overall_max_cost_row["cpmt_total"]
                            - overall_min_cost_row["cpmt_total"]
                        )

                        range_subtitle = (
                            f"Max: {overall_max_cost_row['model']} (TP={int(overall_max_cost_row['tp'])}) | "
                            f"Min: {overall_min_cost_row['model']} (TP={int(overall_min_cost_row['tp'])})"
                        )

                        card_col1, card_col2 = st.columns([4, 1])

                        with card_col1:
                            st.markdown(
                                create_kpi_card(
                                    "💸 Cost Range",
                                    overall_range,
                                    range_subtitle,
                                    lambda x: f"${x:.3f} spread",
                                ),
                                unsafe_allow_html=True,
                            )

                        with card_col2:
                            with st.popover("📊"):
                                st.markdown("**Cost Range by Accelerator:**")
                                accelerators = ["H200", "MI300X", "TPU"]
                                for acc in accelerators:
                                    acc_data = cost_df[
                                        cost_df["accelerator"].str.contains(
                                            acc, case=False, na=False
                                        )
                                    ]
                                    if not acc_data.empty:
                                        max_cost_row = acc_data.loc[
                                            acc_data["cpmt_total"].idxmax()
                                        ]
                                        min_cost_row = acc_data.loc[
                                            acc_data["cpmt_total"].idxmin()
                                        ]
                                        cost_range = (
                                            max_cost_row["cpmt_total"]
                                            - min_cost_row["cpmt_total"]
                                        )

                                        st.markdown(
                                            f"""
                                        **{acc}: ${cost_range:.3f} range**

                                        🔴 **Max:** ${max_cost_row["cpmt_total"]:.3f}
                                        *{max_cost_row["model"]}* (TP={int(max_cost_row["tp"])}, v{max_cost_row["version"]})

                                        🟢 **Min:** ${min_cost_row["cpmt_total"]:.3f}
                                        *{min_cost_row["model"]}* (TP={int(min_cost_row["tp"])}, v{min_cost_row["version"]})
                                        """
                                        )
                                        if acc != "TPU":
                                            st.markdown("---")

                        st.caption(
                            "📊 Click button above for cost breakdown by accelerator type"
                        )

                else:
                    st.warning(
                        "⚠️ No valid cost data available for the current selections."
                    )
            else:
                st.warning(
                    "⚠️ No accelerator pricing data available for the current selections."
                )
        else:
            st.warning("⚠️ No performance data available for cost calculations.")


def render_energy_carbon_methodology_section(full_df):
    """🌱 Energy / Carbon Computation  Section for GPU Services.

    Args:
        full_df: Full DataFrame with all benchmark data (filters are applied independently in this section)
    """
    # GPU power mapping (kW) - average inference power (fallback values)
    # Note: Actual measured values are in GPU_POWER_DATA below for RHAIIS 3.2.5
    GPU_POWER_MAP = {
        "H200": 0.475,
        "MI300X": 0.525,
    }

    # Default power for unknown accelerators
    DEFAULT_GPU_POWER = 0.400

    # Detailed GPU power consumption data from Thanos/Grafana Prometheus queries
    # Structure: accelerator -> version -> profile -> model -> power_watts (or dict with TP-specific values)
    # Power values are in Watts (W), convert to kW when using
    GPU_POWER_DATA = {
        "H200": {
            "RHAIIS-3.2.5": {
                # Profile A: Balanced (1k/1k) and Profile B: Variable Workload (512/2k)
                "Profile A: Balanced (1k/1k)": {
                    "deepseek-ai/DeepSeek-R1-0528": 652.18,
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 531.01,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 547.76,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 394.5,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 410.34,
                    "meta-llama/Llama-3.3-70B-Instruct": 603.45,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 450.49,
                    "openai/gpt-oss-120b": {4: 429.65, 1: 603.58},  # TP-specific values
                },
                "Profile B: Variable Workload (512/2k)": {
                    "deepseek-ai/DeepSeek-R1-0528": 652.18,
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 531.01,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 547.76,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 394.5,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 410.34,
                    "meta-llama/Llama-3.3-70B-Instruct": 603.45,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 450.49,
                    "openai/gpt-oss-120b": {4: 429.65, 1: 603.58},  # TP-specific values
                },
                # Profile E: Prefill Heavy (8k/1k)
                "Profile E: Prefill Heavy (8k/1k)": {
                    "deepseek-ai/DeepSeek-R1-0528": 652.18,
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 555.96,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 629.71,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 450.49,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 410.34,
                    "meta-llama/Llama-3.3-70B-Instruct": 603.45,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 450.49,
                    "openai/gpt-oss-120b": {4: 429.65, 1: 603.58},  # TP-specific values
                },
            },
        },
        "MI300X": {
            "RHAIIS-3.2.5": {
                # Profile A: Balanced (1k/1k) and Profile B: Variable Workload (512/2k)
                "Profile A: Balanced (1k/1k)": {
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 339.41,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 431.71,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 339.11,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 375.16,
                    "deepseek-ai/DeepSeek-R1-0528": 393.71,
                    "meta-llama/Llama-3.3-70B-Instruct": 162.9,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 355.97,
                },
                "Profile B: Variable Workload (512/2k)": {
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 339.41,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 431.71,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 339.11,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 375.16,
                    "deepseek-ai/DeepSeek-R1-0528": 393.71,
                    "meta-llama/Llama-3.3-70B-Instruct": 162.9,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 355.97,
                },
                # Profile E: Prefill Heavy (8k/1k)
                "Profile E: Prefill Heavy (8k/1k)": {
                    "Qwen/Qwen3-235B-A22B-Instruct-2507": 404.65,
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": 437.76,
                    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8": 375.08,
                    "RedHatAI/Qwen3-235B-A22B-FP8-dynamic": 390.74,
                    "deepseek-ai/DeepSeek-R1-0528": 399.13,
                    "meta-llama/Llama-3.3-70B-Instruct": 435.14,
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 390.4,
                },
            },
        },
    }

    def get_gpu_power(accelerator, model, version, profile, tp):
        """Look up GPU power consumption from measured data.

        Returns power in kW. Falls back to GPU_POWER_MAP if no specific data available.

        Args:
            accelerator: GPU type (H200, MI300X, etc.)
            model: Full model name (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            version: RHAIIS version (e.g., "RHAIIS-3.2.5", "RHAIIS-3.2.5-sanity", etc.)
            profile: Profile name (e.g., "Profile A: Balanced (1k/1k)")
            tp: Tensor parallelism value

        Returns:
            Power consumption in kW
        """
        # Normalize version string - map variants like "RHAIIS-3.2.5-sanity" to "RHAIIS-3.2.5"
        normalized_version = version
        if "RHAIIS-3.2.5" in str(version):
            normalized_version = "RHAIIS-3.2.5"

        # Try to find specific power data
        if accelerator in GPU_POWER_DATA:
            acc_data = GPU_POWER_DATA[accelerator]
            if normalized_version in acc_data:
                version_data = acc_data[normalized_version]
                if profile in version_data:
                    profile_data = version_data[profile]
                    if model in profile_data:
                        power_value = profile_data[model]
                        # Handle TP-specific values (dict) vs single values
                        if isinstance(power_value, dict):
                            # Look up by TP value, default to TP=4 if not found
                            power_watts = power_value.get(
                                tp, power_value.get(4, list(power_value.values())[0])
                            )
                        else:
                            power_watts = power_value
                        # Convert Watts to kW
                        return power_watts / 1000.0

        # Fallback to generic GPU_POWER_MAP
        return GPU_POWER_MAP.get(accelerator, DEFAULT_GPU_POWER)

    # Initialize session state for expander
    if "energy_expanded" not in st.session_state:
        st.session_state.energy_expanded = False

    def keep_energy_expander_open():
        """Keep the energy expander open when filters change."""
        st.session_state.energy_expanded = True

    with st.expander(
        "🌱 Energy / Carbon Computation", expanded=st.session_state.energy_expanded
    ):
        st.markdown(
            """
            <p style="font-size: 1.3rem;">
            This section computes energy consumption and carbon footprint for GPU-based inference
            performance benchmarking runs on <strong>RHAIIS 3.2.5</strong> using measured GPU power from Grafana metrics.
            </p>
            """,
            unsafe_allow_html=True,
        )

        # All reference buttons on one line
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 1])

        with btn_col1:
            with st.popover("📐 Formulas & Variables"):
                st.markdown(
                    """
                    **Variables:**
                    | Variable | Description |
                    |----------|-------------|
                    | `n` | Number of GPUs to run model (TP) |
                    | `NR` | Number of replicas |
                    | `PI` | Avg Power per GPU (kW) |
                    | `TH` | Runtime (hours) |

                    ---

                    **Formulas:**

                    Avg Power to run model:
                    ```
                    P_model = PI × n
                    ```

                    GPU Energy (kWh):
                    ```
                    P_GPU = (PI × n × NR) × TH
                    ```

                    Total Energy:
                    ```
                    P_TOTAL = P_GPU + P_CPU + P_STORAGE
                    ```
                    """
                )

        with btn_col2:
            with st.popover("📋 Example"):
                st.markdown(
                    """
                    **Scenario:** granite-13b-instruct-v2 on L40S

                    | Parameter | Symbol | Value |
                    |-----------|--------|-------|
                    | Power per GPU | PI | 0.275 kW |
                    | GPU Count | n | 2 |
                    | Replicas | NR | 2 |
                    | Runtime | TH | 3 hours |

                    ---

                    **Step 1:** Average Power
                    ```
                    P_model = PI × n
                          = 0.275 × 2
                          = 0.550 kW
                    ```

                    **Step 2:** GPU Energy
                    ```
                    P_GPU = P_model × NR × TH
                          = 0.550 × 2 × 3
                          = 3.3 kWh
                    ```
                    """
                )

        with btn_col3:
            with st.popover("⚡ GPU Power Reference"):
                st.markdown(
                    """
                    **How Power Values Are Calculated:**

                    For **RHAIIS 3.2.5** on H200 and MI300X, power values
                    are **measured** from Grafana metrics:
                    - **H200**: DCGM exporter (`DCGM_FI_DEV_POWER_USAGE`)
                    - **MI300X**: ROCm-SMI (`gpu_power_usage`)

                    Values are averaged per GPU across benchmark runs
                    for each model and ISL/OSL profile combination.

                    ---

                    **Reference TDP Values:**

                    | Accelerator | TDP (kW) |
                    |-------------|----------|
                    | H200 | 0.700 |
                    | MI300X | 0.750 |

                    *Actual inference power varies by model and workload,
                    typically 55-85% of TDP.*
                    """
                )

        # Calculate energy for filtered data
        st.markdown("---")
        st.markdown("#### ⚡ Energy Calculation")

        st.info(
            "💡 **Note:** Energy computation uses measured GPU power from Grafana metrics "
            "(DCGM/ROCm-SMI) for **RHAIIS 3.2.5** on H200 and MI300X accelerators."
        )

        # Flag to track if we should show energy calculations
        show_energy_calculations = True
        energy_filtered_df = None

        if full_df is not None and not full_df.empty:
            # Filter to only RHAIIS-3.2.5 data and supported accelerators (H200, MI300X)
            if "version" in full_df.columns and "accelerator" in full_df.columns:
                rhaiis_325_df = full_df[
                    (full_df["version"] == "RHAIIS-3.2.5")
                    & (full_df["accelerator"].isin(["H200", "MI300X"]))
                ]

                if rhaiis_325_df.empty:
                    st.warning(
                        "⚠️ No RHAIIS 3.2.5 data available for H200 or MI300X accelerators in the dataset."
                    )
                    show_energy_calculations = False
                else:
                    # Create independent filters for this section
                    st.markdown("##### Select Filters for Energy Calculation")

                    energy_filter_col1, energy_filter_col2, energy_filter_col3 = (
                        st.columns(3)
                    )

                    # Filter 1: Accelerator
                    with energy_filter_col1:
                        available_accelerators = sorted(
                            rhaiis_325_df["accelerator"].unique().tolist()
                        )
                        selected_energy_accelerators = st.multiselect(
                            "Accelerator(s)",
                            available_accelerators,
                            default=available_accelerators,
                            key="energy_accelerator_filter",
                            help="Select accelerators for energy calculation",
                            on_change=keep_energy_expander_open,
                        )

                    # Filter accelerator first for dependent filters
                    if selected_energy_accelerators:
                        acc_filtered_df = rhaiis_325_df[
                            rhaiis_325_df["accelerator"].isin(
                                selected_energy_accelerators
                            )
                        ]
                    else:
                        acc_filtered_df = rhaiis_325_df

                    # Filter 2: Profile (ISL/OSL) - single select
                    with energy_filter_col2:
                        if "profile" in acc_filtered_df.columns:
                            available_profiles = sorted(
                                acc_filtered_df["profile"].unique().tolist()
                            )
                            # Default to profiles that have measured power data
                            measured_profiles = [
                                "Profile A: Balanced (1k/1k)",
                                "Profile B: Variable Workload (512/2k)",
                                "Profile E: Prefill Heavy (8k/1k)",
                            ]
                            # Find first measured profile available, else use first available
                            default_profile_idx = 0
                            for i, p in enumerate(available_profiles):
                                if p in measured_profiles:
                                    default_profile_idx = i
                                    break

                            selected_energy_profile = st.selectbox(
                                "ISL/OSL Profile",
                                available_profiles,
                                index=default_profile_idx,
                                key="energy_profile_filter",
                                help="Select ISL/OSL profile for energy calculation",
                                on_change=keep_energy_expander_open,
                            )
                        else:
                            selected_energy_profile = None

                    # Filter by profile for model list
                    if selected_energy_profile:
                        profile_filtered_df = acc_filtered_df[
                            acc_filtered_df["profile"] == selected_energy_profile
                        ]
                    else:
                        profile_filtered_df = acc_filtered_df

                    # Filter 3: Model
                    with energy_filter_col3:
                        available_models = sorted(
                            profile_filtered_df["model"].unique().tolist()
                        )
                        # Show short model names in selection
                        model_display = {
                            m: m.split("/")[-1] if "/" in m else m
                            for m in available_models
                        }

                        selected_energy_models = st.multiselect(
                            "Model(s)",
                            available_models,
                            default=available_models,
                            format_func=lambda x: model_display.get(x, x),
                            key="energy_model_filter",
                            help="Select models for energy calculation",
                            on_change=keep_energy_expander_open,
                        )

                    # Apply all filters
                    energy_filtered_df = rhaiis_325_df.copy()

                    if selected_energy_accelerators:
                        energy_filtered_df = energy_filtered_df[
                            energy_filtered_df["accelerator"].isin(
                                selected_energy_accelerators
                            )
                        ]

                    if selected_energy_profile:
                        energy_filtered_df = energy_filtered_df[
                            energy_filtered_df["profile"] == selected_energy_profile
                        ]

                    if selected_energy_models:
                        energy_filtered_df = energy_filtered_df[
                            energy_filtered_df["model"].isin(selected_energy_models)
                        ]

                    if energy_filtered_df.empty:
                        st.warning("⚠️ No data matches the selected filters.")
                        show_energy_calculations = False

                    st.markdown("---")
            else:
                st.warning(
                    "⚠️ Required columns (version, accelerator) not available in the data."
                )
                show_energy_calculations = False
        else:
            st.warning("⚠️ No data available.")
            show_energy_calculations = False

        if (
            show_energy_calculations
            and energy_filtered_df is not None
            and not energy_filtered_df.empty
        ):
            # Group by model, accelerator, TP to calculate energy
            energy_data = []

            # Get unique combinations
            if "intended concurrency" in energy_filtered_df.columns:
                concurrency_col = "intended concurrency"
            elif "measured concurrency" in energy_filtered_df.columns:
                concurrency_col = "measured concurrency"
            else:
                concurrency_col = None

            for (model, accelerator, tp), group in energy_filtered_df.groupby(
                ["model", "accelerator", "TP"]
            ):
                # Count unique concurrency levels
                if concurrency_col:
                    num_concurrencies = group[concurrency_col].nunique()
                else:
                    num_concurrencies = len(group)

                # Calculate runtime: concurrencies × 10 minutes each
                runtime_minutes = num_concurrencies * 10
                runtime_hours = runtime_minutes / 60
                runtime_seconds = runtime_minutes * 60

                # GPU count = TP value
                gpu_count = int(tp)

                # Number of replicas = 1 for RHAIIS
                replicas = 1

                # Get version and profile from the group for power lookup
                version = group["version"].iloc[0] if "version" in group.columns else ""
                profile = group["profile"].iloc[0] if "profile" in group.columns else ""

                # Get GPU power (kW) - use measured data if available, else fallback
                gpu_power = get_gpu_power(accelerator, model, version, profile, int(tp))

                # Calculate energy
                avg_power = gpu_power * gpu_count
                gpu_energy = avg_power * replicas * runtime_hours

                # Calculate average output throughput and total tokens for energy per token
                avg_output_throughput = 0
                total_tokens = 0
                energy_per_1m_tokens = None

                if "output_tok/sec" in group.columns:
                    # Get average throughput across all concurrency levels
                    avg_output_throughput = group["output_tok/sec"].mean()
                    if pd.notna(avg_output_throughput) and avg_output_throughput > 0:
                        # Total tokens = throughput × runtime
                        total_tokens = avg_output_throughput * runtime_seconds
                        # Energy per 1M tokens (in Wh) = (GPU Energy in kWh × 1000) / (total_tokens / 1,000,000)
                        # Simplified: Energy per 1M tokens (Wh) = (GPU Energy × 1e9) / total_tokens
                        if total_tokens > 0:
                            energy_per_1m_tokens = (
                                gpu_energy * 1e9
                            ) / total_tokens  # Wh per 1M tokens

                # Get short model name
                model_short = model.split("/")[-1] if "/" in model else model

                # Check if using measured data (for display purposes)
                # Normalize version for lookup (e.g., "RHAIIS-3.2.5-sanity" -> "RHAIIS-3.2.5")
                normalized_version = (
                    "RHAIIS-3.2.5" if "RHAIIS-3.2.5" in str(version) else version
                )
                using_measured = (
                    accelerator in GPU_POWER_DATA
                    and normalized_version in GPU_POWER_DATA.get(accelerator, {})
                    and profile
                    in GPU_POWER_DATA.get(accelerator, {}).get(normalized_version, {})
                    and model
                    in GPU_POWER_DATA.get(accelerator, {})
                    .get(normalized_version, {})
                    .get(profile, {})
                )

                energy_data.append(
                    {
                        "Model": model_short,
                        "Accelerator": accelerator,
                        "TP (GPUs Used)": gpu_count,
                        "No of Nodes": replicas,
                        "Concurrencies": num_concurrencies,
                        "Benchmark Duration (min)": runtime_minutes,
                        "Benchmark Duration (hrs)": round(runtime_hours, 2),
                        "Avg Throughput (tok/s)": round(avg_output_throughput, 1)
                        if avg_output_throughput > 0
                        else "N/A",
                        "Total Tokens Generated": f"{int(total_tokens):,}"
                        if total_tokens > 0
                        else "N/A",
                        "Power per GPU (kW)": gpu_power,
                        "Total Power Draw (kW)": round(avg_power, 3),
                        "Total Energy Used (kWh)": round(gpu_energy, 3),
                        "Energy/1M Tokens (Wh)": round(energy_per_1m_tokens, 2)
                        if energy_per_1m_tokens
                        else "N/A",
                        "Data Source": "📊 Measured"
                        if using_measured
                        else "📈 Estimated",
                    }
                )

            if energy_data:
                energy_df = pd.DataFrame(energy_data)

                # Sort by energy consumption descending
                energy_df = energy_df.sort_values(
                    "Total Energy Used (kWh)", ascending=False
                )

                # Get highest and lowest energy configs
                max_energy = energy_df["Total Energy Used (kWh)"].max()
                min_energy = energy_df["Total Energy Used (kWh)"].min()
                max_config = energy_df.loc[
                    energy_df["Total Energy Used (kWh)"].idxmax()
                ]
                min_config = energy_df.loc[
                    energy_df["Total Energy Used (kWh)"].idxmin()
                ]

                # Get most efficient config (lowest energy per 1M tokens)
                # Filter out N/A values for energy efficiency comparison
                efficiency_df = energy_df[
                    energy_df["Energy/1M Tokens (Wh)"] != "N/A"
                ].copy()
                best_efficiency_config = None
                best_efficiency_value = None
                if not efficiency_df.empty:
                    efficiency_df["efficiency_numeric"] = pd.to_numeric(
                        efficiency_df["Energy/1M Tokens (Wh)"], errors="coerce"
                    )
                    valid_efficiency = efficiency_df.dropna(
                        subset=["efficiency_numeric"]
                    )
                    if not valid_efficiency.empty:
                        best_efficiency_config = valid_efficiency.loc[
                            valid_efficiency["efficiency_numeric"].idxmin()
                        ]
                        best_efficiency_value = best_efficiency_config[
                            "efficiency_numeric"
                        ]

                # Display meaningful metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric(
                        "⬆️ Highest Energy",
                        f"{max_energy:.3f} kWh",
                        help=f"{max_config['Model']} on {max_config['Accelerator']} (TP={max_config['TP (GPUs Used)']})",
                    )
                    st.caption(
                        f"{max_config['Model'][:20]}... | {max_config['Accelerator']}"
                        if len(max_config["Model"]) > 20
                        else f"{max_config['Model']} | {max_config['Accelerator']}"
                    )
                with metric_col2:
                    st.metric(
                        "⬇️ Lowest Energy",
                        f"{min_energy:.3f} kWh",
                        help=f"{min_config['Model']} on {min_config['Accelerator']} (TP={min_config['TP (GPUs Used)']})",
                    )
                    st.caption(
                        f"{min_config['Model'][:20]}... | {min_config['Accelerator']}"
                        if len(min_config["Model"]) > 20
                        else f"{min_config['Model']} | {min_config['Accelerator']}"
                    )
                with metric_col3:
                    energy_range = max_energy - min_energy
                    st.metric(
                        "📊 Energy Range",
                        f"{energy_range:.3f} kWh",
                        help="Difference between highest and lowest energy configurations",
                    )
                    st.caption(f"{len(energy_df)} configurations")

                with metric_col4:
                    if (
                        best_efficiency_config is not None
                        and best_efficiency_value is not None
                    ):
                        st.metric(
                            "⚡ Best Efficiency",
                            f"{best_efficiency_value:.1f} Wh/1M tok",
                            help=f"Most energy-efficient: {best_efficiency_config['Model']} on {best_efficiency_config['Accelerator']} (TP={best_efficiency_config['TP (GPUs Used)']})",
                        )
                        st.caption(
                            f"{best_efficiency_config['Model'][:15]}... | {best_efficiency_config['Accelerator']}"
                            if len(best_efficiency_config["Model"]) > 15
                            else f"{best_efficiency_config['Model']} | {best_efficiency_config['Accelerator']}"
                        )
                    else:
                        st.metric(
                            "⚡ Best Efficiency",
                            "N/A",
                            help="Throughput data not available to calculate efficiency",
                        )

                st.markdown("##### Detailed Breakdown")
                st.caption(
                    "💡 **Tip:** Hover over column headers to see detailed explanations of each metric."
                )

                # Configure column display
                column_config = {
                    "Model": st.column_config.TextColumn(
                        "Model",
                        width="medium",
                        help="The LLM model being benchmarked (e.g., Llama-3.3-70B, DeepSeek-R1). Short name shown; full path includes organization prefix.",
                    ),
                    "Accelerator": st.column_config.TextColumn(
                        "Accelerator",
                        width="small",
                        help="GPU hardware used for inference. H200 = NVIDIA H200 (700W TDP), MI300X = AMD MI300X (750W TDP).",
                    ),
                    "TP (GPUs Used)": st.column_config.NumberColumn(
                        "TP (GPUs Used)",
                        help="Tensor Parallelism - the number of GPUs the model is distributed across. Higher TP allows larger models but increases power consumption proportionally.",
                        format="%d",
                    ),
                    "No of Nodes": st.column_config.NumberColumn(
                        "No of Nodes",
                        help="Number of server nodes running the model. For RHAIIS single-node deployments, this is always 1.",
                    ),
                    "Concurrencies": st.column_config.NumberColumn(
                        "Concurrencies",
                        help="Number of different concurrent user load levels tested (e.g., 1, 50, 100, 200, 300, 400, 500, 650 users). Each level runs for ~10 minutes.",
                    ),
                    "Benchmark Duration (min)": st.column_config.NumberColumn(
                        "Benchmark Duration (min)",
                        help="Total benchmark duration in minutes. Calculated as: Number of Concurrencies × 10 minutes per level.",
                        format="%d",
                    ),
                    "Benchmark Duration (hrs)": st.column_config.NumberColumn(
                        "Benchmark Duration (hrs)",
                        help="Total benchmark duration converted to hours. Used for energy calculations (kWh = kW × hours).",
                        format="%.2f",
                    ),
                    "Avg Throughput (tok/s)": st.column_config.TextColumn(
                        "Avg Throughput (tok/s)",
                        help="Average output token generation rate across all concurrency levels. Measures how fast the model produces tokens during inference.",
                    ),
                    "Total Tokens Generated": st.column_config.TextColumn(
                        "Total Tokens Generated",
                        help="Estimated total output tokens generated during the benchmark. Formula: Avg Throughput (tok/s) × Runtime (seconds). Used for efficiency calculations.",
                    ),
                    "Power per GPU (kW)": st.column_config.NumberColumn(
                        "Power per GPU (kW)",
                        help="Average power consumption per GPU in kilowatts during inference. Measured from Grafana metrics (DCGM for H200, ROCm-SMI for MI300X) for RHAIIS 3.2.5.",
                        format="%.3f",
                    ),
                    "Total Power Draw (kW)": st.column_config.NumberColumn(
                        "Total Power Draw (kW)",
                        help="Total average power for all GPUs running the model. Formula: Power per GPU × TP (number of GPUs).",
                        format="%.3f",
                    ),
                    "Total Energy Used (kWh)": st.column_config.NumberColumn(
                        "Total Energy Used (kWh)",
                        help="Total GPU energy consumed during the benchmark in kilowatt-hours. Formula: Total Power Draw (kW) × No of Nodes × Benchmark Duration (hrs). Does not include CPU or storage energy.",
                        format="%.3f",
                    ),
                    "Energy/1M Tokens (Wh)": st.column_config.TextColumn(
                        "Energy/1M Tokens (Wh)",
                        help="Energy efficiency metric in Watt-hours per 1 million tokens. Formula: (Total Energy Used in kWh × 1e9) ÷ Total Tokens. LOWER values = MORE efficient. Best metric for comparing energy efficiency across different models and hardware at production scale.",
                    ),
                    "Data Source": st.column_config.TextColumn(
                        "Data Source",
                        help="📊 Measured = Power values from actual Grafana/Prometheus metrics during RHAIIS 3.2.5 benchmarks. 📈 Estimated = Using typical inference power values (fallback when measured data unavailable).",
                        width="small",
                    ),
                }

                st.dataframe(
                    energy_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )

                st.caption(
                    "💡 **Note:** Benchmark Duration calculated as # of Concurrencies × 10 minutes each. "
                    "Power per GPU values are from Grafana metrics (DCGM/ROCm-SMI) for RHAIIS 3.2.5. "
                    "**Energy/1M Tokens** is a normalized efficiency metric for comparing across hardware at production scale."
                )

                # Additional Energy Components below the table
                with st.popover("📝 Additional Energy Components"):
                    st.markdown(
                        """
                        **CPU Energy:**
                        Both deployment types should account for
                        CPU utilization of the server running the GPU.
                        Computed separately and added to total.

                        **Storage Energy:**
                        For dedicated deployments, include model
                        storage energy (e.g., IBM COS).
                        """
                    )
            else:
                st.info(
                    "No RHAIIS 3.2.5 data available for energy calculation with current filter selections."
                )


def render_runtime_configs_section(filtered_df):
    """⚙️ Runtime Server Configs Section - Complete functionality from original."""
    if "runtime_configs_expanded" not in st.session_state:
        st.session_state.runtime_configs_expanded = False

    with st.expander(
        "⚙️ Runtime Server Configs Used",
        expanded=st.session_state.runtime_configs_expanded,
    ):
        if "runtime_args" in filtered_df.columns:
            st.markdown(
                "**Runtime configurations for your current filter selections:**"
            )
            st.info(
                "📊 **Column Legend**: Shows the server runtime arguments used for each Model + Accelerator + Version combination that matches your current filters."
            )

            unique_configs = filtered_df.drop_duplicates(
                subset=["model", "accelerator", "version"]
            )

            if not unique_configs.empty:
                display_runtime_df = unique_configs[
                    ["model", "accelerator", "version", "runtime_args"]
                ].copy()
                display_runtime_df = display_runtime_df.rename(
                    columns={
                        "model": "Model",
                        "accelerator": "Accelerator",
                        "version": "Version",
                        "runtime_args": "Runtime Arguments",
                    }
                )

                # Sort by Version, then Model, then Accelerator
                display_runtime_df = display_runtime_df.sort_values(
                    ["Version", "Model", "Accelerator"]
                )
                display_runtime_df.reset_index(drop=True, inplace=True)
                display_runtime_df.insert(
                    0, "Config #", range(1, len(display_runtime_df) + 1)
                )

                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Total Configurations", len(display_runtime_df))
                with summary_col2:
                    unique_servers = display_runtime_df["Version"].nunique()
                    st.metric("Unique Inference Server versions", unique_servers)
                with summary_col3:
                    unique_models = display_runtime_df["Model"].nunique()
                    st.metric("Unique Models", unique_models)

                df = display_runtime_df.copy()

                row_height = 35
                header_height = 40
                padding = 20
                dynamic_height = min(
                    max(len(df) * row_height + header_height + padding, 150), 600
                )

                st.dataframe(
                    df[
                        [
                            "Config #",
                            "Model",
                            "Accelerator",
                            "Version",
                            "Runtime Arguments",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=dynamic_height,
                    column_config={
                        "Config #": st.column_config.NumberColumn(
                            "Config No", width=80, pinned=True
                        ),
                        "Model": st.column_config.TextColumn(
                            "Model", width=380, pinned=True
                        ),
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator", width=80, pinned=True
                        ),
                        "Version": st.column_config.TextColumn(
                            "Version", width=120, pinned=True
                        ),
                        "Runtime Arguments": st.column_config.TextColumn(
                            "Runtime Args", width=1800
                        ),
                    },
                )

                options = [
                    (
                        i,
                        f"Config {r['Config #']} – {r['Model']} / {r['Accelerator']} / {r['Version']}",
                    )
                    for i, r in df.iterrows()
                ]
                idx = st.selectbox(
                    "Show full runtime args for:",
                    options,
                    format_func=lambda x: x[1],
                    key="runtime_config_selector",
                    on_change=keep_expander_open,
                    args=("runtime_configs_expanded",),
                )[0]

                args = df.loc[idx, "Runtime Arguments"]
                st.code(args, language="bash")

            else:
                st.warning(
                    "No runtime configurations found for the current filter selections."
                )
        else:
            st.error(
                "Runtime arguments column not found in the data. Please ensure the CSV file contains a 'runtime_args' column."
            )


def render_filtered_data_section(filtered_df):
    """📄 Filtered Data Display Section - View only, no download functionality."""
    with st.expander("📄 Filtered Data from the above filters", expanded=False):
        st.info(
            "💡 **Tips**: Hover over column headers to see detailed descriptions of each field."
        )
        display_filtered_df = filtered_df.copy()
        display_filtered_df.reset_index(drop=True, inplace=True)
        display_filtered_df.insert(0, "Row #", range(1, len(display_filtered_df) + 1))

        # Add Run Date column from guidellm_start_time_ms (epoch milliseconds)
        def convert_epoch_to_date(row):
            """Convert epoch milliseconds to date string."""
            start_time = row.get("guidellm_start_time_ms")
            if pd.notna(start_time) and start_time != "":
                try:
                    # Convert milliseconds to seconds and create datetime
                    from datetime import datetime

                    timestamp_sec = int(start_time) / 1000
                    dt = datetime.fromtimestamp(timestamp_sec)
                    return dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    return None
            return None

        display_filtered_df["Run Date"] = display_filtered_df.apply(
            convert_epoch_to_date, axis=1
        )

        # Add Grafana metrics link for rows with timestamp data
        # Dashboard IDs for different accelerators
        # H200 has two dashboards: old (before Jan 1, 2026) and new (Jan 1, 2026 onwards)
        GRAFANA_DASHBOARDS = {
            "H200_OLD": {
                "dashboard_id": "6475e6106c33fe",
                "dashboard_name": "vllm-2b-dcgm-metrics-psap-8xh200-2",
            },
            "H200_NEW": {
                "dashboard_id": "7a3b910e7e827c",
                "dashboard_name": "vllm-2b-dcgm-metrics-psap-rhaiis-h200",
            },
            "MI300X": {
                "dashboard_id": "amd-ods-az-amd-01",
                "dashboard_name": "vllm-2b-rocm-gpu-metrics-ods-az-amd-01",
            },
        }

        # Jan 1, 2026 00:00:00 UTC in milliseconds
        H200_DASHBOARD_CUTOFF_MS = 1767225600000

        def create_grafana_link(row):
            """Create Grafana dashboard link if timestamps are available."""
            start_time = row.get("guidellm_start_time_ms")
            end_time = row.get("guidellm_end_time_ms")
            uuid = row.get("uuid")
            accelerator = row.get("accelerator", "")

            # Only create link if all required fields are present and not NaN
            if (
                pd.notna(start_time)
                and pd.notna(end_time)
                and pd.notna(uuid)
                and start_time != ""
                and end_time != ""
                and uuid != ""
            ):
                # Convert to integers (in case they're floats)
                start_ms = int(start_time)
                end_ms = int(end_time)

                # Determine which dashboard to use based on accelerator and date
                if accelerator == "H200":
                    # Use new dashboard for runs on or after Jan 1, 2026
                    if start_ms >= H200_DASHBOARD_CUTOFF_MS:
                        dashboard_key = "H200_NEW"
                    else:
                        dashboard_key = "H200_OLD"
                elif accelerator in GRAFANA_DASHBOARDS:
                    dashboard_key = accelerator
                else:
                    return None

                dashboard_config = GRAFANA_DASHBOARDS[dashboard_key]
                dashboard_id = dashboard_config["dashboard_id"]
                dashboard_name = dashboard_config["dashboard_name"]

                # Build URL with accelerator-specific dashboard
                base_url = "https://grafana-psap-obs.apps.ocp4.intlab.redhat.com"
                return (
                    f"{base_url}/d/{dashboard_id}/{dashboard_name}"
                    f"?orgId=1&from={start_ms}&to={end_ms}"
                    f"&timezone=browser&var-deployment_uuid={uuid}"
                    f"&var-deployment_pod_name=$__all&var-rate_interval=1m"
                )
            return None

        display_filtered_df["grafana_metrics_link"] = display_filtered_df.apply(
            create_grafana_link, axis=1
        )

        # Reorder columns to place grafana_metrics_link after TP and Run Date at the end
        cols = display_filtered_df.columns.tolist()
        if "grafana_metrics_link" in cols and "TP" in cols:
            cols.remove("grafana_metrics_link")
            tp_idx = cols.index("TP")
            cols.insert(tp_idx + 1, "grafana_metrics_link")
        if "Run Date" in cols:
            cols.remove("Run Date")
            cols.append("Run Date")  # Move to end
            display_filtered_df = display_filtered_df[cols]

        # Define column configurations with help text
        column_config = {
            "Row #": st.column_config.NumberColumn(
                "Row #",
                help="Sequential row number for this filtered dataset",
                pinned=True,
            ),
            "run": st.column_config.TextColumn(
                "run",
                help="Unique identifier combining accelerator, model, and TP configuration",
                pinned=True,
            ),
            "accelerator": st.column_config.TextColumn(
                "accelerator",
                help="Hardware accelerator type (e.g., H200, MI300X, TPU)",
            ),
            "model": st.column_config.TextColumn(
                "model", help="Full path/name of the LLM model being benchmarked"
            ),
            "version": st.column_config.TextColumn(
                "version",
                help="Inference server version (e.g., RHAIIS-3.2.1, vLLM-0.10.0)",
                pinned=True,
            ),
            "prompt toks": st.column_config.NumberColumn(
                "prompt toks",
                help="Target number of prompt tokens used in the benchmark",
            ),
            "output toks": st.column_config.NumberColumn(
                "output toks",
                help="Target number of output tokens to generate in the benchmark",
            ),
            "TP": st.column_config.NumberColumn(
                "TP",
                help="Tensor Parallelism size - number of GPUs used to split the model across",
            ),
            "measured concurrency": st.column_config.NumberColumn(
                "measured concurrency",
                help="Actual concurrency level achieved during the benchmark run",
                format="%.2f",
            ),
            "intended concurrency": st.column_config.NumberColumn(
                "intended concurrency",
                help="Target concurrency level - number of parallel requests sent to the server",
                pinned=True,
            ),
            "measured rps": st.column_config.NumberColumn(
                "measured rps",
                help="Measured requests per second - actual request throughput achieved",
                format="%.4f",
            ),
            "output_tok/sec": st.column_config.NumberColumn(
                "output_tok/sec",
                help="Output tokens per second - key throughput metric (higher is better)",
                format="%.2f",
            ),
            "total_tok/sec": st.column_config.NumberColumn(
                "total_tok/sec",
                help="Total tokens per second (prompt + output tokens combined)",
                format="%.2f",
            ),
            "prompt_token_count_mean": st.column_config.NumberColumn(
                "prompt_token_count_mean",
                help="Average number of prompt tokens across all requests",
                format="%.1f",
            ),
            "prompt_token_count_p99": st.column_config.NumberColumn(
                "prompt_token_count_p99",
                help="99th percentile of prompt token counts",
                format="%.1f",
            ),
            "output_token_count_mean": st.column_config.NumberColumn(
                "output_token_count_mean",
                help="Average number of output tokens generated across all requests",
                format="%.1f",
            ),
            "output_token_count_p99": st.column_config.NumberColumn(
                "output_token_count_p99",
                help="99th percentile of output token counts",
                format="%.1f",
            ),
            "ttft_median": st.column_config.NumberColumn(
                "ttft_median",
                help="Time to First Token median - time until first token is generated (ms, lower is better)",
                format="%.2f",
            ),
            "ttft_p95_s": st.column_config.NumberColumn(
                "ttft_p95_s",
                help="Time to First Token 95th percentile - key latency SLO metric (s, lower is better)",
                format="%.3f",
            ),
            "ttft_p1": st.column_config.NumberColumn(
                "ttft_p1",
                help="Time to First Token 1st percentile - best-case TTFT (ms)",
                format="%.2f",
            ),
            "ttft_p999": st.column_config.NumberColumn(
                "ttft_p999",
                help="Time to First Token 99.9th percentile - worst-case TTFT (ms)",
                format="%.2f",
            ),
            "ttft_mean": st.column_config.NumberColumn(
                "ttft_mean",
                help="Time to First Token average across all requests (ms)",
                format="%.2f",
            ),
            "ttft_p99": st.column_config.NumberColumn(
                "ttft_p99",
                help="Time to First Token 99th percentile (ms, lower is better)",
                format="%.2f",
            ),
            "tpot_median": st.column_config.NumberColumn(
                "tpot_median",
                help="Time Per Output Token median - time to generate each token (ms)",
                format="%.2f",
            ),
            "tpot_p95": st.column_config.NumberColumn(
                "tpot_p95",
                help="Time Per Output Token 95th percentile (ms, lower is better)",
                format="%.2f",
            ),
            "tpot_p99": st.column_config.NumberColumn(
                "tpot_p99",
                help="Time Per Output Token 99th percentile (ms)",
                format="%.2f",
            ),
            "tpot_p999": st.column_config.NumberColumn(
                "tpot_p999",
                help="Time Per Output Token 99.9th percentile (ms)",
                format="%.2f",
            ),
            "tpot_p1": st.column_config.NumberColumn(
                "tpot_p1",
                help="Time Per Output Token 1st percentile - best-case TPOT (ms)",
                format="%.2f",
            ),
            "itl_median": st.column_config.NumberColumn(
                "itl_median",
                help="Inter-Token Latency median - time between consecutive tokens (ms)",
                format="%.2f",
            ),
            "itl_p95": st.column_config.NumberColumn(
                "itl_p95",
                help="Inter-Token Latency 95th percentile - key latency SLO metric (ms, lower is better)",
                format="%.2f",
            ),
            "itl_p999": st.column_config.NumberColumn(
                "itl_p999",
                help="Inter-Token Latency 99.9th percentile - worst-case ITL (ms)",
                format="%.2f",
            ),
            "itl_p1": st.column_config.NumberColumn(
                "itl_p1",
                help="Inter-Token Latency 1st percentile - best-case ITL (ms)",
                format="%.2f",
            ),
            "itl_mean": st.column_config.NumberColumn(
                "itl_mean",
                help="Inter-Token Latency average across all requests (ms)",
                format="%.2f",
            ),
            "itl_p99": st.column_config.NumberColumn(
                "itl_p99",
                help="Inter-Token Latency 99th percentile (ms, lower is better)",
                format="%.2f",
            ),
            "request_latency_median": st.column_config.NumberColumn(
                "request_latency_median",
                help="Total request latency median - end-to-end time per request (seconds)",
                format="%.2f",
            ),
            "request_latency_min": st.column_config.NumberColumn(
                "request_latency_min",
                help="Minimum total request latency observed (seconds)",
                format="%.2f",
            ),
            "request_latency_max": st.column_config.NumberColumn(
                "request_latency_max",
                help="Maximum total request latency observed (seconds)",
                format="%.2f",
            ),
            "successful_requests": st.column_config.NumberColumn(
                "successful_requests",
                help="Number of requests that completed successfully",
            ),
            "errored_requests": st.column_config.NumberColumn(
                "errored_requests",
                help="Number of requests that failed or returned errors",
            ),
            "uuid": st.column_config.TextColumn(
                "uuid", help="Unique identifier for this specific benchmark run"
            ),
            "runtime_args": st.column_config.TextColumn(
                "runtime_args",
                help="Complete runtime arguments and configuration used for this benchmark",
            ),
            "profile": st.column_config.TextColumn(
                "profile",
                help="Workload profile category based on prompt/output token sizes",
            ),
            "error_rate": st.column_config.NumberColumn(
                "error_rate",
                help="Error rate percentage - (errored_requests / total_requests) × 100 (lower is better)",
                format="%.2f",
            ),
            "efficiency_ratio": st.column_config.NumberColumn(
                "efficiency_ratio",
                help="Efficiency ratio - output tokens per second per TP unit (output_tok/sec ÷ TP), measures GPU utilization efficiency (higher is better)",
                format="%.2f",
            ),
            "grafana_metrics_link": st.column_config.LinkColumn(
                "Grafana Metrics",
                help="Link to Grafana dashboard showing detailed metrics for this benchmark run (available only for runs with timestamp data)",
                display_text="View Metrics 📊",
            ),
            "Run Date": st.column_config.TextColumn(
                "Run Date",
                help="Date when the benchmark run was executed (from guidellm_start_time)",
            ),
        }

        st.dataframe(
            display_filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )


def render_header_with_theme_toggle():
    """Render the main header with theme toggle button and view selector."""
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        # View selector
        view_options = ["RHAIIS Dashboard"]
        if MLPERF_AVAILABLE:
            view_options.append("MLPerf Dashboard")
        if LLMD_AVAILABLE:
            view_options.append("LLM-D Dashboard")

        # Determine default index based on current session state
        # This persists the view selection across refreshes
        current_view = st.session_state.get("selected_view", "RHAIIS Dashboard")
        try:
            default_index = view_options.index(current_view)
        except ValueError:
            default_index = 0

        selected_view = st.radio(
            "  Select View:",
            options=view_options,
            index=default_index,
            key="dashboard_view_selector",
            horizontal=False,
        )

        # Store in session state for access outside this function
        st.session_state.selected_view = selected_view

        # Update URL to persist view selection across page refreshes
        st.query_params["view"] = selected_view

    with col2:
        # Load Red Hat logo
        logo_base64 = get_logo_base64()
        if logo_base64:
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 40px; vertical-align: middle; margin-right: 10px;">'
        else:
            logo_html = "📊"

        st.markdown(
            f"""
            <h1 style='text-align: center; margin-bottom: 0.5rem;' class='main-title'>
                {logo_html} Staging Performance Dashboard
            </h1>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align: center; font-size: 1.2rem; margin-bottom: 1.5rem;' class='main-subtitle'>
                Compare benchmark runs across different models, versions, and hardware configurations with advanced analytics.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("<div style='margin-top: 0.5rem;'>", unsafe_allow_html=True)

        # Create two columns for theme toggle and share button
        theme_col1, theme_col2 = st.columns(2)

        with theme_col1:
            current_mode = st.session_state.get("theme_mode", "auto")

            if current_mode == "auto":
                theme_button_text = "🌓 Auto"
                help_text = "Currently: Auto (follows browser preference). Click to switch to Light mode."
            elif current_mode == "light":
                theme_button_text = "☀️ Light"
                help_text = "Currently: Light mode. Click to switch to Dark mode."
            else:
                theme_button_text = "🌙 Dark"
                help_text = "Currently: Dark mode. Click to switch to Auto mode."

            if st.button(theme_button_text, help=help_text, key="theme_toggle"):
                # Set flag to indicate this is just a theme change, not a filter reset
                st.session_state.theme_change_only = True
                if current_mode == "auto":
                    st.session_state.theme_mode = "light"
                elif current_mode == "light":
                    st.session_state.theme_mode = "dark"
                else:
                    st.session_state.theme_mode = "auto"
                st.rerun()

        with theme_col2:
            if st.button(
                "🔗 Share",
                help="Get a shareable URL with current filters applied",
                key="share_view_header",
            ):
                try:
                    st.toast(
                        "🔗 Shareable URL Generated! Copy the browser URL to share this view.",
                        icon="✅",
                    )
                except Exception as e:
                    st.toast(f"❌ Error generating shareable URL: {e}", icon="🚨")

        st.markdown("</div>", unsafe_allow_html=True)


def render_confidentiality_notice():
    """Render the confidentiality notice."""
    st.markdown(
        '<div style="background-color: rgba(255,165,0,0.12); border-left: 4px solid #ffa500; '
        'padding: 8px 12px; border-radius: 4px; font-size: 0.90rem; line-height: 1.4;">'
        "⚠️ <b>Performance Data Disclaimer</b> — Red Hat Confidential.<br>"
        "<b>Sharing Policy:</b> Disclosure to third parties is permitted only under a signed NDA. "
        "Public referencing, blogging, or external publication requires prior consultation and "
        "approval from the PSAP Inference Team (Contact: @psap-inference on #forum-psap). "
        "Users are encouraged to leverage these insights for internal analysis and guided "
        "customer-facing support."
        "</div>",
        unsafe_allow_html=True,
    )


initialize_streamlit_config()
initialize_session_state()

# Check URL for view parameter and set session state accordingly
# This allows the view selection to persist across page refreshes
if "view" in st.query_params:
    view_from_url = st.query_params["view"]
    if view_from_url in ["RHAIIS Dashboard", "MLPerf Dashboard", "LLM-D Dashboard"]:
        st.session_state.selected_view = view_from_url

st.markdown(get_app_css(), unsafe_allow_html=True)
apply_theme_css()

render_header_with_theme_toggle()
render_confidentiality_notice()

# Get selected view from session state (set in render_header_with_theme_toggle)
selected_view = st.session_state.get("selected_view", "RHAIIS Dashboard")

# Reset expander states when switching between views
previous_view = st.session_state.get("previous_view", None)
if previous_view != selected_view and previous_view is not None:
    # Reset all expander states when view changes
    st.session_state.performance_plots_expanded = False
    st.session_state.pareto_expanded = False
    st.session_state.compare_versions_summary_expanded = False
    st.session_state.compare_models_expanded = False
    st.session_state.model_comparison_expanded = False
    st.session_state.runtime_configs_expanded = False
    st.session_state.energy_expanded = False

# Update previous view
st.session_state.previous_view = selected_view

# If MLPerf view is selected, render MLPerf dashboard and exit
if MLPERF_AVAILABLE and selected_view == "MLPerf Dashboard":
    # Version mapping
    mlperf_versions = {
        "v5.1": "mlperf-data/mlperf-5.1.csv",
        "v5.0": "mlperf-data/mlperf-5.0.csv",
    }

    render_mlperf_dashboard(mlperf_versions)
    st.stop()  # Stop execution here, don't load RHAIIS data

# If LLM-D view is selected, render LLM-D dashboard and exit
if LLMD_AVAILABLE and selected_view == "LLM-D Dashboard":
    render_llmd_dashboard("llmd-dashboard.csv")
    st.stop()  # Stop execution here, don't load RHAIIS data

# Otherwise, continue with RHAIIS dashboard
DATA_FILE = "consolidated_dashboard.csv"

cache_key = str(int(time.time() // 300))  # Updates every 5 minutes
df = load_data(DATA_FILE, cache_key=cache_key)


def main():
    """Main application function that orchestrates all components."""
    global df

    if df is None:
        st.warning("⚠️ Data was None, attempting to reload...")
        df = load_data(DATA_FILE, cache_key=cache_key)

    if df is not None:

        def encode_filters_to_url(accelerators, models, versions, profile, tp_sizes):
            """Encode current filter state to URL parameters."""
            url_params = {}

            if accelerators:
                url_params["accelerators"] = ",".join(accelerators)
            if models:
                url_params["models"] = ",".join(models)
            if versions:
                url_params["versions"] = ",".join(versions)
            if profile:
                url_params["profile"] = profile
            if tp_sizes:
                url_params["tp_sizes"] = ",".join(map(str, tp_sizes))

            st.query_params.update(url_params)

        def decode_filters_from_url():
            """Decode filter state from URL parameters."""
            query_params = st.query_params

            all_accelerators = sorted(df["accelerator"].unique().tolist())
            all_models = sorted(df["model"].unique().tolist())
            all_versions = sorted(df["version"].unique().tolist())
            all_profiles = sorted(df["profile"].unique().tolist())
            all_tp_sizes = sorted(df["TP"].dropna().unique().tolist())

            url_accelerators = []
            url_models = []
            url_versions = []
            url_profile = None
            url_tp_sizes = []

            if "accelerators" in query_params:
                url_accelerators = [
                    acc.strip()
                    for acc in query_params["accelerators"].split(",")
                    if acc.strip() in all_accelerators
                ]

            if "models" in query_params:
                url_models = [
                    model.strip()
                    for model in query_params["models"].split(",")
                    if model.strip() in all_models
                ]

            if "versions" in query_params:
                url_versions = [
                    ver.strip()
                    for ver in query_params["versions"].split(",")
                    if ver.strip() in all_versions
                ]

            if "profile" in query_params:
                profile_from_url = query_params["profile"].strip()
                if profile_from_url in all_profiles:
                    url_profile = profile_from_url

            if "tp_sizes" in query_params:
                try:
                    url_tp_sizes = [
                        int(tp.strip())
                        for tp in query_params["tp_sizes"].split(",")
                        if tp.strip().isdigit() and int(tp.strip()) in all_tp_sizes
                    ]
                except:
                    url_tp_sizes = []

            return url_accelerators, url_models, url_versions, url_profile, url_tp_sizes

    df["profile"] = df.apply(assign_profile, axis=1)

    df["error_rate"] = (
        df["errored_requests"]
        / (df["successful_requests"] + df["errored_requests"])
        * 100
    )
    df["error_rate"] = df["error_rate"].fillna(0)

    df["efficiency_ratio"] = df["output_tok/sec"] / df["TP"]

    # Convert TTFT from milliseconds to seconds for display
    if "ttft_p95" in df.columns:
        df["ttft_p95_s"] = df["ttft_p95"] / 1000
    else:
        df["ttft_p95_s"] = np.nan

    if "url_filters_loaded" not in st.session_state:
        st.session_state.url_filters_loaded = True
        url_accelerators, url_models, url_versions, url_profile, url_tp_sizes = (
            decode_filters_from_url()
        )

        if any([url_accelerators, url_models, url_versions, url_profile, url_tp_sizes]):
            preferred_versions = [
                "RHAIIS-3.3",
                "vLLM-0.13.0",
            ]
            available_versions = sorted(df["version"].unique().tolist())
            default_versions = [
                v for v in preferred_versions if v in available_versions
            ]

            preferred_models = [
                "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]
            available_models = sorted(df["model"].unique().tolist())
            default_models = [m for m in preferred_models if m in available_models]

            st.session_state.baseline_accelerators = (
                url_accelerators
                if url_accelerators
                else sorted(df["accelerator"].unique().tolist())
            )
            st.session_state.baseline_models = (
                url_models if url_models else default_models
            )
            st.session_state.baseline_versions = (
                url_versions if url_versions else default_versions
            )
            st.session_state.baseline_profile = (
                url_profile
                if url_profile
                else (
                    "Profile A: Balanced (1k/1k)"
                    if "Profile A: Balanced (1k/1k)"
                    in sorted(df["profile"].unique().tolist())
                    else (
                        sorted(df["profile"].unique().tolist())[0]
                        if sorted(df["profile"].unique().tolist())
                        else None
                    )
                )
            )
            st.session_state.baseline_tp_sizes = (
                url_tp_sizes
                if url_tp_sizes
                else sorted(df["TP"].dropna().unique().tolist())
            )
            st.session_state.use_url_filters = True
        else:
            preferred_versions = [
                "RHAIIS-3.3",
                "vLLM-0.13.0",
            ]
            available_versions = sorted(df["version"].unique().tolist())
            default_versions = [
                v for v in preferred_versions if v in available_versions
            ]

            preferred_models = [
                "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]
            available_models = sorted(df["model"].unique().tolist())
            default_models = [m for m in preferred_models if m in available_models]

            st.session_state.baseline_accelerators = sorted(
                df["accelerator"].unique().tolist()
            )
            st.session_state.baseline_models = default_models
            st.session_state.baseline_versions = default_versions
            st.session_state.baseline_profile = (
                "Profile A: Balanced (1k/1k)"
                if "Profile A: Balanced (1k/1k)"
                in sorted(df["profile"].unique().tolist())
                else (
                    sorted(df["profile"].unique().tolist())[0]
                    if sorted(df["profile"].unique().tolist())
                    else None
                )
            )
            st.session_state.baseline_tp_sizes = sorted(
                df["TP"].dropna().unique().tolist()
            )
            st.session_state.use_url_filters = False

    st.header("Filter Your Data")

    if "filters_initialized" not in st.session_state:
        st.session_state.filters_initialized = True
        st.session_state.filter_change_key = 0
        st.session_state.filters_were_cleared = False

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        # Accelerators filter - filtered by currently selected profile
        temp_df = df.copy()

        # Determine what the current/default profile is by checking session state
        current_profile = st.session_state.get(
            f"profile_filter_{st.session_state.filter_change_key}", None
        )

        # If no profile selected yet, determine the default that will be selected
        if not current_profile:
            available_profiles_raw = sorted(df["profile"].unique().tolist())
            available_profiles = [
                p for p in available_profiles_raw if p != "Custom"
            ] + (["Custom"] if "Custom" in available_profiles_raw else [])

            # Default to Profile A (1k/1k) when clearing or as fallback
            default_profile = "Profile A: Balanced (1k/1k)"

            if st.session_state.get("clear_all_filters", False) or st.session_state.get(
                "filters_were_cleared", False
            ):
                current_profile = (
                    default_profile
                    if default_profile in available_profiles
                    else (available_profiles[0] if available_profiles else None)
                )
            elif st.session_state.get("reset_to_defaults", False):
                baseline_profile = st.session_state.get(
                    "baseline_profile", default_profile
                )
                current_profile = (
                    baseline_profile
                    if baseline_profile in available_profiles
                    else (
                        default_profile
                        if default_profile in available_profiles
                        else (available_profiles[0] if available_profiles else None)
                    )
                )
            else:
                baseline_profile = st.session_state.get(
                    "baseline_profile", default_profile
                )
                current_profile = (
                    baseline_profile
                    if baseline_profile in available_profiles
                    else (
                        default_profile
                        if default_profile in available_profiles
                        else (available_profiles[0] if available_profiles else None)
                    )
                )

        # Filter accelerators by the current/default profile
        if current_profile:
            temp_df = temp_df[temp_df["profile"] == current_profile]

        accelerators = (
            sorted(temp_df["accelerator"].unique().tolist())
            if not temp_df.empty
            else []
        )

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            acc_default = []
        elif st.session_state.get("reset_to_defaults", False):
            baseline_accelerators = st.session_state.get(
                "baseline_accelerators", accelerators
            )
            acc_default = [a for a in baseline_accelerators if a in accelerators]
        else:
            baseline_accelerators = st.session_state.get(
                "baseline_accelerators", accelerators
            )
            acc_default = [a for a in baseline_accelerators if a in accelerators]

        # Get previously selected accelerators from session state
        prev_accel_key = f"accelerators_filter_{st.session_state.filter_change_key}"
        prev_selected = st.session_state.get(prev_accel_key, None)

        # Keep previously selected accelerators that are still available in current profile
        # Use 'is not None' to allow empty list selection
        if prev_selected is not None:
            preserved_selections = [a for a in prev_selected if a in accelerators]
        else:
            preserved_selections = acc_default

        selected_accelerators = st.multiselect(
            "1️⃣ Select Accelerator(s)",
            accelerators,
            default=preserved_selections,
            key=prev_accel_key,
        )
        st.caption("💡 See dropdown for more available models, versions and TP sizes.")

    with filter_col2:
        # ISL/OSL Profile filter - filtered by selected accelerators
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]

        profiles_raw = (
            sorted(temp_df["profile"].unique().tolist()) if not temp_df.empty else []
        )
        # Sort so "Custom" always comes last to avoid it being picked as profiles[0] fallback
        profiles = [p for p in profiles_raw if p != "Custom"] + (
            ["Custom"] if "Custom" in profiles_raw else []
        )

        # Default to Profile A (1k/1k) when clearing or as fallback
        default_profile = "Profile A: Balanced (1k/1k)"

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            profiles_default = (
                default_profile
                if default_profile in profiles
                else (profiles[0] if profiles else None)
            )
        elif st.session_state.get("reset_to_defaults", False):
            baseline_profile = st.session_state.get("baseline_profile", default_profile)
            profiles_default = (
                baseline_profile
                if baseline_profile in profiles
                else (
                    default_profile
                    if default_profile in profiles
                    else (profiles[0] if profiles else None)
                )
            )
        else:
            baseline_profile = st.session_state.get("baseline_profile", default_profile)
            profiles_default = (
                baseline_profile
                if baseline_profile in profiles
                else (
                    default_profile
                    if default_profile in profiles
                    else (profiles[0] if profiles else None)
                )
            )

        # Initialize session state for profile key BEFORE the widget renders.
        # This avoids the "double-click" issue caused by conflicting `index`
        # and session state values — when both are sent to the frontend,
        # a stale `index` (computed from baseline_profile which lags one
        # render behind) can override the user's selection.
        profile_key = f"profile_filter_{st.session_state.filter_change_key}"
        if profile_key not in st.session_state:
            # First render with this key — set the computed default
            if profiles_default and profiles_default in profiles:
                st.session_state[profile_key] = profiles_default
            elif profiles:
                st.session_state[profile_key] = (
                    default_profile if default_profile in profiles else profiles[0]
                )
        elif st.session_state.get(profile_key) not in profiles and profiles:
            # Stored value is no longer in the options (e.g. accelerators
            # changed and the profile is no longer available) — reset
            st.session_state[profile_key] = (
                profiles_default
                if profiles_default and profiles_default in profiles
                else (default_profile if default_profile in profiles else profiles[0])
            )

        selected_profile = (
            st.selectbox(
                "2️⃣ Select Input/Output Sequence Length (ISL/OSL)",
                profiles,
                format_func=clean_profile_name,
                key=profile_key,
            )
            if profiles
            else None
        )

        selected_profiles = [selected_profile] if selected_profile is not None else []

        # Update baseline_profile to remember user's current selection
        # This ensures the selected profile is retained when other filters change
        if selected_profile is not None:
            st.session_state.baseline_profile = selected_profile
            # Clear the "filters_were_cleared" flag so the new selection is preserved
            if st.session_state.get("filters_were_cleared", False):
                st.session_state.filters_were_cleared = False

    with filter_col3:
        # Versions filter - filtered by selected accelerators AND currently selected profile
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]

        # Filter versions by the currently selected profile
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]

        versions = (
            sorted(temp_df["version"].unique().tolist()) if not temp_df.empty else []
        )

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            versions_default = []
        elif st.session_state.get("reset_to_defaults", False):
            baseline_versions = st.session_state.get("baseline_versions", versions)
            versions_default = [v for v in baseline_versions if v in versions]
        else:
            baseline_versions = st.session_state.get("baseline_versions", versions)
            versions_default = [v for v in baseline_versions if v in versions]

        # Get previously selected versions from session state
        prev_versions_key = f"versions_filter_{st.session_state.filter_change_key}"
        prev_selected = st.session_state.get(prev_versions_key, None)

        # Keep previously selected versions that are still available in current profile
        # Use 'is not None' to allow empty list selection
        if prev_selected is not None:
            preserved_selections = [v for v in prev_selected if v in versions]
        else:
            preserved_selections = versions_default

        selected_versions = st.multiselect(
            "3️⃣ Select Version(s)",
            versions,
            default=preserved_selections,
            key=prev_versions_key,
        )
        st.caption(
            "💡 **See Filters Help button to see all valid filter combinations.**"
        )

    # Add negative margin spacer to reduce gap between filter rows
    st.markdown('<div style="margin-top: -2rem;"></div>', unsafe_allow_html=True)

    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col4:
        # Models filter - filtered by selected accelerators, versions, AND currently selected profile
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]

        # Filter models by the currently selected profile
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]

        models = sorted(temp_df["model"].unique().tolist()) if not temp_df.empty else []

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            models_default = []
        elif st.session_state.get("reset_to_defaults", False):
            baseline_models = st.session_state.get("baseline_models", models)
            models_default = [m for m in baseline_models if m in models]
        else:
            baseline_models = st.session_state.get("baseline_models", models)
            models_default = [m for m in baseline_models if m in models]

        # Check if "Select All Models" is checked (from previous render)
        select_all_key = f"select_all_models_{st.session_state.filter_change_key}"
        select_all_checked = st.session_state.get(select_all_key, False)

        # If "Select All" is checked, set default to all models
        models_to_select = models if select_all_checked else models_default

        # Get previously selected models from session state
        prev_models_key = f"models_filter_{st.session_state.filter_change_key}_all_{select_all_checked}"
        prev_selected = st.session_state.get(prev_models_key, None)

        # Keep previously selected models that are still available in current profile
        # Use 'is not None' to allow empty list selection
        if prev_selected is not None:
            preserved_selections = [m for m in prev_selected if m in models]
        else:
            preserved_selections = models_to_select

        selected_models = st.multiselect(
            "4️⃣ Select Model(s)",
            models,
            default=preserved_selections,
            key=prev_models_key,
        )

        # Add checkbox for selecting all models below the dropdown
        st.checkbox(
            "Select All Models",
            value=select_all_checked,
            key=select_all_key,
        )

    with filter_col5:
        # TP sizes filter - filtered by accelerators, versions, models, and profiles
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]

        tp_sizes = (
            sorted(temp_df["TP"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        # Check if "Select All Models" is checked
        select_all_key = f"select_all_models_{st.session_state.filter_change_key}"
        select_all_checked = st.session_state.get(select_all_key, False)

        # Track previous model selection for detecting changes
        tracking_key = "previous_models_for_tp_tracking"
        prev_selected_models = st.session_state.get(tracking_key, None)

        # Get previous TP selection
        prev_tp_key = (
            f"tp_filter_{st.session_state.filter_change_key}_all_{select_all_checked}"
        )
        prev_selected_tp = st.session_state.get(prev_tp_key, None)

        # Check if models selection changed
        models_changed = (
            (prev_selected_models != selected_models)
            if prev_selected_models is not None
            else (bool(selected_models))
        )

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            tp_default = []
        elif models_changed and selected_models:
            # Models changed and some are selected - auto-select all TP sizes for selected models
            tp_default = tp_sizes
        elif select_all_checked:
            # If "Select All Models" is checked, also select all TP sizes
            tp_default = tp_sizes
        elif prev_selected_tp is not None:
            # Models didn't change - preserve user's manual TP selections (filtered to available)
            tp_default = [tp for tp in prev_selected_tp if tp in tp_sizes]
        elif st.session_state.get("reset_to_defaults", False):
            baseline_tp_sizes = st.session_state.get("baseline_tp_sizes", tp_sizes)
            tp_default = [tp for tp in baseline_tp_sizes if tp in tp_sizes]
        else:
            baseline_tp_sizes = st.session_state.get("baseline_tp_sizes", tp_sizes)
            tp_default = [tp for tp in baseline_tp_sizes if tp in tp_sizes]

        selected_tp = st.multiselect(
            "5️⃣ Select TP Size(s)",
            tp_sizes,
            default=tp_default,
            key=prev_tp_key,
        )

        # Update the tracking variable with current selection
        st.session_state[tracking_key] = selected_models

    with filter_col6:
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button(
                "↩ Reset to Defaults", help="Reset filters to system/URL defaults"
            ):
                st.session_state.clear_all_filters = False
                st.session_state.filters_were_cleared = False
                st.session_state.reset_to_defaults = True
                st.session_state.filter_change_key += 1
                # Reset model tracking for smart TP filtering
                st.session_state.previous_models_for_tp_tracking = None
                # Close all expanders when resetting filters
                st.session_state.performance_plots_expanded = False
                st.session_state.model_comparison_expanded = False
                st.session_state.runtime_configs_expanded = False
                st.session_state.energy_expanded = False
                # Reset filter state tracking
                if "previous_filter_state" in st.session_state:
                    del st.session_state.previous_filter_state
                st.rerun()

        with btn_col2:
            if st.button("🧹 Clear Filters", help="Clear all filter selections"):
                st.session_state.clear_all_filters = True
                st.session_state.filters_were_cleared = True
                st.session_state.filter_change_key += 1
                # Reset model tracking for smart TP filtering
                st.session_state.previous_models_for_tp_tracking = None
                # Close all expanders when clearing filters
                st.session_state.performance_plots_expanded = False
                st.session_state.model_comparison_expanded = False
                st.session_state.runtime_configs_expanded = False
                st.session_state.energy_expanded = False
                # Reset filter state tracking
                if "previous_filter_state" in st.session_state:
                    del st.session_state.previous_filter_state
                st.rerun()

        with btn_col3:
            with st.popover("❓ Filters Help", use_container_width=True):
                st.markdown("### ✅ Valid Filter Combinations")
                st.markdown("View all valid combinations of filters:")

                # Selector for tree view type
                tree_view = st.radio(
                    "Group by:",
                    options=["Model", "Version"],
                    horizontal=True,
                    key="filter_help_tree_view",
                )

                if tree_view == "Model":
                    # Group by Model → Accelerator → Version → Profile → TP
                    models = sorted(df["model"].unique())

                    for model in models:
                        model_short = model.split("/")[-1] if "/" in model else model
                        model_data = df[df["model"] == model]

                        with st.expander(f"🤖 {model_short}", expanded=False):
                            combo_dict = {}
                            for _, row in model_data.iterrows():
                                acc = row["accelerator"]
                                version = row["version"]
                                profile = row["profile"]
                                tp = row["TP"]

                                if acc not in combo_dict:
                                    combo_dict[acc] = {}
                                if version not in combo_dict[acc]:
                                    combo_dict[acc][version] = {}
                                if profile not in combo_dict[acc][version]:
                                    combo_dict[acc][version][profile] = []

                                if tp not in combo_dict[acc][version][profile]:
                                    combo_dict[acc][version][profile].append(tp)

                            tree_text = ""
                            for acc in sorted(combo_dict.keys()):
                                tree_text += f"🔧 {acc}\n"

                                versions = sorted(combo_dict[acc].keys())
                                for version in versions:
                                    tree_text += f"    📦 {version}\n"

                                    profiles = sorted(combo_dict[acc][version].keys())
                                    for profile in profiles:
                                        tp_list = ", ".join(
                                            map(
                                                str,
                                                sorted(
                                                    combo_dict[acc][version][profile]
                                                ),
                                            )
                                        )
                                        profile_display = clean_profile_name(profile)
                                        tree_text += f"        📋 {profile_display} → TP: {tp_list}\n"
                                tree_text += "\n"

                            st.code(tree_text, language=None)

                else:  # Group by Version
                    # Group by Version → Accelerator → Model → Profile → TP
                    versions = sorted(df["version"].unique())

                    for version in versions:
                        version_data = df[df["version"] == version]

                        with st.expander(f"📦 {version}", expanded=False):
                            combo_dict = {}
                            for _, row in version_data.iterrows():
                                acc = row["accelerator"]
                                model = row["model"]
                                model_short = (
                                    model.split("/")[-1] if "/" in model else model
                                )
                                profile = row["profile"]
                                tp = row["TP"]

                                if acc not in combo_dict:
                                    combo_dict[acc] = {}
                                if model_short not in combo_dict[acc]:
                                    combo_dict[acc][model_short] = {}
                                if profile not in combo_dict[acc][model_short]:
                                    combo_dict[acc][model_short][profile] = []

                                if tp not in combo_dict[acc][model_short][profile]:
                                    combo_dict[acc][model_short][profile].append(tp)

                            tree_text = ""
                            for acc in sorted(combo_dict.keys()):
                                tree_text += f"🔧 {acc}\n"

                                models = sorted(combo_dict[acc].keys())
                                for model_short in models:
                                    tree_text += f"    🤖 {model_short}\n"

                                    profiles = sorted(
                                        combo_dict[acc][model_short].keys()
                                    )
                                    for profile in profiles:
                                        tp_list = ", ".join(
                                            map(
                                                str,
                                                sorted(
                                                    combo_dict[acc][model_short][
                                                        profile
                                                    ]
                                                ),
                                            )
                                        )
                                        profile_display = clean_profile_name(profile)
                                        tree_text += f"        📋 {profile_display} → TP: {tp_list}\n"
                                tree_text += "\n"

                            st.code(tree_text, language=None)

    if st.session_state.get("clear_all_filters", False):
        st.session_state.clear_all_filters = False
    if st.session_state.get("reset_to_defaults", False):
        st.session_state.reset_to_defaults = False

    with contextlib.suppress(Exception):
        encode_filters_to_url(
            selected_accelerators,
            selected_models,
            selected_versions,
            selected_profile,
            selected_tp,
        )

    filtered_df = df[
        df["accelerator"].isin(selected_accelerators)
        & df["model"].isin(selected_models)
        & df["version"].isin(selected_versions)
        & (df["profile"].isin(selected_profiles) if selected_profiles else True)
        & df["TP"].isin(selected_tp)
    ].copy()

    # Detect if filters have changed and close expanders
    current_filter_state = {
        "accelerators": tuple(sorted(selected_accelerators)),
        "models": tuple(sorted(selected_models)),
        "versions": tuple(sorted(selected_versions)),
        "profile": selected_profile,
        "tp": tuple(sorted(selected_tp)),
    }

    previous_filter_state = st.session_state.get("previous_filter_state", None)

    # If filters have changed (and not first run), close all expanders
    if (
        previous_filter_state is not None
        and previous_filter_state != current_filter_state
    ):
        st.session_state.performance_plots_expanded = False
        st.session_state.model_comparison_expanded = False
        st.session_state.compare_models_expanded = False
        st.session_state.runtime_configs_expanded = False
        st.session_state.energy_expanded = False

    # Store current filter state for next comparison
    st.session_state.previous_filter_state = current_filter_state

    if not filtered_df.empty:
        accelerator_color_map = {
            "H200": "#1f77b4",
            "MI300X": "#ff7f0e",
            "TPU": "#2ca02c",
        }

        st.markdown("---")

        render_performance_plots_section(filtered_df)
        render_dataset_representation_section(selected_profile)
        if selected_profile != "Custom":
            render_pareto_plots_section(preloaded_df=df)
        else:
            render_custom_pareto_tradeoff_section(filtered_df)

        if selected_profile != "Custom":
            render_model_performance_comparison_section(
                filtered_df, accelerator_color_map
            )
        render_compare_versions_summary_section(
            df
        )  # Uses full df for independent filtering
        render_compare_models_section(filtered_df, selected_profile)
        if selected_profile != "Custom":
            render_performance_trends_section(
                df
            )  # Uses full df for independent filtering
            render_cost_analysis_section(filtered_df, accelerator_color_map)
            render_energy_carbon_methodology_section(
                df
            )  # Pass full df - energy section has independent filters
        render_runtime_configs_section(filtered_df)
        render_filtered_data_section(filtered_df)

    else:
        if selected_models:
            available_data_info = []

            for model in selected_models:
                model_data = df[df["model"] == model]
                if not model_data.empty:
                    available_profiles = sorted(model_data["profile"].unique().tolist())
                    available_accelerators = sorted(
                        model_data["accelerator"].unique().tolist()
                    )
                    available_versions = sorted(model_data["version"].unique().tolist())
                    available_tp = sorted(model_data["TP"].unique().tolist())

                    model_short = model.split("/")[-1] if "/" in model else model

                    available_data_info.append(
                        {
                            "model": model_short,
                            "original_model_name": model,
                            "profiles": available_profiles,
                            "accelerators": available_accelerators,
                            "versions": available_versions,
                            "tp_sizes": available_tp,
                        }
                    )

            if available_data_info:
                with st.container():
                    st.markdown(
                        """
                        <div class='no-data-error-banner' style='padding: 5px; border-radius: 2px; margin: 5px 0; text-align: center; box-shadow: 0 6px 12px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0; font-size: 1.8em; font-weight: bold;'>
                                 No Data Matches Your Current Filter Settings
                            </h2>
                            <h3 style='margin: 5px 0 0 0; font-size: 1.2em; opacity: 0.8;'>
                                See available filter combinations for your selected model(s) below:
                            </h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for info in available_data_info:
                        original_model_name = info["original_model_name"]
                        model_data = df[df["model"] == original_model_name]

                        with st.expander(
                            f"📊 {info['model']} - Available Filter Combinations"
                        ):
                            combo_dict = {}
                            for _, row in model_data.iterrows():
                                acc = row["accelerator"]
                                version = row["version"]
                                profile = row["profile"]
                                tp = row["TP"]

                                if acc not in combo_dict:
                                    combo_dict[acc] = {}
                                if version not in combo_dict[acc]:
                                    combo_dict[acc][version] = {}
                                if profile not in combo_dict[acc][version]:
                                    combo_dict[acc][version][profile] = []

                                if tp not in combo_dict[acc][version][profile]:
                                    combo_dict[acc][version][profile].append(tp)

                            tree_text = ""
                            for acc in sorted(combo_dict.keys()):
                                tree_text += f"🔧 {acc}\n"

                                versions = sorted(combo_dict[acc].keys())
                                for version in versions:
                                    tree_text += f"    📦 {version}\n"

                                    profiles = sorted(combo_dict[acc][version].keys())
                                    for profile in profiles:
                                        tp_list = ", ".join(
                                            map(
                                                str,
                                                sorted(
                                                    combo_dict[acc][version][profile]
                                                ),
                                            )
                                        )
                                        # Extract just the ISL/OSL part (e.g., "(32k/256)")
                                        profile_display = clean_profile_name(profile)
                                        tree_text += f"        📋 {profile_display} → TP Sizes: {tp_list}\n"
                                tree_text += "\n"

                            st.code(tree_text, language=None)

            else:
                st.error(
                    "❌ **No data found for the selected model(s).** Please select a different model."
                )
        else:
            st.warning(
                "❌ **No data matches your current filter settings.** Please adjust the filters."
            )


main()
