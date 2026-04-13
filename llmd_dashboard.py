"""LLM-D Dashboard Module.

This module provides functionality to load, process, and visualize
LLM-D benchmark results with disaggregated prefill/decode architecture.
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
S3_KEY = os.environ.get("S3_KEY", "consolidated_dashboard.csv")
S3_KEY_LLMD = os.environ.get("S3_KEY_LLMD", "llmd-dashboard.csv")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def _read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    """Read a CSV file from S3 bucket.

    Args:
        bucket: S3 bucket name.
        key: S3 object key (path to file in bucket).
        region: AWS region name.

    Returns:
        DataFrame with the CSV data.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        s3_client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    else:
        try:
            s3_client = boto3.client("s3", region_name=region)
            s3_client.head_object(Bucket=bucket, Key=key)
        except Exception:
            s3_client = boto3.client(
                "s3",
                region_name=region,
                config=Config(signature_version=UNSIGNED),
            )

    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(io.StringIO(csv_content))


def _geometric_mean(values):
    """Geometric mean of positive values. Accepts a list or pandas Series."""
    if hasattr(values, "values"):
        positive = values[values > 0].values
    else:
        positive = [v for v in values if v > 0]
    if len(positive) == 0:
        return None
    return float(np.exp(np.mean(np.log(positive))))


def _compare_two_datasets(data_a, data_b, metric_config, user_conc_set):
    """Compare two DataFrames on a metric.

    Returns (pct_diff, a_is_better, a_peak_conc, b_peak_conc, is_similar).
    """
    column = metric_config["column"]
    aggregation = metric_config["aggregation"]
    higher_is_better = metric_config["higher_is_better"]

    a_conc = set(data_a["intended concurrency"].dropna().unique())
    b_conc = set(data_b["intended concurrency"].dropna().unique())
    common = a_conc.intersection(b_conc)

    if aggregation == "geom_mean":
        common = common.intersection(user_conc_set)

    if not common:
        return None, None, None, None, None

    a_common = data_a[data_a["intended concurrency"].isin(common)]
    b_common = data_b[data_b["intended concurrency"].isin(common)]

    a_vals = a_common[column].dropna().tolist()
    b_vals = b_common[column].dropna().tolist()

    if not a_vals or not b_vals:
        return None, None, None, None, None

    if aggregation == "peak":
        if higher_is_better:
            a_val, b_val = max(a_vals), max(b_vals)
            a_peak_conc = int(
                a_common.loc[a_common[column].idxmax(), "intended concurrency"]
            )
            b_peak_conc = int(
                b_common.loc[b_common[column].idxmax(), "intended concurrency"]
            )
        else:
            a_val, b_val = min(a_vals), min(b_vals)
            a_peak_conc = int(
                a_common.loc[a_common[column].idxmin(), "intended concurrency"]
            )
            b_peak_conc = int(
                b_common.loc[b_common[column].idxmin(), "intended concurrency"]
            )
    else:
        a_val = _geometric_mean(a_vals)
        b_val = _geometric_mean(b_vals)
        a_peak_conc = None
        b_peak_conc = None

    if a_val is None or b_val is None or b_val == 0:
        return None, None, None, None, None

    pct_diff = ((a_val - b_val) / b_val) * 100
    a_better = pct_diff > 0 if higher_is_better else pct_diff < 0

    return pct_diff, a_better, a_peak_conc, b_peak_conc, abs(pct_diff) < 5


def _keep_expander_open(expander_key):
    """Helper to keep an expander open after widget interaction."""
    st.session_state[expander_key] = True


def assign_profile(row):
    """Assigns a profile label from the actual ISL/OSL values in the data."""
    prompt_toks = int(row["prompt toks"])
    output_toks = int(row["output toks"])
    if prompt_toks == 0 and output_toks == 0:
        return "Custom"
    return f"{prompt_toks}/{output_toks}"


_CUSTOM_ISL_OSL_LABELS = {
    "0/0": "Real Dataset (0/0)",
    "1000/1000": "1000/1000 - Balanced",
    "8000/800": "8000/800 - Heterogeneous",
    "128/128": "128/128 - Multi-turn",
}


def clean_profile_name(profile_name):
    """Extract only the token counts in parentheses from profile names."""
    if profile_name and "(" in profile_name and ")" in profile_name:
        start_idx = profile_name.find("(")
        end_idx = profile_name.find(")", start_idx)
        if start_idx != -1 and end_idx != -1:
            return profile_name[start_idx : end_idx + 1]
    return _CUSTOM_ISL_OSL_LABELS.get(profile_name, profile_name)


def format_custom_isl_osl(pair):
    """Human-readable label for a custom ISL/OSL pair."""
    return _CUSTOM_ISL_OSL_LABELS.get(pair, pair)


@st.cache_data(ttl=300)
def load_llmd_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess LLM-D benchmark data from CSV file or S3.

    If S3_BUCKET environment variable is set, data is loaded from S3.
    Otherwise, falls back to local file system.

    Args:
        file_path: Path to the CSV file to load (fallback).

    Returns:
        DataFrame with loaded and processed data, or None if error occurs.
    """
    try:
        # Try S3 first if configured
        if S3_BUCKET:
            try:
                df = _read_csv_from_s3(S3_BUCKET, S3_KEY_LLMD, S3_REGION)
                logger.info(
                    f"Successfully loaded LLM-D data from S3: s3://{S3_BUCKET}/{S3_KEY_LLMD}"
                )
            except Exception as s3_error:
                logger.warning(
                    f"S3 load failed ({s3_error}), falling back to local file"
                )
                df = pd.read_csv(file_path)
        else:
            logger.info(f"Loading LLM-D data from local file: {file_path}")
            df = pd.read_csv(file_path)
        # Strip whitespace from string columns
        col: str
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()  # type: ignore[assignment]

        # Convert numeric columns
        numeric_cols = [
            "TP",
            "DP",
            "EP",
            "replicas",
            "prefill_pod_count",
            "decode_pod_count",
        ]
        for col_name in numeric_cols:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        # Replace 0 with NaN for pod counts — 0 means "not applicable", not "zero pods"
        for pod_col in ["prefill_pod_count", "decode_pod_count"]:
            if pod_col in df.columns:
                df[pod_col] = df[pod_col].replace(0, pd.NA)

        # Assign profile based on prompt/output tokens
        if "prompt toks" in df.columns and "output toks" in df.columns:
            df["profile"] = df.apply(assign_profile, axis=1)
            df["custom_isl_osl"] = np.where(
                df["profile"] == "Custom",
                df["prompt toks"].astype(int).astype(str)
                + "/"
                + df["output toks"].astype(int).astype(str),
                "",
            )

        # Calculate efficiency ratio (output tokens/sec per TP unit)
        if "output_tok/sec" in df.columns and "TP" in df.columns:
            df["efficiency_ratio"] = df["output_tok/sec"] / df["TP"]

        # Convert TTFT metrics from milliseconds to seconds for consistency with RHAIIS dashboard
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

        # Calculate error rate (percentage of failed requests)
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


def render_llmd_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Render smart cascading filter UI for LLM-D dashboard.

    Args:
        df: LLM-D DataFrame

    Returns:
        Tuple of (filtered_df, filter_selections)
    """
    st.markdown("### Filter your data")

    # Initialize session state for filter management
    if "llmd_filters_initialized" not in st.session_state:
        st.session_state.llmd_filters_initialized = True
        st.session_state.llmd_filter_change_key = 0
        st.session_state.llmd_filters_were_cleared = False

    # Create columns for filters
    # Row 1: Acc (narrow), ISL/OSL (narrow), Version (medium), Model (wide)
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1.5, 2])
    # Row 2: TP, Replicas, Prefill Pods, Decode Pods (equal width)
    filter_col5, filter_col6, filter_col7, filter_col8 = st.columns(4)

    # FILTER 1: Accelerator — filtered by current/default profile (bidirectional cascade)
    with filter_col1:
        temp_df = df.copy()

        current_profile = st.session_state.get(
            f"llmd_profile_filter_{st.session_state.llmd_filter_change_key}", None
        )

        if not current_profile:
            available_profiles_raw = sorted(df["profile"].unique().tolist())
            available_profiles = [
                p for p in available_profiles_raw if p != "Custom"
            ] + (["Custom"] if "Custom" in available_profiles_raw else [])

            if st.session_state.get(
                "llmd_clear_all_filters", False
            ) or st.session_state.get("llmd_filters_were_cleared", False):
                current_profile = available_profiles[0] if available_profiles else None
            else:
                baseline_profile = st.session_state.get("llmd_baseline_profile", None)
                current_profile = (
                    baseline_profile
                    if baseline_profile and baseline_profile in available_profiles
                    else (available_profiles[0] if available_profiles else None)
                )

        if current_profile:
            temp_df = temp_df[temp_df["profile"] == current_profile]

        accelerators = (
            sorted(temp_df["accelerator"].unique().tolist())
            if not temp_df.empty
            else []
        )

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            acc_default = []
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_accelerators = st.session_state.get(
                "llmd_baseline_accelerators", accelerators
            )
            acc_default = [a for a in baseline_accelerators if a in accelerators]
        else:
            baseline_accelerators = st.session_state.get(
                "llmd_baseline_accelerators", accelerators
            )
            acc_default = [a for a in baseline_accelerators if a in accelerators]

        prev_accel_key = f"llmd_acc_filter_{st.session_state.llmd_filter_change_key}"
        prev_selected = st.session_state.get(prev_accel_key, None)

        if prev_selected is not None:
            preserved_selections = [a for a in prev_selected if a in accelerators]
        else:
            persisted = st.session_state.get("llmd_persisted_accelerators", None)
            preserved_selections = (
                [a for a in persisted if a in accelerators]
                if persisted is not None
                else acc_default
            )

        selected_accelerators = st.multiselect(
            "1️⃣ Select Accelerator(s)",
            accelerators,
            default=preserved_selections,
            key=prev_accel_key,
        )

    # FILTER 2: ISL/OSL Profile - filtered by accelerators
    with filter_col2:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]

        profiles_raw = (
            sorted(temp_df["profile"].unique().tolist()) if not temp_df.empty else []
        )
        profiles = [p for p in profiles_raw if p != "Custom"] + (
            ["Custom"] if "Custom" in profiles_raw else []
        )

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            profiles_default = profiles[0] if profiles else None
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_profile = st.session_state.get("llmd_baseline_profile", None)
            profiles_default = (
                baseline_profile
                if baseline_profile and baseline_profile in profiles
                else (profiles[0] if profiles else None)
            )
        else:
            baseline_profile = st.session_state.get("llmd_baseline_profile", None)
            profiles_default = (
                baseline_profile
                if baseline_profile and baseline_profile in profiles
                else (profiles[0] if profiles else None)
            )

        profile_key = f"llmd_profile_filter_{st.session_state.llmd_filter_change_key}"
        if profile_key not in st.session_state:
            persisted_profile = st.session_state.get("llmd_persisted_profile", None)
            if persisted_profile and persisted_profile in profiles:
                st.session_state[profile_key] = persisted_profile
            elif profiles_default and profiles_default in profiles:
                st.session_state[profile_key] = profiles_default
            elif profiles:
                st.session_state[profile_key] = profiles[0]
        elif st.session_state.get(profile_key) not in profiles and profiles:
            st.session_state[profile_key] = (
                profiles_default
                if profiles_default and profiles_default in profiles
                else profiles[0]
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
        st.caption(
            "Please refer to the notes column in the filtered data to understand more about the workload profile."
        )

        selected_profiles = [selected_profile] if selected_profile is not None else []

        # Secondary filter: specific ISL/OSL pair when Custom is selected
        selected_custom_isl_osl = None
        if selected_profile == "Custom" and "custom_isl_osl" in df.columns:
            custom_temp = df.copy()
            if selected_accelerators:
                custom_temp = custom_temp[
                    custom_temp["accelerator"].isin(selected_accelerators)
                ]
            custom_temp = custom_temp[custom_temp["profile"] == "Custom"]
            custom_pairs = sorted(custom_temp["custom_isl_osl"].unique().tolist())
            custom_pairs = [p for p in custom_pairs if p]
            if custom_pairs:
                custom_key = f"llmd_custom_isl_osl_filter_{st.session_state.llmd_filter_change_key}"
                if (
                    custom_key not in st.session_state
                    or st.session_state.get(custom_key) not in custom_pairs
                ):
                    st.session_state[custom_key] = custom_pairs[0]
                selected_custom_isl_osl = st.selectbox(
                    "Select Custom ISL/OSL Pair",
                    custom_pairs,
                    format_func=format_custom_isl_osl,
                    key=custom_key,
                )
                st.session_state.llmd_selected_custom_isl_osl = selected_custom_isl_osl

        if selected_profile is not None:
            st.session_state.llmd_baseline_profile = selected_profile
            if st.session_state.get("llmd_filters_were_cleared", False):
                st.session_state.llmd_filters_were_cleared = False

    # FILTER 3: Version - filtered by accelerators and profile
    with filter_col3:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]

        versions = (
            sorted(temp_df["version"].unique().tolist()) if not temp_df.empty else []
        )

        default_rhoai_versions = [v for v in versions if v.startswith("RHOAI")]

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            version_default = []
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_versions = st.session_state.get(
                "llmd_baseline_versions", default_rhoai_versions or versions
            )
            version_default = [v for v in baseline_versions if v in versions]
        else:
            baseline_versions = st.session_state.get(
                "llmd_baseline_versions", default_rhoai_versions or versions
            )
            version_default = [v for v in baseline_versions if v in versions]

        prev_versions_key = (
            f"llmd_version_filter_{st.session_state.llmd_filter_change_key}"
        )
        prev_selected = st.session_state.get(prev_versions_key, None)

        if prev_selected is not None:
            preserved_selections = [v for v in prev_selected if v in versions]
        else:
            persisted = st.session_state.get("llmd_persisted_versions", None)
            preserved_selections = (
                [v for v in persisted if v in versions]
                if persisted is not None
                else version_default
            )

        selected_versions = st.multiselect(
            "3️⃣ Select Version(s)",
            versions,
            default=preserved_selections,
            key=prev_versions_key,
        )

    # FILTER 4: Model - filtered by accelerators, profile, and version
    with filter_col4:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]  # type: ignore[assignment]

        models = sorted(temp_df["model"].unique().tolist()) if not temp_df.empty else []

        if st.session_state.get("llmd_clear_all_filters", False):
            model_default = []
        else:
            model_default = st.session_state.get(
                "llmd_baseline_models", models[:1] if models else []
            )

        # Check if "Select All Models" is checked
        select_all_models_key = (
            f"select_all_models_{st.session_state.llmd_filter_change_key}"
        )
        select_all_models_checked = st.session_state.get(select_all_models_key, False)

        # If "Select All" is checked, set default to all models
        models_to_select = models if select_all_models_checked else model_default

        # Get previously selected models from session state
        prev_models_key = f"llmd_model_filter_{st.session_state.llmd_filter_change_key}_all_{select_all_models_checked}"
        prev_selected = st.session_state.get(prev_models_key, None)

        # Keep previously selected models that are still available
        if prev_selected is not None:
            preserved_selections = [m for m in prev_selected if m in models]
        else:
            persisted = st.session_state.get("llmd_persisted_models", None)
            preserved_selections = (
                [m for m in persisted if m in models]
                if persisted is not None
                else models_to_select
            )

        selected_models = st.multiselect(
            "4️⃣ Select Model(s)",
            models,
            default=preserved_selections,
            key=prev_models_key,
        )

        # Add checkbox for selecting all models
        st.checkbox(
            "Select All Models",
            value=select_all_models_checked,
            key=select_all_models_key,
        )

    # Track model changes for auto-selecting downstream filters
    select_all_models_key = (
        f"select_all_models_{st.session_state.llmd_filter_change_key}"
    )
    select_all_checked = st.session_state.get(select_all_models_key, False)

    tracking_key = "llmd_previous_models_for_tp_tracking"
    prev_selected_models = st.session_state.get(tracking_key, None)
    models_changed = (
        (prev_selected_models != selected_models)
        if prev_selected_models is not None
        else (bool(selected_models))
    )

    # FILTER 5: TP size
    with filter_col5:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]  # type: ignore[assignment]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]  # type: ignore[assignment]

        tp_sizes = (
            sorted(temp_df["TP"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        prev_tp_key = f"llmd_tp_filter_{st.session_state.llmd_filter_change_key}_all_{select_all_checked}"
        prev_selected_tp = st.session_state.get(prev_tp_key, None)

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            tp_default = []
        elif models_changed and selected_models or select_all_checked:
            tp_default = tp_sizes
        elif prev_selected_tp is not None:
            tp_default = [tp for tp in prev_selected_tp if tp in tp_sizes]
        elif st.session_state.get("llmd_persisted_tp") is not None:
            tp_default = [
                tp for tp in st.session_state.llmd_persisted_tp if tp in tp_sizes
            ]
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_tp = st.session_state.get("llmd_baseline_tp", tp_sizes)
            tp_default = [tp for tp in baseline_tp if tp in tp_sizes]
        else:
            baseline_tp = st.session_state.get("llmd_baseline_tp", tp_sizes)
            tp_default = [tp for tp in baseline_tp if tp in tp_sizes]

        selected_tp = st.multiselect(
            "5️⃣ Select TP Size(s)",
            tp_sizes,
            default=tp_default,
            key=prev_tp_key,
        )

    # FILTER 6: Replicas
    with filter_col6:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]  # type: ignore[assignment]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]  # type: ignore[assignment]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]  # type: ignore[assignment]

        replicas = (
            sorted(temp_df["replicas"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        prev_replicas_key = f"llmd_replicas_filter_{st.session_state.llmd_filter_change_key}_all_{select_all_checked}"
        prev_selected_replicas = st.session_state.get(prev_replicas_key, None)

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            replicas_default = []
        elif models_changed and selected_models or select_all_checked:
            replicas_default = replicas
        elif prev_selected_replicas is not None:
            replicas_default = [r for r in prev_selected_replicas if r in replicas]
        elif st.session_state.get("llmd_persisted_replicas") is not None:
            replicas_default = [
                r for r in st.session_state.llmd_persisted_replicas if r in replicas
            ]
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_replicas = st.session_state.get("llmd_baseline_replicas", replicas)
            if not isinstance(baseline_replicas, list):
                baseline_replicas = [baseline_replicas] if baseline_replicas else []
            replicas_default = [r for r in baseline_replicas if r in replicas]
        else:
            baseline_replicas = st.session_state.get("llmd_baseline_replicas", replicas)
            if not isinstance(baseline_replicas, list):
                baseline_replicas = [baseline_replicas] if baseline_replicas else []
            replicas_default = [r for r in baseline_replicas if r in replicas]

        selected_replicas = st.multiselect(
            "6️⃣ Select # of Replicas",
            replicas,
            default=replicas_default,
            key=prev_replicas_key,
        )

    # FILTER 7: Prefill Pod Count
    with filter_col7:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]  # type: ignore[assignment]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]  # type: ignore[assignment]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]  # type: ignore[assignment]
        if selected_replicas:
            temp_df = temp_df[temp_df["replicas"].isin(selected_replicas)]  # type: ignore[assignment]

        _prefill_numeric = (
            sorted(temp_df["prefill_pod_count"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )
        _has_prefill_na = (
            not temp_df.empty and temp_df["prefill_pod_count"].isna().any()
        )
        prefill_pods = _prefill_numeric + (["N/A"] if _has_prefill_na else [])

        prev_prefill_key = f"llmd_prefill_filter_{st.session_state.llmd_filter_change_key}_all_{select_all_checked}"
        prev_selected_prefill = st.session_state.get(prev_prefill_key, None)

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            prefill_default = []
        elif models_changed and selected_models or select_all_checked:
            prefill_default = prefill_pods
        elif prev_selected_prefill is not None:
            prefill_default = [p for p in prev_selected_prefill if p in prefill_pods]
        elif st.session_state.get("llmd_persisted_prefill") is not None:
            prefill_default = [
                p for p in st.session_state.llmd_persisted_prefill if p in prefill_pods
            ]
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_prefill = st.session_state.get(
                "llmd_baseline_prefill", prefill_pods
            )
            prefill_default = [p for p in baseline_prefill if p in prefill_pods]
        else:
            baseline_prefill = st.session_state.get(
                "llmd_baseline_prefill", prefill_pods
            )
            prefill_default = [p for p in baseline_prefill if p in prefill_pods]

        selected_prefill_pods = st.multiselect(
            "7️⃣ Select Prefill Pod Count",
            prefill_pods,
            default=prefill_default,
            key=prev_prefill_key,
        )

    # FILTER 8: Decode Pod Count
    with filter_col8:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]  # type: ignore[assignment]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]  # type: ignore[assignment]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]  # type: ignore[assignment]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]  # type: ignore[assignment]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]  # type: ignore[assignment]
        if selected_replicas:
            temp_df = temp_df[temp_df["replicas"].isin(selected_replicas)]  # type: ignore[assignment]
        if selected_prefill_pods:
            _pf_nums = [v for v in selected_prefill_pods if v != "N/A"]
            _pf_na = "N/A" in selected_prefill_pods
            temp_df = temp_df[  # type: ignore[assignment]
                temp_df["prefill_pod_count"].isin(_pf_nums)
                | (_pf_na & temp_df["prefill_pod_count"].isna())
            ]

        _decode_numeric = (
            sorted(temp_df["decode_pod_count"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )
        _has_decode_na = not temp_df.empty and temp_df["decode_pod_count"].isna().any()
        decode_pods = _decode_numeric + (["N/A"] if _has_decode_na else [])

        prev_decode_key = f"llmd_decode_filter_{st.session_state.llmd_filter_change_key}_all_{select_all_checked}"
        prev_selected_decode = st.session_state.get(prev_decode_key, None)

        if st.session_state.get(
            "llmd_clear_all_filters", False
        ) or st.session_state.get("llmd_filters_were_cleared", False):
            decode_default = []
        elif models_changed and selected_models or select_all_checked:
            decode_default = decode_pods
        elif prev_selected_decode is not None:
            decode_default = [d for d in prev_selected_decode if d in decode_pods]
        elif st.session_state.get("llmd_persisted_decode") is not None:
            decode_default = [
                d for d in st.session_state.llmd_persisted_decode if d in decode_pods
            ]
        elif st.session_state.get("llmd_reset_to_defaults", False):
            baseline_decode = st.session_state.get("llmd_baseline_decode", decode_pods)
            decode_default = [d for d in baseline_decode if d in decode_pods]
        else:
            baseline_decode = st.session_state.get("llmd_baseline_decode", decode_pods)
            decode_default = [d for d in baseline_decode if d in decode_pods]

        selected_decode_pods = st.multiselect(
            "8️⃣ Select Decode Pod Count",
            decode_pods,
            default=decode_default,
            key=prev_decode_key,
        )

    # Filter action buttons
    _, btn_col2, btn_col3 = st.columns([3.5, 1, 1])

    with btn_col2:
        with st.popover("❓ Filters Help", use_container_width=True):
            st.markdown("### ✅ Valid Filter Combinations")
            st.markdown("View all valid combinations of filters:")

            # Selector for tree view type
            tree_view = st.radio(
                "Group by:",
                options=["Model", "Version"],
                horizontal=True,
                key="llmd_filter_help_tree_view",
            )

            if tree_view == "Model":
                # Group by Model → Accelerator → Version → Profile → TP → Replicas → Pods
                models = sorted(df["model"].unique())

                for model in models:
                    model_short = model.split("/")[-1] if "/" in model else model
                    model_data = df[df["model"] == model]

                    with st.expander(f"🤖 {model_short}", expanded=False):
                        combo_dict: dict[
                            str,
                            dict[
                                str,
                                dict[str, dict[int, dict[int, set[tuple[int, int]]]]],
                            ],
                        ] = {}
                        for _, row in model_data.iterrows():
                            acc = row["accelerator"]
                            version = row["version"]
                            profile = row["profile"]
                            tp = row["TP"]
                            replicas = row["replicas"]
                            prefill = row["prefill_pod_count"]
                            decode = row["decode_pod_count"]

                            if acc not in combo_dict:
                                combo_dict[acc] = {}
                            if version not in combo_dict[acc]:
                                combo_dict[acc][version] = {}
                            if profile not in combo_dict[acc][version]:
                                combo_dict[acc][version][profile] = {}
                            if tp not in combo_dict[acc][version][profile]:
                                combo_dict[acc][version][profile][tp] = {}
                            if replicas not in combo_dict[acc][version][profile][tp]:
                                combo_dict[acc][version][profile][tp][replicas] = set()

                            _p = "N/A" if pd.isna(prefill) else int(prefill)
                            _d = "N/A" if pd.isna(decode) else int(decode)
                            combo_dict[acc][version][profile][tp][replicas].add(
                                (_p, _d)  # type: ignore[arg-type]
                            )

                        tree_text = ""
                        for acc in sorted(combo_dict.keys()):
                            tree_text += f"🔧 {acc}\n"

                            versions = sorted(combo_dict[acc].keys())
                            for version in versions:
                                tree_text += f"    📦 {version}\n"

                                profiles = sorted(combo_dict[acc][version].keys())
                                for profile in profiles:
                                    profile_display = clean_profile_name(profile)
                                    tree_text += f"        📋 {profile_display}\n"

                                    tps = sorted(
                                        combo_dict[acc][version][profile].keys()
                                    )
                                    for tp in tps:
                                        tree_text += f"            🔢 TP: {tp}\n"

                                        replicas_list = sorted(
                                            combo_dict[acc][version][profile][tp].keys()
                                        )
                                        for replica in replicas_list:
                                            pods = sorted(
                                                combo_dict[acc][version][profile][tp][
                                                    replica
                                                ],
                                                key=str,
                                            )
                                            pods_str = ", ".join(
                                                [f"({p}/{d})" for p, d in pods]
                                            )
                                            tree_text += f"                👥 Replicas: {replica} → Pods(P/D): {pods_str}\n"
                            tree_text += "\n"

                        st.code(tree_text, language=None)

            else:  # Group by Version
                # Group by Version → Accelerator → Model → Profile → TP → Replicas → Pods
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
                            replicas = row["replicas"]
                            prefill = row["prefill_pod_count"]
                            decode = row["decode_pod_count"]

                            if acc not in combo_dict:
                                combo_dict[acc] = {}
                            if model_short not in combo_dict[acc]:
                                combo_dict[acc][model_short] = {}
                            if profile not in combo_dict[acc][model_short]:
                                combo_dict[acc][model_short][profile] = {}
                            if tp not in combo_dict[acc][model_short][profile]:
                                combo_dict[acc][model_short][profile][tp] = {}
                            if (
                                replicas
                                not in combo_dict[acc][model_short][profile][tp]
                            ):
                                combo_dict[acc][model_short][profile][tp][replicas] = (
                                    set()
                                )

                            _p = "N/A" if pd.isna(prefill) else int(prefill)
                            _d = "N/A" if pd.isna(decode) else int(decode)
                            combo_dict[acc][model_short][profile][tp][replicas].add(
                                (_p, _d)  # type: ignore[arg-type]
                            )

                        tree_text = ""
                        for acc in sorted(combo_dict.keys()):
                            tree_text += f"🔧 {acc}\n"

                            models = sorted(combo_dict[acc].keys())
                            for model_short in models:
                                tree_text += f"    🤖 {model_short}\n"

                                profiles = sorted(combo_dict[acc][model_short].keys())
                                for profile in profiles:
                                    profile_display = clean_profile_name(profile)
                                    tree_text += f"        📋 {profile_display}\n"

                                    tps = sorted(
                                        combo_dict[acc][model_short][profile].keys()
                                    )
                                    for tp in tps:
                                        tree_text += f"            🔢 TP: {tp}\n"

                                        replicas_list = sorted(
                                            combo_dict[acc][model_short][profile][
                                                tp
                                            ].keys()
                                        )
                                        for replica in replicas_list:
                                            pods = sorted(
                                                combo_dict[acc][model_short][profile][
                                                    tp
                                                ][replica],
                                                key=str,
                                            )
                                            pods_str = ", ".join(
                                                [f"({p}/{d})" for p, d in pods]
                                            )
                                            tree_text += f"                👥 Replicas: {replica} → Pods(P/D): {pods_str}\n"
                            tree_text += "\n"

                        st.code(tree_text, language=None)

    with btn_col3:
        if st.button(
            "↩ Reset to Defaults",
            help="Reset filters to default values",
            key="llmd_reset_btn",
        ):
            st.session_state.llmd_clear_all_filters = False
            st.session_state.llmd_filters_were_cleared = False
            st.session_state.llmd_reset_to_defaults = True
            st.session_state.llmd_filter_change_key += 1
            st.session_state.llmd_previous_models_for_tp_tracking = None
            st.session_state.performance_plots_expanded = False
            st.session_state.rhaiis_comparison_expanded = False
            st.session_state.llmd_compare_versions_expanded = False
            st.session_state.runtime_configs_expanded = False
            if "llmd_previous_filter_state" in st.session_state:
                del st.session_state.llmd_previous_filter_state
            st.rerun()

    # Update model tracking variable for TP/downstream auto-select detection
    st.session_state[tracking_key] = selected_models

    if st.session_state.get("llmd_clear_all_filters", False):
        st.session_state.llmd_clear_all_filters = False
    if st.session_state.get("llmd_reset_to_defaults", False):
        st.session_state.llmd_reset_to_defaults = False

    # Apply all filters unconditionally (empty selection → empty result, not "show all")
    custom_mask = (
        (df["custom_isl_osl"] == selected_custom_isl_osl)
        if selected_custom_isl_osl and "custom_isl_osl" in df.columns
        else True
    )

    _pf_nums = [v for v in selected_prefill_pods if v != "N/A"]
    _pf_na = "N/A" in selected_prefill_pods
    prefill_mask = (
        df["prefill_pod_count"].isin(_pf_nums)
        | (_pf_na & df["prefill_pod_count"].isna())
        if selected_prefill_pods
        else True
    )

    _dc_nums = [v for v in selected_decode_pods if v != "N/A"]
    _dc_na = "N/A" in selected_decode_pods
    decode_mask = (
        df["decode_pod_count"].isin(_dc_nums) | (_dc_na & df["decode_pod_count"].isna())
        if selected_decode_pods
        else True
    )

    filtered_df: pd.DataFrame = df[  # type: ignore[assignment]
        df["accelerator"].isin(selected_accelerators)
        & df["model"].isin(selected_models)
        & df["version"].isin(selected_versions)
        & (df["profile"].isin(selected_profiles) if selected_profiles else True)
        & df["TP"].isin(selected_tp)
        & df["replicas"].isin(selected_replicas)
        & custom_mask
        & prefill_mask
        & decode_mask
    ].copy()

    filter_selections = {
        "accelerators": selected_accelerators,
        "profiles": selected_profiles,
        "versions": selected_versions,
        "models": selected_models,
        "tp_sizes": selected_tp,
        "replicas": selected_replicas,
        "prefill_pods": selected_prefill_pods,
        "decode_pods": selected_decode_pods,
    }

    # Detect filter changes and collapse expanders
    current_filter_state = {
        "accelerators": tuple(sorted(selected_accelerators)),
        "models": tuple(sorted(selected_models)),
        "versions": tuple(sorted(selected_versions)),
        "profile": selected_profile,
        "tp": tuple(sorted(selected_tp)),
        "replicas": tuple(sorted(selected_replicas)),
        "prefill_pods": tuple(sorted(selected_prefill_pods)),
        "decode_pods": tuple(sorted(selected_decode_pods)),
    }

    previous_filter_state = st.session_state.get("llmd_previous_filter_state", None)

    if (
        previous_filter_state is not None
        and previous_filter_state != current_filter_state
    ):
        st.session_state.performance_plots_expanded = False
        st.session_state.rhaiis_comparison_expanded = False
        st.session_state.llmd_compare_versions_expanded = False
        st.session_state.runtime_configs_expanded = False

    st.session_state.llmd_previous_filter_state = current_filter_state

    # Persist filter selections so they survive section switches
    st.session_state.llmd_persisted_accelerators = list(selected_accelerators)
    st.session_state.llmd_persisted_profile = selected_profile
    st.session_state.llmd_persisted_profiles = list(selected_profiles)
    st.session_state.llmd_persisted_versions = list(selected_versions)
    st.session_state.llmd_persisted_models = list(selected_models)
    st.session_state.llmd_persisted_tp = list(selected_tp)
    st.session_state.llmd_persisted_replicas = list(selected_replicas)
    st.session_state.llmd_persisted_prefill = list(selected_prefill_pods)
    st.session_state.llmd_persisted_decode = list(selected_decode_pods)

    # Store baseline values
    if not st.session_state.get("llmd_baseline_accelerators"):
        st.session_state.llmd_baseline_accelerators = accelerators
        st.session_state.llmd_baseline_profile = profiles[0] if profiles else None
        rhoai_versions = [v for v in versions if v.startswith("RHOAI")]
        st.session_state.llmd_baseline_versions = (
            rhoai_versions if rhoai_versions else versions[:1]
        )
        st.session_state.llmd_baseline_models = models[:1] if models else []
        st.session_state.llmd_baseline_tp = tp_sizes
        st.session_state.llmd_baseline_replicas = replicas
        st.session_state.llmd_baseline_prefill = prefill_pods
        st.session_state.llmd_baseline_decode = decode_pods

    return filtered_df, filter_selections


@st.cache_data(ttl=300)
def load_rhaiis_data(file_path: str = "consolidated_dashboard.csv") -> pd.DataFrame:
    """Load RHAIIS data for comparison from S3 or local file.

    If S3_BUCKET environment variable is set, data is loaded from S3.
    Otherwise, falls back to local file system.

    Args:
        file_path: Path to the RHAIIS CSV file (fallback).

    Returns:
        DataFrame with RHAIIS data.
    """
    try:
        # Try S3 first if configured
        if S3_BUCKET:
            try:
                df = _read_csv_from_s3(S3_BUCKET, S3_KEY, S3_REGION)
                logger.info(
                    f"Successfully loaded RHAIIS data from S3: s3://{S3_BUCKET}/{S3_KEY}"
                )
            except Exception as s3_error:
                logger.warning(
                    f"S3 load failed ({s3_error}), falling back to local file"
                )
                df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)

        # Strip whitespace from string columns
        col: str
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()  # type: ignore[assignment]

        # Assign profile based on prompt/output tokens (same as llmd_data)
        if "prompt toks" in df.columns and "output toks" in df.columns:
            df["profile"] = df.apply(assign_profile, axis=1)

        return df
    except Exception as e:
        logger.error(f"Error loading RHAIIS data: {str(e)}")
        st.error(f"Error loading RHAIIS data: {str(e)}")
        return pd.DataFrame()


def render_rhaiis_comparison_section(llmd_filtered_df: pd.DataFrame, use_expander=True):
    """Render comparison section between LLM-D and RHAIIS.

    Args:
        llmd_filtered_df: Filtered LLM-D DataFrame
        use_expander: Whether to wrap content in a collapsible expander.
    """
    if use_expander:
        if "rhaiis_comparison_expanded" not in st.session_state:
            st.session_state.rhaiis_comparison_expanded = False
        ctx = st.expander(
            "🔄 Compare with RHAIIS",
            expanded=st.session_state.rhaiis_comparison_expanded,
        )
    else:
        ctx = contextlib.nullcontext()  # type: ignore[assignment]
    with ctx:
        if not use_expander:
            st.subheader("🔄 Compare with RHAIIS")
        st.markdown(
            """
            **Compare disaggregated LLM-D performance with traditional RHAIIS architecture.**
            This section compares LLM-D runs (with 1 replica) against RHAIIS runs for the same model configurations.
            """
        )

        # Inject CSS to increase tab label size
        st.markdown(
            """
            <style>
            .stTabs [data-baseweb="tab"] {
                font-size: 18px !important;
                font-weight: 600 !important;
                padding-top: 12px !important;
                padding-bottom: 12px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Load RHAIIS data
        rhaiis_df = load_rhaiis_data()

        # Get accelerators from filtered LLM-D data
        llmd_accelerators = (
            set(llmd_filtered_df["accelerator"].unique())
            if "accelerator" in llmd_filtered_df.columns
            else set()
        )

        # Show which accelerators are being compared
        if llmd_accelerators:
            st.info(
                f"🔧 Comparing on accelerator(s): {', '.join(sorted(llmd_accelerators))}"
            )

        # Filter RHAIIS data to match the same accelerators
        if llmd_accelerators and "accelerator" in rhaiis_df.columns:
            rhaiis_df = rhaiis_df[  # type: ignore[assignment]
                rhaiis_df["accelerator"].isin(llmd_accelerators)
            ].copy()

        if rhaiis_df.empty:
            st.warning("⚠️ RHAIIS data could not be loaded.")
            return

        # Filter LLM-D data to only 1 replica runs
        llmd_single_replica = llmd_filtered_df[llmd_filtered_df["replicas"] == 1].copy()

        if llmd_single_replica.empty:
            st.info(
                "ℹ️ No LLM-D runs with 1 replica found in current filters. Please adjust your filters to include single replica runs."
            )
            return

        # Get common models between LLM-D and RHAIIS
        llmd_models = set(llmd_single_replica["model"].unique())
        rhaiis_models = set(rhaiis_df["model"].unique())
        common_models = sorted(llmd_models.intersection(rhaiis_models))

        if not common_models:
            st.warning("⚠️ No common models found between LLM-D and RHAIIS datasets.")
            return

        # Filters for comparison
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            selected_comparison_model = st.selectbox(
                "Select Model to Compare",
                options=common_models,
                format_func=lambda x: x.split("/")[-1] if "/" in x else x,
                key="rhaiis_comparison_model",
            )

        with filter_col2:
            # Get LLM-D versions available for the selected model
            llmd_model_data = llmd_single_replica[
                llmd_single_replica["model"] == selected_comparison_model
            ]
            llmd_versions = sorted(llmd_model_data["version"].unique().tolist())

            if not llmd_versions:
                st.warning("⚠️ No LLM-D versions found for this model.")
                return

            if "rhaiis_comparison_llmd_version_selected" not in st.session_state:
                rhoai_defaults = [v for v in llmd_versions if v.startswith("RHOAI")]
                st.session_state.rhaiis_comparison_llmd_version_selected = (
                    rhoai_defaults[0] if rhoai_defaults else llmd_versions[0]
                )

            llmd_version_default_index = 0
            if (
                st.session_state.rhaiis_comparison_llmd_version_selected
                in llmd_versions
            ):
                llmd_version_default_index = llmd_versions.index(
                    st.session_state.rhaiis_comparison_llmd_version_selected
                )

            selected_llmd_version = st.selectbox(
                "Select LLM-D Version",
                options=llmd_versions,
                index=llmd_version_default_index,
                key="rhaiis_comparison_llmd_version",
            )

            st.session_state.rhaiis_comparison_llmd_version_selected = (
                selected_llmd_version
            )

            llmd_model_data = llmd_model_data[
                llmd_model_data["version"] == selected_llmd_version
            ]

        with filter_col3:
            # Get RHAIIS versions for the selected model (only versions containing "RHAIIS")
            rhaiis_model_data = rhaiis_df[
                rhaiis_df["model"] == selected_comparison_model
            ]
            all_versions = rhaiis_model_data["version"].unique()
            rhaiis_versions = sorted([v for v in all_versions if "RHAIIS" in str(v)])

            if not rhaiis_versions:
                st.warning("⚠️ No RHAIIS versions found for this model.")
                return

            if "rhaiis_comparison_version_selected" not in st.session_state:
                if "RHAIIS-3.2.3" in rhaiis_versions:
                    st.session_state.rhaiis_comparison_version_selected = "RHAIIS-3.2.3"
                else:
                    st.session_state.rhaiis_comparison_version_selected = (
                        rhaiis_versions[0]
                    )

            default_version_index = 0
            if st.session_state.rhaiis_comparison_version_selected in rhaiis_versions:
                default_version_index = rhaiis_versions.index(
                    st.session_state.rhaiis_comparison_version_selected
                )
            elif "RHAIIS-3.2.3" in rhaiis_versions:
                default_version_index = rhaiis_versions.index("RHAIIS-3.2.3")
                st.session_state.rhaiis_comparison_version_selected = "RHAIIS-3.2.3"

            selected_rhaiis_version = st.selectbox(
                "Select RHAIIS Version",
                options=rhaiis_versions,
                index=default_version_index,
                key="rhaiis_comparison_version",
            )

            st.session_state.rhaiis_comparison_version_selected = (
                selected_rhaiis_version
            )

        with filter_col4:
            rhaiis_version_data = rhaiis_model_data[
                rhaiis_model_data["version"] == selected_rhaiis_version
            ]

            if (
                "profile" not in llmd_model_data.columns
                or "profile" not in rhaiis_version_data.columns
            ):
                st.warning(
                    "⚠️ Profile column not found in data. Cannot compare profiles."
                )
                return

            llmd_profiles = set(llmd_model_data["profile"].unique())
            rhaiis_profiles = set(rhaiis_version_data["profile"].unique())
            common_profiles = sorted(llmd_profiles.intersection(rhaiis_profiles))

            if not common_profiles:
                st.warning(
                    "⚠️ No common profiles found for this model/version combination."
                )
                return

            if "rhaiis_comparison_profile_selected" not in st.session_state:
                st.session_state.rhaiis_comparison_profile_selected = common_profiles[0]

            default_profile_index = 0
            if st.session_state.rhaiis_comparison_profile_selected in common_profiles:
                default_profile_index = common_profiles.index(
                    st.session_state.rhaiis_comparison_profile_selected
                )

            selected_comparison_profile = st.selectbox(
                "Select Profile (ISL/OSL)",
                options=common_profiles,
                format_func=clean_profile_name,
                index=default_profile_index,
                key="rhaiis_comparison_profile",
            )

            st.session_state.rhaiis_comparison_profile_selected = (
                selected_comparison_profile
            )

        # Filter both datasets for comparison
        llmd_comparison = llmd_model_data[
            llmd_model_data["profile"] == selected_comparison_profile
        ].copy()
        rhaiis_comparison = rhaiis_version_data[
            rhaiis_version_data["profile"] == selected_comparison_profile
        ].copy()

        if llmd_comparison.empty or rhaiis_comparison.empty:
            st.warning("⚠️ No data available for the selected combination.")
            return

        # Add architecture identifier
        llmd_comparison["architecture"] = "LLM-D"
        rhaiis_comparison["architecture"] = "RHAIIS"

        # Create configuration labels with full version names
        llmd_comparison["config_label"] = (
            "LLM-D ("
            + llmd_comparison["version"].astype(str)
            + ") | TP="
            + llmd_comparison["TP"].astype(str)
            + " | P="
            + llmd_comparison["prefill_pod_count"].fillna("N/A").astype(str)
            + "/D="
            + llmd_comparison["decode_pod_count"].fillna("N/A").astype(str)
        )
        rhaiis_comparison["config_label"] = (
            "RHAIIS ("
            + rhaiis_comparison["version"].astype(str)
            + ") | TP="
            + rhaiis_comparison["TP"].astype(str)
        )

        # Combine for plotting
        combined_df = pd.concat([llmd_comparison, rhaiis_comparison], ignore_index=True)

        # Create comparison plots
        st.markdown("---")
        st.markdown("## 📊 Performance Comparison")

        tab1, tab2, tab3 = st.tabs(
            [
                "📈 Throughput vs Concurrency",
                "⚡ Latency Comparison",
                "📊 Detailed Metrics",
            ]
        )

        with tab1:
            # Throughput comparison
            fig = px.line(
                combined_df.sort_values("intended concurrency"),
                x="intended concurrency",
                y="output_tok/sec",
                color="config_label",
                markers=True,
                title=f"Output Throughput: LLM-D vs RHAIIS<br><sub>Model: {selected_comparison_model.split('/')[-1]} | {clean_profile_name(selected_comparison_profile)}</sub>",
                labels={
                    "intended concurrency": "Concurrency",
                    "output_tok/sec": "Output Tokens/sec",
                    "config_label": "Configuration",
                },
                template="plotly_white_light",
            )
            fig.update_layout(
                legend={
                    "font": {"size": 12},
                    "orientation": "v",
                    "yanchor": "top",
                    "y": 1,
                    "xanchor": "left",
                    "x": 1.02,
                },
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with tab2:
            # Latency comparison
            col1, col2 = st.columns(2)

            with col1:
                fig_ttft = px.line(
                    combined_df.sort_values("intended concurrency"),
                    x="intended concurrency",
                    y="ttft_median",
                    color="config_label",
                    markers=True,
                    title="TTFT Median (ms)<br><sub>Lower is Better</sub>",
                    labels={
                        "intended concurrency": "Concurrency",
                        "ttft_median": "TTFT Median (ms)",
                        "config_label": "Configuration",
                    },
                    template="plotly_white_light",
                )
                fig_ttft.update_layout(
                    legend={"font": {"size": 10}},
                    height=400,
                )
                st.plotly_chart(fig_ttft, use_container_width=True, theme=None)

            with col2:
                fig_tpot = px.line(
                    combined_df.sort_values("intended concurrency"),
                    x="intended concurrency",
                    y="tpot_median",
                    color="config_label",
                    markers=True,
                    title="TPOT Median (ms)<br><sub>Lower is Better</sub>",
                    labels={
                        "intended concurrency": "Concurrency",
                        "tpot_median": "TPOT Median (ms)",
                        "config_label": "Configuration",
                    },
                    template="plotly_white_light",
                )
                fig_tpot.update_layout(
                    legend={"font": {"size": 10}},
                    height=400,
                )
                st.plotly_chart(fig_tpot, use_container_width=True, theme=None)

        with tab3:
            # Summary statistics comparison
            st.markdown("### 📈 Summary Statistics by Configuration")

            # Calculate aggregated metrics
            summary_data = []
            for config in combined_df["config_label"].unique():
                config_data = combined_df[combined_df["config_label"] == config]
                arch = config_data["architecture"].iloc[0]

                summary_data.append(
                    {
                        "Configuration": config,
                        "Architecture": arch,
                        "Avg Output Tok/s": config_data["output_tok/sec"].mean(),
                        "Max Output Tok/s": config_data["output_tok/sec"].max(),
                        "Avg TTFT (ms)": config_data["ttft_median"].mean(),
                        "Avg TPOT (ms)": config_data["tpot_median"].mean(),
                        "Avg Request Latency (s)": config_data[
                            "request_latency_median"
                        ].mean(),
                    }
                )

            summary_df = pd.DataFrame(summary_data)

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Configuration": st.column_config.TextColumn(
                        "Configuration", width=300
                    ),
                    "Architecture": st.column_config.TextColumn(
                        "Architecture", width=100
                    ),
                    "Avg Output Tok/s": st.column_config.NumberColumn(
                        "Avg Output Tok/s",
                        help="Average output tokens per second",
                        format="%.2f",
                    ),
                    "Max Output Tok/s": st.column_config.NumberColumn(
                        "Max Output Tok/s",
                        help="Maximum output tokens per second achieved",
                        format="%.2f",
                    ),
                    "Avg TTFT (ms)": st.column_config.NumberColumn(
                        "Avg TTFT (ms)",
                        help="Average Time to First Token (lower is better)",
                        format="%.2f",
                    ),
                    "Avg TPOT (ms)": st.column_config.NumberColumn(
                        "Avg TPOT (ms)",
                        help="Average Time Per Output Token (lower is better)",
                        format="%.2f",
                    ),
                    "Avg Request Latency (s)": st.column_config.NumberColumn(
                        "Avg Request Latency (s)",
                        help="Average end-to-end request latency (lower is better)",
                        format="%.2f",
                    ),
                },
            )

            st.markdown("---")
            st.markdown("### 📋 Detailed Data")

            # Show detailed comparison data
            detail_cols = [
                "architecture",
                "config_label",
                "intended concurrency",
                "output_tok/sec",
                "total_tok/sec",
                "ttft_median",
                "tpot_median",
                "request_latency_median",
                "successful_requests",
                "errored_requests",
            ]

            detail_df = (
                combined_df[detail_cols]
                .sort_values(["architecture", "config_label", "intended concurrency"])
                .copy()
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "architecture": st.column_config.TextColumn(
                        "Architecture", width=100
                    ),
                    "config_label": st.column_config.TextColumn(
                        "Configuration", width=250
                    ),
                    "intended concurrency": st.column_config.NumberColumn(
                        "Concurrency"
                    ),
                    "output_tok/sec": st.column_config.NumberColumn(
                        "Output Tok/s", format="%.2f"
                    ),
                    "total_tok/sec": st.column_config.NumberColumn(
                        "Total Tok/s", format="%.2f"
                    ),
                    "ttft_median": st.column_config.NumberColumn(
                        "TTFT (ms)", format="%.2f"
                    ),
                    "tpot_median": st.column_config.NumberColumn(
                        "TPOT (ms)", format="%.2f"
                    ),
                    "request_latency_median": st.column_config.NumberColumn(
                        "Latency (s)", format="%.2f"
                    ),
                    "successful_requests": st.column_config.NumberColumn("Success"),
                    "errored_requests": st.column_config.NumberColumn("Errors"),
                },
            )


@st.fragment
def render_compare_versions_section(df, use_expander=True):
    """Compare two LLM-D versions across multiple metrics."""
    if use_expander:
        if "llmd_compare_versions_expanded" not in st.session_state:
            st.session_state.llmd_compare_versions_expanded = False
        ctx = st.expander(
            "⚖️ Compare Versions",
            expanded=st.session_state.llmd_compare_versions_expanded,
        )
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("⚖️ Compare Versions")
        st.markdown(
            "💡 **Generate a summary table comparing performance between two LLM-D versions across all models and metrics.**"
        )

        available_versions = sorted(df["version"].unique().tolist())
        available_accelerators = sorted(df["accelerator"].unique().tolist())
        available_profiles_raw = sorted(df["profile"].unique().tolist())
        available_profiles = [p for p in available_profiles_raw if p != "Custom"] + (
            ["Custom"] if "Custom" in available_profiles_raw else []
        )

        if len(available_versions) < 2:
            st.warning("⚠️ Need at least 2 versions in the data to compare.")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            version_1 = st.selectbox(
                "Select Version 1 (Baseline)",
                options=available_versions,
                index=0,
                key="llmd_compare_v1",
                on_change=_keep_expander_open,
                args=("llmd_compare_versions_expanded",),
            )

        with col2:
            version_2_options = [v for v in available_versions if v != version_1]
            version_2 = (
                st.selectbox(
                    "Select Version 2 (Comparison)",
                    options=version_2_options,
                    index=0 if version_2_options else None,
                    key="llmd_compare_v2",
                    on_change=_keep_expander_open,
                    args=("llmd_compare_versions_expanded",),
                )
                if version_2_options
                else None
            )

        with col3:
            accel_default_index = 0
            if "H200" in available_accelerators:
                accel_default_index = available_accelerators.index("H200")
            selected_accelerator = st.selectbox(
                "Select GPU",
                options=available_accelerators,
                index=accel_default_index,
                key="llmd_compare_accelerator",
                on_change=_keep_expander_open,
                args=("llmd_compare_versions_expanded",),
            )

        with col4:
            selected_profile = st.selectbox(
                "Select ISL/OSL Profile",
                options=available_profiles,
                index=0,
                format_func=clean_profile_name,
                key="llmd_compare_profile",
                on_change=_keep_expander_open,
                args=("llmd_compare_versions_expanded",),
            )

        # Secondary custom ISL/OSL pair filter
        selected_custom_pair = None
        if selected_profile == "Custom" and "custom_isl_osl" in df.columns:
            custom_temp = df[df["profile"] == "Custom"]
            custom_pairs = sorted(custom_temp["custom_isl_osl"].unique().tolist())
            custom_pairs = [p for p in custom_pairs if p]
            if custom_pairs:
                selected_custom_pair = st.selectbox(
                    "Select Custom ISL/OSL Pair",
                    options=custom_pairs,
                    format_func=format_custom_isl_osl,
                    key="llmd_compare_custom_isl_osl",
                    on_change=_keep_expander_open,
                    args=("llmd_compare_versions_expanded",),
                )

        if not version_2:
            st.warning("⚠️ Please select a second version to compare.")
            return

        base_mask_v1 = (
            (df["version"] == version_1)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        )
        base_mask_v2 = (
            (df["version"] == version_2)
            & (df["accelerator"] == selected_accelerator)
            & (df["profile"] == selected_profile)
        )
        if selected_custom_pair:
            base_mask_v1 = base_mask_v1 & (df["custom_isl_osl"] == selected_custom_pair)
            base_mask_v2 = base_mask_v2 & (df["custom_isl_osl"] == selected_custom_pair)
        df_v1 = df[base_mask_v1].copy()
        df_v2 = df[base_mask_v2].copy()

        if df_v1.empty or df_v2.empty:
            st.warning(
                "⚠️ No data available for the selected combination. "
                "Try different accelerator or profile settings."
            )
            return

        v1_model_tp = set(zip(df_v1["model"].tolist(), df_v1["TP"].tolist()))
        v2_model_tp = set(zip(df_v2["model"].tolist(), df_v2["TP"].tolist()))
        common_model_tp = sorted(v1_model_tp.intersection(v2_model_tp))

        if not common_model_tp:
            st.warning(
                f"⚠️ No common model+TP combinations found between {version_1} and {version_2} "
                f"for {selected_accelerator} with profile {clean_profile_name(selected_profile)}."
            )
            return

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
            conc_key = f"llmd_compare_conc_{version_1}_{version_2}_{selected_accelerator}_{selected_profile}"
            selected_concurrencies = st.multiselect(
                "Select Concurrency Level(s) for Geometric Mean",
                options=all_common_concurrencies_sorted,
                default=all_common_concurrencies_sorted,
                key=conc_key,
                on_change=_keep_expander_open,
                args=("llmd_compare_versions_expanded",),
                help=(
                    "Choose which concurrency levels to include in geometric mean calculations. "
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

        if selected_custom_pair:
            profile_short = format_custom_isl_osl(selected_custom_pair)
        else:
            profile_short = clean_profile_name(selected_profile)

        title_col, popover_col = st.columns([5, 1])
        with title_col:
            st.markdown(f"### {selected_accelerator} GPU, ISL/OSL: {profile_short}")
        with popover_col:
            with st.popover("ℹ️ How are these calculated?"):
                st.markdown("""
**Geometric Mean Calculation:**
- Computes the geometric mean of the metric values across the selected concurrency levels
- Better for ratios/percentages because it respects multiplicative relationships

**Peak Calculation:**
- For throughput: compares the maximum throughput across all common concurrency levels
- Higher is better for throughput; lower is better for latency

**Percentage Interpretation:**
- **+X%** means V1's metric value is X% **higher** than V2's
- **-X%** means V1's metric value is X% **lower** than V2's

| Metric Type | +X% means | -X% means |
|-------------|-----------|-----------|
| **Throughput** | V1 is faster | V1 is slower |
| **Latency** | V1 is slower | V1 is faster |

**Status:** 🟢 Better (>=5%) | 🟡 Similar (<5%) | 🔴 Worse (>=5%)
                """)
        st.markdown(f"**Comparing:** {version_1} vs {version_2}")

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

        summary_data = []
        for model, tp in common_model_tp:
            model_short = model.split("/")[-1] if "/" in model else model
            v1_model_data = df_v1[(df_v1["model"] == model) & (df_v1["TP"] == tp)]
            v2_model_data = df_v2[(df_v2["model"] == model) & (df_v2["TP"] == tp)]
            tp_str = f"(TP={int(tp)})" if pd.notna(tp) else ""
            row_data = {"Model": f"{model_short} {tp_str}"}

            for metric_name, metric_config in metrics_config.items():
                pct_diff, v1_better, v1_peak, v2_peak, is_similar = (
                    _compare_two_datasets(
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

            @st.dialog("Version Comparison — Metric Details", width="large")
            def _show_metric_dialog(metric_name):
                mcfg = metrics_config[metric_name]
                col = mcfg["column"]
                agg = mcfg["aggregation"]

                display_title = metric_name.replace(" (Geometric Mean)", "").replace(
                    " (Peak)", ""
                )
                st.markdown(f"#### {display_title} vs Concurrency")
                st.markdown(
                    f"**{version_1}** vs **{version_2}** &nbsp;|&nbsp; "
                    f"**{selected_accelerator}** &nbsp;|&nbsp; ISL/OSL: **{profile_short}**"
                )

                _palette_v1 = [
                    "#EF553B",
                    "#FF7F0E",
                    "#D62728",
                    "#E377C2",
                    "#FF6692",
                    "#FFA15A",
                    "#FECB52",
                    "#F0027F",
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
                ]

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
                        {"label": lbl, "conc": cc_sorted, "v1": v1_by_c, "v2": v2_by_c}
                    )

                if not per_model:
                    st.warning("No data available for this metric.")
                    return

                if col == "ttft_p95":
                    for md in per_model:
                        md["v1"] = [
                            v / 1000 if v is not None else None for v in md["v1"]
                        ]
                        md["v2"] = [
                            v / 1000 if v is not None else None for v in md["v2"]
                        ]

                fig = go.Figure()
                for idx, md in enumerate(per_model):
                    c_v1 = _palette_v1[idx % len(_palette_v1)]
                    c_v2 = _palette_v2[idx % len(_palette_v2)]
                    x_vals = [int(c) for c in md["conc"]]

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
                    key=f"llmd_dlg_line_{metric_name}",
                    theme=None,
                )

                st.caption(
                    "💡 **Tip:** Click a legend entry to toggle it. "
                    "Double-click to isolate a single trace. "
                    f"Warm colors (reds/oranges) = **{version_1}**, "
                    f"cool colors (blues/greens) = **{version_2}**."
                )

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
                        key=f"llmd_cmp_btn_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.llmd_compare_versions_expanded = True
                        _show_metric_dialog(m_name)

            st.markdown("")

            st.markdown(
                "<div style='text-align: right;'>"
                "<span style='font-size: 0.85em; color: gray;'>"
                "💡 <b>Tip:</b> Hover over column headers to see detailed descriptions."
                "</span></div>",
                unsafe_allow_html=True,
            )

            column_config = {
                "Model": st.column_config.TextColumn(
                    "Model",
                    help="Model name with tensor parallelism (TP) configuration",
                ),
                "Peak Output Throughput": st.column_config.TextColumn(
                    "Peak Output Throughput",
                    help="Maximum output tokens/sec achieved. Shows peak concurrency for V1 vs V2.",
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
                f"🟢 {version_1} performs better than {version_2} | "
                f"🔴 {version_1} performs worse than {version_2} | "
                f"🟡 Similar Performance (< 5% difference)"
            )

            st.markdown("---")
            st.markdown("### 📋 Detailed Model Comparisons")
            st.markdown("*Click on a model to see detailed metrics comparison*")

            for idx, (model, tp) in enumerate(common_model_tp, 1):
                model_short = model.split("/")[-1] if "/" in model else model
                v1_model_data = df_v1[(df_v1["model"] == model) & (df_v1["TP"] == tp)]
                v2_model_data = df_v2[(df_v2["model"] == model) & (df_v2["TP"] == tp)]

                tp_val = int(tp) if pd.notna(tp) else "N/A"

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

                v1_peak_idx = v1_common["output_tok/sec"].idxmax()
                v2_peak_idx = v2_common["output_tok/sec"].idxmax()

                v1_peak_throughput = v1_common.loc[v1_peak_idx, "output_tok/sec"]
                v2_peak_throughput = v2_common.loc[v2_peak_idx, "output_tok/sec"]
                v1_peak_conc = int(v1_common.loc[v1_peak_idx, "intended concurrency"])
                v2_peak_conc = int(v2_common.loc[v2_peak_idx, "intended concurrency"])

                v1_total_throughput = v1_common.loc[v1_peak_idx, "total_tok/sec"]
                v2_total_throughput = v2_common.loc[v2_peak_idx, "total_tok/sec"]

                v1_e2e_latency = v1_common.loc[v1_peak_idx, "request_latency_median"]
                v2_e2e_latency = v2_common.loc[v2_peak_idx, "request_latency_median"]

                v1_ttft = v1_common.loc[v1_peak_idx, "ttft_p95"]
                v2_ttft = v2_common.loc[v2_peak_idx, "ttft_p95"]

                v1_itl = v1_common.loc[v1_peak_idx, "itl_p95"]
                v2_itl = v2_common.loc[v2_peak_idx, "itl_p95"]

                def _fmt(val, unit="", decimals=0, round_up=False):
                    import math

                    if pd.isna(val):
                        return "N/A"
                    if round_up:
                        if decimals == 0:
                            return f"~{int(math.ceil(val)):,}{unit}"
                        factor = 10**decimals
                        rounded_val = math.ceil(val * factor) / factor
                        return f"~{rounded_val:,.{decimals}f}{unit}"
                    if decimals == 0:
                        return f"~{int(val):,}{unit}"
                    return f"~{val:,.{decimals}f}{unit}"

                def _winner(v1_val, v2_val, higher_is_better, metric_label):
                    if pd.isna(v1_val) or pd.isna(v2_val) or v2_val == 0:
                        return "N/A"
                    pct = ((v1_val - v2_val) / v2_val) * 100
                    if higher_is_better:
                        if pct > 5:
                            return f"{version_1} has +{abs(pct):.1f}% higher {metric_label}"
                        elif pct < -5:
                            return f"{version_2} has +{abs(pct):.1f}% higher {metric_label}"
                        else:
                            return f"Similar (~{abs(pct):.1f}% difference)"
                    else:
                        if pct < -5:
                            return (
                                f"{version_1} has {abs(pct):.1f}% lower {metric_label}"
                            )
                        elif pct > 5:
                            return (
                                f"{version_2} has {abs(pct):.1f}% lower {metric_label}"
                            )
                        else:
                            return f"Similar (~{abs(pct):.1f}% difference)"

                with st.expander(f"{idx}. {model_short} (TP={tp_val})"):
                    detail_rows = [
                        {
                            "Metric": "Peak Output Throughput (output tok/s)",
                            version_1: f"{_fmt(v1_peak_throughput)} tok/s at {v1_peak_conc} concurrent users",
                            version_2: f"{_fmt(v2_peak_throughput)} tok/s at {v2_peak_conc} concurrent users",
                            "Difference/Winner": _winner(
                                v1_peak_throughput,
                                v2_peak_throughput,
                                True,
                                "peak output throughput",
                            ),
                        },
                        {
                            "Metric": "Total Throughput (input + output tok/s)",
                            version_1: f"{_fmt(v1_total_throughput)} tok/s at {v1_peak_conc} concurrent users",
                            version_2: f"{_fmt(v2_total_throughput)} tok/s at {v2_peak_conc} concurrent users",
                            "Difference/Winner": _winner(
                                v1_total_throughput,
                                v2_total_throughput,
                                True,
                                "total throughput",
                            ),
                        },
                        {
                            "Metric": "Median E2E Latency at Peak Throughput",
                            version_1: _fmt(v1_e2e_latency, "s", 0, round_up=True),
                            version_2: _fmt(v2_e2e_latency, "s", 0, round_up=True),
                            "Difference/Winner": _winner(
                                v1_e2e_latency, v2_e2e_latency, False, "E2E latency"
                            ),
                        },
                        {
                            "Metric": "TTFT P95 at Peak Throughput",
                            version_1: _fmt(v1_ttft / 1000, "s", 2, round_up=True)
                            if pd.notna(v1_ttft)
                            else "N/A",
                            version_2: _fmt(v2_ttft / 1000, "s", 2, round_up=True)
                            if pd.notna(v2_ttft)
                            else "N/A",
                            "Difference/Winner": _winner(
                                v1_ttft, v2_ttft, False, "P95 TTFT"
                            ),
                        },
                        {
                            "Metric": "ITL P95 at Peak Throughput",
                            version_1: _fmt(v1_itl, "ms", 0, round_up=True),
                            version_2: _fmt(v2_itl, "ms", 0, round_up=True),
                            "Difference/Winner": _winner(
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


def render_performance_plots_section(filtered_df, use_expander=True):
    """Render performance plots section for LLM-D dashboard."""
    if use_expander:
        if "performance_plots_expanded" not in st.session_state:
            st.session_state.performance_plots_expanded = False
        ctx = st.expander(
            "📈 Performance Plots",
            expanded=st.session_state.performance_plots_expanded,
        )
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("📈 Performance Plots")
        st.markdown(
            "💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view."
        )

        # Create run identifier for legend (using shortened model name for readability)
        filtered_df["model_short"] = filtered_df["model"].apply(
            lambda x: x.split("/")[-1] if pd.notna(x) else "Unknown"
        )
        filtered_df["run_identifier"] = (
            filtered_df["accelerator"]
            + " | "
            + filtered_df["model_short"]  # Use short model name instead of full name
            + " | "
            + filtered_df["version"]
            + " | TP="
            + filtered_df["TP"].astype(str)
            + " | R="
            + filtered_df["replicas"].astype(str)
            + " | P="
            + filtered_df["prefill_pod_count"].fillna("N/A").astype(str)
            + "/D="
            + filtered_df["decode_pod_count"].fillna("N/A").astype(str)
        )

        filtered_df_sorted = filtered_df.sort_values(
            ["model_short", "accelerator", "version", "TP", "replicas"]
        ).copy()

        col1, col2 = st.columns(2)
        with col1:
            x_axis_options = {
                "Concurrency": "intended concurrency",
                "Throughput (Output Tok/s)": "output_tok/sec",
            }
            x_axis_label = st.selectbox(
                "Select X-Axis",
                options=list(x_axis_options.keys()),
                key="llmd_perf_plots_x_axis",
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
                index=0,
                key="llmd_perf_plots_y_axis",
            )
            y_axis = y_axis_options[y_axis_label]

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
            template="plotly_white_light",
            category_orders={
                "run_identifier": filtered_df_sorted["run_identifier"].unique().tolist()
            },
        )
        fig.update_layout(
            legend_title_text="Run Details (Accelerator | Model | Version | TP | Replicas | Prefill/Decode Count)",
            legend={"font": {"size": 14}},
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Right-align the legend caption
        caption_col1, caption_col2 = st.columns([3, 1])
        with caption_col2:
            st.caption("📜 **Tip**: Scroll within the legend box to see all runs")


def render_runtime_configs_section(filtered_df, use_expander=True):
    """Render runtime server configs section for LLM-D dashboard."""
    if use_expander:
        if "runtime_configs_expanded" not in st.session_state:
            st.session_state.runtime_configs_expanded = False
        ctx = st.expander(
            "⚙️ Runtime Server Configs Used",
            expanded=st.session_state.runtime_configs_expanded,
        )
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("⚙️ Runtime Server Configs Used")
        if "runtime_args" in filtered_df.columns:
            st.markdown(
                "**Runtime configurations for your current filter selections:**"
            )
            st.info(
                "📊 **Column Legend**: Shows the server runtime arguments used for each Model + Accelerator + Version + Pod Configuration combination that matches your current filters."
            )

            unique_configs = filtered_df.drop_duplicates(
                subset=[
                    "model",
                    "accelerator",
                    "version",
                    "replicas",
                    "prefill_pod_count",
                    "decode_pod_count",
                ]
            )

            if not unique_configs.empty:
                configs_df = unique_configs[
                    [
                        "model",
                        "accelerator",
                        "version",
                        "replicas",
                        "prefill_pod_count",
                        "decode_pod_count",
                        "runtime_args",
                    ]
                ].copy()

                configs_df.reset_index(drop=True, inplace=True)
                configs_df.insert(0, "Config #", range(1, len(configs_df) + 1))
                configs_df.rename(
                    columns={
                        "model": "Model",
                        "accelerator": "Accelerator",
                        "version": "Version",
                        "replicas": "Replicas",
                        "prefill_pod_count": "Prefill Pods",
                        "decode_pod_count": "Decode Pods",
                        "runtime_args": "Runtime Arguments",
                    },
                    inplace=True,
                )
                for _pc in ["Prefill Pods", "Decode Pods"]:
                    configs_df[_pc] = configs_df[_pc].apply(
                        lambda v: "N/A" if pd.isna(v) else str(int(v))
                    )

                # Calculate dynamic height
                row_height = 40
                header_height = 40
                padding = 20
                dynamic_height = min(
                    max(len(configs_df) * row_height + header_height + padding, 150),
                    600,
                )

                st.dataframe(
                    configs_df[
                        [
                            "Config #",
                            "Model",
                            "Accelerator",
                            "Version",
                            "Replicas",
                            "Prefill Pods",
                            "Decode Pods",
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
                            "Model", width=300, pinned=True
                        ),
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator", width=80, pinned=True
                        ),
                        "Version": st.column_config.TextColumn(
                            "Version", width=100, pinned=True
                        ),
                        "Replicas": st.column_config.NumberColumn("Replicas", width=80),
                        "Prefill Pods": st.column_config.TextColumn(
                            "Prefill Pods", width=100
                        ),
                        "Decode Pods": st.column_config.TextColumn(
                            "Decode Pods", width=100
                        ),
                        "Runtime Arguments": st.column_config.TextColumn(
                            "Runtime Args", width=1800
                        ),
                    },
                )

                options = [
                    (
                        i,
                        f"Config {r['Config #']} – {r['Model']} / {r['Accelerator']} / {r['Version']} / R{r['Replicas']}-P{r['Prefill Pods']}-D{r['Decode Pods']}",
                    )
                    for i, r in configs_df.iterrows()
                ]
                idx = st.selectbox(
                    "Show full runtime args for:",
                    options,
                    format_func=lambda x: x[1],
                    key="llmd_runtime_config_selector",
                )[0]

                args = configs_df.loc[idx, "Runtime Arguments"]
                st.code(args, language="bash")

            else:
                st.warning(
                    "No runtime configurations found for the current filter selections."
                )
        else:
            st.error(
                "Runtime arguments column not found in the data. Please ensure the CSV file contains a 'runtime_args' column."
            )


def render_filtered_data_section(filtered_df, use_expander=True):
    """Render filtered data table section for LLM-D dashboard."""
    if use_expander:
        ctx = st.expander("📄 Filtered Data from the above filters", expanded=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if not use_expander:
            st.subheader("📄 Filtered Data")
        st.info(
            "💡 **Tips**: Hover over column headers to see detailed descriptions of each field."
        )
        display_filtered_df = filtered_df.copy()
        for _pc in ["prefill_pod_count", "decode_pod_count"]:
            if _pc in display_filtered_df.columns:
                display_filtered_df[_pc] = display_filtered_df[_pc].apply(
                    lambda v: "N/A" if pd.isna(v) else str(int(v))
                )
        display_filtered_df.reset_index(drop=True, inplace=True)
        display_filtered_df.insert(0, "Row #", range(1, len(display_filtered_df) + 1))

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
                help="Inference server version (e.g., llm-d-0.3)",
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
            "DP": st.column_config.NumberColumn(
                "DP",
                help="Data Parallelism size - number of instances running in parallel",
            ),
            "EP": st.column_config.NumberColumn(
                "EP",
                help="Expert Parallelism size (for MoE models)",
            ),
            "replicas": st.column_config.NumberColumn(
                "replicas",
                help="Number of replica instances deployed",
            ),
            "prefill_pod_count": st.column_config.TextColumn(
                "prefill_pod_count",
                help="Number of pods dedicated to prefill (prompt processing). N/A when not applicable.",
            ),
            "decode_pod_count": st.column_config.TextColumn(
                "decode_pod_count",
                help="Number of pods dedicated to decode (token generation). N/A when not applicable.",
            ),
            "router_config": st.column_config.TextColumn(
                "router_config",
                help="Router configuration for disaggregated architecture",
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
            "ttft_p95": st.column_config.NumberColumn(
                "ttft_p95",
                help="Time to First Token 95th percentile - key latency SLO metric (ms, lower is better)",
                format="%.2f",
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
        }

        st.dataframe(
            display_filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )


def _decode_llmd_url_filters(df: pd.DataFrame) -> dict:
    """Decode LLM-D filter state from URL query parameters.

    Returns a dict with keys matching baseline session-state names.
    Values are validated against the actual data.
    """
    qp = st.query_params

    all_accelerators = sorted(df["accelerator"].unique().tolist())
    all_models = sorted(df["model"].unique().tolist())
    all_versions = sorted(df["version"].unique().tolist())
    all_profiles = sorted(df["profile"].unique().tolist())
    all_tp = sorted(df["TP"].dropna().unique().tolist())
    all_replicas = sorted(df["replicas"].dropna().unique().tolist())

    result: dict = {}

    if "accelerators" in qp:
        result["accelerators"] = [
            a.strip()
            for a in qp["accelerators"].split(",")
            if a.strip() in all_accelerators
        ]
    if "models" in qp:
        result["models"] = [
            m.strip() for m in qp["models"].split(",") if m.strip() in all_models
        ]
    if "versions" in qp:
        result["versions"] = [
            v.strip() for v in qp["versions"].split(",") if v.strip() in all_versions
        ]
    if "profile" in qp:
        p = qp["profile"].strip()
        if p in all_profiles:
            result["profile"] = p
    if "custom_isl_osl" in qp:
        result["custom_isl_osl"] = qp["custom_isl_osl"].strip()
    if "tp_sizes" in qp:
        with contextlib.suppress(Exception):
            result["tp"] = [
                int(t.strip())
                for t in qp["tp_sizes"].split(",")
                if t.strip().isdigit() and int(t.strip()) in all_tp
            ]
    if "replicas" in qp:
        with contextlib.suppress(Exception):
            result["replicas"] = [
                int(r.strip())
                for r in qp["replicas"].split(",")
                if r.strip().isdigit() and int(r.strip()) in all_replicas
            ]
    if "prefill_pods" in qp:
        result["prefill_pods"] = [
            v.strip() for v in qp["prefill_pods"].split(",") if v.strip()
        ]
    if "decode_pods" in qp:
        result["decode_pods"] = [
            v.strip() for v in qp["decode_pods"].split(",") if v.strip()
        ]
    if "section" in qp:
        result["section"] = qp["section"].strip()

    if "pp_x" in qp:
        result["pp_x"] = qp["pp_x"].strip()
    if "pp_y" in qp:
        result["pp_y"] = qp["pp_y"].strip()

    return result


def render_llmd_dashboard(llmd_csv_path: str):
    """Render the LLM-D benchmark dashboard.

    Args:
        llmd_csv_path: Path to the LLM-D CSV data file
    """
    # Load data
    df = load_llmd_data(llmd_csv_path)

    if df is None or df.empty:
        st.error("No data available. Please check the data file.")
        return

    # --- URL filter decoding (once per session) ---
    LLMD_SECTION_SLUG_MAP = {
        "📈 Performance Plots": "performance_plots",
        "⚖️ Compare Versions": "compare_versions",
        "🔄 Compare with RHAIIS": "rhaiis_comparison",
        "⚙️ Runtime Server Configs": "runtime_configs",
        "📄 Filtered Data": "filtered_data",
    }
    LLMD_SLUG_TO_SECTION = {v: k for k, v in LLMD_SECTION_SLUG_MAP.items()}

    if "llmd_url_filters_loaded" not in st.session_state:
        st.session_state.llmd_url_filters_loaded = True
        url_filters = _decode_llmd_url_filters(df)

        has_url_filters = bool(url_filters)

        if "section" in url_filters:
            slug = url_filters["section"]
            if slug in LLMD_SLUG_TO_SECTION:
                st.session_state.llmd_active_section = LLMD_SLUG_TO_SECTION[slug]

        if has_url_filters:
            if "accelerators" in url_filters:
                st.session_state.llmd_baseline_accelerators = url_filters[
                    "accelerators"
                ]
            if "models" in url_filters:
                st.session_state.llmd_baseline_models = url_filters["models"]
            if "versions" in url_filters:
                st.session_state.llmd_baseline_versions = url_filters["versions"]
            if "profile" in url_filters:
                st.session_state.llmd_baseline_profile = url_filters["profile"]
            if "custom_isl_osl" in url_filters:
                st.session_state.llmd_selected_custom_isl_osl = url_filters[
                    "custom_isl_osl"
                ]
            if "tp" in url_filters:
                st.session_state.llmd_baseline_tp = url_filters["tp"]
            if "replicas" in url_filters:
                st.session_state.llmd_baseline_replicas = url_filters["replicas"]
            if "prefill_pods" in url_filters:
                st.session_state.llmd_baseline_prefill = url_filters["prefill_pods"]
            if "decode_pods" in url_filters:
                st.session_state.llmd_baseline_decode = url_filters["decode_pods"]
            if "pp_x" in url_filters:
                st.session_state["llmd_perf_plots_x_axis"] = url_filters["pp_x"]
            if "pp_y" in url_filters:
                st.session_state["llmd_perf_plots_y_axis"] = url_filters["pp_y"]

        st.session_state.llmd_use_url_filters = has_url_filters

    # Section navigation via sidebar
    section_list = [
        "📈 Performance Plots",
        "⚖️ Compare Versions",
        "🔄 Compare with RHAIIS",
        "⚙️ Runtime Server Configs",
        "📄 Filtered Data",
    ]

    SECTION_GROUPS = [
        (
            "Performance Analysis",
            [
                "📈 Performance Plots",
                "⚖️ Compare Versions",
                "🔄 Compare with RHAIIS",
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

    current_section = st.session_state.get("llmd_active_section", section_list[0])
    if current_section not in section_list:
        current_section = section_list[0]
    st.session_state.llmd_active_section = current_section

    LLMD_SECTIONS_WITHOUT_GLOBAL_FILTERS = {
        "⚖️ Compare Versions",
        "🔄 Compare with RHAIIS",
    }
    _show_global_filters = current_section not in LLMD_SECTIONS_WITHOUT_GLOBAL_FILTERS

    if _show_global_filters:
        filtered_df, filter_selections = render_llmd_filters(df)
        st.markdown("---")
    else:
        filtered_df = df.copy()
        filter_selections = {}

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
                    key=f"llmd_nav_{section_name}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.llmd_active_section = section_name
                    st.rerun()

    if not filtered_df.empty:

        def _render_selected_section(sel):
            if sel == "📈 Performance Plots":
                render_performance_plots_section(filtered_df, use_expander=False)
            elif sel == "⚖️ Compare Versions":
                render_compare_versions_section(df, use_expander=False)
            elif sel == "🔄 Compare with RHAIIS":
                render_rhaiis_comparison_section(df, use_expander=False)
            elif sel == "⚙️ Runtime Server Configs":
                render_runtime_configs_section(filtered_df, use_expander=False)
            elif sel == "📄 Filtered Data":
                render_filtered_data_section(filtered_df, use_expander=False)

        _render_selected_section(current_section)

    else:
        selected_models = filter_selections.get("models", [])
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
                            combo_dict: dict[str, dict[str, dict[str, list]]] = {}
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

    # --- Sync filter state to URL query params ---
    with contextlib.suppress(Exception):
        desired_params: dict[str, str] = {}
        if "view" in st.query_params:
            desired_params["view"] = st.query_params["view"]

        if _show_global_filters:
            sel_acc = filter_selections.get("accelerators", [])
            sel_mod = filter_selections.get("models", [])
            sel_ver = filter_selections.get("versions", [])
            sel_prof = filter_selections.get("profiles", [])
            sel_tp = filter_selections.get("tp_sizes", [])
            sel_rep = filter_selections.get("replicas", [])
            sel_pf = filter_selections.get("prefill_pods", [])
            sel_dc = filter_selections.get("decode_pods", [])

            if sel_acc:
                desired_params["accelerators"] = ",".join(sel_acc)
            if sel_mod:
                desired_params["models"] = ",".join(sel_mod)
            if sel_ver:
                desired_params["versions"] = ",".join(sel_ver)
            if sel_prof:
                desired_params["profile"] = sel_prof[0]
            custom_val = st.session_state.get("llmd_selected_custom_isl_osl")
            if custom_val:
                desired_params["custom_isl_osl"] = custom_val
            if sel_tp:
                desired_params["tp_sizes"] = ",".join(map(str, sel_tp))
            if sel_rep:
                desired_params["replicas"] = ",".join(map(str, sel_rep))
            if sel_pf:
                desired_params["prefill_pods"] = ",".join(map(str, sel_pf))
            if sel_dc:
                desired_params["decode_pods"] = ",".join(map(str, sel_dc))

        if current_section and current_section in LLMD_SECTION_SLUG_MAP:
            slug = LLMD_SECTION_SLUG_MAP[current_section]
            desired_params["section"] = slug
            if slug == "performance_plots":
                for url_key, ss_key in (
                    ("pp_x", "llmd_perf_plots_x_axis"),
                    ("pp_y", "llmd_perf_plots_y_axis"),
                ):
                    val = st.session_state.get(ss_key)
                    if val is not None:
                        desired_params[url_key] = str(val)

        st.query_params.from_dict(desired_params)

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
            if (hdr) hdr.classList.add('hamburger-btn');
        }}
        var sidebarOpen = sb && sb.getAttribute('aria-expanded') === 'true';
        if (!sidebarOpen) {{
            var header = doc.querySelector('[data-testid="stHeader"]');
            if (header) {{
                var firstBtn = header.querySelector('button');
                if (firstBtn) firstBtn.classList.add('hamburger-btn');
            }}
        }}
    }}

    scan();
    doc._hamburgerInterval = setInterval(scan, 500);
}})();
</script>
""",
        height=0,
    )
