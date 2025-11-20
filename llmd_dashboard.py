"""LLM-D Dashboard Module.

This module provides functionality to load, process, and visualize
LLM-D benchmark results with disaggregated prefill/decode architecture.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


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
    elif prompt_toks == 8000 and output_toks == 1000:
        return "Profile E: Prefill Heavy (8k/1k)"
    elif prompt_toks == 1000 and output_toks == 100:
        return "Profile F: Prefill Heavy (1k/100)"
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


def load_llmd_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess LLM-D benchmark data from CSV file.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        DataFrame with loaded and processed data, or None if error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Convert numeric columns
        numeric_cols = [
            "TP",
            "DP",
            "EP",
            "replicas",
            "prefill_pod_count",
            "decode_pod_count",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Assign profile based on prompt/output tokens
        if "prompt toks" in df.columns and "output toks" in df.columns:
            df["profile"] = df.apply(assign_profile, axis=1)

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
        for col in ttft_cols:
            if col in df.columns:
                df[f"{col}_s"] = df[col] / 1000

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
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns(3)
    filter_col7, filter_col8, filter_col9 = st.columns(3)

    # FILTER 1: Accelerator
    with filter_col1:
        accelerators = sorted(df["accelerator"].unique().tolist())

        if st.session_state.get("llmd_clear_all_filters", False):
            acc_default = []
        else:
            baseline_accelerators = st.session_state.get(
                "llmd_baseline_accelerators", accelerators
            )
            # Ensure baseline values are in current available options
            acc_default = [a for a in baseline_accelerators if a in accelerators]

        selected_accelerators = st.multiselect(
            "1️⃣ Select Accelerator(s)",
            accelerators,
            default=acc_default,
            key=f"llmd_acc_filter_{st.session_state.llmd_filter_change_key}",
        )

    # FILTER 2: ISL/OSL Profile - filtered by accelerators
    with filter_col2:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]

        profiles = (
            sorted(temp_df["profile"].unique().tolist()) if not temp_df.empty else []
        )

        default_profile = "Profile A: Balanced (1k/1k)"
        if st.session_state.get("llmd_clear_all_filters", False):
            profile_default = None
        else:
            baseline_profile = st.session_state.get(
                "llmd_baseline_profile", default_profile
            )
            profile_default = (
                baseline_profile
                if baseline_profile in profiles
                else (
                    default_profile
                    if default_profile in profiles
                    else (profiles[0] if profiles else None)
                )
            )

        selected_profile = (
            st.selectbox(
                "2️⃣ Select Input/Output Sequence Length (ISL/OSL)",
                profiles,
                index=(
                    profiles.index(profile_default)
                    if profile_default in profiles
                    else 0
                ),
                format_func=clean_profile_name,
                key=f"llmd_profile_filter_{st.session_state.llmd_filter_change_key}",
            )
            if profiles
            else None
        )

        selected_profiles = [selected_profile] if selected_profile is not None else []

        # Update baseline_profile to remember user's current selection
        if selected_profile is not None:
            st.session_state.llmd_baseline_profile = selected_profile

    # FILTER 3: Version - filtered by accelerators and profile
    with filter_col3:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]

        versions = (
            sorted(temp_df["version"].unique().tolist()) if not temp_df.empty else []
        )

        if st.session_state.get("llmd_clear_all_filters", False):
            version_default = []
        else:
            baseline_versions = st.session_state.get(
                "llmd_baseline_versions", versions[:1] if versions else []
            )
            # Ensure baseline values are in current available options
            version_default = [v for v in baseline_versions if v in versions]

        selected_versions = st.multiselect(
            "3️⃣ Select Version(s)",
            versions,
            default=version_default,
            key=f"llmd_version_filter_{st.session_state.llmd_filter_change_key}",
        )

    # FILTER 4: Model - filtered by accelerators, profile, and version
    with filter_col4:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]

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
            preserved_selections = models_to_select

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

    # FILTER 5: TP size
    with filter_col5:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]

        tp_sizes = (
            sorted(temp_df["TP"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        # Check if "Select All Models" is checked or if models are selected
        select_all_models_key = (
            f"select_all_models_{st.session_state.llmd_filter_change_key}"
        )
        select_all_models_checked = st.session_state.get(select_all_models_key, False)

        if st.session_state.get("llmd_clear_all_filters", False):
            tp_default = []
        elif select_all_models_checked or selected_models:
            # If "Select All Models" is checked OR models are selected, auto-select all TP sizes
            tp_default = tp_sizes
        else:
            baseline_tp = st.session_state.get("llmd_baseline_tp", tp_sizes)
            # Ensure baseline values are in current available options
            tp_default = [tp for tp in baseline_tp if tp in tp_sizes]

        selected_tp = st.multiselect(
            "5️⃣ Select TP Size(s)",
            tp_sizes,
            default=tp_default,
            key=f"llmd_tp_filter_{st.session_state.llmd_filter_change_key}",
        )

    # FILTER 6: Replicas
    with filter_col6:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]

        replicas = (
            sorted(temp_df["replicas"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        # Check if "Select All Models" is checked or if models are selected
        select_all_models_key = (
            f"select_all_models_{st.session_state.llmd_filter_change_key}"
        )
        select_all_models_checked = st.session_state.get(select_all_models_key, False)

        if st.session_state.get("llmd_clear_all_filters", False):
            replicas_default = []
        elif select_all_models_checked or selected_models:
            # If "Select All Models" is checked OR models are selected, auto-select all replicas
            replicas_default = replicas
        else:
            baseline_replicas = st.session_state.get("llmd_baseline_replicas", replicas)
            # Ensure baseline_replicas is always a list
            if not isinstance(baseline_replicas, list):
                baseline_replicas = [baseline_replicas] if baseline_replicas else []
            # Ensure baseline values are in current available options
            replicas_default = [r for r in baseline_replicas if r in replicas]

        selected_replicas = st.multiselect(
            "6️⃣ Select # of Replicas",
            replicas,
            default=replicas_default,
            key=f"llmd_replicas_filter_{st.session_state.llmd_filter_change_key}",
        )

    # FILTER 7: Prefill Pod Count
    with filter_col7:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]
        if selected_replicas:
            temp_df = temp_df[temp_df["replicas"].isin(selected_replicas)]

        prefill_pods = (
            sorted(temp_df["prefill_pod_count"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        # Check if "Select All Models" is checked or if models are selected
        select_all_models_key = (
            f"select_all_models_{st.session_state.llmd_filter_change_key}"
        )
        select_all_models_checked = st.session_state.get(select_all_models_key, False)

        if st.session_state.get("llmd_clear_all_filters", False):
            prefill_default = []
        elif select_all_models_checked or selected_models:
            # If "Select All Models" is checked OR models are selected, auto-select all prefill pod counts
            prefill_default = prefill_pods
        else:
            baseline_prefill = st.session_state.get(
                "llmd_baseline_prefill", prefill_pods
            )
            # Ensure baseline values are in current available options
            prefill_default = [p for p in baseline_prefill if p in prefill_pods]

        selected_prefill_pods = st.multiselect(
            "7️⃣ Select Prefill Pod Count",
            prefill_pods,
            default=prefill_default,
            key=f"llmd_prefill_filter_{st.session_state.llmd_filter_change_key}",
        )

    # FILTER 8: Decode Pod Count
    with filter_col8:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]
        if selected_tp:
            temp_df = temp_df[temp_df["TP"].isin(selected_tp)]
        if selected_replicas:
            temp_df = temp_df[temp_df["replicas"].isin(selected_replicas)]
        if selected_prefill_pods:
            temp_df = temp_df[temp_df["prefill_pod_count"].isin(selected_prefill_pods)]

        decode_pods = (
            sorted(temp_df["decode_pod_count"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        # Check if "Select All Models" is checked or if models are selected
        select_all_models_key = (
            f"select_all_models_{st.session_state.llmd_filter_change_key}"
        )
        select_all_models_checked = st.session_state.get(select_all_models_key, False)

        if st.session_state.get("llmd_clear_all_filters", False):
            decode_default = []
        elif select_all_models_checked or selected_models:
            # If "Select All Models" is checked OR models are selected, auto-select all decode pod counts
            decode_default = decode_pods
        else:
            baseline_decode = st.session_state.get("llmd_baseline_decode", decode_pods)
            # Ensure baseline values are in current available options
            decode_default = [d for d in baseline_decode if d in decode_pods]

        selected_decode_pods = st.multiselect(
            "8️⃣ Select Decode Pod Count",
            decode_pods,
            default=decode_default,
            key=f"llmd_decode_filter_{st.session_state.llmd_filter_change_key}",
        )

    # Filter buttons next to decode pod count
    with filter_col9:
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button(
                "🔄 Reset to Defaults",
                help="Reset filters to default values",
                key="llmd_reset_btn",
            ):
                st.session_state.llmd_clear_all_filters = False
                st.session_state.llmd_filters_were_cleared = False
                st.session_state.llmd_reset_to_defaults = True
                st.session_state.llmd_filter_change_key += 1
                st.rerun()

        with btn_col2:
            if st.button(
                "🧹 Clear Filters",
                help="Clear all filter selections",
                key="llmd_clear_btn",
            ):
                st.session_state.llmd_clear_all_filters = True
                st.session_state.llmd_filters_were_cleared = True
                st.session_state.llmd_filter_change_key += 1
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
                                    dict[
                                        str, dict[int, dict[int, set[tuple[int, int]]]]
                                    ],
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
                                if (
                                    replicas
                                    not in combo_dict[acc][version][profile][tp]
                                ):
                                    combo_dict[acc][version][profile][tp][replicas] = (
                                        set()
                                    )

                                combo_dict[acc][version][profile][tp][replicas].add(
                                    (int(prefill), int(decode))
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
                                                combo_dict[acc][version][profile][
                                                    tp
                                                ].keys()
                                            )
                                            for replica in replicas_list:
                                                pods = sorted(
                                                    combo_dict[acc][version][profile][
                                                        tp
                                                    ][replica]
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
                                    combo_dict[acc][model_short][profile][tp][
                                        replicas
                                    ] = set()

                                combo_dict[acc][model_short][profile][tp][replicas].add(
                                    (int(prefill), int(decode))
                                )

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
                                                    combo_dict[acc][model_short][
                                                        profile
                                                    ][tp][replica]
                                                )
                                                pods_str = ", ".join(
                                                    [f"({p}/{d})" for p, d in pods]
                                                )
                                                tree_text += f"                👥 Replicas: {replica} → Pods(P/D): {pods_str}\n"
                                tree_text += "\n"

                            st.code(tree_text, language=None)

    # Apply all filters
    filtered_df = df.copy()

    if selected_accelerators:
        filtered_df = filtered_df[
            filtered_df["accelerator"].isin(selected_accelerators)
        ]
    if selected_profiles:
        filtered_df = filtered_df[filtered_df["profile"].isin(selected_profiles)]
    if selected_versions:
        filtered_df = filtered_df[filtered_df["version"].isin(selected_versions)]
    if selected_models:
        filtered_df = filtered_df[filtered_df["model"].isin(selected_models)]
    if selected_tp:
        filtered_df = filtered_df[filtered_df["TP"].isin(selected_tp)]
    if selected_replicas:
        filtered_df = filtered_df[filtered_df["replicas"].isin(selected_replicas)]
    if selected_prefill_pods:
        filtered_df = filtered_df[
            filtered_df["prefill_pod_count"].isin(selected_prefill_pods)
        ]
    if selected_decode_pods:
        filtered_df = filtered_df[
            filtered_df["decode_pod_count"].isin(selected_decode_pods)
        ]

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

    # Reset state flags after processing
    if st.session_state.get("llmd_clear_all_filters", False):
        st.session_state.llmd_clear_all_filters = False
    if st.session_state.get("llmd_reset_to_defaults", False):
        st.session_state.llmd_reset_to_defaults = False

    # Store baseline values
    if not st.session_state.get("llmd_baseline_accelerators"):
        st.session_state.llmd_baseline_accelerators = accelerators
        st.session_state.llmd_baseline_profile = (
            default_profile
            if default_profile in profiles
            else (profiles[0] if profiles else None)
        )
        st.session_state.llmd_baseline_versions = versions[:1] if versions else []
        st.session_state.llmd_baseline_models = models[:1] if models else []
        st.session_state.llmd_baseline_tp = tp_sizes
        st.session_state.llmd_baseline_replicas = replicas
        st.session_state.llmd_baseline_prefill = prefill_pods
        st.session_state.llmd_baseline_decode = decode_pods

    return filtered_df, filter_selections


def load_rhaiis_data(file_path: str = "consolidated_dashboard.csv") -> pd.DataFrame:
    """Load RHAIIS data for comparison.

    Args:
        file_path: Path to the RHAIIS CSV file

    Returns:
        DataFrame with RHAIIS data
    """
    try:
        df = pd.read_csv(file_path)
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Assign profile based on prompt/output tokens (same as llmd_data)
        if "prompt toks" in df.columns and "output toks" in df.columns:
            df["profile"] = df.apply(assign_profile, axis=1)

        return df
    except Exception as e:
        st.error(f"Error loading RHAIIS data: {str(e)}")
        return pd.DataFrame()


def render_rhaiis_comparison_section(llmd_filtered_df: pd.DataFrame):
    """Render comparison section between LLM-D and RHAIIS.

    Args:
        llmd_filtered_df: Filtered LLM-D DataFrame
    """
    if "rhaiis_comparison_expanded" not in st.session_state:
        st.session_state.rhaiis_comparison_expanded = False

    with st.expander(
        "🔄 Compare with RHAIIS", expanded=st.session_state.rhaiis_comparison_expanded
    ):
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
            rhaiis_df = rhaiis_df[
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
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            selected_comparison_model = st.selectbox(
                "Select Model to Compare",
                options=common_models,
                format_func=lambda x: x.split("/")[-1] if "/" in x else x,
                key="rhaiis_comparison_model",
            )

        with filter_col2:
            # Get RHAIIS versions for the selected model (only versions containing "RHAIIS")
            rhaiis_model_data = rhaiis_df[
                rhaiis_df["model"] == selected_comparison_model
            ]
            all_versions = rhaiis_model_data["version"].unique()
            rhaiis_versions = sorted([v for v in all_versions if "RHAIIS" in str(v)])

            if not rhaiis_versions:
                st.warning("⚠️ No RHAIIS versions found for this model.")
                return

            # Preserve version selection in session state
            if "rhaiis_comparison_version_selected" not in st.session_state:
                # Default to RHAIIS-3.2.3 if available, otherwise use first available
                if "RHAIIS-3.2.3" in rhaiis_versions:
                    st.session_state.rhaiis_comparison_version_selected = "RHAIIS-3.2.3"
                else:
                    st.session_state.rhaiis_comparison_version_selected = (
                        rhaiis_versions[0]
                    )

            # Use preserved version if it exists in current options, otherwise default to RHAIIS-3.2.3 or first available
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

            # Update session state with current selection
            st.session_state.rhaiis_comparison_version_selected = (
                selected_rhaiis_version
            )

        with filter_col3:
            # Get common profiles
            llmd_model_data = llmd_single_replica[
                llmd_single_replica["model"] == selected_comparison_model
            ]
            rhaiis_version_data = rhaiis_model_data[
                rhaiis_model_data["version"] == selected_rhaiis_version
            ]

            # Check if 'profile' column exists, handle missing column gracefully
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

            # Preserve profile selection in session state
            if "rhaiis_comparison_profile_selected" not in st.session_state:
                st.session_state.rhaiis_comparison_profile_selected = common_profiles[0]

            # Use preserved profile if it exists in current options, otherwise use first available
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

            # Update session state with current selection
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
            + llmd_comparison["prefill_pod_count"].astype(str)
            + "/D="
            + llmd_comparison["decode_pod_count"].astype(str)
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
                template="plotly_white",
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
            st.plotly_chart(fig, use_container_width=True)

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
                    template="plotly_white",
                )
                fig_ttft.update_layout(
                    legend={"font": {"size": 10}},
                    height=400,
                )
                st.plotly_chart(fig_ttft, use_container_width=True)

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
                    template="plotly_white",
                )
                fig_tpot.update_layout(
                    legend={"font": {"size": 10}},
                    height=400,
                )
                st.plotly_chart(fig_tpot, use_container_width=True)

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


def render_performance_plots_section(filtered_df):
    """Render performance plots section for LLM-D dashboard."""
    if "performance_plots_expanded" not in st.session_state:
        st.session_state.performance_plots_expanded = False

    with st.expander(
        "📈 Performance Plots", expanded=st.session_state.performance_plots_expanded
    ):
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
            + filtered_df["prefill_pod_count"].astype(str)
            + "/D="
            + filtered_df["decode_pod_count"].astype(str)
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
            template="plotly_white",
            category_orders={
                "run_identifier": filtered_df_sorted["run_identifier"].unique().tolist()
            },
        )
        fig.update_layout(
            legend_title_text="Run Details (Accelerator | Model | Version | TP | Replicas | Prefill/Decode Count)",
            legend={"font": {"size": 14}},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Right-align the legend caption
        caption_col1, caption_col2 = st.columns([3, 1])
        with caption_col2:
            st.caption("📜 **Tip**: Scroll within the legend box to see all runs")


def render_runtime_configs_section(filtered_df):
    """Render runtime server configs section for LLM-D dashboard."""
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
                        "Prefill Pods": st.column_config.NumberColumn(
                            "Prefill Pods", width=100
                        ),
                        "Decode Pods": st.column_config.NumberColumn(
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


def render_filtered_data_section(filtered_df):
    """Render filtered data table section for LLM-D dashboard."""
    with st.expander("📄 Filtered Data from the above filters", expanded=False):
        st.info(
            "💡 **Tips**: Hover over column headers to see detailed descriptions of each field."
        )
        display_filtered_df = filtered_df.copy()
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
            "prefill_pod_count": st.column_config.NumberColumn(
                "prefill_pod_count",
                help="Number of pods dedicated to prefill (prompt processing)",
            ),
            "decode_pod_count": st.column_config.NumberColumn(
                "decode_pod_count",
                help="Number of pods dedicated to decode (token generation)",
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

    # Render filters
    filtered_df, filter_selections = render_llmd_filters(df)

    if filtered_df.empty:
        st.warning("⚠️ No runs match the selected filters.")
        return

    st.markdown("---")

    # Render sections
    render_performance_plots_section(filtered_df)
    render_rhaiis_comparison_section(filtered_df)
    render_runtime_configs_section(filtered_df)
    render_filtered_data_section(filtered_df)
