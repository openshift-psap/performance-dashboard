"""LLM Inference Performance Dashboard.

A comprehensive dashboard for analyzing and comparing LLM inference performance
across different models, versions, and hardware configurations.
"""

import contextlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Import styling functions
from dashboard_styles import (
    apply_theme_css,
    get_app_css,
    initialize_session_state,
    initialize_streamlit_config,
)


@st.cache_data(ttl=300)  # Cache for 5 minutes max
def load_data(file_path, cache_key=None):
    """Load and preprocess performance data from CSV file.

    Args:
        file_path: Path to the CSV file to load.
        cache_key: Optional cache key for cache invalidation.

    Returns:
        DataFrame with loaded and processed data, or None if error occurs.
    """
    try:
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


def render_performance_plots_section(filtered_df):
    """📈 Performance Plots Section - Complete functionality from original."""
    if "performance_plots_expanded" not in st.session_state:
        st.session_state.performance_plots_expanded = False

    with st.expander(
        "📈 Performance Plots", expanded=st.session_state.performance_plots_expanded
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

        col1, col2 = st.columns(2)
        with col1:
            x_axis_options = {
                "Concurrency": "intended concurrency",
                "Throughput (Output Tok/s)": "output_tok/sec",
            }
            x_axis_label = st.selectbox(
                "Select X-Axis",
                options=list(x_axis_options.keys()),
                key="perf_plots_x_axis",
            )
            x_axis = x_axis_options[x_axis_label]

        with col2:
            y_axis_options = {
                "Throughput (Output tokens/second generated)": "output_tok/sec",
                "Efficiency (Output tokens/sec per TP unit)": "efficiency_ratio",
                "Inter-Token Latency P95 (Time between tokens)": "itl_p95",
                "Time to First Token P95 (Response start delay)": "ttft_p95",
                "Request Latency Median (Total request processing time)": "request_latency_median",
                "Request Latency Max (Maximum request processing time)": "request_latency_max",
                "Total Throughput (Total tokens/second processed)": "total_tok/sec",
                "Request Count (Successful completions)": "successful_requests",
                "Error Rate (% Failed requests)": "error_rate",
            }
            y_axis_label = st.selectbox(
                "Select Y-Axis",
                options=list(y_axis_options.keys()),
                key="perf_plots_y_axis",
            )
            y_axis = y_axis_options[y_axis_label]

        fig = px.line(
            filtered_df_sorted.sort_values(by=x_axis),
            x=x_axis,
            y=y_axis,
            color="run_identifier",
            markers=True,
            title=f"{x_axis_label} vs. {y_axis_label}",
            labels={
                x_axis: x_axis_label,
                y_axis: y_axis_label,
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


def render_regression_analysis_section(filtered_df, analyze_performance_changes):
    """🔍 Performance Regression Analysis Section - Complete functionality from original."""
    with st.expander("🔍 Performance Regression Analysis", expanded=False):
        st.markdown(
            "Compare performance changes between versions for the same model, accelerator, and TP configuration."
        )
        st.info(
            "ℹ️ **Note**: This analysis compares versions within the same inference server (RHAIIS to RHAIIS, vLLM to vLLM, sglang to sglang) across **all common concurrency levels**. The **Median Change** columns show median percentage change across all concurrency levels. The **(Old → New)** columns show **peak performance values from the same concurrency level** (e.g., best throughput or lowest latency). Eg: **(C=50)** means the comparison is at **concurrency level 50**. This ensures we're comparing apples-to-apples performance at the exact same concurrency level."
        )

        regression_df = analyze_performance_changes(filtered_df)

        if not regression_df.empty:
            reg_col1, reg_col2 = st.columns(2)

            with reg_col1:
                # Threshold for significant changes
                significance_threshold = st.slider(
                    "Significance Threshold (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Changes below this percentage are considered insignificant",
                )

            with reg_col2:
                st.info("📊 **Analysis Metric**: Throughput")

            # Function to categorize changes
            def categorize_change(
                value, is_higher_better=True, threshold=significance_threshold
            ):
                if pd.isna(value):
                    return "No Data", "⚪"

                abs_value = abs(value)
                if abs_value < threshold:
                    return "No Change", "🟡"

                if is_higher_better:
                    if value > 0:
                        return "Improvement", "🟢"
                    else:
                        return "Regression", "🔴"
                else:
                    if value < 0:
                        return "Improvement", "🟢"
                    else:
                        return "Regression", "🔴"

            if "throughput_change" in regression_df.columns:
                regression_df[["throughput_status", "throughput_icon"]] = regression_df[
                    "throughput_change"
                ].apply(
                    lambda x: pd.Series(
                        categorize_change(x, True, significance_threshold)
                    )
                )

            if "ttft_change" in regression_df.columns:
                regression_df[["ttft_status", "ttft_icon"]] = regression_df[
                    "ttft_change"
                ].apply(
                    lambda x: pd.Series(
                        categorize_change(x, False, significance_threshold)
                    )
                )

            if "itl_change" in regression_df.columns:
                regression_df[["itl_status", "itl_icon"]] = regression_df[
                    "itl_change"
                ].apply(
                    lambda x: pd.Series(
                        categorize_change(x, False, significance_threshold)
                    )
                )

            if "efficiency_change" in regression_df.columns:
                regression_df[["efficiency_status", "efficiency_icon"]] = regression_df[
                    "efficiency_change"
                ].apply(
                    lambda x: pd.Series(
                        categorize_change(x, True, significance_threshold)
                    )
                )

            st.subheader("📊 Change Summary")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                total_comparisons = len(regression_df)
                st.metric("Total Comparisons", total_comparisons)

            with summary_col2:
                if "throughput_status" in regression_df.columns:
                    improvements = len(
                        regression_df[
                            regression_df["throughput_status"] == "Improvement"
                        ]
                    )
                    st.metric(
                        "Throughput Improvements",
                        improvements,
                        delta=f"{improvements}/{total_comparisons}",
                    )

            with summary_col3:
                if "throughput_status" in regression_df.columns:
                    regressions = len(
                        regression_df[
                            regression_df["throughput_status"] == "Regression"
                        ]
                    )
                    st.metric(
                        "Throughput Regressions",
                        regressions,
                        delta=f"-{regressions}/{total_comparisons}",
                    )

            with summary_col4:
                if "throughput_status" in regression_df.columns:
                    no_change = len(
                        regression_df[regression_df["throughput_status"] == "No Change"]
                    )
                    st.metric("No Significant Change", no_change)

            st.subheader("🔍 Detailed Version Comparisons")

            show_all = st.checkbox(
                "Show all changes", value=False, help="Include insignificant changes"
            )

            if not show_all:
                significant_df = regression_df.copy()
                if "throughput_status" in significant_df.columns:
                    significant_df = significant_df[
                        significant_df["throughput_status"] != "No Change"
                    ]
                regression_display_df = significant_df
            else:
                regression_display_df = regression_df

            if not regression_display_df.empty:
                display_columns = [
                    "model",
                    "accelerator",
                    "tp",
                    "profile",
                    "version_type",
                    "older_version",
                    "newer_version",
                    "common_concurrencies",
                ]

                if "throughput_change" in regression_display_df.columns:
                    regression_display_df["Median Throughput Change"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['throughput_icon']} {row['throughput_change']:.1f}%"
                                if pd.notna(row["throughput_change"])
                                else "No Data"
                            ),
                            axis=1,
                        )
                    )
                    regression_display_df["Throughput (Old → New)"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['throughput_peak_older']:.1f} → {row['throughput_peak_newer']:.1f} (C={int(row['throughput_peak_concurrency'])})"
                                if pd.notna(row.get("throughput_peak_older"))
                                and pd.notna(row.get("throughput_peak_newer"))
                                else "No Peak Match"
                            ),
                            axis=1,
                        )
                    )
                    display_columns.extend(
                        ["Median Throughput Change", "Throughput (Old → New)"]
                    )

                if "ttft_change" in regression_display_df.columns:
                    regression_display_df["Median TTFT Change"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['ttft_icon']} {row['ttft_change']:.1f}%"
                                if pd.notna(row["ttft_change"])
                                else "No Data"
                            ),
                            axis=1,
                        )
                    )
                    regression_display_df["TTFT (Old → New)"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['ttft_peak_older']:.1f} → {row['ttft_peak_newer']:.1f} ms (C={int(row['ttft_peak_concurrency'])})"
                                if pd.notna(row.get("ttft_peak_older"))
                                and pd.notna(row.get("ttft_peak_newer"))
                                else "No Peak Match"
                            ),
                            axis=1,
                        )
                    )
                    display_columns.extend(["Median TTFT Change", "TTFT (Old → New)"])

                if "itl_change" in regression_display_df.columns:
                    regression_display_df["Median ITL Change"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['itl_icon']} {row['itl_change']:.1f}%"
                                if pd.notna(row["itl_change"])
                                else "No Data"
                            ),
                            axis=1,
                        )
                    )
                    regression_display_df["ITL (Old → New)"] = (
                        regression_display_df.apply(
                            lambda row: (
                                f"{row['itl_peak_older']:.1f} → {row['itl_peak_newer']:.1f} ms (C={int(row['itl_peak_concurrency'])})"
                                if pd.notna(row.get("itl_peak_older"))
                                and pd.notna(row.get("itl_peak_newer"))
                                else "No Peak Match"
                            ),
                            axis=1,
                        )
                    )
                    display_columns.extend(["Median ITL Change", "ITL (Old → New)"])

                # Format concurrency levels for display
                regression_display_df["Concurrency Levels"] = regression_display_df[
                    "common_concurrencies"
                ].apply(
                    lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
                )
                display_columns = [
                    col if col != "common_concurrencies" else "Concurrency Levels"
                    for col in display_columns
                ]

                # Rename columns for display
                display_df = regression_display_df[display_columns].copy()
                display_df = display_df.rename(
                    columns={
                        "model": "Model",
                        "accelerator": "Accelerator",
                        "tp": "TP",
                        "profile": "Profile",
                        "version_type": "Inference Server",
                        "older_version": "From Version",
                        "newer_version": "To Version",
                    }
                )

                # Sort by most significant changes
                if "throughput_change" in regression_display_df.columns:
                    display_df = display_df.reindex(
                        regression_display_df["throughput_change"]
                        .abs()
                        .sort_values(ascending=False)
                        .index
                    )

                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No significant performance changes detected with current threshold."
                )

            # Visualization of changes (Collapsible)
            with st.expander("📈 Performance Change Visualization", expanded=False):
                if (
                    not regression_display_df.empty
                    and "throughput_change" in regression_display_df.columns
                ):
                    # Create a scatter plot of performance changes
                    regression_display_df["comparison"] = (
                        regression_display_df["model"]
                        + " | "
                        + regression_display_df["accelerator"]
                        + " | TP"
                        + regression_display_df["tp"].astype(str)
                        + " | "
                        + regression_display_df["version_type"]
                        + ": "
                        + regression_display_df["older_version"]
                        + "→"
                        + regression_display_df["newer_version"]
                    )

                    fig_regression = px.scatter(
                        regression_display_df,
                        x="throughput_change",
                        y="comparison",
                        color="throughput_status",
                        size=regression_display_df["throughput_change"].abs(),
                        hover_data=[
                            "model",
                            "accelerator",
                            "older_version",
                            "newer_version",
                        ],
                        title="Median Throughput Performance Changes by Configuration",
                        labels={
                            "throughput_change": "Median Throughput Change (%)",
                            "comparison": "Configuration",
                        },
                        color_discrete_map={
                            "Improvement": "green",
                            "Regression": "red",
                            "No Change": "gray",
                        },
                        template="plotly_white",
                    )

                    fig_regression.add_vline(
                        x=0, line_dash="dash", line_color="black", opacity=0.5
                    )
                    fig_regression.add_vline(
                        x=significance_threshold,
                        line_dash="dot",
                        line_color="orange",
                        opacity=0.7,
                    )
                    fig_regression.add_vline(
                        x=-significance_threshold,
                        line_dash="dot",
                        line_color="orange",
                        opacity=0.7,
                    )

                    fig_regression.update_layout(
                        height=max(400, len(regression_display_df) * 30)
                    )
                    st.plotly_chart(fig_regression, use_container_width=True)
                else:
                    st.info("No performance change data available for visualization.")

        else:
            st.error(
                "⚠️ **No version comparisons available!** Need at least 2 versions of the same inference server type (e.g., RHAIIS-3.1 and RHAIIS-3.2) for the same model, accelerator, and TP combination with common concurrency levels."
            )


def render_model_performance_comparison_section(filtered_df, accelerator_color_map):
    """📊 Model Performance Comparison Section - Complete functionality with SLO analysis from original."""
    with st.expander("📊 Model Performance Comparison", expanded=False):
        st.markdown(
            "💡 **Tip:** Click on the full screen view (⛶) of any graph to get a detailed view."
        )
        st.markdown("")

        def get_performance_at_fixed_concurrency(group, target_concurrency=100):
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
            .apply(get_performance_at_fixed_concurrency)
            .reset_index()
        )

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
                "ttft_p95",
                "itl_p95",
                "efficiency_ratio",
            ]
            existing_cols = [
                col for col in required_cols if col in model_comparison.columns
            ]
            if existing_cols:
                model_comparison = model_comparison.dropna(subset=existing_cols)

        if not model_comparison.empty:
            st.info(
                "📊 **Fair Comparison**: All metrics shown are at **Concurrency Level 100** for apples-to-apples comparison across models and accelerators."
            )
        else:
            st.warning(
                "⚠️ No data available at concurrency level 100 for the selected filters."
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
                title="Peak Throughput by Model & Accelerator (at Concurrency 100)<br><sub>Higher is Better ↑</sub>",
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
                model_comparison.sort_values("ttft_p95", ascending=False),
                x="ttft_p95",
                y="model_accelerator",
                color="accelerator",
                color_discrete_map=accelerator_color_map,
                orientation="h",
                title="Best TTFT P95 Latency by Model & Accelerator (at Concurrency 100)<br><sub>Lower is Better ↓</sub>",
                labels={
                    "ttft_p95": "Best TTFT P95 (ms)",
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
                title="Peak Efficiency Ratio by Model & Accelerator (at Concurrency 100)<br><sub>Higher is Better ↑</sub>",
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
                title="Best Inter-Token Latency P95 by Model & Accelerator (at Concurrency 100)<br><sub>Lower is Better ↓</sub>",
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
        st.markdown("")

        with st.expander(
            "💰 Cloud Instance Pricing (as of September 11th, 2025)", expanded=False
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

        # FIRST PASS: Identify which model/accelerator combinations have SLO-compliant configurations
        slo_analysis_data = []
        model_accelerator_groups = filtered_df.groupby(["model", "accelerator"])

        for (model, accelerator), group in model_accelerator_groups:
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

            slo_analysis_data.append(result_dict)

        slo_analysis = pd.DataFrame(slo_analysis_data)

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
                    cost_col1, cost_col2 = st.columns(2)

                    with cost_col1:
                        # Time to Million Tokens chart
                        fig_time = px.bar(
                            cost_df.sort_values("ttmt_minutes", ascending=True),
                            x="ttmt_minutes",
                            y="model",
                            color="accelerator",
                            color_discrete_map=accelerator_color_map,
                            orientation="h",
                            title="Time to Million Tokens (minutes) - Lower is Better",
                            labels={
                                "ttmt_minutes": "Time to Million Tokens (minutes)",
                                "model": "Model",
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
                            y="model",
                            color="accelerator",
                            color_discrete_map=accelerator_color_map,
                            orientation="h",
                            title="Cost per Million Tokens (USD) - Lower is Better",
                            labels={
                                "cpmt_total": "Cost per Million Tokens (USD)",
                                "model": "Model",
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
                        st.caption(
                            "📊 Multiple accelerator types used for results, see 'Formulas' for calculation details. Click legend items to show/hide accelerator types."
                        )

                    # Cost efficiency ranking table
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

                    st.dataframe(
                        cost_display_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
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


def render_performance_rankings_section(filtered_df):
    """🏆 Performance Rankings Section - Complete functionality from original."""
    with st.expander("🏆 Performance Rankings", expanded=False):
        st.info(
            """
        📊 **How Rankings Are Calculated**:

        **🚀 Top 10 by Throughput**: Ranked by **highest Output Tokens/Second** - shows configurations that generate the most tokens per second, regardless of latency or concurrency level.

        **⚡ Top 10 by Low Latency**: Ranked by **lowest TTFT P95** (Time to First Token, 95th percentile) - shows configurations with the fastest response start times, prioritizing user experience.

        - Rankings include **all concurrency levels** and **TP configurations** from your current filters
        - Each row shows the specific configuration (Model, Version, TP, Concurrency) that achieved that performance
        """
        )

        ranking_col1, ranking_col2 = st.columns(2)

        with ranking_col1:
            st.subheader("Top 10 by Throughput")
            top_throughput = filtered_df.nlargest(10, "output_tok/sec")[
                [
                    "accelerator",
                    "model",
                    "version",
                    "TP",
                    "intended concurrency",
                    "output_tok/sec",
                    "ttft_p95",
                ]
            ].copy()
            top_throughput["model"] = top_throughput["model"].apply(
                lambda x: x.split("/")[-1]
            )
            top_throughput = top_throughput.rename(
                columns={"intended concurrency": "Concurrency"}
            )
            top_throughput = top_throughput.round(2)
            top_throughput.reset_index(drop=True, inplace=True)
            top_throughput.insert(0, "Rank", range(1, len(top_throughput) + 1))
            st.dataframe(top_throughput, use_container_width=True, hide_index=True)

        with ranking_col2:
            st.subheader("Top 10 by Low Latency")
            top_latency = filtered_df.nsmallest(10, "ttft_p95")[
                [
                    "accelerator",
                    "model",
                    "version",
                    "TP",
                    "intended concurrency",
                    "ttft_p95",
                    "output_tok/sec",
                ]
            ].copy()
            top_latency["model"] = top_latency["model"].apply(
                lambda x: x.split("/")[-1]
            )
            top_latency = top_latency.rename(
                columns={"intended concurrency": "Concurrency"}
            )
            top_latency = top_latency.round(2)
            top_latency.reset_index(drop=True, inplace=True)
            top_latency.insert(0, "Rank", range(1, len(top_latency) + 1))
            st.dataframe(top_latency, use_container_width=True, hide_index=True)


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
                            "Config No", width=80
                        ),
                        "Model": st.column_config.TextColumn("Model", width=380),
                        "Accelerator": st.column_config.TextColumn(
                            "Accelerator", width=80
                        ),
                        "Version": st.column_config.TextColumn("Version", width=120),
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
    """📊 Filtered Data Display Section - View only, no download functionality."""
    with st.expander("📊 Filtered Data from the above filters", expanded=False):
        display_filtered_df = filtered_df.copy()
        display_filtered_df.reset_index(drop=True, inplace=True)
        display_filtered_df.insert(0, "Row #", range(1, len(display_filtered_df) + 1))

        st.dataframe(display_filtered_df, use_container_width=True, hide_index=True)


def render_kpi_section(filtered_df):
    """📊 Key Performance Indicators Section - Complete functionality from original."""
    with st.expander("📊 Key Performance Indicators", expanded=False):
        st.info(
            "🎯 **KPI Details**: Shows best performance across all configurations. Format: Accelerator | Model | TP=Tensor Parallelism | Version | C=Concurrency Level"
        )

        best_throughput_idx = filtered_df["output_tok/sec"].idxmax()
        best_throughput_config = filtered_df.loc[best_throughput_idx]
        best_throughput = best_throughput_config["output_tok/sec"]

        best_latency_idx = filtered_df["ttft_p95"].idxmin()
        best_latency_config = filtered_df.loc[best_latency_idx]
        best_latency = best_latency_config["ttft_p95"]

        best_efficiency_idx = filtered_df["efficiency_ratio"].idxmax()
        best_efficiency_config = filtered_df.loc[best_efficiency_idx]
        best_efficiency = best_efficiency_config["efficiency_ratio"]

        best_itl_idx = filtered_df["itl_p95"].idxmin()
        best_itl_config = filtered_df.loc[best_itl_idx]
        best_itl = best_itl_config["itl_p95"]

        kpi_col1, kpi_col3, kpi_col2, kpi_col4 = st.columns(4)

        with kpi_col1:
            throughput_subtitle = (
                f"{best_throughput_config['accelerator']} | {best_throughput_config['model'].split('/')[-1]} | "
                f"TP={int(best_throughput_config['TP'])} | {best_throughput_config['version']} | "
                f"C={int(best_throughput_config['intended concurrency'])}"
            )
            st.markdown(
                create_kpi_card(
                    "🚀 Best Throughput",
                    best_throughput,
                    throughput_subtitle,
                    lambda x: f"{x:.1f} tok/s",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col3:
            efficiency_subtitle = (
                f"{best_efficiency_config['accelerator']} | {best_efficiency_config['model'].split('/')[-1]} | "
                f"TP={int(best_efficiency_config['TP'])} | {best_efficiency_config['version']} | "
                f"C={int(best_efficiency_config['intended concurrency'])} | "
                f"(Throughput ÷ TP)"
            )
            st.markdown(
                create_kpi_card(
                    "🎯 Most Efficient",
                    best_efficiency,
                    efficiency_subtitle,
                    lambda x: f"{x:.1f} tok/s/TP",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col2:
            ttft_subtitle = (
                f"{best_latency_config['accelerator']} | {best_latency_config['model'].split('/')[-1]} | "
                f"TP={int(best_latency_config['TP'])} | {best_latency_config['version']} | "
                f"C={int(best_latency_config['intended concurrency'])}"
            )
            st.markdown(
                create_kpi_card(
                    "⚡ Lowest TTFT Latency",
                    best_latency,
                    ttft_subtitle,
                    lambda x: f"{x:.1f} ms",
                ),
                unsafe_allow_html=True,
            )

        with kpi_col4:
            itl_subtitle = (
                f"{best_itl_config['accelerator']} | {best_itl_config['model'].split('/')[-1]} | "
                f"TP={int(best_itl_config['TP'])} | {best_itl_config['version']} | "
                f"C={int(best_itl_config['intended concurrency'])}"
            )
            st.markdown(
                create_kpi_card(
                    "⚡ Lowest Inter-Token Latency",
                    best_itl,
                    itl_subtitle,
                    lambda x: f"{x:.1f} ms",
                ),
                unsafe_allow_html=True,
            )


def render_header_with_theme_toggle():
    """Render the main header with theme toggle button."""
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        pass

    with col2:
        st.markdown(
            """
            <h1 style='text-align: center; margin-bottom: 0.5rem;' class='main-title'>
                📊 LLM Inference Performance Dashboard
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
            if current_mode == "auto":
                st.session_state.theme_mode = "light"
            elif current_mode == "light":
                st.session_state.theme_mode = "dark"
            else:
                st.session_state.theme_mode = "auto"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_confidentiality_notice():
    """Render the confidentiality notice."""
    st.warning(
        "⚠️ **CONFIDENTIAL**: Any data displayed here is only for internal use. If in doubt, please contact Ashish Kamra at #forum-psap."
    )


initialize_streamlit_config()
initialize_session_state()

st.markdown(get_app_css(), unsafe_allow_html=True)
apply_theme_css()

render_header_with_theme_toggle()
render_confidentiality_notice()

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

    if "url_filters_loaded" not in st.session_state:
        st.session_state.url_filters_loaded = True
        url_accelerators, url_models, url_versions, url_profile, url_tp_sizes = (
            decode_filters_from_url()
        )

        if any([url_accelerators, url_models, url_versions, url_profile, url_tp_sizes]):
            preferred_versions = [
                "RHAIIS-3.2.1",
                "RHAIIS-3.2.2",
                "vLLM-0.10.0",
                "vLLM-0.10.1.1",
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
                    "Profile B: Variable Workload (512/2k)"
                    if "Profile B: Variable Workload (512/2k)"
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
                "RHAIIS-3.2.1",
                "RHAIIS-3.2.2",
                "vLLM-0.10.0",
                "vLLM-0.10.1.1",
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
                "Profile B: Variable Workload (512/2k)"
                if "Profile B: Variable Workload (512/2k)"
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
    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col1:
        accelerators = sorted(df["accelerator"].unique().tolist())

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            acc_default = []
        elif st.session_state.get("reset_to_defaults", False):
            acc_default = st.session_state.get("baseline_accelerators", accelerators)
        else:
            acc_default = st.session_state.get("baseline_accelerators", accelerators)

        selected_accelerators = st.multiselect(
            "Select Accelerator(s)",
            accelerators,
            default=acc_default,
            key=f"accelerators_filter_{st.session_state.filter_change_key}",
        )

    with filter_col2:
        if selected_accelerators:
            available_models_for_accelerators = (
                df[df["accelerator"].isin(selected_accelerators)]["model"]
                .unique()
                .tolist()
            )
            models = sorted(available_models_for_accelerators)
        else:
            models = sorted(df["model"].unique().tolist())

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

        selected_models = st.multiselect(
            "Select Model(s)",
            models,
            default=models_default,
            key=f"models_filter_{st.session_state.filter_change_key}",
        )
        st.caption("💡 See dropdown for more available models.")

    with filter_col3:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]

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

        selected_versions = st.multiselect(
            "Select Version(s)",
            versions,
            default=versions_default,
            key=f"versions_filter_{st.session_state.filter_change_key}",
        )
        st.caption("💡 See dropdown for more available versions.")

    with filter_col4:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]

        profiles = (
            sorted(temp_df["profile"].unique().tolist()) if not temp_df.empty else []
        )

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            profiles_default = profiles[0] if profiles else None
        elif st.session_state.get("reset_to_defaults", False):
            baseline_profile = st.session_state.get("baseline_profile", None)
            profiles_default = (
                baseline_profile
                if baseline_profile in profiles
                else (profiles[0] if profiles else None)
            )
        else:
            baseline_profile = st.session_state.get("baseline_profile", None)
            profiles_default = (
                baseline_profile
                if baseline_profile in profiles
                else (profiles[0] if profiles else None)
            )

        selected_profile = (
            st.selectbox(
                "Select Input/Output Sequence Length (ISL/OSL)",
                profiles,
                index=(
                    profiles.index(profiles_default)
                    if profiles_default in profiles
                    else 0
                ),
                format_func=clean_profile_name,
                key=f"profile_filter_{st.session_state.filter_change_key}",
            )
            if profiles
            else None
        )

        selected_profiles = [selected_profile] if selected_profile is not None else []

    with filter_col5:
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_profiles:
            temp_df = temp_df[temp_df["profile"].isin(selected_profiles)]

        tp_sizes = (
            sorted(temp_df["TP"].dropna().unique().tolist())
            if not temp_df.empty
            else []
        )

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            tp_default = []
        elif st.session_state.get("reset_to_defaults", False):
            baseline_tp_sizes = st.session_state.get("baseline_tp_sizes", tp_sizes)
            tp_default = [tp for tp in baseline_tp_sizes if tp in tp_sizes]
        else:
            baseline_tp_sizes = st.session_state.get("baseline_tp_sizes", tp_sizes)
            tp_default = [tp for tp in baseline_tp_sizes if tp in tp_sizes]

        selected_tp = st.multiselect(
            "Select TP Size(s)",
            tp_sizes,
            default=tp_default,
            key=f"tp_filter_{st.session_state.filter_change_key}",
        )

    with filter_col6:
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button(
                "🔄 Reset to Defaults", help="Reset filters to system/URL defaults"
            ):
                st.session_state.clear_all_filters = False
                st.session_state.filters_were_cleared = False
                st.session_state.reset_to_defaults = True
                st.session_state.filter_change_key += 1
                st.rerun()

        with btn_col2:
            if st.button("🧹 Clear Filters", help="Clear all filter selections"):
                st.session_state.clear_all_filters = True
                st.session_state.filters_were_cleared = True
                st.session_state.filter_change_key += 1
                st.rerun()

        with btn_col3:
            if st.button(
                "🔗 Share Current View",
                help="Get a shareable URL with current filters applied",
            ):
                try:
                    st.toast(
                        "🔗 Shareable URL Generated! Copy the browser URL to share this view.",
                        icon="✅",
                    )
                except Exception as e:
                    st.toast(f"❌ Error generating shareable URL: {e}", icon="🚨")

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

    if not filtered_df.empty:
        accelerator_color_map = {
            "H200": "#1f77b4",
            "MI300X": "#ff7f0e",
            "TPU": "#2ca02c",
        }

        render_performance_plots_section(filtered_df)
        render_kpi_section(filtered_df)

        def analyze_performance_changes(df):
            comparison_data = []

            def get_version_type(version):
                return version.split("-")[0] if "-" in version else version

            df_with_type = df.copy()
            df_with_type["version_type"] = df_with_type["version"].apply(
                get_version_type
            )

            for (
                model,
                accelerator,
                tp,
                profile,
                version_type,
            ), group in df_with_type.groupby(
                ["model", "accelerator", "TP", "profile", "version_type"]
            ):
                if len(group["version"].unique()) < 2:
                    continue

                versions = sorted(group["version"].unique())

                # Compare each version with the previous one (within same type)
                for i in range(1, len(versions)):
                    older_version = versions[i - 1]
                    newer_version = versions[i]

                    older_version_data = group[group["version"] == older_version]
                    newer_version_data = group[group["version"] == newer_version]

                    # Find common concurrency levels between versions
                    older_concurrencies = set(
                        older_version_data["intended concurrency"].dropna()
                    )
                    newer_concurrencies = set(
                        newer_version_data["intended concurrency"].dropna()
                    )
                    common_concurrencies = older_concurrencies.intersection(
                        newer_concurrencies
                    )

                    if not common_concurrencies:
                        continue

                    # Calculate performance changes for each common concurrency level
                    metric_changes = {
                        "throughput": [],
                        "ttft": [],
                        "itl": [],
                        "efficiency": [],
                    }
                    metric_values = {
                        "throughput_older": [],
                        "throughput_newer": [],
                        "ttft_older": [],
                        "ttft_newer": [],
                        "itl_older": [],
                        "itl_newer": [],
                        "efficiency_older": [],
                        "efficiency_newer": [],
                    }

                    # Store all concurrency-level data for peak analysis
                    concurrency_data = []

                    for concurrency in common_concurrencies:
                        older_row = older_version_data[
                            older_version_data["intended concurrency"] == concurrency
                        ]
                        newer_row = newer_version_data[
                            newer_version_data["intended concurrency"] == concurrency
                        ]

                        if older_row.empty or newer_row.empty:
                            continue

                        older_row = older_row.iloc[0]
                        newer_row = newer_row.iloc[0]

                        # Store concurrency data for peak analysis
                        concurrency_data.append(
                            {
                                "concurrency": concurrency,
                                "older_throughput": (
                                    older_row["output_tok/sec"]
                                    if pd.notna(older_row["output_tok/sec"])
                                    else None
                                ),
                                "newer_throughput": (
                                    newer_row["output_tok/sec"]
                                    if pd.notna(newer_row["output_tok/sec"])
                                    else None
                                ),
                                "older_ttft": (
                                    older_row["ttft_p95"]
                                    if pd.notna(older_row["ttft_p95"])
                                    else None
                                ),
                                "newer_ttft": (
                                    newer_row["ttft_p95"]
                                    if pd.notna(newer_row["ttft_p95"])
                                    else None
                                ),
                                "older_itl": (
                                    older_row["itl_p95"]
                                    if pd.notna(older_row["itl_p95"])
                                    else None
                                ),
                                "newer_itl": (
                                    newer_row["itl_p95"]
                                    if pd.notna(newer_row["itl_p95"])
                                    else None
                                ),
                                "older_efficiency": (
                                    older_row["efficiency_ratio"]
                                    if pd.notna(older_row["efficiency_ratio"])
                                    else None
                                ),
                                "newer_efficiency": (
                                    newer_row["efficiency_ratio"]
                                    if pd.notna(newer_row["efficiency_ratio"])
                                    else None
                                ),
                            }
                        )

                        # Throughput comparison
                        if (
                            pd.notna(older_row["output_tok/sec"])
                            and pd.notna(newer_row["output_tok/sec"])
                            and older_row["output_tok/sec"] > 0
                        ):
                            change = (
                                (
                                    newer_row["output_tok/sec"]
                                    - older_row["output_tok/sec"]
                                )
                                / older_row["output_tok/sec"]
                            ) * 100
                            metric_changes["throughput"].append(change)
                            metric_values["throughput_older"].append(
                                older_row["output_tok/sec"]
                            )
                            metric_values["throughput_newer"].append(
                                newer_row["output_tok/sec"]
                            )

                        # TTFT comparison
                        if (
                            pd.notna(older_row["ttft_p95"])
                            and pd.notna(newer_row["ttft_p95"])
                            and older_row["ttft_p95"] > 0
                        ):
                            change = (
                                (newer_row["ttft_p95"] - older_row["ttft_p95"])
                                / older_row["ttft_p95"]
                            ) * 100
                            metric_changes["ttft"].append(change)
                            metric_values["ttft_older"].append(older_row["ttft_p95"])
                            metric_values["ttft_newer"].append(newer_row["ttft_p95"])

                        # ITL comparison
                        if (
                            pd.notna(older_row["itl_p95"])
                            and pd.notna(newer_row["itl_p95"])
                            and older_row["itl_p95"] > 0
                        ):
                            change = (
                                (newer_row["itl_p95"] - older_row["itl_p95"])
                                / older_row["itl_p95"]
                            ) * 100
                            metric_changes["itl"].append(change)
                            metric_values["itl_older"].append(older_row["itl_p95"])
                            metric_values["itl_newer"].append(newer_row["itl_p95"])

                        # Efficiency comparison
                        if (
                            pd.notna(older_row["efficiency_ratio"])
                            and pd.notna(newer_row["efficiency_ratio"])
                            and older_row["efficiency_ratio"] > 0
                        ):
                            change = (
                                (
                                    newer_row["efficiency_ratio"]
                                    - older_row["efficiency_ratio"]
                                )
                                / older_row["efficiency_ratio"]
                            ) * 100
                            metric_changes["efficiency"].append(change)
                            metric_values["efficiency_older"].append(
                                older_row["efficiency_ratio"]
                            )
                            metric_values["efficiency_newer"].append(
                                newer_row["efficiency_ratio"]
                            )

                    # Find peak values from same concurrency level for "(Old → New)" display
                    def find_peak_from_same_concurrency(
                        concurrency_data,
                        metric_older,
                        metric_newer,
                        higher_is_better=True,
                    ):
                        """Find peak values from the same concurrency level for both older and newer versions."""
                        valid_data = [
                            d
                            for d in concurrency_data
                            if d[metric_older] is not None
                            and d[metric_newer] is not None
                        ]

                        if not valid_data:
                            return None, None, None

                        if higher_is_better:
                            # For throughput: find max older, then check if newer at same concurrency is available
                            older_peak = max(valid_data, key=lambda x: x[metric_older])
                            # Check if newer has data at the same concurrency level
                            same_concurrency_newer = next(
                                (
                                    d[metric_newer]
                                    for d in valid_data
                                    if d["concurrency"] == older_peak["concurrency"]
                                ),
                                None,
                            )

                            if same_concurrency_newer is not None:
                                return (
                                    older_peak[metric_older],
                                    same_concurrency_newer,
                                    older_peak["concurrency"],
                                )
                        else:
                            # For latency: find min older, then check if newer at same concurrency is available
                            older_peak = min(valid_data, key=lambda x: x[metric_older])
                            # Check if newer has data at the same concurrency level
                            same_concurrency_newer = next(
                                (
                                    d[metric_newer]
                                    for d in valid_data
                                    if d["concurrency"] == older_peak["concurrency"]
                                ),
                                None,
                            )

                            if same_concurrency_newer is not None:
                                return (
                                    older_peak[metric_older],
                                    same_concurrency_newer,
                                    older_peak["concurrency"],
                                )

                        return None, None, None

                    # Find peak comparisons from same concurrency levels
                    peak_comparisons = {}

                    # Throughput (higher is better)
                    (
                        throughput_older_peak,
                        throughput_newer_peak,
                        throughput_peak_concurrency,
                    ) = find_peak_from_same_concurrency(
                        concurrency_data,
                        "older_throughput",
                        "newer_throughput",
                        higher_is_better=True,
                    )
                    if throughput_older_peak is not None:
                        peak_comparisons["throughput_peak_older"] = (
                            throughput_older_peak
                        )
                        peak_comparisons["throughput_peak_newer"] = (
                            throughput_newer_peak
                        )
                        peak_comparisons["throughput_peak_concurrency"] = (
                            throughput_peak_concurrency
                        )

                    # TTFT (lower is better)
                    ttft_older_peak, ttft_newer_peak, ttft_peak_concurrency = (
                        find_peak_from_same_concurrency(
                            concurrency_data,
                            "older_ttft",
                            "newer_ttft",
                            higher_is_better=False,
                        )
                    )
                    if ttft_older_peak is not None:
                        peak_comparisons["ttft_peak_older"] = ttft_older_peak
                        peak_comparisons["ttft_peak_newer"] = ttft_newer_peak
                        peak_comparisons["ttft_peak_concurrency"] = (
                            ttft_peak_concurrency
                        )

                    # ITL (lower is better)
                    itl_older_peak, itl_newer_peak, itl_peak_concurrency = (
                        find_peak_from_same_concurrency(
                            concurrency_data,
                            "older_itl",
                            "newer_itl",
                            higher_is_better=False,
                        )
                    )
                    if itl_older_peak is not None:
                        peak_comparisons["itl_peak_older"] = itl_older_peak
                        peak_comparisons["itl_peak_newer"] = itl_newer_peak
                        peak_comparisons["itl_peak_concurrency"] = itl_peak_concurrency

                    # Aggregate the changes (use median to be robust to outliers)
                    metrics_comparison = {}
                    metrics_comparison["concurrency_levels_compared"] = len(
                        common_concurrencies
                    )
                    metrics_comparison["common_concurrencies"] = sorted(
                        common_concurrencies
                    )

                    if metric_changes["throughput"]:
                        metrics_comparison["throughput_change"] = np.median(
                            metric_changes["throughput"]
                        )
                        metrics_comparison["throughput_older"] = np.median(
                            metric_values["throughput_older"]
                        )
                        metrics_comparison["throughput_newer"] = np.median(
                            metric_values["throughput_newer"]
                        )

                    if metric_changes["ttft"]:
                        metrics_comparison["ttft_change"] = np.median(
                            metric_changes["ttft"]
                        )
                        metrics_comparison["ttft_older"] = np.median(
                            metric_values["ttft_older"]
                        )
                        metrics_comparison["ttft_newer"] = np.median(
                            metric_values["ttft_newer"]
                        )

                    if metric_changes["itl"]:
                        metrics_comparison["itl_change"] = np.median(
                            metric_changes["itl"]
                        )
                        metrics_comparison["itl_older"] = np.median(
                            metric_values["itl_older"]
                        )
                        metrics_comparison["itl_newer"] = np.median(
                            metric_values["itl_newer"]
                        )

                    if metric_changes["efficiency"]:
                        metrics_comparison["efficiency_change"] = np.median(
                            metric_changes["efficiency"]
                        )
                        metrics_comparison["efficiency_older"] = np.median(
                            metric_values["efficiency_older"]
                        )
                        metrics_comparison["efficiency_newer"] = np.median(
                            metric_values["efficiency_newer"]
                        )

                    comparison_data.append(
                        {
                            "model": model.split("/")[-1] if "/" in model else model,
                            "accelerator": accelerator,
                            "tp": tp,
                            "profile": (
                                profile.split(":")[0] if ":" in profile else profile
                            ),
                            "version_type": version_type,
                            "older_version": older_version,
                            "newer_version": newer_version,
                            **metrics_comparison,
                            **peak_comparisons,
                        }
                    )

            return pd.DataFrame(comparison_data)

        render_regression_analysis_section(filtered_df, analyze_performance_changes)
        render_model_performance_comparison_section(filtered_df, accelerator_color_map)
        render_cost_analysis_section(filtered_df, accelerator_color_map)
        render_performance_rankings_section(filtered_df)
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
                                        tree_text += f"        📋 {profile} → TP Sizes: {tp_list}\n"
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
