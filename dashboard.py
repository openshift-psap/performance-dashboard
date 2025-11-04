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

# Import MLPerf dashboard
try:
    from mlperf_datacenter import render_mlperf_dashboard

    MLPERF_AVAILABLE = True
except ImportError:
    MLPERF_AVAILABLE = False
    print("Warning: mlperf_datacenter module not found. MLPerf view will be disabled.")


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

        # Add units to y-axis label for certain metrics
        y_axis_display_label = y_axis_label
        if y_axis == "ttft_p95_s":
            y_axis_display_label = f"{y_axis_label} (s)"
        elif y_axis == "itl_p95":
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


def load_pareto_data(csv_file_path):
    """Load benchmark results from consolidated CSV file for Pareto analysis.

    Args:
        csv_file_path: Path to the consolidated CSV file.

    Returns:
        List of result dictionaries for Pareto tradeoff analysis.
    """
    try:
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


def render_pareto_plots_section():
    """📊 Pareto Tradeoff Analysis Section - Interactive plots showing performance vs latency tradeoffs."""
    if "pareto_expanded" not in st.session_state:
        st.session_state.pareto_expanded = False

    with st.expander(
        "📊 Pareto Tradeoff Analysis", expanded=st.session_state.pareto_expanded
    ):
        # Load data
        results = load_pareto_data("consolidated_dashboard.csv")

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
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

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

            # Set default version
            default_version = "RHAIIS-3.2.3"
            if default_version not in unique_versions and unique_versions:
                default_version = unique_versions[0]

            default_idx = (
                unique_versions.index(default_version)
                if default_version in unique_versions
                else 0
            )

            selected_version = st.selectbox(
                "Select Version",
                options=unique_versions,
                index=default_idx,
                key="pareto_version_select",
                on_change=keep_expander_open,
                args=("pareto_expanded",),
            )

        # Filter by selected version
        results = [
            r for r in results if r.get("version", "Unknown") == selected_version
        ]
        if not results:
            st.warning(f"No results found for version: '{selected_version}'")
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

        # Filter by selected hardware
        if selected_hw != "All Accelerators":
            results = [r for r in results if r.get("hw", "").upper() == selected_hw]
            if not results:
                st.warning(f"No results found for accelerator: '{selected_hw}'")
                return

        # Get unique accelerators and TP sizes
        unique_hw = sorted({r.get("hw", "unknown") for r in results})
        unique_tps = sorted({r.get("tp", 1) for r in results})

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

        # Create unique color mapping for each accelerator+TP combination
        hw_tp_color_map = {}
        color_idx = 0
        for hw in sorted(unique_hw):
            for tp in sorted(unique_tps):
                hw_tp_key = f"{hw.lower()}_{tp}"
                hw_tp_color_map[hw_tp_key] = color_palette[
                    color_idx % len(color_palette)
                ]
                color_idx += 1

        # Create tabs for different plot types
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

                # Group by accelerator first, then by TP size
                for hw in sorted(unique_hw):
                    for tp_size in sorted(unique_tps):
                        # Filter results for this accelerator and TP combination
                        hw_tp_results = [
                            r
                            for r in filtered_results
                            if r.get("hw", "unknown").lower() == hw.lower()
                            and r.get("tp", 1) == tp_size
                        ]

                        if hw_tp_results:
                            # Sort by concurrency for proper line drawing
                            hw_tp_results_sorted = sorted(
                                hw_tp_results, key=lambda x: x.get("conc", 0)
                            )

                            xs = [r.get("median_e2el", 0) for r in hw_tp_results_sorted]
                            ys = [
                                r.get("tput_per_gpu", 0) for r in hw_tp_results_sorted
                            ]
                            models = [
                                r.get("model", "Unknown") for r in hw_tp_results_sorted
                            ]
                            concs = [r.get("conc", "N/A") for r in hw_tp_results_sorted]
                            versions = [
                                r.get("version", "N/A") for r in hw_tp_results_sorted
                            ]
                            isl_osls = [
                                r.get("isl_osl", "N/A") for r in hw_tp_results_sorted
                            ]

                            # Get unique color for this accelerator+TP combination
                            hw_tp_key = f"{hw.lower()}_{tp_size}"
                            color = hw_tp_color_map.get(hw_tp_key, "#999999")

                            hover_text = [
                                f"Accelerator: {hw.upper()}<br>"
                                f"TP Size: {tp_size}<br>"
                                f"Concurrent Requests: {conc} Users<br>"
                                f"Latency: {x:.2f}s<br>"
                                f"Throughput: {y:.2f} tok/s/gpu"
                                for conc, isl_osl, model, version, x, y in zip(
                                    concs, isl_osls, models, versions, xs, ys
                                )
                            ]

                            fig.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="markers+lines",
                                    name=f"{hw.upper()} (TP={tp_size})",
                                    marker={
                                        "size": 10,
                                        "color": color,
                                        "line": {"width": 1, "color": "white"},
                                    },
                                    line={"color": color, "width": 2},
                                    hovertext=hover_text,
                                    hoverinfo="text",
                                    legendgroup=hw.upper(),  # Group by accelerator in legend
                                )
                            )

                fig.update_layout(
                    title="Note: Throughput is Total Tokens per second(prompt + output tokens combined)",
                    xaxis_title="End-to-end Latency (s)",
                    yaxis_title="Token Throughput per GPU (tok/s/gpu)",
                    template="plotly_dark",
                    hovermode="closest",
                    showlegend=True,
                    legend={"title": "Accelerator (TP Size)", "font": {"size": 12}},
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

                # Group by accelerator first, then by TP size
                for hw in sorted(unique_hw):
                    for tp_size in sorted(unique_tps):
                        # Filter results for this accelerator and TP combination
                        hw_tp_results = [
                            r
                            for r in filtered_results
                            if r.get("hw", "unknown").lower() == hw.lower()
                            and r.get("tp", 1) == tp_size
                        ]

                        if hw_tp_results:
                            # Sort by concurrency for proper line drawing
                            hw_tp_results_sorted = sorted(
                                hw_tp_results, key=lambda x: x.get("conc", 0)
                            )

                            xs = [
                                r.get("median_intvty", 0) for r in hw_tp_results_sorted
                            ]
                            ys = [
                                r.get("tput_per_gpu", 0) for r in hw_tp_results_sorted
                            ]
                            models = [
                                r.get("model", "Unknown") for r in hw_tp_results_sorted
                            ]
                            concs = [r.get("conc", "N/A") for r in hw_tp_results_sorted]
                            versions = [
                                r.get("version", "N/A") for r in hw_tp_results_sorted
                            ]
                            isl_osls = [
                                r.get("isl_osl", "N/A") for r in hw_tp_results_sorted
                            ]

                            # Get unique color for this accelerator+TP combination
                            hw_tp_key = f"{hw.lower()}_{tp_size}"
                            color = hw_tp_color_map.get(hw_tp_key, "#999999")

                            hover_text = [
                                f"Accelerator: {hw.upper()}<br>"
                                f"TP Size: {tp_size}<br>"
                                f"Concurrent Requests: {conc} Users<br>"
                                f"Interactivity: {x:.2f} tok/s/user<br>"
                                f"Throughput: {y:.2f} tok/s/gpu"
                                for conc, isl_osl, model, version, x, y in zip(
                                    concs, isl_osls, models, versions, xs, ys
                                )
                            ]

                            fig.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="markers+lines",
                                    name=f"{hw.upper()} (TP={tp_size})",
                                    marker={
                                        "size": 10,
                                        "color": color,
                                        "line": {"width": 1, "color": "white"},
                                    },
                                    line={"color": color, "width": 2},
                                    hovertext=hover_text,
                                    hoverinfo="text",
                                    legendgroup=hw.upper(),  # Group by accelerator in legend
                                )
                            )

                fig.update_layout(
                    title="Note: Throughput is Total Tokens per second(prompt + output tokens combined)",
                    xaxis_title="Interactivity (tok/s/user)",
                    yaxis_title="Token Throughput per GPU (tok/s/gpu)",
                    template="plotly_dark",
                    hovermode="closest",
                    showlegend=True,
                    legend={"title": "Accelerator (TP Size)", "font": {"size": 12}},
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        with st.expander("📊 Summary Statistics"):
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

                st.dataframe(
                    df_display.sort_values(
                        by="tput_per_gpu", ascending=False
                    ).reset_index(drop=True),
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
                            help="Throughput per GPU: Total tokens/second divided by TP size. Calculated as (total_tok/sec ÷ TP). Higher is better.",
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

                    # Calculate peak throughput change percentage
                    def calc_peak_throughput_change(row):
                        if (
                            pd.notna(row.get("throughput_peak_older"))
                            and pd.notna(row.get("throughput_peak_newer"))
                            and row.get("throughput_peak_older") > 0
                        ):
                            pct_change = (
                                (
                                    row["throughput_peak_newer"]
                                    - row["throughput_peak_older"]
                                )
                                / row["throughput_peak_older"]
                            ) * 100
                            # Categorize the change
                            abs_change = abs(pct_change)
                            if abs_change < significance_threshold:
                                icon = "🟡"
                            elif pct_change > 0:
                                icon = "🟢"
                            else:
                                icon = "🔴"
                            return f"{icon} {pct_change:.1f}%"
                        return "No Data"

                    regression_display_df["Peak Throughput Change"] = (
                        regression_display_df.apply(calc_peak_throughput_change, axis=1)
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
                        [
                            "Median Throughput Change",
                            "Peak Throughput Change",
                            "Throughput (Old → New)",
                        ]
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
                                f"{row['ttft_peak_older'] / 1000:.3f} → {row['ttft_peak_newer'] / 1000:.3f} s (C={int(row['ttft_peak_concurrency'])})"
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


def render_version_comparison_section(filtered_df):
    """⚖️ Version Comparison Section - Compare performance between two versions."""
    if "version_comparison_expanded" not in st.session_state:
        st.session_state.version_comparison_expanded = False

    with st.expander(
        "⚖️ Compare Versions", expanded=st.session_state.version_comparison_expanded
    ):
        st.markdown(
            "💡 **Compare performance metrics between two Inference Server versions for common models with same configurations.**"
        )
        st.info(
            "💡 **Tip**: Check the **'Select All Models'** checkbox in the filters above to see all available comparisons."
        )

        # Get available versions from filtered data
        available_versions = sorted(filtered_df["version"].unique().tolist())

        if len(available_versions) < 2:
            st.warning(
                "⚠️ Need at least 2 versions in the filtered data to compare. Please adjust your filters."
            )
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            version_1 = st.selectbox(
                "Select Version 1",
                options=available_versions,
                index=0,
                key="version_comparison_v1",
                on_change=keep_expander_open,
                args=("version_comparison_expanded",),
            )

        with col2:
            # Filter out version_1 from version_2 options
            version_2_options = [v for v in available_versions if v != version_1]
            version_2 = (
                st.selectbox(
                    "Select Version 2",
                    options=version_2_options,
                    index=0 if version_2_options else None,
                    key="version_comparison_v2",
                    on_change=keep_expander_open,
                    args=("version_comparison_expanded",),
                )
                if version_2_options
                else None
            )

        with col3:
            metric_options = {
                "Throughput (Output Tokens/sec)": "output_tok/sec",
                "TTFT P95 (s)": "ttft_p95_s",
                "ITL P95 (ms)": "itl_p95",
            }
            metric_label = st.selectbox(
                "Select Metric to Compare",
                options=list(metric_options.keys()),
                key="version_comparison_metric",
                on_change=keep_expander_open,
                args=("version_comparison_expanded",),
            )
            metric_column = metric_options[metric_label]

        st.info(
            "ℹ️ **Comparison Direction**: **Version 1** is the baseline/reference. "
            "All results show how **Version 1** performs relative to **Version 2**. "
            "Positive changes indicate Version 1 is better; negative changes indicate Version 2 was better."
        )

        if not version_2:
            st.warning("⚠️ Please select a second version to compare.")
            return

        # Filter data for each version
        df_v1 = filtered_df[filtered_df["version"] == version_1].copy()
        df_v2 = filtered_df[filtered_df["version"] == version_2].copy()

        # Find common models with same configurations
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

        if common_configs.empty:
            st.warning(
                f"⚠️ No common model configurations found between {version_1} and {version_2}."
            )
            return

        # Get unique models from common configs
        common_models = common_configs["model"].unique()

        st.success(
            f"✅ Found {len(common_models)} common model(s) between {version_1} and {version_2}"
        )

        # Create summary table showing quick comparison results
        summary_col1, summary_col2 = st.columns([4, 1])
        with summary_col1:
            st.markdown("### 📊 Quick Comparison Summary")
            st.markdown(f"Performance of **{version_1}** compared to **{version_2}**:")
        with summary_col2:
            with st.popover("ℹ️ How are these calculated?"):
                st.markdown("""
                **Mean Change Calculation:**
                - Calculated by taking the percentage change at each common concurrency level, then taking the mean (average) of all those changes
                - Shows the average performance difference across all concurrency levels
                - Can be affected by outliers (extreme values)
                - Formula: `mean([(v1 - v2) / v2 × 100 for each concurrency level])`

                **Median Change Calculation:**
                - Calculated by taking the percentage change at each common concurrency level, then taking the median of all those changes
                - Shows the typical performance difference across all concurrency levels
                - More robust to outliers than mean - better represents typical performance
                - Formula: `median([(v1 - v2) / v2 × 100 for each concurrency level])`

                **Peak Change Calculation:**
                - **For Throughput**: `((Version 1 Max - Version 2 Max) / Version 2 Max) × 100`
                  - Compares maximum throughput values (best = highest performance)
                  - Higher is better
                - **For Latency (TTFT/ITL)**: `((Version 1 Latency @ Max Throughput - Version 2 Latency @ Max Throughput) / Version 2 Latency @ Max Throughput) × 100`
                  - Compares latency values at the concurrency where max throughput occurs for each version
                  - This shows latency characteristics at peak performance
                  - Lower is better

                **Status Classification:**
                The status emoji is determined by consensus across all three metrics (Mean, Median, and Peak change):
                - 🟢 **Better**: At least 2 out of 3 metrics show ≥5% improvement
                - 🟡 **Similar**: Mixed signals (some metrics up, some down) or all metrics show <5% difference
                - 🔴 **Worse**: At least 2 out of 3 metrics show ≥5% decline

                This consensus approach provides a more robust assessment by requiring multiple metrics to agree before declaring a clear winner or loser.

                **Note**: Each accelerator-TP combination is compared independently across all common concurrency levels.
                """)

        summary_rows = []
        is_higher_better = metric_column == "output_tok/sec"

        for model in sorted(common_models):
            model_short = model.split("/")[-1] if "/" in model else model
            model_configs = common_configs[common_configs["model"] == model]
            common_acc_tp = model_configs[["accelerator", "TP"]].drop_duplicates()

            # Calculate summary for each accelerator-TP combination
            for _, acc_tp_row in common_acc_tp.iterrows():
                accelerator = acc_tp_row["accelerator"]
                tp = int(acc_tp_row["TP"])

                # Filter data for this specific model and configuration
                model_v1_data = df_v1[
                    (df_v1["model"] == model)
                    & (df_v1["accelerator"] == accelerator)
                    & (df_v1["TP"] == tp)
                ].copy()

                model_v2_data = df_v2[
                    (df_v2["model"] == model)
                    & (df_v2["accelerator"] == accelerator)
                    & (df_v2["TP"] == tp)
                ].copy()

                # Get common concurrencies
                v1_concurrencies = set(model_v1_data["intended concurrency"].unique())
                v2_concurrencies = set(model_v2_data["intended concurrency"].unique())
                common_concurrencies = sorted(
                    v1_concurrencies.intersection(v2_concurrencies)
                )

                if common_concurrencies:
                    v1_common = model_v1_data[
                        model_v1_data["intended concurrency"].isin(common_concurrencies)
                    ]
                    v2_common = model_v2_data[
                        model_v2_data["intended concurrency"].isin(common_concurrencies)
                    ]

                    # Calculate median change across all common concurrencies
                    metric_changes = []
                    for concurrency in common_concurrencies:
                        v1_row = v1_common[
                            v1_common["intended concurrency"] == concurrency
                        ]
                        v2_row = v2_common[
                            v2_common["intended concurrency"] == concurrency
                        ]

                        if not v1_row.empty and not v2_row.empty:
                            v1_val = v1_row.iloc[0][metric_column]
                            v2_val = v2_row.iloc[0][metric_column]

                            if pd.notna(v1_val) and pd.notna(v2_val) and v2_val > 0:
                                change = ((v1_val - v2_val) / v2_val) * 100
                                metric_changes.append(change)

                    # Calculate median and mean change
                    median_change = np.median(metric_changes) if metric_changes else 0
                    mean_change = np.mean(metric_changes) if metric_changes else 0

                    # Calculate percentage difference (v1 compared to v2) for peak values
                    if is_higher_better:
                        # For throughput, compare max values
                        v1_max = v1_common[metric_column].max()
                        v2_max = v2_common[metric_column].max()
                        if v2_max > 0:
                            pct_change = ((v1_max - v2_max) / v2_max) * 100
                        else:
                            pct_change = 0
                    else:
                        # For latency, compare values at max throughput
                        v1_max_throughput_idx = v1_common["output_tok/sec"].idxmax()
                        v2_max_throughput_idx = v2_common["output_tok/sec"].idxmax()
                        v1_latency_at_max = v1_common.loc[
                            v1_max_throughput_idx, metric_column
                        ]
                        v2_latency_at_max = v2_common.loc[
                            v2_max_throughput_idx, metric_column
                        ]
                        if v2_latency_at_max > 0:
                            pct_change = (
                                (v1_latency_at_max - v2_latency_at_max)
                                / v2_latency_at_max
                            ) * 100
                        else:
                            pct_change = 0

                    # Determine emoji and status based on all three metrics (mean, median, peak)
                    # For throughput: positive is good, negative is bad
                    # For latency: negative is good (lower), positive is bad (higher)
                    mean_improvement = mean_change if is_higher_better else -mean_change
                    median_improvement = (
                        median_change if is_higher_better else -median_change
                    )
                    peak_improvement = pct_change if is_higher_better else -pct_change

                    # Count how many metrics show improvement, decline, or similarity
                    improvements = sum(
                        [
                            1 if mean_improvement >= 5 else 0,
                            1 if median_improvement >= 5 else 0,
                            1 if peak_improvement >= 5 else 0,
                        ]
                    )

                    declines = sum(
                        [
                            1 if mean_improvement <= -5 else 0,
                            1 if median_improvement <= -5 else 0,
                            1 if peak_improvement <= -5 else 0,
                        ]
                    )

                    # Determine status based on consensus across all three metrics
                    if improvements >= 2:
                        # At least 2 out of 3 show improvement
                        emoji = "🟢"
                        status = "Better"
                    elif declines >= 2:
                        # At least 2 out of 3 show decline
                        emoji = "🔴"
                        status = "Worse"
                    else:
                        # Mixed signals or all three show similar performance
                        emoji = "🟡"
                        status = "Similar"

                    # Format the change text
                    if is_higher_better:
                        change_text = (
                            f"{'+' if pct_change > 0 else ''}{pct_change:.1f}%"
                        )
                        median_change_text = (
                            f"{'+' if median_change > 0 else ''}{median_change:.1f}%"
                        )
                        mean_change_text = (
                            f"{'+' if mean_change > 0 else ''}{mean_change:.1f}%"
                        )
                    else:
                        change_text = (
                            f"{'+' if pct_change > 0 else ''}{pct_change:.1f}%"
                        )
                        median_change_text = (
                            f"{'+' if median_change > 0 else ''}{median_change:.1f}%"
                        )
                        mean_change_text = (
                            f"{'+' if mean_change > 0 else ''}{mean_change:.1f}%"
                        )

                    summary_rows.append(
                        {
                            "": emoji,
                            "Model": model_short,
                            "Accelerator": accelerator,
                            "TP": tp,
                            "Status": status,
                            "Mean change": mean_change_text,
                            "Median change": median_change_text,
                            "Peak change": change_text,
                        }
                    )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.caption(
                "**Status**: Based on consensus across all three metrics (Mean, Median, Peak) | "
                "🟢 Better: ≥2 metrics show ≥5% improvement | "
                "🟡 Similar: Mixed signals or <5% difference | "
                "🔴 Worse: ≥2 metrics show ≥5% decline | "
                "**Mean**: average difference | **Median**: typical difference (robust to outliers) | **Peak**: best-case difference"
            )

        st.markdown("---")
        st.markdown("### 📈 Detailed Comparisons")

        # For each common model, create an expander with comparison graph
        for model in sorted(common_models):
            model_short = model.split("/")[-1] if "/" in model else model
            model_configs = common_configs[common_configs["model"] == model]

            # Get unique accelerator and TP combinations for this model that are common in BOTH versions
            config_summary = []
            for _, row in model_configs.iterrows():
                config_summary.append(f"{row['accelerator']} (TP={int(row['TP'])})")

            with st.expander(
                f"📈 {model_short} - {', '.join(set(config_summary))}", expanded=False
            ):
                # Get data for this model from both versions
                # But ONLY for the common accelerator/TP combinations
                common_acc_tp = model_configs[["accelerator", "TP"]].drop_duplicates()

                # Filter to only common configurations
                model_v1_data = pd.merge(
                    df_v1[df_v1["model"] == model],
                    common_acc_tp,
                    on=["accelerator", "TP"],
                    how="inner",
                ).copy()

                model_v2_data = pd.merge(
                    df_v2[df_v2["model"] == model],
                    common_acc_tp,
                    on=["accelerator", "TP"],
                    how="inner",
                ).copy()

                # Add version label
                model_v1_data["version_label"] = version_1
                model_v2_data["version_label"] = version_2

                # Combine data
                combined_data = pd.concat([model_v1_data, model_v2_data])

                # Create identifier for grouping
                combined_data["config_id"] = (
                    combined_data["accelerator"]
                    + " | TP="
                    + combined_data["TP"].astype(str)
                    + " | "
                    + combined_data["version_label"]
                )

                # Sort by concurrency
                combined_data = combined_data.sort_values("intended concurrency")

                # Create line plot
                fig = px.line(
                    combined_data,
                    x="intended concurrency",
                    y=metric_column,
                    color="config_id",
                    markers=True,
                    title=f"{model_short}: Concurrency vs {metric_label}",
                    labels={
                        "intended concurrency": "Concurrency",
                        metric_column: metric_label,
                        "config_id": "Configuration",
                    },
                    template="plotly_white",
                )

                fig.update_layout(
                    legend_title_text="Configuration (Accelerator | TP | Version)",
                    legend={"font": {"size": 12}},
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Now show separate summary statistics for each accelerator-TP combination
                for _, acc_tp_row in common_acc_tp.iterrows():
                    accelerator = acc_tp_row["accelerator"]
                    tp = int(acc_tp_row["TP"])

                    st.markdown("---")
                    st.markdown(f"### {accelerator} (TP={tp})")

                    # Filter data for this specific accelerator-TP combination
                    v1_config_data = model_v1_data[
                        (model_v1_data["accelerator"] == accelerator)
                        & (model_v1_data["TP"] == tp)
                    ].copy()

                    v2_config_data = model_v2_data[
                        (model_v2_data["accelerator"] == accelerator)
                        & (model_v2_data["TP"] == tp)
                    ].copy()

                    # Get common concurrencies for this specific configuration
                    v1_concurrencies = set(
                        v1_config_data["intended concurrency"].unique()
                    )
                    v2_concurrencies = set(
                        v2_config_data["intended concurrency"].unique()
                    )
                    common_concurrencies = sorted(
                        v1_concurrencies.intersection(v2_concurrencies)
                    )

                    if common_concurrencies:
                        # Filter to only common concurrencies for comparison
                        v1_common = v1_config_data[
                            v1_config_data["intended concurrency"].isin(
                                common_concurrencies
                            )
                        ]
                        v2_common = v2_config_data[
                            v2_config_data["intended concurrency"].isin(
                                common_concurrencies
                            )
                        ]

                        # Calculate statistics at common concurrencies
                        v1_median = v1_common[metric_column].median()
                        v1_max = v1_common[metric_column].max()
                        v1_min = v1_common[metric_column].min()

                        v2_median = v2_common[metric_column].median()
                        v2_max = v2_common[metric_column].max()
                        v2_min = v2_common[metric_column].min()

                        # Find concurrency at which max and min occurred
                        v1_max_concurrency = v1_common[
                            v1_common[metric_column] == v1_max
                        ]["intended concurrency"].iloc[0]
                        v1_min_concurrency = v1_common[
                            v1_common[metric_column] == v1_min
                        ]["intended concurrency"].iloc[0]
                        v2_max_concurrency = v2_common[
                            v2_common[metric_column] == v2_max
                        ]["intended concurrency"].iloc[0]
                        v2_min_concurrency = v2_common[
                            v2_common[metric_column] == v2_min
                        ]["intended concurrency"].iloc[0]

                        # Create summary statistics table
                        st.markdown("**Summary Statistics (at common concurrencies):**")

                        # Get unit and precision for display
                        unit = ""
                        precision = 2
                        if metric_column == "output_tok/sec":
                            unit = " tokens/s"
                            precision = 2
                        elif metric_column == "ttft_p95_s":
                            unit = " s"
                            precision = 3
                        elif metric_column == "itl_p95":
                            unit = " ms"
                            precision = 2

                        summary_data = {
                            "Statistic": [
                                f"Median {metric_label}",
                                f"Max {metric_label} (at concurrency)",
                                f"Min {metric_label} (at concurrency)",
                            ],
                            version_1: [
                                f"{v1_median:.{precision}f}{unit}",
                                f"{v1_max:.{precision}f}{unit} (C={int(v1_max_concurrency)})",
                                f"{v1_min:.{precision}f}{unit} (C={int(v1_min_concurrency)})",
                            ],
                            version_2: [
                                f"{v2_median:.{precision}f}{unit}",
                                f"{v2_max:.{precision}f}{unit} (C={int(v2_max_concurrency)})",
                                f"{v2_min:.{precision}f}{unit} (C={int(v2_min_concurrency)})",
                            ],
                        }
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(
                            summary_df, use_container_width=True, hide_index=True
                        )

                        # Determine which version is better and by how much (based on best values)
                        # For throughput, higher is better; for latency, lower is better
                        is_higher_better = metric_column == "output_tok/sec"

                        if is_higher_better:
                            # For throughput, compare max values (best = highest)
                            # Calculate v1 compared to v2
                            if v2_max > 0:
                                pct_change = ((v1_max - v2_max) / v2_max) * 100
                                abs_pct_change = abs(pct_change)

                                if abs_pct_change < 5:
                                    # Similar performance (< 5% difference)
                                    st.warning(
                                        f"🟡 **{version_1}** and **{version_2}** have similar performance (**{abs_pct_change:.1f}%** difference)"
                                    )
                                elif pct_change > 0:
                                    # v1 is better
                                    st.success(
                                        f"✅ **{version_1}** is **{abs_pct_change:.1f}% better** than **{version_2}** (higher max {metric_label.lower()} at C={int(v1_max_concurrency)})"
                                    )
                                else:
                                    # v1 is worse
                                    st.error(
                                        f"❌ **{version_1}** is **{abs_pct_change:.1f}% worse** than **{version_2}** (lower max {metric_label.lower()} at C={int(v1_max_concurrency)})"
                                    )
                        else:  # Lower is better (for latency metrics)
                            # For latency, compare values at max throughput concurrency
                            # First, find where max throughput occurs for each version
                            v1_max_throughput_idx = v1_common["output_tok/sec"].idxmax()
                            v2_max_throughput_idx = v2_common["output_tok/sec"].idxmax()

                            v1_latency_at_max = v1_common.loc[
                                v1_max_throughput_idx, metric_column
                            ]
                            v2_latency_at_max = v2_common.loc[
                                v2_max_throughput_idx, metric_column
                            ]

                            v1_concurrency_at_max = v1_common.loc[
                                v1_max_throughput_idx, "intended concurrency"
                            ]
                            v2_concurrency_at_max = v2_common.loc[
                                v2_max_throughput_idx, "intended concurrency"
                            ]

                            # Calculate v1 compared to v2
                            if v2_latency_at_max > 0:
                                pct_change = (
                                    (v1_latency_at_max - v2_latency_at_max)
                                    / v2_latency_at_max
                                ) * 100
                                abs_pct_change = abs(pct_change)

                                if abs_pct_change < 5:
                                    # Similar performance (< 5% difference)
                                    st.warning(
                                        f"🟡 **{version_1}** and **{version_2}** have similar performance (**{abs_pct_change:.1f}%** difference)"
                                    )
                                elif pct_change < 0:
                                    # v1 is better (lower latency)
                                    st.success(
                                        f"✅ **{version_1}** is **{abs_pct_change:.1f}% better** than **{version_2}** (lower {metric_label.lower()} at max throughput: C={int(v1_concurrency_at_max)})"
                                    )
                                else:
                                    # v1 is worse (higher latency)
                                    st.error(
                                        f"❌ **{version_1}** is **{abs_pct_change:.1f}% worse** than **{version_2}** (higher {metric_label.lower()} at max throughput: C={int(v2_concurrency_at_max)})"
                                    )

                        st.caption(
                            f"💡 Comparison based on {len(common_concurrencies)} common concurrency level(s): {', '.join(map(str, common_concurrencies))}"
                        )
                    else:
                        st.warning(
                            f"⚠️ No common concurrency levels found for {accelerator} (TP={tp})."
                        )

                # Add explanation of calculations (once at the end)
                st.markdown("---")
                with st.expander(
                    "ℹ️ How are these statistics calculated?", expanded=False
                ):
                    st.markdown(f"""
                    **Median Calculation:**
                    - The median is the middle value when all {metric_label.lower()} measurements at common concurrency levels are sorted
                    - More robust than mean as it's not affected by outliers
                    - Calculated across all data points at the common concurrency levels for each accelerator-TP combination

                    **Max/Min Calculation:**
                    - Max: Highest {metric_label.lower()} value observed across all common concurrency levels
                    - Min: Lowest {metric_label.lower()} value observed across all common concurrency levels
                    - The concurrency level (C=X) shows where this extreme value occurred

                    **Percentage Difference Calculation:**
                    - **Mean Change**: Calculated by taking the percentage change at each common concurrency level, then taking the mean (average)
                      - Shows the average performance difference across all concurrency levels
                      - Can be affected by outliers (extreme values)
                      - Formula: `mean([(v1 - v2) / v2 × 100 for each concurrency level])`
                    - **Median Change**: Calculated by taking the percentage change at each common concurrency level, then taking the median
                      - Shows the typical performance difference across all concurrency levels
                      - More robust to outliers - better represents typical performance
                      - Formula: `median([(v1 - v2) / v2 × 100 for each concurrency level])`
                    - **Peak Change**:
                      - **For Throughput**: `((Version 1 Max - Version 2 Max) / Version 2 Max) × 100`
                        - Compares maximum throughput values (best = highest performance)
                        - Higher is better
                      - **For Latency (TTFT/ITL)**: `((Version 1 Latency @ Max Throughput - Version 2 Latency @ Max Throughput) / Version 2 Latency @ Max Throughput) × 100`
                        - Compares latency values at the concurrency where max throughput occurs for each version
                        - This shows latency characteristics at peak performance
                        - Lower is better
                    - **Note**: Each accelerator-TP combination is compared independently
                    """)


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
        if MLPERF_AVAILABLE:
            view_options = ["RHAIIS Dashboard", "MLPerf Dashboard"]
        else:
            view_options = ["RHAIIS Dashboard"]

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
    st.warning(
        "⚠️ **CONFIDENTIAL**: Any data displayed here is only for internal use. If in doubt, please contact Ashish Kamra at #forum-psap."
    )


initialize_streamlit_config()
initialize_session_state()

# Check URL for view parameter and set session state accordingly
# This allows the view selection to persist across page refreshes
if "view" in st.query_params:
    view_from_url = st.query_params["view"]
    if view_from_url in ["RHAIIS Dashboard", "MLPerf Dashboard"]:
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
    st.session_state.version_comparison_expanded = False
    st.session_state.model_comparison_expanded = False
    st.session_state.runtime_configs_expanded = False

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
                "RHAIIS-3.2.2",
                "RHAIIS-3.2.3",
                "vLLM-0.10.1.1",
                "vLLM-0.11.0",
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
                "RHAIIS-3.2.2",
                "RHAIIS-3.2.3",
                "vLLM-0.10.1.1",
                "vLLM-0.11.0",
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

    with filter_col1:
        # Accelerators filter - filtered by currently selected profile
        temp_df = df.copy()

        # Determine what the current/default profile is by checking session state
        current_profile = st.session_state.get(
            f"profile_filter_{st.session_state.filter_change_key}", None
        )

        # If no profile selected yet, determine the default that will be selected
        if not current_profile:
            available_profiles = sorted(df["profile"].unique().tolist())

            # Default to Profile B (512/2k) when clearing or as fallback
            default_profile = "Profile B: Variable Workload (512/2k)"

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
            "Select Accelerator(s)",
            accelerators,
            default=preserved_selections,
            key=prev_accel_key,
        )
        st.caption("💡 See dropdown for more available models, versions and TP sizes.")

    with filter_col2:
        # Versions filter - filtered by selected accelerators AND currently selected profile
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]

        # Determine what the current/default profile is by checking session state
        current_profile = st.session_state.get(
            f"profile_filter_{st.session_state.filter_change_key}", None
        )

        # If no profile selected yet, determine the default that will be selected
        if not current_profile:
            # Get available profiles to determine default
            profile_temp_df = df.copy()
            if selected_accelerators:
                profile_temp_df = profile_temp_df[
                    profile_temp_df["accelerator"].isin(selected_accelerators)
                ]

            available_profiles = (
                sorted(profile_temp_df["profile"].unique().tolist())
                if not profile_temp_df.empty
                else []
            )

            # Default to Profile B (512/2k) when clearing or as fallback
            default_profile = "Profile B: Variable Workload (512/2k)"

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

        # Filter versions by the current/default profile
        if current_profile:
            temp_df = temp_df[temp_df["profile"] == current_profile]

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
            "Select Version(s)",
            versions,
            default=preserved_selections,
            key=prev_versions_key,
        )
        st.caption(
            "💡 **See Filters Help button to see all valid filter combinations.**"
        )

    with filter_col3:
        # Models filter - filtered by selected accelerators, versions, AND currently selected profile
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]

        # Determine what the current/default profile is by checking session state
        # If no profile in session state yet, determine what the default will be
        current_profile = st.session_state.get(
            f"profile_filter_{st.session_state.filter_change_key}", None
        )

        # If no profile selected yet, determine the default that will be selected
        if not current_profile:
            # Get available profiles for accelerators/versions to determine default
            profile_temp_df = df.copy()
            if selected_accelerators:
                profile_temp_df = profile_temp_df[
                    profile_temp_df["accelerator"].isin(selected_accelerators)
                ]
            if selected_versions:
                profile_temp_df = profile_temp_df[
                    profile_temp_df["version"].isin(selected_versions)
                ]

            available_profiles = (
                sorted(profile_temp_df["profile"].unique().tolist())
                if not profile_temp_df.empty
                else []
            )

            # Use the same logic as the profile filter to determine default
            # Default to Profile B (512/2k) when clearing or as fallback
            default_profile = "Profile B: Variable Workload (512/2k)"

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

        # Filter models by the current/default profile
        if current_profile:
            temp_df = temp_df[temp_df["profile"] == current_profile]

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
            "Select Model(s)",
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

    # Add negative margin spacer to reduce gap between filter rows
    st.markdown('<div style="margin-top: -2rem;"></div>', unsafe_allow_html=True)

    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col4:
        # ISL/OSL Profile filter - filtered by selected accelerators, versions, and models
        temp_df = df.copy()
        if selected_accelerators:
            temp_df = temp_df[temp_df["accelerator"].isin(selected_accelerators)]
        if selected_versions:
            temp_df = temp_df[temp_df["version"].isin(selected_versions)]
        if selected_models:
            temp_df = temp_df[temp_df["model"].isin(selected_models)]

        profiles = (
            sorted(temp_df["profile"].unique().tolist()) if not temp_df.empty else []
        )

        # Default to Profile B (512/2k) when clearing or as fallback
        default_profile = "Profile B: Variable Workload (512/2k)"

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

        if st.session_state.get("clear_all_filters", False) or st.session_state.get(
            "filters_were_cleared", False
        ):
            tp_default = []
        elif select_all_checked:
            # If "Select All Models" is checked, also select all TP sizes
            tp_default = tp_sizes
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
            key=f"tp_filter_{st.session_state.filter_change_key}_all_{select_all_checked}",
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
                # Close all expanders when resetting filters
                st.session_state.performance_plots_expanded = False
                st.session_state.model_comparison_expanded = False
                st.session_state.version_comparison_expanded = False
                st.session_state.runtime_configs_expanded = False
                # Reset filter state tracking
                if "previous_filter_state" in st.session_state:
                    del st.session_state.previous_filter_state
                st.rerun()

        with btn_col2:
            if st.button("🧹 Clear Filters", help="Clear all filter selections"):
                st.session_state.clear_all_filters = True
                st.session_state.filters_were_cleared = True
                st.session_state.filter_change_key += 1
                # Close all expanders when clearing filters
                st.session_state.performance_plots_expanded = False
                st.session_state.model_comparison_expanded = False
                st.session_state.version_comparison_expanded = False
                st.session_state.runtime_configs_expanded = False
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
        st.session_state.version_comparison_expanded = False
        st.session_state.runtime_configs_expanded = False

    # Store current filter state for next comparison
    st.session_state.previous_filter_state = current_filter_state

    if not filtered_df.empty:
        accelerator_color_map = {
            "H200": "#1f77b4",
            "MI300X": "#ff7f0e",
            "TPU": "#2ca02c",
        }

        render_performance_plots_section(filtered_df)
        render_pareto_plots_section()

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

        render_model_performance_comparison_section(filtered_df, accelerator_color_map)
        render_version_comparison_section(filtered_df)
        render_regression_analysis_section(filtered_df, analyze_performance_changes)
        render_cost_analysis_section(filtered_df, accelerator_color_map)
        # render_performance_rankings_section(filtered_df)
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
