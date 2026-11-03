"""Build links to Grafana and MLflow for benchmark result rows."""

import math

import pandas as pd

GRAFANA_DASHBOARDS = {
    "H200_OLD": {
        "dashboard_id": "6475e6106c33fe",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-8xh200-2",
    },
    "H200_NEW": {
        "dashboard_id": "7a3b910e7e827c",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-rhaiis-h200",
    },
    "H200_HERA": {
        "dashboard_id": "cd77e3aa2b40e0",
        "dashboard_name": "vllm-2b-dcgm-metrics-rhaiis-ibm-dc-h200",
    },
    "H200_HERA2": {
        "dashboard_id": "psap-hera-dashboard",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-hera-h200",
    },
    "MI300X": {
        "dashboard_id": "amd-ods-az-amd-01",
        "dashboard_name": "vllm-2b-rocm-gpu-metrics-ods-az-amd-01",
    },
    "MI355X": {
        "dashboard_id": "mi355x-rocm-gpu",
        "dashboard_name": "vllm-2b-rocm-gpu-metrics-mi355x",
        "extra_params": "&var-cluster_name=sno-mi355x",
    },
    "B200": {
        "dashboard_id": "psap-b200-mlperf",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-b200-mlperf",
    },
    "H200_ZEUS2": {
        "dashboard_id": "d35f19c8f56250",
        "dashboard_name": "vllm-2b-dcgm-metrics-psap-zeus-syd",
    },
    "H200_ATHENA": {
        "dashboard_id": "df28i54z7okcgf",
        "dashboard_name": "vllm-2b-dcgm-complete-metrics-combined",
        "extra_params": "&var-cluster_name=psap-fire-athena",
    },
    "H200_JANUS": {
        "dashboard_id": "df28i54z7okcgf",
        "dashboard_name": "vllm-2b-dcgm-complete-metrics-combined",
        "extra_params": "&var-cluster_name=psap-de-h200-cluster",
    },
    "B200_PIRATE": {
        "dashboard_id": "df28i54z7okcgf",
        "dashboard_name": "vllm-2b-dcgm-complete-metrics-combined",
        "extra_params": "&var-cluster_name=psap-b200-nm",
    },
}

_SGLANG_H200_DASHBOARD = {
    "dashboard_id": "sglang-dcgm-metrics-psap-rhaiis-h200",
    "dashboard_name": "sglang-2b-dcgm-metrics-psap-rhaiis-h200",
    "extra_params": "&var-cluster_name=$__all&var-model_name=$__all",
}
_SGLANG_ATHENA_DASHBOARD = {
    "dashboard_id": "sglang-dcgm-metrics-psap-rhaiis-h200",
    "dashboard_name": "sglang-2b-dcgm-metrics-psap-rhaiis-h200",
    "extra_params": "&var-cluster_name=psap-fire-athena&var-model_name=$__all",
}
_SGLANG_JANUS_DASHBOARD = {
    "dashboard_id": "sglang-dcgm-metrics-psap-rhaiis-h200",
    "dashboard_name": "sglang-2b-dcgm-metrics-psap-rhaiis-h200",
    "extra_params": "&var-cluster_name=psap-de-h200-cluster&var-model_name=$__all",
}
SGLANG_GRAFANA_DASHBOARDS = {
    "H200": _SGLANG_H200_DASHBOARD,
    "H200_HERA2": _SGLANG_H200_DASHBOARD,
    "H200_ZEUS2": _SGLANG_H200_DASHBOARD,
    "H200_ATHENA": _SGLANG_ATHENA_DASHBOARD,
    "H200_JANUS": _SGLANG_JANUS_DASHBOARD,
}

# Jan 1, 2026 00:00:00 UTC in milliseconds.
H200_DASHBOARD_CUTOFF_MS = 1767225600000
GRAFANA_BASE_URL = "https://grafana-psap-obs.apps.ocp4.intlab.redhat.com"


def _has_value(value):
    """Return whether a scalar contains a usable value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return False


def _as_int(value):
    """Convert a numeric value to int, returning None for invalid values."""
    try:
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        return int(numeric_value)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_hera_run(run_name):
    """Check if a run belongs to the H200 Hera cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("H200-HERA-") or upper.startswith("H200_HERA-")


def _is_hera2_run(run_name):
    """Check if a run belongs to the H200 Hera2 cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("H200-HERA2-") or upper.startswith("H200_HERA2-")


def _is_zeus2_run(run_name):
    """Check if a run belongs to the H200 ZEUS2 cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("H200-ZEUS2-") or upper.startswith("H200_ZEUS2-")


def _is_athena_run(run_name):
    """Check if a run belongs to the Athena cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("ATHENA-") or upper.startswith("ATHENA_")


def _is_janus_run(run_name):
    """Check if a run belongs to the Janus cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("JANUS-") or upper.startswith("JANUS_")


def _is_b200_nm_run(run_name):
    """Check if a run belongs to the B200 NM cluster."""
    if not isinstance(run_name, str):
        return False
    upper = run_name.upper()
    return upper.startswith("B200-B200-") or upper.startswith("B200_B200-")


def create_grafana_link(row):
    """Create the same cluster- and date-aware Grafana link used by Filtered Data."""
    start_time = row.get("guidellm_start_time_ms")
    end_time = row.get("guidellm_end_time_ms")
    uuid = row.get("uuid")
    accelerator = row.get("accelerator", "")
    run_name = row.get("run", "")
    version = row.get("version", "")

    if not all(_has_value(value) for value in (start_time, end_time, uuid)):
        return None

    start_ms = _as_int(start_time)
    end_ms = _as_int(end_time)
    if start_ms is None or end_ms is None:
        return None

    if _is_athena_run(run_name):
        dashboard_key = "H200_ATHENA"
    elif _is_janus_run(run_name):
        dashboard_key = "H200_JANUS"
    elif _is_hera2_run(run_name):
        dashboard_key = "H200_HERA2"
    elif _is_hera_run(run_name):
        dashboard_key = "H200_HERA"
    elif _is_zeus2_run(run_name):
        dashboard_key = "H200_ZEUS2"
    elif _is_b200_nm_run(run_name):
        dashboard_key = "B200_PIRATE"
    elif accelerator == "H200":
        dashboard_key = (
            "H200_NEW" if start_ms >= H200_DASHBOARD_CUTOFF_MS else "H200_OLD"
        )
    elif accelerator in GRAFANA_DASHBOARDS:
        dashboard_key = accelerator
    else:
        return None

    is_sglang = isinstance(version, str) and version.lower().startswith("sglang")
    if is_sglang:
        sglang_key = dashboard_key
        if sglang_key not in SGLANG_GRAFANA_DASHBOARDS and sglang_key.startswith(
            "H200"
        ):
            sglang_key = "H200"
        dashboard_config = SGLANG_GRAFANA_DASHBOARDS.get(sglang_key)
        if dashboard_config is None:
            return None
    else:
        dashboard_config = GRAFANA_DASHBOARDS[dashboard_key]

    extra_params = dashboard_config.get("extra_params", "")
    return (
        f"{GRAFANA_BASE_URL}/d/{dashboard_config['dashboard_id']}/"
        f"{dashboard_config['dashboard_name']}"
        f"?orgId=1&from={start_ms}&to={end_ms}"
        f"&timezone=browser&var-deployment_uuid={uuid}"
        f"&var-deployment_pod_name=$__all&var-rate_interval=1m{extra_params}"
    )


def create_mlflow_link(row, base_url, workspace):
    """Create a link to a tracked MLflow run when both IDs are available."""
    run_id = row.get("mlflow_run_id")
    experiment_id = row.get("mlflow_experiment_id")
    if not _has_value(run_id) or not _has_value(experiment_id):
        return None

    exp_id = _as_int(experiment_id)
    if exp_id is None or not base_url:
        return None
    return (
        f"{base_url}/#/experiments/{exp_id}"
        f"/runs/{run_id}/artifacts?workspace={workspace}"
    )


def add_observability_links(dataframe, mlflow_base_url, mlflow_workspace):
    """Return a copy with Grafana and MLflow link columns added."""
    linked_dataframe = dataframe.copy()
    linked_dataframe["grafana_metrics_link"] = linked_dataframe.apply(
        create_grafana_link, axis=1
    )
    linked_dataframe["mlflow_link"] = linked_dataframe.apply(
        lambda row: create_mlflow_link(row, mlflow_base_url, mlflow_workspace), axis=1
    )
    return linked_dataframe


def build_compare_filtered_rows(
    version_1_rows,
    version_2_rows,
    comparison_pairs,
    slice_by_config,
    selected_concurrencies,
    label_1=None,
    label_2=None,
):
    """Collect only latest source rows represented by the comparison pairs."""
    filtered_rows = []
    sources = (
        (version_1_rows, label_1),
        (version_2_rows, label_2),
    )

    for model, config_1, config_2 in comparison_pairs:
        for (source_df, label), config in zip(sources, (config_1, config_2)):
            rows = slice_by_config(source_df, model, config)
            if selected_concurrencies:
                rows = rows[rows["intended concurrency"].isin(selected_concurrencies)]
            if rows.empty:
                continue
            rows = rows.copy()
            if "label" in source_df.columns and label is not None:
                rows["label"] = label
            filtered_rows.append(rows)

    if not filtered_rows:
        return version_1_rows.iloc[0:0].copy()
    return pd.concat(filtered_rows, ignore_index=True)
