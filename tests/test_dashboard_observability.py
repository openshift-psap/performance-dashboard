"""Tests for links and source-row selection in dashboard filtered tables."""

import pandas as pd

from dashboard_observability import (
    add_observability_links,
    build_compare_filtered_rows,
    create_grafana_link,
    create_mlflow_link,
)


def test_grafana_link_uses_cluster_specific_h200_dashboard():
    row = pd.Series(
        {
            "accelerator": "H200",
            "run": "H200-ZEUS2-model-tp1",
            "version": "RHAIIS-3.6",
            "guidellm_start_time_ms": 1767000000000,
            "guidellm_end_time_ms": 1767000005000,
            "uuid": "run-uuid",
        }
    )

    link = create_grafana_link(row)

    assert "/d/d35f19c8f56250/vllm-2b-dcgm-metrics-psap-zeus-syd" in link
    assert "from=1767000000000&to=1767000005000" in link
    assert "var-deployment_uuid=run-uuid" in link


def test_grafana_link_uses_accelerator_dashboard():
    row = pd.Series(
        {
            "accelerator": "B200",
            "run": "B200-model-tp1",
            "version": "vLLM-0.28.0",
            "guidellm_start_time_ms": 1767000000000,
            "guidellm_end_time_ms": 1767000005000,
            "uuid": "run-uuid",
        }
    )

    link = create_grafana_link(row)

    assert "/d/psap-b200-mlperf/vllm-2b-dcgm-metrics-psap-b200-mlperf" in link


def test_grafana_link_uses_mi355x_dashboard():
    row = pd.Series(
        {
            "accelerator": "MI355X",
            "run": "MI355X-model-tp1",
            "version": "RHAIIS-3.6",
            "guidellm_start_time_ms": 1767000000000,
            "guidellm_end_time_ms": 1767000005000,
            "uuid": "mi355x-run-uuid",
        }
    )

    link = create_grafana_link(row)

    assert "/d/mi355x-rocm-gpu/vllm-2b-rocm-gpu-metrics-mi355x" in link
    assert "var-deployment_uuid=mi355x-run-uuid" in link
    assert "var-cluster_name=sno-mi355x" in link


def test_mlflow_link_uses_experiment_and_run_ids():
    row = pd.Series({"mlflow_run_id": "abc123", "mlflow_experiment_id": "42"})

    link = create_mlflow_link(row, "https://mlflow.example", "forge-rhaiis")

    assert (
        link == "https://mlflow.example/#/experiments/42/runs/abc123/artifacts"
        "?workspace=forge-rhaiis"
    )


def test_add_observability_links_does_not_mutate_input():
    source = pd.DataFrame([{"mlflow_run_id": "abc123", "mlflow_experiment_id": 42}])

    linked = add_observability_links(source, "https://mlflow.example", "ws")

    assert "grafana_metrics_link" not in source
    assert "mlflow_link" not in source
    assert linked.loc[0, "mlflow_link"].endswith("/runs/abc123/artifacts?workspace=ws")


def test_compare_filtered_rows_only_include_paired_models_and_selected_concurrency():
    v1 = pd.DataFrame(
        [
            {"model": "model/a", "TP": 1, "intended concurrency": 1},
            {"model": "model/a", "TP": 1, "intended concurrency": 2},
            {"model": "model/a", "TP": 2, "intended concurrency": 2},
            {
                "model": "model/unpaired",
                "TP": 1,
                "intended concurrency": 2,
            },
        ]
    )
    v2 = pd.DataFrame(
        [
            {"model": "model/a", "TP": 1, "intended concurrency": 2},
            {"model": "model/a", "TP": 3, "intended concurrency": 2},
        ]
    )

    def slice_by_config(frame, model, config):
        return frame[(frame["model"] == model) & (frame["TP"] == config[1])]

    result = build_compare_filtered_rows(
        v1,
        v2,
        [("model/a", ("TP", 1), ("TP", 1))],
        slice_by_config,
        {2},
    )

    assert set(result["model"]) == {"model/a"}
    assert set(result["TP"]) == {1}
    assert set(result["intended concurrency"]) == {2}
