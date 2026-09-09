"""Tests for the dashboard's version, label, and run taxonomy."""

import json

import pandas as pd

from dashboard_taxonomy import (
    DEFAULT_LABEL,
    UNLABELED,
    add_trace_metadata,
    compact_series_label,
    decode_query_mapping,
    derive_product_family,
    deterministic_color_map,
    encode_query_mapping,
    filter_taxonomy,
    normalize_label,
    normalize_taxonomy_columns,
    parse_filter_values,
    split_legacy_version,
    sync_selected_options,
    taxonomy_query_params,
)
from manual_runs.scripts.vllm.import_manual_runs_json_v2 import parse_guidellm_json


def test_legacy_rows_get_default_label_and_family_columns():
    result = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": ["vLLM-0.28.0", "RHAIIS-3.5-GA", "other-1"],
                "uuid": ["run-1", None, "run-3"],
            }
        )
    )

    assert result["label"].tolist() == [DEFAULT_LABEL, DEFAULT_LABEL, DEFAULT_LABEL]
    assert result["product_family"].tolist() == ["vLLM", "RHAIIS", "Other"]
    assert result["uuid"].tolist() == ["run-1", "", "run-3"]


def test_normalization_drops_submitter_columns():
    result = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": ["vLLM-0.28.0"],
                "username": ["test-owner"],
                "uuid": ["run-1"],
            }
        )
    )

    assert not any(
        column in result.columns
        for column in ("user", "username", "submitted_by", "furnace_user")
    )


def test_label_and_family_values_are_normalized():
    assert normalize_label("  pcon-mnbt ") == "pcon-mnbt"
    assert normalize_label("") == UNLABELED
    assert derive_product_family("sglang-0.5.2") == "sglang"
    assert derive_product_family("AIC-0.10.0-vLLM-0.24.0") == "vLLM"
    assert derive_product_family("unknown-1") == "Other"


def test_known_composite_versions_become_release_and_label():
    assert split_legacy_version("vLLM-0.24.0-nn-d1") == ("vLLM-0.24.0", "nn-d1")
    assert split_legacy_version("RHAIIS-3.5-GA-FIBF16") == (
        "RHAIIS-3.5-GA",
        "FIBF16",
    )
    assert split_legacy_version("vLLM-0.24.0") == ("vLLM-0.24.0", DEFAULT_LABEL)
    assert split_legacy_version("vLLM-0.24-nn-d-exc1") == (
        "vLLM-0.24",
        "nn-d-exc1",
    )
    assert split_legacy_version("RHAIIS-3.4GA-fubon-no-caching") == (
        "RHAIIS-3.4GA",
        "fubon-no-caching",
    )
    assert split_legacy_version("AIC-0.10.0-vLLM-0.24.0-pcon") == (
        "AIC-0.10.0-vLLM-0.24.0",
        "pcon",
    )
    assert split_legacy_version("AIC-0.10.0-vLLM-0.24.0") == (
        "AIC-0.10.0-vLLM-0.24.0",
        DEFAULT_LABEL,
    )


def test_normalization_preserves_composite_source_version():
    result = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": ["vLLM-0.24.0-nn-d1", "RHAIIS-3.5-GA"],
                "uuid": ["run-1", "run-2"],
            }
        )
    )

    assert result["version"].tolist() == ["vLLM-0.24.0", "RHAIIS-3.5-GA"]
    assert result["label"].tolist() == ["nn-d1", DEFAULT_LABEL]
    assert result["legacy_version"].tolist() == ["vLLM-0.24.0-nn-d1", ""]


def test_cascading_taxonomy_filter_keeps_all_concurrency_rows():
    data = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": ["vLLM-0.28.0"] * 3,
                "label": ["pcon"] * 3,
                "uuid": ["run-1"] * 2 + ["run-2"],
                "intended concurrency": [1, 50, 1],
            }
        )
    )

    result = filter_taxonomy(
        data,
        families=["vLLM"],
        versions=["vLLM-0.28.0"],
        labels=["pcon"],
        uuids=["run-1"],
    )
    assert result["intended concurrency"].tolist() == [1, 50]


def test_url_taxonomy_filters_round_trip():
    params = taxonomy_query_params(["vLLM"], ["pcon-mnbt"], ["run-1"])

    assert params == {
        "families": "vLLM",
        "labels": "pcon-mnbt",
        "uuids": "run-1",
    }
    assert parse_filter_values(params["families"], ["vLLM", "RHAIIS"]) == ["vLLM"]
    assert parse_filter_values(params["labels"], ["pcon-mnbt"]) == ["pcon-mnbt"]
    assert parse_filter_values(params["uuids"], ["run-1"]) == ["run-1"]
    assert parse_filter_values("stale", ["run-1"]) == []


def test_appearance_query_mapping_round_trip():
    mapping = {"series | pcon": "#0072B2", "run-2": "triangle-up"}

    encoded = encode_query_mapping(mapping)

    assert decode_query_mapping(encoded) == mapping
    assert decode_query_mapping(encoded, ["triangle-up"]) == {"run-2": "triangle-up"}
    assert decode_query_mapping("not-json") == {}


def test_select_all_state_tracks_new_options_without_overriding_manual_selection():
    assert sync_selected_options(
        ["model-a"], ["model-a", "model-b"], select_all=True
    ) == ["model-a", "model-b"]
    assert sync_selected_options(
        ["tp-1"], ["tp-1", "tp-2"], select_all=False
    ) == ["tp-1"]


def test_duplicate_runs_are_dotted_and_single_runs_are_solid():
    data = pd.DataFrame(
        {
            "run_identifier": ["base", "base", "base", "single"],
            "uuid": ["run-1", "run-1", "run-2", "run-3"],
        }
    )

    result = add_trace_metadata(data)
    assert set(result.loc[result["run_identifier"] == "base", "line_style"]) == {"dot"}
    assert set(result.loc[result["run_identifier"] == "single", "line_style"]) == {
        "solid"
    }
    assert any("UUID=run-1" in label for label in result["trace_label"])
    assert any("UUID=run-2" in label for label in result["trace_label"])
    assert set(result.loc[result["run_identifier"] == "base", "marker_symbol"]) == {
        "circle",
        "triangle-up",
    }
    assert result.loc[result["run_identifier"] == "single", "marker_symbol"].iloc[0] == "circle"
    assert (
        result.loc[result["run_identifier"] == "single", "trace_label"].iloc[0]
        == "single"
    )


def test_compact_series_label_keeps_config_readable():
    label = compact_series_label(
        {
            "accelerator": "B200",
            "model_short": "Llama-3.1-8B",
            "version": "RHAIIS-3.5-GA",
            "label": "pcon-mnbt",
            "TP": 2.0,
            "prefix_caching": "yes",
        }
    )

    assert label == "B200 | Llama-3.1-8B | RHAIIS-3.5-GA · pcon-mnbt | TP=2 | PC=yes"


def test_color_assignment_is_deterministic():
    colors = deterministic_color_map(["b", "a"])
    assert colors == deterministic_color_map(["a", "b"])
    assert colors["a"] != colors["b"]


def test_manual_importer_emits_optional_label(tmp_path):
    benchmark_json = {
        "metadata": {"guidellm_version": "0.6.0"},
        "args": {"data": [json.dumps({"prompt_tokens": 1000, "output_tokens": 1000})]},
        "benchmarks": [
            {
                "config": {"run_id": "run-1", "strategy": {"streams": 1}},
                "scheduler_metrics": {"start_time": 1, "end_time": 2},
                "metrics": {},
            }
        ],
    }
    json_path = tmp_path / "benchmark.json"
    json_path.write_text(json.dumps(benchmark_json))

    result = parse_guidellm_json(
        json_path,
        "B200",
        "test-model",
        "vLLM-0.28.0",
        2,
        "tensor-parallel-size: 2",
        "image:tag",
        "0.6.0",
        label="pcon-mnbt",
    )

    assert result["label"].tolist() == ["pcon-mnbt"]
    assert "user" not in result.columns
