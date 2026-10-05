"""Tests for the dashboard's version, label, and run taxonomy."""

import json

import pandas as pd

from dashboard_taxonomy import (
    DEFAULT_LABEL,
    UNLABELED,
    add_trace_metadata,
    compact_series_label,
    decode_query_mapping,
    decode_version_label_pairs,
    deterministic_color_map,
    display_label,
    encode_query_mapping,
    encode_version_label_pairs,
    filter_taxonomy,
    is_valid_hex_color,
    normalize_label,
    normalize_taxonomy_columns,
    parse_filter_values,
    split_legacy_version,
    sync_selected_options,
    taxonomy_query_params,
    uses_legacy_methodology,
    version_label_pair_mask,
)
from manual_runs.scripts.vllm.import_manual_runs_json_v2 import parse_guidellm_json


def test_legacy_rows_get_default_label_columns():
    result = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": ["vLLM-0.28.0", "RHAIIS-3.5-GA", "other-1"],
                "uuid": ["run-1", None, "run-3"],
            }
        )
    )

    assert result["label"].tolist() == [DEFAULT_LABEL, DEFAULT_LABEL, DEFAULT_LABEL]
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


def test_label_values_are_normalized():
    assert normalize_label("  pcon-mnbt ") == "pcon-mnbt"
    assert normalize_label("") == UNLABELED


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


def test_methodology_release_thresholds():
    assert uses_legacy_methodology("vLLM-0.24.0")
    assert uses_legacy_methodology("AIC-0.10.0-vLLM-0.24.0-pcon")
    assert uses_legacy_methodology("RHAIIS-3.5-GA")
    assert not uses_legacy_methodology("vLLM-0.26.0")
    assert not uses_legacy_methodology("vLLM-0.28.0")
    assert not uses_legacy_methodology("RHAIIS-3.6-GA")
    assert not uses_legacy_methodology("sglang-0.5.5")


def test_release_selection_defaults_to_plain_runs_until_label_is_selected():
    data = normalize_taxonomy_columns(
        pd.DataFrame(
            {
                "version": [
                    "vLLM-0.24.0",
                    "vLLM-0.24.0-pcon",
                    "vLLM-0.24.0-pcoff",
                ],
                "uuid": ["default", "pcon", "pcoff"],
            }
        )
    )

    release_rows = data[data["version"] == "vLLM-0.24.0"]
    assert release_rows.loc[
        release_rows["label"] == DEFAULT_LABEL, "uuid"
    ].tolist() == ["default"]
    assert version_label_pair_mask(
        release_rows, [("vLLM-0.24.0", "pcon")]
    ).tolist() == [False, True, False]
    assert version_label_pair_mask(
        release_rows, [("vLLM-0.24.0", "pcon")], include_default=True
    ).tolist() == [True, True, False]


def test_unlabeled_display_uses_dash():
    assert display_label(DEFAULT_LABEL) == "—"
    assert display_label(UNLABELED) == "—"
    assert display_label("pcon") == "pcon"


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
        versions=["vLLM-0.28.0"],
        labels=["pcon"],
        uuids=["run-1"],
    )
    assert result["intended concurrency"].tolist() == [1, 50]


def test_url_taxonomy_filters_round_trip():
    params = taxonomy_query_params(["pcon-mnbt"], ["run-1"])

    assert params == {
        "labels": "pcon-mnbt",
        "uuids": "run-1",
    }
    assert parse_filter_values(params["labels"], ["pcon-mnbt"]) == ["pcon-mnbt"]
    assert parse_filter_values(params["uuids"], ["run-1"]) == ["run-1"]
    assert parse_filter_values("stale", ["run-1"]) == []


def test_version_label_pairs_preserve_release_scope():
    pairs = [("vLLM-0.23.0", "pcon"), ("vLLM-0.24.0", "pcon")]
    encoded = encode_version_label_pairs(pairs)

    assert decode_version_label_pairs(encoded) == pairs
    assert (
        decode_version_label_pairs(
            taxonomy_query_params(version_label_pairs=pairs)["version_labels"]
        )
        == pairs
    )

    data = pd.DataFrame(
        {
            "version": ["vLLM-0.23.0", "vLLM-0.24.0", "vLLM-0.24.0"],
            "label": ["pcon", "pcon", "pcoff"],
        }
    )
    assert version_label_pair_mask(data, [("vLLM-0.23.0", "pcon")]).tolist() == [
        True,
        False,
        False,
    ]


def test_appearance_query_mapping_round_trip():
    mapping = {"series | pcon": "#0072B2", "run-2": "triangle-up"}

    encoded = encode_query_mapping(mapping)

    assert decode_query_mapping(encoded) == mapping
    assert decode_query_mapping(encoded, ["triangle-up"]) == {"run-2": "triangle-up"}
    assert decode_query_mapping("not-json") == {}


def test_hex_color_validation_rejects_malformed_url_values():
    assert is_valid_hex_color("#abc")
    assert is_valid_hex_color("#A1b2C3")
    assert is_valid_hex_color("#A1b2C3d4")
    assert not is_valid_hex_color("#ggg")
    assert not is_valid_hex_color("#12345")
    assert not is_valid_hex_color("rgb(0, 0, 0)")


def test_select_all_state_tracks_new_options_without_overriding_manual_selection():
    assert sync_selected_options(
        ["model-a"], ["model-a", "model-b"], select_all=True
    ) == ["model-a", "model-b"]
    assert sync_selected_options(["tp-1"], ["tp-1", "tp-2"], select_all=False) == [
        "tp-1"
    ]


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
        "circle"
    }
    assert (
        result.loc[result["run_identifier"] == "single", "marker_symbol"].iloc[0]
        == "circle"
    )
    assert (
        result.loc[result["run_identifier"] == "single", "trace_label"].iloc[0]
        == "single"
    )


def test_runtime_identical_repeats_are_faded_but_similar_runs_are_not():
    data = pd.DataFrame(
        {
            "run_identifier": ["exact", "exact", "similar", "similar"],
            "uuid": ["run-1", "run-2", "run-3", "run-4"],
            "runtime_args": [
                "tensor-parallel-size: 2; max-model-len: 8192",
                "max-model-len: 8192; tensor-parallel-size: 2",
                "max-model-len: 8192",
                "max-model-len: 16384",
            ],
        }
    )

    result = add_trace_metadata(data)

    exact = result[result["run_identifier"] == "exact"]
    similar = result[result["run_identifier"] == "similar"]
    assert exact["line_style"].tolist() == ["dot", "dot"]
    assert exact["line_opacity"].tolist() == [1.0, 0.55]
    assert set(exact["marker_symbol"]) == {"circle"}
    assert similar["line_style"].tolist() == ["dot", "dot"]
    assert similar["line_opacity"].tolist() == [1.0, 1.0]
    assert set(similar["marker_symbol"]) == {"circle"}


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
    assert "default" not in compact_series_label(
        {
            "accelerator": "B200",
            "model_short": "Llama-3.1-8B",
            "version": "RHAIIS-3.5-GA",
            "label": DEFAULT_LABEL,
            "TP": 2,
        }
    )


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
