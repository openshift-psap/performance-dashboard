"""Tests for the regression threshold policy."""

import ast
from pathlib import Path

from llmd_dashboard import (
    P50_THRESHOLD_PCT,
    P95_THRESHOLD_PCT,
    is_similar_change,
    p50_regressed,
    pct_threshold_for_column,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _constant_value(path, name):
    tree = ast.parse((_REPO_ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_dashboard_threshold_copies_match():
    for name in ("P50_THRESHOLD_PCT", "P95_THRESHOLD_PCT"):
        assert _constant_value("dashboard.py", name) == _constant_value(
            "llmd_dashboard.py", name
        )


def test_p50_columns_use_five_percent():
    for column in ("output_tok/sec", "total_tok/sec", "request_latency_median"):
        assert pct_threshold_for_column(column) == P50_THRESHOLD_PCT == 5.0


def test_p95_columns_use_ten_percent():
    for column in ("ttft_p95", "itl_p95"):
        assert pct_threshold_for_column(column) == P95_THRESHOLD_PCT == 10.0


def test_boundary_is_inclusive():
    assert is_similar_change(5.0, "output_tok/sec")
    assert not is_similar_change(5.01, "output_tok/sec")
    assert is_similar_change(-5.0, "request_latency_median")
    assert not is_similar_change(-5.01, "request_latency_median")
    assert is_similar_change(10.0, "ttft_p95")
    assert not is_similar_change(10.01, "ttft_p95")
    assert is_similar_change(-10.0, "itl_p95")
    assert not is_similar_change(-10.01, "itl_p95")


def test_e2e_latency_has_no_absolute_tolerance():
    # A 6 % E2E change is outside the 5 % band regardless of its absolute delta.
    assert not is_similar_change(6.0, "request_latency_median")
    assert not is_similar_change(-6.0, "request_latency_median")


def test_p50_regressed_marker():
    assert p50_regressed(6.0)
    assert p50_regressed(29.5)
    assert not p50_regressed(4.9)
    assert not p50_regressed(None)
