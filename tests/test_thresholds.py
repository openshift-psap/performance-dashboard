"""Tests for the shared regression threshold policy."""

from thresholds import (
    P50_THRESHOLD_PCT,
    P95_THRESHOLD_PCT,
    is_similar_change,
    p50_only_regression,
    pct_threshold_for_column,
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


def test_p50_only_marker():
    assert p50_only_regression(5.0, 6.0)
    assert p50_only_regression(-15.0, 29.5)
    assert not p50_only_regression(11.0, 6.0)  # P95 already caused the red
    assert not p50_only_regression(5.0, 4.9)
