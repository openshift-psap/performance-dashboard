"""Regression threshold policy shared by both dashboards."""

# P50/central metrics: throughput and the E2E latency median.
P50_THRESHOLD_PCT = 5.0

# P95 latency metrics: TTFT P95 and ITL P95.
P95_THRESHOLD_PCT = 10.0

P95_COLUMNS = frozenset({"ttft_p95", "itl_p95"})


def pct_threshold_for_column(column):
    """Return the percentage dead-zone for a metric column."""
    return P95_THRESHOLD_PCT if column in P95_COLUMNS else P50_THRESHOLD_PCT


def is_similar_change(pct_diff, column):
    """True when the percentage change is inside the column's dead-zone."""
    return abs(pct_diff) <= pct_threshold_for_column(column)
