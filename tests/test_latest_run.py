"""Tests for selecting the latest run in Compare Versions."""

import pandas as pd

from dashboard_taxonomy import latest_run_rows, run_identity


def test_latest_run_rows_keeps_all_rows_from_newest_run():
    data = pd.DataFrame(
        {
            "uuid": ["new", "old", "old", "new"],
            "guidellm_start_time_ms": [2000, 1000, 1000, 2000],
            "row": ["new-a", "old-a", "old-b", "new-b"],
        }
    )

    result = latest_run_rows(data)

    assert result["uuid"].tolist() == ["new", "new"]
    assert result["row"].tolist() == ["new-a", "new-b"]


def test_latest_run_rows_uses_csv_order_when_timestamps_are_missing():
    data = pd.DataFrame(
        {
            "uuid": ["old", "old", "new", "new"],
            "guidellm_start_time_ms": [None, None, None, None],
            "row": ["old-a", "old-b", "new-a", "new-b"],
        }
    )

    result = latest_run_rows(data)

    assert result["row"].tolist() == ["new-a", "new-b"]


def test_latest_run_rows_uses_timestamp_when_run_id_is_missing():
    data = pd.DataFrame(
        {
            "guidellm_start_time_ms": [1000, 3000, 3000, 2000],
            "row": ["old", "new-a", "new-b", "middle"],
        }
    )

    result = latest_run_rows(data)

    assert result["row"].tolist() == ["new-a", "new-b"]


def test_latest_run_rows_keeps_rows_when_run_metadata_is_missing():
    data = pd.DataFrame({"row": ["first", "second"]})

    result = latest_run_rows(data)

    assert result.equals(data)


def test_run_identity_groups_rows_by_uuid_then_timestamp():
    data = pd.DataFrame(
        {
            "uuid": ["run-a", "run-a", "", ""],
            "guidellm_start_time_ms": [1000, 1000, 2000, 3000],
        }
    )

    assert run_identity(data).tolist() == [
        "run-a",
        "run-a",
        "legacy:2000",
        "legacy:3000",
    ]
