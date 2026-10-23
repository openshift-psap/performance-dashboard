"""Run-level statistics used by the Compare Versions dashboard."""

import numpy as np
import pandas as pd

MIN_RUNS = 5
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_BATCH_SIZE = 256
_CONCURRENCY_COLUMN = "intended concurrency"
_SETTING_COLUMNS = (
    "model",
    "accelerator",
    "profile",
    "TP",
    "DP",
    "custom_isl_osl",
    "dataset",
    "spec_decoding",
    "prefix_caching",
    "turns",
    "multiturn_isl_osl",
    "prefix_tokens",
    "prefix_count",
    "runtime_args",
)


def _run_metric_values(data, metric_config, common_concurrencies):
    """Return one valid metric value per UUID after grouping its settings."""
    metric_column = metric_config["column"]
    if (
        "uuid" not in data.columns
        or _CONCURRENCY_COLUMN not in data.columns
        or metric_column not in data.columns
    ):
        return {}, set(), set(), len(data), set()

    uuids = data["uuid"].astype("string").str.strip()
    has_uuid = (
        uuids.notna() & uuids.ne("") & ~uuids.str.lower().isin({"nan", "none", "null"})
    )
    missing_uuid_rows = int((~has_uuid).sum())
    rows = data.loc[has_uuid].copy()
    rows["uuid"] = uuids.loc[has_uuid]
    rows[_CONCURRENCY_COLUMN] = pd.to_numeric(
        rows[_CONCURRENCY_COLUMN], errors="coerce"
    )
    run_ids = set(rows["uuid"].unique())
    rows = rows[rows[_CONCURRENCY_COLUMN].isin(common_concurrencies)]

    setting_columns = [column for column in _SETTING_COLUMNS if column in rows.columns]
    duplicate_key = ["uuid", *setting_columns, _CONCURRENCY_COLUMN]
    duplicate_rows = rows.duplicated(subset=duplicate_key, keep=False)
    duplicate_run_ids = set(rows.loc[duplicate_rows, "uuid"].unique())
    duplicate_concurrencies = set(
        rows.loc[duplicate_rows, _CONCURRENCY_COLUMN].dropna().unique()
    )
    rows = rows[~rows["uuid"].isin(duplicate_run_ids)]

    values_by_run = {}
    invalid_runs = set()
    aggregation = metric_config["aggregation"]
    higher_is_better = metric_config["higher_is_better"]
    group_columns = ["uuid", *setting_columns]
    group_key = group_columns if len(group_columns) > 1 else group_columns[0]
    for group_key_value, setting_rows in rows.groupby(
        group_key, sort=True, dropna=False
    ):
        run_id = (
            group_key_value[0]
            if isinstance(group_key_value, tuple)
            else group_key_value
        )
        run_concurrencies = set(setting_rows[_CONCURRENCY_COLUMN].dropna().unique())
        if not common_concurrencies.issubset(run_concurrencies):
            invalid_runs.add(run_id)
            continue

        metric_values = (
            pd.to_numeric(setting_rows[metric_column], errors="coerce")
            .astype("float64")
            .to_numpy()
        )
        if (
            not len(metric_values)
            or not np.isfinite(metric_values).all()
            or (metric_values <= 0).any()
        ):
            invalid_runs.add(run_id)
            continue

        if aggregation == "geom_mean":
            value = float(np.exp(np.log(metric_values).mean()))
        elif aggregation == "peak":
            value = float(
                np.max(metric_values) if higher_is_better else np.min(metric_values)
            )
        else:
            raise ValueError(
                f"Unsupported version comparison aggregation: {aggregation}"
            )

        values_by_run.setdefault(run_id, []).append(value)

    values = {}
    for run_id, setting_values in values_by_run.items():
        if run_id in invalid_runs:
            continue
        if aggregation == "geom_mean":
            value = float(np.exp(np.log(setting_values).mean()))
        else:
            value = float(
                max(setting_values) if higher_is_better else min(setting_values)
            )
        values[run_id] = value

    return (
        values,
        duplicate_run_ids,
        duplicate_concurrencies,
        missing_uuid_rows,
        run_ids,
    )


def compare_version_runs(data_v1, data_v2, metric_config, selected_concurrencies):
    """Compare run-level metrics using medians and a bootstrap CI for V2 minus V1."""
    result = {
        "median_v1": None,
        "median_v2": None,
        "delta": None,
        "ci_low": None,
        "ci_high": None,
        "n_v1": 0,
        "n_v2": 0,
        "status": "N/A",
        "duplicate_runs_v1": 0,
        "duplicate_runs_v2": 0,
        "duplicate_run_ids_v1": [],
        "duplicate_run_ids_v2": [],
        "duplicate_concurrencies_v1": [],
        "duplicate_concurrencies_v2": [],
        "missing_uuid_rows_v1": 0,
        "missing_uuid_rows_v2": 0,
        "shared_runs": 0,
    }
    if (
        _CONCURRENCY_COLUMN not in data_v1.columns
        or _CONCURRENCY_COLUMN not in data_v2.columns
    ):
        return result

    concurrencies_v1 = set(
        pd.to_numeric(data_v1[_CONCURRENCY_COLUMN], errors="coerce").dropna().unique()
    )
    concurrencies_v2 = set(
        pd.to_numeric(data_v2[_CONCURRENCY_COLUMN], errors="coerce").dropna().unique()
    )
    common_concurrencies = concurrencies_v1 & concurrencies_v2
    if metric_config["aggregation"] == "geom_mean":
        selected = set(
            pd.to_numeric(pd.Series(list(selected_concurrencies)), errors="coerce")
            .dropna()
            .unique()
        )
        common_concurrencies &= selected
    if not common_concurrencies:
        return result

    (
        values_v1,
        duplicates_v1,
        duplicate_concurrencies_v1,
        missing_v1,
        run_ids_v1,
    ) = _run_metric_values(data_v1, metric_config, common_concurrencies)
    (
        values_v2,
        duplicates_v2,
        duplicate_concurrencies_v2,
        missing_v2,
        run_ids_v2,
    ) = _run_metric_values(data_v2, metric_config, common_concurrencies)
    shared_runs = run_ids_v1 & run_ids_v2
    for run_id in shared_runs:
        values_v1.pop(run_id, None)
        values_v2.pop(run_id, None)

    sample_v1 = np.asarray(sorted(values_v1.values()), dtype=float)
    sample_v2 = np.asarray(sorted(values_v2.values()), dtype=float)
    result.update(
        {
            "median_v1": float(np.median(sample_v1)) if len(sample_v1) else None,
            "median_v2": float(np.median(sample_v2)) if len(sample_v2) else None,
            "delta": (
                float(np.median(sample_v2) - np.median(sample_v1))
                if len(sample_v1) and len(sample_v2)
                else None
            ),
            "n_v1": len(sample_v1),
            "n_v2": len(sample_v2),
            "duplicate_runs_v1": len(duplicates_v1),
            "duplicate_runs_v2": len(duplicates_v2),
            "duplicate_run_ids_v1": sorted(duplicates_v1),
            "duplicate_run_ids_v2": sorted(duplicates_v2),
            "duplicate_concurrencies_v1": sorted(duplicate_concurrencies_v1),
            "duplicate_concurrencies_v2": sorted(duplicate_concurrencies_v2),
            "missing_uuid_rows_v1": missing_v1,
            "missing_uuid_rows_v2": missing_v2,
            "shared_runs": len(shared_runs),
        }
    )
    if len(sample_v1) < MIN_RUNS or len(sample_v2) < MIN_RUNS:
        result["status"] = "Inconclusive"
        return result

    rng = np.random.default_rng(0)
    bootstrap_deltas = np.empty(_BOOTSTRAP_RESAMPLES)
    for start in range(0, _BOOTSTRAP_RESAMPLES, _BOOTSTRAP_BATCH_SIZE):
        stop = min(start + _BOOTSTRAP_BATCH_SIZE, _BOOTSTRAP_RESAMPLES)
        draws_v1 = rng.choice(
            sample_v1, size=(stop - start, len(sample_v1)), replace=True
        )
        draws_v2 = rng.choice(
            sample_v2, size=(stop - start, len(sample_v2)), replace=True
        )
        bootstrap_deltas[start:stop] = np.median(draws_v2, axis=1) - np.median(
            draws_v1, axis=1
        )

    ci_low, ci_high = np.percentile(bootstrap_deltas, [2.5, 97.5])
    result["ci_low"] = float(ci_low)
    result["ci_high"] = float(ci_high)

    if metric_config["higher_is_better"]:
        if ci_high < 0:
            result["status"] = "Regression"
        elif ci_low > 0:
            result["status"] = "Improvement"
        else:
            result["status"] = "No clear change"
    elif ci_low > 0:
        result["status"] = "Regression"
    elif ci_high < 0:
        result["status"] = "Improvement"
    else:
        result["status"] = "No clear change"

    return result
