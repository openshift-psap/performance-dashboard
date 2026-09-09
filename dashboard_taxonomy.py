"""Small, dependency-light helpers for dashboard run taxonomy."""

import json
import re
from itertools import cycle

import pandas as pd

DEFAULT_LABEL = "default"
UNLABELED = "Unlabeled"

_PRODUCT_FAMILIES = {
    "rhaiis": "RHAIIS",
    "vllm": "vLLM",
    "sglang": "sglang",
}

_PRODUCT_RELEASE_RE = re.compile(
    r"(?P<prefix>RHAIIS|vLLM|sglang)-?"
    r"(?P<release>\d+(?:\.\d+)+(?:[A-Za-z]+)?(?:-(?:GA|EA\d*|RC\d+))?)",
    re.IGNORECASE,
)

# Okabe-Ito colors plus a few high-contrast extensions.
COLORBLIND_SAFE_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

MARKER_SYMBOLS = (
    "circle",
    "triangle-up",
    "square",
    "diamond",
    "x",
    "triangle-down",
    "star",
    "hexagon",
)


def normalize_label(value):
    """Return a display-safe label while preserving free-form label text."""
    if value is None or pd.isna(value):
        return UNLABELED
    label = str(value).strip()
    return label or UNLABELED


def derive_product_family(version):
    """Derive the product family from a release version or wrapper name."""
    if not isinstance(version, str):
        return "Other"
    match = _PRODUCT_RELEASE_RE.search(version.strip())
    if match:
        return _PRODUCT_FAMILIES[match.group("prefix").lower()]
    prefix = version.strip().split("-", 1)[0].lower()
    return _PRODUCT_FAMILIES.get(prefix, "Other")


def split_legacy_version(version):
    """Split a release and label already encoded in a version value."""
    if not isinstance(version, str):
        return version, DEFAULT_LABEL
    raw_version = version.strip()
    match = _PRODUCT_RELEASE_RE.search(raw_version)
    if not match:
        return raw_version, DEFAULT_LABEL

    prefix = _PRODUCT_FAMILIES[match.group("prefix").lower()]
    if match.start() == 0:
        release = f"{prefix}-{match.group('release')}"
    else:
        # Preserve wrappers such as AIC-0.10.0-vLLM-0.24.0 as the release.
        release = raw_version[: match.end()]

    suffix = raw_version[match.end() :].strip()
    label = suffix.removeprefix("-").strip() if suffix.startswith("-") else ""
    return release, normalize_label(label) if label else DEFAULT_LABEL


def normalize_taxonomy_columns(df):
    """Add compatible taxonomy columns without changing the source data."""
    result = df.copy()
    source_versions = result["version"].astype(str).str.strip()
    split_versions = source_versions.map(split_legacy_version)
    canonical_versions = split_versions.map(lambda value: value[0])
    inferred_labels = split_versions.map(lambda value: value[1])
    result["version"] = canonical_versions
    if "label" in result.columns:
        labels = result["label"].map(normalize_label)
        labels = labels.mask(labels.eq(UNLABELED), inferred_labels)
    else:
        labels = inferred_labels
    result["label"] = labels
    changed_versions = source_versions.ne(canonical_versions)
    if changed_versions.any():
        result["legacy_version"] = source_versions.where(changed_versions, "")
    result["product_family"] = result["version"].map(derive_product_family)
    if "uuid" not in result.columns:
        result["uuid"] = ""
    else:
        result["uuid"] = result["uuid"].fillna("").astype(str).str.strip()
    result = result.drop(
        columns=["user", "username", "submitted_by", "furnace_user"],
        errors="ignore",
    )
    return result


def filter_taxonomy(df, families=None, versions=None, labels=None, uuids=None):
    """Apply the optional family/version/label/run filters to a DataFrame."""
    result = df
    for column, values in (
        ("product_family", families),
        ("version", versions),
        ("label", labels),
        ("uuid", uuids),
    ):
        if values:
            result = result[result[column].isin(values)]
    return result


def parse_filter_values(raw_value, available_values):
    """Parse a comma-separated URL filter and discard stale values."""
    if not raw_value:
        return []
    return [
        value.strip()
        for value in str(raw_value).split(",")
        if value.strip() in available_values
    ]


def sync_selected_options(previous, available, *, select_all=False):
    """Return widget state that stays aligned with its current options."""
    available = list(available)
    if select_all:
        return available
    return [value for value in (previous or []) if value in available]


def taxonomy_query_params(families=None, labels=None, uuids=None):
    """Return URL parameters for the taxonomy filters."""
    return {
        key: ",".join(values)
        for key, values in (
            ("families", families),
            ("labels", labels),
            ("uuids", uuids),
        )
        if values
    }


def encode_query_mapping(mapping):
    """Encode a small mapping compactly for a shareable query parameter."""
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"))


def decode_query_mapping(raw_value, allowed_values=None):
    """Decode a small JSON mapping and optionally restrict its values."""
    if not raw_value:
        return {}
    try:
        parsed = json.loads(str(raw_value))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in parsed.items()
        if key and (allowed_values is None or value in allowed_values)
    }


def deterministic_color_map(series_keys):
    """Assign stable colors to sorted series keys."""
    keys = sorted(set(series_keys))
    return dict(zip(keys, cycle(COLORBLIND_SAFE_PALETTE)))


def compact_series_label(
    row,
    *,
    include_accelerator=True,
    include_model=True,
    include_profile=False,
    include_label=True,
    include_tp=True,
    include_dp=True,
):
    """Build a concise legend label while keeping the full row in tooltips."""
    accelerator = str(row.get("accelerator") or "?")
    model = str(row.get("model_short") or row.get("model") or "?").rsplit("/", 1)[-1]
    for prefix in ("NVIDIA-", "RedHatAI/"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
    for suffix in ("-Instruct", "-instruct", "-dynamic"):
        model = model.removesuffix(suffix)
    if len(model) > 28:
        model = model[:27] + "…"
    version = str(row.get("version") or "?")
    label = normalize_label(row.get("label"))
    release = version + (f" · {label}" if include_label else "")
    parts = []
    if include_accelerator:
        parts.append(accelerator)
    if include_model:
        parts.append(model)
    parts.append(release)

    if include_profile:
        profile = str(row.get("profile") or "?")
        parts.append(profile)

    tp = row.get("TP")
    if include_tp and pd.notna(tp):
        parts.append(f"TP={int(tp) if float(tp).is_integer() else tp}")

    dp = row.get("DP")
    if include_dp and pd.notna(dp) and float(dp) > 1:
        parts.append(f"DP={int(dp) if float(dp).is_integer() else dp}")

    spec_decoding = str(row.get("spec_decoding") or "").strip()
    if spec_decoding and spec_decoding.lower() not in {"no", "false", "none", "nan"}:
        parts.append(f"SD={spec_decoding}")

    prefix_caching = str(row.get("prefix_caching") or "").strip()
    if prefix_caching and prefix_caching.lower() not in {
        "no",
        "false",
        "none",
        "nan",
    }:
        parts.append(f"PC={prefix_caching}")

    turns = row.get("turns")
    if pd.notna(turns) and float(turns) > 1:
        parts.append(f"{int(turns) if float(turns).is_integer() else turns}T")

    return " | ".join(parts)


def add_trace_metadata(df, series_column="run_identifier", legend_options=None):
    """Add duplicate-run legend labels, line styles, and marker symbols."""
    result = df.copy()
    legend_options = legend_options or {}
    if "uuid" not in result.columns:
        result["uuid"] = ""
    result["uuid"] = result["uuid"].fillna("").astype(str).str.strip()
    run_counts = result.groupby(series_column)["uuid"].transform(
        lambda values: values[values != ""].nunique()
    )
    result["line_style"] = run_counts.gt(1).map({True: "dot", False: "solid"})
    result["trace_label"] = result[series_column].astype(str)
    result["legend_label"] = result.apply(
        lambda row: compact_series_label(row, **legend_options), axis=1
    )
    duplicate_runs = run_counts.gt(1) & result["uuid"].ne("")
    result.loc[duplicate_runs, "trace_label"] += (
        " | UUID=" + result.loc[duplicate_runs, "uuid"]
    )
    result.loc[duplicate_runs, "legend_label"] += (
        " | run " + result.loc[duplicate_runs, "uuid"].str[:8]
    )

    result["marker_symbol"] = "circle"
    for _, indexes in result.groupby(series_column, sort=False).groups.items():
        run_ids = sorted(
            run_id for run_id in result.loc[indexes, "uuid"].unique() if run_id
        )
        if len(run_ids) > 1:
            symbols = {
                run_id: MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)]
                for index, run_id in enumerate(run_ids)
            }
            result.loc[indexes, "marker_symbol"] = (
                result.loc[indexes, "uuid"].map(symbols).fillna("circle")
            )
    return result
