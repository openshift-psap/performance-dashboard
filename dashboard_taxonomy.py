"""Small, dependency-light helpers for dashboard run taxonomy."""

import json
import re
from itertools import cycle

import pandas as pd

DEFAULT_LABEL = "default"
UNLABELED = "Unlabeled"

_VERSION_PREFIXES = {
    "rhaiis": "RHAIIS",
    "vllm": "vLLM",
    "sglang": "sglang",
}

_PRODUCT_RELEASE_RE = re.compile(
    r"(?P<prefix>RHAIIS|vLLM|sglang)-?"
    r"(?P<release>\d+(?:\.\d+)+(?:[A-Za-z]+)?(?:-(?:GA|EA\d*|RC\d+))?)",
    re.IGNORECASE,
)

# Dark, colorblind-safe colors with enough contrast against the light plot theme.
COLORBLIND_SAFE_PALETTE = (
    "#005AB5",
    "#D55E00",
    "#009E73",
    "#7A3E9D",
    "#A6761D",
    "#0072B2",
    "#B2182B",
    "#333333",
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

_HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")


def normalize_label(value):
    """Return a display-safe label while preserving free-form label text."""
    if value is None or pd.isna(value):
        return UNLABELED
    label = str(value).strip()
    return label or UNLABELED


def display_label(value):
    """Return the compact UI value for an optional label."""
    normalized = normalize_label(value)
    return "—" if normalized in {DEFAULT_LABEL, UNLABELED} else normalized


def split_legacy_version(version):
    """Split a release and label already encoded in a version value."""
    if not isinstance(version, str):
        return version, DEFAULT_LABEL
    raw_version = version.strip()
    match = _PRODUCT_RELEASE_RE.search(raw_version)
    if not match:
        return raw_version, DEFAULT_LABEL

    prefix = _VERSION_PREFIXES[match.group("prefix").lower()]
    if match.start() == 0:
        release = f"{prefix}-{match.group('release')}"
    else:
        # Preserve wrappers such as AIC-0.10.0-vLLM-0.24.0 as the release.
        release = raw_version[: match.end()]

    suffix = raw_version[match.end() :].strip()
    label = suffix.removeprefix("-").strip() if suffix.startswith("-") else ""
    return release, normalize_label(label) if label else DEFAULT_LABEL


def uses_legacy_methodology(version):
    """Return whether a RHAIIS/vLLM release predates the methodology change."""
    release, _ = split_legacy_version(version)
    match = re.search(
        r"(?P<prefix>RHAIIS|vLLM)-?(?P<release>\d+(?:\.\d+)+)",
        str(release),
        re.IGNORECASE,
    )
    if not match:
        return False

    release_parts = tuple(int(part) for part in match.group("release").split("."))
    release_parts += (0,) * (3 - len(release_parts))
    threshold = (
        (3, 6, 0)
        if match.group("prefix").casefold() == "rhaiis"
        else (0, 26, 0)
    )
    return release_parts < threshold


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
    if "uuid" not in result.columns:
        result["uuid"] = ""
    else:
        result["uuid"] = result["uuid"].fillna("").astype(str).str.strip()
    result = result.drop(
        columns=["user", "username", "submitted_by", "furnace_user"],
        errors="ignore",
    )
    return result


def filter_taxonomy(df, versions=None, labels=None, uuids=None):
    """Apply the optional version/label/run filters to a DataFrame."""
    result = df
    for column, values in (
        ("version", versions),
        ("label", labels),
        ("uuid", uuids),
    ):
        if values:
            result = result[result[column].isin(values)]
    return result


def encode_version_label_pairs(pairs):
    """Encode exact version/label selections for a shareable URL."""
    return json.dumps(
        [list(pair) for pair in sorted(set(pairs))],
        separators=(",", ":"),
    )


def decode_version_label_pairs(raw_value):
    """Decode exact version/label selections from a URL value."""
    if not raw_value:
        return []
    try:
        parsed = json.loads(str(raw_value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        (str(pair[0]), str(pair[1]))
        for pair in parsed
        if isinstance(pair, list) and len(pair) == 2 and all(pair)
    ]


def version_label_pair_mask(df, pairs, *, include_default=False):
    """Return a mask for rows matching selected version/label pairs."""
    if not pairs:
        return pd.Series(True, index=df.index)
    if not {"version", "label"}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    pair_index = pd.MultiIndex.from_frame(df[["version", "label"]].astype(str))
    mask = pd.Series(pair_index.isin(set(pairs)), index=df.index)
    if include_default:
        selected_versions = {version for version, _ in pairs}
        mask |= df["version"].isin(selected_versions) & df["label"].isin(
            {DEFAULT_LABEL, UNLABELED}
        )
    return mask


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


def taxonomy_query_params(
    labels=None,
    uuids=None,
    version_label_pairs=None,
):
    """Return URL parameters for the taxonomy filters."""
    params = {
        key: ",".join(values)
        for key, values in (
            ("labels", labels),
            ("uuids", uuids),
        )
        if values
    }
    if version_label_pairs:
        params["version_labels"] = encode_version_label_pairs(version_label_pairs)
    return params


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


def is_valid_hex_color(value):
    """Return whether a value is a supported CSS hex color."""
    return isinstance(value, str) and bool(_HEX_COLOR_RE.fullmatch(value))


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
    release = version
    if include_label and label not in {DEFAULT_LABEL, UNLABELED}:
        release += f" · {label}"
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


def _runtime_args_key(value):
    """Normalize runtime-argument ordering for duplicate-run comparison."""
    if value is None or pd.isna(value):
        return ""
    return ";".join(
        sorted(" ".join(part.split()) for part in str(value).split(";") if part.strip())
    )


def add_trace_metadata(df, series_column="run_identifier", legend_options=None):
    """Add duplicate-run legend labels, styles, shapes, and opacity."""
    result = df.copy()
    legend_options = legend_options or {}
    if "uuid" not in result.columns:
        result["uuid"] = ""
    result["uuid"] = result["uuid"].fillna("").astype(str).str.strip()
    if "runtime_args" not in result.columns:
        result["runtime_args"] = ""
    result["runtime_config_key"] = result["runtime_args"].map(_runtime_args_key)
    for column in ("spec_decoding", "prefix_caching", "turns", "prefix_tokens", "prefix_count"):
        if column in result.columns:
            result["runtime_config_key"] += (
                ";" + column + "=" + result[column].fillna("").astype(str).str.strip()
            )

    run_info = result.loc[
        result["uuid"].ne(""),
        [series_column, "uuid", "runtime_config_key"],
    ].drop_duplicates([series_column, "uuid"])
    run_info["runtime_variant_count"] = run_info.groupby(series_column)[
        "runtime_config_key"
    ].transform("nunique")
    run_info = run_info.sort_values([series_column, "uuid"])
    run_info["run_rank"] = run_info.groupby(series_column).cumcount()
    result = result.merge(
        run_info[[series_column, "uuid", "runtime_variant_count", "run_rank"]],
        how="left",
        on=[series_column, "uuid"],
    )

    run_counts = result.groupby(series_column)["uuid"].transform(
        lambda values: values[values != ""].nunique()
    )
    result["line_style"] = run_counts.gt(1).map({True: "dot", False: "solid"})
    result["line_opacity"] = 1.0
    exact_repeats = (
        run_counts.gt(1)
        & result["uuid"].ne("")
        & result["runtime_variant_count"].eq(1)
        & result["run_rank"].gt(0)
    )
    result.loc[exact_repeats, "line_opacity"] = 0.55
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
    return result
