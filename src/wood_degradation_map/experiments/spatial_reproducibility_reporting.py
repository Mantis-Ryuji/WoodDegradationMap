"""Complete-case summaries, display-only label alignment, and publication figures."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .global_k8_repeats import repeat_paths
from .manifests import _digest, _read_json, _require
from .spatial_reproducibility_metrics import WINDOWS, boundary_frequency, validate_map

MAP_IDS = tuple((encoder, kmeans) for encoder in (1, 2, 3) for kmeans in (1, 2, 3))
GROUPS = ("within_encoder", "between_encoder")
SUMMARY_GROUPS = (*GROUPS, "initial_reference")
EXPECTED_PAIRS = {"within_encoder": 9, "between_encoder": 27, "initial_reference": 8}
VI_METRICS = ("h_b_given_a_bits", "h_a_given_b_bits", "vi_bits")
BOUNDARY_METRICS = tuple(
    f"boundary_{window}_{side}" for window in WINDOWS for side in ("lower", "upper")
)
METRICS = ("ari", *VI_METRICS, *BOUNDARY_METRICS)
COLORS = ("#0072B2", "#D55E00", "#009E73")
SAMPLE_FILES = (
    "01_nine_maps.png", "03_boundary_frequency.png",
    "boundary_frequency.npz",
)
SUMMARY_FILES = ("01_ari_vs_boundary_coverage.png",)


def color_alignment_contract() -> dict[str, object]:
    """Describe the display-only assignment independently of numerical evaluation."""
    return {
        "reference": [1, 1], "K": 8, "background": 0,
        "method": "Hungarian maximum total pixel overlap; one-to-one permutation",
        "scope": "all valid pixels pooled over all planned samples; one mapping per run",
        "direction": "target raw -> initial (1,1) raw -> saved global-fit M11 display ID",
        "global_display": "reuse M11/matching.csv and the palette saved in global_k8_v1/report.json",
        "ties": "SciPy assignment on ascending raw IDs; absent target IDs use unused colors in order",
        "application": "3x3 label-map display only; source labels and all metrics unchanged",
    }


def label_overlap_counts(
    reference: np.ndarray, target: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Count initial/raw label pairs on exactly the original valid-pixel set."""
    validate_map(reference, mask)
    validate_map(target, mask)
    valid = mask.astype(bool, copy=False)
    pairs = (reference[valid].astype(np.int64) - 1) * 8 + target[valid].astype(np.int64) - 1
    return np.bincount(pairs, minlength=64).reshape(8, 8)


def display_mappings(
    counts: Mapping[tuple[int, int], np.ndarray],
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict[str, object]]]:
    """Align each run directly to (1,1) using pooled counts, never merging labels.

    Rows of each 8x8 table are initial raw IDs and columns are target raw IDs.
    Empty target clusters receive the remaining colors in ascending order. Every
    zero-overlap assignment is explicitly unsupported, including present clusters
    assigned to an absent reference ID. Color similarity is not a scientific score.
    """
    from scipy.optimize import linear_sum_assignment

    if set(counts) != set(MAP_IDS):
        raise ValueError("Color alignment requires all nine overlap tables")
    tables = {}
    for identity in MAP_IDS:
        table = counts[identity]
        if (not isinstance(table, np.ndarray) or table.shape != (8, 8)
                or table.dtype.kind not in "iu" or np.any(table < 0)
                or np.any(table > np.iinfo(np.int64).max)):
            raise ValueError("Expected nonnegative integer 8x8 overlap counts")
        tables[identity] = table.astype(np.int64, copy=False)
    initial = tables[1, 1]
    reference_counts = initial.sum(axis=1)
    total = int(reference_counts.sum())
    if total <= 0 or not np.array_equal(initial, np.diag(reference_counts)):
        raise ValueError("Initial overlap table must be a nonempty self-comparison")
    lookups, records = {}, []
    for identity in MAP_IDS:
        table = tables[identity]
        if not np.array_equal(table.sum(axis=1), reference_counts):
            raise ValueError("Overlap tables must cover the same initial valid pixels")
        target_counts = table.sum(axis=0)
        present = np.flatnonzero(target_counts > 0)
        lookup = np.arange(9, dtype=np.uint8)
        if identity != (1, 1):
            reference_ids, target_columns = linear_sum_assignment(table[:, present], maximize=True)
            lookup[present[target_columns] + 1] = reference_ids + 1
            unused = sorted(set(range(1, 9)) - set(int(value) for value in lookup[present + 1]))
            lookup[np.flatnonzero(target_counts == 0) + 1] = unused
        assignments = []
        for target_id in range(1, 9):
            display_id = int(lookup[target_id])
            overlap = int(table[display_id - 1, target_id - 1])
            assignments.append({
                "raw_cluster": target_id, "display_cluster": display_id,
                "overlap_pixels": overlap, "zero_overlap": overlap == 0,
                "target_pixels": int(target_counts[target_id - 1]),
                "reference_pixels": int(reference_counts[display_id - 1]),
            })
        lookups[identity] = lookup
        records.append({
            "encoder_repeat": identity[0], "kmeans_repeat": identity[1],
            "target_to_display_including_background": lookup.tolist(),
            "contingency_rows": "initial (1,1) raw IDs 1..8",
            "contingency_columns": "target raw IDs 1..8",
            "contingency": table.tolist(), "valid_pixels": total,
            "assignments": assignments,
        })
    return lookups, records


def _validate_display_lookups(lookups: Mapping[tuple[int, int], np.ndarray]) -> None:
    if set(lookups) != set(MAP_IDS):
        raise ValueError("Display mappings must cover all nine maps")
    for lookup in lookups.values():
        validate_permutation(lookup)
    if not np.array_equal(lookups[1, 1], np.arange(9)):
        raise ValueError("Initial (1,1) display labels must remain unchanged")


def validate_permutation(lookup: np.ndarray) -> None:
    """Require a display-only permutation that preserves background zero."""
    if (not isinstance(lookup, np.ndarray) or lookup.shape != (9,)
            or lookup.dtype.kind not in "iu" or lookup[0] != 0
            or not np.array_equal(np.sort(lookup[1:]), np.arange(1, 9))):
        raise ValueError("Display lookup must preserve background and permute labels 1..8")


@dataclass(frozen=True)
class GlobalDisplay:
    """The saved M11 raw-to-display assignment and its exact saved palette."""

    lookup: np.ndarray
    palette: tuple[str, ...]
    provenance: dict[str, object]


def load_global_display(source: Path, sample_ids: tuple[str, ...]) -> GlobalDisplay:
    """Verify the two display-source files without regenerating any global result."""
    from matplotlib.colors import is_color_like, to_rgba

    directory = source / "results/figures/global_k8_v1"
    completion = _read_json(directory / "completion.json")
    _require(completion.get("status") == "global_report_completed",
             "The global-fit figure report is incomplete")
    hashes = completion.get("artifact_sha256")
    _require(isinstance(hashes, dict), "Global-fit report has no artifact hashes")
    source_hashes = {}
    for name in ("report.json", "M11/matching.csv"):
        source_hashes[name] = _digest(directory / name)
        _require(source_hashes[name] == hashes.get(name),
                 f"Global display source changed: {name}")
    report = _read_json(directory / "report.json")
    contract = report["contract"]
    _require(report["sample_ids"] == list(sample_ids)
             and contract.get("K") == 8 and contract.get("repeat") == 1
             and contract.get("reference") == "M00+Cosine-KMeans",
             "Global display uses a different sample set, fit or reference")
    regenerated, _ = repeat_paths(source, "M11", 1)
    _require(contract["sources"]["M11"] == _digest(regenerated / "completion.json"),
             "Global display refers to a different M11 source")
    colors = report["definitions"]["palette"]
    _require(isinstance(colors, list) and len(colors) == 9
             and all(isinstance(color, str) and is_color_like(color) for color in colors),
             "Global display requires nine valid saved colors")
    _require(to_rgba(colors[0]) == to_rgba("white"), "Global background must be white")
    with (directory / "M11/matching.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 8 and all(row["condition"] == "M11" for row in rows),
             "Expected eight M11 display assignments")
    originals = [int(row["original_cluster"]) for row in rows]
    displays = [int(row["display_cluster"]) for row in rows]
    _require(sorted(originals) == list(range(1, 9))
             and sorted(displays) == list(range(1, 9)),
             "Global M11 matching must be a one-to-one permutation")
    lookup = np.zeros(9, dtype=np.uint8)
    lookup[originals] = displays
    validate_permutation(lookup)
    return GlobalDisplay(lookup, tuple(colors), {
        "directory": str(directory.resolve()),
        "completion_sha256": _digest(directory / "completion.json"),
        "artifact_sha256": source_hashes,
        "initial_raw_to_global_display": lookup.tolist(), "palette": colors,
        "reference": "saved global-fit M11 display, aligned to M00+Cosine-KMeans",
        "initial_map": "canonical original map; not replaced by the regenerated copy",
    })


def compose_display_mappings(
    initial_lookups: Mapping[tuple[int, int], np.ndarray], global_lookup: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Compose target raw -> initial raw -> global display; preserve each cluster."""
    _validate_display_lookups(initial_lookups)
    validate_permutation(global_lookup)
    return {identity: global_lookup[initial_lookups[identity]] for identity in MAP_IDS}


def _record(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _rows(value: object, context: str) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a sequence of records")
    return [_record(row, context) for row in value]


def _number(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number or null")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be a finite number or null")
    return result


def _pair_key(pair: Mapping[str, object]) -> tuple[tuple[int, int], tuple[int, int]]:
    values = tuple(pair.get(key) for key in ("encoder_a", "kmeans_a", "encoder_b", "kmeans_b"))
    if any(type(value) is not int for value in values):
        raise ValueError("Pair identities must contain integer encoder and Cosine-KMeans repeats")
    encoder_a, kmeans_a, encoder_b, kmeans_b = cast(tuple[int, int, int, int], values)
    left, right = (encoder_a, kmeans_a), (encoder_b, kmeans_b)
    if left not in MAP_IDS or right not in MAP_IDS or left >= right:
        raise ValueError("Pair identities must be distinct, valid, and in canonical order")
    expected_group = "within_encoder" if encoder_a == encoder_b else "between_encoder"
    if pair.get("group") != expected_group:
        raise ValueError("Pair group does not match its encoder identities")
    if pair.get("initial_reference") is not (left == (1, 1)):
        raise ValueError("Initial-reference flag does not match the canonical initial map")
    return left, right


def vi_values(pair: Mapping[str, object]) -> dict[str, float | None]:
    """Read both conditional entropies in bits; an absent diagnostic is not zero."""
    if "overlap" not in pair:
        return dict.fromkeys(VI_METRICS)
    overlap = _record(pair["overlap"], "Overlap")
    components = {name: _number(overlap.get(name), name) for name in VI_METRICS[:2]}
    if any(value is None or value < 0 for value in components.values()):
        raise ValueError("VI components must both be finite nonnegative numbers")
    return {**components, "vi_bits": sum(cast(float, value) for value in components.values())}


def _metric_values(
    pair: Mapping[str, object],
) -> tuple[dict[str, float | None], dict[str, str], dict[str, object]]:
    ari = _record(pair.get("ari"), "ARI")
    values = {"ari": _number(ari.get("value"), "ARI value"), **vi_values(pair)}
    reasons: dict[str, str] = {}
    for name in VI_METRICS:
        if values[name] is None:
            reasons[name] = "overlap_not_requested"
    if values["ari"] is None:
        reasons["ari"] = str(ari.get("undefined_reason") or "unspecified")
    windows: dict[int, Mapping[str, object]] = {}
    for row in _rows(pair.get("boundaries"), "Boundary results"):
        window = row.get("window")
        if type(window) is not int or window not in WINDOWS or window in windows:
            raise ValueError("Boundary results must contain each of windows 3, 5, 9 once")
        windows[window] = row
    if set(windows) != set(WINDOWS):
        raise ValueError("Boundary results must contain all three fixed windows")
    diagnostics: dict[str, object] = {}
    for window in WINDOWS:
        row = windows[window]
        directions = [_record(row.get(direction), direction) for direction in ("a_to_b", "b_to_a")]
        directional_values = [_number(direction.get("value"), "Coverage")
                              for direction in directions]
        missing = [str(direction.get("undefined_reason") or "unspecified")
                   for direction, value in zip(directions, directional_values, strict=True)
                   if value is None]
        diagnostic = {
            "defined_directions": sum(value is not None for value in directional_values),
            "zero_directions": sum(value == 0 for value in directional_values),
            "undefined_reasons": missing,
        }
        diagnostics[str(window)] = diagnostic
        for side in ("lower", "upper"):
            name = f"boundary_{window}_{side}"
            saved = _number(row.get(side), name)
            if missing:
                if saved is not None:
                    raise ValueError("Boundary extrema require both directions to be defined")
                values[name] = None
                reasons[name] = ";".join(sorted(set(missing)))
            else:
                actual = [cast(float, value) for value in directional_values]
                expected = min(actual) if side == "lower" else max(actual)
                if saved is None or saved != expected:
                    raise ValueError("Saved boundary extrema disagree with directional values")
                values[name] = expected
    return values, reasons, diagnostics


def _distribution(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "minimum": None, "maximum": None, "sample_sd": None}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else None}


def _sample_summary(evaluation: Mapping[str, object]) -> dict[str, object]:
    sample_id = evaluation.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Each evaluation requires a nonempty sample_id")
    pairs = _rows(evaluation.get("pairs"), "Pairs")
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    by_group: dict[str, list[tuple[Mapping[str, object], dict[str, float | None],
                                        dict[str, str], dict[str, object]]]] = {
        group: [] for group in SUMMARY_GROUPS
    }
    for pair in pairs:
        key = _pair_key(pair)
        if key in seen:
            raise ValueError(f"Duplicate pair in sample {sample_id}")
        seen.add(key)
        values, reasons, diagnostics = _metric_values(pair)
        by_group[cast(str, pair["group"])].append((pair, values, reasons, diagnostics))
        if pair["initial_reference"]:
            by_group["initial_reference"].append((pair, values, reasons, diagnostics))
    map_rows = _rows(evaluation.get("maps"), "Map diagnostics")
    map_seen: set[tuple[int, int]] = set()
    for row in map_rows:
        identity = (row.get("encoder_repeat"), row.get("kmeans_repeat"))
        if any(type(value) is not int for value in identity) or identity not in MAP_IDS:
            raise ValueError("Map diagnostics have an invalid encoder/Cosine-KMeans identity")
        key = cast(tuple[int, int], identity)
        if key in map_seen:
            raise ValueError(f"Duplicate map diagnostics in sample {sample_id}")
        map_seen.add(key)
    groups: dict[str, object] = {}
    for group in SUMMARY_GROUPS:
        rows = by_group[group]
        expected = EXPECTED_PAIRS[group]
        planned = [pair for pair in combinations(MAP_IDS, 2)
                   if (pair[0] == (1, 1) if group == "initial_reference" else
                       (pair[0][0] == pair[1][0]) == (group == "within_encoder"))]
        metrics: dict[str, object] = {}
        for metric in METRICS:
            defined = [cast(float, values[metric]) for _, values, _, _ in rows
                       if values[metric] is not None]
            counts = Counter(reasons[metric] for _, values, reasons, _ in rows
                             if values[metric] is None)
            if len(rows) < expected:
                counts["missing_pair"] = expected - len(rows)
            complete = len(defined) == expected
            metrics[metric] = {
                "expected_pairs": expected, "observed_pairs": len(rows),
                "defined_pairs": len(defined), "complete_case": complete,
                "undefined_pair_reasons": dict(sorted(counts.items())),
                "partial_defined_pairs": _distribution(defined),
                **_distribution(defined if complete else []),
            }
        boundary_diagnostics: dict[str, object] = {}
        for window in WINDOWS:
            entries = [_record(diagnostic[str(window)], "Direction diagnostics")
                       for _, _, _, diagnostic in rows]
            reason_counts: Counter[str] = Counter()
            for entry in entries:
                reason_counts.update(cast(list[str], entry["undefined_reasons"]))
            boundary_diagnostics[str(window)] = {
                "expected_directions": 2 * expected,
                "observed_directions": 2 * len(rows),
                "defined_directions": sum(cast(int, entry["defined_directions"])
                                          for entry in entries),
                "zero_directions": sum(cast(int, entry["zero_directions"]) for entry in entries),
                "one_direction_undefined_pairs": sum(entry["defined_directions"] == 1
                                                      for entry in entries),
                "both_directions_undefined_pairs": sum(entry["defined_directions"] == 0
                                                       for entry in entries),
                "undefined_direction_reasons": dict(sorted(reason_counts.items())),
            }
        groups[group] = {
            "expected_pairs": expected, "observed_pairs": len(rows),
            "missing_pairs": [[list(left), list(right)] for left, right in planned
                              if (left, right) not in seen],
            "metrics": metrics, "boundary_diagnostics": boundary_diagnostics,
        }
    return {
        "sample_id": sample_id, "groups": groups, "map_diagnostics": map_rows,
        "empty_boundary_map_count": sum(row.get("boundary_count") == 0 for row in map_rows),
        "missing_maps": [list(identity) for identity in MAP_IDS if identity not in map_seen],
        "initial_reference_pairs": [dict(pair) for pair in pairs if pair["initial_reference"]],
    }


def _metric_row(sample: Mapping[str, object], group: str, metric: str) -> Mapping[str, object]:
    groups = _record(sample["groups"], "Groups")
    metrics = _record(_record(groups[group], "Group")["metrics"], "Metrics")
    return _record(metrics[metric], "Metric summary")


def _macro(values: Mapping[str, float], all_ids: Sequence[str]) -> dict[str, object]:
    distribution = _distribution(list(values.values()))
    return {
        "sample_count": len(values), "common_sample_ids": sorted(values),
        "excluded_sample_ids": sorted(set(all_ids) - set(values)),
        "sample_values": dict(sorted(values.items())), "macro_mean": distribution["mean"],
        "sample_minimum": distribution["minimum"], "sample_maximum": distribution["maximum"],
        "sample_sd": distribution["sample_sd"],
        "undefined_reason": None if values else "no_complete_case_samples",
    }


def summarize_samples(evaluations: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize complete planned pair sets, retaining partial availability separately.

    All boundary windows and both extrema use a common sample base within a
    comparison group. Differences between groups use their intersection.
    No pair is treated as an independent sample for a confidence interval.
    """
    samples = [_sample_summary(evaluation) for evaluation in evaluations]
    sample_ids = [cast(str, sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id in evaluations")
    samples.sort(key=lambda sample: cast(str, sample["sample_id"]))
    complete: dict[tuple[str, str], set[str]] = {}
    for group in SUMMARY_GROUPS:
        for metric in METRICS:
            complete[group, metric] = {
                cast(str, sample["sample_id"]) for sample in samples
                if _metric_row(sample, group, metric)["complete_case"]
            }
        boundary_common = set.intersection(*(complete[group, metric]
                                            for metric in BOUNDARY_METRICS))
        for metric in BOUNDARY_METRICS:
            complete[group, metric] = boundary_common
    metrics: dict[str, object] = {}
    for metric in METRICS:
        groups: dict[str, object] = {}
        values: dict[str, dict[str, float]] = {}
        for group in SUMMARY_GROUPS:
            values[group] = {
                cast(str, sample["sample_id"]): cast(
                    float, _metric_row(sample, group, metric)["mean"])
                for sample in samples if sample["sample_id"] in complete[group, metric]
            }
            groups[group] = {"expected_pairs_per_sample": EXPECTED_PAIRS[group],
                             **_macro(values[group], sample_ids)}
        common = set(values[GROUPS[0]]) & set(values[GROUPS[1]])
        differences = {sample_id: values["between_encoder"][sample_id]
                       - values["within_encoder"][sample_id] for sample_id in sorted(common)}
        groups["group_difference"] = {
            "definition": "between_encoder minus within_encoder; paired common samples",
            **_macro(differences, sample_ids),
        }
        metrics[metric] = groups
    return {
        "schema_version": 2, "sample_ids": sorted(sample_ids), "sample_count": len(samples),
        "windows": list(WINDOWS), "samples": samples, "metrics": metrics,
        "aggregation": {
            "unit": "equal-weight sample; average over all planned pairs within each sample",
            "boundary_sample_base": "common across 3/5/9 windows and lower/upper directions",
            "missingness": "complete case per group and metric; partial values retained separately",
            "group_difference": "paired intersection of complete-case samples",
            "vi": "bits; H(B|A) split-side, H(A|B) merge-side; A/B follow saved pair order; "
                  "sum is VI; absent overlap diagnostics are undefined",
            "initial_reference": "eight pairs with A=(1,1); subset of the other two groups",
            "inference": "descriptive; no pair-level significance test or confidence interval",
        },
    }


def summary_table_rows(
    summary: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Flatten sample and equal-weight macro summaries without duplicating boundary directions."""
    sample_rows, macro_rows = [], []
    for metric in METRICS:
        unit = "bits" if metric in VI_METRICS else "dimensionless"
        for group in SUMMARY_GROUPS:
            identity = {"group": group, "metric": metric, "unit": unit}
            for sample in _rows(summary["samples"], "Sample summaries"):
                row = _metric_row(sample, group, metric)
                reasons = _record(row["undefined_pair_reasons"], "Undefined reasons")
                sample_rows.append({"sample_id": sample["sample_id"], **identity,
                                    **{key: row[key] for key in (
                                        "expected_pairs", "observed_pairs", "defined_pairs",
                                        "complete_case", "mean", "minimum", "maximum", "sample_sd")},
                                    "undefined_pair_reasons": ";".join(
                                        f"{reason}:{count}" for reason, count in sorted(reasons.items()))})
            groups = _record(_record(summary["metrics"], "Metrics")[metric], metric)
            row = _record(groups[group], group)
            macro_rows.append({**identity, "total_samples": summary["sample_count"],
                               **{key: row[key] for key in (
                                   "expected_pairs_per_sample", "sample_count", "macro_mean",
                                   "sample_minimum", "sample_maximum", "sample_sd", "undefined_reason")}})
    return sample_rows, macro_rows


def _save(figure: Figure, path: Path, dpi: int) -> Path:
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()
    return path


def _summary_values(summary: Mapping[str, object], metric: str, group: str) -> dict[str, float]:
    metrics = _record(summary["metrics"], "Summary metrics")
    group_row = _record(_record(metrics[metric], metric)[group], group)
    values = _record(group_row["sample_values"], "Sample values")
    return {key: cast(float, _number(value, "Sample value")) for key, value in values.items()}


def _figure(rows: int, columns: int, *, height: float, spatial: bool) -> tuple[Figure, np.ndarray]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, height))
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    # Reserve the bottom band for the sample ID and shared legend/colorbar.
    figure.subplots_adjust(left=0.025 if spatial else 0.08, right=0.98,
                           bottom=0.22 if spatial else 0.19, top=0.98,
                           wspace=0.04 if spatial else 0.25,
                           hspace=0.07 if spatial else 0.22)
    for index, axis in enumerate(axes.flat):
        axis.text(0.02, 0.97, f"({chr(97 + index)})", transform=axis.transAxes,
                  ha="left", va="top", fontsize=11,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    return figure, axes


def _spatial_axis(axis: Axes, mask: np.ndarray) -> None:
    axis.set_xlim(-0.5, mask.shape[1] - 0.5)
    axis.set_ylim(mask.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()


def _sample_label(figure: Figure, sample_id: str) -> None:
    figure.text(0.5, 0.165, sample_id, ha="center", va="center",
                fontsize=15, fontweight="bold")


def _legend(figure: Figure, axis: Axes, *, columns: int) -> None:
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
                  ncol=columns, frameon=False, fontsize=9)


def render_sample(
    sample_id: str, maps: Mapping[tuple[int, int], np.ndarray], mask: np.ndarray,
    evaluation: dict[str, object], output: Path, *, display_lookups: Mapping[
        tuple[int, int], np.ndarray], palette: Sequence[str], dpi: int = 160,
) -> list[Path]:
    """Render raw maps through composed global display IDs without altering any input."""
    from matplotlib.colors import BoundaryNorm, ListedColormap, is_color_like

    if set(maps) != set(MAP_IDS) or set(display_lookups) != set(MAP_IDS):
        raise ValueError("Publication figures require all nine maps and display lookups")
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    if evaluation.get("sample_id") != sample_id:
        raise ValueError("Figure sample_id differs from its evaluation")
    for lookup in display_lookups.values():
        validate_permutation(lookup)
    if len(palette) != 9 or any(not is_color_like(color) for color in palette):
        raise ValueError("Expected the nine saved global-fit colors, including background")
    for identity in MAP_IDS:
        validate_map(maps[identity], mask)
    _sample_summary(evaluation)
    pairs = {_pair_key(pair): pair for pair in _rows(evaluation.get("pairs"), "Pairs")}
    if len(pairs) != 36:
        raise ValueError("Sample figures require all 36 evaluated pairs")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axes = _figure(3, 3, height=11, spatial=True)
    # Background is masked, so it is not presented as a scientific cluster in the bar.
    cmap = ListedColormap(palette[1:])
    cmap.set_bad(palette[0])
    for identity in MAP_IDS:
        encoder, kmeans = identity
        axis = axes[encoder - 1, kmeans - 1]
        display = display_lookups[identity][maps[identity]]
        image = axis.imshow(np.ma.array(display, mask=~mask.astype(bool)), cmap=cmap,
                            norm=BoundaryNorm(np.arange(0.5, 9), 8),
                            interpolation="nearest", origin="upper")
        _spatial_axis(axis, mask)
    _sample_label(figure, sample_id)
    figure.colorbar(image, cax=figure.add_axes((0.22, 0.075, 0.56, 0.018)),
                    orientation="horizontal", ticks=range(1, 9), label="Cluster ID")
    paths.append(_save(figure, output / SAMPLE_FILES[0], dpi))

    frequencies = {window: boundary_frequency([maps[key] for key in MAP_IDS], mask, window)
                   for window in WINDOWS}
    frequency_path = output / "boundary_frequency.npz"
    np.savez_compressed(frequency_path, valid_mask=mask.astype(bool),
                        **{f"frequency_w{w}": values for w, values in frequencies.items()})
    paths.append(frequency_path)
    figure, axes = _figure(1, 3, height=4.6, spatial=True)
    for axis, window in zip(axes[0], WINDOWS, strict=True):
        image = axis.imshow(np.ma.masked_invalid(frequencies[window]), cmap="viridis",
                            vmin=0, vmax=1, origin="upper", interpolation="nearest")
        _spatial_axis(axis, mask)
    _sample_label(figure, sample_id)
    figure.colorbar(image, cax=figure.add_axes((0.25, 0.085, 0.5, 0.022)),
                    orientation="horizontal", label="Fraction of 9 maps")
    paths.append(_save(figure, output / SAMPLE_FILES[1], dpi))
    return paths


def render_summary(summary: dict[str, object], output: Path, *, dpi: int = 160) -> list[Path]:
    """Render saved per-sample values; all panel definitions and counts live in captions."""
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    figure, axes = _figure(2, 3, height=7.2, spatial=False)
    for row_index, group in enumerate(GROUPS):
        ari = _summary_values(summary, "ari", group)
        for column, window in enumerate(WINDOWS):
            axis = axes[row_index, column]
            lower = _summary_values(summary, f"boundary_{window}_lower", group)
            upper = _summary_values(summary, f"boundary_{window}_upper", group)
            common = sorted(set(ari) & set(lower) & set(upper))
            x = [ari[sample] for sample in common]
            axis.vlines(x, [lower[sample] for sample in common],
                        [upper[sample] for sample in common], color="0.65", linewidth=0.8)
            for values, color, label in ((lower, COLORS[0], "Smaller directional coverage"),
                                          (upper, COLORS[1], "Larger directional coverage")):
                axis.scatter(x, [values[sample] for sample in common], c=color, s=16, label=label)
            axis.set(xlim=(-1.02, 1.02), ylim=(-0.02, 1.02))
            if row_index == 1:
                axis.set_xlabel("ARI")
            if column == 0:
                axis.set_ylabel("Boundary coverage")
            axis.grid(alpha=0.2)
    _legend(figure, axes[0, 0], columns=2)
    paths.append(_save(figure, output / SUMMARY_FILES[0], dpi))
    return paths


def captions(summary: dict[str, object]) -> dict[str, object]:
    """Keep manuscript explanations outside the rasters, with actual sample counts."""
    panels = []
    for group in GROUPS:
        ari = _summary_values(summary, "ari", group)
        for window in WINDOWS:
            lower = _summary_values(summary, f"boundary_{window}_lower", group)
            upper = _summary_values(summary, f"boundary_{window}_upper", group)
            panels.append({"panel": chr(97 + len(panels)), "group": group, "window": window,
                           "radius_pixels": (window - 1) // 2,
                           "sample_count": len(set(ari) & set(lower) & set(upper))})
    counts = {metric: {group: len(_summary_values(summary, metric, group)) for group in GROUPS}
              for metric in ("ari", *(f"boundary_{w}_{side}" for w in WINDOWS
                                      for side in ("lower", "upper")))}
    return {
        "panel_order": "row-major; (a), (b), ...",
        "main_text": {
            "file": "summary/01_ari_vs_boundary_coverage.png", "panels": panels,
            "caption": "M11の反復間ARIと境界一致率。上段は同一encoder内のCosine-KMeans反復9対、"
                       "下段は異なるencoder間の27対。左から3×3、5×5、9×9近傍。"
                       "各試料についてARIを対平均し、境界一致率は対ごとの双方向の小さい側・"
                       "大きい側をそれぞれ平均した。線は同一試料の2値を結び、信頼区間ではない。"
                       "右ほど分割が一致し、上ほど境界が許容距離内に対応する。"
                       "対平均なので、特定の反復のずれは3×3マップと対別評価値で確認する。",
        },
        "sample_figure_templates": {
            "01_nine_maps.png": "行はencoder反復1–3、列はCosine-KMeans反復1–3。"
                                 "(a)が初回(1,1)。各反復を初回の元ラベルへ重なりで対応付け、"
                                 "既存の全体fit図の表示IDと配色へ変換した。白は背景。"
                                 "同じ行で初期化間の違いを、行をまたいで再学習を含む違いを確認する。"
                                 "色合わせは表示専用で、化学的対応を保証しない。",
            "03_boundary_frequency.png": "左から3×3、5×5、9×9近傍に境界が存在したマップの割合"
                                          "（全9マップ）。許容距離は1、2、4画素。"
                                          "1は全反復で近傍内に境界があり、0は全反復で近傍内に境界がない。"
                                          "中間値は反復によって境界の有無が変わる場所を表す。"
                                          "近傍幅を広げると頻度は下がらず、値の上昇だけで安定とは判断しない。"
                                          "共有encoderを含む記述的頻度であり、信頼度や化学的対応ではない。",
        },
        "sample_counts": counts,
        "appendix_selection": "既選定の代表7試料を候補とし、全49試料を保持する。"
                              "追加例には選定理由を明記し、安定例だけで代表させない。",
        "canonical_initial": "初回は元のglobal_v1のマップ。全体fit図が使う同seed再生成版へ"
                             "置換しない。両者の微小なラベル差はprotocol/initial_map_audit.jsonを参照。",
    }
