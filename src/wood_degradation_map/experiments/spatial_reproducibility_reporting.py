"""Complete-case summaries and spatially registered figures for nine M11 maps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .spatial_reproducibility_metrics import (
    WINDOWS,
    boundary_frequency,
    boundary_lattice,
    boundary_neighborhood,
)

MAP_IDS = tuple((encoder, kmeans) for encoder in (1, 2, 3) for kmeans in (1, 2, 3))
GROUPS = ("within_encoder", "between_encoder")
EXPECTED_PAIRS = {"within_encoder": 9, "between_encoder": 27}
BOUNDARY_METRICS = tuple(
    f"boundary_{window}_{side}" for window in WINDOWS for side in ("lower", "upper")
)
METRICS = ("ari", *BOUNDARY_METRICS)
GROUP_LABELS = {
    "within_encoder": "Within encoder (KMeans initializations)",
    "between_encoder": "Between encoders (training + KMeans)",
}
COLORS = ("#0072B2", "#D55E00", "#009E73")


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
        raise ValueError("Pair identities must contain integer encoder and KMeans repeats")
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


def _metric_values(
    pair: Mapping[str, object],
) -> tuple[dict[str, float | None], dict[str, str], dict[str, object]]:
    ari = _record(pair.get("ari"), "ARI")
    values = {"ari": _number(ari.get("value"), "ARI value")}
    reasons: dict[str, str] = {}
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
        group: [] for group in GROUPS
    }
    for pair in pairs:
        key = _pair_key(pair)
        if key in seen:
            raise ValueError(f"Duplicate pair in sample {sample_id}")
        seen.add(key)
        values, reasons, diagnostics = _metric_values(pair)
        by_group[cast(str, pair["group"])].append((pair, values, reasons, diagnostics))
    map_rows = _rows(evaluation.get("maps"), "Map diagnostics")
    map_seen: set[tuple[int, int]] = set()
    for row in map_rows:
        identity = (row.get("encoder_repeat"), row.get("kmeans_repeat"))
        if any(type(value) is not int for value in identity) or identity not in MAP_IDS:
            raise ValueError("Map diagnostics have an invalid encoder/KMeans identity")
        key = cast(tuple[int, int], identity)
        if key in map_seen:
            raise ValueError(f"Duplicate map diagnostics in sample {sample_id}")
        map_seen.add(key)
    groups: dict[str, object] = {}
    for group in GROUPS:
        rows = by_group[group]
        expected = EXPECTED_PAIRS[group]
        planned = [pair for pair in combinations(MAP_IDS, 2)
                   if (pair[0][0] == pair[1][0]) == (group == "within_encoder")]
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
    for group in GROUPS:
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
        for group in GROUPS:
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
        "schema_version": 1, "sample_ids": sorted(sample_ids), "sample_count": len(samples),
        "windows": list(WINDOWS), "samples": samples, "metrics": metrics,
        "aggregation": {
            "unit": "equal-weight sample; average over all planned pairs within each sample",
            "boundary_sample_base": "common across 3/5/9 windows and lower/upper directions",
            "missingness": "complete case per group and metric; partial values retained separately",
            "group_difference": "paired intersection of complete-case samples",
            "inference": "descriptive; no pair-level significance test or confidence interval",
        },
    }


def _figure(
    rows: int, columns: int, *, width: float = 12.0, height: float = 8.0,
) -> tuple[Figure, np.ndarray]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    return figure, axes


def _save(figure: Figure, path: Path, dpi: int) -> Path:
    figure.savefig(path, dpi=dpi, facecolor="white")
    figure.clear()
    return path


def _spatial_axis(axis: Axes, mask: np.ndarray) -> None:
    axis.imshow(np.where(mask, 0.94, 1.0), cmap="gray", vmin=0, vmax=1,
                interpolation="nearest", origin="upper")
    axis.set_xlim(-0.5, mask.shape[1] - 0.5)
    axis.set_ylim(mask.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_xlabel("Column (pixels)")
    axis.set_ylabel("Row (pixels)")


def _points(axis: Axes, lattice: np.ndarray, *, color: str, label: str,
            size: float = 3, alpha: float = 0.7) -> None:
    points = np.argwhere(lattice)
    axis.scatter(points[:, 1] / 2, points[:, 0] / 2, c=color, label=label,
                 s=size, alpha=alpha, linewidths=0, rasterized=True)


def _coverage_text(direction: Mapping[str, object]) -> str:
    value = direction["value"]
    if value is None:
        return f"undefined ({direction['undefined_reason']})"
    return (f"{cast(float, value):.3f} "
            f"({direction['matched_count']}/{direction['boundary_count']})")


def render_sample(
    sample_id: str, maps: Mapping[tuple[int, int], np.ndarray], mask: np.ndarray,
    evaluation: dict[str, object], output: Path, *, dpi: int = 160,
) -> list[Path]:
    """Save all nine maps, boundary locations, frequency arrays, and initial comparisons.

    The same image coordinates are used in every panel. Cluster colors are raw
    labels without cross-run alignment and do not imply chemical correspondence.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if set(maps) != set(MAP_IDS):
        raise ValueError("Sample figures and frequencies require exactly all nine maps")
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    if evaluation.get("sample_id") != sample_id:
        raise ValueError("Figure sample_id differs from its evaluation")
    # Validation is shared with the metrics; no mask intersection or label repair.
    lattices = {identity: boundary_lattice(maps[identity], mask) for identity in MAP_IDS}
    _sample_summary(evaluation)
    pair_lookup = {_pair_key(pair): pair for pair in _rows(evaluation.get("pairs"), "Pairs")}
    if len(pair_lookup) != 36:
        raise ValueError("Sample figures require all 36 evaluated pairs")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figure, axes = _figure(3, 3, height=11)
    palette = ListedColormap(["white", "#0072B2", "#E69F00", "#009E73", "#CC79A7",
                              "#56B4E9", "#D55E00", "#F0E442", "#666666"])
    image = None
    for (encoder, kmeans), labels in maps.items():
        axis = axes[encoder - 1, kmeans - 1]
        image = axis.imshow(labels, cmap=palette, norm=BoundaryNorm(np.arange(-0.5, 9), 9),
                            interpolation="nearest", origin="upper")
        axis.set_title(f"Encoder {encoder}, KMeans {kmeans}" +
                       (" (initial map)" if (encoder, kmeans) == (1, 1) else ""))
        axis.set_xlabel("Column (pixels)")
        axis.set_ylabel("Row (pixels)")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), ticks=range(9), shrink=0.6,
                        label="Raw cluster ID; 0 = excluded/background")
    figure.suptitle(f"{sample_id}: nine maps, raw label colors (not aligned across runs)")
    paths.append(_save(figure, output / "01_nine_maps.png", dpi))

    figure, axes = _figure(2, 3, height=8)
    for encoder in (1, 2, 3):
        axis = axes[0, encoder - 1]
        _spatial_axis(axis, mask)
        for kmeans, color in zip((1, 2, 3), COLORS, strict=True):
            _points(axis, lattices[encoder, kmeans], color=color, label=f"KMeans {kmeans}")
        if encoder == 1:
            _points(axis, lattices[1, 1], color="black", label="Initial (1,1)", size=1)
        axis.set_title(f"Within encoder {encoder}")
        axis.legend(fontsize=7, loc="upper right")
    for axis, (left, right) in zip(axes[1], combinations((1, 2, 3), 2), strict=True):
        _spatial_axis(axis, mask)
        for encoder in (left, right):
            union = np.logical_or.reduce([lattices[encoder, kmeans] for kmeans in (1, 2, 3)])
            _points(axis, union, color=COLORS[encoder - 1], label=f"Encoder {encoder}, all KMeans")
        if left == 1:
            _points(axis, lattices[1, 1], color="black", label="Initial (1,1)", size=1)
        axis.set_title(f"Between encoders {left} and {right}")
        axis.legend(fontsize=7, loc="upper right")
    figure.suptitle(f"{sample_id}: internal boundary midpoints; background edges excluded")
    paths.append(_save(figure, output / "02_boundary_overlays.png", dpi))

    frequencies = {
        window: boundary_frequency([maps[identity] for identity in MAP_IDS], mask, window)
        for window in WINDOWS
    }
    frequency_path = output / "boundary_frequency.npz"
    np.savez_compressed(frequency_path, valid_mask=mask.astype(bool),
                        **{f"frequency_w{window}": value for window, value in frequencies.items()})
    paths.append(frequency_path)
    figure, axes = _figure(1, 3, height=4.6)
    for axis, window in zip(axes[0], WINDOWS, strict=True):
        image = axis.imshow(np.ma.masked_invalid(frequencies[window]), cmap="viridis",
                            vmin=0, vmax=1, origin="upper", interpolation="nearest")
        axis.set_title(f"{window} x {window}; radius {(window - 1) // 2} pixels")
        axis.set_xlabel("Column (pixels)")
        axis.set_ylabel("Row (pixels)")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Fraction of 9 maps", shrink=0.8)
    figure.suptitle(f"{sample_id}: nearby boundary frequency (descriptive, not confidence)")
    paths.append(_save(figure, output / "03_boundary_frequency.png", dpi))

    for identity in MAP_IDS[1:]:
        pair = pair_lookup[((1, 1), identity)]
        results = {cast(int, row["window"]): row for row in _rows(pair["boundaries"], "Boundaries")}
        figure, axes = _figure(2, 3, height=8)
        for row_index, (source, target, direction) in enumerate(
            (((1, 1), identity, "a_to_b"), (identity, (1, 1), "b_to_a"))
        ):
            for axis, window in zip(axes[row_index], WINDOWS, strict=True):
                _spatial_axis(axis, mask)
                matched = lattices[source] & boundary_neighborhood(lattices[target], window)
                unmatched = lattices[source] & ~matched
                _points(axis, matched, color=COLORS[0], label="Matched source", size=4)
                _points(axis, unmatched, color="#CC0077", label="Unmatched source", size=5, alpha=1)
                _points(axis, lattices[target], color="black", label="Target", size=0.7, alpha=0.35)
                score = _coverage_text(_record(results[window][direction], "Direction"))
                axis.set_title(f"{source} -> {target}, w={window}\n{score}", fontsize=9)
                axis.legend(fontsize=6, loc="upper right")
        figure.suptitle(f"{sample_id}: initial map vs encoder {identity[0]}, KMeans {identity[1]}")
        paths.append(_save(figure, output / f"initial_vs_e{identity[0]}_k{identity[1]}.png", dpi))
    return paths


def _summary_values(summary: Mapping[str, object], metric: str, group: str) -> dict[str, float]:
    metrics = _record(summary["metrics"], "Summary metrics")
    group_row = _record(_record(metrics[metric], metric)[group], group)
    values = _record(group_row["sample_values"], "Sample values")
    return {key: cast(float, _number(value, "Sample value")) for key, value in values.items()}


def _ecdf(axis: Axes, values: Mapping[str, float], *, label: str, color: str) -> None:
    if values:
        ordered = np.sort(list(values.values()))
        axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
                  label=f"{label} (n={len(ordered)})", color=color)
        axis.scatter(ordered, np.arange(1, len(ordered) + 1) / len(ordered), s=7, color=color)
    else:
        axis.plot([], [], color=color, label=f"{label} (n=0, undefined)")


def render_summary(summary: dict[str, object], output: Path, *, dpi: int = 160) -> list[Path]:
    """Plot sample distributions and ARI/coverage correspondence without pair-level CI."""
    if type(dpi) is not int or dpi <= 0:
        raise ValueError("DPI must be a positive integer")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figure, axes = _figure(2, 3, height=7.5)
    for row_index, group in enumerate(GROUPS):
        ari = _summary_values(summary, "ari", group)
        for axis, window in zip(axes[row_index], WINDOWS, strict=True):
            lower = _summary_values(summary, f"boundary_{window}_lower", group)
            upper = _summary_values(summary, f"boundary_{window}_upper", group)
            common = sorted(set(ari) & set(lower) & set(upper))
            x = [ari[sample] for sample in common]
            axis.vlines(x, [lower[sample] for sample in common],
                        [upper[sample] for sample in common], color="0.65", linewidth=0.8)
            axis.scatter(x, [lower[sample] for sample in common], c=COLORS[0], s=16,
                         label="Mean smaller directional coverage")
            axis.scatter(x, [upper[sample] for sample in common], c=COLORS[1], s=16,
                         label="Mean larger directional coverage")
            axis.set_title(f"{GROUP_LABELS[group]}\nw={window}; common samples n={len(common)}",
                           fontsize=9)
            axis.set_xlabel("Mean pairwise ARI within sample")
            axis.set_ylabel("Mean directional extrema within sample")
            axis.set_xlim(-1.02, 1.02)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, loc="lower left")
    figure.suptitle("ARI and boundary coverage: each point is one equally weighted sample")
    paths.append(_save(figure, output / "01_ari_vs_boundary_coverage.png", dpi))

    figure, axes = _figure(2, 3, height=7.5)
    for row_index, side in enumerate(("lower", "upper")):
        for axis, window in zip(axes[row_index], WINDOWS, strict=True):
            for group, color in zip(GROUPS, COLORS[:2], strict=True):
                _ecdf(axis, _summary_values(summary, f"boundary_{window}_{side}", group),
                      label=GROUP_LABELS[group], color=color)
            axis.set_title(f"w={window}; {side} directional coverage")
            axis.set_xlabel("Mean over planned pairs within sample")
            axis.set_ylabel("Cumulative fraction of complete samples")
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(0, 1.02)
            axis.legend(fontsize=6, loc="upper left")
            axis.grid(alpha=0.2)
    figure.suptitle("All complete-case sample values; common sample base across boundary scales")
    paths.append(_save(figure, output / "02_boundary_sample_distributions.png", dpi))

    figure, axes = _figure(1, 1, width=7, height=4.8)
    axis = axes[0, 0]
    for group, color in zip(GROUPS, COLORS[:2], strict=True):
        _ecdf(axis, _summary_values(summary, "ari", group), label=GROUP_LABELS[group], color=color)
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Mean pairwise ARI within sample")
    axis.set_ylabel("Cumulative fraction of complete samples")
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(alpha=0.2)
    figure.suptitle("ARI sample distribution; no pair-level inference")
    paths.append(_save(figure, output / "03_ari_sample_distribution.png", dpi))
    return paths
