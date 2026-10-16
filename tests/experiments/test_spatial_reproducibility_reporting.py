"""Synthetic reporting contracts; these are not scientific acceptance thresholds."""

from __future__ import annotations

import csv
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from wood_degradation_map.experiments import spatial_reproducibility_reporting as reporting
from wood_degradation_map.experiments.global_k8_repeats import repeat_paths
from wood_degradation_map.experiments.manifests import _digest, _write_json
from wood_degradation_map.experiments.spatial_reproducibility_metrics import (
    WINDOWS, boundary_frequency, compare_maps, map_diagnostics,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PALETTE = ("#ffffff", "#0072b2", "#e69f00", "#009e73", "#cc79a7",
           "#56b4e9", "#d55e00", "#f0e442", "#332288")
GLOBAL_LOOKUP = np.array([0, 1, 2, 3, 4, 5, 7, 6, 8], dtype=np.uint8)


def _fixture(
    sample_id: str = "sample", *, empty_initial: bool = False,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, dict[str, object]]:
    mask = np.ones((5, 8), dtype=bool)
    mask[0, 0] = False
    labels = np.ones(mask.shape, dtype=np.uint8)
    labels[:, 4:] = 2
    labels[~mask] = 0
    maps = {identity: labels.copy() for identity in reporting.MAP_IDS}
    if empty_initial:
        maps[1, 1][mask] = 1
    rows = [{"encoder_repeat": encoder, "kmeans_repeat": kmeans,
             **map_diagnostics(maps[encoder, kmeans], mask)}
            for encoder, kmeans in reporting.MAP_IDS]
    pairs = []
    for left, right in combinations(reporting.MAP_IDS, 2):
        pairs.append({
            "encoder_a": left[0], "kmeans_a": left[1],
            "encoder_b": right[0], "kmeans_b": right[1],
            "group": "within_encoder" if left[0] == right[0] else "between_encoder",
            "initial_reference": left == (1, 1),
            **compare_maps(maps[left], maps[right], mask),
        })
    return maps, mask, {"sample_id": sample_id, "maps": rows, "pairs": pairs}


def _counts(
    maps: dict[tuple[int, int], np.ndarray], mask: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    return {identity: reporting.label_overlap_counts(maps[1, 1], maps[identity], mask)
            for identity in reporting.MAP_IDS}


def test_display_mapping_recovers_permutations_without_mutating_labels_or_metrics() -> None:
    reference = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8)
    permutation = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)
    maps = {identity: (reference.copy() if identity == (1, 1) else permutation[reference])
            for identity in reporting.MAP_IDS}
    mask = reference > 0
    before = {identity: labels.copy() for identity, labels in maps.items()}
    metrics = compare_maps(maps[1, 1], maps[1, 2], mask, include_overlap=True)
    counts = _counts(maps, mask)
    original_counts = {identity: table.copy() for identity, table in counts.items()}
    lookups, records = reporting.display_mappings(counts)
    assert len(records) == 9
    np.testing.assert_array_equal(lookups[1, 1], np.arange(9))
    np.testing.assert_array_equal(lookups[1, 2], permutation)
    for identity in reporting.MAP_IDS:
        np.testing.assert_array_equal(lookups[identity][maps[identity]], reference)
        np.testing.assert_array_equal(maps[identity], before[identity])
        np.testing.assert_array_equal(counts[identity], original_counts[identity])
    assert compare_maps(maps[1, 1], maps[1, 2], mask, include_overlap=True) == metrics


def test_display_assignment_uses_pooled_samples_even_if_a_local_assignment_differs() -> None:
    first = np.array([[1, 1, 1, 2, 2, 2]], dtype=np.uint8)
    second = np.array([[1, 2]], dtype=np.uint8)
    first_maps = {identity: first.copy() for identity in reporting.MAP_IDS}
    second_maps = {identity: second.copy() for identity in reporting.MAP_IDS}
    first_maps[1, 2] = 3 - first
    pooled = _counts(first_maps, first > 0)
    local = _counts(second_maps, second > 0)
    for identity in reporting.MAP_IDS:
        pooled[identity] += local[identity]
    lookups, records = reporting.display_mappings(pooled)
    local_lookups, _ = reporting.display_mappings(local)
    assert lookups[1, 2][1:3].tolist() == [2, 1]
    assert local_lookups[1, 2][1:3].tolist() == [1, 2]
    assert records[1]["valid_pixels"] == 8
    assert sum(row["overlap_pixels"] for row in records[1]["assignments"]) == 6


def test_split_and_absent_clusters_remain_a_permutation_with_unsupported_colors_recorded() -> None:
    reference = np.array([[1, 1, 1, 1, 2, 2]], dtype=np.uint8)
    split = np.array([[5, 5, 5, 6, 7, 7]], dtype=np.uint8)
    maps = {identity: reference.copy() for identity in reporting.MAP_IDS}
    maps[1, 2] = split
    lookups, records = reporting.display_mappings(_counts(maps, reference > 0))
    lookup = lookups[1, 2]
    assert lookup[0] == 0 and sorted(lookup[1:]) == list(range(1, 9))
    assert lookup[5] == 1 and lookup[7] == 2
    assert lookup[6] not in (1, 2)  # The split is not merged to make the picture agree.
    assert len(np.unique(lookup[split])) == 3
    unsupported = records[1]["assignments"][5]
    assert unsupported["target_pixels"] == 1 and unsupported["reference_pixels"] == 0
    assert unsupported["overlap_pixels"] == 0 and unsupported["zero_overlap"] is True
    assert all(row["zero_overlap"] for row in records[1]["assignments"]
               if row["target_pixels"] == 0)


@pytest.mark.parametrize("change", ["missing_run", "negative", "noninteger", "shape",
                                     "reference", "different_pixels"])
def test_invalid_alignment_counts_are_rejected(change: str) -> None:
    maps, mask, _ = _fixture()
    counts = _counts(maps, mask)
    if change == "missing_run":
        counts.pop((3, 3))
    elif change == "negative":
        counts[1, 2][0, 0] = -1
    elif change == "noninteger":
        counts[1, 2] = counts[1, 2].astype(float)
    elif change == "shape":
        counts[1, 2] = counts[1, 2][:, :-1]
    elif change == "reference":
        counts[1, 1][0, 1] = 1
    else:
        counts[1, 2][0, 0] += 1
    with pytest.raises(ValueError):
        reporting.display_mappings(counts)


def test_complete_pairs_and_equal_sample_macro_do_not_weight_pair_counts() -> None:
    _, _, first = _fixture("first")
    _, _, second = _fixture("second")
    for pair in second["pairs"]:
        pair["ari"]["value"] = 0.25 if pair["group"] == "within_encoder" else 0.75
    summary = reporting.summarize_samples([first, second])
    assert summary["sample_ids"] == ["first", "second"]
    assert summary["metrics"]["ari"]["within_encoder"]["macro_mean"] == 0.625
    assert summary["metrics"]["ari"]["between_encoder"]["macro_mean"] == 0.875
    assert summary["metrics"]["ari"]["group_difference"]["macro_mean"] == 0.25
    groups = summary["samples"][0]["groups"]
    assert groups["within_encoder"]["metrics"]["ari"]["defined_pairs"] == 9
    assert groups["between_encoder"]["metrics"]["ari"]["defined_pairs"] == 27
    assert len(summary["samples"][0]["initial_reference_pairs"]) == 8
    assert summary["metrics"]["boundary_3_lower"]["within_encoder"]["macro_mean"] == 1


def test_empty_source_excludes_entire_sample_group_and_preserves_directional_zero() -> None:
    _, _, complete = _fixture("complete")
    _, _, partial = _fixture("partial", empty_initial=True)
    summary = reporting.summarize_samples([complete, partial])
    row = summary["samples"][1]
    assert row["empty_boundary_map_count"] == 1
    within = row["groups"]["within_encoder"]
    assert within["metrics"]["boundary_3_lower"]["mean"] is None
    assert within["metrics"]["boundary_3_lower"]["defined_pairs"] == 7
    assert within["metrics"]["boundary_3_lower"]["partial_defined_pairs"]["mean"] == 1
    assert within["boundary_diagnostics"]["3"]["zero_directions"] == 2
    assert within["boundary_diagnostics"]["3"]["one_direction_undefined_pairs"] == 2
    between = row["groups"]["between_encoder"]
    assert between["metrics"]["boundary_3_lower"]["defined_pairs"] == 21
    assert between["boundary_diagnostics"]["3"]["zero_directions"] == 6
    for window in WINDOWS:
        for side in ("lower", "upper"):
            metric = summary["metrics"][f"boundary_{window}_{side}"]
            for group in reporting.GROUPS:
                assert metric[group]["common_sample_ids"] == ["complete"]
                assert metric[group]["excluded_sample_ids"] == ["partial"]
    assert summary["metrics"]["ari"]["within_encoder"]["sample_count"] == 2


def test_missing_pair_is_not_replaced_by_remaining_pair_mean() -> None:
    _, _, evaluation = _fixture()
    evaluation["pairs"].pop(0)
    summary = reporting.summarize_samples([evaluation])
    within = summary["samples"][0]["groups"]["within_encoder"]
    assert within["observed_pairs"] == 8
    assert len(within["missing_pairs"]) == 1
    assert within["metrics"]["ari"]["mean"] is None
    assert within["metrics"]["ari"]["partial_defined_pairs"]["mean"] == 1
    assert within["metrics"]["ari"]["undefined_pair_reasons"] == {"missing_pair": 1}
    assert summary["metrics"]["ari"]["within_encoder"]["sample_count"] == 0
    assert summary["metrics"]["ari"]["within_encoder"]["macro_mean"] is None
    assert summary["metrics"]["ari"]["between_encoder"]["sample_count"] == 1
    assert summary["metrics"]["ari"]["group_difference"]["sample_count"] == 0


def test_boundary_scales_share_a_sample_base_even_with_scale_specific_missingness() -> None:
    _, _, complete = _fixture("complete")
    _, _, partial = _fixture("partial")
    boundary = partial["pairs"][0]["boundaries"][0]
    boundary["a_to_b"]["value"] = None
    boundary["a_to_b"]["undefined_reason"] = "fixture_missing_value"
    boundary["lower"] = boundary["upper"] = None
    summary = reporting.summarize_samples([complete, partial])
    for window in WINDOWS:
        for side in ("lower", "upper"):
            metric = summary["metrics"][f"boundary_{window}_{side}"]
            assert metric["within_encoder"]["common_sample_ids"] == ["complete"]
            assert metric["between_encoder"]["common_sample_ids"] == ["complete", "partial"]
            assert metric["group_difference"]["common_sample_ids"] == ["complete"]


def test_empty_summary_is_explicitly_undefined() -> None:
    summary = reporting.summarize_samples([])
    assert summary["samples"] == []
    for metric in reporting.METRICS:
        for group in (*reporting.GROUPS, "group_difference"):
            result = summary["metrics"][metric][group]
            assert result["sample_count"] == 0
            assert result["macro_mean"] is None
            assert result["undefined_reason"] == "no_complete_case_samples"


def test_vi_components_distinguish_split_and_merge_and_sum_in_bits() -> None:
    coarse = np.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=np.uint8)
    fine = np.array([[3, 3, 4, 4, 5, 5, 5, 5]], dtype=np.uint8)
    split = reporting.vi_values(compare_maps(coarse, fine, coarse > 0, include_overlap=True))
    merge = reporting.vi_values(compare_maps(fine, coarse, coarse > 0, include_overlap=True))
    assert split == {"h_b_given_a_bits": 0.5, "h_a_given_b_bits": 0.0, "vi_bits": 0.5}
    assert merge == {"h_b_given_a_bits": 0.0, "h_a_given_b_bits": 0.5, "vi_bits": 0.5}


def test_vi_summaries_include_initial_reference_and_keep_samples_equally_weighted() -> None:
    _, _, first = _fixture("first")
    _, _, second = _fixture("second")
    for evaluation, split, merge in ((first, 0.25, 0.75), (second, 0.5, 1.0)):
        for pair in evaluation["pairs"]:
            pair["overlap"] = {"h_b_given_a_bits": split, "h_a_given_b_bits": merge}
    summary = reporting.summarize_samples([first, second])
    for group, expected in (("within_encoder", 9), ("between_encoder", 27), ("initial_reference", 8)):
        assert summary["metrics"]["h_b_given_a_bits"][group]["macro_mean"] == 0.375
        assert summary["metrics"]["h_a_given_b_bits"][group]["macro_mean"] == 0.875
        assert summary["metrics"]["vi_bits"][group]["macro_mean"] == 1.25
        assert summary["metrics"]["vi_bits"][group]["expected_pairs_per_sample"] == expected
    samples, macro = reporting.summary_table_rows(summary)
    assert len(samples) == 2 * 3 * len(reporting.METRICS)
    assert len(macro) == 3 * len(reporting.METRICS)
    initial = next(row for row in macro if row["group"] == "initial_reference" and row["metric"] == "vi_bits")
    assert initial["unit"] == "bits" and initial["macro_mean"] == 1.25
    second["pairs"][0].pop("overlap")
    partial = reporting.summarize_samples([first, second])
    for metric in reporting.VI_METRICS:
        assert partial["metrics"][metric]["within_encoder"]["common_sample_ids"] == ["first"]
        assert partial["metrics"][metric]["initial_reference"]["common_sample_ids"] == ["first"]
        assert partial["metrics"][metric]["between_encoder"]["sample_count"] == 2
    assert partial["metrics"]["ari"]["within_encoder"]["sample_count"] == 2


def test_unrequested_vi_stays_undefined_in_tables() -> None:
    _, _, evaluation = _fixture()
    summary = reporting.summarize_samples([evaluation])
    sample_rows, macro_rows = reporting.summary_table_rows(summary)
    for row in sample_rows:
        if row["metric"] in reporting.VI_METRICS:
            assert row["mean"] is None and row["defined_pairs"] == 0
            assert "overlap_not_requested" in row["undefined_pair_reasons"]
    assert all(row["macro_mean"] is None for row in macro_rows if row["metric"] in reporting.VI_METRICS)


@pytest.mark.parametrize("bad", [None, -0.1, float("nan"), float("inf"), True])
def test_invalid_vi_is_not_silently_treated_as_missing(bad: object) -> None:
    with pytest.raises(ValueError):
        reporting.vi_values({"overlap": {"h_b_given_a_bits": bad, "h_a_given_b_bits": 0.2}})


@pytest.mark.parametrize("failure", ["duplicate_pair", "duplicate_sample", "wrong_group",
                                     "wrong_initial", "inconsistent_extrema", "missing_window"])
def test_malformed_evaluations_are_rejected(failure: str) -> None:
    _, _, evaluation = _fixture()
    evaluations = [evaluation]
    if failure == "duplicate_pair":
        evaluation["pairs"].append(deepcopy(evaluation["pairs"][0]))
    elif failure == "duplicate_sample":
        evaluations.append(deepcopy(evaluation))
    elif failure == "wrong_group":
        evaluation["pairs"][0]["group"] = "between_encoder"
    elif failure == "wrong_initial":
        evaluation["pairs"][0]["initial_reference"] = False
    elif failure == "inconsistent_extrema":
        evaluation["pairs"][0]["boundaries"][0]["lower"] = 0.25
    else:
        evaluation["pairs"][0]["boundaries"].pop()
    with pytest.raises(ValueError):
        reporting.summarize_samples(evaluations)


def _publication_fixture() -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, dict[str, object]]:
    labels = np.tile(np.arange(9, dtype=np.uint8), (4, 1))
    mask = labels > 0
    reverse = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)
    maps = {key: (labels.copy() if key == (1, 1) else reverse[labels])
            for key in reporting.MAP_IDS}
    evaluation = {
        "sample_id": "KYOw02752",
        "maps": [{"encoder_repeat": e, "kmeans_repeat": k,
                  **map_diagnostics(maps[e, k], mask)} for e, k in reporting.MAP_IDS],
        "pairs": [{"encoder_a": a[0], "kmeans_a": a[1], "encoder_b": b[0], "kmeans_b": b[1],
                   "group": "within_encoder" if a[0] == b[0] else "between_encoder",
                   "initial_reference": a == (1, 1), **compare_maps(maps[a], maps[b], mask)}
                  for a, b in combinations(reporting.MAP_IDS, 2)],
    }
    return maps, mask, evaluation


def _global_reference(source: Path, *, duplicate: bool = False) -> Path:
    root = source / "results/figures/global_k8_v1"
    (root / "M11").mkdir(parents=True)
    regenerated, _ = repeat_paths(source, "M11", 1)
    regenerated.mkdir(parents=True)
    _write_json(regenerated / "completion.json", {"fixture": "regenerated initial map"})
    _write_json(root / "report.json", {
        "sample_ids": ["KYOw02752"], "definitions": {"palette": list(PALETTE)},
        "contract": {"K": 8, "repeat": 1, "reference": "M00+Cosine-KMeans",
                     "sources": {"M11": _digest(regenerated / "completion.json")}},
    })
    with (root / "M11/matching.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "original_cluster", "display_cluster"])
        writer.writerows(["M11", raw, int(GLOBAL_LOOKUP[raw]) if not duplicate else 1]
                         for raw in range(1, 9))
    _write_json(root / "completion.json", {
        "status": "global_report_completed", "artifact_sha256": {
            name: _digest(root / name) for name in ("report.json", "M11/matching.csv")},
    })
    return root


def test_global_display_uses_saved_assignment_and_palette_and_rejects_changed_source(
    tmp_path: Path,
) -> None:
    root = _global_reference(tmp_path)
    reference = reporting.load_global_display(tmp_path, ("KYOw02752",))
    np.testing.assert_array_equal(reference.lookup, GLOBAL_LOOKUP)
    assert reference.palette == PALETTE
    with pytest.raises(ValueError, match="different sample set"):
        reporting.load_global_display(tmp_path, ("another",))
    (root / "M11/matching.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="display source changed"):
        reporting.load_global_display(tmp_path, ("KYOw02752",))


def test_global_display_rejects_many_to_one_assignment(tmp_path: Path) -> None:
    _global_reference(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="one-to-one"):
        reporting.load_global_display(tmp_path, ("KYOw02752",))


def test_composition_preserves_every_cluster_and_both_input_lookups() -> None:
    maps, mask, _ = _publication_fixture()
    raw, _ = reporting.display_mappings({key: reporting.label_overlap_counts(maps[1, 1], labels, mask)
                                        for key, labels in maps.items()})
    before = {key: lookup.copy() for key, lookup in raw.items()}
    global_before = GLOBAL_LOOKUP.copy()
    composed = reporting.compose_display_mappings(raw, GLOBAL_LOOKUP)
    for key in reporting.MAP_IDS:
        np.testing.assert_array_equal(composed[key][maps[key]], GLOBAL_LOOKUP[maps[1, 1]])
        np.testing.assert_array_equal(raw[key], before[key])
        assert sorted(composed[key].tolist()) == list(range(9))
    np.testing.assert_array_equal(GLOBAL_LOOKUP, global_before)


def test_publication_rasters_have_no_titles_or_pixel_axes_and_keep_data_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.colors import to_rgba

    maps, mask, evaluation = _publication_fixture()
    maps_before, evaluation_before = deepcopy(maps), deepcopy(evaluation)
    raw, _ = reporting.display_mappings({key: reporting.label_overlap_counts(maps[1, 1], labels, mask)
                                        for key, labels in maps.items()})
    composed = reporting.compose_display_mappings(raw, GLOBAL_LOOKUP)
    saved = reporting._save
    inspected = []

    def capture(figure: Figure, path: Path, dpi: int) -> Path:
        assert figure._suptitle is None
        assert all(not axis.get_title() for axis in figure.axes)
        if path.parent.name == "sample":
            count = {"01_nine_maps.png": 9, "03_boundary_frequency.png": 3}[path.name]
            spatial_axes = figure.axes[:count]
            assert all(not axis.axison and not axis.get_xlabel() and not axis.get_ylabel()
                       for axis in spatial_axes)
            footer = [text for text in figure.texts if text.get_text() == "KYOw02752"]
            assert len(footer) == 1
            assert footer[0].get_position()[1] < min(axis.get_position().y0 for axis in spatial_axes)
            if path.name == "01_nine_maps.png":
                for axis in spatial_axes:
                    image = axis.images[0]
                    np.testing.assert_array_equal(image.get_array().data, GLOBAL_LOOKUP[maps[1, 1]])
                    np.testing.assert_allclose(image.to_rgba(np.array([6, 7, 8])),
                                               [to_rgba(PALETTE[k]) for k in (6, 7, 8)])
                    assert image.get_interpolation() == "nearest"
                assert figure.axes[-1].get_xticks().tolist() == list(range(1, 9))
        if path.name == "01_ari_vs_boundary_coverage.png":
            for axis in figure.axes:
                # One specimen, unchanged ARI and boundary values, both exactly one.
                for points in axis.collections[1:]:
                    np.testing.assert_allclose(points.get_offsets(), [[1, 1]])
        inspected.append(path.name)
        return saved(figure, path, dpi)

    monkeypatch.setattr(reporting, "_save", capture)
    paths = reporting.render_sample("KYOw02752", maps, mask, evaluation, tmp_path / "sample",
                                    display_lookups=composed, palette=PALETTE, dpi=30)
    summary = reporting.summarize_samples([evaluation])
    paths += reporting.render_summary(summary, tmp_path / "summary", dpi=30)
    assert sorted(inspected) == sorted([
        "01_nine_maps.png", "03_boundary_frequency.png", "01_ari_vs_boundary_coverage.png",
    ])
    assert {path.name for path in (tmp_path / "sample").iterdir()} == {
        "01_nine_maps.png", "03_boundary_frequency.png", "boundary_frequency.npz",
    }
    assert {path.name for path in (tmp_path / "summary").iterdir()} == {
        "01_ari_vs_boundary_coverage.png",
    }
    assert len(paths) == 4 and all(path.is_file() for path in paths)
    for path in paths:
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    with np.load(tmp_path / "sample/boundary_frequency.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["valid_mask"], mask)
        for window in WINDOWS:
            np.testing.assert_allclose(payload[f"frequency_w{window}"],
                                       boundary_frequency(list(maps.values()), mask, window),
                                       equal_nan=True)
    assert evaluation == evaluation_before
    for key in maps:
        np.testing.assert_array_equal(maps[key], maps_before[key])
    notes = reporting.captions(summary)
    assert notes["main_text"]["file"] == "summary/01_ari_vs_boundary_coverage.png"
    assert all(panel["sample_count"] == 1 for panel in notes["main_text"]["panels"])
    assert set(notes["sample_figure_templates"]) == {"01_nine_maps.png", "03_boundary_frequency.png"}
    assert "Cosine-KMeans" in notes["sample_figure_templates"]["01_nine_maps.png"]


@pytest.mark.parametrize("change", ["missing_run", "background", "merge", "missing_map", "invalid_label"])
def test_invalid_display_does_not_create_output(tmp_path: Path, change: str) -> None:
    maps, mask, evaluation = _publication_fixture()
    lookups = {key: GLOBAL_LOOKUP.copy() for key in maps}
    if change == "missing_run":
        lookups.pop((3, 3))
    elif change == "background":
        lookups[1, 1][0] = 1
    elif change == "merge":
        lookups[1, 1][7] = lookups[1, 1][6]
    elif change == "missing_map":
        maps.pop((3, 3))
    else:
        maps[3, 3][0, 1] = 0
    output = tmp_path / "figures"
    with pytest.raises(ValueError):
        reporting.render_sample("KYOw02752", maps, mask, evaluation, output,
                                display_lookups=lookups, palette=PALETTE)
    assert not output.exists()
