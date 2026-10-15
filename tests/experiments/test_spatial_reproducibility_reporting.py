"""Synthetic reporting contracts; these are not scientific acceptance thresholds."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from wood_degradation_map.experiments import spatial_reproducibility_reporting as reporting
from wood_degradation_map.experiments.spatial_reproducibility_metrics import (
    WINDOWS, compare_maps, map_diagnostics,
)


def _fixture(
    sample_id: str = "sample", *, empty_initial: bool = False, swap_target: bool = False,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, dict[str, object]]:
    mask = np.ones((5, 8), dtype=bool)
    mask[0, 0] = False
    labels = np.ones(mask.shape, dtype=np.uint8)
    labels[:, 4:] = 2
    labels[~mask] = 0
    maps = {identity: labels.copy() for identity in reporting.MAP_IDS}
    if empty_initial:
        maps[1, 1][mask] = 1
    if swap_target:
        maps[1, 2] = np.array([0, 2, 1], dtype=np.uint8)[maps[1, 2]]
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


def test_figures_preserve_frequency_values_and_all_initial_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps, mask, evaluation = _fixture(swap_target=True)
    before = {identity: labels.copy() for identity, labels in maps.items()}
    original_evaluation = deepcopy(evaluation)
    lookups, _ = reporting.display_mappings(_counts(maps, mask))
    drawn = []
    save = reporting._save

    def capture(figure: Figure, path: Path, dpi: int) -> Path:
        if path.name == "01_nine_maps.png":
            drawn.extend(np.asarray(axis.images[0].get_array()).copy() for axis in figure.axes[:9])
        return save(figure, path, dpi)

    monkeypatch.setattr(reporting, "_save", capture)
    paths = reporting.render_sample("sample", maps, mask, evaluation, tmp_path / "sample",
                                    dpi=40, display_lookups=lookups)
    assert len(paths) == 12
    assert len(drawn) == 9
    assert all(np.array_equal(labels, maps[1, 1]) for labels in drawn)
    assert evaluation == original_evaluation
    assert len(list((tmp_path / "sample").glob("initial_vs_*.png"))) == 8
    for path in paths:
        assert path.is_file()
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    with np.load(tmp_path / "sample/boundary_frequency.npz", allow_pickle=False) as payload:
        assert np.array_equal(payload["valid_mask"], mask)
        for window in WINDOWS:
            values = payload[f"frequency_w{window}"]
            assert values.shape == mask.shape
            assert np.isnan(values[~mask]).all()
            assert set(np.unique(values[mask])) <= {0, 1}
    assert all(np.array_equal(maps[identity], before[identity]) for identity in maps)
    summary_paths = reporting.render_summary(reporting.summarize_samples([evaluation]),
                                             tmp_path / "summary", dpi=40)
    assert len(summary_paths) == 3
    assert all(path.is_file() for path in summary_paths)


def test_rendering_rejects_missing_maps_before_writing(tmp_path: Path) -> None:
    maps, mask, evaluation = _fixture()
    maps.pop((3, 3))
    with pytest.raises(ValueError, match="exactly all nine"):
        reporting.render_sample("sample", maps, mask, evaluation, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("change", ["missing_run", "background", "merge", "initial"])
def test_rendering_rejects_invalid_display_permutations(tmp_path: Path, change: str) -> None:
    maps, mask, evaluation = _fixture()
    lookups, _ = reporting.display_mappings(_counts(maps, mask))
    if change == "missing_run":
        lookups.pop((3, 3))
    elif change == "background":
        lookups[1, 2][0] = 1
    elif change == "merge":
        lookups[1, 2][2] = lookups[1, 2][1]
    else:
        lookups[1, 1][1:3] = [2, 1]
    with pytest.raises(ValueError):
        reporting.render_sample("sample", maps, mask, evaluation, tmp_path / "figures",
                                display_lookups=lookups)
    assert not (tmp_path / "figures").exists()
