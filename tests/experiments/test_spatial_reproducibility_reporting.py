"""Synthetic reporting contracts; these are not scientific acceptance thresholds."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wood_degradation_map.experiments import spatial_reproducibility_reporting as reporting
from wood_degradation_map.experiments.spatial_reproducibility_metrics import (
    WINDOWS, compare_maps, map_diagnostics,
)


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


def test_figures_preserve_frequency_values_and_all_initial_comparisons(tmp_path: Path) -> None:
    maps, mask, evaluation = _fixture()
    before = {identity: labels.copy() for identity, labels in maps.items()}
    paths = reporting.render_sample("sample", maps, mask, evaluation, tmp_path / "sample", dpi=40)
    assert len(paths) == 12
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
