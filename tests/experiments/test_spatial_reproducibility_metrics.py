"""Small coordinate oracles for the spatial reproducibility definitions."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from wood_degradation_map.experiments.spatial_reproducibility_metrics import (
    WINDOWS,
    boundary_frequency,
    boundary_lattice,
    boundary_neighborhood,
    compare_maps,
    map_diagnostics,
    validate_map,
)


def _midpoints(labels: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Independent edge enumeration in original, not doubled, coordinates."""
    positions = []
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if not mask[row, col]:
                continue
            for dy, dx in ((0, 1), (1, 0)):
                other = row + dy, col + dx
                if (other[0] < labels.shape[0] and other[1] < labels.shape[1]
                        and mask[other] and labels[row, col] != labels[other]):
                    positions.append((row + dy / 2, col + dx / 2))
    return positions


def _within(point: tuple[float, float], targets: list[tuple[float, float]],
            radius: float) -> bool:
    return any(max(abs(point[0] - other[0]), abs(point[1] - other[1])) <= radius
               for other in targets)


def test_midpoints_are_unique_and_exclude_background_holes_and_exterior() -> None:
    labels = np.array([[1, 2, 0], [1, 1, 3]], dtype=np.uint8)
    mask = labels > 0
    boundary = boundary_lattice(labels, mask)
    assert boundary.shape == (3, 5)
    np.testing.assert_array_equal(np.argwhere(boundary), [[0, 1], [1, 2], [2, 3]])
    result = map_diagnostics(labels, mask)
    assert result == {
        "valid_pixels": 5, "valid_edges": 5, "boundary_count": 3,
        "boundary_density": 0.6, "boundary_density_undefined_reason": None,
        "cluster_counts": [3, 1, 1, 0, 0, 0, 0, 0],
        "occupancy": [0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        "used_clusters": 3, "maximum_occupancy": 0.6,
    }


def test_widths_are_one_two_four_pixel_radii_with_inclusive_edges() -> None:
    left = np.array([[1, 2, 2, 2, 2, 2, 2, 2]])
    right = np.array([[1, 1, 1, 2, 2, 2, 2, 2]])
    result = compare_maps(left, right, np.ones_like(left, dtype=bool))
    assert [row["radius_pixels"] for row in result["boundaries"]] == [1, 2, 4]
    assert [row["a_to_b"]["value"] for row in result["boundaries"]] == [0.0, 1.0, 1.0]
    boundary = boundary_lattice(left, left > 0)
    near = boundary_neighborhood(boundary, 3)
    assert near[0, 3]  # 1.5 is exactly one original pixel from the 0.5 midpoint.
    assert not near[0, 4]  # 2.0 is 1.5 pixels away.
    assert not near[0, -1]  # No image-edge wraparound.


def test_boundary_orientation_and_cluster_identity_do_not_constrain_matching() -> None:
    left = np.array([[1, 2], [1, 2]])
    right = np.array([[7, 7], [3, 3]])
    result = compare_maps(left, right, left > 0)
    assert result["ari"]["value"] == -0.5
    for row in result["boundaries"]:
        assert row["a_to_b"]["value"] == row["b_to_a"]["value"] == 1.0
    # Distinct source midpoints may share a single nearby target midpoint.
    more = np.array([[1, 2, 3]])
    fewer = np.array([[4, 4, 5]])
    match = compare_maps(more, fewer, more > 0)["boundaries"][0]
    assert match["a_to_b"]["matched_count"] == 2
    assert match["b_to_a"]["matched_count"] == 1
    assert match["lower"] == match["upper"] == 1.0


def test_split_retains_both_directions_and_conditional_entropies() -> None:
    left = np.array([[1] * 3 + [2] * 9])
    right = np.array([[5] * 3 + [6] * 5 + [7] * 4])
    result = compare_maps(left, right, left > 0, include_overlap=True)
    for row in result["boundaries"]:
        assert row["a_to_b"] == {
            "matched_count": 1, "boundary_count": 1, "value": 1.0,
            "undefined_reason": None,
        }
        assert row["b_to_a"] == {
            "matched_count": 1, "boundary_count": 2, "value": 0.5,
            "undefined_reason": None,
        }
        assert row["lower"] == 0.5 and row["upper"] == 1.0
    overlap = result["overlap"]
    assert overlap["h_b_given_a_bits"] > 0
    assert overlap["h_a_given_b_bits"] == 0.0
    assert overlap["counts"][1][5:7] == [5, 4]
    assert overlap["row_fractions"][1][5:7] == pytest.approx([5 / 9, 4 / 9])
    assert overlap["row_fractions"][2] == [None] * 8
    assert all(row[0] is None for row in overlap["column_fractions"])
    swapped = compare_maps(right, left, left > 0, include_overlap=True)
    assert swapped["overlap"]["h_b_given_a_bits"] == 0.0
    assert swapped["overlap"]["h_a_given_b_bits"] == overlap["h_b_given_a_bits"]
    assert "overlap" not in compare_maps(left, right, left > 0)


def test_empty_boundary_is_not_perfect_agreement_or_missing_prediction() -> None:
    constant = np.array([[1, 1, 1]])
    split = np.array([[1, 1, 2]])
    result = compare_maps(constant, split, constant > 0)
    assert result["ari"]["degeneracy_flags"] == ["a_single_cluster"]
    for row in result["boundaries"]:
        assert row["a_to_b"]["boundary_count"] == 0
        assert row["a_to_b"]["value"] is None
        assert row["a_to_b"]["undefined_reason"] == "empty_source_boundary"
        assert row["b_to_a"]["boundary_count"] == 1
        assert row["b_to_a"]["value"] == 0.0
        assert row["b_to_a"]["undefined_reason"] is None
        assert row["lower"] is row["upper"] is None
    both_empty = compare_maps(constant, constant * 8, constant > 0)
    assert both_empty["ari"]["value"] == 1.0
    assert all(row["a_to_b"]["value"] is row["b_to_a"]["value"] is None
               for row in both_empty["boundaries"])
    json.dumps(result, allow_nan=False)
    json.dumps(both_empty, allow_nan=False)


def test_no_valid_adjacent_edges_and_one_pixel_retain_explicit_reasons() -> None:
    labels = np.array([[1, 0], [0, 2]])
    diagnostics = map_diagnostics(labels, labels > 0)
    assert diagnostics["valid_edges"] == 0
    assert diagnostics["boundary_density"] is None
    assert diagnostics["boundary_density_undefined_reason"] == "no_valid_adjacent_edges"
    result = compare_maps(labels, labels, labels > 0)
    assert result["ari"]["value"] == 1.0
    assert result["ari"]["degeneracy_flags"] == ["a_all_singletons", "b_all_singletons"]
    for row in result["boundaries"]:
        assert row["a_to_b"]["undefined_reason"] == "no_valid_adjacent_edges"
        assert row["b_to_a"]["undefined_reason"] == "no_valid_adjacent_edges"
    one = np.array([[8]])
    single = compare_maps(one, one, one > 0, include_overlap=True)
    assert single["ari"]["value"] is None
    assert single["ari"]["undefined_reason"] == "fewer_than_two_valid_pixels"
    assert single["overlap"]["h_b_given_a_bits"] == 0.0
    assert boundary_lattice(one, one > 0).shape == (1, 1)
    json.dumps(single, allow_nan=False)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (2, 3), (5, 7), (9, 9)])
def test_directed_counts_match_independent_original_coordinate_oracle(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(831)
    mask = rng.random(shape) > 0.25
    mask[0, 0] = True
    left = rng.integers(1, 9, shape, dtype=np.uint8)
    right = rng.integers(1, 9, shape, dtype=np.uint8)
    left[~mask] = right[~mask] = 0
    a, b = _midpoints(left, mask), _midpoints(right, mask)
    actual = compare_maps(left, right, mask)
    np.testing.assert_array_equal(
        np.argwhere(boundary_lattice(left, mask)) / 2,
        np.asarray(sorted(a)).reshape(-1, 2),
    )
    for row in actual["boundaries"]:
        radius = (row["window"] - 1) / 2
        for key, source, target in (("a_to_b", a, b), ("b_to_a", b, a)):
            expected = sum(_within(point, target, radius) for point in source)
            assert row[key]["matched_count"] == expected
            assert row[key]["boundary_count"] == len(source)
            assert row[key]["value"] == (expected / len(source) if source else None)
    if np.count_nonzero(mask) >= 2:
        assert actual["ari"]["value"] == pytest.approx(
            adjusted_rand_score(left[mask], right[mask]), abs=1e-7,
        )


def test_permutation_transpose_padding_and_read_only_inputs_preserve_scores() -> None:
    left = np.array([[1, 0, 2, 1], [2, 1, 0, 1], [0, 2, 1, 0]])
    right = np.array([[2, 0, 2, 2], [1, 1, 0, 2], [0, 1, 1, 0]])
    mask = left > 0
    original = compare_maps(left, right, mask)
    before = left.copy(), right.copy(), mask.copy()
    left.flags.writeable = right.flags.writeable = mask.flags.writeable = False
    lookup = np.array([0, 8, 5])
    assert compare_maps(lookup[left], lookup[right], mask) == original
    assert compare_maps(left.T, right.T, mask.T) == original
    assert compare_maps(np.pad(left, 2), np.pad(right, 2), np.pad(mask, 2)) == original
    assert compare_maps(left, right, mask) == original
    for actual, expected in zip((left, right, mask), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_nine_map_frequency_uses_pixel_centers_and_counts_empty_maps_as_zero() -> None:
    split = np.array([[1, 1, 2, 2, 0]])
    constant = np.array([[7, 7, 7, 7, 0]])
    maps = [split] * 3 + [constant] * 6
    result = boundary_frequency(maps, split > 0, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, :4], [0, 1 / 3, 1 / 3, 0])
    assert np.isnan(result[0, 4])
    np.testing.assert_allclose(boundary_frequency(maps, split > 0, 5)[0, :4], 1 / 3)
    with pytest.raises(ValueError, match="Exactly nine"):
        boundary_frequency(maps[:-1], split > 0, 3)
    missing = split.copy()
    missing[0, 0] = 0
    with pytest.raises(ValueError, match="Missing predictions"):
        boundary_frequency([missing] + maps[1:], split > 0, 3)


@pytest.mark.parametrize("window", WINDOWS)
def test_frequency_matches_original_coordinate_oracle(window: int) -> None:
    rng = np.random.default_rng(93)
    mask = np.ones((4, 8), dtype=bool)
    mask[1, 2] = False
    maps = [rng.integers(1, 4, mask.shape, dtype=np.uint8) for _ in range(9)]
    for labels in maps:
        labels[~mask] = 0
    boundaries = [_midpoints(labels, mask) for labels in maps]
    actual = boundary_frequency(maps, mask, window)
    for point in np.argwhere(mask):
        row, col = (int(value) for value in point)
        expected = sum(_within((row, col), points, (window - 1) / 2)
                       for points in boundaries) / 9
        assert actual[row, col] == pytest.approx(expected, abs=1e-7)
    assert np.isnan(actual[1, 2])


@pytest.mark.parametrize("labels,mask,message", [
    (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=bool), "Empty"),
    (np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=bool), "Empty"),
    (np.array([[1.0]]), np.array([[1]]), "integer"),
    (np.array([[True]]), np.array([[True]]), "integer"),
    (np.array([1]), np.array([1]), "two-dimensional"),
    (np.array([[1]]), np.ones((2, 2), dtype=bool), "matching"),
    (np.array([[1]]), np.array([[1.0]]), "binary"),
    (np.array([[1]]), np.array([[2]]), "binary"),
    (np.array([[-1]]), np.array([[1]]), "0..8"),
    (np.array([[9]]), np.array([[1]]), "0..8"),
    (np.array([[2**64 - 1]], dtype=np.uint64), np.array([[1]]), "0..8"),
    (np.array([[1, 2]]), np.array([[1, 0]]), "Background"),
    (np.array([[1, 0]]), np.array([[1, 1]]), "Missing"),
])
def test_validation_rejects_changed_mask_and_invalid_values(
    labels: np.ndarray, mask: np.ndarray, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_map(labels, mask)


@pytest.mark.parametrize("window", [1, 2, 4, 7, 3.0, True])
def test_unplanned_windows_are_rejected(window: object) -> None:
    labels = np.array([[1, 2]])
    with pytest.raises(ValueError, match="fixed widths"):
        boundary_neighborhood(boundary_lattice(labels, labels > 0), window)


@pytest.mark.parametrize("boundary", [
    np.zeros((2, 3), dtype=bool), np.zeros((3, 3), dtype=np.uint8),
    np.zeros((0, 3), dtype=bool), np.ones((1, 1), dtype=bool),
    np.array([[False, False, False], [False, True, False], [False, False, False]]),
])
def test_boundary_neighborhood_rejects_nonmidpoint_lattices(boundary: np.ndarray) -> None:
    with pytest.raises(ValueError, match="lattice|midpoints"):
        boundary_neighborhood(boundary, 3)
