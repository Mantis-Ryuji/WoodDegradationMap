"""Hand-counted maps and an independent pair oracle for sample-level LLA."""

from __future__ import annotations

import json
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from wood_degradation_map.experiments.spatial_metrics import _scores, local_label_agreement


def test_unequal_neighbor_degrees_use_pair_weighting_and_each_window() -> None:
    labels = np.array([[1, 1, 1, 2]], dtype=np.uint8)
    result = local_label_agreement(labels, np.ones_like(labels), k=4)
    assert result.valid_pixels == 4 and result.cluster_counts == (3, 1, 0, 0)
    assert result.occupancy == (0.75, 0.25, 0.0, 0.0)
    assert result.used_clusters == 2 and result.maximum_occupancy == 0.75
    assert result.expected_agreement == 0.5  # Without replacement, not sum(p_k^2)=0.625.
    for row, (window, matches, pairs, adjusted) in zip(result.windows, (
        (3, 4, 6, Fraction(1, 3)), (5, 6, 10, Fraction(1, 5)), (9, 6, 12, 0),
    ), strict=True):
        assert (row.window, row.matching_pairs, row.valid_pairs) == (window, matches, pairs)
        assert row.lla == pytest.approx(matches / pairs, abs=1e-7)
        assert row.adjusted_lla == pytest.approx(float(adjusted), abs=1e-7)
        assert row.pixels_with_neighbors == 4 and row.neighbor_pixel_fraction == 1.0
        assert row.lla_undefined_reason is None and row.adjusted_undefined_reasons == ()
    assert result.windows[0].lla != pytest.approx((1 + 1 + 0.5 + 0) / 4)


def test_alternating_labels_retain_negative_adjusted_score() -> None:
    labels = np.array([[1, 2, 1]], dtype=np.int32)
    result = local_label_agreement(labels, labels > 0, k=2)
    assert result.expected_agreement == pytest.approx(1 / 3)
    assert result.windows[0].matching_pairs == 0 and result.windows[0].valid_pairs == 4
    assert result.windows[0].lla == 0.0 and result.windows[0].adjusted_lla == -0.5
    assert result.windows[1].adjusted_lla == result.windows[2].adjusted_lla == 0.0


def test_diagonal_neighbors_count_and_background_and_center_do_not() -> None:
    labels = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = local_label_agreement(labels, labels > 0, k=2)
    for row in result.windows:
        assert row.valid_pairs == row.matching_pairs == 2
        assert row.lla == 1.0 and row.adjusted_lla is None
        assert row.adjusted_undefined_reasons == ("single_cluster",)
    assert result.expected_agreement == 1.0 and result.used_clusters == 1


def test_neighborhood_width_is_not_radius_and_image_edges_do_not_wrap() -> None:
    labels = np.array([[1, 0, 0, 2, 0, 0, 1]], dtype=np.uint8)
    result = local_label_agreement(labels, labels > 0, k=2)
    for row in result.windows[:2]:
        assert row.valid_pairs == 0 and row.lla is None and row.adjusted_lla is None
        assert row.neighbor_pixel_fraction == 0.0
        assert row.lla_undefined_reason == "no_valid_neighbor_pairs"
        assert row.adjusted_undefined_reasons == ("no_valid_neighbor_pairs",)
    assert result.windows[2].valid_pairs == 4 and result.windows[2].matching_pairs == 0
    assert result.windows[2].adjusted_lla == -0.5
    endpoints = np.array([[1, 0, 0, 0, 0, 1]], dtype=np.int32)
    assert all(row.valid_pairs == 0 for row in local_label_agreement(
        endpoints, endpoints > 0, k=2,
    ).windows)


def test_isolated_pixels_remain_in_occupancy_and_coverage_denominator() -> None:
    labels = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    result = local_label_agreement(labels, labels > 0, k=2)
    assert result.cluster_counts == (2, 1) and result.expected_agreement == pytest.approx(1 / 3)
    for row in result.windows:
        assert row.valid_pairs == 2 and row.matching_pairs == 0
        assert row.pixels_with_neighbors == 2
        assert row.neighbor_pixel_fraction == pytest.approx(2 / 3)
        assert row.adjusted_lla == -0.5


def test_one_valid_pixel_is_undefined_and_json_serializes_without_nan() -> None:
    result = local_label_agreement(np.array([[2]]), np.array([[True]]), k=2)
    assert result.expected_agreement is None and result.cluster_counts == (0, 1)
    for row in result.windows:
        assert row.lla is row.adjusted_lla is None
        assert row.matching_pairs == row.valid_pairs == row.pixels_with_neighbors == 0
        assert row.adjusted_undefined_reasons == (
            "fewer_than_two_valid_pixels", "no_valid_neighbor_pairs", "single_cluster",
        )
    restored = json.loads(json.dumps(asdict(result), allow_nan=False))
    assert restored["windows"][0]["lla"] is None
    assert restored["windows"][0]["adjusted_lla"] is None


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (2, 3), (6, 10), (9, 9)])
def test_exact_counts_match_independent_pixel_pair_oracle(shape: tuple[int, int]) -> None:
    random = np.random.default_rng(78)
    mask = random.random(shape) > 0.3
    mask[0, 0] = True
    labels = random.integers(1, 5, size=shape, dtype=np.int32)
    labels[~mask] = 0
    result = local_label_agreement(labels, mask, k=4)
    coordinates = [tuple(point) for point in np.argwhere(mask)]
    n = len(coordinates)
    counts = [sum(labels[point] == label for point in coordinates) for label in range(1, 5)]
    expected = (Fraction(sum(int(count) * (int(count) - 1) for count in counts), n * (n - 1))
                if n > 1 else None)
    for row in result.windows:
        pairs = [(p, q) for p in coordinates for q in coordinates if p != q
                 and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= row.window // 2]
        matching = sum(labels[p] == labels[q] for p, q in pairs)
        assert row.valid_pairs == len(pairs) and row.matching_pairs == matching
        assert row.pixels_with_neighbors == len({p for p, _ in pairs})
        if pairs:
            exact = Fraction(int(matching), len(pairs))
            assert row.lla == pytest.approx(float(exact), abs=1e-7)
            if expected != 1:
                assert row.adjusted_lla == pytest.approx(
                    float((exact - expected) / (1 - expected)), abs=1e-7,
                )
        else:
            assert row.lla is None


def test_label_permutation_padding_transpose_and_read_only_inputs() -> None:
    labels = np.array([[1, 0, 2, 1], [2, 1, 0, 1], [0, 2, 1, 0]], dtype=np.uint8)
    mask = labels > 0
    before = labels.copy()
    labels.flags.writeable = False
    mask.flags.writeable = False
    original = local_label_agreement(labels, mask, k=4)
    relabeled = np.array([0, 4, 3, 2, 1], dtype=np.uint8)[labels]
    permuted = local_label_agreement(relabeled, mask, k=4)
    assert original.windows == permuted.windows
    assert permuted.cluster_counts == original.cluster_counts[::-1]
    assert local_label_agreement(labels.T, mask.T, k=4) == original
    assert local_label_agreement(np.pad(labels, 5), np.pad(mask, 5), k=4) == original
    np.testing.assert_array_equal(labels, before)


def test_near_one_expected_agreement_does_not_hide_multicluster_sample() -> None:
    # Count-only stress case, with no huge array. FP32 rounds P to 1, although
    # one of the 2^28 valid pixels belongs to a second cluster.
    n = 2**28
    counts = (n - 1, 1)
    assert np.float32((n - 1) * (n - 2)) / np.float32(n * (n - 1)) == np.float32(1)
    lla, adjusted, reasons = _scores(100, 100, counts)
    assert lla == adjusted == 1.0 and reasons == ()
    _, adjusted, reasons = _scores(0, 2, counts)
    exact = -Fraction((n - 1) * (n - 2), 2 * (n - 1))
    assert adjusted == pytest.approx(float(exact), rel=1e-7) and reasons == ()


@pytest.mark.parametrize("labels,mask,message", [
    (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=bool), "Empty"),
    (np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=bool), "Empty"),
    (np.array([[1.0]]), np.array([[1]]), "integer"),
    (np.array([[np.nan]]), np.array([[1]]), "integer"),
    (np.array([[True]]), np.array([[True]]), "integer"),
    (np.array([1]), np.array([1]), "two-dimensional"),
    (np.array([[1]]), np.ones((2, 2), dtype=bool), "matching"),
    (np.array([[1]]), np.array([[1.0]]), "binary"),
    (np.array([[1]]), np.array([[2]]), "binary"),
    (np.array([[1]]), np.array([[-1]]), "binary"),
    (np.array([[-1]]), np.array([[1]]), "0..K"),
    (np.array([[3]]), np.array([[1]]), "0..K"),
    (np.array([[2**64 - 1]], dtype=np.uint64), np.array([[1]]), "0..K"),
    (np.array([[1, 2]]), np.array([[1, 0]]), "Background"),
    (np.array([[1, 0]]), np.array([[1, 1]]), "Missing"),
])
def test_invalid_inputs_fail_instead_of_dropping_pixels(
    labels: np.ndarray, mask: np.ndarray, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        local_label_agreement(labels, mask, k=2)


@pytest.mark.parametrize("k", [1, 3, 16, 2.0, True])
def test_unplanned_or_noninteger_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="fixed experiment plan"):
        local_label_agreement(np.array([[1, 2]]), np.array([[1, 1]]), k=k)
