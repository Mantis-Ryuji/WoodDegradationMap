"""Small independent references for pooled silhouette and three-repeat ARI."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from wood_degradation_map.experiments import diagnostic_metrics as metrics
from wood_degradation_map.experiments.data import SpectrumInputError
from wood_degradation_map.experiments.lfr import PixelLabels

CPU = torch.device("cpu")


def _pixels(labels: list[int] | np.ndarray, sample_id: str = "KYOw02702") -> PixelLabels:
    values = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(values))
    return PixelLabels(sample_id, rows, np.column_stack((rows // 4, rows % 4)), values)


def _features(points: list[list[float]], labels: list[int], sample_id: str = "KYOw02702") -> metrics.SampleFeatures:
    values = np.zeros((len(points), 16), dtype=np.float32)
    values[:, :2] = points
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return metrics.SampleFeatures(_pixels(labels, sample_id), values)


def _silhouette(samples: tuple[metrics.SampleFeatures, ...], *, k: int = 4, chunk: int = 2) -> metrics.FoldSilhouette:
    return metrics.fold_silhouette(samples, expected_test_pixels={s.pixels.sample_id: len(s.values) for s in samples},
                                   condition_id="B1", fold=1, repeat=1, k=k, device=CPU,
                                   chunk_pixels=chunk)


def test_fold_is_pooled_once_and_macro_is_not_pixel_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _features([[0, 1]], [2], "KYOw02702")
    second = _features([[1, 0], [1, 0], [1, 0], [-1, 0], [0, 1]],
                       [1, 1, 1, 2, 2], "KYOw02707")
    reference = metrics.silhouette_samples_cosine_gpu
    calls = []

    def checked(values: np.ndarray, labels: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(len(values))
        assert kwargs["dtype"] == torch.float32 and kwargs["eps"] == 1e-12
        assert kwargs["device"] == "cpu" and kwargs["return_numpy"] is True
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert not torch.is_autocast_enabled("cpu")
        assert np.all(labels > 0)
        return reference(values, labels, **kwargs)

    monkeypatch.setattr(metrics, "silhouette_samples_cosine_gpu", checked)
    precision = torch.backends.cuda.matmul.fp32_precision
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = _silhouette((second, first))  # Caller ordering cannot change sample boundaries.
    assert torch.backends.cuda.matmul.fp32_precision == precision
    assert calls == [6] and result.pixel_scores.dtype == np.float32
    assert result.cluster_counts == (3, 3, 0, 0) and result.used_clusters == 2
    np.testing.assert_allclose(result.pixel_scores, [0.5, 1, 1, 1, 0.5, 0.5], atol=1e-7)
    assert [row.offset for row in result.samples] == [0, 1]
    assert [row.mean for row in result.samples] == pytest.approx([0.5, 0.8])
    assert result.macro_mean == pytest.approx(0.65)
    assert result.macro_mean != pytest.approx(float(result.pixel_scores.mean()))
    assert result.defined_samples == 2 and result.undefined_reason is None


def test_reference_agrees_with_pairwise_sklearn_and_chunk_changes() -> None:
    random = np.random.default_rng(192)
    values = random.standard_normal((12, 16), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    pixels = _pixels([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4])
    sample = metrics.SampleFeatures(pixels, values)
    expected = silhouette_samples(values, pixels.labels, metric="cosine")
    first = _silhouette((sample,), chunk=1)
    second = _silhouette((sample,), chunk=100)
    np.testing.assert_allclose(first.pixel_scores, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(first.pixel_scores, second.pixel_scores, atol=2e-6, rtol=2e-6)
    assert first.pixel_scores[-1] == 0.0 and first.samples[0].singleton_pixels == 1


def test_exact_zero_intra_and_inter_distances_return_zero() -> None:
    sample = _features([[1, 0]] * 4, [1, 1, 2, 2])
    result = _silhouette((sample,))
    np.testing.assert_array_equal(result.pixel_scores, np.zeros(4, dtype=np.float32))
    assert result.macro_mean == 0.0


def test_b0_uses_256_columns_and_preserves_inputs() -> None:
    small = _features([[1, 0], [1, 0], [0, 1], [-1, 0]], [1, 1, 2, 2])
    values = np.pad(small.values, ((0, 0), (0, 240)))
    original = values.copy()
    values.flags.writeable = False
    result = metrics.fold_silhouette(
        (metrics.SampleFeatures(small.pixels, values),), expected_test_pixels={"KYOw02702": 4},
        condition_id="B0", fold=1, repeat=1, k=2, device=CPU,
    )
    np.testing.assert_allclose(result.pixel_scores, [1, 1, 0, 0.5], atol=1e-7)
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError, match="256 columns"):
        metrics.fold_silhouette((small,), expected_test_pixels={"KYOw02702": 4},
                                condition_id="B0", fold=1, repeat=1, k=2, device=CPU)


@pytest.mark.parametrize("labels,reason", [([1, 1, 1], "single_cluster"),
                                           ([1, 2, 3], "all_singletons"), ([1], "single_cluster")])
def test_undefined_fold_never_calls_reference(
    labels: list[int], reason: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Undefined fold must not enter reference metric")

    monkeypatch.setattr(metrics, "silhouette_samples_cosine_gpu", forbidden)
    result = _silhouette((_features([[1, 0]] * len(labels), labels),))
    assert result.pixel_scores is None and result.macro_mean is None
    assert result.defined_samples == 0 and result.undefined_reason == reason
    assert result.samples[0].mean is None and result.samples[0].undefined_reason == reason


def test_missing_test_sample_or_row_is_rejected() -> None:
    sample = _features([[1, 0], [0, 1]], [1, 2])
    for expected in ({"KYOw02702": 2, "KYOw02707": 3}, {"KYOw02702": 3}):
        with pytest.raises(ValueError):
            metrics.fold_silhouette((sample,), expected_test_pixels=expected,
                                    condition_id="B1", fold=1, repeat=1, k=2, device=CPU)
    with pytest.raises(ValueError, match="every test sample"):
        _silhouette((sample, sample))


@pytest.mark.parametrize("value", [0.0, 1e-15, np.nan, np.inf, 1e30])
def test_invalid_representations_are_failures_even_for_undefined_fold(value: float) -> None:
    sample = _features([[1, 0], [0, 1]], [1, 1])
    sample.values[1] = value
    with pytest.raises(SpectrumInputError) as error:
        _silhouette((sample,))
    assert error.value.hdf5_rows == (1,)


@pytest.mark.parametrize("value", [np.nan, np.inf, 2.0, -2.0])
def test_numerically_invalid_reference_output_is_not_clipped(
    value: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _features([[1, 0], [1, 0], [0, 1]], [1, 1, 2])
    monkeypatch.setattr(metrics, "silhouette_samples_cosine_gpu",
                        lambda *args, **kwargs: np.array([0, value, 0], dtype=np.float32))
    with pytest.raises(SpectrumInputError) as error:
        _silhouette((sample,))
    assert error.value.hdf5_rows == (1,)


def test_ari_hand_counted_negative_and_label_permuted_partitions() -> None:
    predictions = {1: _pixels([1, 1, 2, 2]), 2: _pixels([1, 2, 1, 2]), 3: _pixels([4, 4, 3, 3])}
    result = metrics.repeat_ari(predictions, expected_pixel_count=4, condition_id="M11", fold=1, k=4)
    assert [pair.repeats for pair in result.pairs] == [(1, 2), (1, 3), (2, 3)]
    assert [pair.value for pair in result.pairs] == [-0.5, 1.0, -0.5]
    assert result.mean == 0.0
    assert result.pairs[0].contingency == ((1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert all(pair.degeneracy_flags == () for pair in result.pairs)


@pytest.mark.parametrize("seed", [3, 43, 191])
def test_ari_matches_sklearn(seed: int) -> None:
    random = np.random.default_rng(seed)
    predictions = {repeat: _pixels(random.integers(1, 5, size=23)) for repeat in (1, 2, 3)}
    result = metrics.repeat_ari(predictions, expected_pixel_count=23, condition_id="B0", fold=3, k=4)
    expected = []
    for pair in result.pairs:
        left, right = pair.repeats
        expected.append(adjusted_rand_score(predictions[left].labels, predictions[right].labels))
        assert pair.value == pytest.approx(expected[-1], abs=1e-7)
    assert result.mean == pytest.approx(np.mean(expected), abs=1e-7)


@pytest.mark.parametrize("labels,flag", [([2, 2, 2, 2], "single_cluster"),
                                        ([1, 2, 3, 4], "all_singletons")])
def test_degenerate_perfect_ari_has_flags(labels: list[int], flag: str) -> None:
    predictions = {repeat: _pixels(labels) for repeat in (1, 2, 3)}
    result = metrics.repeat_ari(predictions, expected_pixel_count=4, condition_id="B0", fold=1, k=4)
    assert result.mean == 1.0
    for pair in result.pairs:
        assert pair.value == 1.0 and pair.undefined_reason is None
        assert pair.degeneracy_flags == tuple(f"repeat_{repeat}_{flag}" for repeat in pair.repeats)


def test_one_pixel_ari_is_undefined_and_mixed_degeneracy_is_not_perfect() -> None:
    result = metrics.repeat_ari({repeat: _pixels([repeat]) for repeat in (1, 2, 3)},
                                expected_pixel_count=1, condition_id="B0", fold=1, k=4)
    assert result.mean is None and result.undefined_reason == "fewer_than_two_valid_pixels"
    assert all(pair.value is None for pair in result.pairs)
    result = metrics.repeat_ari({1: _pixels([1, 1, 1, 1]), 2: _pixels([1, 2, 3, 4]),
                                 3: _pixels([2, 2, 2, 2])},
                                expected_pixel_count=4, condition_id="B0", fold=1, k=4)
    assert [pair.value for pair in result.pairs] == [0, 1, 0]
    assert result.mean == pytest.approx(1 / 3)


def test_ari_integer_products_do_not_overflow_int64() -> None:
    # Only two 0.8 MiB label vectors; the intermediate pair-count products exceed int64.
    labels = np.repeat(np.array([1, 2]), 50_000)
    result = metrics._ari_pair(labels, 3 - labels, 2, (1, 2))
    assert result.value == 1.0


@pytest.mark.parametrize("problem", ["missing_repeat", "row", "coordinate", "duplicate_coordinate",
                                     "sample", "background", "float", "count"])
def test_ari_rejects_incomplete_or_misaligned_pixels(problem: str) -> None:
    predictions = {repeat: _pixels([1, 1, 2, 2]) for repeat in (1, 2, 3)}
    if problem == "missing_repeat":
        predictions.pop(3)
    else:
        changes = {
            "row": {"hdf5_rows": np.array([0, 1, 3, 2])},
            "coordinate": {"pixel_row_col": predictions[3].pixel_row_col[::-1]},
            "duplicate_coordinate": {"pixel_row_col": np.zeros((4, 2), dtype=np.int64)},
            "sample": {"sample_id": "KYOw02707"},
            "background": {"labels": np.array([1, 0, 2, 2])},
            "float": {"labels": np.array([1, 1, 2, 2], dtype=np.float32)},
            "count": {"labels": np.array([1, 1, 2])},
        }
        predictions[3] = replace(predictions[3], **changes[problem])
    with pytest.raises(ValueError):
        metrics.repeat_ari(predictions, expected_pixel_count=4, condition_id="B0", fold=1, k=2)
