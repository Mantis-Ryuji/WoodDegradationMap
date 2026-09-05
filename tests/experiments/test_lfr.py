"""CPU fixtures for shared evaluation inputs, aligned flips and fixed predictors."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import perturbations
from wood_degradation_map.experiments.baselines import B0Baseline, NormalizedRepresentation
from wood_degradation_map.experiments.cluster_pipeline import NeuralRepresentation
from wood_degradation_map.experiments.clustering import TrainFeatures, fit_clusters
from wood_degradation_map.experiments.config import perturbation_seed
from wood_degradation_map.experiments.data import SpectrumBatch, SpectrumInputError
from wood_degradation_map.experiments.lfr import (
    LFRAccumulator, PerturbedLabels, PixelLabels, accumulate_lfr_block,
)
from wood_degradation_map.experiments.neural import TorchRandomStream, fp32_inference
from wood_degradation_map.experiments.perturbations import DRAW_KEYS, SharedPerturbations

CPU = torch.device("cpu")
SAMPLE = "KYOw02702"


def _spectra(n: int) -> np.ndarray:
    values = np.random.default_rng(316).standard_normal((n, 256), dtype=np.float32)
    values -= values.mean(axis=1, keepdims=True)
    values /= values.std(axis=1, ddof=1, keepdims=True)
    return values


def _batches(values: np.ndarray, size: int) -> list[SpectrumBatch]:
    rows = np.arange(len(values), dtype=np.int64)
    coordinates = np.column_stack((rows // 5, rows % 5))
    return [SpectrumBatch(SAMPLE, rows[start:start + size], coordinates[start:start + size],
                          values[start:start + size]) for start in range(0, len(values), size)]


def _labels(values: list[int], start: int = 0) -> PixelLabels:
    rows = np.arange(start, start + len(values))
    return PixelLabels(SAMPLE, rows, np.column_stack((rows // 5, rows % 5)), np.array(values))


def _draws(pixels: PixelLabels) -> tuple[PerturbedLabels, ...]:
    return tuple(PerturbedLabels(kind, draw, pixels) for kind, draw in DRAW_KEYS)


def test_production_generation_width_and_probability_settings() -> None:
    assert perturbations.GENERATION_BATCH_PIXELS == 1024
    for kind, noise, shift in (("noise", 1, 0), ("shift", 0, 1), ("both", 1, 1)):
        augmenter = perturbations.evaluation_augmenter(kind)
        assert augmenter.training
        assert (augmenter.config.noise_prob, augmenter.config.shift_prob) == (noise, shift)
        assert augmenter.config.noise_angle_deg_range == (0.0, 2.5)
        assert augmenter.config.shift_delta_range == (-2.0, 2.0)
        assert augmenter.config.shuffle_order_per_batch
        assert augmenter.config.recenter_after_each_op and augmenter.config.renorm_to_input_norm


def test_loader_chunks_rng_and_consumer_order_do_not_change_shared_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Small fixture width exercises full and tail generation batches without
    # making CPU verification depend on a 1024-row augmentation workload.
    monkeypatch.setattr(perturbations, "GENERATION_BATCH_PIXELS", 7)
    clean = _spectra(17)
    original = clean.copy()
    stream = SharedPerturbations(SAMPLE, len(clean), device=CPU)
    before = torch.get_rng_state().clone()
    reference = list(stream.batches(_batches(clean, 4)))
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(11)
    before = torch.get_rng_state().clone()
    replay = list(stream.batches(_batches(clean, 11)))
    assert torch.equal(torch.get_rng_state(), before)
    assert [len(block.clean.snv) for block in reference] == [7, 7, 3]
    for left, right in zip(reference, replay, strict=True):
        assert [(item.kind, item.draw) for item in left.perturbed] == list(DRAW_KEYS)
        for a, b in zip(left.perturbed, right.perturbed, strict=True):
            assert a.seed == perturbation_seed(SAMPLE, a.kind, a.draw)
            np.testing.assert_array_equal(a.batch.snv, b.batch.snv)
            np.testing.assert_array_equal(a.batch.hdf5_rows, left.clean.hdf5_rows)
            np.testing.assert_array_equal(a.batch.pixel_row_col, left.clean.pixel_row_col)
            assert a.batch.snv.dtype == np.float32 and not a.batch.snv.flags.writeable
            np.testing.assert_allclose(a.batch.snv.mean(axis=1), 0, atol=2e-6)
            np.testing.assert_allclose(np.linalg.norm(a.batch.snv, axis=1),
                                       np.linalg.norm(left.clean.snv, axis=1), rtol=2e-6)
        assert not np.array_equal(left.perturbed[0].batch.snv, left.perturbed[1].batch.snv)
        with pytest.raises(ValueError):
            left.perturbed[0].batch.snv[:] = 0
    np.testing.assert_array_equal(clean, original)
    record = stream.record()
    assert len(record["draws"]) == 15 and record["generation_batch_pixels"] == 7
    assert not ({"condition", "K", "training_repeat", "fold"} & set(record))


@pytest.mark.parametrize("kind", ["noise", "shift", "both"])
def test_matches_reference_augmenter_with_one_continuous_stream_per_draw(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    monkeypatch.setattr(perturbations, "GENERATION_BATCH_PIXELS", 7)
    clean = _spectra(10)
    blocks = list(SharedPerturbations(SAMPLE, 10, device=CPU).batches(_batches(clean, 2)))
    augmenter = perturbations.evaluation_augmenter(kind)
    for draw in range(1, 6):
        rng = TorchRandomStream(perturbation_seed(SAMPLE, kind, draw), CPU)
        for block in blocks:
            with fp32_inference(CPU), rng.scope():
                expected = augmenter(torch.from_numpy(block.clean.snv.copy())).numpy()
            actual = next(item.batch.snv for item in block.perturbed
                          if (item.kind, item.draw) == (kind, draw))
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_sample", "reversed"])
def test_generation_requires_full_ordered_sample(problem: str) -> None:
    batches = _batches(_spectra(8), 4)
    if problem == "missing":
        batches.pop()
    elif problem == "duplicate":
        batches[1] = batches[0]
    elif problem == "wrong_sample":
        batches[0] = replace(batches[0], sample_id="KYOw02707")
    else:
        batches.reverse()
    with pytest.raises(ValueError):
        list(SharedPerturbations(SAMPLE, 8, device=CPU).batches(batches))


@pytest.mark.parametrize("value", [0.0, 1e-15, np.nan, np.inf])
def test_generation_failure_preserves_rng_and_reports_source_rows(value: float) -> None:
    clean = _spectra(5)
    clean[3] = value
    before = torch.get_rng_state().clone()
    with pytest.raises(SpectrumInputError) as error:
        list(SharedPerturbations(SAMPLE, 5, device=CPU).batches(_batches(clean, 2)))
    assert error.value.sample_id == SAMPLE and error.value.hdf5_rows == (3,)
    assert torch.equal(torch.get_rng_state(), before)


def test_generator_does_not_leave_global_scopes_active_while_yielding() -> None:
    generator = SharedPerturbations(SAMPLE, 5, device=CPU).batches(_batches(_spectra(5), 5))
    before = torch.get_rng_state().clone()
    precision = torch.backends.cuda.matmul.fp32_precision
    next(generator)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.backends.cuda.matmul.fp32_precision == precision
    generator.close()


def test_flip_counts_five_means_and_occupancy_are_sample_weighted() -> None:
    accumulator = LFRAccumulator(SAMPLE, 5, k=2)
    for clean in (_labels([1, 1, 2], 0), _labels([1, 2], 3)):
        variants = []
        for kind, draw in DRAW_KEYS:
            labels = clean.labels.copy()
            # Each draw flips exactly its first draw rows in the full sample.
            flip = clean.hdf5_rows < draw
            if kind == "noise":
                labels[flip] = 3 - labels[flip]
            elif kind == "both":
                labels[:] = 3 - labels
            variants.append(PerturbedLabels(kind, draw, replace(clean, labels=labels)))
        accumulator.add(clean, tuple(variants))
    result = accumulator.finish()
    assert result.clean_occupancy.counts == (3, 2)
    assert [row.flipped_pixels for row in result.draws[:5]] == [1, 2, 3, 4, 5]
    assert result.mean_by_kind == pytest.approx({"noise": 0.6, "shift": 0.0, "both": 1.0})
    assert result.draws[0].rate == pytest.approx(1 / 5)  # Not the mean of 1/3 and 0/2.
    assert all(sum(row.occupancy.counts) == 5 for row in result.draws)
    assert result.draws[0].occupancy.counts == (2, 3)


def test_single_pixel_and_single_cluster_flip_rates_are_defined() -> None:
    accumulator = LFRAccumulator(SAMPLE, 1, k=4)
    clean = _labels([3])
    accumulator.add(clean, _draws(clean))
    result = accumulator.finish()
    assert result.clean_occupancy.used_clusters == 1
    assert result.clean_occupancy.maximum_fraction == 1.0
    assert all(row.rate == 0.0 for row in result.draws)
    # Swapping cluster IDs changes LFR: do not apply Hungarian matching.
    other = LFRAccumulator(SAMPLE, 1, k=4)
    other.add(clean, _draws(replace(clean, labels=np.array([4]))))
    assert all(row.rate == 1.0 for row in other.finish().draws)


@pytest.mark.parametrize("problem", ["missing_draw", "duplicate_draw", "row", "coordinate",
                                     "sample", "zero", "float", "length"])
def test_rejected_chunk_does_not_partially_update_counts(problem: str) -> None:
    accumulator = LFRAccumulator(SAMPLE, 3, k=2)
    clean = _labels([1, 2, 1])
    variants = list(_draws(clean))
    if problem == "missing_draw":
        variants.pop()
    elif problem == "duplicate_draw":
        variants[-1] = variants[0]
    else:
        changes = {
            "row": {"hdf5_rows": np.array([0, 2, 1])},
            "coordinate": {"pixel_row_col": clean.pixel_row_col[::-1]},
            "sample": {"sample_id": "KYOw02707"},
            "zero": {"labels": np.array([1, 0, 1])},
            "float": {"labels": clean.labels.astype(np.float32)},
            "length": {"labels": np.array([1, 2])},
        }
        variants[-1] = replace(variants[-1], pixels=replace(clean, **changes[problem]))
    with pytest.raises(ValueError):
        accumulator.add(clean, tuple(variants))
    accumulator.add(clean, _draws(clean))
    assert accumulator.finish().clean_occupancy.counts == (2, 1)


def test_missing_rows_and_duplicate_coordinates_cannot_finish() -> None:
    accumulator = LFRAccumulator(SAMPLE, 3, k=2)
    first = _labels([1, 2])
    accumulator.add(first, _draws(first))
    with pytest.raises(ValueError, match="Incomplete"):
        accumulator.finish()
    with pytest.raises(ValueError, match="HDF5"):
        accumulator.add(first, _draws(first))
    last = replace(_labels([1], 2), pixel_row_col=first.pixel_row_col[:1])
    accumulator.add(last, _draws(last))
    with pytest.raises(ValueError, match="Duplicate"):
        accumulator.finish()


def test_predictors_reuse_each_input_across_k_without_refitting_or_mutating_shared_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = B0Baseline()
    features = TrainFeatures("B0", 1, 1, ("train-only",), baseline.transform(_spectra(32)).values)
    clusters = {k: fit_clusters(features, k, device=CPU) for k in (2, 4)}
    centers = {k: cluster.centroids for k, cluster in clusters.items()}

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("Evaluation must not refit centers")

    monkeypatch.setattr(CosineKMeans, "fit", forbidden_fit)
    block = next(SharedPerturbations(SAMPLE, 8, device=CPU).batches(_batches(_spectra(8), 3)))
    original = block.clean.snv.copy()
    seen = []

    class Consumer:
        def transform(self, values: np.ndarray) -> NormalizedRepresentation:
            seen.append(values.copy())
            result = baseline.transform(values)
            values[:] = 0  # Must not corrupt input shared with other consumers.
            return result

    results = []
    for _ in range(2):
        accumulators = {k: LFRAccumulator(SAMPLE, 8, k=k) for k in clusters}
        accumulate_lfr_block(block, Consumer(), clusters, accumulators)
        results.append({k: accumulator.finish() for k, accumulator in accumulators.items()})
        torch.rand(5)
    assert len(seen) == 2 * 16  # 1 clean + 15 perturbed transforms per consumer, not per K.
    for left, right in zip(seen[:16], seen[16:], strict=True):
        np.testing.assert_array_equal(left, right)
    assert results[0] == results[1]
    np.testing.assert_array_equal(block.clean.snv, original)
    for k, cluster in clusters.items():
        np.testing.assert_array_equal(cluster.centroids, centers[k])


def test_neural_consumer_uses_eval_full_visible_fp32_without_decoder_or_rng() -> None:
    with TorchRandomStream(183, CPU).scope():
        model = ChemoMAE(seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                         dim_feedforward=16, dropout=0.0, latent_dim=16,
                         latent_normalize=True, decoder_num_layers=1, n_mask=8)
    representation = NeuralRepresentation(model, CPU)
    features = TrainFeatures("M11", 1, 1, ("train-only",),
                             representation.transform(_spectra(24)).values)
    cluster = fit_clusters(features, 2, device=CPU)
    block = next(SharedPerturbations(SAMPLE, 4, device=CPU).batches(_batches(_spectra(4), 3)))
    seen = []

    def inspect(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        assert not module.training and not torch.is_autocast_enabled("cpu")
        assert inputs[0].dtype == torch.float32 and bool(inputs[1].all())
        seen.append(len(inputs[0]))

    def forbidden_decoder(*args: object) -> None:
        raise AssertionError("LFR extraction must not call the decoder")

    encoder_hook = model.encoder.register_forward_pre_hook(inspect)
    decoder_hook = model.decoder.register_forward_pre_hook(forbidden_decoder)
    before = torch.get_rng_state().clone()
    try:
        accumulator = LFRAccumulator(SAMPLE, 4, k=2)
        accumulate_lfr_block(block, representation, {2: cluster}, {2: accumulator})
        assert len(accumulator.finish().draws) == 15
    finally:
        encoder_hook.remove()
        decoder_hook.remove()
    assert seen == [4] * 16 and torch.equal(before, torch.get_rng_state())
