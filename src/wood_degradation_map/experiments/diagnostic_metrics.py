"""Pooled test-fold cosine silhouette and aligned three-repeat sample ARI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import torch
from chemomae.clustering.metric import silhouette_samples_cosine_gpu

from .clustering import _dimension
from .config import CLUSTER_COUNTS, REPEATS, experiment_config, kmeans_seed
from .data import SpectrumInputError
from .lfr import PixelLabels
from .neural import fp32_inference


def _validate_pixels(pixels: PixelLabels, n: int, k: int) -> None:
    if type(n) is not int or n < 1:
        raise ValueError("An empty valid pixel set is an input failure")
    if (not pixels.sample_id or not isinstance(pixels.labels, np.ndarray)
            or pixels.labels.shape != (n,) or pixels.labels.dtype.kind not in "iu"
            or np.any(pixels.labels < 1) or np.any(pixels.labels > k)):
        raise ValueError("Expected all valid pixels with integer labels in 1..K; background is excluded")
    if (not isinstance(pixels.hdf5_rows, np.ndarray) or pixels.hdf5_rows.dtype.kind not in "iu"
            or not np.array_equal(pixels.hdf5_rows, np.arange(n))):
        raise ValueError("Expected all HDF5 rows in canonical order, without omission or duplication")
    coordinates = pixels.pixel_row_col
    if (not isinstance(coordinates, np.ndarray) or coordinates.shape != (n, 2)
            or coordinates.dtype.kind not in "iu" or np.any(coordinates < 0)
            or len(np.unique(coordinates, axis=0)) != n):
        raise ValueError("Expected unique, nonnegative source coordinates")


@dataclass(frozen=True)
class SampleFeatures:
    pixels: PixelLabels
    values: np.ndarray


@dataclass(frozen=True)
class SampleSilhouette:
    sample_id: str
    valid_pixels: int
    offset: int
    mean: float | None
    undefined_reason: str | None
    cluster_counts: tuple[int, ...]
    singleton_pixels: int  # Singleton membership in the pooled test fold, not within this sample.


@dataclass(frozen=True)
class FoldSilhouette:
    condition_id: str
    fold: int
    repeat: int
    k: int
    valid_pixels: int
    used_clusters: int
    cluster_counts: tuple[int, ...]
    unit_norm_absolute_error_max: float
    pixel_scores: np.ndarray | None  # Concatenated in the returned sample order; no NaN placeholders.
    samples: tuple[SampleSilhouette, ...]
    macro_mean: float | None
    defined_samples: int
    undefined_reason: str | None
    device: str
    chunk_pixels: int
    torch_version: str
    chemomae_version: str


def fold_silhouette(
    samples: tuple[SampleFeatures, ...], *, expected_test_pixels: Mapping[str, int],
    condition_id: str, fold: int, repeat: int, k: int, device: torch.device,
    chunk_pixels: int = 1_000_000,
) -> FoldSilhouette:
    """Call the reference once on ALL test samples, then compute sample means.

    expected_test_pixels must come from the validated fold/inventory, not from
    the supplied feature arrays. The caller extracts values and labels using
    the same SpectrumBatch coordinates. Missing samples/rows are errors, even
    for a fold where silhouette would otherwise be undefined. No spectra I/O,
    fit, extra normalization or K/condition selection occurs here.

    The reference moves the complete N x D representation to device and builds
    several additional N x D arrays. chunk_pixels bounds only the N x K tile,
    not the full-fold memory footprint; there is no automatic device fallback.
    CPU exists for fixtures and CUDA for production evaluation.
    """
    dimension = _dimension(condition_id)
    if type(k) is not int:
        raise ValueError("K must be an integer in the fixed plan")
    kmeans_seed(fold, repeat, k)
    if device.type not in ("cpu", "cuda") or type(chunk_pixels) is not int or chunk_pixels < 1:
        raise ValueError("Expected CPU/CUDA and a positive integer chunk size")
    if version("chemomae") != "0.2.1":
        raise ValueError("The fixed silhouette protocol requires ChemoMAE 0.2.1")
    ids = [sample.pixels.sample_id for sample in samples]
    if (not expected_test_pixels or len(ids) != len(set(ids))
            or set(ids) != set(expected_test_pixels)):
        raise ValueError("Expected exactly every test sample from the validated fold")
    ordered = sorted(samples, key=lambda sample: sample.pixels.sample_id)
    norm_error = 0.0
    for sample in ordered:
        n = expected_test_pixels[sample.pixels.sample_id]
        _validate_pixels(sample.pixels, n, k)
        values = sample.values
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                or values.shape != (n, dimension)):
            raise ValueError(f"Expected FP32 test representations with {dimension} columns")
        # Use the silhouette epsilon here, not the KMeans epsilon. Values are
        # already normalized by the representation; report drift without rescaling.
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            norms = torch.linalg.vector_norm(torch.from_numpy(values.copy()), dim=1).numpy()
        bad = (~np.isfinite(values).all(axis=1) | ~np.isfinite(norms)
               | (norms < experiment_config()["evaluation"]["silhouette_eps"]))
        if bad.any():
            raise SpectrumInputError(sample.pixels.sample_id, sample.pixels.hdf5_rows[bad],
                                     "Nonfinite or zero/epsilon-clamped silhouette representation")
        norm_error = max(norm_error, float(np.max(np.abs(norms - np.float32(1)))))
    labels = np.concatenate([sample.pixels.labels.astype(np.int64) for sample in ordered])
    counts = np.bincount(labels, minlength=k + 1)[1:]
    n, used = len(labels), int(np.count_nonzero(counts))
    reason = "single_cluster" if used == 1 else "all_singletons" if used == n else None
    scores = None
    if reason is None:
        values = np.concatenate([sample.values for sample in ordered])
        with fp32_inference(device):
            scores = silhouette_samples_cosine_gpu(
                values, labels, device=str(device), chunk=chunk_pixels,
                dtype=torch.float32, return_numpy=True,
                eps=experiment_config()["evaluation"]["silhouette_eps"],
            )
        if not isinstance(scores, np.ndarray) or scores.dtype != np.float32 or scores.shape != (n,):
            raise ValueError("Reference silhouette returned an invalid score array")
        bad = ~np.isfinite(scores) | (scores < -1) | (scores > 1)
        if bad.any():
            offset = 0
            for sample in ordered:
                stop = offset + len(sample.values)
                if bad[offset:stop].any():
                    raise SpectrumInputError(
                        sample.pixels.sample_id, sample.pixels.hdf5_rows[bad[offset:stop]],
                        "Reference silhouette produced nonfinite or out-of-range scores; no clipping",
                    )
                offset = stop
        scores.flags.writeable = False
    reports = []
    offset = 0
    for sample in ordered:
        sample_n = len(sample.values)
        sample_counts = np.bincount(sample.pixels.labels.astype(np.int64), minlength=k + 1)[1:]
        singleton_pixels = int(np.count_nonzero(counts[sample.pixels.labels.astype(np.int64) - 1] == 1))
        mean = float(scores[offset:offset + sample_n].mean(dtype=np.float32)) if scores is not None else None
        reports.append(SampleSilhouette(sample.pixels.sample_id, sample_n, offset, mean, reason,
                                        tuple(int(value) for value in sample_counts), singleton_pixels))
        offset += sample_n
    macro = (float(np.mean(np.array([row.mean for row in reports], dtype=np.float32), dtype=np.float32))
             if scores is not None else None)
    return FoldSilhouette(condition_id, fold, repeat, k, n, used, tuple(int(value) for value in counts),
                          norm_error, scores, tuple(reports), macro,
                          len(reports) if scores is not None else 0, reason, str(device), chunk_pixels,
                          str(torch.__version__), version("chemomae"))


@dataclass(frozen=True)
class ARIPair:
    repeats: tuple[int, int]
    value: float | None
    undefined_reason: str | None
    used_clusters: tuple[int, int]
    degeneracy_flags: tuple[str, ...]
    contingency: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RepeatARI:
    sample_id: str
    condition_id: str
    fold: int
    k: int
    valid_pixels: int
    pairs: tuple[ARIPair, ...]
    mean: float | None
    undefined_reason: str | None


def _ari_pair(left: np.ndarray, right: np.ndarray, k: int, repeats: tuple[int, int]) -> ARIPair:
    n = len(left)
    table = np.bincount((left.astype(np.int64) - 1) * k + right.astype(np.int64) - 1,
                        minlength=k * k).reshape(k, k)
    rows, columns = table.sum(axis=1), table.sum(axis=0)
    used = (int(np.count_nonzero(rows)), int(np.count_nonzero(columns)))
    flags = tuple(f"repeat_{repeat}_{kind}"
                  for repeat, count in zip(repeats, used, strict=True)
                  for kind, applies in (("single_cluster", count == 1), ("all_singletons", count == n))
                  if applies)
    contingency = tuple(tuple(int(value) for value in row) for row in table)
    if n < 2:
        return ARIPair(repeats, None, "fewer_than_two_valid_pixels", used, flags, contingency)

    def pairs(count: int) -> int:
        value = int(count)
        return value * (value - 1) // 2

    total = pairs(n)
    a, b = sum(pairs(count) for count in rows), sum(pairs(count) for count in columns)
    c = sum(pairs(count) for count in table.flat)
    # Python integer products avoid int64 overflow and cancellation in C-AB/T.
    numerator = 2 * (c * total - a * b)
    denominator = (a + b) * total - 2 * a * b
    value = 1.0 if denominator == 0 else float(np.float32(numerator) / np.float32(denominator))
    return ARIPair(repeats, value, None, used, flags, contingency)


def repeat_ari(
    predictions: Mapping[int, PixelLabels], *, expected_pixel_count: int,
    condition_id: str, fold: int, k: int,
) -> RepeatARI:
    """Compare the three completed runs for one sample using exactly matching pixels.

    expected_pixel_count comes from the input inventory. Run loading must check
    the condition/fold/K metadata before supplying the three label arrays here.
    Return each pair, degeneracy flags and the per-sample mean; pair SD is not an
    estimate of three independent training repetitions. Background is never accepted.
    """
    _dimension(condition_id)
    if type(k) is not int or k not in CLUSTER_COUNTS:
        raise ValueError("K must be an integer in the fixed plan")
    kmeans_seed(fold, 1, k)
    if set(predictions) != set(REPEATS) or any(type(key) is not int for key in predictions):
        raise ValueError("Exactly the completed repeats 1, 2, 3 are required")
    first = predictions[1]
    for pixels in predictions.values():
        _validate_pixels(pixels, expected_pixel_count, k)
        if (pixels.sample_id != first.sample_id or not np.array_equal(pixels.hdf5_rows, first.hdf5_rows)
                or not np.array_equal(pixels.pixel_row_col, first.pixel_row_col)):
            raise ValueError("ARI repetitions must have identical sample/row/coordinate identities")
    pairs = tuple(_ari_pair(predictions[left].labels, predictions[right].labels, k, (left, right))
                  for left, right in experiment_config()["evaluation"]["ari_pairs"])
    reason = "fewer_than_two_valid_pixels" if expected_pixel_count < 2 else None
    mean = (None if reason else float(np.mean(np.array([pair.value for pair in pairs], dtype=np.float32),
                                             dtype=np.float32)))
    return RepeatARI(first.sample_id, condition_id, fold, k, expected_pixel_count, pairs, mean, reason)
