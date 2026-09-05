"""Aligned label flip counts, five-draw means and fixed-center block prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .cluster_pipeline import _transform_batch
from .clustering import FittedClusters, Representation
from .config import CLUSTER_COUNTS, PERTURBATIONS, Perturbation, perturbation_seed
from .data import SpectrumBatch
from .perturbations import DRAW_KEYS, SharedPerturbationBlock


@dataclass(frozen=True)
class PixelLabels:
    sample_id: str
    hdf5_rows: np.ndarray
    pixel_row_col: np.ndarray
    labels: np.ndarray  # 1..K, never background or reference-library zero-based IDs.


@dataclass(frozen=True)
class PerturbedLabels:
    kind: Perturbation
    draw: int
    pixels: PixelLabels


@dataclass(frozen=True)
class Occupancy:
    counts: tuple[int, ...]
    fractions: tuple[float, ...]
    used_clusters: int
    maximum_fraction: float


@dataclass(frozen=True)
class LFRDraw:
    kind: Perturbation
    draw: int
    flipped_pixels: int
    rate: float
    occupancy: Occupancy


@dataclass(frozen=True)
class LFRResult:
    sample_id: str
    k: int
    valid_pixels: int
    clean_occupancy: Occupancy
    draws: tuple[LFRDraw, ...]
    mean_by_kind: dict[str, float]


def _occupancy(counts: np.ndarray, n: int) -> Occupancy:
    fractions = counts.astype(np.float32) / np.float32(n)
    return Occupancy(tuple(int(value) for value in counts), tuple(float(value) for value in fractions),
                     int(np.count_nonzero(counts)), float(fractions.max()))


class LFRAccumulator:
    """Accumulate one sample/condition/K/repeat without averaging chunk rates.

    Every add requires all 15 draw labels aligned with the clean source rows and
    coordinates. Updates are committed only after validating the entire chunk.
    The caller supplies the full valid pixel count from the input inventory.
    Missing/duplicate rows, coordinates or draws are failures, not missing scores.
    """

    def __init__(self, sample_id: str, pixel_count: int, *, k: int) -> None:
        if (not sample_id or type(pixel_count) is not int or pixel_count < 1
                or type(k) is not int or k not in CLUSTER_COUNTS):
            raise ValueError("Expected a sample ID, positive valid pixel count and planned K")
        self.sample_id, self.pixel_count, self.k = sample_id, pixel_count, k
        self._received = 0
        self._coordinates = np.empty((pixel_count, 2), dtype=np.int64)
        self._clean_counts = np.zeros(k, dtype=np.int64)
        self._draw_counts = np.zeros((len(DRAW_KEYS), k), dtype=np.int64)
        self._flips = np.zeros(len(DRAW_KEYS), dtype=np.int64)

    def _validate(self, pixels: PixelLabels) -> None:
        labels = pixels.labels
        if (not isinstance(labels, np.ndarray) or labels.ndim != 1 or len(labels) == 0
                or labels.dtype.kind not in "iu" or np.any(labels < 1) or np.any(labels > self.k)
                or pixels.sample_id != self.sample_id):
            raise ValueError("Expected the same sample's nonempty integer labels in 1..K")
        if (pixels.hdf5_rows.shape != labels.shape or pixels.hdf5_rows.dtype.kind not in "iu"
                or pixels.pixel_row_col.shape != (len(labels), 2)
                or pixels.pixel_row_col.dtype.kind not in "iu" or np.any(pixels.pixel_row_col < 0)
                or np.any(pixels.pixel_row_col > np.iinfo(np.int64).max)):
            raise ValueError("Invalid pixel row/coordinate identity")

    def add(self, clean: PixelLabels, perturbed: tuple[PerturbedLabels, ...]) -> None:
        self._validate(clean)
        n = len(clean.labels)
        if (self._received + n > self.pixel_count or not np.array_equal(
                clean.hdf5_rows, np.arange(self._received, self._received + n))):
            raise ValueError("Missing, duplicated or out-of-order clean HDF5 rows")
        by_key = {(item.kind, item.draw): item.pixels for item in perturbed}
        if (len(perturbed) != len(DRAW_KEYS) or set(by_key) != set(DRAW_KEYS)
                or any(type(item.draw) is not int for item in perturbed)):
            raise ValueError("All three perturbation kinds and draws 1..5 are required exactly once")
        draw_counts = np.zeros_like(self._draw_counts)
        flips = np.zeros_like(self._flips)
        for index, key in enumerate(DRAW_KEYS):
            pixels = by_key[key]
            self._validate(pixels)
            if (not np.array_equal(pixels.hdf5_rows, clean.hdf5_rows)
                    or not np.array_equal(pixels.pixel_row_col, clean.pixel_row_col)):
                raise ValueError("Perturbed labels do not align with the clean rows/coordinates")
            flips[index] = np.count_nonzero(pixels.labels != clean.labels)
            draw_counts[index] = np.bincount(pixels.labels.astype(np.int64), minlength=self.k + 1)[1:]
        self._coordinates[self._received:self._received + n] = clean.pixel_row_col
        self._clean_counts += np.bincount(clean.labels.astype(np.int64), minlength=self.k + 1)[1:]
        self._draw_counts += draw_counts
        self._flips += flips
        self._received += n

    def finish(self) -> LFRResult:
        if self._received != self.pixel_count:
            raise ValueError("Incomplete sample; do not average partial LFR results")
        if len(np.unique(self._coordinates, axis=0)) != self.pixel_count:
            raise ValueError("Duplicate source coordinates across LFR chunks")
        draws = tuple(LFRDraw(
            kind, draw, int(self._flips[index]),
            float(np.float32(self._flips[index]) / np.float32(self.pixel_count)),
            _occupancy(self._draw_counts[index], self.pixel_count),
        ) for index, (kind, draw) in enumerate(DRAW_KEYS))
        means = {kind: float(np.mean(np.array([row.rate for row in draws if row.kind == kind],
                                              dtype=np.float32), dtype=np.float32))
                 for kind in PERTURBATIONS}
        return LFRResult(self.sample_id, self.k, self.pixel_count,
                         _occupancy(self._clean_counts, self.pixel_count), draws, means)


@dataclass(frozen=True)
class LFRBlockPrediction:
    values: np.ndarray
    clean: dict[int, PixelLabels]


def accumulate_lfr_block(
    block: SharedPerturbationBlock, representation: Representation,
    clusters: Mapping[int, FittedClusters], accumulators: Mapping[int, LFRAccumulator],
) -> LFRBlockPrediction:
    """Transform clean/15 perturbed inputs once each, reusing each across all supplied K.

    No fit method is called. NeuralRepresentation retains eval/all-visible FP32
    inference while the independent generation augmenter uses training mode.
    Use the SAME block for other conditions and training repeats. Inputs passed
    to transform are private copies so one consumer cannot corrupt shared data.
    Return the clean features/labels for map verification and pooled silhouette,
    avoiding a second clean extraction in the evaluation pipeline.
    """
    if not clusters or set(clusters) != set(accumulators):
        raise ValueError("Matching nonempty cluster and accumulator K sets are required")
    runs = {(cluster.record.condition_id, cluster.record.fold, cluster.record.repeat)
            for cluster in clusters.values()}
    if len(runs) != 1:
        raise ValueError("All K must belong to the same representation/fold/repeat")
    condition, _, _ = next(iter(runs))
    for k, cluster in clusters.items():
        accumulator = accumulators[k]
        if (cluster.record.k != k or accumulator.k != k
                or accumulator.sample_id != block.clean.sample_id):
            raise ValueError("Cluster or accumulator identity mismatch")
    keys = [(item.kind, item.draw) for item in block.perturbed]
    if len(keys) != len(DRAW_KEYS) or set(keys) != set(DRAW_KEYS):
        raise ValueError("Shared block must contain exactly the 15 planned draws")
    for item in block.perturbed:
        if (type(item.draw) is not int
                or item.seed != perturbation_seed(block.clean.sample_id, item.kind, item.draw)
                or item.batch.sample_id != block.clean.sample_id
                or not np.array_equal(item.batch.hdf5_rows, block.clean.hdf5_rows)
                or not np.array_equal(item.batch.pixel_row_col, block.clean.pixel_row_col)):
            raise ValueError("Shared perturbation coordinates/rows differ from clean input")

    def predict(batch: SpectrumBatch) -> LFRBlockPrediction:
        private = SpectrumBatch(batch.sample_id, batch.hdf5_rows, batch.pixel_row_col, batch.snv.copy())
        features = _transform_batch(representation, private, condition).values
        return LFRBlockPrediction(features, {
            k: PixelLabels(batch.sample_id, batch.hdf5_rows, batch.pixel_row_col,
                           cluster.predict(features) + 1) for k, cluster in clusters.items()
        })

    clean = predict(block.clean)
    draws: dict[int, list[PerturbedLabels]] = {k: [] for k in clusters}
    for item in block.perturbed:
        predictions = predict(item.batch)
        for k in clusters:
            draws[k].append(PerturbedLabels(item.kind, item.draw, predictions.clean[k]))
    for k in clusters:
        accumulators[k].add(clean.clean[k], tuple(draws[k]))
    return clean
