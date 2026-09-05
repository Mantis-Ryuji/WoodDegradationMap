"""Train-only reference Cosine-KMeans and complete coordinate-based label maps."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans

from .baselines import (
    NormalizationDiagnostics, NormalizedRepresentation, PCABaseline, RepresentationError,
)
from .config import CLUSTER_COUNTS, CONDITIONS, experiment_config, kmeans_seed
from .data import FoldData, SpectrumInputError
from .neural import fp32_inference


class Representation(Protocol):
    def transform(self, spectra: np.ndarray) -> NormalizedRepresentation: ...


def _dimension(condition_id: str) -> int:
    for condition in CONDITIONS:
        if condition.condition_id == condition_id:
            return condition.output_dim
    raise ValueError(f"Unknown representation condition: {condition_id}")


def _diagnose(values: np.ndarray, dimension: int) -> NormalizationDiagnostics:
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32
            or values.ndim != 2 or values.shape[1] != dimension
            or len(values) == 0):
        raise ValueError(f"Expected a nonempty FP32 representation with {dimension} columns")
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        norms = torch.linalg.vector_norm(tensor, dim=1).numpy()
    finite = np.isfinite(values).all(axis=1)
    finite_norm = np.isfinite(norms)
    diagnostics = NormalizationDiagnostics(
        len(values), dimension, tuple(np.flatnonzero(~finite).tolist()),
        tuple(np.flatnonzero(finite & ~finite_norm).tolist()),
        tuple(np.flatnonzero(finite & (norms == 0)).tolist()),
        tuple(np.flatnonzero(finite & finite_norm & (norms > 0) & (norms < 1e-6)).tolist()),
        None,
    )
    if (diagnostics.nonfinite_rows or diagnostics.nonfinite_norm_rows
            or diagnostics.zero_norm_rows or diagnostics.epsilon_clamped_rows):
        raise RepresentationError(diagnostics)
    # Inspect the supplied values without an extra normalization operation.
    return NormalizationDiagnostics(len(values), dimension, (), (), (), (),
                                    float(np.max(np.abs(norms - np.float32(1)))))


@dataclass(frozen=True)
class TrainFeatures:
    condition_id: str
    fold: int
    repeat: int
    sample_ids: tuple[str, ...]
    values: np.ndarray


def collect_train_features(
    data: FoldData, representation: Representation, *, condition_id: str, repeat: int,
    chunk_pixels: int = 2048,
) -> TrainFeatures:
    """Extract the shared train selection once for reuse across all planned K.

    No tail is dropped, and transform is never allowed to return fewer rows.
    This function calls no fit method and never opens test spectra.
    """
    dimension = _dimension(condition_id)
    kmeans_seed(data.fold, repeat, CLUSTER_COUNTS[0])
    if isinstance(representation, PCABaseline):
        record = representation.record
        if (condition_id != "B1" or record.fold != data.fold
                or record.sample_ids != data.train_sample_ids
                or record.train_pixel_count != data.train_pixel_count
                or (record.repeat != repeat and not representation.reusable_across_repeats)):
            raise ValueError("PCA was not fitted to this fold's shared train selection")
    features = np.empty((data.train_pixel_count, dimension), dtype=np.float32)
    offset = 0
    for batch in data.batches("train", chunk_pixels=chunk_pixels):
        try:
            transformed = representation.transform(batch.snv)
            if len(transformed.values) != len(batch.snv):
                raise ValueError(f"{batch.sample_id}: transform changed the number of pixels")
            _diagnose(transformed.values, dimension)
        except RepresentationError as error:
            diagnostic = error.diagnostics
            bad = sorted(set(diagnostic.nonfinite_rows + diagnostic.nonfinite_norm_rows
                             + diagnostic.zero_norm_rows + diagnostic.epsilon_clamped_rows))
            raise SpectrumInputError(batch.sample_id, batch.hdf5_rows[bad], str(error)) from error
        stop = offset + len(batch.snv)
        features[offset:stop] = transformed.values
        offset = stop
    if offset != len(features):
        raise ValueError("Train feature count differs from the shared manifest")
    return TrainFeatures(condition_id, data.fold, repeat, data.train_sample_ids, features)


@dataclass(frozen=True)
class ClusterFitRecord:
    condition_id: str
    fold: int
    repeat: int
    k: int
    seed: int
    dimension: int
    train_sample_ids: tuple[str, ...]
    train_pixels: int
    train_occupancy: tuple[int, ...]
    reference_inertia: float
    final_center_inertia: float
    max_iter: int
    tol: float
    iterations: int | None
    stop_reason: str
    train_unit_norm_error_max: float
    center_unit_norm_error_max: float
    fit_seconds: float
    chemomae_version: str
    torch_version: str
    fit_device: str


class FittedClusters:
    """Prediction with fixed centers; no fit method or automatic center update."""

    def __init__(self, module: CosineKMeans, record: ClusterFitRecord) -> None:
        self._module = module
        self.record = record

    @property
    def centroids(self) -> np.ndarray:
        return self._module.centroids.detach().cpu().numpy().copy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict one FP32 chunk; labels remain 0..K-1 until placed in a map."""
        _diagnose(features, self.record.dimension)
        with fp32_inference(self._module.device):
            labels = self._module.predict(torch.from_numpy(np.ascontiguousarray(features)))
        return labels.cpu().numpy().copy()

    def save(self, path: Path) -> None:
        """Save exact center values under checkpoints/, without pickle or renormalizing."""
        _diagnose(self.centroids, self.record.dimension)
        with path.open("xb") as destination:
            np.savez_compressed(destination, centroids=self.centroids,
                                metadata=np.array(json.dumps({"schema_version": 1,
                                                              "fit": asdict(self.record)})))

    @classmethod
    def load(
        cls, path: Path, *, condition_id: str, fold: int, repeat: int, k: int,
        device: torch.device,
    ) -> FittedClusters:
        if type(k) is not int:
            raise ValueError("K must be an integer in the fixed plan")
        expected_seed = kmeans_seed(fold, repeat, k)
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"centroids", "metadata"}:
                raise ValueError("Unexpected centroid checkpoint fields")
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata.get("schema_version") != 1:
                raise ValueError("Unsupported centroid checkpoint schema")
            fit = metadata["fit"]
            record = ClusterFitRecord(**{
                **fit, "train_sample_ids": tuple(fit["train_sample_ids"]),
                "train_occupancy": tuple(fit["train_occupancy"]),
            })
            centers = saved["centroids"].copy()
        settings = experiment_config()["clustering"]
        if ((record.condition_id, record.fold, record.repeat, record.k, record.seed)
                != (condition_id, fold, repeat, k, expected_seed)
                or record.dimension != _dimension(condition_id)
                or record.max_iter != settings["max_iter"] or record.tol != settings["tol"]
                or record.chemomae_version != version("chemomae")
                or record.torch_version != str(torch.__version__)):
            raise ValueError("Centroid checkpoint run/config/version mismatch")
        if (centers.shape != (k, record.dimension) or record.train_pixels < k
                or not record.train_sample_ids or len(record.train_occupancy) != k
                or sum(record.train_occupancy) != record.train_pixels
                or any(type(count) is not int or count < 0 for count in record.train_occupancy)
                or not math.isfinite(record.reference_inertia)
                or not math.isfinite(record.final_center_inertia)):
            raise ValueError("Invalid centroid checkpoint shape or training provenance")
        _diagnose(centers, record.dimension)
        module = CosineKMeans(k, tol=record.tol, max_iter=record.max_iter,
                              device=device, random_state=record.seed).to(dtype=torch.float32)
        # The reference save/load helpers normalize centers again. Populate the
        # same fitted state directly to preserve the exact fit-time predictors.
        module.centroids.resize_(centers.shape)
        module.centroids.copy_(torch.from_numpy(centers).to(device))
        module.latent_dim = record.dimension
        module.inertia_ = record.reference_inertia
        module._fitted = True
        return cls(module, record)


def fit_clusters(features: TrainFeatures, k: int, *, device: torch.device) -> FittedClusters:
    """Call reference fit once on all train features; never perform restarts or select K."""
    dimension = _dimension(features.condition_id)
    if type(k) is not int:
        raise ValueError("K must be an integer in the fixed plan")
    seed = kmeans_seed(features.fold, features.repeat, k)
    if version("chemomae") != "0.2.1":
        raise ValueError("The fixed protocol requires ChemoMAE 0.2.1")
    diagnostics = _diagnose(features.values, dimension)
    if len(features.values) < k or not features.sample_ids:
        raise ValueError("Insufficient train features or missing sample provenance")
    settings = experiment_config()["clustering"]
    module = CosineKMeans(k, tol=settings["tol"], max_iter=settings["max_iter"],
                          device=device, random_state=seed).to(dtype=torch.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(features.values))
    started = time.perf_counter()
    with fp32_inference(device):
        module.fit(tensor)  # Full-device reference default; no streaming algorithm substitution.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        center_diagnostic = _diagnose(module.centroids.detach().cpu().numpy(), dimension)
        labels, distances = module.predict(tensor, return_dist=True)
        inertia = float(distances.gather(1, labels[:, None]).mean())
        occupancy = tuple(torch.bincount(labels, minlength=k).cpu().tolist())
    if not math.isfinite(module.inertia_) or not math.isfinite(inertia):
        raise ValueError("KMeans produced a nonfinite objective")
    # Reference inertia is evaluated before its final M-step, not against the
    # final saved centers. Keep the two objectives explicitly distinguishable.
    record = ClusterFitRecord(
        features.condition_id, features.fold, features.repeat, k, seed, dimension,
        features.sample_ids, len(features.values), occupancy, float(module.inertia_), inertia,
        settings["max_iter"], settings["tol"], None, "not_exposed_by_ChemoMAE_0.2.1",
        diagnostics.unit_norm_absolute_error_max, center_diagnostic.unit_norm_absolute_error_max,
        elapsed, version("chemomae"), str(torch.__version__), str(device),
    )
    return FittedClusters(module, record)


class LabelMap:
    """Restore chunk predictions to one sample and require complete valid-mask coverage."""

    def __init__(self, valid_mask: np.ndarray, k: int) -> None:
        if (type(k) is not int or k not in CLUSTER_COUNTS or not isinstance(valid_mask, np.ndarray)
                or valid_mask.ndim != 2 or valid_mask.dtype.kind not in "biu"
                or not np.isin(valid_mask, (0, 1)).all() or not valid_mask.any()):
            raise ValueError("Expected a nonempty binary valid mask and a planned K")
        self.k = k
        self._valid = valid_mask.astype(bool, copy=True)
        self._labels = np.zeros(valid_mask.shape, dtype=np.uint8)

    def add(self, coordinates: np.ndarray, labels: np.ndarray) -> None:
        if (not isinstance(coordinates, np.ndarray) or not isinstance(labels, np.ndarray)
                or coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0
                or coordinates.dtype.kind not in "iu" or labels.shape != (len(coordinates),)
                or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= self.k)):
            raise ValueError("Expected integer pixel coordinates and 0..K-1 labels")
        height, width = self._valid.shape
        rows, columns = coordinates[:, 0], coordinates[:, 1]
        if (np.any(rows < 0) or np.any(rows >= height)
                or np.any(columns < 0) or np.any(columns >= width)):
            raise ValueError("Pixel coordinate is outside the sample image")
        flat = rows.astype(np.int64) * width + columns.astype(np.int64)
        if (len(np.unique(flat)) != len(flat) or np.any(self._labels[rows, columns] != 0)):
            raise ValueError("Duplicate pixel assignment within or across chunks")
        if not self._valid[rows, columns].all():
            raise ValueError("Cannot assign a cluster to background/invalid pixels")
        self._labels[rows, columns] = labels + 1

    def finish(self) -> np.ndarray:
        missing = int(np.count_nonzero(self._valid & (self._labels == 0)))
        if missing:
            raise ValueError(f"Missing predictions for {missing} valid pixels; do not drop rows")
        return self._labels.copy()
