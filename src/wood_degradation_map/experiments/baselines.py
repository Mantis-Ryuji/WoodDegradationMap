"""FP32 B0 and train-only sklearn PCA with explicit numerical diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA

from .config import FOLDS, REPEATS, experiment_config, run_seed
from .data import FoldData


@dataclass(frozen=True)
class NormalizationDiagnostics:
    row_count: int
    dimension: int
    nonfinite_rows: tuple[int, ...]
    nonfinite_norm_rows: tuple[int, ...]
    zero_norm_rows: tuple[int, ...]
    epsilon_clamped_rows: tuple[int, ...]
    unit_norm_absolute_error_max: float | None


@dataclass(frozen=True)
class NormalizedRepresentation:
    values: np.ndarray
    diagnostics: NormalizationDiagnostics


class RepresentationError(ValueError):
    """Numerical failure with inspectable row indices; no row is silently dropped."""

    def __init__(self, diagnostics: NormalizationDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "Invalid cosine representation: "
            f"nonfinite={len(diagnostics.nonfinite_rows)}, "
            f"nonfinite_norm={len(diagnostics.nonfinite_norm_rows)}, "
            f"zero_norm={len(diagnostics.zero_norm_rows)}, "
            f"epsilon_clamped={len(diagnostics.epsilon_clamped_rows)}",
        )


def _matrix(values: np.ndarray, dimension: int) -> None:
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32")
            or values.ndim != 2 or values.shape[1] != dimension or len(values) == 0):
        raise ValueError(f"Expected a nonempty float32 matrix with {dimension} columns")


def normalize_representation(values: np.ndarray) -> NormalizedRepresentation:
    """Use the reference helper on CPU FP32, without redefining small norms as valid.

    Norms below its fixed epsilon would be clamped by the helper and fail to
    become unit length. Report those rows as a run failure. Other unit-norm
    errors are diagnostics, without introducing an empirical exclusion cutoff.
    """
    import torch
    from chemomae.clustering.ops import l2_normalize_rows

    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected a two-dimensional representation")
    _matrix(values, values.shape[1])
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        norms = torch.linalg.vector_norm(tensor, dim=1).numpy()
        finite = np.isfinite(values).all(axis=1)
        finite_norm = np.isfinite(norms)
        eps = 1e-6
        diagnostics = NormalizationDiagnostics(
            len(values), values.shape[1], tuple(np.flatnonzero(~finite).tolist()),
            tuple(np.flatnonzero(finite & ~finite_norm).tolist()),
            tuple(np.flatnonzero(finite & (norms == 0)).tolist()),
            tuple(np.flatnonzero(finite & finite_norm & (norms > 0) & (norms < eps)).tolist()),
            None,
        )
        if (diagnostics.nonfinite_rows or diagnostics.nonfinite_norm_rows
                or diagnostics.zero_norm_rows or diagnostics.epsilon_clamped_rows):
            raise RepresentationError(diagnostics)
        normalized = l2_normalize_rows(tensor, eps=eps)
        error = float(torch.max(torch.abs(torch.linalg.vector_norm(normalized, dim=1) - 1)))
        return NormalizedRepresentation(normalized.numpy(), NormalizationDiagnostics(
            len(values), values.shape[1], (), (), (), (), error,
        ))


def _b0_spec() -> dict[str, object]:
    return {"schema_version": 1, "condition": "B0", "input": "SNV", "dimension": 256,
            "dtype": "float32", "normalization": "chemomae.clustering.ops.l2_normalize_rows",
            "eps": 1e-6, "fitted_parameters": None}


class B0Baseline:
    """B0 has no fit step or learned parameters; only its transform contract is saved."""

    def transform(self, spectra: np.ndarray) -> NormalizedRepresentation:
        _matrix(spectra, 256)
        return normalize_representation(spectra)

    def save(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as destination:
            json.dump(_b0_spec(), destination, indent=2)
            destination.write("\n")

    @classmethod
    def load(cls, path: Path) -> B0Baseline:
        if json.loads(path.read_text(encoding="utf-8")) != _b0_spec():
            raise ValueError("Saved B0 contract differs from the fixed protocol")
        return cls()


@dataclass(frozen=True)
class PCAFitRecord:
    fold: int
    repeat: int
    seed: int
    sample_ids: tuple[str, ...]
    train_pixel_count: int
    solver: str
    sklearn_version: str
    numpy_version: str


@dataclass(frozen=True)
class PCABaseline:
    estimator: PCA
    record: PCAFitRecord

    @property
    def reusable_across_repeats(self) -> bool:
        return self.record.solver in ("full", "covariance_eigh")

    def transform(self, spectra: np.ndarray) -> NormalizedRepresentation:
        _matrix(spectra, 256)
        finite = np.isfinite(spectra).all(axis=1)
        zero = ~(spectra != 0).any(axis=1)
        if not finite.all() or zero.any():
            raise RepresentationError(NormalizationDiagnostics(
                len(spectra), 256, tuple(np.flatnonzero(~finite).tolist()), (),
                tuple(np.flatnonzero(zero).tolist()), (), None,
            ))
        projected = self.estimator.transform(spectra)
        _matrix(projected, 16)
        return normalize_representation(projected)

    def save(self, path: Path) -> None:
        """Save numerical state without pickle; place coefficients under checkpoints/."""
        record = {"schema_version": 1, "condition": "B1", "fit": asdict(self.record)}
        with path.open("xb") as destination:
            np.savez_compressed(
                destination, metadata=np.array(json.dumps(record)),
                mean=self.estimator.mean_, components=self.estimator.components_,
                explained_variance=self.estimator.explained_variance_,
                explained_variance_ratio=self.estimator.explained_variance_ratio_,
                singular_values=self.estimator.singular_values_,
                noise_variance=np.array(self.estimator.noise_variance_, dtype=np.float32),
            )

    @classmethod
    def load(cls, path: Path, *, fold: int, repeat: int) -> PCABaseline:
        """Restore the fitted estimator; stochastic fits cannot cross repeat IDs."""
        if type(fold) is not int or fold not in FOLDS or repeat not in REPEATS:
            raise ValueError("Invalid fold or repeat")
        with np.load(path, allow_pickle=False) as saved:
            expected = {"metadata", "mean", "components", "explained_variance",
                        "explained_variance_ratio", "singular_values", "noise_variance"}
            if set(saved.files) != expected:
                raise ValueError("Unexpected PCA checkpoint fields")
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata.get("schema_version") != 1 or metadata.get("condition") != "B1":
                raise ValueError("Unsupported PCA checkpoint schema")
            fit = metadata["fit"]
            record = PCAFitRecord(**{**fit, "sample_ids": tuple(fit["sample_ids"])})
            if record.fold != fold or record.seed != run_seed("pca", fold, record.repeat):
                raise ValueError("PCA checkpoint fold or seed mismatch")
            if record.sklearn_version != sklearn.__version__:
                raise ValueError("PCA checkpoint sklearn version differs from the current runtime")
            if record.solver not in ("full", "covariance_eigh", "randomized", "arpack"):
                raise ValueError("Unknown saved PCA solver")
            if record.train_pixel_count < 16 or not record.sample_ids:
                raise ValueError("Invalid PCA training provenance")
            if repeat != record.repeat and record.solver not in ("full", "covariance_eigh"):
                raise ValueError("Stochastic PCA checkpoint belongs to a different repeat")
            estimator = PCA(n_components=16)
            shapes = {"mean": (256,), "components": (16, 256), "explained_variance": (16,),
                      "explained_variance_ratio": (16,), "singular_values": (16,),
                      "noise_variance": ()}
            for name, shape in shapes.items():
                values = saved[name]
                if (values.shape != shape or values.dtype != np.dtype("float32")
                        or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid PCA checkpoint array: {name}")
                setattr(estimator, f"{name}_", values.copy())
        estimator.n_features_in_ = 256
        estimator.n_components_ = 16
        estimator.n_samples_ = record.train_pixel_count
        estimator._fit_svd_solver = record.solver
        return cls(estimator, record)


def fit_pca(data: FoldData, *, repeat: int, chunk_pixels: int = 2048) -> PCABaseline:
    """Fit PCA only to the common train selection, preserving the caller's NumPy RNG.

    random_state=None is the fixed sklearn setting. This synchronous context
    seeds and restores NumPy's legacy RNG for stochastic solvers; it must not
    run concurrently with other threads consuming that global RNG.
    """
    seed = run_seed("pca", data.fold, repeat)
    spectra = data.train_matrix(chunk_pixels=chunk_pixels)
    if len(spectra) < 16:
        raise ValueError("PCA needs at least 16 train rows")
    estimator = PCA(n_components=16)
    fixed = experiment_config()["pca"]
    if any(name not in fixed or value != fixed[name]
           for name, value in estimator.get_params().items()):
        raise ValueError("Installed PCA defaults differ from the fixed protocol")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        estimator.fit(spectra)
    finally:
        np.random.set_state(state)
    for name in ("mean_", "components_", "explained_variance_",
                 "explained_variance_ratio_", "singular_values_"):
        values = getattr(estimator, name)
        if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
            raise ValueError(f"PCA fit produced non-finite or non-FP32 {name}")
    return PCABaseline(estimator, PCAFitRecord(
        data.fold, repeat, seed, data.train_sample_ids, len(spectra),
        estimator._fit_svd_solver, sklearn.__version__, np.__version__,
    ))
