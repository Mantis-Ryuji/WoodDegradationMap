"""One global PCA fit and a parameter-free B0 contract, with checked reload."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sklearn
from sklearn.decomposition import PCA

from .baselines import B0Baseline, PCABaseline
from .global_manifest import GlobalData, global_config, global_seed
from .manifests import _digest, _read_json, _require, _write_json


@dataclass(frozen=True)
class GlobalPCARecord:
    scope: str
    repeat: int
    seed: int
    sample_ids: tuple[str, ...]
    train_pixel_count: int
    solver: str
    sklearn_version: str
    numpy_version: str


@dataclass(frozen=True)
class GlobalPCA(PCABaseline):
    record: GlobalPCARecord

    @classmethod
    def load(cls, path: Path) -> GlobalPCA:
        with np.load(path, allow_pickle=False) as saved:
            shapes = {"mean": (256,), "components": (16, 256), "explained_variance": (16,),
                      "explained_variance_ratio": (16,), "singular_values": (16,),
                      "noise_variance": ()}
            _require(set(saved.files) == {"metadata", *shapes}, "Unexpected global PCA fields")
            metadata = json.loads(str(saved["metadata"].item()))
            _require(metadata.get("schema_version") == 1 and metadata.get("condition") == "B1",
                     "Unsupported global PCA schema")
            fit = metadata["fit"]
            record = GlobalPCARecord(**{**fit, "sample_ids": tuple(fit["sample_ids"])})
            _require(record.scope == "global_fit" and record.repeat == 1
                     and record.seed == global_seed("pca"), "Global PCA scope or seed differs")
            _require(record.sklearn_version == sklearn.__version__
                     and record.numpy_version == np.__version__, "Global PCA runtime differs")
            _require(record.solver in ("full", "covariance_eigh", "randomized", "arpack")
                     and record.train_pixel_count >= 16 and bool(record.sample_ids),
                     "Invalid global PCA fit provenance")
            estimator = PCA(n_components=16)
            for name, shape in shapes.items():
                values = saved[name]
                _require(values.shape == shape and values.dtype == np.dtype("float32")
                         and np.isfinite(values).all(), f"Invalid global PCA array: {name}")
                # NPZ preserves C/F layout. Changing it can change FP32 BLAS
                # accumulation and break the transform roundtrip after centering.
                setattr(estimator, f"{name}_", values.copy(order="K"))
        estimator.n_features_in_ = 256
        estimator.n_components_ = 16
        estimator.n_samples_ = record.train_pixel_count
        estimator._fit_svd_solver = record.solver
        return cls(estimator, record)


def fit_global_pca(data: GlobalData) -> GlobalPCA:
    spectra = data.train_matrix()
    _require(len(spectra) >= 16, "Global PCA needs at least 16 fit rows")
    estimator = PCA(n_components=16)
    fixed = global_config()["pca"]
    _require(all(name in fixed and value == fixed[name]
                 for name, value in estimator.get_params().items()), "PCA defaults changed")
    state = np.random.get_state()
    try:
        np.random.seed(global_seed("pca"))
        estimator.fit(spectra)
    finally:
        np.random.set_state(state)
    for name in ("mean_", "components_", "explained_variance_",
                 "explained_variance_ratio_", "singular_values_"):
        values = getattr(estimator, name)
        _require(values.dtype == np.dtype("float32") and np.isfinite(values).all(),
                 f"Global PCA produced invalid {name}")
    return GlobalPCA(estimator, GlobalPCARecord(
        "global_fit", 1, global_seed("pca"), data.train_sample_ids, len(spectra),
        estimator._fit_svd_solver, sklearn.__version__, np.__version__,
    ))


def _baseline_contract(experiment: Path) -> dict[str, object]:
    directory = Path(__file__).parent
    return {
        "scope": "global_fit", "config": global_config(),
        "manifest_artifact_sha256": _read_json(
            experiment / "manifests/complete.json",
        )["artifact_sha256"],
        "code_sha256": {name: _digest(directory / name) for name in (
            "global_baselines.py", "global_manifest.py", "baselines.py", "data.py",
            "config.py", "manifests.py", "input_validation.py",
        )},
    }


def global_baselines(experiment: Path, data: GlobalData, *, fit: bool) -> dict[str, object]:
    results = experiment / "results/baselines"
    checkpoints = experiment / "checkpoints/baselines"
    contract = _baseline_contract(experiment)
    # Probe one selected chunk. This is transform validation, not a held-out metric.
    batches = data.batches("train", chunk_pixels=2048)
    try:
        probe = next(batches)
    finally:
        batches.close()
    if fit:
        if results.exists() or checkpoints.exists():
            raise FileExistsError("Global baseline output exists; use baseline-check")
        b0, pca = B0Baseline(), fit_global_pca(data)
        b0_diagnostics = asdict(b0.transform(probe.snv).diagnostics)
        expected = pca.transform(probe.snv)
        # Validate staged files first, so a failed roundtrip leaves the output
        # paths available for a retry without deleting a partly published fit.
        with TemporaryDirectory(prefix=".baseline-fit-", dir=experiment) as temporary:
            staging = Path(temporary)
            b0.save(staging / "b0.json")
            pca.save(staging / "pca.npz")
            restored = GlobalPCA.load(staging / "pca.npz")
            error = float(np.max(np.abs(expected.values - restored.transform(probe.snv).values)))
            _require(error <= 1e-6,
                     f"Global PCA roundtrip mismatch: maximum absolute error {error:.9g} > 1e-6")
            results.mkdir(parents=True, exist_ok=False)
            checkpoints.mkdir(parents=True, exist_ok=False)
            (staging / "b0.json").replace(results / "b0.json")
            (staging / "pca.npz").replace(checkpoints / "pca.npz")
        report = {
            "status": "fitted_and_roundtrip_checked", "contract": contract,
            "pca_fit": json.loads(json.dumps(asdict(pca.record))),
            "artifact_sha256": {"b0.json": _digest(results / "b0.json"),
                                "pca.npz": _digest(checkpoints / "pca.npz")},
            "pca_save_load_probe_absolute_error_max": error,
            "probe": {"sample_id": probe.sample_id, "hdf5_rows": probe.hdf5_rows.tolist(),
                      "B0": b0_diagnostics, "B1": asdict(expected.diagnostics)},
        }
        _write_json(results / "completion.json", report)
    else:
        report = _read_json(results / "completion.json")
        _require(report.get("status") == "fitted_and_roundtrip_checked"
                 and report.get("contract") == contract, "Global baseline contract changed")
        _require(report.get("artifact_sha256") == {
            "b0.json": _digest(results / "b0.json"), "pca.npz": _digest(checkpoints / "pca.npz"),
        }, "Global baseline artifact changed")
        b0, pca = B0Baseline.load(results / "b0.json"), GlobalPCA.load(checkpoints / "pca.npz")
        _require(pca.record.sample_ids == data.train_sample_ids
                 and pca.record.train_pixel_count == data.train_pixel_count
                 and report.get("pca_fit") == json.loads(json.dumps(asdict(pca.record))),
                 "Global PCA fit provenance changed")
        b0.transform(probe.snv)
        pca.transform(probe.snv)
    return report
