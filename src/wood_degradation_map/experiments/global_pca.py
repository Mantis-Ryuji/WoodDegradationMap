"""CPU PCA projections of the shared global clustering representations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

from .artifact_output import publish_directory

CONDITIONS = ("B0", "B1", "A0", "A1", "M00", "M11")
INPUT_DIRECTORY = "checkpoints/pca_projection/global_k8_v1/inputs"
RESULT_DIRECTORY = "results/pca/global_k8_v1"
FIGURE_DIRECTORY = "results/figures/global_k8_v1/pca-latent-2d"
BASELINE_PROJECTION = "B1_projection.npz"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def settings(condition: str | None = None) -> dict:
    if condition == "B1":
        return {"n_components": 2, "source": "saved_global_baseline_pca",
                "component_order": "descending_singular_values", "score_normalization": "none"}
    return {"n_components": 2, "svd_solver": "covariance_eigh", "whiten": False}


def baseline_component_indices(singular_values: np.ndarray) -> np.ndarray:
    require(singular_values.shape == (16,) and np.isfinite(singular_values).all()
            and (singular_values >= 0).all() and singular_values.max() > 0,
            "Invalid baseline PCA singular values")
    return np.argsort(-singular_values, kind="stable")[:2]


def load_baseline_projection(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / BASELINE_PROJECTION, allow_pickle=False) as saved:
        shapes = {"mean": (256,), "components": (2, 256), "explained_variance": (2,),
                  "explained_variance_ratio": (2,), "singular_values": (2,),
                  "source_component_index": (2,)}
        require(set(saved.files) == set(shapes), "Invalid baseline PCA projection fields")
        projection = {name: saved[name].copy() for name in shapes}
    require(all(projection[name].shape == shape and np.isfinite(projection[name]).all()
                for name, shape in shapes.items()), "Invalid baseline PCA projection arrays")
    indices = projection["source_component_index"]
    require(indices.dtype.kind in "iu" and len(np.unique(indices)) == 2
            and np.isin(indices, np.arange(16)).all()
            and projection["singular_values"][0] >= projection["singular_values"][1] >= 0,
            "Invalid baseline PCA component order")
    return projection


def finish_artifacts(directory: Path, status: str) -> dict:
    hashes = {str(p.relative_to(directory)).replace("\\", "/"): digest(p)
              for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "completion.json"}
    record = {"status": status, "artifact_sha256": hashes}
    write_json(directory / "completion.json", record)
    return record


def check_artifacts(directory: Path, status: str) -> dict:
    record = read_json(directory / "completion.json")
    require(record.get("status") == status, f"Incomplete {directory.name}")
    hashes = record["artifact_sha256"]
    actual = {str(p.relative_to(directory)).replace("\\", "/")
              for p in directory.rglob("*") if p.is_file() and p.name != "completion.json"}
    require(set(hashes) == actual, f"Artifact coverage differs: {directory}")
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and digest(path) == expected,
                f"Artifact changed: {name}")
    return record


def check_inputs(directory: Path) -> tuple[dict, pd.DataFrame]:
    check_artifacts(directory, "pca_inputs_completed")
    record = read_json(directory / "inputs.json")
    require(record.get("schema_version") == 2, "PCA inputs need regeneration for direct B1 axes")
    require(record.get("conditions") == list(CONDITIONS), "PCA condition order differs")
    require(type(record.get("plot_seed")) is int, "Missing scatter-order seed")
    pixels = pd.read_csv(directory / "pixels.csv")
    require(len(pixels) == record["pixels"] and len(pixels) > 2
            and np.array_equal(pixels.point_id, np.arange(len(pixels)))
            and not pixels.duplicated(["sample_id", "hdf5_row"]).any()
            and not pixels.duplicated(["sample_id", "pixel_row", "pixel_col"]).any(),
            "Invalid PCA pixel identity")
    counts = pixels.groupby("sample_id").size()
    require(counts.index.tolist() == record["sample_ids"]
            and (counts == record["pixels_per_sample"]).all(), "PCA sample balance differs")
    for condition in CONDITIONS:
        values = np.load(directory / f"{condition}.npy", mmap_mode="r", allow_pickle=False)
        dimension = 256 if condition == "B0" else 2 if condition == "B1" else 16
        require(values.shape == (len(pixels), dimension) and values.dtype == np.float32,
                f"{condition}: invalid representation shape/dtype")
        for start in range(0, len(values), 8192):
            block = values[start:start + 8192]
            require(np.isfinite(block).all(), f"{condition}: non-finite representation")
            if condition != "B1":
                require(np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5, rtol=0),
                        f"{condition}: invalid unit representation")
        labels = pixels[f"{condition}_cluster"].to_numpy()
        require(labels.dtype.kind in "iu" and np.isin(labels, np.arange(1, 9)).all(),
                f"{condition}: invalid display clusters")
    load_baseline_projection(directory)
    return record, pixels


def fit_projection(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    import scipy
    import sklearn
    from sklearn.decomposition import PCA

    require(values.ndim == 2 and min(values.shape) >= 2 and np.isfinite(values).all(),
            "Invalid PCA input matrix")
    started = time.perf_counter()
    # Accumulate the covariance in FP64; the saved source vectors remain unchanged.
    model = PCA(**settings())
    coordinates = model.fit_transform(np.asarray(values, dtype=np.float64))
    projection = {"mean": model.mean_, "components": model.components_,
                  "explained_variance": model.explained_variance_,
                  "explained_variance_ratio": model.explained_variance_ratio_,
                  "singular_values": model.singular_values_}
    require(np.isfinite(coordinates).all()
            and all(np.isfinite(value).all() for value in projection.values())
            and model.explained_variance_[0] > 0, "PCA input has no finite variance")
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "numpy": np.__version__, "scipy": scipy.__version__,
               "scikit_learn": sklearn.__version__, "device": "cpu", "dtype": "float64",
               "wall_seconds": time.perf_counter() - started}
    return coordinates, projection, runtime


def check_projection(
    directory: Path, input_digest: str, condition: str, values: np.ndarray,
    baseline_projection: dict[str, np.ndarray] | None = None,
) -> dict:
    completion = directory / "completion.json"
    if not completion.is_file():
        raise FileNotFoundError(
            f"{condition}: PCA fit is not complete ({completion}). "
            "Run global_pca.py fit before rendering."
        )
    check_artifacts(directory, "pca_projection_completed")
    record = read_json(directory / "run.json")
    require(record["condition"] == condition and record["input_completion_sha256"] == input_digest
            and record["code_sha256"] == digest(Path(__file__))
            and record["settings"] == settings(condition), "PCA source/settings differ")
    coordinates = np.load(directory / "coordinates.npy", allow_pickle=False)
    require(coordinates.shape == (len(values), 2) and coordinates.dtype == np.float64
            and np.isfinite(coordinates).all(), f"{condition}: invalid saved coordinates")
    dimension = 256 if condition == "B1" else values.shape[1]
    basis_tolerance = 1e-5 if condition == "B1" else 1e-10
    with np.load(directory / "projection.npz", allow_pickle=False) as saved:
        mean, components = saved["mean"], saved["components"]
        ratio, variance = saved["explained_variance_ratio"], saved["explained_variance"]
        require(mean.shape == (dimension,) and components.shape == (2, dimension)
                and ratio.shape == variance.shape == (2,)
                and all(np.isfinite(saved[name]).all() for name in saved.files)
                and (variance >= 0).all() and (ratio >= 0).all()
                and ratio.sum() <= 1 + basis_tolerance
                and np.allclose(components @ components.T, np.eye(2),
                                atol=basis_tolerance, rtol=0),
                f"{condition}: invalid saved PCA basis")
        if condition == "B1":
            require(baseline_projection is not None
                    and set(saved.files) == set(baseline_projection)
                    and all(np.array_equal(saved[name], baseline_projection[name])
                            for name in saved.files), "B1: saved baseline PCA basis differs")
            require(np.array_equal(coordinates, values.astype(np.float64)),
                    "B1: direct baseline PCA scores differ")
            return record
        # Deterministic source-row probes validate the stored affine transform without refitting.
        probe = np.unique(np.linspace(0, len(values) - 1, min(32, len(values)), dtype=int))
        expected = (values[probe].astype(np.float64) - mean) @ components.T
        require(np.allclose(coordinates[probe], expected, atol=1e-10, rtol=1e-10),
                f"{condition}: PCA projection roundtrip differs")
    return record


def run_pca(experiment: Path, *, resume: bool = False, overwrite: bool = False) -> dict:
    require(not (resume and overwrite), "Choose resume or overwrite")
    inputs = experiment / INPUT_DIRECTORY
    _, pixels = check_inputs(inputs)
    baseline_projection = load_baseline_projection(inputs)
    input_digest = digest(inputs / "completion.json")
    output = experiment / RESULT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        target = output / condition
        values = np.load(inputs / f"{condition}.npy", mmap_mode="r", allow_pickle=False)
        if target.exists() and not overwrite:
            if not resume:
                raise FileExistsError(f"{target} exists; use --resume to verify/skip")
            check_projection(target, input_digest, condition, values, baseline_projection)
            print(f"{condition}: verified existing PCA", flush=True)
            continue
        if condition == "B1":
            print("B1: using saved baseline PC1/PC2 scores; no PCA fit", flush=True)
            coordinates = values.astype(np.float64)
            projection = baseline_projection
            runtime = {"python": platform.python_version(), "platform": platform.platform(),
                       "numpy": np.__version__, "device": "cpu", "dtype": "float64",
                       "operation": "copy saved FP32 baseline scores without normalization"}
        else:
            print(f"{condition}: CPU PCA {values.shape} -> PC1/PC2", flush=True)
            coordinates, projection, runtime = fit_projection(values)
        with TemporaryDirectory(prefix=".pca-fit-", dir=output) as temporary:
            stage = Path(temporary) / "result"
            stage.mkdir()
            np.save(stage / "coordinates.npy", coordinates, allow_pickle=False)
            np.savez_compressed(stage / "projection.npz", **projection)
            pd.DataFrame({"point_id": pixels.point_id, "pc1": coordinates[:, 0],
                          "pc2": coordinates[:, 1],
                          "cluster": pixels[f"{condition}_cluster"]}).to_csv(
                              stage / "coordinates.csv", index=False)
            pd.DataFrame({"component": ["PC1", "PC2"],
                          "explained_variance": projection["explained_variance"],
                          "explained_variance_ratio": projection["explained_variance_ratio"],
                          "cumulative_ratio": np.cumsum(projection["explained_variance_ratio"]),
                          "singular_value": projection["singular_values"]}).to_csv(
                              stage / "explained_variance.csv", index=False)
            pd.DataFrame({"feature_index": np.arange(len(projection["mean"])),
                          "mean": projection["mean"], "pc1": projection["components"][0],
                          "pc2": projection["components"][1]}).to_csv(
                              stage / "components.csv", index=False)
            write_json(stage / "run.json", {"condition": condition,
                "input_completion_sha256": input_digest, "settings": settings(condition),
                "runtime": runtime, "code_sha256": digest(Path(__file__)),
                "fit_scope": ("reuse baseline PCA fitted on shared global fit pixels; raw scores"
                              if condition == "B1" else
                              "shared global fit pixels; centered, no extra scaling or whitening")})
            finish_artifacts(stage, "pca_projection_completed")
            require(digest(inputs / "completion.json") == input_digest, "PCA inputs changed")
            check_projection(stage, input_digest, condition, values, baseline_projection)
            publish_directory(stage, target, root=experiment)
        del values, coordinates
    return check_pca(experiment)


def check_pca(experiment: Path) -> dict:
    inputs = experiment / INPUT_DIRECTORY
    _, pixels = check_inputs(inputs)
    baseline_projection = load_baseline_projection(inputs)
    input_digest = digest(inputs / "completion.json")
    for condition in CONDITIONS:
        values = np.load(inputs / f"{condition}.npy", mmap_mode="r", allow_pickle=False)
        check_projection(experiment / RESULT_DIRECTORY / condition, input_digest, condition,
                         values, baseline_projection)
    return {"status": "validated_global_pca", "checks_passed": True,
            "conditions": list(CONDITIONS), "pixels": len(pixels)}
