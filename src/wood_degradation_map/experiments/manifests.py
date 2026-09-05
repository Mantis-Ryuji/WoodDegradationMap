"""Create and reload fixed CV inputs without training or reading spectra."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .config import (
    FOLDS,
    PIXELS_PER_SAMPLE,
    experiment_config,
    sampling_seed,
    seed_plan,
    split_seed,
)
from .input_validation import InputInventory, SampleInput


@dataclass(frozen=True)
class CVManifest:
    """One test-fold assignment per KYOw and shared train pixels per fold."""

    folds: pd.DataFrame
    train_pixels: dict[int, pd.DataFrame]
    q: int


FOLD_COLUMNS = ["sample_id", "test_fold", "file", "saved_pixel_count", "height", "width"]
PIXEL_COLUMNS = ["fold", "sample_id", "sampling_seed", "hdf5_row", "pixel_row", "pixel_col"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _samples(inventory: InputInventory) -> dict[str, SampleInput]:
    samples = {sample.sample_id: sample for sample in inventory.samples}
    _require(len(samples) == len(inventory.samples), "Duplicate inventory sample IDs")
    _require(len(samples) >= len(FOLDS), "At least five samples are required")
    return samples


def _read_coordinates(sample: SampleInput) -> tuple[np.ndarray, np.ndarray]:
    """Load only coordinates and mask for one sample; never read spectra."""
    with h5py.File(sample.path, "r") as handle:
        for name, expected in (
            ("sample_id", sample.sample_id), ("saved_pixel_count", sample.saved_pixel_count),
            ("schema_version", 2),
        ):
            _require(handle.attrs.get(name) == expected,
                     f"{sample.sample_id}: HDF5 attribute {name} mismatch")
        for name, shape in (
            ("pixel_row_col", (sample.saved_pixel_count, 2)),
            ("valid_spectrum_mask", (sample.height, sample.width)),
        ):
            _require(name in handle and isinstance(handle[name], h5py.Dataset),
                     f"{sample.sample_id}: missing dataset {name}")
            _require(handle[name].shape == shape and handle[name].dtype.kind in "biu",
                     f"{sample.sample_id}: invalid {name} shape or dtype")
        _require(handle["pixel_row_col"].dtype.kind in "iu", "Coordinates must be integers")
        return handle["pixel_row_col"][:], handle["valid_spectrum_mask"][:]


def _selected_coordinates(
    sample: SampleInput, rows: np.ndarray, coordinates: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    selected = coordinates[rows]
    _require(((selected >= 0) & (selected < [sample.height, sample.width])).all(),
             f"{sample.sample_id}: sampled coordinate outside image")
    _require(len(np.unique(selected, axis=0)) == len(rows),
             f"{sample.sample_id}: duplicate sampled coordinates")
    _require((mask[selected[:, 0], selected[:, 1]] == 1).all(),
             f"{sample.sample_id}: sampled coordinate outside valid mask")
    return selected


def create_cv_manifest(inventory: InputInventory, *, q: int = PIXELS_PER_SAMPLE) -> CVManifest:
    """Plan balanced random folds and uniform nonreplacement train samples.

    The low-level q argument supports small fixtures. Production saving only
    accepts the protocol's q=8192. Each sample's coordinate/mask arrays are read
    once and released; no SNV or reflectance dataset is loaded.
    """
    _require(type(q) is int and q > 0, "q must be a positive integer")
    samples = _samples(inventory)
    _require(all(sample.saved_pixel_count >= q for sample in samples.values()),
             "A sample has fewer than q saved pixels; sampling with replacement is forbidden")
    rng = np.random.Generator(np.random.PCG64(split_seed()))
    test_folds = {
        str(sample_id): fold
        for fold, ids in zip(FOLDS, np.array_split(rng.permutation(sorted(samples)), len(FOLDS)),
                             strict=True)
        for sample_id in ids
    }
    records: list[dict[str, object]] = []
    pixel_frames: dict[int, list[pd.DataFrame]] = {fold: [] for fold in FOLDS}
    for sample_id, sample in sorted(samples.items()):
        records.append({"sample_id": sample_id, "test_fold": test_folds[sample_id],
                        "file": f"samples/{sample_id}.h5",
                        "saved_pixel_count": sample.saved_pixel_count,
                        "height": sample.height, "width": sample.width})
        coordinates, mask = _read_coordinates(sample)
        for fold in FOLDS:
            if test_folds[sample_id] == fold:
                continue
            seed = sampling_seed(fold, sample_id)
            generator = np.random.Generator(np.random.PCG64(seed))
            rows = np.sort(generator.choice(sample.saved_pixel_count, size=q, replace=False))
            selected = _selected_coordinates(sample, rows, coordinates, mask)
            pixel_frames[fold].append(pd.DataFrame({
                "fold": fold, "sample_id": sample_id, "sampling_seed": seed,
                "hdf5_row": rows, "pixel_row": selected[:, 0], "pixel_col": selected[:, 1],
            }))
    plan = CVManifest(pd.DataFrame(records, columns=FOLD_COLUMNS), {
        fold: pd.concat(frames, ignore_index=True)[PIXEL_COLUMNS]
        for fold, frames in pixel_frames.items()
    }, q)
    validate_cv_manifest(plan, inventory, check_coordinates=False)
    return plan


def _columns(table: pd.DataFrame, expected: list[str], integers: list[str]) -> None:
    _require(list(table.columns) == expected, f"Expected columns {expected}")
    _require(not table.isna().any().any(), "Manifest contains missing values")
    _require(table["sample_id"].map(lambda value: isinstance(value, str)).all(),
             "Manifest sample IDs must be strings")
    for column in integers:
        _require(is_integer_dtype(table[column].dtype), f"{column} must contain integers")


def validate_cv_manifest(
    plan: CVManifest, inventory: InputInventory, *, check_coordinates: bool = True,
) -> None:
    """Reject leakage, omissions, duplicate rows, wrong seeds and coordinate drift."""
    samples = _samples(inventory)
    _require(type(plan.q) is int and plan.q > 0, "q must be positive")
    _columns(plan.folds, FOLD_COLUMNS, ["test_fold", "saved_pixel_count", "height", "width"])
    _require(not plan.folds["sample_id"].duplicated().any(), "A KYOw has multiple test folds")
    _require(set(plan.folds["sample_id"]) == set(samples), "Fold sample IDs differ from inventory")
    _require(set(plan.folds["test_fold"]) == set(FOLDS), "All five test folds must be present")
    counts = plan.folds.groupby("test_fold").size()
    _require(int(counts.max() - counts.min()) <= 1, "Test folds must be balanced")
    test_folds: dict[str, int] = {}
    for row in plan.folds.itertuples(index=False):
        sample = samples[row.sample_id]
        _require((row.saved_pixel_count, row.height, row.width, row.file) == (
            sample.saved_pixel_count, sample.height, sample.width, f"samples/{sample.sample_id}.h5",
        ), f"{sample.sample_id}: saved input identity differs")
        test_folds[sample.sample_id] = int(row.test_fold)
    _require(set(plan.train_pixels) == set(FOLDS),
             "Train pixel manifests must contain all five folds")
    for fold, pixels in plan.train_pixels.items():
        _columns(pixels, PIXEL_COLUMNS,
                 ["fold", "sampling_seed", "hdf5_row", "pixel_row", "pixel_col"])
        _require((pixels["fold"] == fold).all(), f"Wrong fold column in fold {fold}")
        expected_train = {sample_id for sample_id, test in test_folds.items() if test != fold}
        _require(set(pixels["sample_id"]) == expected_train,
                 f"Fold {fold}: train/test leakage or missing train samples")
        _require((pixels.groupby("sample_id").size() == plan.q).all(),
                 f"Fold {fold}: each train sample must contain exactly q rows")
        _require(not pixels.duplicated(["sample_id", "hdf5_row"]).any(),
                 f"Fold {fold}: duplicate sampled HDF5 rows")
        _require(not pixels.duplicated(["sample_id", "pixel_row", "pixel_col"]).any(),
                 f"Fold {fold}: duplicate sampled coordinates")
        for sample_id, group in pixels.groupby("sample_id", sort=False):
            sample = samples[sample_id]
            rows = group["hdf5_row"].to_numpy()
            _require(((rows >= 0) & (rows < sample.saved_pixel_count)).all(),
                     f"{sample_id}: sampled HDF5 row out of bounds")
            _require((np.diff(rows) > 0).all(), f"{sample_id}: HDF5 rows must be sorted")
            _require((group["sampling_seed"] == sampling_seed(fold, sample_id)).all(),
                     f"{sample_id}: sampling seed differs from fixed plan")
            _require(((group["pixel_row"] >= 0) & (group["pixel_row"] < sample.height)
                      & (group["pixel_col"] >= 0) & (group["pixel_col"] < sample.width)).all(),
                     f"{sample_id}: manifest coordinate out of bounds")
    if check_coordinates:
        # Read each source once even though it appears in four train folds.
        for sample_id, sample in samples.items():
            coordinates, mask = _read_coordinates(sample)
            for fold, pixels in plan.train_pixels.items():
                if fold == test_folds[sample_id]:
                    continue
                group = pixels.loc[pixels["sample_id"] == sample_id]
                selected = _selected_coordinates(sample, group["hdf5_row"].to_numpy(),
                                                 coordinates, mask)
                _require(np.array_equal(selected, group[["pixel_row", "pixel_col"]].to_numpy()),
                         f"{sample_id}: saved coordinates differ from HDF5 rows")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def input_snapshot(
    inventory: InputInventory, processed_dir: Path, metadata: Path,
) -> dict[str, object]:
    """Fingerprint small input tables; HDF5 identity uses file size and mtime.

    No full HDF5 hashes are computed. Equal sizes/timestamps cannot prove byte
    identity; selected coordinates are checked again when the manifest is loaded.
    """
    return {
        "preprocessing_id": inventory.preprocessing_id,
        "small_file_sha256": {
            name: _digest(processed_dir / name)
            for name in ("config.json", "manifest.parquet", "sample_quality.parquet")
        },
        "metadata_sha256": _digest(metadata),
        "samples": [
            {"sample_id": sample.sample_id, "file": f"samples/{sample.sample_id}.h5",
             "saved_pixel_count": sample.saved_pixel_count, "height": sample.height,
             "width": sample.width, "bytes": sample.path.stat().st_size,
             "mtime_ns": sample.path.stat().st_mtime_ns}
            for sample in sorted(inventory.samples, key=lambda item: item.sample_id)
        ],
    }


def _write_json(path: Path, value: object) -> None:
    # Exclusive creation also protects against accidental reuse of an output directory.
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def fold_summary(plan: CVManifest) -> list[dict[str, int]]:
    """Summarize planned counts, not completed model updates."""
    return [
        {"fold": fold, "test_samples": int((plan.folds["test_fold"] == fold).sum()),
         "train_samples": int((plan.folds["test_fold"] != fold).sum()),
         "train_pixels": len(plan.train_pixels[fold]),
         "batches_per_epoch": len(plan.train_pixels[fold]) // 1024,
         "planned_updates_per_run": 800 * (len(plan.train_pixels[fold]) // 1024)}
        for fold in FOLDS
    ]


def create_manifest_bundle(
    output_dir: Path, inventory: InputInventory, processed_dir: Path, metadata: Path,
) -> CVManifest:
    """Create a new production bundle; existing directories are never overwritten."""
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; use the check command: {output_dir}")
    config = experiment_config()
    _require(inventory.preprocessing_id == config["preprocessing_id"],
             "Production preprocessing ID differs from fixed config")
    _require(sorted(sample.sample_id for sample in inventory.samples)
             == sorted(config["split"]["adopted_sample_ids"]),
             "Input sample IDs differ from the approved sample set")
    before = input_snapshot(inventory, processed_dir, metadata)
    plan = create_cv_manifest(inventory)
    test_folds = dict(zip(plan.folds["sample_id"], plan.folds["test_fold"], strict=True))
    seeds = seed_plan(test_folds)
    _require(input_snapshot(inventory, processed_dir, metadata) == before,
             "Inputs changed while creating the manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config").mkdir()
    manifests = output_dir / "manifests"
    (manifests / "train_pixels").mkdir(parents=True)
    files = ["config/experiment.json", "config/seeds.json", "manifests/inputs.json",
             "manifests/folds.parquet"]
    _write_json(output_dir / files[0], config)
    _write_json(output_dir / files[1], seeds)
    _write_json(output_dir / files[2], before)
    plan.folds.to_parquet(output_dir / files[3], index=False)
    for fold, pixels in plan.train_pixels.items():
        name = f"manifests/train_pixels/fold_{fold}.parquet"
        pixels.to_parquet(output_dir / name, index=False)
        files.append(name)
    # The completion record is written last. Interrupted bundles fail on reload.
    _write_json(manifests / "complete.json", {
        "schema_version": 1, "status": "complete",
        "artifact_sha256": {name: _digest(output_dir / name) for name in files},
        "fold_summary": fold_summary(plan),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "h5py_version": h5py.__version__,
    })
    return plan


def load_manifest_bundle(
    output_dir: Path, inventory: InputInventory, processed_dir: Path, metadata: Path,
) -> CVManifest:
    """Read the saved selections and validate them; never regenerate a split."""
    completion = _read_json(output_dir / "manifests/complete.json")
    _require(completion.get("schema_version") == 1 and completion.get("status") == "complete",
             "Manifest bundle is incomplete or has an unsupported schema")
    expected_files = {"config/experiment.json", "config/seeds.json", "manifests/inputs.json",
                      "manifests/folds.parquet"} | {
        f"manifests/train_pixels/fold_{fold}.parquet" for fold in FOLDS
    }
    digests = completion.get("artifact_sha256")
    _require(isinstance(digests, dict) and set(digests) == expected_files,
             "Manifest completion record has unexpected artifact paths")
    for name in sorted(expected_files):
        _require(_digest(output_dir / name) == digests[name], f"Manifest artifact changed: {name}")
    _require(_read_json(output_dir / "config/experiment.json") == experiment_config(),
             "Saved experiment config differs from the fixed protocol")
    before = input_snapshot(inventory, processed_dir, metadata)
    _require(_read_json(output_dir / "manifests/inputs.json") == before,
             "Source inputs changed since manifest creation; do not silently regenerate")
    plan = CVManifest(pd.read_parquet(output_dir / "manifests/folds.parquet"), {
        fold: pd.read_parquet(output_dir / f"manifests/train_pixels/fold_{fold}.parquet")
        for fold in FOLDS
    }, PIXELS_PER_SAMPLE)
    validate_cv_manifest(plan, inventory)
    test_folds = dict(zip(plan.folds["sample_id"], plan.folds["test_fold"], strict=True))
    _require(_read_json(output_dir / "config/seeds.json") == seed_plan(test_folds),
             "Saved seed plan differs from fixed seed derivation")
    _require(completion.get("fold_summary") == fold_summary(plan), "Fold summary differs")
    _require(input_snapshot(inventory, processed_dir, metadata) == before,
             "Inputs changed during manifest validation")
    return plan
