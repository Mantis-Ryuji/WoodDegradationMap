"""Bacteria-ID source checks, fixed splits and per-spectrum SNV preparation."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.model_selection import StratifiedKFold, train_test_split

from .bacteria_config import TABLE4_CLASSES, bacteria_config, seed_manifest
from .manifests import _read_json, _write_json


SOURCE_COUNTS = {"reference": 60_000, "finetune": 3_000, "test": 3_000}
PER_CLASS = {"reference": 2_000, "finetune": 100, "test": 100}
CHANNELS = 1000


def snv_rows(values: np.ndarray, *, subset: str, start: int) -> np.ndarray:
    """SNV with sample standard deviation; fail on invalid rows without exclusion."""
    if values.ndim != 2 or values.shape[1] != CHANNELS or len(values) == 0:
        raise ValueError(f"{subset}: expected a nonempty (N, {CHANNELS}) chunk")
    bad = ~np.isfinite(values).all(axis=1)
    if bad.any():
        raise ValueError(f"{subset}: nonfinite spectrum at source row {start + int(np.flatnonzero(bad)[0])}")
    work = values.astype(np.float64, copy=False)
    center = work.mean(axis=1, keepdims=True)
    scale = work.std(axis=1, ddof=1, keepdims=True)
    bad = ~np.isfinite(scale[:, 0]) | (scale[:, 0] <= 0)
    if bad.any():
        raise ValueError(f"{subset}: zero/nonfinite sample SD at source row {start + int(np.flatnonzero(bad)[0])}")
    result = ((work - center) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{subset}: nonfinite SNV output in chunk starting at {start}")
    return result


def _source_array(path: Path, shape: tuple[int, ...], *, labels: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required Bacteria-ID source file is missing: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != shape or (value.dtype.kind not in "iuf" if labels
                                else value.dtype.kind != "f"):
        kind = "numeric labels" if labels else "floating spectra"
        raise ValueError(f"{path}: expected {shape} {kind}, got {value.shape} {value.dtype}")
    return value


def _checked_labels(path: Path, count: int, per_class: int) -> np.ndarray:
    raw = _source_array(path, (count,), labels=True)
    if raw.dtype.kind == "f" and (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()):
        raise ValueError(f"{path}: floating labels must be finite exact integers")
    labels = np.asarray(raw, dtype=np.int64)
    if not np.all((labels >= 0) & (labels < 30)):
        raise ValueError(f"{path}: labels must be in 0..29")
    observed = np.bincount(labels, minlength=30)
    if observed.shape != (30,) or not np.all(observed == per_class):
        raise ValueError(f"{path}: expected labels 0..29, each with {per_class} rows")
    return labels


def _table4_split(labels: np.ndarray, ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.isin(labels, ids))
    classes = labels[rows]
    train_val, test = train_test_split(rows, test_size=0.2, stratify=classes, random_state=42)
    train, validation = train_test_split(
        train_val, test_size=0.25, stratify=labels[train_val], random_state=42,
    )
    result = tuple(np.sort(part).astype(np.int64) for part in (train, validation, test))
    for part, expected in zip(result, (1200, 400, 400), strict=True):
        if any(np.count_nonzero(labels[part] == label) != expected for label in ids):
            raise ValueError("Table 4 stratified split has unexpected per-isolate counts")
    if len(np.unique(np.concatenate(result))) != len(rows):
        raise ValueError("Table 4 split contains duplicate or missing source rows")
    return result


def _input_snapshot(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_inputs(raw_dir: Path, processed_dir: Path, *, chunk_rows: int = 1024) -> dict[str, object]:
    """Prepare three SNV memmaps and fixed source-row splits once; never edit raw files."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if processed_dir.exists():
        raise FileExistsError(f"Prepared directory exists: {processed_dir}")
    seeds = seed_manifest()
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_snapshot: dict[str, dict[str, object]] = {}
    for subset, count in SOURCE_COUNTS.items():
        x_path, y_path = raw_dir / f"X_{subset}.npy", raw_dir / f"y_{subset}.npy"
        spectra = _source_array(x_path, (count, CHANNELS), labels=False)
        labels = _checked_labels(y_path, count, PER_CLASS[subset])
        sources[subset] = (spectra, labels)
        source_snapshot[x_path.name] = _input_snapshot(x_path)
        source_snapshot[y_path.name] = _input_snapshot(y_path)

    axis_path = raw_dir / "wavenumbers.npy"
    if axis_path.is_file():
        axis = np.load(axis_path, allow_pickle=False)
        if axis.shape != (CHANNELS,) or axis.dtype.kind != "f":
            raise ValueError("wavenumbers.npy must contain 1000 floating values")
        differences = np.diff(axis)
        if (not np.isfinite(axis).all()
                or not (np.all(differences > 0) or np.all(differences < 0))):
            raise ValueError("wavenumbers.npy must be finite and strictly monotone")
        wave_info: dict[str, object] = {"source": _input_snapshot(axis_path),
                                        "first": float(axis[0]), "last": float(axis[-1]),
                                        "order": "ascending" if differences[0] > 0 else "descending"}
    else:
        wave_info = {"source": None, "status": "not present in supplied files; no axis inferred"}

    ref_labels = sources["reference"][1]
    splits: dict[str, np.ndarray] = {"reference30_pretrain": np.arange(60_000, dtype=np.int64),
                                     "test_all": np.arange(3_000, dtype=np.int64)}
    for corpus, ids in TABLE4_CLASSES.items():
        train, validation, test = _table4_split(ref_labels, ids)
        splits.update({f"{corpus}_train": train, f"{corpus}_validation": validation,
                       f"{corpus}_test": test})
    fine_labels = sources["finetune"][1]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fine_train, fine_val = next(folds.split(np.zeros(len(fine_labels)), fine_labels))
    splits["finetune_train"] = fine_train.astype(np.int64)
    splits["finetune_validation"] = fine_val.astype(np.int64)
    if (len(fine_train) != 2400 or len(fine_val) != 600
            or any(np.count_nonzero(fine_labels[fine_val] == label) != 20 for label in range(30))):
        raise ValueError("SMAE fold[1] does not have the expected 2400/600 stratified split")

    processed_dir.mkdir(parents=True, exist_ok=False)
    for subset, (spectra, labels) in sources.items():
        target = open_memmap(processed_dir / f"X_{subset}_snv.npy", mode="w+",
                             dtype=np.float32, shape=spectra.shape)
        for start in range(0, len(spectra), chunk_rows):
            end = min(start + chunk_rows, len(spectra))
            target[start:end] = snv_rows(np.asarray(spectra[start:end]), subset=subset, start=start)
        target.flush()
        del target
        np.save(processed_dir / f"y_{subset}.npy", labels, allow_pickle=False)
    np.savez_compressed(processed_dir / "splits.npz", **splits)
    acquisition_path = raw_dir / "acquisition.json"
    acquisition = _read_json(acquisition_path) if acquisition_path.is_file() else {
        "status": "user supplied files; acquisition URL not recorded",
    }
    record: dict[str, object] = {
        "schema_version": 1, "status": "prepared", "config": bacteria_config(),
        "source": source_snapshot, "acquisition": acquisition,
        "versions": {"numpy": version("numpy"), "scikit_learn": version("scikit-learn")},
        "wavenumbers": wave_info, "split_file": "splits.npz",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "seeds": seeds,
        "run_manifest": [
            {"corpus": corpus, "condition": condition, "run_seed": seed,
             "pretrain": True, "cluster": corpus != "reference30",
             "finetune": corpus == "reference30"}
            for corpus in ("bacteria4", "bacteria6", "reference30")
            for condition in ("M00", "M11") for seed in range(5)
        ],
    }
    _write_json(processed_dir / "manifest.json", record)
    return record


class PreparedBacteria:
    """Read-only access to verified prepared spectra, labels and source-row indices."""

    def __init__(self, processed_dir: Path) -> None:
        manifest = _read_json(processed_dir / "manifest.json")
        if (manifest.get("schema_version") != 1 or manifest.get("status") != "prepared"
                or manifest.get("config") != bacteria_config()
                or manifest.get("seeds") != seed_manifest()):
            raise ValueError("Prepared Bacteria-ID manifest does not match the fixed protocol")
        self.manifest = manifest
        self.root = processed_dir
        self.spectra: dict[str, np.ndarray] = {}
        self.labels: dict[str, np.ndarray] = {}
        for subset, count in SOURCE_COUNTS.items():
            self.spectra[subset] = _source_array(
                processed_dir / f"X_{subset}_snv.npy", (count, CHANNELS), labels=False,
            )
            if self.spectra[subset].dtype != np.float32:
                raise ValueError(f"Processed {subset} spectra must be FP32 SNV")
            self.labels[subset] = _checked_labels(
                processed_dir / f"y_{subset}.npy", count, PER_CLASS[subset],
            )
        with np.load(processed_dir / "splits.npz", allow_pickle=False) as saved:
            self.splits = {name: saved[name].copy() for name in saved.files}
        if (set(self.splits) != set(manifest["split_sizes"])
                or any(len(rows) != manifest["split_sizes"][name] for name, rows in self.splits.items())):
            raise ValueError("Prepared split file differs from manifest")
        expected: dict[str, np.ndarray] = {
            "reference30_pretrain": np.arange(60_000, dtype=np.int64),
            "test_all": np.arange(3_000, dtype=np.int64),
        }
        for corpus, ids in TABLE4_CLASSES.items():
            train, validation, test = _table4_split(self.labels["reference"], ids)
            expected.update({f"{corpus}_train": train, f"{corpus}_validation": validation,
                             f"{corpus}_test": test})
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train, validation = next(folds.split(np.zeros(3_000), self.labels["finetune"]))
        expected.update(finetune_train=train, finetune_validation=validation)
        if (set(self.splits) != set(expected)
                or any(not np.array_equal(self.splits[name], rows)
                       for name, rows in expected.items())):
            raise ValueError("Prepared source-row splits differ from the fixed protocol")

    def rows(self, subset: str, selection: str) -> tuple[np.ndarray, np.ndarray]:
        """Return an FP32 copy and labels in persisted source-row order."""
        if subset not in self.spectra or selection not in self.splits:
            raise ValueError("Unknown prepared subset or split")
        indices = self.splits[selection]
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(self.labels[subset])):
            raise ValueError(f"Invalid row indices in {selection}")
        return np.array(self.spectra[subset][indices], dtype=np.float32, copy=True), self.labels[subset][indices]
