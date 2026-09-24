"""Fold-free, fixed all-sample fit inputs, kept separate from CV artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ADOPTED_SAMPLE_IDS, PIXELS_PER_SAMPLE, ROOT_SEED, RUN_PURPOSES, _seed
from .config import experiment_config
from .data import FoldData, _Selection
from .input_validation import InputInventory
from .manifests import (
    _columns,
    _digest,
    _read_coordinates,
    _read_json,
    _require,
    _selected_coordinates,
    _write_json,
    input_snapshot,
)

GLOBAL_CONDITIONS = ("B0", "B1", "A0", "A1", "M00", "M11")
NEURAL_CONDITIONS = ("A0", "A1", "M00", "M11")
SAMPLE_COLUMNS = ["sample_id", "file", "saved_pixel_count", "height", "width"]
PIXEL_COLUMNS = ["sample_id", "sampling_seed", "hdf5_row", "pixel_row", "pixel_col"]
BUNDLE_FILES = (
    "config/experiment.json", "config/seeds.json", "manifests/inputs.json",
    "manifests/samples.parquet", "manifests/fit_pixels.parquet",
)


def global_seed(purpose: str, sample_id: str | None = None) -> int:
    """Use the approved global scope and repeat 1, never a surrogate CV fold."""
    if purpose == "sampling" and isinstance(sample_id, str) and sample_id:
        return _seed(purpose, "global", sample_id)
    if sample_id is None and purpose in RUN_PURPOSES:
        return _seed(purpose, "global", 1)
    if sample_id is None and purpose == "kmeans":
        return _seed(purpose, "global", 1, 8)
    raise ValueError("Invalid global seed purpose or sample ID")


def global_config() -> dict[str, object]:
    cv = experiment_config()
    config = {key: cv[key] for key in (
        "preprocessing_id", "input", "representation", "pca", "chemomae", "training",
        "augmentation", "extraction",
    )}
    config.update({
        "schema_version": 1, "scope": "global_fit", "repeat": 1,
        "protocol": "docs/design/visualization_and_interpretation.md",
        "sample_ids": list(ADOPTED_SAMPLE_IDS),
        "conditions": [c for c in cv["conditions"] if c["condition_id"] in GLOBAL_CONDITIONS],
        "sampling": {**{k: cv["sampling"][k] for k in
                         ("q", "replace", "distribution", "algorithm")},
                     "shared_across": ["condition", "clustering_method", "epoch"]},
        "seeds": {"root_seed": ROOT_SEED, "algorithm": "sha256-json-v1-first-32-bits",
                  "context": "global replaces fold; repeat=1; sampling has no repeat"},
        "clustering_plan": {"K": 8, "methods": ["cosine_kmeans", "vmf"]},
        "prediction_scope": "all saved valid pixels; descriptive, not OOF",
    })
    config["pca"]["fit_scope"] = "shared global fit pixels"
    return config


def global_seed_plan(sample_ids: tuple[str, ...]) -> dict[str, object]:
    records = [{"purpose": "sampling", "sample_id": sample_id,
                "seed": global_seed("sampling", sample_id)} for sample_id in sorted(sample_ids)]
    records.extend({"purpose": purpose, "repeat": 1, "seed": global_seed(purpose)}
                   for purpose in RUN_PURPOSES)
    records.append({"purpose": "kmeans", "repeat": 1, "K": 8, "seed": global_seed("kmeans")})
    _require(len({row["seed"] for row in records}) == len(records), "Global seed collision")
    return {**global_config()["seeds"], "scope": "global_fit", "records": records}


@dataclass(frozen=True)
class GlobalManifest:
    samples: pd.DataFrame
    fit_pixels: pd.DataFrame
    q: int


def create_global_manifest(
    inventory: InputInventory, *, q: int = PIXELS_PER_SAMPLE,
) -> GlobalManifest:
    """Sample coordinates only; q is overridable solely for synthetic fixtures."""
    _require(type(q) is int and q > 0, "q must be a positive integer")
    _require(bool(inventory.samples), "Empty inventory")
    records, frames = [], []
    for sample in sorted(inventory.samples, key=lambda item: item.sample_id):
        _require(sample.saved_pixel_count >= q, "Sampling with replacement is forbidden")
        records.append({"sample_id": sample.sample_id, "file": f"samples/{sample.sample_id}.h5",
                        "saved_pixel_count": sample.saved_pixel_count,
                        "height": sample.height, "width": sample.width})
        seed = global_seed("sampling", sample.sample_id)
        rows = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(
            sample.saved_pixel_count, size=q, replace=False,
        ))
        coordinates, mask = _read_coordinates(sample)
        selected = _selected_coordinates(sample, rows, coordinates, mask)
        frames.append(pd.DataFrame({"sample_id": sample.sample_id, "sampling_seed": seed,
                                    "hdf5_row": rows, "pixel_row": selected[:, 0],
                                    "pixel_col": selected[:, 1]}))
    plan = GlobalManifest(pd.DataFrame(records, columns=SAMPLE_COLUMNS),
                          pd.concat(frames, ignore_index=True)[PIXEL_COLUMNS], q)
    validate_global_manifest(plan, inventory, check_coordinates=False)
    return plan


def validate_global_manifest(
    plan: GlobalManifest, inventory: InputInventory, *, check_coordinates: bool = True,
) -> None:
    samples = {sample.sample_id: sample for sample in inventory.samples}
    _require(bool(samples) and len(samples) == len(inventory.samples), "Invalid inventory IDs")
    _require(type(plan.q) is int and plan.q > 0, "q must be positive")
    _columns(plan.samples, SAMPLE_COLUMNS, ["saved_pixel_count", "height", "width"])
    _columns(plan.fit_pixels, PIXEL_COLUMNS, PIXEL_COLUMNS[1:])
    _require(plan.samples.sample_id.tolist() == sorted(samples), "Global sample IDs differ")
    pixels = plan.fit_pixels
    _require(set(pixels.sample_id) == set(samples), "Missing or unexpected global fit samples")
    _require((pixels.groupby("sample_id").size() == plan.q).all(), "Wrong pixels per sample")
    _require(pixels.sample_id.tolist() == sorted(pixels.sample_id), "Sample order differs")
    _require(not pixels.duplicated(["sample_id", "pixel_row", "pixel_col"]).any(),
             "Duplicate global fit coordinates")
    for row in plan.samples.itertuples(index=False):
        sample = samples[row.sample_id]
        _require((row.file, row.saved_pixel_count, row.height, row.width) == (
            f"samples/{sample.sample_id}.h5", sample.saved_pixel_count, sample.height, sample.width,
        ), "Global sample identity differs")
        group = pixels.loc[pixels.sample_id == sample.sample_id]
        rows = group.hdf5_row.to_numpy()
        seed = global_seed("sampling", sample.sample_id)
        _require((group.sampling_seed == seed).all(), "Global sampling seed differs")
        _require(sample.saved_pixel_count >= plan.q, "Insufficient pixels")
        expected_rows = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(
            sample.saved_pixel_count, size=plan.q, replace=False,
        ))
        _require(np.array_equal(rows, expected_rows), "Global sampled rows differ from seed plan")
        selected = group[["pixel_row", "pixel_col"]].to_numpy()
        _require(((selected >= 0) & (selected < [sample.height, sample.width])).all(),
                 "Global coordinates out of bounds")
        if check_coordinates:
            coordinates, mask = _read_coordinates(sample)
            _require(np.array_equal(selected, _selected_coordinates(sample, rows, coordinates, mask)),
                     "Global coordinates differ from source")


def global_summary(plan: GlobalManifest) -> dict[str, int]:
    steps = len(plan.fit_pixels) // global_config()["training"]["batch_size"]
    return {"samples": len(plan.samples), "pixels_per_sample": plan.q,
            "fit_pixels": len(plan.fit_pixels), "steps_per_epoch": steps,
            "planned_updates_per_run": steps * global_config()["training"]["epochs"]}


def _production_inventory(inventory: InputInventory) -> None:
    _require(inventory.preprocessing_id == global_config()["preprocessing_id"],
             "Wrong global preprocessing ID")
    _require(sorted(s.sample_id for s in inventory.samples) == sorted(ADOPTED_SAMPLE_IDS),
             "Global fit requires the approved 49 samples")


def create_global_bundle(
    output: Path, inventory: InputInventory, processed: Path, metadata: Path,
) -> GlobalManifest:
    if output.exists():
        raise FileExistsError("Global output exists; check the saved manifest instead")
    _production_inventory(inventory)
    before = input_snapshot(inventory, processed, metadata)
    plan = create_global_manifest(inventory)
    seeds = global_seed_plan(tuple(plan.samples.sample_id))
    _require(input_snapshot(inventory, processed, metadata) == before, "Inputs changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config").mkdir()
    (output / "manifests").mkdir()
    for name, value in zip(BUNDLE_FILES[:3], (global_config(), seeds, before), strict=True):
        _write_json(output / name, value)
    plan.samples.to_parquet(output / BUNDLE_FILES[3], index=False)
    plan.fit_pixels.to_parquet(output / BUNDLE_FILES[4], index=False)
    _write_json(output / "manifests/complete.json", {
        "schema_version": 1, "scope": "global_fit", "status": "complete",
        "artifact_sha256": {name: _digest(output / name) for name in BUNDLE_FILES},
        "summary": global_summary(plan), "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    })
    return plan


def load_global_bundle(
    output: Path, inventory: InputInventory, processed: Path, metadata: Path,
) -> GlobalManifest:
    _production_inventory(inventory)
    record = _read_json(output / "manifests/complete.json")
    _require(record.get("schema_version") == 1 and record.get("scope") == "global_fit"
             and record.get("status") == "complete", "Not a complete global manifest")
    hashes = record.get("artifact_sha256")
    _require(isinstance(hashes, dict) and set(hashes) == set(BUNDLE_FILES),
             "Unexpected global manifest artifacts")
    for name in BUNDLE_FILES:
        _require(_digest(output / name) == hashes[name], f"Global artifact changed: {name}")
    _require(_read_json(output / BUNDLE_FILES[0]) == global_config(), "Global config changed")
    before = input_snapshot(inventory, processed, metadata)
    _require(_read_json(output / BUNDLE_FILES[2]) == before, "Global source inputs changed")
    plan = GlobalManifest(pd.read_parquet(output / BUNDLE_FILES[3]),
                          pd.read_parquet(output / BUNDLE_FILES[4]), PIXELS_PER_SAMPLE)
    validate_global_manifest(plan, inventory)
    _require(_read_json(output / BUNDLE_FILES[1]) == global_seed_plan(tuple(plan.samples.sample_id)),
             "Global seed plan differs")
    _require(record.get("summary") == global_summary(plan), "Global summary differs")
    _require(input_snapshot(inventory, processed, metadata) == before, "Inputs changed")
    return plan


class GlobalData(FoldData):
    """Reuse validated chunked SNV reading, with no fold or held-out split.

    Only the loader implementation is inherited. The CV constructor is never
    called and no fake fold ID is assigned. All saved pixels are future map inputs.
    """

    def __init__(self, inventory: InputInventory, manifest: GlobalManifest) -> None:
        validate_global_manifest(manifest, inventory, check_coordinates=False)
        samples = {sample.sample_id: sample for sample in inventory.samples}
        self._train = tuple(
            _Selection(samples[sample_id], group.hdf5_row.to_numpy(copy=True),
                       group[["pixel_row", "pixel_col"]].to_numpy(copy=True))
            for sample_id, group in manifest.fit_pixels.groupby("sample_id", sort=True)
        )
        self._test = ()
        self.train_sample_ids = tuple(selection.sample.sample_id for selection in self._train)
        self.train_pixel_count = len(manifest.fit_pixels)
        self.test_sample_ids = ()
        self.test_pixel_count = 0
