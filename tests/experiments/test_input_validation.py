from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments.input_validation import (
    InputInventory,
    SampleProbe,
    load_input_inventory,
    probe_sample,
)
from wood_degradation_map.preprocessing.production_preprocessing import (
    ProductionPreprocessingConfig,
)


@pytest.fixture
def saved_input(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "production"
    (root / "samples").mkdir(parents=True)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("KYOw\n2702\n2782\n", encoding="utf-8")
    config = ProductionPreprocessingConfig().to_json_dict()
    config.update(target_wavelength_start_nm=900.0, target_wavelength_end_nm=2300.0)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame([{
        "sample_id": "KYOw02702", "file": "samples/KYOw02702.h5",
        "height": 2, "width": 320, "bands": 256, "saved_pixel_count": 4,
        "preprocessing_id": "production",
    }]).to_parquet(root / "manifest.parquet", index=False)
    pd.DataFrame([{"sample_id": "KYOw02702", "saved_pixel_count": 4}]).to_parquet(
        root / "sample_quality.parquet", index=False,
    )
    coordinates = np.array([[0, 0], [0, 319], [1, 0], [1, 319]], dtype=np.int32)
    wavelength = np.linspace(900, 2300, 256)
    # Distinct shapes let the fixture detect a spectrum-row permutation.
    reflectance = np.stack([
        np.linspace(0.1, 0.8, 256) ** exponent for exponent in (1, 2, 3, 4)
    ]).astype(np.float32)
    values = reflectance.astype(np.float64)
    snv = ((values - values.mean(axis=1, keepdims=True))
           / values.std(axis=1, ddof=1, keepdims=True)).astype(np.float32)
    mask = np.zeros((2, 320), dtype=np.uint8)
    mask[coordinates[:, 0], coordinates[:, 1]] = 1
    with h5py.File(root / "samples/KYOw02702.h5", "w") as handle:
        for key, value in {
            "wavelength_nm": wavelength, "snv": snv, "reflectance": reflectance,
            "pixel_row_col": coordinates, "valid_spectrum_mask": mask,
        }.items():
            handle.create_dataset(key, data=value)
        handle.attrs.update({
            "schema_version": 2, "sample_id": "KYOw02702", "preprocessing_id": "production",
            "source": "200 Hz only", "snr_threshold": 10.0, "snv_ddof": 1,
            "target_bands": 256, "source_height": 2, "source_width": 320,
            "source_bands": 256, "saved_pixel_count": 4,
        })
    return root, metadata


def _probe(inventory: InputInventory, rows: int = 8) -> SampleProbe:
    return probe_sample(
        inventory.samples[0], preprocessing_id=inventory.preprocessing_id,
        wavelength_start_nm=inventory.wavelength_start_nm,
        wavelength_end_nm=inventory.wavelength_end_nm, rows_per_sample=rows,
    )


def test_tables_and_snv_coordinate_correspondence(saved_input: tuple[Path, Path]) -> None:
    inventory = load_input_inventory(*saved_input)
    report = _probe(inventory)
    assert inventory.metadata_only_ids == ("KYOw02782",)
    assert report.checked_hdf5_rows == (0, 1, 2, 3)
    assert report.checked_pixel_row_col == ((0, 0), (0, 319), (1, 0), (1, 319))
    assert report.snv_reconstruction_absolute_error_max < 1e-6


@pytest.fixture
def production_input(saved_input: tuple[Path, Path]) -> tuple[Path, Path]:
    old, metadata = saved_input
    root = old.with_name("production_v1")
    old.rename(root)
    manifest_path = root / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest["preprocessing_id"] = root.name
    manifest.to_parquet(manifest_path, index=False)
    with h5py.File(root / "samples/KYOw02702.h5", "r+") as handle:
        handle.attrs["preprocessing_id"] = root.name
    return root, metadata


def test_production_policy_and_nonnegative_reflectance(production_input: tuple[Path, Path]) -> None:
    assert _probe(load_input_inventory(*production_input)).checked_hdf5_rows == (0, 1, 2, 3)


@pytest.mark.parametrize("change", ["missing", "criterion", "action"])
def test_production_rejects_missing_or_different_background_policy(
    production_input: tuple[Path, Path], change: str,
) -> None:
    path = production_input[0] / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if change == "missing":
        del config["spectral"]["negative_reflectance_policy"]
    else:
        config["spectral"]["negative_reflectance_policy"][change] = "changed"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="negative_reflectance_policy"):
        load_input_inventory(*production_input)


def test_production_probe_rejects_negative_reflectance(production_input: tuple[Path, Path]) -> None:
    with h5py.File(production_input[0] / "samples/KYOw02702.h5", "r+") as handle:
        handle["reflectance"][0, 10] = -0.01
        values = handle["reflectance"][:].astype(np.float64)
        handle["snv"][:] = ((values - values.mean(axis=1, keepdims=True))
                            / values.std(axis=1, ddof=1, keepdims=True)).astype(np.float32)
    with pytest.raises(ValueError, match="reflectance is negative"):
        _probe(load_input_inventory(*production_input))


@pytest.mark.parametrize("content", ["2702\n02702", "2782", "", "2702\nunknown"])
def test_invalid_metadata_fails(saved_input: tuple[Path, Path], content: str) -> None:
    root, metadata = saved_input
    metadata.write_text(f"KYOw\n{content}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="[Mm]etadata"):
        load_input_inventory(root, metadata)


@pytest.mark.parametrize("problem", ["duplicate", "count", "id", "fractional_count"])
def test_inconsistent_quality_fails(saved_input: tuple[Path, Path], problem: str) -> None:
    path = saved_input[0] / "sample_quality.parquet"
    quality = pd.read_parquet(path)
    if problem == "duplicate":
        quality = pd.concat([quality, quality], ignore_index=True)
    elif problem == "count":
        quality["saved_pixel_count"] = 3
    elif problem == "id":
        quality["sample_id"] = "KYOw02782"
    else:
        quality["saved_pixel_count"] = 3.5
    quality.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        load_input_inventory(*saved_input)


def test_changed_fixed_config_fails(saved_input: tuple[Path, Path]) -> None:
    path = saved_input[0] / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["spectral"]["snv"] = "ddof=0"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config.spectral.snv"):
        load_input_inventory(*saved_input)


def test_manifest_path_cannot_escape(saved_input: tuple[Path, Path]) -> None:
    path = saved_input[0] / "manifest.parquet"
    manifest = pd.read_parquet(path)
    manifest["file"] = "../../outside.h5"
    manifest.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="expected samples/"):
        load_input_inventory(*saved_input)


@pytest.mark.parametrize("problem, message", [
    ("coordinate", "outside the image"),
    ("duplicate", "duplicate sampled coordinates"),
    ("mask", "not valid mask pixels"),
    ("snv", "SNV differs"),
    ("permutation", "SNV differs"),
    ("constant", "nonpositive standard deviation"),
    ("nonfinite", "non-finite"),
    ("wavelength", "grid differs"),
    ("attribute", "attribute snv_ddof"),
    ("missing", "missing dataset snv"),
    ("shape", "snv shape"),
    ("dtype", "snv must be float32"),
])
def test_corrupt_hdf5_fails(
    saved_input: tuple[Path, Path], problem: str, message: str,
) -> None:
    inventory = load_input_inventory(*saved_input)
    with h5py.File(inventory.samples[0].path, "r+") as handle:
        if problem == "coordinate":
            handle["pixel_row_col"][0] = [-1, 0]
        elif problem == "duplicate":
            handle["pixel_row_col"][1] = handle["pixel_row_col"][0]
        elif problem == "mask":
            handle["valid_spectrum_mask"][0, 0] = 0
        elif problem == "snv":
            handle["snv"][0, 0] += 0.1
        elif problem == "permutation":
            handle["snv"][:] = handle["snv"][:][::-1]
        elif problem == "constant":
            handle["reflectance"][0] = 1
        elif problem == "nonfinite":
            handle["reflectance"][0, 0] = np.nan
        elif problem == "wavelength":
            handle["wavelength_nm"][:] += 1
        elif problem == "attribute":
            handle.attrs["snv_ddof"] = 0
        else:
            values = handle["snv"][:]
            del handle["snv"]
            if problem == "shape":
                handle.create_dataset("snv", data=values[:, :-1])
            elif problem == "dtype":
                handle.create_dataset("snv", data=values.astype(np.float64))
    with pytest.raises(ValueError, match=message):
        _probe(inventory)


def test_probe_scope_is_bounded_and_reported(saved_input: tuple[Path, Path]) -> None:
    inventory = load_input_inventory(*saved_input)
    with h5py.File(inventory.samples[0].path, "r+") as handle:
        handle["snv"][1, 0] = np.nan
    # The unselected row must not be advertised as validated by a two-row probe.
    assert _probe(inventory, rows=2).checked_hdf5_rows == (0, 3)
    with pytest.raises(ValueError, match="non-finite"):
        _probe(inventory)


@pytest.mark.parametrize("rows", [0, 65])
def test_unbounded_probe_rejected(saved_input: tuple[Path, Path], rows: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 64"):
        _probe(load_input_inventory(*saved_input), rows)
