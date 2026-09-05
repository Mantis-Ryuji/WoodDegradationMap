"""Read-only checks of saved production inputs before experimental splitting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


@dataclass(frozen=True)
class SampleInput:
    """A manifest entry, without any decision about experimental inclusion."""

    sample_id: str
    path: Path
    height: int
    width: int
    saved_pixel_count: int


@dataclass(frozen=True)
class InputInventory:
    """Validated table correspondence; source relationships remain unconfirmed."""

    preprocessing_id: str
    samples: tuple[SampleInput, ...]
    metadata_only_ids: tuple[str, ...]
    wavelength_start_nm: float
    wavelength_end_nm: float


@dataclass(frozen=True)
class SampleProbe:
    """Results limited to the specified HDF5 rows, not a full pixel audit."""

    sample_id: str
    saved_pixel_count: int
    checked_hdf5_rows: tuple[int, ...]
    checked_pixel_row_col: tuple[tuple[int, int], ...]
    snv_reconstruction_absolute_error_max: float
    snv_mean_absolute_max: float
    snv_sample_std_absolute_error_max: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_config(actual: object, expected: dict[str, object], prefix: str) -> None:
    """Check the fixed input contract while allowing unrelated provenance keys."""
    if not isinstance(actual, dict):
        raise ValueError(f"{prefix} must be an object")
    for key, value in expected.items():
        location = f"{prefix}.{key}"
        if isinstance(value, dict):
            _check_config(actual.get(key), value, location)
        else:
            _require(actual.get(key) == value, f"Unexpected {location}: expected {value!r}")


def _check_ids(table: pd.DataFrame, label: str) -> None:
    _require(not table.empty, f"{label} is empty")
    ids = table["sample_id"]
    _require(
        ids.map(lambda value: isinstance(value, str) and bool(
            re.fullmatch(r"KYOw[0-9]{5}", value)
        )).all(),
        f"{label}: sample_id must use KYOw followed by five digits",
    )
    _require(not ids.duplicated().any(), f"{label}: duplicate sample_id")


def _check_integer_column(table: pd.DataFrame, column: str, minimum: int) -> None:
    values = table[column]
    _require(
        is_integer_dtype(values.dtype) and not values.isna().any()
        and (values >= minimum).all(),
        f"{column} must contain integers >= {minimum} without missing values",
    )


def load_input_inventory(processed_dir: Path, metadata_path: Path) -> InputInventory:
    """Validate small saved tables/config and match numeric metadata KYOw IDs.

    Metadata KYOw numbers are formatted as five digits after the literal KYOw
    prefix, matching the production filenames. Missing/duplicate IDs are errors;
    metadata-only rows are reported rather than silently adopted or removed.
    No HDF5 spectra, raw data, or output files are read or written here.
    """
    root = processed_dir.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    _check_config(config, {
        "schema_version": 2,
        "source": "200 Hz only",
        "expected_width": 320,
        "expected_bands": 256,
        "mask": {
            "score": "sum of all 256 raw intensity bands",
            "threshold": "per-sample three-class Multi-Otsu",
            "wood_classes": [1, 2],
            "erosion_radius": 1,
            "min_object_size": 1,
            "connectivity": 2,
            "postprocessing_order": ["erosion", "remove_small_objects"],
        },
        "spectral": {
            "reflectance": "(I_200-D_200)/(W_200-D_200)",
            "snr_threshold": 10.0,
            "cutoff_uses_sample_spectra": False,
            "target_bands": 256,
            "interpolation": "linear in wavelength without extrapolation",
            "snv": "pixel-wise mean and sample std with ddof=1",
            "processing_order": [
                "reflectance", "automatic_terminal_snr_cutoff",
                "linear_interpolation_to_256", "SNV",
            ],
            "clip": False,
            "smoothing": False,
        },
        "storage": {"format": "HDF5", "dtype": "float32"},
    }, "config")
    if root.name == "production_v1":
        _check_config(config, {"spectral": {"negative_reflectance_policy": {
            "stage": "interpolated reflectance before SNV",
            "criterion": "any band < 0",
            "action": "background in valid_spectrum_mask; exclude from train and test",
            "reason_code": 3,
            "reason_priority": [1, 2, 3],
        }}}, "config")
    endpoints = [config.get(key) for key in (
        "target_wavelength_start_nm", "target_wavelength_end_nm",
    )]
    _require(all(isinstance(value, (int, float)) for value in endpoints),
             "Wavelength endpoints must be numeric")
    start, end = (float(value) for value in endpoints)
    _require(np.isfinite([start, end]).all() and start < end,
             "Wavelength endpoints must be finite and increasing")

    manifest = pd.read_parquet(root / "manifest.parquet", columns=[
        "sample_id", "file", "height", "width", "bands", "saved_pixel_count",
        "preprocessing_id",
    ])
    quality = pd.read_parquet(root / "sample_quality.parquet", columns=[
        "sample_id", "saved_pixel_count",
    ])
    for table, label in ((manifest, "manifest"), (quality, "sample_quality")):
        _check_ids(table, label)
        _check_integer_column(table, "saved_pixel_count", 1)
    for column in ("height", "width", "bands"):
        _check_integer_column(manifest, column, 1)
    _require((manifest["width"] == 320).all() and (manifest["bands"] == 256).all(),
             "Manifest must describe width 320 and 256 bands")
    _require((manifest["preprocessing_id"] == root.name).all(),
             "Manifest preprocessing_id differs from input directory")
    _require(set(manifest["sample_id"]) == set(quality["sample_id"]),
             "Manifest and sample_quality sample IDs differ")
    counts = quality.set_index("sample_id")["saved_pixel_count"]
    _require(all(int(counts[row.sample_id]) == int(row.saved_pixel_count)
                 for row in manifest.itertuples(index=False)),
             "Manifest and sample_quality pixel counts differ")

    metadata = pd.read_csv(metadata_path, usecols=["KYOw"], dtype="string")
    ids = metadata["KYOw"]
    _require(not metadata.empty and not ids.isna().any()
             and ids.str.fullmatch(r"[0-9]{1,5}").all(),
             "Metadata KYOw must contain one to five digits without missing values")
    metadata_ids = ids.map(lambda value: f"KYOw{int(value):05d}")
    _require(not metadata_ids.duplicated().any(), "Metadata contains duplicate KYOw IDs")
    missing = sorted(set(manifest["sample_id"]) - set(metadata_ids))
    _require(not missing, f"Manifest samples missing from metadata: {missing}")

    samples: list[SampleInput] = []
    for row in manifest.sort_values("sample_id").itertuples(index=False):
        expected_file = f"samples/{row.sample_id}.h5"
        _require(row.file == expected_file, f"{row.sample_id}: expected {expected_file}")
        path = (root / expected_file).resolve()
        _require(path.is_relative_to(root), f"HDF5 path escapes input directory: {row.sample_id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        count, height, width = int(row.saved_pixel_count), int(row.height), int(row.width)
        _require(count <= height * width, f"{row.sample_id}: pixel count exceeds image size")
        samples.append(SampleInput(row.sample_id, path, height, width, count))
    return InputInventory(
        root.name, tuple(samples),
        tuple(sorted(set(metadata_ids) - set(manifest["sample_id"]))), start, end,
    )


def probe_sample(
    sample: SampleInput,
    *,
    preprocessing_id: str,
    wavelength_start_nm: float,
    wavelength_end_nm: float,
    rows_per_sample: int = 8,
) -> SampleProbe:
    """Check schema and up to 64 evenly spaced saved rows using read-only HDF5.

    Only selected spectra/coordinates, their mask cells and the 256 wavelengths
    are read. HDF5 may decompress whole chunks to access these cells. Unsampled
    rows, global coordinate uniqueness and total mask coverage are not verified.
    The 1e-5 absolute tolerance checks float32 storage, not scientific quality.
    """
    if isinstance(rows_per_sample, bool) or not isinstance(rows_per_sample, int):
        raise ValueError("rows_per_sample must be an integer between 1 and 64")
    _require(1 <= rows_per_sample <= 64, "rows_per_sample must be between 1 and 64")
    n = sample.saved_pixel_count
    _require(n > 0, f"{sample.sample_id}: no saved pixels")
    rows = np.linspace(0, n - 1, min(n, rows_per_sample), dtype=np.int64)

    def require(condition: bool, message: str) -> None:
        _require(condition, f"{sample.sample_id}: {message}")

    with h5py.File(sample.path, "r") as handle:
        attributes = {
            "schema_version": 2, "sample_id": sample.sample_id,
            "preprocessing_id": preprocessing_id, "source": "200 Hz only",
            "snr_threshold": 10.0, "snv_ddof": 1, "target_bands": 256,
            "source_height": sample.height, "source_width": sample.width,
            "source_bands": 256, "saved_pixel_count": n,
        }
        for key, expected in attributes.items():
            require(handle.attrs.get(key) == expected, f"attribute {key} must equal {expected!r}")
        shapes = {
            "snv": (n, 256), "reflectance": (n, 256), "pixel_row_col": (n, 2),
            "valid_spectrum_mask": (sample.height, sample.width), "wavelength_nm": (256,),
        }
        for key, shape in shapes.items():
            require(key in handle and isinstance(handle[key], h5py.Dataset),
                    f"missing dataset {key}")
            require(handle[key].shape == shape, f"{key} shape must be {shape}")
        for key in ("snv", "reflectance"):
            require(handle[key].dtype == np.dtype("float32"), f"{key} must be float32")
        require(handle["pixel_row_col"].dtype.kind in "iu", "coordinates must be integers")
        require(handle["valid_spectrum_mask"].dtype.kind in "biu", "mask must be binary integers")
        wavelength = handle["wavelength_nm"][:]
        require(np.isfinite(wavelength).all() and (np.diff(wavelength) > 0).all(),
                "wavelengths must be finite and increasing")
        require(np.allclose(wavelength, np.linspace(
            wavelength_start_nm, wavelength_end_nm, 256,
        ), rtol=0, atol=1e-5), "wavelength grid differs from config")
        coordinates = handle["pixel_row_col"][rows]
        require(((coordinates >= 0) & (coordinates < [sample.height, sample.width])).all(),
                "sampled coordinates are outside the image")
        require(len(np.unique(coordinates, axis=0)) == len(rows),
                "duplicate sampled coordinates")
        require(all(handle["valid_spectrum_mask"][int(y), int(x)] == 1 for y, x in coordinates),
                "sampled coordinates are not valid mask pixels")
        reflectance = handle["reflectance"][rows].astype(np.float64)
        snv = handle["snv"][rows].astype(np.float64)
    require(np.isfinite(reflectance).all() and np.isfinite(snv).all(),
            "sampled spectra contain non-finite values")
    if preprocessing_id == "production_v1":
        require((reflectance >= 0.0).all(), "sampled production_v1 reflectance is negative")
    std = reflectance.std(axis=1, ddof=1)
    require((std > 0).all(), "sampled reflectance has nonpositive standard deviation")
    reconstructed = (reflectance - reflectance.mean(axis=1, keepdims=True)) / std[:, None]
    error = float(np.max(np.abs(snv - reconstructed)))
    require(error <= 1e-5, "SNV differs from corresponding reflectance (ddof=1)")
    return SampleProbe(
        sample.sample_id, n, tuple(int(row) for row in rows),
        tuple((int(y), int(x)) for y, x in coordinates), error,
        float(np.max(np.abs(snv.mean(axis=1)))),
        float(np.max(np.abs(snv.std(axis=1, ddof=1) - 1))),
    )
