from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.preprocessing import production_preprocessing as production
from wood_degradation_map.preprocessing.intensity_multiotsu_masking import ThreeClassMaskResult
from wood_degradation_map.preprocessing.masking import MaskResult
from wood_degradation_map.preprocessing.production_preprocessing import (
    ProductionPreprocessingConfig,
    _interpolated_exclusion_reasons,
    _validate_output_locations,
)


def test_fixed_production_configuration_records_science_settings() -> None:
    config = ProductionPreprocessingConfig()

    recorded = config.to_json_dict()

    assert recorded["schema_version"] == 2
    assert recorded["source"] == "200 Hz only"
    assert recorded["mask"]["erosion_radius"] == 1
    assert recorded["mask"]["min_object_size"] == 1
    assert recorded["spectral"]["snr_threshold"] == 10.0
    assert recorded["spectral"]["target_bands"] == 256
    assert recorded["spectral"]["negative_reflectance_policy"]["reason_code"] == 3
    assert config.output_dir.name == config.report_dir.name == "production_v1"
    assert config.output_dir == Path("data/processed/production_v1")


def test_production_outputs_are_separated_by_purpose(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "data" / "processed" / "fixed"
    report_dir = tmp_path / "outputs" / "preprocessing" / "fixed"

    _validate_output_locations(data_dir, output_dir, report_dir)


def test_processed_data_cannot_be_written_below_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs" / "preprocessing" / "fixed"
    report_dir = tmp_path / "outputs" / "preprocessing" / "fixed"

    with pytest.raises(ValueError, match="Processed data"):
        _validate_output_locations(data_dir, output_dir, report_dir)


def test_pre_snv_negative_filter_keeps_zero_bands_and_above_one_values() -> None:
    rows = np.tile(np.linspace(0.0, 1.2, 256, dtype=np.float32), (4, 1))
    rows[1, 100] = -np.finfo(np.float32).tiny
    rows[2] = 0.0
    rows[3] = -0.5
    before = rows.copy()
    np.testing.assert_array_equal(_interpolated_exclusion_reasons(rows), [0, 3, 2, 2])
    np.testing.assert_array_equal(rows, before)
    # A retained nonnegative spectrum normally becomes partly negative after SNV.
    normalized = (rows[0] - rows[0].mean()) / rows[0].std(ddof=1)
    assert (normalized < 0).any()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_interpolated_variance_retains_existing_reason(value: float) -> None:
    rows = np.linspace(0.1, 0.8, 256)[None, :]
    rows[0, 128] = value
    np.testing.assert_array_equal(_interpolated_exclusion_reasons(rows), [2])


def _write_test_envi(path: Path, values: np.ndarray) -> None:
    """Write a tiny BIP fixture, without reading any production data."""
    height, width, bands = values.shape
    wavelengths = ", ".join(str(value) for value in np.linspace(900, 2300, bands))
    path.write_text(
        f"ENVI\nsamples = {width}\nlines = {height}\nbands = {bands}\n"
        "header offset = 0\nfile type = ENVI Standard\ndata type = 5\n"
        "interleave = bip\nbyte order = 0\nfps = 200\n"
        f"wavelength = {{{wavelengths}}}\n", encoding="utf-8",
    )
    values.astype("<f8").tofile(path.with_suffix(".raw"))


def test_negative_pixels_are_background_in_saved_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    dark = np.full((1, 320, 256), 100.0)
    denominator = np.broadcast_to(1000.0 + np.arange(320)[None, :, None], dark.shape)
    _write_test_envi(data / "200hz_dark.hdr", dark)
    _write_test_envi(data / "200hz_white.hdr", dark + denominator)
    spectra = np.tile(np.linspace(0.1, 0.8, 256), (5, 1))
    spectra[1, 100] = -0.1
    spectra[2] = 0.0
    spectra[3, 0] = 0.0
    spectra[4, -1] = 1.2
    raw = np.broadcast_to(dark, (2, 320, 256)).copy()
    raw[0, :5] += spectra * denominator[0, :5]
    _write_test_envi(data / "200hz_KYOw00001.hdr", raw)

    # Isolate the spectral policy from Multi-Otsu decisions on artificial intensities.
    mask = np.zeros((2, 320), dtype=bool)
    mask[0, :5] = True
    mask_result = MaskResult(mask.astype(float), 0.5, mask, mask, mask, 1, 1)

    def fixed_mask(*args: object, **kwargs: object) -> ThreeClassMaskResult:
        return ThreeClassMaskResult((0.5, 1.5), mask.astype(np.uint8), mask_result)

    def skip_plot(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(production, "build_three_class_multiotsu_mask", fixed_mask)
    for name in ("plot_snr_cutoff_decision", "plot_band_distribution",
                 "plot_ranked_snv_spectra", "plot_masked_scalar_map"):
        monkeypatch.setattr(production, name, skip_plot)
    config = ProductionPreprocessingConfig(
        data_dir=data, output_dir=tmp_path / "data/processed/test_v2",
        report_dir=tmp_path / "outputs/preprocessing/test_v2",
    )
    production.run_production_preprocessing(config)
    with h5py.File(config.output_dir / "samples/KYOw00001.h5", "r") as saved:
        np.testing.assert_array_equal(saved["mask"][:], mask)
        np.testing.assert_array_equal(saved["valid_spectrum_mask"][0, :5], [1, 0, 0, 1, 1])
        np.testing.assert_array_equal(saved["pixel_row_col"][:], [[0, 0], [0, 3], [0, 4]])
        np.testing.assert_array_equal(saved["excluded_pixel_row_col"][:], [[0, 1], [0, 2]])
        np.testing.assert_array_equal(saved["excluded_reason_code"][:], [3, 2])
        np.testing.assert_allclose(saved["reflectance"][:], spectra[[0, 3, 4]], atol=1e-6)
        assert (saved["reflectance"][:] >= 0).all()
        assert (saved["snv"][:] < 0).any()
        assert np.isnan(saved["reflectance_l2_norm"][0, 1:3]).all()
        assert "3" in json.loads(saved.attrs["excluded_reason_codes"])
    quality = pd.read_parquet(config.output_dir / "sample_quality.parquet").iloc[0]
    assert quality["mask_pixel_count"] == 5
    assert quality["saved_pixel_count"] == 3
    assert quality["excluded_pixel_count"] == 2
    assert quality["negative_interpolated_reflectance_excluded_pixel_count"] == 1
    assert quality["pixels_with_any_negative_interpolated_reflectance"] == 0
    assert quality["pixels_with_any_interpolated_reflectance_above_one"] == 1
    summary = json.loads((config.output_dir / "preprocessing_summary.json").read_text())
    assert summary["negative_interpolated_reflectance_excluded_pixel_count"] == 1
    with pytest.raises(FileExistsError):
        production.run_production_preprocessing(config)
