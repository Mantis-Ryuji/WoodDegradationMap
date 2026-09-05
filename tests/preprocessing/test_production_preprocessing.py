from __future__ import annotations

from pathlib import Path

import pytest

from wood_degradation_map.preprocessing.production_preprocessing import (
    ProductionPreprocessingConfig,
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


def test_production_outputs_are_separated_by_purpose(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "data" / "processed" / "preprocessing" / "fixed"
    report_dir = tmp_path / "outputs" / "preprocessing" / "fixed"

    _validate_output_locations(data_dir, output_dir, report_dir)


def test_processed_data_cannot_be_written_below_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs" / "preprocessing" / "fixed"
    report_dir = tmp_path / "outputs" / "preprocessing" / "fixed"

    with pytest.raises(ValueError, match="Processed data"):
        _validate_output_locations(data_dir, output_dir, report_dir)
