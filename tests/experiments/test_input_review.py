"""Small table-only review export; no source-image decoding or HDF5 scan."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import h5py
import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments.input_review import OVERVIEW_IMAGES, TABLES, prepare_input_review
from wood_degradation_map.experiments.input_validation import InputInventory, load_input_inventory
from wood_degradation_map.experiments.manifests import _digest, _read_json

from test_input_validation import saved_input  # noqa: F401

Fixture = tuple[Path, Path, Path, InputInventory]


@pytest.fixture
def review_input(saved_input: tuple[Path, Path], tmp_path: Path) -> Fixture:
    processed, metadata = saved_input
    inventory = load_input_inventory(processed, metadata)
    quality = pd.read_parquet(processed / "sample_quality.parquet")
    quality["snv_error"] = 1.0797521099448204e-8
    quality["optional_diagnostic"] = np.nan
    quality["text"] = '<script>alert("escaped")</script>'
    quality.to_parquet(processed / "sample_quality.parquet", index=False)
    pd.DataFrame({"sample_id": ["KYOw02702"], "component_count": [2]}).to_parquet(
        processed / "mask_quality.parquet", index=False)
    pd.DataFrame({"band_index": [0, 1], "snr_proxy": [12.1, 15.9]}).to_parquet(
        processed / "reference_band_quality.parquet", index=False)
    pd.DataFrame({"stage": ["snv"], "band_index": [0], "median": [0.23]}).to_parquet(
        processed / "output_band_summary.parquet", index=False)
    cutoff = {"last_retained_band": 221}
    (processed / "cutoff_decision.json").write_text(json.dumps(cutoff), encoding="utf-8")
    (processed / "preprocessing_summary.json").write_text(json.dumps({
        "preprocessing_id": processed.name, "sample_count": 1, "saved_pixel_count": 4,
        "cutoff_decision": cutoff,
    }), encoding="utf-8")
    figures = tmp_path / "figures with spaces"
    (figures / "reflectance_l2_norm").mkdir(parents=True)
    for name in [filename for filename, _ in OVERVIEW_IMAGES] + ["reflectance_l2_norm/KYOw02702.png"]:
        # Only file presence/linking is promised; decoding images is not part of export.
        (figures / name).write_bytes(b"image placeholder")
    (figures / "report_config.json").write_text(json.dumps({
        "preprocessing_id": processed.name, "processed_data_dir": str(processed),
        "reflectance_l2_norm": {"scope": "all samples", "vmin": 4.18, "vmax": 9.62, "cmap": "plasma"},
    }), encoding="utf-8")
    return processed, figures, tmp_path / "review", inventory


def test_export_preserves_all_tables_values_links_and_source_files(
    review_input: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed, figures, output, inventory = review_input
    before = {path: _digest(path) for path in processed.iterdir() if path.is_file()}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Review must not open HDF5 or regenerate spectra")

    monkeypatch.setattr(h5py, "File", forbidden)
    page = prepare_input_review(processed, figures, output, inventory)
    text = page.read_text(encoding="utf-8")
    assert text.count('<img ') == 5 and 'KYOw02702' in text
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert format(1.0797521099448204e-8, ".17g") in text
    assert "NA（保存表の欠損）" in text
    assert "figures%20with%20spaces" in text
    assert "../figures with spaces/reflectance_l2_norm/KYOw02702.png" in unquote(text)
    for name in TABLES:
        original = pd.read_parquet(processed / f"{name}.parquet")
        restored = pd.read_csv(output / f"{name}.csv")
        assert list(restored.columns) == list(original.columns)
        pd.testing.assert_frame_equal(restored, original, check_dtype=False)
    report = _read_json(output / "review.json")
    assert report["status"] == "input_review_prepared" and report["manual_review_required"] is True
    assert report["missing_images"] == [] and report["sample_count"] == 1
    assert report["html_sha256"] == _digest(page)
    assert {path: _digest(path) for path in before} == before


def test_missing_image_is_visible_without_removing_sample(review_input: Fixture) -> None:
    processed, figures, output, inventory = review_input
    (figures / "reflectance_l2_norm/KYOw02702.png").unlink()
    page = prepare_input_review(processed, figures, output, inventory)
    report = _read_json(output / "review.json")
    assert len(report["missing_images"]) == 1
    assert report["sample_ids"] == ["KYOw02702"]
    assert '画像がありません' in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("change", ["mask_id", "quality_count", "summary_count", "cutoff", "figure_source", "figure_id"])
def test_mismatched_records_rejected_before_output(review_input: Fixture, change: str) -> None:
    processed, figures, output, inventory = review_input
    if change == "mask_id":
        pd.DataFrame({"sample_id": ["KYOw99999"]}).to_parquet(processed / "mask_quality.parquet")
    elif change == "quality_count":
        quality = pd.read_parquet(processed / "sample_quality.parquet")
        quality["saved_pixel_count"] = 3
        quality.to_parquet(processed / "sample_quality.parquet")
    elif change in ("summary_count", "cutoff"):
        path = processed / "preprocessing_summary.json"
        record = _read_json(path)
        record["sample_count" if change == "summary_count" else "cutoff_decision"] = 9
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path = figures / "report_config.json"
        record = _read_json(path)
        record["processed_data_dir" if change == "figure_source" else "preprocessing_id"] = "different"
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_input_review(processed, figures, output, inventory)
    assert not output.exists()


def test_existing_output_is_never_overwritten(review_input: Fixture) -> None:
    processed, figures, output, inventory = review_input
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_input_review(processed, figures, output, inventory)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_output_cannot_be_created_inside_processed_data(review_input: Fixture) -> None:
    processed, figures, _, inventory = review_input
    with pytest.raises(ValueError, match="separate"):
        prepare_input_review(processed, figures, processed / "new_review", inventory)
