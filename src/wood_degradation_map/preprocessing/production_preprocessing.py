"""Self-contained production preprocessing from raw 200 Hz acquisitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .diagnostics import compute_reference_band_quality
from .envi_io import (
    CubeDescriptor,
    discover_sample_cubes,
    load_reference_set,
    open_cube_memmap,
    validate_reference_against_cubes,
)
from .intensity_multiotsu_masking import (
    ThreeClassMaskResult,
    build_three_class_multiotsu_mask,
)
from .masking import connected_component_records, integrate_intensity
from .reflectance import reflectance_l2_norm
from .spectral_grid import (
    build_linear_interpolation_plan,
    derive_terminal_snr_cutoff,
    interpolate_spectra_linear,
    wavelength_grid_table,
)
from .spectral_quality import (
    ANOMALY_COLUMNS,
    aggregate_band_statistics,
    anomaly_candidates,
    apply_snv_in_place,
    band_statistics,
    extract_masked_reflectance,
    select_global_anomalies,
    snv_pixel_metrics,
)
from .visualization import (
    plot_band_distribution,
    plot_masked_scalar_map,
    plot_ranked_snv_spectra,
    plot_snr_cutoff_decision,
    shared_robust_display_limits,
)


_SCHEMA_VERSION = 2
_SNR_THRESHOLD = 10.0
_TARGET_BANDS = 256
_MASK_MIN_OBJECT_SIZE = 1
_MASK_EROSION_RADIUS = 1
_MASK_CONNECTIVITY = 2
_TOP_ANOMALY_SPECTRA = 20


@dataclass(frozen=True)
class ProductionPreprocessingConfig:
    """I/O and non-scientific chunk settings for the fixed preprocessing."""

    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed/preprocessing/200hz_snr10_linear256")
    report_dir: Path = Path("outputs/preprocessing/200hz_snr10_linear256")
    row_chunk_size: int = 64
    spectrum_chunk_size: int = 8192
    hdf5_chunk_pixels: int = 2048
    expected_width: int = 320
    expected_bands: int = 256

    def validate(self) -> None:
        """Validate implementation parameters without exposing fixed science settings."""

        if self.row_chunk_size <= 0 or self.spectrum_chunk_size <= 0:
            raise ValueError("Chunk sizes must be positive")
        if self.hdf5_chunk_pixels <= 0:
            raise ValueError("hdf5_chunk_pixels must be positive")
        if self.expected_width != 320 or self.expected_bands != 256:
            raise ValueError("The fixed input layout is width 320 with 256 bands")

    def to_json_dict(self) -> dict[str, object]:
        """Return the complete fixed production configuration."""

        values = asdict(self)
        for key in ("data_dir", "output_dir", "report_dir"):
            values[key] = str(Path(values[key]).resolve())
        values.update(
            {
                "schema_version": _SCHEMA_VERSION,
                "status": "fixed production preprocessing",
                "source": "200 Hz only",
                "mask": {
                    "score": "sum of all 256 raw intensity bands",
                    "threshold": "per-sample three-class Multi-Otsu",
                    "wood_classes": [1, 2],
                    "erosion_radius": _MASK_EROSION_RADIUS,
                    "min_object_size": _MASK_MIN_OBJECT_SIZE,
                    "connectivity": _MASK_CONNECTIVITY,
                    "postprocessing_order": ["erosion", "remove_small_objects"],
                },
                "spectral": {
                    "reflectance": "(I_200-D_200)/(W_200-D_200)",
                    "snr_threshold": _SNR_THRESHOLD,
                    "cutoff": (
                        "terminal contiguous reference bands with non-finite "
                        "or SNR proxy <= 10"
                    ),
                    "cutoff_uses_sample_spectra": False,
                    "target_bands": _TARGET_BANDS,
                    "interpolation": "linear in wavelength without extrapolation",
                    "snv": "pixel-wise mean and sample std with ddof=1",
                    "processing_order": [
                        "reflectance",
                        "automatic_terminal_snr_cutoff",
                        "linear_interpolation_to_256",
                        "SNV",
                    ],
                    "clip": False,
                    "smoothing": False,
                },
                "storage": {
                    "format": "HDF5",
                    "dtype": "float32",
                    "compression": "gzip level 4 with shuffle",
                    "unit": "one file per sample",
                },
            }
        )
        return values


def _validate_output_locations(
    data_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> None:
    processed_root = (data_dir.parent / "processed" / "preprocessing").resolve()
    repository_root = data_dir.parent.parent
    report_root = (repository_root / "outputs" / "preprocessing").resolve()
    if not output_dir.is_relative_to(processed_root):
        raise ValueError(f"Processed data must be stored below {processed_root}")
    if not report_dir.is_relative_to(report_root):
        raise ValueError(f"Preprocessing reports must be stored below {report_root}")
    if output_dir == processed_root or report_dir == report_root:
        raise ValueError("Output paths must include a preprocessing identifier")
    if output_dir.name != report_dir.name:
        raise ValueError("Data and report directories must use the same identifier")


def _validate_empty_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")


def _prepare_output_directories(output_dir: Path, report_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "reflectance_l2_norm").mkdir()


def _mask_record(
    sample_id: str,
    result: ThreeClassMaskResult,
) -> dict[str, object]:
    mask_result = result.mask_result
    record: dict[str, object] = {
        "sample_id": sample_id,
        "lower_threshold": result.thresholds[0],
        "upper_threshold": result.thresholds[1],
        "background_class_pixels": int((result.class_map == 0).sum()),
        "dark_wood_class_pixels": int((result.class_map == 1).sum()),
        "bright_wood_class_pixels": int((result.class_map == 2).sum()),
    }
    record.update(mask_result.metrics())
    return record


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else np.nan


def _write_compressed_dataset(
    handle: h5py.File,
    name: str,
    data: np.ndarray,
    *,
    chunk_rows: int | None = None,
) -> h5py.Dataset:
    data = np.asarray(data)
    if data.size == 0:
        return handle.create_dataset(name, data=data)
    chunks = None
    if chunk_rows is not None and data.ndim >= 1:
        chunks = (min(chunk_rows, data.shape[0]), *data.shape[1:])
    return handle.create_dataset(
        name,
        data=data,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _sample_id(cube: CubeDescriptor) -> str:
    if cube.sample_id is None:
        raise ValueError(f"Expected a sample cube, got reference {cube.hdr_path}")
    return cube.sample_id


def run_production_preprocessing(config: ProductionPreprocessingConfig) -> Path:
    """Run the fixed mask and spectral preprocessing directly from raw data."""

    config.validate()
    data_dir = config.data_dir.resolve()
    output_dir = config.output_dir.resolve()
    report_dir = config.report_dir.resolve()
    _validate_output_locations(data_dir, output_dir, report_dir)
    _validate_empty_directory(output_dir)
    _validate_empty_directory(report_dir)

    cubes = discover_sample_cubes(
        data_dir,
        "200",
        expected_width=config.expected_width,
        expected_bands=config.expected_bands,
    )
    reference = load_reference_set(
        data_dir,
        "200",
        expected_width=config.expected_width,
        expected_bands=config.expected_bands,
    )
    validate_reference_against_cubes(cubes, reference)
    reference_quality = compute_reference_band_quality(
        reference.white,
        reference.dark,
        reference.wavelengths_nm,
        mode="200",
        snr_threshold=_SNR_THRESHOLD,
        dtype_max=reference.dtype_max,
    )
    decision = derive_terminal_snr_cutoff(
        reference_quality,
        snr_threshold=_SNR_THRESHOLD,
    )
    interpolation_plan = build_linear_interpolation_plan(
        reference.wavelengths_nm[: decision.retained_source_bands],
        target_bands=_TARGET_BANDS,
    )

    masks: dict[str, np.ndarray] = {}
    mask_records: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []
    for cube in cubes:
        sample_id = _sample_id(cube)
        intensity_integral = integrate_intensity(open_cube_memmap(cube))
        mask_result = build_three_class_multiotsu_mask(
            intensity_integral,
            min_object_size=_MASK_MIN_OBJECT_SIZE,
            erosion_radius=_MASK_EROSION_RADIUS,
            connectivity=_MASK_CONNECTIVITY,
        )
        masks[sample_id] = mask_result.mask_result.final_mask
        mask_records.append(_mask_record(sample_id, mask_result))
        for component in connected_component_records(
            mask_result.mask_result.after_erosion,
            connectivity=_MASK_CONNECTIVITY,
        ):
            component_records.append({"sample_id": sample_id, **component})

    _prepare_output_directories(output_dir, report_dir)

    manifest_records: list[dict[str, object]] = []
    sample_quality_records: list[dict[str, object]] = []
    source_band_frames: list[pd.DataFrame] = []
    output_band_frames: list[pd.DataFrame] = []
    anomaly_candidate_records: list[dict[str, object]] = []
    reflectance_l2_maps: dict[str, np.ndarray] = {}
    source_low_snr = reference_quality["low_snr"].to_numpy(dtype=bool)
    output_low_snr = np.zeros(_TARGET_BANDS, dtype=bool)
    for cube in cubes:
        sample_id = _sample_id(cube)
        mask = masks[sample_id]
        full_reflectance, all_coordinates = extract_masked_reflectance(
            open_cube_memmap(cube),
            cube,
            reference,
            mask,
            row_chunk_size=config.row_chunk_size,
        )
        source_band_frames.append(
            band_statistics(
                full_reflectance,
                sample_id=sample_id,
                stage="reflectance",
                wavelengths_nm=reference.wavelengths_nm,
                low_snr=source_low_snr,
            )
        )
        retained_reflectance = full_reflectance[
            :, : decision.retained_source_bands
        ].copy()
        full_snv_valid, _, _, _ = apply_snv_in_place(
            full_reflectance,
            spectrum_chunk_size=config.spectrum_chunk_size,
        )
        source_band_frames.append(
            band_statistics(
                full_reflectance,
                sample_id=sample_id,
                stage="snv",
                wavelengths_nm=reference.wavelengths_nm,
                low_snr=source_low_snr,
            )
        )

        finite_retained = np.isfinite(retained_reflectance).all(axis=1)
        excluded_reason = np.zeros(len(retained_reflectance), dtype=np.uint8)
        excluded_reason[~finite_retained] = 1
        finite_indices = np.flatnonzero(finite_retained)
        interpolated = interpolate_spectra_linear(
            retained_reflectance[finite_retained],
            interpolation_plan,
            spectrum_chunk_size=config.spectrum_chunk_size,
        )
        interpolated_std = interpolated.std(axis=1, ddof=1, dtype=np.float64)
        valid_variance = np.isfinite(interpolated_std) & (interpolated_std > 0.0)
        excluded_reason[finite_indices[~valid_variance]] = 2
        final_indices = finite_indices[valid_variance]
        if not final_indices.size:
            raise ValueError(f"No valid spectra remain for {sample_id}")
        final_reflectance = (
            interpolated if valid_variance.all() else interpolated[valid_variance]
        )
        negative_pixel_count = int((final_reflectance < 0.0).any(axis=1).sum())
        above_one_pixel_count = int((final_reflectance > 1.0).any(axis=1).sum())
        coordinates = all_coordinates[final_indices]
        valid_spectrum_mask = np.zeros(mask.shape, dtype=bool)
        valid_spectrum_mask[coordinates[:, 0], coordinates[:, 1]] = True
        excluded = excluded_reason > 0
        excluded_coordinates = all_coordinates[excluded]
        excluded_codes = excluded_reason[excluded]

        output_band_frames.append(
            band_statistics(
                final_reflectance,
                sample_id=sample_id,
                stage="reflectance",
                wavelengths_nm=interpolation_plan.target_wavelength_nm,
                low_snr=output_low_snr,
            )
        )
        reflectance_l2, finite_band_count = reflectance_l2_norm(final_reflectance)
        if not (finite_band_count == _TARGET_BANDS).all():
            raise RuntimeError(f"Unexpected non-finite L2 input: {sample_id}")
        reflectance_l2_map = np.full(mask.shape, np.nan, dtype=np.float32)
        reflectance_l2_map[coordinates[:, 0], coordinates[:, 1]] = reflectance_l2
        reflectance_l2_maps[sample_id] = reflectance_l2_map

        hdf5_path = output_dir / "samples" / f"{sample_id}.h5"
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset(
                "wavelength_nm",
                data=interpolation_plan.target_wavelength_nm,
            )
            handle.create_dataset(
                "source_wavelength_nm",
                data=interpolation_plan.source_wavelength_nm,
            )
            handle.create_dataset(
                "retained_source_band_index",
                data=np.arange(decision.retained_source_bands, dtype=np.int32),
            )
            _write_compressed_dataset(handle, "mask", mask.astype(np.uint8))
            _write_compressed_dataset(
                handle,
                "valid_spectrum_mask",
                valid_spectrum_mask.astype(np.uint8),
            )
            _write_compressed_dataset(
                handle,
                "pixel_row_col",
                coordinates.astype(np.int32),
                chunk_rows=config.hdf5_chunk_pixels,
            )
            _write_compressed_dataset(
                handle,
                "excluded_pixel_row_col",
                excluded_coordinates.astype(np.int32),
                chunk_rows=config.hdf5_chunk_pixels,
            )
            _write_compressed_dataset(
                handle,
                "excluded_reason_code",
                excluded_codes.astype(np.uint8),
                chunk_rows=config.hdf5_chunk_pixels,
            )
            _write_compressed_dataset(
                handle,
                "reflectance",
                final_reflectance,
                chunk_rows=config.hdf5_chunk_pixels,
            )
            _write_compressed_dataset(
                handle,
                "reflectance_l2_norm",
                reflectance_l2_map,
            )

            snv_valid, input_mean, input_std, nonfinite_count = apply_snv_in_place(
                final_reflectance,
                spectrum_chunk_size=config.spectrum_chunk_size,
            )
            if not snv_valid.all() or nonfinite_count.any():
                raise RuntimeError(f"Unexpected invalid SNV after filtering: {sample_id}")
            output_band_frames.append(
                band_statistics(
                    final_reflectance,
                    sample_id=sample_id,
                    stage="snv",
                    wavelengths_nm=interpolation_plan.target_wavelength_nm,
                    low_snr=output_low_snr,
                )
            )
            snv_metrics = snv_pixel_metrics(
                final_reflectance,
                snv_valid,
                output_low_snr,
                spectrum_chunk_size=config.spectrum_chunk_size,
            )
            anomaly_candidate_records.extend(
                anomaly_candidates(
                    sample_id=sample_id,
                    coordinates=coordinates,
                    snv=final_reflectance,
                    input_mean=input_mean,
                    input_std=input_std,
                    metrics=snv_metrics,
                    per_sample_count=_TOP_ANOMALY_SPECTRA,
                )
            )
            _write_compressed_dataset(
                handle,
                "snv",
                final_reflectance,
                chunk_rows=config.hdf5_chunk_pixels,
            )
            handle.attrs["schema_version"] = _SCHEMA_VERSION
            handle.attrs["sample_id"] = sample_id
            handle.attrs["preprocessing_id"] = output_dir.name
            handle.attrs["source"] = "200 Hz only"
            handle.attrs["snr_threshold"] = _SNR_THRESHOLD
            handle.attrs["snv_ddof"] = 1
            handle.attrs["source_hsi_200_hdr"] = cube.hdr_path.name
            handle.attrs["source_hsi_200_raw"] = cube.raw_path.name
            handle.attrs["source_height"] = cube.shape[0]
            handle.attrs["source_width"] = cube.shape[1]
            handle.attrs["source_bands"] = cube.shape[2]
            handle.attrs["target_bands"] = _TARGET_BANDS
            handle.attrs["first_excluded_band"] = (
                decision.first_excluded_band
                if decision.first_excluded_band is not None
                else -1
            )
            handle.attrs["mask_pixel_count"] = int(mask.sum())
            handle.attrs["saved_pixel_count"] = len(coordinates)
            handle.attrs["excluded_pixel_count"] = int(excluded.sum())
            handle.attrs["excluded_reason_codes"] = json.dumps(
                {
                    "1": "nonfinite retained reflectance",
                    "2": "nonpositive SNV input std",
                }
            )

        hdf5_bytes = hdf5_path.stat().st_size
        uncompressed_reflectance_snv_bytes = int(
            2 * final_reflectance.size * np.dtype(np.float32).itemsize
        )
        expected_l2 = np.sqrt(_TARGET_BANDS - 1)
        snv_means = final_reflectance.mean(axis=1, dtype=np.float64)
        snv_stds = final_reflectance.std(axis=1, ddof=1, dtype=np.float64)
        snv_l2 = np.linalg.norm(final_reflectance.astype(np.float64), axis=1)
        sample_quality_records.append(
            {
                "sample_id": sample_id,
                "mask_pixel_count": int(mask.sum()),
                "source_full_band_invalid_snv_pixel_count": int(
                    (~full_snv_valid).sum()
                ),
                "nonfinite_retained_reflectance_pixel_count": int(
                    (~finite_retained).sum()
                ),
                "nonpositive_snv_input_std_pixel_count": int(
                    (~valid_variance).sum()
                ),
                "saved_pixel_count": len(coordinates),
                "excluded_pixel_count": int(excluded.sum()),
                "excluded_pixel_fraction": float(excluded.mean()),
                "pixels_with_any_negative_interpolated_reflectance": (
                    negative_pixel_count
                ),
                "pixels_with_any_interpolated_reflectance_above_one": (
                    above_one_pixel_count
                ),
                "final_snv_first_difference_rms_q99": _finite_quantile(
                    snv_metrics["first_difference_rms"],
                    0.99,
                ),
                "final_snv_max_abs_second_difference_q99": _finite_quantile(
                    snv_metrics["max_abs_second_difference"],
                    0.99,
                ),
                "final_snv_max_abs_second_difference_max": _finite_quantile(
                    snv_metrics["max_abs_second_difference"],
                    1.0,
                ),
                "snv_mean_absolute_max": float(np.max(np.abs(snv_means))),
                "snv_sample_std_absolute_error_max": float(
                    np.max(np.abs(snv_stds - 1.0))
                ),
                "snv_l2_norm_absolute_error_max": float(
                    np.max(np.abs(snv_l2 - expected_l2))
                ),
            }
        )
        manifest_records.append(
            {
                "sample_id": sample_id,
                "file": (Path("samples") / f"{sample_id}.h5").as_posix(),
                "source_hsi_200_hdr": cube.hdr_path.name,
                "source_hsi_200_raw": cube.raw_path.name,
                "height": cube.shape[0],
                "width": cube.shape[1],
                "bands": _TARGET_BANDS,
                "saved_pixel_count": len(coordinates),
                "excluded_pixel_count": int(excluded.sum()),
                "hdf5_bytes": hdf5_bytes,
                "uncompressed_reflectance_snv_bytes": (
                    uncompressed_reflectance_snv_bytes
                ),
                "preprocessing_id": output_dir.name,
            }
        )

    mask_quality = pd.DataFrame(mask_records)
    mask_quality.to_parquet(output_dir / "mask_quality.parquet", index=False)
    component_columns = [
        "sample_id",
        "component_label",
        "area_pixels",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row_exclusive",
        "bbox_max_col_exclusive",
        "touches_image_boundary",
    ]
    pd.DataFrame(component_records, columns=component_columns).to_parquet(
        output_dir / "mask_components.parquet",
        index=False,
    )
    reference_quality = reference_quality.copy()
    reference_quality["retained"] = (
        reference_quality["band_index"] < decision.retained_source_bands
    )
    reference_quality.to_parquet(
        output_dir / "reference_band_quality.parquet",
        index=False,
    )
    wavelength_grid_table(interpolation_plan).to_parquet(
        output_dir / "wavelength_grid.parquet",
        index=False,
    )
    manifest = pd.DataFrame(manifest_records)
    manifest.to_parquet(output_dir / "manifest.parquet", index=False)
    sample_quality = pd.DataFrame(sample_quality_records)
    sample_quality.to_parquet(output_dir / "sample_quality.parquet", index=False)

    source_band_statistics = pd.concat(source_band_frames, ignore_index=True)
    source_band_summary = aggregate_band_statistics(source_band_statistics)
    source_band_statistics.to_parquet(
        output_dir / "source_band_statistics.parquet",
        index=False,
    )
    source_band_summary.to_parquet(
        output_dir / "source_band_summary.parquet",
        index=False,
    )
    output_band_statistics = pd.concat(output_band_frames, ignore_index=True)
    output_band_summary = aggregate_band_statistics(output_band_statistics)
    output_band_statistics.to_parquet(
        output_dir / "output_band_statistics.parquet",
        index=False,
    )
    output_band_summary.to_parquet(
        output_dir / "output_band_summary.parquet",
        index=False,
    )

    selected_anomalies = select_global_anomalies(
        anomaly_candidate_records,
        top_spectra=_TOP_ANOMALY_SPECTRA,
        wavelengths_nm=interpolation_plan.target_wavelength_nm,
    )
    anomaly_frames = [
        frame for frame in selected_anomalies.values() if not frame.empty
    ]
    ranked_anomalies = (
        pd.concat(anomaly_frames, ignore_index=True)
        if anomaly_frames
        else pd.DataFrame(columns=ANOMALY_COLUMNS)
    )
    ranked_anomalies.to_parquet(
        output_dir / "ranked_final_snv_spectra.parquet",
        index=False,
    )

    plot_snr_cutoff_decision(
        reference_quality,
        source_band_summary,
        report_dir / "cutoff_decision.png",
        snr_threshold=_SNR_THRESHOLD,
        cutoff_boundary_nm=decision.cutoff_boundary_nm,
    )
    output_reflectance_summary = output_band_summary.loc[
        output_band_summary["stage"] == "reflectance"
    ]
    output_snv_summary = output_band_summary.loc[
        output_band_summary["stage"] == "snv"
    ]
    plot_band_distribution(
        output_reflectance_summary,
        report_dir / "interpolated_reflectance_band_distribution.png",
        y_label="Reflectance",
    )
    plot_band_distribution(
        output_snv_summary,
        report_dir / "interpolated_snv_band_distribution.png",
        y_label="SNV",
    )
    final_anomalies = selected_anomalies["max_abs_second_difference"]
    if not final_anomalies.empty:
        plot_ranked_snv_spectra(
            final_anomalies,
            output_reflectance_summary,
            output_snv_summary,
            report_dir / "final_snv_anomaly_candidates.png",
        )
    reflectance_l2_display_limits = shared_robust_display_limits(
        list(reflectance_l2_maps.values())
    )
    for sample_id, reflectance_l2_map in reflectance_l2_maps.items():
        plot_masked_scalar_map(
            reflectance_l2_map,
            report_dir / "reflectance_l2_norm" / f"{sample_id}.png",
            display_limits=reflectance_l2_display_limits,
            cmap="plasma",
        )

    fixed_config = config.to_json_dict()
    fixed_config["cutoff_decision"] = decision.to_json_dict()
    fixed_config["target_wavelength_start_nm"] = float(
        interpolation_plan.target_wavelength_nm[0]
    )
    fixed_config["target_wavelength_end_nm"] = float(
        interpolation_plan.target_wavelength_nm[-1]
    )
    fixed_config["target_wavelength_spacing_nm"] = float(
        interpolation_plan.target_wavelength_nm[1]
        - interpolation_plan.target_wavelength_nm[0]
    )
    (output_dir / "config.json").write_text(
        json.dumps(fixed_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cutoff_decision.json").write_text(
        json.dumps(decision.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_mask_pixels = int(sample_quality["mask_pixel_count"].sum())
    total_saved_pixels = int(sample_quality["saved_pixel_count"].sum())
    total_excluded_pixels = int(sample_quality["excluded_pixel_count"].sum())
    total_hdf5_bytes = int(manifest["hdf5_bytes"].sum())
    total_uncompressed_spectra_bytes = int(
        manifest["uncompressed_reflectance_snv_bytes"].sum()
    )
    preprocessing_summary = {
        "schema_version": _SCHEMA_VERSION,
        "preprocessing_id": output_dir.name,
        "sample_count": len(cubes),
        "cutoff_decision": decision.to_json_dict(),
        "target_bands": _TARGET_BANDS,
        "mask_pixel_count": total_mask_pixels,
        "saved_pixel_count": total_saved_pixels,
        "excluded_pixel_count": total_excluded_pixels,
        "excluded_pixel_fraction": total_excluded_pixels / total_mask_pixels,
        "hdf5_total_bytes": total_hdf5_bytes,
        "hdf5_total_gib": total_hdf5_bytes / (1024**3),
        "uncompressed_reflectance_snv_bytes": total_uncompressed_spectra_bytes,
        "hdf5_total_to_uncompressed_reflectance_snv_ratio": (
            total_hdf5_bytes / total_uncompressed_spectra_bytes
        ),
        "snv_expected_l2_norm": float(np.sqrt(_TARGET_BANDS - 1)),
        "snv_mean_absolute_max": float(
            sample_quality["snv_mean_absolute_max"].max()
        ),
        "snv_sample_std_absolute_error_max": float(
            sample_quality["snv_sample_std_absolute_error_max"].max()
        ),
        "snv_l2_norm_absolute_error_max": float(
            sample_quality["snv_l2_norm_absolute_error_max"].max()
        ),
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_config = {
        "schema_version": _SCHEMA_VERSION,
        "preprocessing_id": output_dir.name,
        "processed_data_dir": str(output_dir),
        "title_policy": "no figure or axes titles",
        "reflectance_l2_norm": {
            "spectra": "interpolated reflectance before SNV",
            "cmap": "plasma",
            "vmin": reflectance_l2_display_limits[0],
            "vmax": reflectance_l2_display_limits[1],
            "scope": "all samples",
            "background": "transparent",
            "colorbar": False,
        },
    }
    (report_dir / "report_config.json").write_text(
        json.dumps(report_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir
