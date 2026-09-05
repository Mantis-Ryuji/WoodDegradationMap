"""Run the fixed production preprocessing directly from raw 200 Hz data."""

from __future__ import annotations

import argparse
from pathlib import Path

from wood_degradation_map.preprocessing import (
    ProductionPreprocessingConfig,
    run_production_preprocessing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the raw ENVI acquisitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/preprocessing/200hz_snr10_linear256"),
        help="New or empty directory for processed data.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/preprocessing/200hz_snr10_linear256"),
        help="New or empty directory for preprocessing figures.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=64,
        help="Number of image rows converted to reflectance at once.",
    )
    parser.add_argument(
        "--spectrum-chunk-size",
        type=int,
        default=8192,
        help="Number of spectra interpolated or SNV-transformed at once.",
    )
    parser.add_argument(
        "--hdf5-chunk-pixels",
        type=int,
        default=2048,
        help="Pixel rows per compressed HDF5 spectrum chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProductionPreprocessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        row_chunk_size=args.row_chunk_size,
        spectrum_chunk_size=args.spectrum_chunk_size,
        hdf5_chunk_pixels=args.hdf5_chunk_pixels,
    )
    output_dir = run_production_preprocessing(config)
    print(f"Production dataset written to {output_dir}")
    print(f"Preprocessing report written to {config.report_dir.resolve()}")


if __name__ == "__main__":
    main()
