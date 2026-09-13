"""Redraw spectral reports from saved QC tables without rerunning preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wood_degradation_map.preprocessing.visualization import (
    plot_band_distribution,
    plot_ranked_snv_spectra,
    plot_snr_cutoff_decision,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data/processed/production_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=root / "outputs/preprocessing/production_v1",
        help="Destination for the four spectral PNGs; existing PNGs are replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.processed_dir.resolve()
    report = args.report_dir.resolve()
    reference = pd.read_parquet(processed / "reference_band_quality.parquet")
    source_summary = pd.read_parquet(processed / "source_band_summary.parquet")
    output_summary = pd.read_parquet(processed / "output_band_summary.parquet")
    candidates = pd.read_parquet(processed / "ranked_final_snv_spectra.parquet")
    decision = json.loads((processed / "cutoff_decision.json").read_text(encoding="utf-8"))
    reflectance = output_summary.loc[output_summary["stage"] == "reflectance"]
    snv = output_summary.loc[output_summary["stage"] == "snv"]
    candidates = candidates.loc[candidates["metric_name"] == "max_abs_second_difference"]

    plot_snr_cutoff_decision(
        reference, source_summary, report / "cutoff_decision.png",
        snr_threshold=decision["snr_threshold"],
        cutoff_boundary_nm=decision["cutoff_boundary_nm"],
    )
    for summary, stage, label in (
        (reflectance, "reflectance", "Reflectance"),
        (snv, "snv", "SNV"),
    ):
        plot_band_distribution(
            summary, report / f"interpolated_{stage}_band_distribution.png", y_label=label,
        )
    if not candidates.empty:
        plot_ranked_snv_spectra(
            candidates, reflectance, snv, report / "final_snv_anomaly_candidates.png",
        )
    print(f"Spectral reports redrawn in {report}; reflectance_l2_norm was not accessed.")


if __name__ == "__main__":
    main()
