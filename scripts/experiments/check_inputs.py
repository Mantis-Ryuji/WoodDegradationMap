"""Check saved production tables and bounded HDF5 rows without writing files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from wood_degradation_map.experiments.input_validation import (
    load_input_inventory,
    probe_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--processed-dir", type=Path,
        default=root / "data/processed/preprocessing/200hz_snr10_linear256",
    )
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--rows-per-sample", type=int, choices=range(1, 65), default=8,
                        metavar="1..64")
    parser.add_argument("--sample-id", action="append",
                        help="HDF5 sample to probe; repeat to select several. Default: all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    ids = {sample.sample_id for sample in inventory.samples}
    selected = set(args.sample_id) if args.sample_id else ids
    if selected - ids:
        raise ValueError(f"Unknown sample IDs: {sorted(selected - ids)}")
    probes = [
        probe_sample(
            sample, preprocessing_id=inventory.preprocessing_id,
            wavelength_start_nm=inventory.wavelength_start_nm,
            wavelength_end_nm=inventory.wavelength_end_nm,
            rows_per_sample=args.rows_per_sample,
        )
        for sample in inventory.samples if sample.sample_id in selected
    ]
    report = {
        "status": "passed_table_and_sampled_row_checks",
        "processed_dir": str(args.processed_dir.resolve()),
        "metadata": str(args.metadata.resolve()),
        "preprocessing_id": inventory.preprocessing_id,
        "candidate_sample_count": len(inventory.samples),
        "candidate_sample_ids": sorted(ids),
        "metadata_only_ids": inventory.metadata_only_ids,
        "total_saved_pixel_count": sum(sample.saved_pixel_count for sample in inventory.samples),
        "unchecked_hdf5_sample_ids": sorted(ids - selected),
        "probes": [asdict(probe) for probe in probes],
        "limitations": [
            "This check does not determine experimental inclusion or infer source relationships.",
            "Only reported HDF5 rows were checked; unsampled pixels were not validated.",
            "Full mask coverage and global coordinate uniqueness were not checked.",
            "No split, pixel sampling manifest, or preprocessing data was generated.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
