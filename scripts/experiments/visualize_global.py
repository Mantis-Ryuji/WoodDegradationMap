"""Render/check global K=8 PNG/CSV reports with M00 observed SNV cosine matching.

Run replaces the previous report directory only after successful generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wood_degradation_map.experiments.global_manifest import GlobalData, load_global_bundle
from wood_degradation_map.experiments.global_reporting import (
    DEFAULT_DIRECTORY, check_global_report, render_global_report,
)
from wood_degradation_map.experiments.input_validation import load_input_inventory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--chunk-pixels", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--resume", action="store_true", help="Verify/skip an already complete report")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or args.dpi < 1:
        parser.error("Require positive chunk size and DPI")
    if args.resume and args.action != "run":
        parser.error("--resume is for run only")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    if args.action == "check" or (args.resume and (experiment / DEFAULT_DIRECTORY).exists()):
        report = check_global_report(experiment, data, inventory)
    else:
        render_global_report(experiment, data, inventory, dpi=args.dpi,
                             chunk_pixels=args.chunk_pixels)
        report = check_global_report(experiment, data, inventory)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
