"""Save or check complete OOF summaries, three-repeat ARI and planned comparisons (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wood_degradation_map.experiments.config import CONDITIONS
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, load_manifest_bundle
from wood_degradation_map.experiments.oof_pipeline import check_oof, run_oof


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--conditions", nargs="+", choices=[c.condition_id for c in CONDITIONS],
                        help="Run only: all five folds and three repeats are required for every condition")
    parser.add_argument("--snapshot", help="Output name; run defaults to UTC timestamp, check requires a name")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/cv_200hz_snr10_linear256_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/preprocessing/200hz_snr10_linear256")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    args = parser.parse_args()
    if args.action == "run" and not args.conditions:
        parser.error("run requires --conditions")
    if args.action == "check" and (args.conditions is not None or args.snapshot is None):
        parser.error("check requires --snapshot and reads conditions from that snapshot")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_manifest_bundle(experiment, inventory, args.processed_dir, args.metadata)
    if args.action == "check":
        report = check_oof(experiment, inventory, manifest, args.snapshot)
    else:
        output = run_oof(experiment, inventory, manifest, tuple(args.conditions), snapshot=args.snapshot)
        report = {"output_dir": str(output), **_read_json(output / "completion.json")}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
