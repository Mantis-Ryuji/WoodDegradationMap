"""Create or check the fixed KYOw-level CV manifest; no spectra or GPU are used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import (
    create_manifest_bundle,
    fold_summary,
    load_manifest_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("action", choices=("create", "check"))
    parser.add_argument("--experiment-id", default="preflight_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Only a single directory name is accepted so outputs cannot alias source data.
    if not args.experiment_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.experiment_id
    ):
        raise ValueError("experiment-id must contain only ASCII letters, digits, '_' and '-'")
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs/experiments" / args.experiment_id
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    operation = create_manifest_bundle if args.action == "create" else load_manifest_bundle
    plan = operation(output, inventory, args.processed_dir, args.metadata)
    print(json.dumps({
        "status": "created" if args.action == "create" else "validated_existing_manifest",
        "output_dir": str(output), "sample_count": len(plan.folds),
        "fold_summary": fold_summary(plan),
        "scope": "KYOw split and shared train coordinates; no model training or evaluation",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
