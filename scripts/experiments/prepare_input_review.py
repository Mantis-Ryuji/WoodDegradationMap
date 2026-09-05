"""Prepare a local quality-table and existing-image index without regenerating preprocessing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from wood_degradation_map.experiments.config import ADOPTED_SAMPLE_IDS
from wood_degradation_map.experiments.input_review import prepare_input_review
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/production_v1")
    parser.add_argument("--figures-dir", type=Path,
                        default=root / "outputs/preprocessing/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="New review directory; defaults to an experiment results/input_review UTC directory")
    args = parser.parse_args()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    if {sample.sample_id for sample in inventory.samples} != set(ADOPTED_SAMPLE_IDS):
        raise ValueError("Review input differs from the fixed 49 adopted sample IDs")
    output = args.output_dir or (root / "outputs/experiments/preflight_v1/results/input_review"
                                 / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    page = prepare_input_review(args.processed_dir, args.figures_dir, output, inventory)
    print(json.dumps({"html": str(page), **_read_json(page.parent / "review.json")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
