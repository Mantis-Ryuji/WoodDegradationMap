"""Check saved preflight sources on 1025 synthetic pixels using CUDA; no CV metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.config import CONDITIONS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.evaluation_smoke import run_evaluation_smoke
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, load_manifest_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=[c.condition_id for c in CONDITIONS], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--clustering-smoke-id", required=True, help="Explicit completed clustering smoke ID")
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/preflight_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise RuntimeError("A valid CUDA device is required; no automatic CPU fallback")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_manifest_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = FoldData(inventory, manifest, args.fold)
    results = run_evaluation_smoke(experiment, data, args.condition, args.repeat,
                                   clustering_smoke_id=args.clustering_smoke_id, device=device)
    print(json.dumps({"output_dir": str(results), **_read_json(results / "completion.json")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
