"""Fit/check global K=8 Cosine-KMeans and all-pixel maps for the six fixed conditions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.global_clustering import (
    check_global_clustering, condition_paths, run_global_clustering,
)
from wood_degradation_map.experiments.global_manifest import (
    GLOBAL_CONDITIONS, GlobalData, load_global_bundle,
)
from wood_degradation_map.experiments.input_validation import load_input_inventory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--conditions", nargs="+", choices=GLOBAL_CONDITIONS,
                        default=list(GLOBAL_CONDITIONS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--resume", action="store_true",
                        help="Verify/skip completed conditions, then run unstarted conditions")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or len(set(args.conditions)) != len(args.conditions):
        parser.error("Require a positive chunk size and unique conditions")
    if args.resume and args.action != "run":
        parser.error("--resume is for run only")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    device = torch.device("cpu")
    if args.action == "run":
        if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
            raise RuntimeError("A valid single CUDA device is required; no CPU fallback")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
    for condition in args.conditions:
        results, checkpoints = condition_paths(experiment, condition)
        if args.action == "check" or (args.resume and (results.exists() or checkpoints.exists())):
            report = check_global_clustering(experiment, data, inventory, condition)
            if args.action == "run":
                print(f"{condition}: verified complete; skipped", flush=True)
        else:
            run_global_clustering(experiment, data, inventory, condition, device=device,
                                  chunk_pixels=args.chunk_pixels)
            report = check_global_clustering(experiment, data, inventory, condition)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
