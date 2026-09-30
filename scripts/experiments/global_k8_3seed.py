"""Fit three global K=8 KMeans seeds, evaluate them, and aggregate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.global_k8_aggregate import check_report, write_report
from wood_degradation_map.experiments.global_k8_evaluation import (
    check_condition_evaluation, condition_output, evaluate_condition,
)
from wood_degradation_map.experiments.global_k8_repeats import (
    check_repeat_clustering, repeat_paths, run_repeat_clustering,
)
from wood_degradation_map.experiments.global_manifest import (
    GLOBAL_CONDITIONS, GlobalData, load_global_bundle,
)
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_plan
from wood_degradation_map.experiments.input_validation import load_input_inventory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cluster", "evaluate", "aggregate", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--conditions", nargs="+", choices=GLOBAL_CONDITIONS,
                        default=list(GLOBAL_CONDITIONS))
    parser.add_argument("--repeats", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--resume", action="store_true",
                        help="Verify and skip completed cluster/evaluation outputs")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or len(set(args.conditions)) != len(args.conditions):
        parser.error("Require a positive chunk size and unique conditions")
    if len(set(args.repeats)) != len(args.repeats):
        parser.error("Repeated repeat IDs")
    if args.resume and args.action not in ("cluster", "evaluate"):
        parser.error("--resume is for cluster or evaluate only")
    if args.action in ("aggregate", "check") and args.conditions != list(GLOBAL_CONDITIONS):
        parser.error("aggregate/check require all six conditions")
    if args.action != "cluster" and args.repeats != [1, 2, 3]:
        parser.error("--repeats is for cluster only")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    plan = global_kmeans_plan()
    print(json.dumps({"K": 8, "KMeans_seeds": plan,
                      "representation": "saved global fit, fixed across repeats"}, indent=2), flush=True)
    device = torch.device("cpu")
    if args.action in ("cluster", "evaluate"):
        if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
            raise RuntimeError("A valid CUDA device is required for global feature extraction")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    if args.action == "cluster":
        for condition in args.conditions:
            for repeat in args.repeats:
                results, checkpoints = repeat_paths(experiment, condition, repeat)
                if args.resume and (results.exists() or checkpoints.exists()):
                    report = check_repeat_clustering(experiment, data, inventory, condition, repeat)
                    print(f"{condition} repeat {repeat}: verified complete; skipped", flush=True)
                else:
                    run_repeat_clustering(experiment, data, inventory, condition, repeat,
                                          device=device, chunk_pixels=args.chunk_pixels)
                    report = check_repeat_clustering(experiment, data, inventory, condition, repeat)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    elif args.action == "evaluate":
        for condition in args.conditions:
            output = condition_output(experiment, condition)
            if args.resume and output.exists():
                report = check_condition_evaluation(experiment, data, inventory, condition)
                print(f"{condition}: verified complete; skipped", flush=True)
            else:
                evaluate_condition(experiment, data, inventory, condition,
                                   device=device, chunk_pixels=args.chunk_pixels)
                report = check_condition_evaluation(experiment, data, inventory, condition)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    elif args.action == "aggregate":
        report = write_report(experiment, data, inventory)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps(check_report(experiment, data, inventory),
                         ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps(check_report(experiment, data, inventory),
                         ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
