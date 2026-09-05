"""Evaluate completed production maps with shared test perturbations, or verify saved metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.config import CONDITIONS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.evaluation_pipeline import check_evaluations, run_evaluation
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, load_manifest_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check"))
    parser.add_argument("--conditions", nargs="+", required=True,
                        choices=[condition.condition_id for condition in CONDITIONS])
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--repeats", type=int, nargs="+", choices=range(1, 4), default=[1])
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--chunk-pixels", type=int, default=1024,
                        help="HDF5 read width; perturbation generation remains fixed at 1024")
    parser.add_argument("--silhouette-chunk-pixels", type=int, default=1_000_000,
                        help="Bounds N x K workspace only; full-fold feature arrays remain required")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/cv_200hz_snr10_linear256_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/preprocessing/200hz_snr10_linear256")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    args = parser.parse_args()
    if args.chunk_pixels < 1 or args.silhouette_chunk_pixels < 1:
        parser.error("Chunk sizes must be positive")
    if len(set(args.conditions)) != len(args.conditions) or len(set(args.repeats)) != len(args.repeats):
        parser.error("Conditions and repeats must be unique")
    return args


def main() -> int:
    args = parse_args()
    device = torch.device("cpu")
    if args.action == "run":
        if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
            raise RuntimeError("A valid CUDA device is required; no automatic CPU fallback")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_manifest_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = FoldData(inventory, manifest, args.fold)
    if args.action == "check":
        reports = check_evaluations(experiment, data, inventory,
                                    tuple(args.conditions), tuple(args.repeats))
    else:
        paths = run_evaluation(
            experiment, data, inventory, tuple(args.conditions), tuple(args.repeats), device=device,
            chunk_pixels=args.chunk_pixels, silhouette_chunk_pixels=args.silhouette_chunk_pixels,
        )
        reports = [{"output_dir": str(path), **_read_json(path / "completion.json")} for path in paths]
    print(json.dumps({"evaluations": reports}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
