"""Fit all planned K and save clean test maps, check artifacts, or run a train-only CUDA probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.cluster_pipeline import check_clustering, run_clustering
from wood_degradation_map.experiments.config import CONDITIONS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, load_manifest_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "check", "smoke"))
    parser.add_argument("--condition", choices=[c.condition_id for c in CONDITIONS], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--chunk-pixels", type=int, default=1024,
                        help="Extraction/I/O chunk size; does not change the training batch size")
    parser.add_argument("--neural-smoke-id", help="Explicit training smoke ID; smoke action only")
    parser.add_argument("--pca-repeat", type=int, choices=range(1, 4),
                        help="Explicit saved PCA source repeat for B1; no automatic fallback")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/preflight_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    args = parser.parse_args()
    if args.chunk_pixels <= 0:
        parser.error("--chunk-pixels must be positive")
    if args.neural_smoke_id is not None and (
            args.action != "smoke" or args.condition in ("B0", "B1")):
        parser.error("--neural-smoke-id requires a neural condition and smoke action")
    if args.pca_repeat is not None and (args.condition != "B1" or args.action == "check"):
        parser.error("--pca-repeat requires B1 run/smoke; check uses the saved source")
    return args


def main() -> int:
    args = parse_args()
    if args.action == "check":
        device = torch.device("cpu")
    else:
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
        report = check_clustering(experiment, data, inventory, args.condition, args.repeat,
                                  device=device)
    else:
        results = run_clustering(
            experiment, data, inventory, args.condition, args.repeat, device=device,
            smoke=args.action == "smoke", neural_smoke_id=args.neural_smoke_id,
            pca_repeat=args.pca_repeat, chunk_pixels=args.chunk_pixels,
        )
        report = {"output_dir": str(results), **_read_json(results / "completion.json")}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
