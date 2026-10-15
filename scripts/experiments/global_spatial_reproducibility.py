"""Run the isolated M11 encoder x KMeans spatial reproducibility supplement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from wood_degradation_map.experiments.global_manifest import GlobalData, load_global_bundle
from wood_degradation_map.experiments.global_training import GlobalTrainingData, check_global_training
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, _require, _write_json
from wood_degradation_map.experiments.spatial_reproducibility_pipeline import (
    check_evaluation, check_prepared, check_report, cluster, evaluate, prepare, separate_roots,
    write_report,
)
from wood_degradation_map.experiments.spatial_reproducibility_training import (
    SpatialTrainer, training_paths,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "training-check", "cluster",
                                           "evaluate", "report", "check"))
    parser.add_argument("--source-experiment-dir", type=Path,
                        default=root / "outputs/experiments/global_v1")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/global_spatial_reproducibility_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--encoders", type=int, nargs="+", choices=(2, 3), default=None,
                        help="Added encoder IDs for train, training-check or cluster (default: 2 3)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--include-overlap", action="store_true",
                        help="Also save 8x8 overlap fractions and VI components during evaluate")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted training epoch, or verify and skip sealed outputs")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or args.dpi < 1 or args.device < 0:
        parser.error("Require positive chunk size/DPI and a nonnegative device index")
    if args.encoders is not None and args.action not in ("train", "training-check", "cluster"):
        parser.error("--encoders applies only to train, training-check or cluster")
    if args.encoders is not None and len(set(args.encoders)) != len(args.encoders):
        parser.error("Do not repeat encoder IDs")
    if args.include_overlap and args.action != "evaluate":
        parser.error("--include-overlap applies only to evaluate")
    if args.resume and args.action in ("training-check", "check"):
        parser.error("Check actions do not take --resume")
    args.encoders = args.encoders or [2, 3]
    return args


def _fit(trainer: SpatialTrainer) -> dict[str, object]:
    trainer.fit()
    return check_global_training(trainer)


def _run(args: argparse.Namespace) -> dict[str, object]:
    experiment, source = separate_roots(args.experiment_dir, args.source_experiment_dir)
    # Refuse a custom output that could contain or be inside the observed input tree.
    separate_roots(experiment, args.processed_dir)
    _require(not args.metadata.resolve().is_relative_to(experiment),
             "The metadata source must not be inside the output experiment")
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(source, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    if args.action == "prepare":
        if not (args.resume and (experiment / "protocol").exists()):
            prepare(experiment, source, data, inventory)
        return check_prepared(experiment, source, data, inventory)
    check_prepared(experiment, source, data, inventory)
    device = torch.device("cpu")
    if args.action in ("train", "training-check", "cluster"):
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError("A valid single CUDA device is required; no CPU fallback")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    if args.action in ("train", "training-check"):
        train = GlobalTrainingData.from_data(data)
        reports = []
        for encoder in args.encoders:
            result, weights = training_paths(experiment, encoder)
            existing = result.exists() or weights.exists()
            resume_from = (weights / "checkpoints/last.pt"
                           if args.action == "training-check" or (args.resume and existing) else None)
            trainer = SpatialTrainer(train, experiment, source, data, encoder,
                                     device=device, resume_from=resume_from)
            if args.action == "train" and not (args.resume and (result / "completion.json").exists()):
                report = _fit(trainer)
            else:
                report = check_global_training(trainer)
            reports.append({"encoder_repeat": encoder, "check": report})
            del trainer
        return {"status": "spatial_training_verified", "encoders": reports}
    if args.action == "cluster":
        for encoder in args.encoders:
            cluster(experiment, source, data, inventory, encoder, device=device,
                    chunk_pixels=args.chunk_pixels, resume=args.resume)
        return {"status": "spatial_clustering_verified", "encoders": args.encoders}
    if args.action == "evaluate":
        output = experiment / "results/evaluation"
        if args.resume and output.exists():
            _require(_read_json(output / "run.json")["contract"]["include_overlap"]
                     == args.include_overlap, "--include-overlap differs from saved evaluation")
        else:
            evaluate(experiment, source, data, inventory, include_overlap=args.include_overlap)
        return check_evaluation(experiment, source, data, inventory)
    if args.action == "report":
        output = experiment / "results/report"
        if args.resume and output.exists():
            _require(_read_json(output / "run.json")["contract"]["dpi"] == args.dpi,
                     "--dpi differs from saved report")
        else:
            write_report(experiment, source, data, inventory, dpi=args.dpi)
        return check_report(experiment, source, data, inventory)
    return check_report(experiment, source, data, inventory)


def main() -> int:
    args = parse_args()
    # Validate before even the failure logger can create anything in the source tree.
    experiment, _ = separate_roots(args.experiment_dir, args.source_experiment_dir)
    separate_roots(experiment, args.processed_dir)
    _require(not args.metadata.resolve().is_relative_to(experiment),
             "The metadata source must not be inside the output experiment")
    started, status = time.perf_counter(), "failed_or_interrupted"
    try:
        report = _run(args)
        status = "completed"
    finally:
        error_type = sys.exc_info()[0]
        # Checks are read-only; stage attempts are retained even when their temporary
        # outputs were removed. They are outside sealed scientific artifacts.
        if args.action not in ("check", "training-check"):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            try:
                (experiment / "attempts").mkdir(parents=True, exist_ok=True)
                _write_json(experiment / "attempts" / f"{args.action}_{stamp}.json", {
                    "action": args.action, "status": status,
                    "error_type": error_type.__name__ if error_type else None,
                    "wall_seconds": time.perf_counter() - started,
                    "encoders": args.encoders, "resume": args.resume,
                    "include_overlap": args.include_overlap, "device_index": args.device,
                    "resume_rule": "Training replays an interrupted epoch from last.pt; "
                                   "other stages verify completed outputs and retry incomplete work",
                })
            except OSError as logging_error:
                if error_type is None:
                    raise
                print(f"Could not save attempt record: {logging_error}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
