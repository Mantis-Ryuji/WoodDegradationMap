"""Train M11-75 once, compare three fixed KMeans seeds, and export the appendix figure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from wood_degradation_map.experiments.global_clustering import source_record
from wood_degradation_map.experiments.global_manifest import GlobalData, load_global_bundle
from wood_degradation_map.experiments.global_mask_rate_pipeline import (
    check_evaluation, cluster, evaluate, evaluation_path,
)
from wood_degradation_map.experiments.global_mask_rate_reporting import (
    check_report, report_path, write_report,
)
from wood_degradation_map.experiments.global_mask_rate_training import (
    MaskRateTrainer, training_paths,
)
from wood_degradation_map.experiments.global_seed_plan import global_kmeans_plan
from wood_degradation_map.experiments.global_training import GlobalTrainingData, check_global_training
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "training-check", "cluster", "evaluate",
                                           "report", "check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--resume", action="store_true",
                        help="Resume M11-75 training; verify and skip completed later outputs")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or args.dpi < 1 or args.device < 0:
        parser.error("Require positive chunk size/DPI and a nonnegative device index")
    if args.resume and args.action in ("training-check", "check"):
        parser.error("Check actions do not take --resume")
    return args


def _fit_recorded(trainer: MaskRateTrainer) -> None:
    started = time.perf_counter()
    status = "failed_or_interrupted"
    try:
        trainer.fit()
        status = "completed"
    finally:
        error_type = sys.exc_info()[0]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        try:
            _write_json(trainer.results_dir / f"attempt_{stamp}.json", {
                "status": status, "error_type": error_type.__name__ if error_type else None,
                "wall_seconds": time.perf_counter() - started,
                "resume_from": str(trainer.cfg.resume_from) if trainer.cfg.resume_from else None,
                "completed_epochs_in_memory": trainer.completed_epochs,
                "checkpoint": str(trainer.ckpt_dir / "last.pt"),
                "resume_rule": "Use the saved checkpoint; an interrupted epoch is replayed",
            })
        except OSError as logging_error:
            if error_type is None:
                raise
            print(f"Could not save failed-attempt record: {logging_error}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    reference = source_record(experiment, data, "M11")
    device = torch.device("cpu")
    if args.action in ("train", "training-check", "cluster", "evaluate"):
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError("A valid single CUDA device is required; no CPU fallback")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    print(json.dumps({"condition": "M11-75", "neural_fits": 1,
                      "KMeans_seed_plan": global_kmeans_plan(),
                      "fit_pixels": data.train_pixel_count,
                      "publication": "appendix supplementary comparison"}, indent=2), flush=True)
    if args.action in ("train", "training-check"):
        results, weights = training_paths(experiment)
        existing = results.exists() or weights.exists()
        resume_from = (weights / "checkpoints/last.pt"
                       if args.action == "training-check" or (args.resume and existing) else None)
        train = GlobalTrainingData.from_data(data)
        trainer = MaskRateTrainer(train, experiment, reference, device=device,
                                  resume_from=resume_from)
        if args.action == "train" and not (args.resume and (results / "completion.json").exists()):
            _fit_recorded(trainer)
        report = check_global_training(trainer)
    elif args.action == "cluster":
        cluster(experiment, data, inventory, device=device,
                chunk_pixels=args.chunk_pixels, resume=args.resume)
        report = {"status": "three_M11_75_KMeans_fits_verified"}
    elif args.action == "evaluate":
        if not (args.resume and evaluation_path(experiment).exists()):
            evaluate(experiment, data, inventory, device=device, chunk_pixels=args.chunk_pixels)
        report = check_evaluation(experiment, data, inventory)
    elif args.action == "report":
        if not (args.resume and report_path(experiment).exists()):
            write_report(experiment, data, inventory, dpi=args.dpi)
        report = check_report(experiment, data, inventory)
    else:
        report = check_report(experiment, data, inventory)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
