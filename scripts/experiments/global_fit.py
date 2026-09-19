"""Prepare global B0/PCA inputs and sequentially train A0, M00 and M11 once."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wood_degradation_map.experiments.global_baselines import global_baselines
from wood_degradation_map.experiments.global_manifest import (
    NEURAL_CONDITIONS,
    GlobalData,
    create_global_bundle,
    global_summary,
    load_global_bundle,
)
from wood_degradation_map.experiments.global_training import (
    GlobalTrainer,
    GlobalTrainingData,
    check_global_training,
)
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _write_json
from wood_degradation_map.experiments.neural import extract_full_visible


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "check", "baseline-fit", "baseline-check",
                                           "smoke", "train", "training-check"))
    parser.add_argument("--experiment-dir", type=Path, default=root / "outputs/experiments/global_v1")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--conditions", nargs="+", choices=NEURAL_CONDITIONS,
                        default=None, help="Default: A0 M00 M11, sequentially")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Train: resume last.pt; verify and skip completed runs")
    parser.add_argument("--smoke-batches", type=int, choices=range(2, 17), default=None)
    args = parser.parse_args(argv)
    if args.resume and args.action != "train":
        parser.error("--resume is for train only")
    if args.smoke_batches is not None and args.action != "smoke":
        parser.error("--smoke-batches is for smoke only")
    if args.conditions is not None and args.action not in ("train", "smoke", "training-check"):
        parser.error("--conditions is for neural operations only")
    args.conditions = args.conditions or list(NEURAL_CONDITIONS)
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions cannot contain duplicates")
    return args


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _fit_recorded(trainer: GlobalTrainer) -> None:
    started = time.perf_counter()
    status = "failed_or_interrupted"
    try:
        trainer.fit()
        status = "completed"
    finally:
        error_type = sys.exc_info()[0]
        record = {
            "status": status, "error_type": error_type.__name__ if error_type else None,
            "wall_seconds": time.perf_counter() - started,
            "resume_from": str(trainer.cfg.resume_from) if trainer.cfg.resume_from else None,
            "in_memory_completed_epochs": trainer.completed_epochs,
            "in_memory_attempted_updates": trainer.attempted_updates,
            "checkpoint": str(trainer.ckpt_dir / "last.pt"),
            "resume_rule": "Use the saved checkpoint; an interrupted epoch is replayed",
            "execution": trainer.run_record["execution"],
        }
        try:
            _write_json(trainer.results_dir / f"attempt_{_stamp()}.json", record)
        except OSError as logging_error:
            if error_type is None:
                raise
            print(f"Could not save failed-attempt record: {logging_error}", file=sys.stderr)


def smoke_run(
    train: GlobalTrainingData, experiment: Path, condition: str, device: torch.device,
    *, batches: int,
) -> dict[str, object]:
    smoke_id = _stamp()
    trainer = GlobalTrainer(train, experiment, condition, device=device,
                            smoke_batches=batches, smoke_id=smoke_id)
    _fit_recorded(trainer)
    trace = trainer.trace[batches:]
    reference = {key: value.detach().cpu().clone()
                 for key, value in trainer.model.state_dict().items()}
    saved = torch.load(trainer.out_dir / "smoke_model.pt", map_location="cpu", weights_only=True)
    if saved.keys() != reference.keys() or not all(
        torch.equal(saved[key], value) for key, value in reference.items()
    ):
        raise ValueError("Global smoke raw-weight roundtrip failed")
    del saved
    probe = train.spectra[:8].to(device)
    expected = extract_full_visible(trainer.model, probe).values
    resume = trainer.ckpt_dir / "epoch_1.pt"
    del trainer
    gc.collect()
    restored = GlobalTrainer(train, experiment, condition, device=device, resume_from=resume,
                             smoke_batches=batches, smoke_id=smoke_id)
    _fit_recorded(restored)
    keys = ("epoch", "batch", "lr", "optimizer_step", "amp_scale", "train_rows_sha256",
            "augmented_sha256", "visible_mask_sha256")
    exact = len(trace) == len(restored.trace) and all(
        all(left[key] == right[key] for key in keys)
        for left, right in zip(trace, restored.trace, strict=True)
    )
    weight_error = max(float((value - restored.model.state_dict()[key].detach().cpu()).abs().max())
                       for key, value in reference.items())
    latent_error = float(np.abs(expected - extract_full_visible(restored.model, probe).values).max())
    passed = exact and weight_error <= 1e-6 and latent_error <= 1e-6 and restored.nonzero_lr_updates > 0
    report = {
        "status": "smoke_and_resume_probe_completed", "scope": "global_fit",
        "condition": condition, "smoke_id": smoke_id, "checks_passed": passed,
        "batch_size": restored.recipe["batch_size"], "batches_per_smoke_epoch": batches,
        "executed_batches_including_replay": 3 * batches,
        "raw_weights_save_load_exact": True, "resume_inputs_lr_scaler_and_steps_exact": exact,
        "resume_weight_absolute_error_max": weight_error,
        "resume_latent_absolute_error_max": latent_error,
        "reference_epoch_2": trace, "resumed_epoch_2": restored.trace,
    }
    _write_json(restored.results_dir / "smoke.json", report)
    if not passed:
        raise ValueError(f"Global smoke/resume validation failed: {restored.results_dir}")
    return report


def train_runs(
    train: GlobalTrainingData, experiment: Path, conditions: list[str], device: torch.device,
    *, resume: bool, check_only: bool = False,
) -> None:
    for condition in conditions:
        results = experiment / f"results/neural/{condition}/repeat_1"
        weights = experiment / f"checkpoints/neural/{condition}/repeat_1"
        checkpoint = weights / "checkpoints/last.pt"
        existing = results.exists() or weights.exists()
        if check_only or (resume and existing):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"No resumable global checkpoint: {checkpoint}")
            resume_from = checkpoint
        else:
            resume_from = None
        print(f"Global {condition}: {'check' if check_only else 'train'}", flush=True)
        trainer = GlobalTrainer(train, experiment, condition, device=device, resume_from=resume_from)
        if check_only or (resume and (results / "completion.json").exists()):
            report = check_global_training(trainer)
            print(f"Global {condition}: completed run verified; skipped.", flush=True)
        else:
            _fit_recorded(trainer)
            report = check_global_training(trainer)
        print(json.dumps(report, indent=2), flush=True)
        del trainer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    operation = create_global_bundle if args.action == "create" else load_global_bundle
    manifest = operation(experiment, inventory, args.processed_dir, args.metadata)
    if args.action in ("create", "check"):
        print(json.dumps({"status": args.action, "scope": "global_fit",
                          "output": str(experiment), **global_summary(manifest)}, indent=2))
        return 0
    data = GlobalData(inventory, manifest)
    if args.action in ("baseline-fit", "baseline-check"):
        print(json.dumps(global_baselines(experiment, data, fit=args.action == "baseline-fit"),
                         indent=2, allow_nan=False))
        return 0
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise RuntimeError("A valid single CUDA device is required; no CPU/batch-size fallback")
    # Validate PCA/B0 before launching the long neural jobs.
    global_baselines(experiment, data, fit=False)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    print(f"Loading {data.train_pixel_count} shared global FP32 spectra on CPU.", flush=True)
    train = GlobalTrainingData.from_data(data)
    if args.action == "smoke":
        for condition in args.conditions:
            report = smoke_run(train, experiment, condition, device, batches=args.smoke_batches or 2)
            print(json.dumps(report, indent=2), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        train_runs(train, experiment, args.conditions, device, resume=args.resume,
                   check_only=args.action == "training-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
