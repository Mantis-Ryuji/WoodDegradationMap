"""Run one fixed ChemoMAE training job, or a separate short CUDA/resume probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wood_degradation_map.experiments.config import CONDITIONS
from wood_degradation_map.experiments.data import FoldData
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, _write_json, load_manifest_bundle
from wood_degradation_map.experiments.neural import extract_full_visible
from wood_degradation_map.experiments.training import ExperimentTrainer, TrainingData


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "smoke"))
    parser.add_argument("--condition", required=True,
                        choices=[c.condition_id for c in CONDITIONS if c.n_mask is not None])
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--resume", type=Path, help="Explicit same-run checkpoint; train only")
    parser.add_argument("--smoke-batches", type=int, choices=range(2, 17),
                        help="Batches in each of two truncated smoke epochs; default 2")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/preflight_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    args = parser.parse_args()
    if args.action == "smoke" and args.resume is not None:
        parser.error("smoke performs its own resume probe; --resume is for train")
    if args.action == "train" and args.smoke_batches is not None:
        parser.error("--smoke-batches cannot alter a production run")
    return args


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _fit_recorded(trainer: ExperimentTrainer) -> None:
    started = time.perf_counter()
    status = "failed_or_interrupted"
    try:
        trainer.fit()
        status = "completed"
    finally:
        error_type = sys.exc_info()[0]
        attempt = {
            "status": status, "error_type": error_type.__name__ if error_type else None,
            "wall_seconds": time.perf_counter() - started,
            "resume_from": str(trainer.cfg.resume_from) if trainer.cfg.resume_from else None,
            "in_memory_completed_epochs": trainer.completed_epochs,
            "in_memory_attempted_updates": trainer.attempted_updates,
            "checkpoint": str(trainer.ckpt_dir / "last.pt"),
            "resume_rule": "Use the saved checkpoint; an interrupted epoch is replayed",
            "runtime": trainer.run_record["execution"]["runtime"],
            "code_sha256": trainer.run_record["execution"]["code_sha256"],
        }
        try:
            _write_json(trainer.results_dir / f"attempt_{_stamp()}.json", attempt)
        except OSError as logging_error:
            # A full disk must not hide the original training/save exception.
            if error_type is None:
                raise
            print(f"Could not persist the failed-attempt record: {logging_error}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        raise RuntimeError("A valid CUDA device is required; there is no CPU/batch-size fallback")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    experiment = args.experiment_dir.resolve()
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_manifest_bundle(experiment, inventory, args.processed_dir, args.metadata)
    fold_data = FoldData(inventory, manifest, args.fold)
    print(f"Loading shared train selection: {fold_data.train_pixel_count} FP32 spectra on CPU.",
          flush=True)
    train = TrainingData.from_fold(fold_data)
    smoke_batches = (args.smoke_batches or 2) if args.action == "smoke" else None
    smoke_id = _stamp() if args.action == "smoke" else None
    trainer = ExperimentTrainer(
        train, experiment, args.condition, args.repeat, device=device,
        resume_from=args.resume, smoke_batches=smoke_batches, smoke_id=smoke_id,
    )
    print(f"Results: {trainer.results_dir}", flush=True)
    _fit_recorded(trainer)
    if args.action == "train":
        print(json.dumps(_read_json(trainer.results_dir / "completion.json"), indent=2))
        return 0

    # Retain the original second-epoch outputs, then repeat only that epoch
    # after reconstructing the Trainer from the saved first-epoch checkpoint.
    reference_trace = trainer.trace[smoke_batches:]
    reference_weights = {key: value.detach().cpu().clone()
                         for key, value in trainer.model.state_dict().items()}
    saved_weights = torch.load(trainer.out_dir / "smoke_model.pt",
                               map_location="cpu", weights_only=True)
    if not all(torch.equal(value, saved_weights[key]) for key, value in reference_weights.items()):
        raise ValueError("Raw weights changed during save/load")
    del saved_weights
    probe = train.spectra[:8].to(device)
    expected = extract_full_visible(trainer.model, probe)
    resume_path = trainer.ckpt_dir / "epoch_1.pt"
    reference_completion = _read_json(trainer.results_dir / "completion.json")
    del trainer
    restored = ExperimentTrainer(
        train, experiment, args.condition, args.repeat, device=device,
        resume_from=resume_path, smoke_batches=smoke_batches, smoke_id=smoke_id,
    )
    _fit_recorded(restored)
    comparison_fields = ("epoch", "batch", "lr", "optimizer_step", "amp_scale",
                         "train_rows_sha256", "augmented_sha256", "visible_mask_sha256")
    exact = len(reference_trace) == len(restored.trace) and all(
        all(left[key] == right[key] for key in comparison_fields)
        for left, right in zip(reference_trace, restored.trace, strict=True)
    )
    weight_error = max(float((value - restored.model.state_dict()[key].detach().cpu()).abs().max())
                       for key, value in reference_weights.items())
    actual = extract_full_visible(restored.model, probe)
    latent_error = float(np.max(np.abs(expected.values - actual.values)))
    report = {
        "status": "smoke_and_resume_probe_completed", "condition": args.condition,
        "scope": "Two truncated train-only epochs and replay of epoch 2; no CV metrics",
        "smoke_id": smoke_id, "batch_size": restored.recipe["batch_size"],
        "batches_per_smoke_epoch": smoke_batches,
        "executed_batches_including_replay": 3 * smoke_batches,
        "reference_completion": reference_completion,
        "resume_inputs_lr_scaler_and_step_decisions_exact": exact,
        "raw_weights_save_load_exact": True,
        "resume_weight_absolute_error_max": weight_error,
        "resume_latent_absolute_error_max": latent_error,
        "reference_epoch_2": reference_trace, "resumed_epoch_2": restored.trace,
        "full_visible_probe": asdict(actual.diagnostics),
        "wall_seconds_including_io_and_replay": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    # This is an engineering roundtrip tolerance, never a pixel exclusion rule.
    passed = (exact and weight_error <= 1e-6 and latent_error <= 1e-6
              and restored.nonzero_lr_updates > 0)
    report["checks_passed"] = passed
    _write_json(restored.results_dir / "smoke.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
