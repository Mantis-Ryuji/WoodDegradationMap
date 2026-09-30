"""Prepare Bacteria-ID and run one fixed M00/M11 job or render SMAE tables."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from wood_degradation_map.experiments.bacteria_config import CONDITIONS, CORPORA, RUN_SEEDS
from wood_degradation_map.experiments.bacteria_data import PreparedBacteria, prepare_inputs
from wood_degradation_map.experiments.bacteria_evaluation import evaluate_table4, render_tables
from wood_degradation_map.experiments.bacteria_training import (
    BacteriaFinetuner, BacteriaPretrainer, BacteriaTrainingData,
)
from wood_degradation_map.experiments.bacteria_variant_training import (
    ALL_VARIANTS, BacteriaVariantFinetuner,
)
from wood_degradation_map.experiments.bacteria_variant_reporting import render_integrated_tables


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pretrain", "cluster", "finetune", "tables", "tables-all"))
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/bacteria_id")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/bacteria_id_v1")
    parser.add_argument("--output-dir", type=Path,
                        help="Required for historical runs; select an explicit output directory")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--seed", type=int, choices=RUN_SEEDS)
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--resume", type=Path, help="Explicit same-run last.pt; train actions only")
    parser.add_argument("--variant", choices=ALL_VARIANTS, default="cls_unfrozen",
                        help="Table 5 classifier path; default preserves the original CLS run")
    parser.add_argument("--chunk-rows", type=int, default=1024, help="SNV preparation chunk size")
    args = parser.parse_args()
    if args.action != "prepare" and args.output_dir is None:
        parser.error("--output-dir is required for historical Bacteria-ID runs")
    if args.action != "finetune" and args.variant != "cls_unfrozen":
        parser.error("--variant applies only to finetune")
    if args.action in ("pretrain", "cluster", "finetune"):
        if args.condition is None or args.seed is None:
            parser.error("--condition and --seed are required for run actions")
        if args.action != "finetune" and args.corpus is None:
            parser.error("--corpus is required for pretrain/cluster")
        if args.action == "finetune" and args.corpus is not None:
            parser.error("finetune always uses reference30; omit --corpus")
        if args.action == "cluster" and args.corpus == "reference30":
            parser.error("Table 4 clustering is only for bacteria4/bacteria6")
        if args.action == "cluster" and args.resume is not None:
            parser.error("cluster has no training checkpoint to resume")
    elif args.condition is not None or args.corpus is not None or args.seed is not None or args.resume is not None:
        parser.error("prepare/tables/tables-all do not take run selectors or --resume")
    if args.action != "prepare" and args.chunk_rows != 1024:
        parser.error("--chunk-rows is only for prepare")
    return args


def _device(index: int) -> torch.device:
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise RuntimeError("A valid single CUDA device is required")
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    return device


def _json_replace_denied(
    exc: PermissionError, results_dir: Path, filenames: tuple[str, ...],
) -> bool:
    """Recognize a Windows lock on one of this run's atomic JSON outputs."""
    if getattr(exc, "winerror", None) != 5 or exc.filename is None or exc.filename2 is None:
        return False
    source, target = Path(exc.filename), Path(exc.filename2)
    return (target.parent == results_dir and target.name in filenames
            and source == target.with_name(target.name + ".tmp"))


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        record = prepare_inputs(args.raw_dir, args.processed_dir, chunk_rows=args.chunk_rows)
        print(json.dumps({"prepared": str(args.processed_dir),
                          "split_sizes": record["split_sizes"],
                          "wavenumbers": record["wavenumbers"]}, indent=2))
        return 0
    data = PreparedBacteria(args.processed_dir)
    if args.action == "tables":
        paths = render_tables(data, args.output_dir)
        print("\n".join(str(path) for path in paths))
        return 0
    if args.action == "tables-all":
        paths = render_integrated_tables(data, args.output_dir)
        print("\n".join(str(path) for path in paths))
        return 0
    device = _device(args.device)
    if args.action == "pretrain":
        train = BacteriaTrainingData.from_prepared(data, args.corpus)
        resume_from = args.resume
        for attempt in range(5):
            trainer = BacteriaPretrainer(
                train, data, args.output_dir, args.condition, args.seed,
                device=device, resume_from=resume_from,
            )
            try:
                trainer.fit()
                break
            except PermissionError as exc:
                # Resume from last.pt; a newer .tmp JSON is never the training state.
                if (not _json_replace_denied(
                        exc, trainer.results_dir,
                        ("training_history.json", "checkpoint.json")) or attempt == 4):
                    raise
                checkpoint = trainer.ckpt_dir / "last.pt"
                if not checkpoint.is_file():
                    raise
                delay = min(0.5 * 2**attempt, 4.0)
                print(f"JSON replace was denied; retrying from {checkpoint} "
                      f"after {delay:g}s (attempt {attempt + 2}/5)", file=sys.stderr)
                time.sleep(delay)
                resume_from = checkpoint
        print((trainer.results_dir / "completion.json").read_text(encoding="utf-8"))
    elif args.action == "cluster":
        report = evaluate_table4(
            data, args.output_dir, args.corpus, args.condition, args.seed, device=device,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        resume_from = args.resume
        for attempt in range(5):
            if args.variant == "cls_unfrozen":
                trainer = BacteriaFinetuner(
                    data, args.output_dir, args.condition, args.seed,
                    device=device, resume_from=resume_from,
                )
            else:
                trainer = BacteriaVariantFinetuner(
                    data, args.output_dir, args.condition, args.seed, args.variant,
                    device=device, resume_from=resume_from,
                )
            try:
                result = trainer.fit()
                break
            except PermissionError as exc:
                if (not _json_replace_denied(
                        exc, trainer.results_dir, ("training_history.json",)) or attempt == 4):
                    raise
                if not trainer.last_path.is_file():
                    raise
                delay = min(0.5 * 2**attempt, 4.0)
                print(f"JSON replace was denied; retrying from {trainer.last_path} "
                      f"after {delay:g}s (attempt {attempt + 2}/5)", file=sys.stderr)
                time.sleep(delay)
                resume_from = trainer.last_path
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
