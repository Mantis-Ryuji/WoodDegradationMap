"""Run M11 five-fold HPO and shared M00/M11 Table 5 final evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wood_degradation_map.experiments.bacteria_data import PreparedBacteria
from wood_degradation_map.experiments.bacteria_hpo import (
    evaluate_selected, render_table5, search, select_completed, train_selected,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("search", "select", "train", "evaluate", "tables"))
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/bacteria_id_v1")
    parser.add_argument("--pretrain-output-dir", type=Path,
                        default=root / "outputs/experiments/bacteria_id_table5_reference30_pretrain_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/experiments/bacteria_id_table5_hpo_cv_v1")
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument(
        "--completed-trials", type=int,
        help="For select: rank only the first N unique completed settings (30-100)",
    )
    args = parser.parse_args()
    if args.action == "select":
        if args.completed_trials is None or not 30 <= args.completed_trials <= 100:
            parser.error("select requires --completed-trials between 30 and 100")
    elif args.completed_trials is not None:
        parser.error("--completed-trials is only valid with select")
    return args


def _device(index: int) -> torch.device:
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise RuntimeError("A valid single CUDA device is required")
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    return device


def main() -> int:
    args = parse_args()
    data = PreparedBacteria(args.processed_dir)
    if args.action == "select":
        select_completed(
            data, args.pretrain_output_dir, args.output_dir,
            completed_trials=args.completed_trials,
        )
        return 0
    if args.action == "tables":
        print("\n".join(str(path) for path in render_table5(
            data, args.pretrain_output_dir, args.output_dir,
        )))
        return 0
    device = _device(args.device)
    if args.action == "search":
        search(data, args.pretrain_output_dir, args.output_dir, device)
    elif args.action == "train":
        train_selected(data, args.pretrain_output_dir, args.output_dir, device)
    else:
        evaluate_selected(data, args.pretrain_output_dir, args.output_dir, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
