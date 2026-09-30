"""Prepare, run and report the isolated 80/20 Bacteria-ID Table 4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wood_degradation_map.experiments.bacteria_config import CONDITIONS, RUN_SEEDS, TABLE4_CLASSES
from wood_degradation_map.experiments.bacteria_table4_80 import (
    Table4Data, Table4Pretrainer, discard_incomplete_score, evaluate_table4_80,
    prepare_table4_80, render_table4_80, verify_pretraining, verify_score,
)
from wood_degradation_map.experiments.bacteria_training import _run_paths


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = ROOT / "data/processed/bacteria_id_v1"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/bacteria_id_table4_80_20_v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pretrain", "cluster", "report", "all"))
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", choices=tuple(TABLE4_CLASSES))
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=RUN_SEEDS)
    parser.add_argument("--device", type=int, default=0, help="Single CUDA device index")
    parser.add_argument("--resume", type=Path, help="Same 80/20 run's last.pt; pretrain only")
    args = parser.parse_args()
    if args.action in ("pretrain", "cluster") and (
        args.corpus is None or args.condition is None or args.seed is None
    ):
        parser.error("pretrain and cluster require --corpus, --condition and --seed")
    if args.action not in ("pretrain", "cluster") and any(
        value is not None for value in (args.corpus, args.condition, args.seed)
    ):
        parser.error("run identity applies only to pretrain and cluster")
    if args.resume is not None and args.action != "pretrain":
        parser.error("--resume applies only to pretrain")
    return args


def _device(index: int) -> torch.device:
    if index < 0 or not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is unavailable")
    return torch.device("cuda", index)


def _all(data: Table4Data, output_dir: Path, device: torch.device) -> None:
    for corpus in TABLE4_CLASSES:
        for condition in CONDITIONS:
            for seed in RUN_SEEDS:
                label = f"{corpus} {condition} seed {seed}"
                pre_results, pre_weights = _run_paths(
                    output_dir.resolve(), "pretrain", corpus, condition, seed,
                )
                completion = pre_results / "completion.json"
                if completion.is_file():
                    verify_pretraining(data, output_dir, corpus, condition, seed)
                else:
                    checkpoint = pre_weights / "checkpoints/last.pt"
                    if pre_results.exists() != pre_weights.exists() or (
                        pre_results.exists() and not checkpoint.is_file()
                    ):
                        raise RuntimeError(
                            f"Incomplete run has no usable same-run checkpoint: {label}"
                        )
                    resume = checkpoint if pre_results.exists() else None
                    print(f"[Table 4 80/20] pretrain {label}", flush=True)
                    Table4Pretrainer(
                        data, output_dir, corpus, condition, seed,
                        device=device, resume_from=resume,
                    ).fit()
                    verify_pretraining(data, output_dir, corpus, condition, seed)
                score_results, score_weights = _run_paths(
                    output_dir.resolve(), "table4", corpus, condition, seed,
                )
                if (score_results / "metrics.json").is_file():
                    verify_score(data, output_dir, corpus, condition, seed)
                else:
                    if score_results.exists() or score_weights.exists():
                        if not discard_incomplete_score(score_results, score_weights):
                            raise RuntimeError(
                                f"Incomplete clustering output requires inspection: {label}"
                            )
                        print(f"[Table 4 80/20] discard incomplete score {label}", flush=True)
                    print(f"[Table 4 80/20] cluster {label}", flush=True)
                    evaluate_table4_80(data, output_dir, corpus, condition, seed, device=device)
                    verify_score(data, output_dir, corpus, condition, seed)
    report_paths = (
        output_dir / "table4.md", output_dir / "table4_seed_scores.csv",
        output_dir / "table4_statistics.json", output_dir / "table4_seed_scores.png",
    )
    if all(path.is_file() for path in report_paths):
        for path in report_paths:
            print(path)
    else:
        for path in render_table4_80(data, output_dir):
            print(path)


def main() -> int:
    args = _args()
    if args.action == "prepare":
        record = prepare_table4_80(args.processed_dir, args.output_dir)
        print(json.dumps({"protocol": record["protocol"], "split_sizes": record["split_sizes"],
                          "planned_runs": len(record["planned_runs"])}, indent=2))
        return 0
    data = Table4Data(args.processed_dir, args.output_dir)
    if args.action == "report":
        for path in render_table4_80(data, args.output_dir):
            print(path)
        return 0
    device = _device(args.device)
    if args.action == "all":
        _all(data, args.output_dir, device)
    elif args.action == "pretrain":
        trainer = Table4Pretrainer(
            data, args.output_dir, args.corpus, args.condition, args.seed,
            device=device, resume_from=args.resume,
        )
        trainer.fit()
        verify_pretraining(data, args.output_dir, args.corpus, args.condition, args.seed)
    else:
        result = evaluate_table4_80(
            data, args.output_dir, args.corpus, args.condition, args.seed, device=device,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
