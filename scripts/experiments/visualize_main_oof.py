"""Render main OOF figures and CSV tables from saved artifacts only (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from wood_degradation_map.experiments.oof_reporting import render_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/production_v1")
    parser.add_argument("--snapshot", default="main_oof_v1")
    parser.add_argument("--output-dir", type=Path,
                        help="Report files are replaced and obsolete figures removed; "
                        "default: experiment/results/figures/SNAPSHOT")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or args.experiment_dir / "results/figures" / args.snapshot
    result = render_report(args.experiment_dir, args.snapshot, output, dpi=args.dpi)
    print(json.dumps({"output_dir": str(result), "status": "oof_figures_completed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
