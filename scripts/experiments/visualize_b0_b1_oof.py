"""Render B0/B1 OOF sanity figures from existing CV artifacts only (CPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from wood_degradation_map.experiments.oof_sanity import render_sanity  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/production_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/sanity_checks/b0_b1_oof_visualization",
                        help="Must not already exist; CV artifacts are never overwritten")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = render_sanity(args.experiment_dir, args.output_dir, dpi=args.dpi)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
