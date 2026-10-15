"""Evaluate M11 spatial reproducibility using completed encoders and cached maps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from wood_degradation_map.experiments.global_manifest import GlobalData, load_global_bundle
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _read_json, _require, _write_json
from wood_degradation_map.experiments.spatial_reproducibility_pipeline import (
    SpatialSources, check_evaluation, check_report, clustering_seed_plan, evaluate,
    separate_roots, write_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cluster", "evaluate", "report", "check"))
    parser.add_argument("--source-experiment-dir", type=Path,
                        default=root / "outputs/experiments/global_v1")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=root / "outputs/experiments/global_spatial_reproducibility_v1",
                        help="Completed encoder 2/3 weights and reusable clustering maps")
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/global_spatial_reproducibility_v2")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed/production_v1")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    parser.add_argument("--encoders", type=int, nargs="+", choices=(1, 2, 3), default=None,
                        help="Encoders for the shared second Cosine-KMeans seed; default: 1 2 3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-pixels", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--include-overlap", action="store_true",
                        help="Also save overlap fractions and VI components during evaluate")
    parser.add_argument("--resume", action="store_true",
                        help="Verify and skip completed outputs")
    args = parser.parse_args(argv)
    if args.chunk_pixels < 1 or args.dpi < 1 or args.device < 0:
        parser.error("Require positive chunk size/DPI and a nonnegative device index")
    if args.encoders is not None and args.action != "cluster":
        parser.error("--encoders applies only to cluster")
    if args.encoders is not None and len(set(args.encoders)) != len(args.encoders):
        parser.error("Do not repeat encoder IDs")
    if args.include_overlap and args.action != "evaluate":
        parser.error("--include-overlap applies only to evaluate")
    if args.resume and args.action == "check":
        parser.error("Check does not take --resume")
    args.encoders = args.encoders or [1, 2, 3]
    return args


def _inputs(args: argparse.Namespace) -> tuple[Path, Path, SpatialSources]:
    experiment, source = separate_roots(args.experiment_dir, args.source_experiment_dir)
    sources = SpatialSources(args.artifacts_dir.resolve())
    sources.validate_roots(experiment, source)
    separate_roots(experiment, args.processed_dir)
    _require(not args.metadata.resolve().is_relative_to(experiment),
             "The metadata source must not be inside the output experiment")
    return experiment, source, sources


def _run(args: argparse.Namespace) -> dict[str, object]:
    experiment, source, sources = _inputs(args)
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_global_bundle(source, inventory, args.processed_dir, args.metadata)
    data = GlobalData(inventory, manifest)
    if args.action == "prepare":
        if not (args.resume and (experiment / "protocol").exists()):
            sources.prepare(experiment, source, data, inventory)
        return {**sources.check_prepared(experiment, source, data, inventory),
                "seed_plan": clustering_seed_plan()}
    sources.check_prepared(experiment, source, data, inventory)
    if args.action == "cluster":
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError("A valid single CUDA device is required; no CPU fallback")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        for encoder in args.encoders:
            sources.cluster(experiment, source, data, inventory, encoder, device=device,
                            chunk_pixels=args.chunk_pixels, resume=args.resume)
        return {"status": "spatial_clustering_verified", "encoders": args.encoders}
    if args.action == "evaluate":
        output = experiment / "results/evaluation"
        if args.resume and output.exists():
            _require(_read_json(output / "run.json")["contract"]["include_overlap"]
                     == args.include_overlap, "--include-overlap differs from saved evaluation")
        else:
            evaluate(experiment, source, data, inventory,
                     include_overlap=args.include_overlap, sources=sources)
        return check_evaluation(experiment, source, data, inventory, sources=sources)
    if args.action == "report":
        output = experiment / "results/report"
        if args.resume and output.exists():
            _require(_read_json(output / "run.json")["contract"]["dpi"] == args.dpi,
                     "--dpi differs from saved report")
        else:
            write_report(experiment, source, data, inventory, dpi=args.dpi, sources=sources)
    return check_report(experiment, source, data, inventory, sources=sources)


def main() -> int:
    args = parse_args()
    # Validate before the failure logger can create anything in an input tree.
    experiment, _, _ = _inputs(args)
    started, status = time.perf_counter(), "failed_or_interrupted"
    try:
        report = _run(args)
        status = "completed"
    finally:
        error_type = sys.exc_info()[0]
        if args.action != "check":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            try:
                (experiment / "attempts").mkdir(parents=True, exist_ok=True)
                _write_json(experiment / "attempts" / f"{args.action}_{stamp}.json", {
                    "action": args.action, "status": status,
                    "error_type": error_type.__name__ if error_type else None,
                    "wall_seconds": time.perf_counter() - started,
                    "encoders": args.encoders, "resume": args.resume,
                    "include_overlap": args.include_overlap, "device_index": args.device,
                    "artifacts_dir": str(args.artifacts_dir.resolve()),
                })
            except OSError as logging_error:
                if error_type is None:
                    raise
                print(f"Could not save attempt record: {logging_error}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
