"""Fit or check one fold's B0/PCA baseline state using the saved train manifest."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from wood_degradation_map.experiments.baselines import B0Baseline, PCABaseline, fit_pca
from wood_degradation_map.experiments.data import FoldData, SpectrumBatch
from wood_degradation_map.experiments.input_validation import load_input_inventory
from wood_degradation_map.experiments.manifests import _digest, load_manifest_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "check"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--experiment-dir", type=Path,
                        default=root / "outputs/experiments/cv_200hz_snr10_linear256_v1")
    parser.add_argument("--processed-dir", type=Path,
                        default=root / "data/processed/preprocessing/200hz_snr10_linear256")
    parser.add_argument("--metadata", type=Path, default=root / "data/metadata/古材メタデータ.csv")
    return parser.parse_args()


def _probe(data: FoldData, split: Literal["train", "test"]) -> SpectrumBatch:
    # These bounded probes test transform/save/load only, not full test performance.
    with closing(data.batches(split, chunk_pixels=8)) as iterator:
        return next(iterator)


def main() -> int:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    results = experiment / f"results/baselines/fold_{args.fold}/repeat_{args.repeat}"
    checkpoints = experiment / f"checkpoints/baselines/fold_{args.fold}/repeat_{args.repeat}"
    if args.action == "fit" and (results.exists() or checkpoints.exists()):
        raise FileExistsError("Baseline output already exists; use check to reload it")
    inventory = load_input_inventory(args.processed_dir, args.metadata)
    manifest = load_manifest_bundle(experiment, inventory, args.processed_dir, args.metadata)
    data = FoldData(inventory, manifest, args.fold)
    artifact_hashes = json.loads((experiment / "manifests/complete.json").read_text(
        encoding="utf-8",
    ))["artifact_sha256"]
    train_probe, test_probe = _probe(data, "train"), _probe(data, "test")
    if args.action == "fit":
        b0 = B0Baseline()
        pca = fit_pca(data, repeat=args.repeat)
        expected = pca.transform(test_probe.snv).values
        results.mkdir(parents=True, exist_ok=False)
        checkpoints.mkdir(parents=True, exist_ok=False)
        b0.save(results / "b0.json")
        pca.save(checkpoints / "pca.npz")
        restored = PCABaseline.load(checkpoints / "pca.npz", fold=args.fold, repeat=args.repeat)
        error = float(np.max(np.abs(expected - restored.transform(test_probe.snv).values)))
        if error > 1e-6:
            raise ValueError(f"PCA save/load probe mismatch: maximum absolute error {error}")
        report = {
            "status": "fitted_and_roundtrip_checked", "pca_fit": asdict(pca.record),
            "manifest_artifact_sha256": artifact_hashes,
            "pca_checkpoint_sha256": _digest(checkpoints / "pca.npz"),
            "pca_reusable_across_repeats": pca.reusable_across_repeats,
            "pca_save_load_probe_absolute_error_max": error,
            "probes": [{
                "split": split, "sample_id": batch.sample_id,
                "hdf5_rows": batch.hdf5_rows.tolist(),
                "B0": asdict(b0.transform(batch.snv).diagnostics),
                "B1": asdict(pca.transform(batch.snv).diagnostics),
            } for split, batch in (("train", train_probe), ("test", test_probe))],
            "scope": "Train PCA fit and bounded transform probes; no clustering or test metrics",
        }
        with (results / "fit.json").open("x", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    else:
        report = json.loads((results / "fit.json").read_text(encoding="utf-8"))
        if (report["manifest_artifact_sha256"] != artifact_hashes
                or report["pca_checkpoint_sha256"] != _digest(checkpoints / "pca.npz")):
            raise ValueError("Manifest or PCA checkpoint differs from the fit record")
        b0 = B0Baseline.load(results / "b0.json")
        pca = PCABaseline.load(checkpoints / "pca.npz", fold=args.fold, repeat=args.repeat)
        if (pca.record.sample_ids != data.train_sample_ids
                or pca.record.train_pixel_count != data.train_pixel_count):
            raise ValueError("PCA training provenance differs from the current fold")
        for batch in (train_probe, test_probe):
            b0.transform(batch.snv)
            pca.transform(batch.snv)
        report["status"] = "validated_existing_baselines"
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
