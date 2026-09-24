"""Export fixed global representations on the host that owns the audited fits."""

from __future__ import annotations

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from .config import ROOT_SEED
from .artifact_output import publish_directory
from .global_baselines import GlobalPCA
from .global_clustering import collect_global_features, condition_paths, load_global_representation
from .global_clustering import read_label_map
from .global_manifest import GLOBAL_CONDITIONS, GlobalData, load_global_bundle
from .global_reporting import DEFAULT_DIRECTORY, check_global_report
from .global_pca import (
    BASELINE_PROJECTION, CONDITIONS, INPUT_DIRECTORY, baseline_component_indices,
    check_inputs, digest, finish_artifacts, require, write_json,
)
from .input_validation import load_input_inventory
from .oof_sanity import LABEL_COLORS
from .training import runtime_record


def collect_baseline_scores(
    data: GlobalData, representation: GlobalPCA, stage: Path, *, chunk_pixels: int,
) -> np.ndarray:
    """Keep the fitted baseline axes and raw scores, before cosine L2 normalization."""
    model = representation.estimator
    indices = baseline_component_indices(model.singular_values_)
    projection = {name: getattr(model, f"{name}_")[indices]
                  for name in ("components", "explained_variance",
                               "explained_variance_ratio", "singular_values")}
    np.savez_compressed(stage / BASELINE_PROJECTION, mean=model.mean_,
                        source_component_index=indices, **projection)
    features = np.empty((data.train_pixel_count, 2), dtype=np.float32)
    offset = 0
    for batch in data.batches("train", chunk_pixels=chunk_pixels):
        # estimator.transform is the existing FP32 PCA transform without the L2 wrapper.
        scores = model.transform(batch.snv)[:, indices]
        stop = offset + len(scores)
        require(stop <= len(features), "Too many baseline PCA input rows")
        features[offset:stop] = scores
        offset = stop
    require(offset == len(features), "Missing baseline PCA input rows")
    return features


def prepare_inputs(
    experiment: Path, processed: Path, metadata: Path, *, device: torch.device,
    chunk_pixels: int = 1024, resume: bool = False, overwrite: bool = False,
) -> dict:
    require(chunk_pixels > 0, "Invalid extraction chunk size")
    require(not (resume and overwrite), "Choose resume or overwrite")
    require(CONDITIONS == GLOBAL_CONDITIONS, "PCA conditions differ from global fits")
    inventory = load_input_inventory(processed, metadata)
    manifest = load_global_bundle(experiment, inventory, processed, metadata)
    data = GlobalData(inventory, manifest)
    check_global_report(experiment, data, inventory)
    report_path = experiment / DEFAULT_DIRECTORY
    source = {"manifest": digest(experiment / "manifests/complete.json"),
              "snv_report": digest(report_path / "completion.json"),
              "baseline_pca": digest(experiment / "checkpoints/baselines/pca.npz"),
              "prepare_code": digest(Path(__file__))}
    output = experiment / INPUT_DIRECTORY
    if output.exists() and not overwrite:
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume to verify/skip")
        record, _ = check_inputs(output)
        require(record["source"] == source, "Prepared PCA source changed")
        return {"status": "validated_pca_inputs", "pixels": record["pixels"]}
    pixels = manifest.fit_pixels.sort_values(["sample_id", "hdf5_row"]).reset_index(drop=True)
    pixels = pixels[["sample_id", "hdf5_row", "pixel_row", "pixel_col"]].copy()
    pixels.insert(0, "point_id", np.arange(len(pixels)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pca-inputs-", dir=output.parent) as temporary:
        stage = Path(temporary) / "inputs"
        stage.mkdir()
        for condition in CONDITIONS:
            print(f"{condition}: extracting {len(pixels)} shared PCA input rows", flush=True)
            representation, _ = load_global_representation(experiment, data, condition,
                                                           device=device)
            if condition == "B1":
                require(isinstance(representation, GlobalPCA), "B1 requires the saved global PCA")
                features = collect_baseline_scores(data, representation, stage,
                                                   chunk_pixels=chunk_pixels)
            else:
                features = collect_global_features(data, representation, condition,
                                                   chunk_pixels=chunk_pixels)
            require(len(features) == len(pixels), "PCA representation/pixel count differs")
            np.save(stage / f"{condition}.npy", features, allow_pickle=False)
            matching = pd.read_csv(report_path / condition / "matching.csv")
            require("snv_cosine_similarity" in matching, "PCA requires SNV-aligned labels")
            mapping = np.zeros(9, dtype=np.uint8)
            original = matching.original_cluster.to_numpy()
            display = matching.display_cluster.to_numpy()
            require(np.array_equal(np.sort(original), np.arange(1, 9))
                    and np.array_equal(np.sort(display), np.arange(1, 9)), "Invalid SNV mapping")
            mapping[original] = display
            raw_labels = np.zeros(len(pixels), dtype=np.uint8)
            for sample in inventory.samples:
                selected = pixels.sample_id == sample.sample_id
                coordinates = pixels.loc[selected, ["pixel_row", "pixel_col"]].to_numpy()
                label_map = read_label_map(condition_paths(experiment, condition)[0] / "maps"
                                          / f"{sample.sample_id}.npz", sample)
                raw_labels[selected] = label_map[tuple(coordinates.T)]
            pixels[f"{condition}_original_cluster"] = raw_labels
            pixels[f"{condition}_cluster"] = mapping[raw_labels]
            del representation, features
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pixels.to_csv(stage / "pixels.csv", index=False)
        write_json(stage / "inputs.json", {
            "schema_version": 2, "conditions": list(CONDITIONS), "source": source,
            "sample_ids": list(data.train_sample_ids), "pixels_per_sample": manifest.q,
            "pixels": len(pixels), "plot_seed": ROOT_SEED,
            "palette": list(LABEL_COLORS[1:]), "runtime": runtime_record(device),
            "extraction_chunk_pixels": chunk_pixels,
            "representation": {c: ("saved baseline PCA top two raw scores, FP32"
                                    if c == "B1" else "global clustering FP32 unit vectors")
                               for c in CONDITIONS},
            "selection": "all existing shared global fit rows, ordered by sample_id/hdf5_row",
        })
        require(source["snv_report"] == digest(report_path / "completion.json")
                and source["manifest"] == digest(experiment / "manifests/complete.json")
                and source["baseline_pca"] == digest(experiment / "checkpoints/baselines/pca.npz"),
                "PCA source changed during extraction")
        finish_artifacts(stage, "pca_inputs_completed")
        check_inputs(stage)
        publish_directory(stage, output, root=experiment)
    return {"status": "pca_inputs_completed", "pixels": len(pixels), "output": str(output)}
