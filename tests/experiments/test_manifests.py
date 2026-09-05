from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from wood_degradation_map.experiments.config import (
    ADOPTED_SAMPLE_IDS,
    CLUSTER_COUNTS,
    CONDITIONS,
    FOLDS,
    PERTURBATIONS,
    REPEATS,
    RUN_PURPOSES,
    experiment_config,
    kmeans_seed,
    perturbation_seed,
    run_seed,
    sampling_seed,
    seed_plan,
)
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import (
    CVManifest,
    create_cv_manifest,
    create_manifest_bundle,
    load_manifest_bundle,
    validate_cv_manifest,
)


def _inventory(tmp_path: Path, *, count: int = 10, pixels: int = 12) -> InputInventory:
    root = tmp_path / "200hz_snr10_linear256"
    (root / "samples").mkdir(parents=True)
    height = (pixels + 319) // 320
    flat = np.arange(pixels, dtype=np.int32)
    coordinates = np.column_stack((flat // 320, flat % 320))
    mask = np.zeros((height, 320), dtype=np.uint8)
    mask[coordinates[:, 0], coordinates[:, 1]] = 1
    samples = []
    for index in range(count):
        sample_id = f"KYOw{2700 + index:05d}"
        path = root / "samples" / f"{sample_id}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pixel_row_col", data=coordinates)
            handle.create_dataset("valid_spectrum_mask", data=mask)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=pixels, schema_version=2)
            # No spectra exist: planning must depend only on coordinate/mask data.
        samples.append(SampleInput(sample_id, path, height, 320, pixels))
    return InputInventory(root.name, tuple(samples), (), 900.0, 2300.0)


def test_fixed_condition_matrix_and_recipe() -> None:
    conditions = {item.condition_id: item for item in CONDITIONS}
    assert set(conditions) == {"B0", "B1", "A0", "M00", "M10", "M01", "M11", "M11-25", "M11-75"}
    assert conditions["B0"].output_dim == 256
    assert all(item.output_dim == 16 for key, item in conditions.items() if key != "B0")
    assert conditions["A0"].n_mask == 0 and conditions["A0"].loss_region == "all"
    assert [conditions[name].n_mask for name in ("M11-25", "M11", "M11-75")] == [4, 8, 12]
    assert [(conditions[name].noise_prob, conditions[name].shift_prob)
            for name in ("M00", "M10", "M01", "M11")] == [
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
    ]
    config = experiment_config()
    assert config["training"]["epochs"] == 800
    assert config["training"]["peak_lr"] == 6e-4
    assert config["training"]["batch_size"] == 1024
    assert config["training"]["amp_dtype"] == "fp16"
    assert config["extraction"]["amp"] is False
    assert config["evaluation"]["tf32"] is False
    config["training"]["epochs"] = 1
    assert experiment_config()["training"]["epochs"] == 800


def test_seed_scopes_and_complete_49_sample_plan() -> None:
    assignments = {sample_id: index % 5 + 1 for index, sample_id in enumerate(ADOPTED_SAMPLE_IDS)}
    records = seed_plan(assignments)["records"]
    # 1 split + 49*4 sampling + 49*15 perturbations + 5*3*(5 run purposes + 7 K).
    assert len(records) == 1112
    assert len({record["seed"] for record in records}) == len(records)
    assert all(0 <= record["seed"] < 2**32 for record in records)
    assert all("condition" not in record for record in records)
    evaluation = [record for record in records if record["purpose"] == "evaluation_perturbation"]
    assert all(not ({"repeat", "K", "fold"} & set(record)) for record in evaluation)
    sampling = [record for record in records if record["purpose"] == "sampling"]
    assert all(not ({"repeat", "K"} & set(record)) for record in sampling)
    assert len({run_seed(purpose, fold, repeat) for purpose in RUN_PURPOSES
                for fold in FOLDS for repeat in REPEATS}) == 75
    assert len({kmeans_seed(1, 1, k) for k in CLUSTER_COUNTS}) == 7
    assert len({perturbation_seed("KYOw02700", kind, draw)
                for kind in PERTURBATIONS for draw in range(1, 6)}) == 15


def test_all_49_samples_have_one_test_fold_and_shared_unique_train_rows(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, count=49)
    plan = create_cv_manifest(inventory, q=3)
    validate_cv_manifest(plan, inventory)
    assert sorted(plan.folds.groupby("test_fold").size().tolist()) == [9, 10, 10, 10, 10]
    assert plan.folds["sample_id"].nunique() == 49
    for fold, pixels in plan.train_pixels.items():
        test_ids = set(plan.folds.loc[plan.folds["test_fold"] == fold, "sample_id"])
        assert not test_ids & set(pixels["sample_id"])
        assert len(test_ids | set(pixels["sample_id"])) == 49
        assert set(pixels.groupby("sample_id").size()) == {3}
        assert not pixels.duplicated(["sample_id", "hdf5_row"]).any()
        assert "condition" not in pixels and "repeat" not in pixels and "K" not in pixels


def test_input_order_and_unrelated_rng_do_not_change_plan(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    first = create_cv_manifest(inventory, q=3)
    state = np.random.get_state()
    try:
        np.random.seed(12)
        np.random.random(100)
        reversed_inventory = InputInventory(
            inventory.preprocessing_id, tuple(reversed(inventory.samples)), (), 900.0, 2300.0,
        )
        second = create_cv_manifest(reversed_inventory, q=3)
    finally:
        np.random.set_state(state)
    pd.testing.assert_frame_equal(first.folds, second.folds)
    for fold in FOLDS:
        pd.testing.assert_frame_equal(first.train_pixels[fold], second.train_pixels[fold])


@pytest.mark.parametrize("problem", ["leak", "duplicate_test", "missing_test", "duplicate_pixel",
                                     "missing_pixel", "coordinate", "seed"])
def test_invalid_manifest_is_rejected(tmp_path: Path, problem: str) -> None:
    inventory = _inventory(tmp_path)
    plan = create_cv_manifest(inventory, q=3)
    fold = 1
    pixels = plan.train_pixels[fold]
    if problem == "leak":
        test_ids = plan.folds.loc[plan.folds["test_fold"] == fold, "sample_id"]
        pixels.loc[0, "sample_id"] = test_ids.iloc[0]
    elif problem == "duplicate_test":
        plan.folds.loc[1, "sample_id"] = plan.folds.loc[0, "sample_id"]
    elif problem == "missing_test":
        plan = CVManifest(plan.folds.iloc[1:].copy(), plan.train_pixels, plan.q)
    elif problem == "duplicate_pixel":
        pixels.loc[1, "hdf5_row"] = pixels.loc[0, "hdf5_row"]
    elif problem == "missing_pixel":
        plan.train_pixels[fold] = pixels.iloc[1:].copy()
    elif problem == "coordinate":
        pixels.loc[0, "pixel_col"] = 319
    else:
        pixels.loc[0, "sampling_seed"] = sampling_seed(2, pixels.loc[0, "sample_id"])
    with pytest.raises(ValueError):
        validate_cv_manifest(plan, inventory)


def test_insufficient_pixels_fail_instead_of_replacement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fewer than q"):
        create_cv_manifest(_inventory(tmp_path), q=13)


def test_invalid_mask_pixel_cannot_be_sampled(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    with h5py.File(inventory.samples[0].path, "r+") as handle:
        handle["valid_spectrum_mask"][:] = 0
    with pytest.raises(ValueError, match="outside valid mask"):
        create_cv_manifest(inventory, q=3)


@pytest.fixture
def bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, InputInventory, Path, Path]:
    inventory = _inventory(tmp_path, count=5, pixels=8192)

    # Use five synthetic KYOw IDs while retaining the production q and recipe.
    def fixture_config() -> dict[str, object]:
        config = experiment_config()
        config["split"]["adopted_sample_ids"] = [sample.sample_id for sample in inventory.samples]
        return config

    monkeypatch.setattr(
        "wood_degradation_map.experiments.manifests.experiment_config", fixture_config,
    )
    processed = inventory.samples[0].path.parents[1]
    # Bundle tests supply inventory directly; these small files are provenance inputs.
    for name in ("config.json", "manifest.parquet", "sample_quality.parquet"):
        (processed / name).write_bytes(b"fixture provenance")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("KYOw\n2700\n", encoding="utf-8")
    output = tmp_path / "outputs/experiments/fixture"
    create_manifest_bundle(output, inventory, processed, metadata)
    return output, inventory, processed, metadata


def test_production_bundle_rejects_unapproved_samples_before_writing(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="approved sample set"):
        create_manifest_bundle(output, inventory, tmp_path, tmp_path / "metadata.csv")
    assert not output.exists()


def test_bundle_roundtrip_without_resampling(
    bundle: tuple[Path, InputInventory, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_create(*args: object, **kwargs: object) -> None:
        raise AssertionError("Reload must not regenerate the selection")

    monkeypatch.setattr(
        "wood_degradation_map.experiments.manifests.create_cv_manifest", forbid_create,
    )
    plan = load_manifest_bundle(*bundle)
    assert len(plan.folds) == 5
    assert all(len(pixels) == 4 * 8192 for pixels in plan.train_pixels.values())
    assert plan.q == 8192


def test_existing_bundle_is_never_overwritten(
    bundle: tuple[Path, InputInventory, Path, Path],
) -> None:
    completion = bundle[0] / "manifests/complete.json"
    before = completion.read_bytes()
    with pytest.raises(FileExistsError):
        create_manifest_bundle(*bundle)
    assert completion.read_bytes() == before


@pytest.mark.parametrize("problem", ["artifact", "source_table", "source_hdf5", "incomplete"])
def test_changed_or_incomplete_bundle_fails(
    bundle: tuple[Path, InputInventory, Path, Path], problem: str,
) -> None:
    output, inventory, processed, metadata = bundle
    if problem == "artifact":
        path = output / "manifests/train_pixels/fold_1.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "hdf5_row"] = 2
        table.to_parquet(path, index=False)
    elif problem == "source_table":
        metadata.write_text("changed", encoding="utf-8")
    elif problem == "source_hdf5":
        with h5py.File(inventory.samples[0].path, "r+") as handle:
            handle.attrs["changed"] = True
    else:
        path = output / "manifests/complete.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["status"] = "interrupted"
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest_bundle(output, inventory, processed, metadata)
