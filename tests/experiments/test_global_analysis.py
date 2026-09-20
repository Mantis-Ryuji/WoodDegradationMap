"""Small CPU global fits and independent spectral/matching/report regression fixtures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from chemomae.clustering.cosine_kmeans import CosineKMeans
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import global_clustering as gc
from wood_degradation_map.experiments import global_reporting as gr
from wood_degradation_map.experiments.baselines import B0Baseline
from wood_degradation_map.experiments.global_baselines import GlobalPCA, global_baselines
from wood_degradation_map.experiments.global_manifest import GlobalData, create_global_manifest
from wood_degradation_map.experiments.input_validation import InputInventory, SampleInput
from wood_degradation_map.experiments.manifests import _digest, _write_json
from wood_degradation_map.experiments.records import make_run_record
from wood_degradation_map.experiments.training import runtime_record


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, GlobalData, InputInventory]:
    experiment = tmp_path / "global"
    (experiment / "manifests").mkdir(parents=True)
    _write_json(experiment / "manifests/complete.json", {"artifact_sha256": {"fixture": "input"}})
    rng = np.random.default_rng(120)
    wave = np.linspace(913.1, 2305.59, 256)
    samples = []
    for si, (sample_id, count) in enumerate((("KYOw02789", 125), ("KYOw02777", 73))):
        path = tmp_path / f"{sample_id}.h5"
        coordinates = np.column_stack((np.arange(count) // 16, np.arange(count) % 16))
        coordinates = coordinates[rng.permutation(count)].astype(np.int32)
        mask = np.zeros((9, 16), dtype=np.uint8)
        mask[coordinates[:, 0], coordinates[:, 1]] = 1
        x = np.linspace(0, 1, 256)
        reflectance = np.stack([.5 + .02 * si + .09 * np.sin((2 + r % 8) * np.pi * x)
                                + .002 * rng.normal(size=256) for r in range(count)])
        reflectance[-1, 10] = 0
        reflectance = reflectance.astype(np.float32)
        snv = reflectance - reflectance.mean(axis=1, keepdims=True)
        snv /= snv.std(axis=1, ddof=1, keepdims=True)
        with h5py.File(path, "w") as handle:
            for name, value in (("pixel_row_col", coordinates), ("valid_spectrum_mask", mask),
                                ("reflectance", reflectance), ("snv", snv), ("wavelength_nm", wave)):
                handle.create_dataset(name, data=value)
            handle.attrs.update(sample_id=sample_id, saved_pixel_count=count, schema_version=2)
        samples.append(SampleInput(sample_id, path, 9, 16, count))
    inventory = InputInventory("fixture", tuple(samples), (), wave[0], wave[-1])
    data = GlobalData(inventory, create_global_manifest(inventory, q=32))
    global_baselines(experiment, data, fit=True)
    return experiment, data, inventory


@pytest.mark.parametrize("condition", ("B0", "B1"))
def test_global_fit_fixed_seed_roundtrip_complete_coordinates_and_tamper(
    inputs: tuple[Path, GlobalData, InputInventory], condition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, data, inventory = inputs
    fit_method = CosineKMeans.fit
    calls = []

    def count_fit(self: CosineKMeans, values: torch.Tensor, chunk: int | None = None) -> CosineKMeans:
        calls.append(values.shape)
        assert self.random_state == gc.global_seed("kmeans")
        assert self.n_components == 8 and self.max_iter == 500 and self.tol == 1e-4
        assert chunk is None
        return fit_method(self, values, chunk=chunk)

    monkeypatch.setattr(CosineKMeans, "fit", count_fit)
    gc.run_global_clustering(experiment, data, inventory, condition,
                             device=torch.device("cpu"), chunk_pixels=17)
    report = gc.check_global_clustering(experiment, data, inventory, condition)
    assert report["prediction_pixels"] == 198 and report["fit_pixels"] == 64
    assert calls == [(64, 256 if condition == "B0" else 16)]
    results, checkpoints = gc.condition_paths(experiment, condition)
    assert '"fold":' not in (results / "run.json").read_text()
    representation, _ = gc.load_global_representation(experiment, data, condition,
                                                     device=torch.device("cpu"))
    fitted = gc.GlobalClusters.load(checkpoints / "centers_k8.npz", condition=condition,
                                    device=torch.device("cpu"))
    for sample in inventory.samples:
        labels = gc.read_label_map(results / "maps" / f"{sample.sample_id}.npz", sample)
        with h5py.File(sample.path, "r") as handle:
            coordinates = handle["pixel_row_col"][:]
            expected = fitted.predict(representation.transform(handle["snv"][:]).values) + 1
        np.testing.assert_array_equal(labels[tuple(coordinates.T)], expected)
    with pytest.raises(ValueError, match="output exists"):
        gc.run_global_clustering(experiment, data, inventory, condition, device=torch.device("cpu"))
    path = results / "maps" / f"{inventory.samples[0].sample_id}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        gc.check_global_clustering(experiment, data, inventory, condition)


def test_extraction_uses_only_manifest_fit_rows_and_retains_tail(
    inputs: tuple[Path, GlobalData, InputInventory],
) -> None:
    _, data, inventory = inputs
    expected = B0Baseline().transform(data.train_matrix()).values
    for selection in data._train:
        with h5py.File(selection.sample.path, "r+") as handle:
            unused = np.setdiff1d(np.arange(selection.sample.saved_pixel_count), selection.rows)
            handle["snv"][unused] = np.full((len(unused), 256), np.nan, dtype=np.float32)
    actual = gc.collect_global_features(data, B0Baseline(), "B0", chunk_pixels=13)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == 64
    with pytest.raises(ValueError, match="non-finite SNV"):
        list(gc.AllPixelData(inventory.samples).all_batches(chunk_pixels=13))


def test_pca_export_preserves_shared_pixel_and_snv_label_identity(
    inputs: tuple[Path, GlobalData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wood_degradation_map.experiments import global_pca_inputs as ui

    experiment, _, inventory = inputs
    report = experiment / ui.DEFAULT_DIRECTORY
    report.mkdir(parents=True)
    _write_json(report / "completion.json", {"fixture": "SNV report"})
    monkeypatch.setattr(ui, "load_input_inventory", lambda *args: inventory)
    # Reconstruct the same deterministic manifest used by the fixture.
    data_manifest = create_global_manifest(inventory, q=32)
    monkeypatch.setattr(ui, "load_global_bundle", lambda *args: data_manifest)
    monkeypatch.setattr(ui, "check_global_report", lambda *args: None)
    expected = {}
    baseline = GlobalPCA.load(experiment / "checkpoints/baselines/pca.npz")
    selected = np.argsort(-baseline.estimator.singular_values_, kind="stable")[:2]
    expected["B1"] = baseline.estimator.transform(
        GlobalData(inventory, data_manifest).train_matrix())[:, selected]

    def extract(
        current_data: GlobalData, representation: object, condition: str, *, chunk_pixels: int,
    ) -> np.ndarray:
        assert condition != "B1"  # B1 must bypass the normalized representation wrapper.
        values = B0Baseline().transform(current_data.train_matrix()).values
        if condition != "B0":
            values = values[:, :16].copy()
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        expected[condition] = values.copy()
        return values

    def label_map(path: Path, sample: SampleInput) -> np.ndarray:
        return (np.arange(sample.height * sample.width).reshape(sample.height, sample.width)
                % 8 + 1).astype(np.uint8)

    monkeypatch.setattr(ui, "load_global_representation",
                        lambda exp, data, condition, **kwargs:
                        (baseline if condition == "B1" else None, {}))
    monkeypatch.setattr(ui, "collect_global_features", extract)
    monkeypatch.setattr(ui, "read_label_map", label_map)
    for condition in ui.CONDITIONS:
        (report / condition).mkdir()
        pd.DataFrame({"original_cluster": np.arange(1, 9),
                      "display_cluster": np.arange(8, 0, -1),
                      "snv_cosine_similarity": np.ones(8)}).to_csv(
                          report / condition / "matching.csv", index=False)
    ui.prepare_inputs(experiment, Path("unused"), Path("unused"), device=torch.device("cpu"))
    output = experiment / ui.INPUT_DIRECTORY
    record, pixels = ui.check_inputs(output)
    assert record["plot_seed"] == ui.ROOT_SEED
    expected_pixels = data_manifest.fit_pixels[["sample_id", "hdf5_row", "pixel_row", "pixel_col"]]
    pd.testing.assert_frame_equal(pixels[expected_pixels.columns], expected_pixels,
                                  check_dtype=False)
    raw = (pixels.pixel_row * 16 + pixels.pixel_col) % 8 + 1
    for condition in ui.CONDITIONS:
        np.testing.assert_allclose(np.load(output / f"{condition}.npy"), expected[condition],
                                   rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(pixels[f"{condition}_original_cluster"], raw)
        np.testing.assert_array_equal(pixels[f"{condition}_cluster"], 9 - raw)


def test_failed_global_fit_does_not_publish_partial_condition(
    inputs: tuple[Path, GlobalData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, data, inventory = inputs

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic fitting failure")

    monkeypatch.setattr(gc, "fit_global_clusters", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        gc.run_global_clustering(experiment, data, inventory, "B0", device=torch.device("cpu"))
    assert not any(path.exists() for path in gc.condition_paths(experiment, "B0"))
    assert not list(experiment.glob(".global-clustering-*"))


def _neural_source(
    experiment: Path, data: GlobalData, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    original = gc.global_config

    def config() -> dict:
        value = original()
        value["training"]["batch_size"] = 4
        return value

    def build(condition: str) -> ChemoMAE:
        return ChemoMAE(seq_len=256, n_patches=16, d_model=8, nhead=2, num_layers=1,
                        dim_feedforward=16, dropout=0, latent_dim=16, latent_normalize=True,
                        decoder_num_layers=1, n_mask=8)

    monkeypatch.setattr(gc, "global_config", config)
    monkeypatch.setattr(gc, "build_global_model", build)
    result = experiment / "results/neural/M00/repeat_1"
    weights = experiment / "checkpoints/neural/M00/repeat_1/last_model.pt"
    result.mkdir(parents=True)
    weights.parent.mkdir(parents=True)
    torch.save(build("M00").state_dict(), weights)
    steps = data.train_pixel_count // 4
    fields = {"scope": "global_fit", "mode": "training", "condition": "M00", "repeat": 1,
        "smoke_id": None, "smoke_batches_per_epoch": None, "config": config(),
        "train_sample_ids": list(data.train_sample_ids), "train_pixels": data.train_pixel_count,
        "batch_size": 4, "steps_per_full_epoch": steps, "planned_production_updates": steps * 800,
        "code_sha256": gc.global_code_hashes(), "runtime": runtime_record(torch.device("cpu")),
        "manifest_artifact_sha256": {"fixture": "input"},
        "seeds": {p: gc.global_seed(p) for p in ("model_init", "pixel_order", "mask", "train_aug")}}
    _write_json(result / "run.json", make_run_record(fields))
    counters = {"completed_epochs": 800, "attempted_updates": steps * 800,
                "optimizer_updates": steps * 800, "nonzero_lr_updates": steps * 800 - 1}
    _write_json(result / "completion.json", {**counters, "status": "training_completed",
        "amp_skips": 0, "weights_file": str(weights), "weights_sha256": _digest(weights)})
    _write_json(result / "checkpoint.json", counters)
    return result


def test_neural_loader_uses_completed_global_raw_weights_and_rejects_smoke(
    inputs: tuple[Path, GlobalData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, data, _ = inputs
    result = _neural_source(experiment, data, monkeypatch)
    representation, source = gc.load_global_representation(experiment, data, "M00",
                                                           device=torch.device("cpu"))
    spectra = data.train_matrix()[:5]
    values = representation.transform(spectra).values
    assert values.shape == (5, 16) and source["completed_epochs"] == 800
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
    path = result / "run.json"
    record = json.loads(path.read_text())
    record["mode"] = "smoke"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source contract"):
        gc.source_record(experiment, data, "M00")


def _make_maps(experiment: Path, inventory: InputInventory) -> np.ndarray:
    permutation = np.roll(np.arange(8), 3)
    for ci, condition in enumerate(gr.CONDITIONS):
        directory = gc.condition_paths(experiment, condition)[0] / "maps"
        directory.mkdir(parents=True, exist_ok=True)
        for sample in inventory.samples:
            with h5py.File(sample.path, "r") as handle:
                coords = handle["pixel_row_col"][:]
            labels = np.arange(sample.saved_pixel_count) % 8
            if ci != gr.REFERENCE:
                labels = permutation[labels]
            output = np.zeros((sample.height, sample.width), dtype=np.uint8)
            output[tuple(coords.T)] = labels + 1
            np.savez_compressed(directory / f"{sample.sample_id}.npz", labels_k8=output)
    return permutation


def test_spectral_aggregation_log_order_exclusions_sample_weights_and_matching(
    inputs: tuple[Path, GlobalData, InputInventory],
) -> None:
    experiment, _, inventory = inputs
    permutation = _make_maps(experiment, inventory)
    summary = gr.collect_spectra(experiment, inventory, chunk_pixels=11)
    for sample in inventory.samples:
        si = summary.sample_ids.index(sample.sample_id)
        with h5py.File(sample.path, "r") as handle:
            reflectance = handle["reflectance"][:].astype(float)
        for k in range(8):
            subset = reflectance[np.arange(len(reflectance)) % 8 == k]
            positive = subset[(subset > 0).all(axis=1)]
            expected = (-np.log10(positive)).mean(axis=0)
            np.testing.assert_allclose(summary.values[si, gr.REFERENCE, k, 2], expected, atol=1e-14)
            assert summary.pseudoabsorbance_pixels[si, gr.REFERENCE, k] == len(positive)
    mean, low, high, n = gr.equal_sample_summary(summary.values[:, gr.REFERENCE, 0, 0])
    assert n == 2
    np.testing.assert_allclose(mean, summary.values[:, gr.REFERENCE, 0, 0].mean(axis=0))
    counts = summary.pixels[:, gr.REFERENCE, 0]
    pooled = np.average(summary.values[:, gr.REFERENCE, 0, 0], axis=0, weights=counts)
    assert not np.allclose(mean, pooled, atol=1e-6)
    assert np.all(low <= high)
    # Reflectance and derivative curves do not contribute to the SNV assignment.
    summary.values[:, :, :, [0, 2, 3], :] = np.nan
    mapping, similarity = gr.match_snv(summary)
    np.testing.assert_array_equal(mapping[0, permutation + 1], np.arange(1, 9))
    np.testing.assert_allclose(similarity[0, np.arange(8), permutation], 1, atol=1e-12)
    np.testing.assert_array_equal(mapping[gr.REFERENCE], np.arange(9))


def test_sg_nm_scale_and_absent_curves() -> None:
    wave = np.linspace(913.1, 2305.59, 256)
    quadratic = 2e-6 * (wave - 1600) ** 2 + .1
    np.testing.assert_allclose(gr.second_derivative(quadratic, wave), 4e-6, atol=1e-16)
    values = np.stack((quadratic, np.full(256, np.nan)))
    mean, low, high, n = gr.equal_sample_summary(values)
    assert n == 1
    np.testing.assert_array_equal(mean, quadratic)
    np.testing.assert_array_equal(low, high)
    assert gr.equal_sample_summary(values[1:])[3] == 0
    broken = wave.copy()
    broken[20] += 1
    with pytest.raises(ValueError, match="uniform"):
        gr.second_derivative(quadratic, broken)


def test_empty_snv_cluster_fails_without_fabricated_labels(
    inputs: tuple[Path, GlobalData, InputInventory],
) -> None:
    experiment, _, inventory = inputs
    _make_maps(experiment, inventory)
    summary = gr.collect_spectra(experiment, inventory)
    summary.pixels[:, 0, 1] += summary.pixels[:, 0, 0]
    summary.pixels[:, 0, 0] = 0
    summary.contingency[0, :, 1] += summary.contingency[0, :, 0]
    summary.contingency[0, :, 0] = 0
    with pytest.raises(ValueError, match="Empty cluster.*B0 cluster 1"):
        gr.match_snv(summary)


def test_snv_assignment_uses_equal_sample_spectra_instead_of_spatial_overlap() -> None:
    basis = np.zeros((8, 256))
    basis[np.arange(8), 2 * np.arange(8)] = 1
    basis[np.arange(8), 2 * np.arange(8) + 1] = -1
    values = np.full((2, 5, 8, 4, 256), np.nan)
    values[:, :, :, 1] = basis
    values[0, 0, :2, 1] = .8 * basis[[1, 0]]
    values[1, 0, :2, 1] = .6 * basis[:2]
    pixels = np.broadcast_to(np.array([1, 100])[:, None, None], (2, 5, 8)).copy()
    contingency = np.repeat((101 * np.eye(8, dtype=np.int64))[None], 5, axis=0)
    summary = gr.SpectralSummaries(("small", "large"), np.arange(256), values,
                                  pixels, pixels.copy(), contingency)
    mapping, similarity = gr.match_snv(summary)
    # Spatial overlap and pixel-weighted spectra favor identity; equal-sample SNV swaps 1/2.
    np.testing.assert_allclose(similarity[0, :2, :2], [[.6, .8], [.8, .6]])
    np.testing.assert_array_equal(gr._pooled_iou(summary)[0], np.eye(8))
    np.testing.assert_array_equal(mapping[0], [0, 2, 1, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(mapping[gr.REFERENCE], np.arange(9))
    # Cosine retains sign rather than taking absolute similarity.
    values[:, 0, 2, 1] *= -3
    _, similarity = gr.match_snv(summary)
    assert similarity[0, 2, 2] == pytest.approx(-1)
    # An absent sample/cluster curve is excluded, without zero-filling or sample weighting.
    values[0, 0, 0, 1] = np.nan
    _, similarity = gr.match_snv(summary)
    assert similarity[0, 0, 0] == pytest.approx(1)
    for invalid in (0, np.nan):
        values[:, 0, 0, 1] = invalid
        with pytest.raises(ValueError, match="Undefined or zero-norm SNV.*B0 cluster 1"):
            gr.match_snv(summary)


def test_report_png_csv_alignment_and_tamper_detection(
    inputs: tuple[Path, GlobalData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, data, inventory = inputs
    permutation = _make_maps(experiment, inventory)
    # The separate clustering tests cover source audits; this fixture isolates report arithmetic.
    monkeypatch.setattr(gr, "report_contract", lambda *args: {"fixture": "saved global maps"})
    monkeypatch.setattr(gr, "REPRESENTATIVES", tuple(("fixture", s.sample_id)
                                                    for s in inventory.samples))
    output = experiment / gr.DEFAULT_DIRECTORY
    output.mkdir(parents=True)
    latent = output / "pca-latent-2d"
    latent.mkdir()
    (latent / "projection.png").write_bytes(b"independently managed PCA figure")
    (latent / "completion.json").write_text('{"scope":"latent"}', encoding="utf-8")
    obsolete_latent = output / "latent"
    obsolete_latent.mkdir()
    (obsolete_latent / "projection.png").write_bytes(b"obsolete output location")
    old = output / "obsolete.png"
    old.write_text("old report", encoding="utf-8")
    with monkeypatch.context() as failure:
        def fail(*args: object, **kwargs: object) -> gr.SpectralSummaries:
            raise ValueError("fixture render failure")
        failure.setattr(gr, "collect_spectra", fail)
        with pytest.raises(ValueError, match="fixture render failure"):
            gr.render_global_report(experiment, data, inventory, dpi=45)
    assert old.read_text(encoding="utf-8") == "old report"
    report = gr.render_global_report(experiment, data, inventory, dpi=45, chunk_pixels=19)
    assert report["png_count"] == 26 and report["csv_count"] == 37
    assert gr.check_global_report(experiment, data, inventory)["checks_passed"]
    assert (latent / "projection.png").read_bytes() == b"independently managed PCA figure"
    assert json.loads((latent / "completion.json").read_text()) == {"scope": "latent"}
    assert not obsolete_latent.exists()
    assert not old.exists()
    assert [p.name for p in output.glob("*.png")] == [gr.REPRESENTATIVE_FIGURE]
    sample = inventory.samples[0]
    with np.load(output / "M00/label_maps.npz") as saved:
        expected = saved[sample.sample_id]
        for condition in gr.CONDITIONS:
            with np.load(output / condition / "label_maps.npz") as target:
                np.testing.assert_array_equal(target[sample.sample_id], expected)
    matching = pd.read_csv(output / "B0/matching.csv")
    assert matching.loc[matching.condition == "B0", "display_cluster"].tolist() == (
        np.argsort(permutation) + 1).tolist()
    np.testing.assert_allclose(matching.condition_overlap_fraction, 1)
    assert (matching.common_pixels == 198).all()
    np.testing.assert_allclose(matching.iou, 1)
    np.testing.assert_allclose(matching.snv_cosine_similarity, 1)
    matrices = pd.read_csv(output / "B0/matching_matrices.csv")
    with np.load(output / "spectral_summary.npz") as saved:
        means = saved["means"][:, :, gr.KINDS.index("snv")]
        means /= np.linalg.norm(means, axis=-1, keepdims=True)
        expected_similarity = means[gr.REFERENCE] @ means[0].T
    actual_similarity = matrices.pivot(index="reference_cluster", columns="display_target_cluster",
                                       values="snv_cosine_similarity").to_numpy()
    np.testing.assert_allclose(actual_similarity, expected_similarity, atol=1e-12)
    differences = pd.read_csv(output / "B0/difference_spectra.csv")
    np.testing.assert_allclose(differences.difference, 0, atol=1e-12)
    occupancy = pd.read_csv(output / "B0/occupancy.csv")
    assert np.allclose(occupancy.groupby(["sample_id", "condition"]).fraction.sum(), 1)
    for condition in gr.CONDITIONS:
        assert sorted(p.name for p in (output / condition / "labels").glob("*.png")) == [
            "00_samples_01-02_1x7.png", "07_representative_samples_1x7.png",
            "08_all_samples_7x7.png"]
        assert (output / condition / "01_representative_spectra.png").is_file()
        assert (output / condition / "02_snv_cosine_matrix.png").is_file()
        assert not (output / condition / "02_iou_matrix.png").exists()
    completion_path = output / "completion.json"
    completion = completion_path.read_text(encoding="utf-8")
    completion_path.write_text(json.dumps({**json.loads(completion), "png_count": 12}),
                               encoding="utf-8")
    with pytest.raises(ValueError, match="artifact counts"):
        gr.check_global_report(experiment, data, inventory)
    completion_path.write_text(completion, encoding="utf-8")
    (output / "B0/matching.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        gr.check_global_report(experiment, data, inventory)


def test_publish_report_restores_latent_when_directory_swap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "global"
    output = experiment / gr.DEFAULT_DIRECTORY
    (output / "pca-latent-2d").mkdir(parents=True)
    (output / "old.csv").write_text("old report", encoding="utf-8")
    (output / "pca-latent-2d/projection.png").write_bytes(b"preserved PCA")
    stage = experiment / ".staging/report"
    stage.mkdir(parents=True)
    (stage / "new.csv").write_text("new report", encoding="utf-8")
    rename = Path.rename

    def fail_publication(source: Path, target: Path) -> Path:
        if source == stage and target == output:
            raise PermissionError("synthetic directory lock")
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(PermissionError, match="synthetic directory lock"):
        gr._publish_report(stage, output, experiment)
    assert (output / "old.csv").read_text() == "old report"
    assert (output / "pca-latent-2d/projection.png").read_bytes() == b"preserved PCA"
    assert not (stage / "pca-latent-2d").exists()


def _cli(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / f"scripts/experiments/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_separates_clustering_and_rendering() -> None:
    cluster, visual = _cli("global_cluster"), _cli("visualize_global")
    assert cluster.parse_args(["run"]).conditions == list(gr.CONDITIONS)
    assert visual.parse_args(["run"]).dpi == 240
    for arguments in (["check", "--resume"], ["run", "--conditions", "M11", "M11"],
                      ["run", "--conditions", "M10"]):
        with pytest.raises(SystemExit):
            cluster.parse_args(arguments)
    with pytest.raises(SystemExit):
        visual.parse_args(["run", "--conditions", "M11"])


def test_cluster_resume_validates_before_skipping_and_stops_on_failure(
    inputs: tuple[Path, GlobalData, InputInventory], monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, data, inventory = inputs
    cli = _cli("global_cluster")
    args = cli.parse_args(["run", "--conditions", "B0", "B1", "--resume",
                           "--experiment-dir", str(experiment)])
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "load_input_inventory", lambda *args: inventory)
    monkeypatch.setattr(cli, "load_global_bundle", lambda *args: None)
    monkeypatch.setattr(cli, "GlobalData", lambda *args: data)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    gc.condition_paths(experiment, "B0")[0].mkdir(parents=True)
    fitted, checked = [], []

    def fit(*args: object, **kwargs: object) -> dict[str, object]:
        fitted.append(args[3])
        return {}

    def check(*args: object) -> dict[str, object]:
        checked.append(args[3])
        return {"checks_passed": True}

    monkeypatch.setattr(cli, "run_global_clustering", fit)
    monkeypatch.setattr(cli, "check_global_clustering", check)
    assert cli.main() == 0
    assert fitted == ["B1"] and checked == ["B0", "B1"]
    fitted.clear()

    def failed_check(*args: object) -> dict[str, object]:
        raise ValueError("Incomplete existing condition")

    monkeypatch.setattr(cli, "check_global_clustering", failed_check)
    with pytest.raises(ValueError, match="Incomplete existing"):
        cli.main()
    assert not fitted
