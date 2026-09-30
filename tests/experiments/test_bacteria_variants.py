"""Small CPU checks for the three additional Table 5 classifier paths."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import bacteria_variant_reporting as reporting
from wood_degradation_map.experiments.bacteria_variant_reporting import select_variant
from wood_degradation_map.experiments.bacteria_variant_training import (
    BacteriaVariantClassifier, variant_config, variant_paths,
)


def _model() -> ChemoMAE:
    return ChemoMAE(
        seq_len=1000, n_patches=20, d_model=256, nhead=8, num_layers=1,
        dim_feedforward=512, dropout=0.0, latent_dim=128,
        latent_normalize=True, decoder_num_layers=1, n_mask=10,
    )


@pytest.mark.parametrize("variant", ("cls_frozen", "z_unfrozen", "z_frozen"))
def test_variant_updates_only_its_documented_parameters(variant: str) -> None:
    classifier = BacteriaVariantClassifier(_model(), run_seed=0, variant=variant)
    encoder = classifier.cls.encoder if variant == "cls_frozen" else classifier.encoder
    before_patch = encoder.patch_proj.weight.detach().clone()
    before_projection = encoder.to_latent.weight.detach().clone()
    trainable = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    assert len(trainable) > 0
    assert encoder.to_latent.weight.requires_grad == (variant == "z_unfrozen")
    assert encoder.patch_proj.weight.requires_grad == (variant == "z_unfrozen")
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    spectra = torch.randn(2, 1000, generator=torch.Generator().manual_seed(19))
    spectra = (spectra - spectra.mean(dim=1, keepdim=True)) / spectra.std(dim=1, keepdim=True)
    logits = classifier(spectra)
    assert logits.shape == (2, 30)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert torch.equal(before_patch, encoder.patch_proj.weight) == (variant != "z_unfrozen")
    assert torch.equal(before_projection, encoder.to_latent.weight) == (variant != "z_unfrozen")
    assert variant_config(variant)["table5"]["finetune"]["encoder_trainable"] == (
        variant == "z_unfrozen"
    )


def test_variant_paths_are_separate_from_original_and_each_other(tmp_path) -> None:
    directories = [variant_paths(tmp_path, "M00", variant, 0) for variant in
                   ("cls_frozen", "z_unfrozen", "z_frozen")]
    assert len({path for pair in directories for path in pair}) == 6
    assert all("finetune_variants" in path.parts for pair in directories for path in pair)


def test_selection_uses_both_conditions_test_means_and_fixed_tie_order() -> None:
    means = {
        "cls_unfrozen": {"M00": 79.0, "M11": 80.0},
        "z_unfrozen": {"M00": 82.0, "M11": 79.0},
        "cls_frozen": {"M00": 78.0, "M11": 82.0},
        "z_frozen": {"M00": 79.0, "M11": 80.0},
    }
    assert select_variant(means) == "z_unfrozen"
    means["cls_unfrozen"] = {"M00": 80.5, "M11": 80.5}
    assert select_variant(means) == "cls_unfrozen"
    with pytest.raises(ValueError, match="four"):
        select_variant({"cls_unfrozen": means["cls_unfrozen"]})


def test_integrated_tables_show_common_selected_path_and_all_runs(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    data = SimpleNamespace(root=prepared)

    def table4_score(_data, _output, _stage, _corpus, _condition, seed):
        return {"metrics_percent": {"ACC": 70 + seed, "NMI": 60 + seed,
                                     "AMI": 50 + seed}}

    def table5_score(_data, _output, condition, variant, seed):
        base = {"cls_unfrozen": 79, "z_unfrozen": 83,
                "cls_frozen": 78, "z_frozen": 80}[variant]
        return {"test_accuracy_percent": base + seed / 10 + (condition == "M11")}

    monkeypatch.setattr(reporting, "_read_score", table4_score)
    monkeypatch.setattr(reporting, "_verified_score", table5_score)
    output = tmp_path / "results"
    paths = reporting.render_integrated_tables(data, output)
    main = paths[1].read_text(encoding="utf-8")
    appendix = paths[2].read_text(encoding="utf-8")
    statistics = json.loads((output / "table_statistics.json").read_text(encoding="utf-8"))
    assert "Selected classifier path: **z_unfrozen**" in main
    assert "test-based selection is exploratory" in main
    assert "ChemoMAE(M00) | — | — | 83.20" in main
    assert "ChemoMAE(M11) | — | — | 84.20" in main
    assert all(variant in appendix for variant in reporting.SELECTION_ORDER)
    assert "ChemoMAE(M00) | 72.00 ±" in paths[0].read_text(encoding="utf-8")
    assert statistics["table5_selected_variant"] == "z_unfrozen"
    assert statistics["table5_variants"]["z_unfrozen"]["M00"]["accuracy_percent"] == [
        83, 83.1, 83.2, 83.3, 83.4,
    ]
    assert reporting.render_integrated_tables(data, output) == paths
