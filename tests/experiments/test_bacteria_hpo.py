"""Small CPU checks for the Bacteria-ID five-fold Table 5 protocol."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from chemomae.models.chemo_mae import ChemoMAE

from wood_degradation_map.experiments import bacteria_hpo
from wood_degradation_map.experiments.bacteria_hpo import (
    AUG_MODES, BATCH_SIZES, ENCODER_LRS, HEAD_LRS, WEIGHT_DECAYS,
    FineTuneData, FineTuneSetting, classifier_and_optimizer, cv_folds,
    fit_classifier, full_finetune_data, set_encoder_updates,
    setting_from_record, suggest_setting,
)

if TYPE_CHECKING:
    import optuna


def _model() -> ChemoMAE:
    return ChemoMAE(
        seq_len=1000, n_patches=20, d_model=256, nhead=8, num_layers=8,
        dim_feedforward=512, dropout=0.0, latent_dim=128,
        latent_normalize=True, decoder_num_layers=1, n_mask=10,
    )


class _Trial:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values
        self.asked: list[str] = []

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.asked.append(name)
        value = self.values[name]
        assert low <= value <= high
        return value


def test_search_space_omits_encoder_lr_at_k_zero() -> None:
    trial = _Trial({
        "k": 0, "head_lr_index": 9, "batch_index": 4,
        "aug_index": 3, "weight_decay_index": 4,
    })
    setting = suggest_setting(trial)
    assert setting == FineTuneSetting(
        0, HEAD_LRS[-1], None, BATCH_SIZES[-1], *AUG_MODES[-1], WEIGHT_DECAYS[-1],
    )
    assert "encoder_lr_index" not in trial.asked
    assert setting_from_record(setting.__dict__) == setting
    with pytest.raises(ValueError, match="search space"):
        FineTuneSetting(
            0, HEAD_LRS[0], ENCODER_LRS[0], BATCH_SIZES[0],
            False, False, WEIGHT_DECAYS[0],
        )


def test_duplicate_fallback_selects_a_new_setting() -> None:
    used = FineTuneSetting(
        0, HEAD_LRS[0], None, 8, False, False, 0.0,
    )
    assert bacteria_hpo._unseen_random_setting({used}, 123) != used


@pytest.mark.parametrize("k", [0, 3, 8])
def test_first_five_epochs_freeze_encoder_then_update_last_k(k: int) -> None:
    setting = FineTuneSetting(
        k, HEAD_LRS[0], ENCODER_LRS[2] if k else None,
        16, False, False, WEIGHT_DECAYS[0],
    )
    model, optimizer = classifier_and_optimizer(_model(), 0, setting)
    layers = model.encoder.encoder.layers
    assert len(optimizer.param_groups) == (1 if k == 0 else 2)
    assert optimizer.param_groups[0]["lr"] == setting.head_lr
    if k:
        assert optimizer.param_groups[1]["lr"] == setting.encoder_lr
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.norm.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    set_encoder_updates(model, k, enabled=True)
    for index, layer in enumerate(layers):
        assert all(parameter.requires_grad == (index >= 8 - k)
                   for parameter in layer.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.patch_proj.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.to_latent.parameters())
    assert [group["lr"] for group in optimizer.param_groups] == (
        [setting.head_lr, setting.encoder_lr] if k else [setting.head_lr]
    )


def test_all_five_folds_and_final_training_use_correct_rows() -> None:
    labels = np.repeat(np.arange(30, dtype=np.int64), 100)
    spectra = np.zeros((3000, 1000), dtype=np.float32)
    spectra[:, 0] = np.arange(3000)
    data = SimpleNamespace(
        spectra={"finetune": spectra}, labels={"finetune": labels},
    )
    folds = cv_folds(data)
    assert len(folds) == 5
    held_out: list[int] = []
    for fold in folds:
        assert len(fold.train_x) == 2400
        assert len(fold.validation_x) == 600
        assert torch.equal(torch.bincount(fold.train_y, minlength=30),
                           torch.full((30,), 80))
        assert torch.equal(torch.bincount(fold.validation_y, minlength=30),
                           torch.full((30,), 20))
        held_out.extend(fold.validation_x[:, 0].int().tolist())
    assert sorted(held_out) == list(range(3000))
    final = full_finetune_data(data)
    assert len(final.train_x) == 3000
    assert final.validation_x is None


def test_fixed_epoch_score_is_last_epoch_not_best_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(123)
    spectra = torch.randn((2, 1000), generator=generator)
    spectra = (spectra - spectra.mean(dim=1, keepdim=True)) / spectra.std(dim=1, keepdim=True)
    labels = torch.tensor([0, 1])
    train = FineTuneData(spectra, labels, spectra.clone(), labels.clone())
    setting = FineTuneSetting(
        0, HEAD_LRS[0], None, 16, False, False, WEIGHT_DECAYS[0],
    )
    scores = iter((0.9, 0.2))
    monkeypatch.setattr(bacteria_hpo, "_accuracy", lambda *_args, **_kwargs: next(scores))
    result = fit_classifier(
        _model(), train, setting, 0, torch.device("cpu"), epochs=2,
    )
    assert len(result.history) == 2
    assert result.history[0]["validation_accuracy"] == 0.9
    assert result.epoch50_validation_accuracy == result.history[-1]["validation_accuracy"] == 0.2
    assert result.final_weights is None


@pytest.fixture
def cutoff_study() -> optuna.study.Study:
    import optuna

    study = optuna.create_study(direction="maximize")
    for index in range(64):
        if index == 10:
            study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
        setting = FineTuneSetting(
            0, HEAD_LRS[index % 10], None, BATCH_SIZES[(index // 10) % 5],
            index >= 50, False, 0.0,
        )
        score = 0.99 if index >= 60 else (0.94 if index in (5, 59) else 0.8)
        study.add_trial(optuna.trial.create_trial(
            value=score, user_attrs={"setting": asdict(setting), "fold_accuracies": [score] * 5},
        ))
    for state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING):
        study.add_trial(optuna.trial.create_trial(state=state))
    return study


def test_cutoff_excludes_later_and_incomplete_trials_and_keeps_first_tie(
    cutoff_study: optuna.study.Study,
) -> None:
    selected = bacteria_hpo._selection_record(cutoff_study, {"trials": 100}, 60)
    assert selected["selected_trial_number"] == 5
    assert selected["cv_mean_accuracy"] == 0.94
    assert selected["schema_version"] == 2
    assert selected["selection_policy"]["trial_numbers"] == [
        number for number in range(61) if number != 10
    ]
    assert selected["selection_policy"]["completed_settings"] == 60
    assert selected["selection_policy"]["planned_settings"] == 100
    with pytest.raises(ValueError, match="requires 100"):
        bacteria_hpo._selection_record(cutoff_study, {"trials": 100})
    with pytest.raises(ValueError, match="at least 65"):
        bacteria_hpo._selection_record(cutoff_study, {"trials": 100}, 65)


@pytest.mark.parametrize("count", [29, 101, True, 60.0])
def test_cutoff_rejects_invalid_count(cutoff_study: optuna.study.Study, count: object) -> None:
    with pytest.raises(ValueError, match="between 30 and 100"):
        bacteria_hpo._selection_trials(cutoff_study, count)


def test_legacy_selection_accepts_only_known_code_and_matching_protocol() -> None:
    previous = {
        "code_sha256": bacteria_hpo.EARLY_SELECTION_COMPATIBLE_CODE,
        "trials": 100, "folds": 5, "processed_manifest_sha256": "data-v1",
        "torch_version": "original-version",
    }
    current = {**previous, "code_sha256": "new-selection-code"}
    assert bacteria_hpo._matching_study_protocol(
        previous, current, allow_legacy_selection=True,
    )
    assert not bacteria_hpo._matching_study_protocol(
        previous, current, allow_legacy_selection=False,
    )
    for key, changed in (
        ("code_sha256", "unknown-code"), ("trials", 60), ("folds", 4),
        ("processed_manifest_sha256", "other-data"), ("torch_version", "new-version"),
    ):
        assert not bacteria_hpo._matching_study_protocol(
            {**previous, key: changed}, current, allow_legacy_selection=True,
        )


def test_cutoff_selection_reloads_without_rewriting_original_study(
    cutoff_study: optuna.study.Study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {"code_sha256": bacteria_hpo.EARLY_SELECTION_COMPATIBLE_CODE, "trials": 100}
    current = {**protocol, "code_sha256": "new-selection-code"}
    cutoff_study.set_user_attr("protocol", protocol)

    def open_study(
        output_dir: Path, contract: dict[str, object], *, create: bool,
        allow_legacy_selection: bool = False,
    ) -> optuna.study.Study:
        assert output_dir == tmp_path and not create
        assert bacteria_hpo._matching_study_protocol(
            protocol, contract, allow_legacy_selection=allow_legacy_selection,
        )
        return cutoff_study

    monkeypatch.setattr(bacteria_hpo, "_open_study", open_study)
    monkeypatch.setattr(bacteria_hpo, "_study_contract", lambda *_args: current)
    trials_before = cutoff_study.trials
    selected = bacteria_hpo.select_completed(
        SimpleNamespace(), tmp_path, tmp_path, completed_trials=60,
    )
    loaded, setting, original_contract = bacteria_hpo._selected(tmp_path, current)
    assert loaded == selected
    assert setting == setting_from_record(selected["setting"])
    assert original_contract == protocol
    assert selected["study_contract_sha256"] == bacteria_hpo._contract_hash(protocol)
    assert cutoff_study.user_attrs["protocol"] == protocol
    assert cutoff_study.trials == trials_before
    assert bacteria_hpo.select_completed(
        SimpleNamespace(), tmp_path, tmp_path, completed_trials=60,
    ) == selected
    with pytest.raises(ValueError, match="Existing HPO selection differs"):
        bacteria_hpo.select_completed(SimpleNamespace(), tmp_path, tmp_path, completed_trials=64)
    assert bacteria_hpo._selected(tmp_path, current)[0] == selected
    with pytest.raises(ValueError, match="finalized at a requested cutoff"):
        bacteria_hpo.search(SimpleNamespace(), tmp_path, tmp_path, torch.device("cpu"))

    selected["selection_policy"]["trial_numbers"][0] = 999
    bacteria_hpo._replace_json(tmp_path / "selected.json", selected)
    with pytest.raises(ValueError, match="differs from the completed"):
        bacteria_hpo._selected(tmp_path, current)
