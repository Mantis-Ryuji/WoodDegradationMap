"""Successful and failed replacement of generated output directories."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from wood_degradation_map.experiments.artifact_output import publish_directory


@pytest.mark.parametrize("fail", [False, True])
def test_replacement_removes_stale_files_only_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("original")
    with TemporaryDirectory(dir=tmp_path) as temporary:
        stage = Path(temporary) / "new"
        stage.mkdir()
        (stage / "new.txt").write_text("regenerated")
        rename = Path.rename

        def fail_publication(path: Path, target: Path) -> Path:
            if path == stage:
                raise OSError("fixture publication failure")
            return rename(path, target)

        if fail:
            monkeypatch.setattr(Path, "rename", fail_publication)
            with pytest.raises(OSError, match="fixture publication"):
                publish_directory(stage, output, root=tmp_path)
            assert (output / "old.txt").read_text() == "original"
            assert not (output / "new.txt").exists()
        else:
            publish_directory(stage, output, root=tmp_path)
            assert not (output / "old.txt").exists()
            assert (output / "new.txt").read_text() == "regenerated"


def test_replacement_rejects_target_outside_declared_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    stage = root / "stage"
    stage.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        publish_directory(stage, tmp_path / "outside", root=root)
    assert stage.exists()
