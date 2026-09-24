"""Publish regenerated artifacts inside a fixed experiment directory."""

from __future__ import annotations

from pathlib import Path
import stat


def publish_directory(stage: Path, output: Path, *, root: Path) -> None:
    """Replace an output after generation, rolling back a failed rename.

    The caller owns the staging TemporaryDirectory; it removes the previous
    output only after publication succeeds. Linked output trees are rejected.
    """
    root = root.resolve()
    for path in (stage, output):
        if (path.absolute() != path.resolve() or not path.resolve().is_relative_to(root)
                or path.resolve() == root):
            raise ValueError(f"Artifact replacement escapes its root: {path}")
    stage, output = stage.resolve(), output.resolve()
    if stage.is_relative_to(output) or output.is_relative_to(stage):
        raise ValueError("Staging and output directories must be separate")
    previous = stage.parent / "previous-output"
    if previous.exists():
        raise FileExistsError(previous)
    if output.exists():
        for path in (output, *output.rglob("*")):
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            if (path.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    or not path.resolve().is_relative_to(output)):
                raise ValueError(f"Refuse replacement through a linked artifact: {path}")
        output.rename(previous)
    try:
        stage.rename(output)
    except OSError:
        if previous.exists():
            previous.rename(output)
        raise
