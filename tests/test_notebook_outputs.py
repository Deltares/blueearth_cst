"""The tracked notebooks must not carry rendered outputs.

THIS TEST IS THE GATE, not the pre-commit hook beside it. `core.hooksPath` is a
per-clone setting that cloning does not install (AGENTS.md says so explicitly,
next to the same caveat for the ruff pre-push hook), so a hook protects only the
machines that opted in. This runs on both CI legs and on every checkout.

Why it exists: a notebook carrying outputs embeds each figure as base64 PNG and
therefore does not delta-compress, so every edit mints a fresh multi-megabyte
blob that stays in history. Measured 2026-08-14, the day before this landed:
`Model building.ipynb` was 6.43 MB with 82 blob versions already in history, and
a rename sweep that rewrote three short strings inside the notebooks turned a
few hundred KB of text change into a 7.1 MB push. Stripping the three took them
from 8.8 MB to 0.08 MB.

This REVERSES the 2026-08-13 owner ruling (fao assessment §6.3 option C,
"commit outputs with a dated banner"). What that ruling was buying -- a reader
seeing the results without running the pipeline -- is preserved by publishing
rendered copies as Artifacts instead; see docs/notebooks/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "docs" / "notebooks"

sys.path.insert(0, str(REPO_ROOT / "dev" / "scripts"))
import notebook_outputs as no  # noqa: E402


def _notebooks() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_there_are_notebooks_to_check():
    """A glob that matches nothing must fail, not pass vacuously.

    The same defect class the 2026-08-14 rename hit: three tests globbed
    `Snakefile_*`, matched nothing after the rename, and kept passing while
    checking an empty set.
    """
    assert _notebooks(), f"no notebooks under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.name)
def test_notebook_carries_no_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    offenders = no.cells_with_outputs(notebook)
    assert not offenders, (
        f"{path.name}: {len(offenders)} cell(s) carry outputs or an "
        f"execution_count (cells {offenders[:5]}...). Clear them with "
        "`python dev/scripts/notebook_outputs.py --strip`."
    )


def test_strip_is_idempotent_and_reports_no_change_on_clean_input(tmp_path):
    """The fixer's own contract, proven on a synthetic notebook.

    Layer-1 style: this runs whatever the tracked notebooks happen to contain,
    so the checker's logic stays proven even if `docs/notebooks/` is empty or
    every notebook is already clean.
    """
    dirty = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["1+1"],
                "outputs": [{"x": 1}],
                "execution_count": 3,
            },
            {"cell_type": "markdown", "source": ["# heading"]},
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
    }
    assert no.cells_with_outputs(dirty) == [0]

    assert no.strip(dirty) is True
    assert no.cells_with_outputs(dirty) == []
    # Second pass changes nothing -- so a clean tree never produces a diff.
    assert no.strip(dirty) is False


def test_markdown_cells_are_left_alone(tmp_path):
    """Stripping must not touch prose, which is the half worth keeping."""
    nb = {
        "cells": [{"cell_type": "markdown", "source": ["# Rendered against abc123"]}],
        "metadata": {},
        "nbformat": 4,
    }
    assert no.strip(nb) is False
    assert nb["cells"][0]["source"] == ["# Rendered against abc123"]
