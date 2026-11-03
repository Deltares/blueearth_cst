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


# ---------------------------------------------------------------------------
# The staged (index) check.
#
# A commit records the INDEX. Until 2026-09-06 the pre-commit hook selected
# staged filenames and the checker then read working-tree files, so the two
# could disagree and the local check passed on a commit that carried outputs.
# These tests pin both directions of that disagreement.
# ---------------------------------------------------------------------------

DIRTY_NB = {
    "cells": [
        {
            "cell_type": "code",
            "source": ["1+1"],
            "outputs": [{"output_type": "stream", "text": ["2"]}],
            "execution_count": 3,
        }
    ],
    "metadata": {"kernelspec": {"name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

CLEAN_NB = {
    "cells": [
        {
            "cell_type": "code",
            "source": ["1+1"],
            "outputs": [],
            "execution_count": None,
        }
    ],
    "metadata": {"kernelspec": {"name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _git(repo, *args):
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True, text=True
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo whose notebook name contains a space, as ours do."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "docs" / "notebooks").mkdir(parents=True)
    # Point the module's repo-relative git calls at the throwaway repo.
    monkeypatch.setattr(no, "REPO_ROOT", tmp_path)
    return tmp_path


def _write(repo, rel, notebook):
    (repo / rel).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def test_staged_check_rejects_outputs_left_in_the_index(repo, capsys):
    """THE BYPASS: stage with outputs, strip the working copy, do not re-stage.

    The working tree is clean, so a working-tree check says "ok" while the
    commit would still carry the outputs.
    """
    rel = "docs/notebooks/Climate Stress Test.ipynb"
    _write(repo, rel, DIRTY_NB)
    _git(repo, "add", "--", rel)
    _write(repo, rel, CLEAN_NB)  # stripped, deliberately NOT re-staged

    # The old behaviour, kept here as the contrast: the working tree passes.
    assert no.cells_with_outputs(json.loads((repo / rel).read_text())) == []

    assert no._check_staged([]) == 1
    err = capsys.readouterr().err
    assert "Climate Stress Test.ipynb" in err
    assert "git add" in err  # the divergence note explains why stripping missed


def test_staged_check_ignores_outputs_that_are_only_in_the_working_tree(repo, capsys):
    """The reverse direction must NOT reject.

    The hook checks staged notebooks precisely so a working copy you are
    running does not block an unrelated commit.
    """
    rel = "docs/notebooks/Model building.ipynb"
    _write(repo, rel, CLEAN_NB)
    _git(repo, "add", "--", rel)
    _write(repo, rel, DIRTY_NB)  # actively running it; not staged

    assert no._check_staged([]) == 0
    assert "working copy differs" in capsys.readouterr().out


def test_staged_check_passes_when_nothing_is_staged(repo, capsys):
    assert no._check_staged([]) == 0
    assert "no notebooks staged" in capsys.readouterr().out


def test_staged_check_errors_on_a_path_the_index_does_not_hold(repo, capsys):
    """A named path with no blob is an error, never a silent skip.

    Same rule the working-tree mode already enforced, restated in index terms:
    this is what a word-split filename looks like from the index side.
    """
    assert no._check_staged(["docs/notebooks/Climate"]) == 2
    assert "word-split" in capsys.readouterr().err


def test_staged_notebooks_finds_names_containing_spaces(repo):
    """The list is NUL-delimited and never crosses a shell boundary."""
    rel = "docs/notebooks/Climate Stress Test.ipynb"
    _write(repo, rel, CLEAN_NB)
    _git(repo, "add", "--", rel)
    assert no.staged_notebooks(repo) == [rel]
