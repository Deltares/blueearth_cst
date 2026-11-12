"""Project layout version stamp (`blueearth_cst.shared.layout_stamp`, rule R18)."""

import json

import pytest

from blueearth_cst.shared import layout_stamp
from blueearth_cst.shared.layout_paths import LAYOUT_VERSION


def test_a_new_folder_is_stamped(tmp_path):
    root = tmp_path / "outputs"
    assert layout_stamp.admit(root) == LAYOUT_VERSION
    assert layout_stamp.read(root) == LAYOUT_VERSION


def test_an_empty_folder_is_stamped(tmp_path):
    layout_stamp.admit(tmp_path)
    assert (tmp_path / layout_stamp.STAMP).exists()


def test_an_unstamped_project_is_layout_1_and_never_backfilled(tmp_path):
    (tmp_path / "models").mkdir()
    assert layout_stamp.admit(tmp_path) == layout_stamp.UNSTAMPED_VERSION
    assert layout_stamp.read(tmp_path) is None


def test_a_different_layout_is_refused_with_the_fix(tmp_path):
    layout_stamp.write(tmp_path, LAYOUT_VERSION + 1)
    with pytest.raises(layout_stamp.LayoutMismatch, match="new folder"):
        layout_stamp.admit(tmp_path)


def test_an_unstamped_project_is_refused_once_the_layout_moves_on(
    tmp_path, monkeypatch
):
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(layout_stamp, "LAYOUT_VERSION", 2)
    with pytest.raises(layout_stamp.LayoutMismatch, match="layout 1"):
        layout_stamp.admit(tmp_path)


def test_an_unreadable_stamp_is_refused(tmp_path):
    path = tmp_path / layout_stamp.STAMP
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"layout_version": "1"}), encoding="utf-8")
    with pytest.raises(layout_stamp.LayoutMismatch, match="unreadable"):
        layout_stamp.admit(tmp_path)


def test_admission_is_idempotent(tmp_path):
    layout_stamp.admit(tmp_path)
    before = (tmp_path / layout_stamp.STAMP).read_bytes()
    layout_stamp.admit(tmp_path)
    assert (tmp_path / layout_stamp.STAMP).read_bytes() == before
