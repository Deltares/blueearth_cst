"""Tests for the WF2 log-gather concatenation (blueearth_cst/shared/merge_logs.py)."""
import os
import sys

from blueearth_cst.shared.merge_logs import merge_logs  # noqa: E402


def test_merge_concatenates_in_order_with_marker(tmp_path):
    parts = []
    for name in ("modelA", "modelB"):
        p = tmp_path / "_parts" / f"{name}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"line from {name}\n", encoding="utf-8")
        parts.append(str(p))
    out = tmp_path / "merged.log"
    merge_logs(parts, str(out))
    text = out.read_text(encoding="utf-8")
    assert "# merged log: merged (2 per-model parts)" in text
    assert "line from modelA" in text and "line from modelB" in text
    assert text.index("modelA") < text.index("modelB")  # order preserved


def test_merge_creates_parent_dir(tmp_path):
    part = tmp_path / "p.log"
    part.write_text("x\n", encoding="utf-8")
    out = tmp_path / "nested" / "deep" / "merged.log"
    merge_logs([str(part)], str(out))
    assert out.exists()


def test_merge_notes_missing_part(tmp_path):
    out = tmp_path / "merged.log"
    merge_logs([str(tmp_path / "absent" / "modelX.log")], str(out))
    assert "no log produced for modelX" in out.read_text(encoding="utf-8")
