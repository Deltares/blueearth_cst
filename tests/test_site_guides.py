"""The site consumes independent guide sources without changing them."""

from pathlib import Path

import pytest

from dev.scripts.stage_site_guides import GUIDES, stage_guides


def test_default_staging_covers_the_canonical_guide_inventory():
    source_dir = Path(__file__).resolve().parents[1] / "docs/guide"
    assert set(GUIDES) == {source.stem for source in source_dir.glob("*.md")}


def test_staging_refreshes_links_and_assets_without_changing_sources(tmp_path: Path):
    source_dir = tmp_path / "guide"
    (source_dir / "assets").mkdir(parents=True)
    original = "[Setup](../site/setup/install.md) [Next](outputs.md#results)\n"
    for name in GUIDES:
        (source_dir / f"{name}.md").write_text(original, encoding="utf-8")
    asset = source_dir / "assets/shape-of-a-run.html"
    asset.write_text("<html>diagram</html>", encoding="utf-8")

    stage_guides(tmp_path)

    target_dir = tmp_path / "site/guide"
    for name in GUIDES:
        assert (source_dir / f"{name}.md").read_text(encoding="utf-8") == original
        assert (target_dir / f"{name}.md").read_text(encoding="utf-8") == (
            "[Setup](../setup/install.md) [Next](outputs.md#results)\n"
        )
    assert (
        target_dir / "assets/shape-of-a-run.html"
    ).read_bytes() == asset.read_bytes()

    (source_dir / "outputs.md").write_text(
        "Updated output guidance\n", encoding="utf-8"
    )
    stage_guides(tmp_path)
    assert (target_dir / "outputs.md").read_text(encoding="utf-8") == (
        "Updated output guidance\n"
    )


def test_missing_source_fails_before_staging(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        stage_guides(tmp_path)
    assert not (tmp_path / "site").exists()
