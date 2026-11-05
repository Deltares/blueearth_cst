"""The pilot preserves source ownership, boundaries and source-path compatibility."""

import json
import re
from pathlib import Path
from shutil import copyfile

import pytest
import yaml

from dev.scripts.build_docs_pilot import diagram_fallback, select_text, stage_pilot

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_copy(tmp_path: Path) -> Path:
    """Copy only real pilot inputs, including the linked module contract."""
    manifest = json.loads(
        (
            REPO
            / "tests/fixtures/docs-pilot/source/docs/publishing-pilot/manifest.json"
        ).read_text()
    )
    compatibility = manifest["compatibility"]
    paths = {
        "docs/publishing-pilot/manifest.json",
        "docs/publishing-pilot/format.yml",
        "blueearth_cst/indicators/README.md",
        "docs/technical-manual/indicator-requests.md",
        manifest["asset"],
        *(section["source"] for section in manifest["sections"]),
        *(
            f"{compatibility['fixture']}/{compatibility[key]}"
            for key in ("old", "canonical")
        ),
    }
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frozen = REPO / "tests/fixtures/docs-pilot/source" / relative
        original = frozen if frozen.exists() else REPO / relative
        if relative == manifest["asset"]:
            original = REPO / "tests/fixtures/docs-pilot/shape-of-a-run.html"
        copyfile(original, target)
    return tmp_path


def test_staging_preserves_sources_and_refreshes(source_copy: Path) -> None:
    before = {p: p.read_bytes() for p in source_copy.rglob("*") if p.is_file()}
    output = source_copy / ".tmp/build"
    hashes = stage_pilot(source_copy, output)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert len(hashes) == 10
    assembled = (output / "index.qmd").read_text(encoding="utf-8")
    assert "completed `_engine/simulation.json` record" in assembled
    assert "python scripts/simulate_system.py --config <config> --cores 3" in assembled
    assert "```yaml\n# project_config_simulate_system.yml\n" in assembled
    assert "workflows.simulate_system.config_path" in assembled
    assert "max(ceil(T),10)" in assembled
    assert "Return levels pool blocks instead." in assembled
    assert "native pandas missingness and partial-block weighting" in assembled
    assert "(repository/online reference)" in assembled
    assert "/blueearth_cst/indicators/README.md#add-a-calculation)" in assembled
    assert "The toolbox's registry and calculation scripts define" in assembled
    assert "## Advanced implementation details" not in assembled
    for token in (
        "registry.py",
        "calculations.py",
        "checks.py",
        "ASSUMPTIONS",
        "PARAMETERS",
    ):
        assert token not in assembled
    assert "[Reading the outputs](#reading-an-indicator-set)" in assembled
    assert "### Migrating existing outputs" not in assembled
    assert "## Return levels and their limits" not in assembled
    assert "operation:" not in assembled
    assert "example_subset.yml" not in assembled
    assert (
        "1. In `operations`, select `simulations`, `indicators`, or both." in assembled
    )
    assert (
        "It does not change arithmetic, units, calendars, missingness or grain."
        in assembled
    )
    assert (
        "The values are the same; they are reported under your indicator name."
        in assembled
    )
    assert yaml.safe_load(assembled.split("---")[1])["subtitle"] == (
        "P3 named-files working tree (base 865803b2)"
    )
    canonical = (output / "technical-manual/indicator-requests.qmd").read_text(
        encoding="utf-8"
    )
    assert yaml.safe_load(canonical.split("---")[1])["format-links"] is False
    for name in ("configuration", "reading-an-indicator-set", "default-indicators"):
        assert f"{{#{name}}}" in assembled
    source = source_copy / "blueearth_cst/indicators/sets/default/README.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "## Advanced implementation details",
            "REFRESH-MARKER\n\n## Advanced implementation details",
        ),
        encoding="utf-8",
    )
    (output / "index.qmd").write_text("GENERATED-EDIT", encoding="utf-8")
    stage_pilot(source_copy, output)
    refreshed = (output / "index.qmd").read_text(encoding="utf-8")
    assert "REFRESH-MARKER" in refreshed
    assert "GENERATED-EDIT" not in refreshed


@pytest.mark.parametrize("field", ["source", "asset", "old", "canonical"])
def test_missing_input_fails_without_partial_staging(
    source_copy: Path, field: str
) -> None:
    manifest = json.loads(
        (source_copy / "docs/publishing-pilot/manifest.json").read_text()
    )
    if field == "source":
        relative = manifest["sections"][0]["source"]
    elif field == "asset":
        relative = manifest["asset"]
    else:
        relative = (
            f"{manifest['compatibility']['fixture']}/{manifest['compatibility'][field]}"
        )
    (source_copy / relative).unlink()
    output = source_copy / ".tmp/build"
    with pytest.raises(FileNotFoundError):
        stage_pilot(source_copy, output)
    assert not output.exists()


@pytest.mark.parametrize("text", ["absent", "START\nSTOP\nSTART", "STOP\nSTART"])
def test_boundaries_fail_loudly(text: str) -> None:
    with pytest.raises(ValueError):
        select_text(text, {"start": "START", "stop": "STOP"})


def test_every_manifest_boundary_is_required(source_copy: Path) -> None:
    manifest = json.loads(
        (source_copy / "docs/publishing-pilot/manifest.json").read_text()
    )
    for section in manifest["sections"]:
        original = (source_copy / section["source"]).read_text(encoding="utf-8")
        for boundary in ("start", "stop"):
            if boundary in section:
                with pytest.raises(ValueError, match="Expected one boundary"):
                    select_text(original.replace(section[boundary], "REMOVED"), section)


def test_source_compatibility_before_rendering() -> None:
    fixture = REPO / "tests/fixtures/docs-pilot"
    old = fixture / "docs/site/toolbox-reference/workflow-simulate-system.qmd"
    text = old.read_text(encoding="utf-8")
    assert "## Configuration {#configuration}" in text
    link = re.search(r"\]\(([^)]+)\)", text).group(1)
    path, anchor = link.split("#")
    canonical = (old.parent / path).resolve()
    assert canonical == fixture / "docs/technical-manual/indicator-requests.md"
    assert f"## {anchor.title()}" in canonical.read_text(encoding="utf-8")


def test_static_fallback_covers_svg_workflows_and_all_details() -> None:
    asset = (REPO / "tests/fixtures/docs-pilot/shape-of-a-run.html").read_text(
        encoding="utf-8"
    )
    svg, fallback = diagram_fallback(asset)
    details = re.findall(r'wf\d: \["([^"]+)", "([^"]+)", "([^"]+)"\]', asset)
    assert len(details) == 5
    for label, title, body in details:
        assert f"### {label}: {title}" in fallback
        assert body in fallback
    assert "Dashed arrows show order, not data dependencies." in fallback
    assert "DASHED ARROWS = RUN ORDER, NOT DATA DEPENDENCIES" in svg
    with pytest.raises(ValueError, match="all five"):
        diagram_fallback(asset.replace('data-id="wf4"', 'data-id="unexpected"'))


def test_output_cannot_overwrite_authored_sources(source_copy: Path) -> None:
    with pytest.raises(ValueError, match=".tmp"):
        stage_pilot(source_copy, source_copy / "docs/guide")
