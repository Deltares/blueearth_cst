"""Full publication staging keeps canonical content and offline navigation intact."""

import hashlib
import json
import re
from pathlib import Path
from shutil import copyfile

import pytest
import yaml

from dev.scripts import build_publications
from dev.scripts.build_publications import stage_publications
from dev.scripts.stage_site_guides import stage_guides

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication_source(tmp_path: Path) -> Path:
    """Pair the declared six-source appendix and interactive asset with two chapters."""
    source = tmp_path / "source"
    appendix_path = "docs/technical-manual/local-source-appendix.yml"
    appendix = yaml.safe_load((REPO / appendix_path).read_text(encoding="utf-8"))
    paths = [
        appendix_path,
        "docs/guide/assets/shape-of-a-run.html",
        "docs/guide/outputs.md",
        "docs/guide/new-case.md",
        "blueearth_cst/indicators/climate/README.md",
    ]
    paths += [entry["path"] for entry in appendix["external_evidence"]]
    paths += [s["path"] for s in appendix["sources"]]
    for relative in paths:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = (
            REPO / "tests/fixtures/docs-pilot/shape-of-a-run.html"
            if relative == "docs/guide/assets/shape-of-a-run.html"
            else REPO / relative
        )
        copyfile(original, target)
    manifest = {
        "schema_version": 1,
        "repository_url": "https://example.org/repository/blob/revision/",
        "format": "docs/publication/format.yml",
        "asset": "docs/guide/assets/shape-of-a-run.html",
        "appendix_manifest": appendix_path,
        "publications": [
            {
                "id": "user-guide",
                "title": "User guide",
                "chapters": ["docs/guide/quick-start.md"],
                "include_appendices": False,
            },
            {
                "id": "technical-manual",
                "title": "Technical manual",
                "chapters": ["docs/technical-manual/indicator-requests.md"],
                "include_appendices": True,
            },
        ],
    }
    publication = source / "docs/publication"
    publication.mkdir(parents=True)
    (publication / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (publication / "format.yml").write_text(
        "format:\n  html:\n    embed-resources: true\n  typst: default\n",
        encoding="utf-8",
    )
    (source / "docs/guide/quick-start.md").write_text(
        "---\ntitle: Start\n---\n\n## Reading\n\n"
        "[Requests](../technical-manual/indicator-requests.md#standalone-definitions)\n"
        "[Default](../../blueearth_cst/indicators/sets/default/README.md)\n"
        "[Again](#reading-1)\n\n## Reading\n\n"
        "[`registry.py`](../technical-manual/indicator-requests.md#standalone-definitions)\n"
        "[Wrapped\nlabel](../technical-manual/indicator-requests.md#standalone-definitions)\n"
        "`[Example](does-not-exist.md)`\n"
        "[Reference][requests]\n\n[requests]: ../technical-manual/indicator-requests.md#standalone-definitions\n"
        '<iframe src="assets/shape-of-a-run.html" title="Diagram"></iframe>\n'
        "[Open the interactive diagram](assets/shape-of-a-run.html) if needed.\n"
        '```python\nprint("DO-NOT-EXECUTE")\n[Example](does-not-exist.md)\n```\n',
        encoding="utf-8",
    )
    (source / "docs/technical-manual/indicator-requests.md").write_text(
        "---\ntitle: Requests\nformat: html\n---\n\n"
        "## Standalone definitions\n\n```yaml\noperations: [indicators]\n```\n\n[Start](../guide/quick-start.md#reading)\n"
        "[Module](../../blueearth_cst/indicators/README.md#add-a-calculation)\n",
        encoding="utf-8",
    )
    wrapper = source / "docs/site/toolbox-reference/workflow-simulate-system.qmd"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "## Configuration {#configuration}\n\n[Requests](../../technical-manual/indicator-requests.md#standalone-definitions)\n",
        encoding="utf-8",
    )
    return source


def test_static_illustrations_are_bundled_for_site_and_manual(publication_source: Path):
    manifest_path = publication_source / "docs/publication/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["asset"] = "docs/guide/assets/workflow-overview.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    asset = publication_source / manifest["asset"]
    asset.write_bytes(b"static-image-fixture")
    chapter = publication_source / "docs/guide/quick-start.md"
    chapter.write_text(
        "---\ntitle: Example\n---\n\n## Reading\n\n"
        '<img src="assets/workflow-overview.png" alt="Workflow">\n',
        encoding="utf-8",
    )
    generated, hashes = build_publications.prepare_publications(publication_source)
    assert generated["guide/assets/workflow-overview.png"] == asset.read_bytes()
    assert generated["manuals/assets/workflow-overview.png"] == asset.read_bytes()
    assert b'src="assets/workflow-overview.png"' in generated["guide/quick-start.qmd"]
    assert manifest["asset"] in hashes


def test_manual_illustrations_are_bundled_offline(publication_source: Path) -> None:
    asset = publication_source / "docs/technical-manual/assets/processes.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"manual-image-fixture")
    chapter = publication_source / "docs/technical-manual/indicator-requests.md"
    with chapter.open("a", encoding="utf-8") as stream:
        stream.write('\n![Processes](assets/processes.png){fig-alt="Water fluxes"}\n')
    generated, hashes = build_publications.prepare_publications(publication_source)
    assert generated["technical-manual/assets/processes.png"] == asset.read_bytes()
    assert (
        generated["manuals/assets/technical-manual/processes.png"] == asset.read_bytes()
    )
    assert (
        b"](assets/processes.png)"
        in generated["technical-manual/indicator-requests.qmd"]
    )
    assert (
        b"](assets/technical-manual/processes.png)"
        in generated["manuals/technical-manual.qmd"]
    )
    assert "docs/technical-manual/assets/processes.png" in hashes
    stage_guides(publication_source / "docs")
    assert (
        publication_source / "docs/site/technical-manual/assets/processes.png"
    ).read_bytes() == asset.read_bytes()


def test_complete_sources_rebased_offline_and_refresh(
    publication_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "build"
    before = {p: p.read_bytes() for p in publication_source.rglob("*") if p.is_file()}
    hashes = stage_publications(publication_source, output)
    user = (output / "manuals/user-guide.qmd").read_text(encoding="utf-8")
    technical = (output / "manuals/technical-manual.qmd").read_text(encoding="utf-8")
    assert (
        "technical-manual.html#technical-manual-indicator-requests--standalone-definitions"
        in user
    )
    assert "technical-manual.html#indicator-default" in user
    assert "[Again](#guide-quick-start--reading-1)" in user
    assert "user-guide.html#guide-quick-start--reading" in technical
    assert "(#indicator-module--add-a-calculation)" in technical
    assert "[Example](does-not-exist.md)" in user
    assert (
        "[`registry.py`](technical-manual.html#technical-manual-indicator-requests--standalone-definitions)"
        in user
    )
    assert "`[Example](does-not-exist.md)`" in user
    assert (
        "[Wrapped\nlabel](technical-manual.html#technical-manual-indicator-requests--standalone-definitions)"
        in user
    )
    assert (
        "{#technical-manual-indicator-requests--standalone-definitions}\n\n```yaml\n"
        in technical
    )
    assert "data:text/html;base64," in user
    assert 'when-format="html"' in user
    metadata = yaml.safe_load(technical.split("---")[1])
    assert set(metadata["format"]) == {"html", "typst"}
    assert metadata["execute"] == {"enabled": False}
    for document in (user, technical):
        assert yaml.safe_load(document.split("---")[1])["format-links"] == [
            "html",
            {"format": "typst", "text": "PDF", "icon": "file-pdf"},
        ]
    appendix = yaml.safe_load(
        (
            publication_source / "docs/technical-manual/local-source-appendix.yml"
        ).read_text()
    )
    for entry in appendix["sources"]:
        assert (output / f"technical-manual/appendices/{entry['anchor']}.qmd").is_file()
        assert f"{{#{entry['anchor']}}}" in technical
        # Complete source, including the content after the pilot's excerpt stop.
        original = (
            (publication_source / entry["path"]).read_text(encoding="utf-8").strip()
        )
        plain_original = re.sub(
            r"\[([^]]+)\]\([^)]+\)", r"\1", original.splitlines()[-1]
        )
        plain_technical = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", technical)
        plain_technical = plain_technical.replace(" (repository/online reference)", "")
        assert plain_original in plain_technical
    assert "## Interpreting return levels" in technical
    assert "repository/online reference" in technical
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(
        hashlib.sha256((publication_source / p).read_bytes()).hexdigest() == h
        for p, h in hashes.items()
    )
    old_output = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    stage_publications(publication_source, output)
    assert all(p.read_bytes() == data for p, data in old_output.items())
    source = publication_source / "docs/guide/quick-start.md"
    source.write_text(
        source.read_text(encoding="utf-8") + "\nREFRESH-MARKER\n", encoding="utf-8"
    )
    (output / "manuals/user-guide.qmd").write_text("GENERATED EDIT", encoding="utf-8")
    refreshed = stage_publications(publication_source, output)
    assert refreshed["docs/guide/quick-start.md"] != hashes["docs/guide/quick-start.md"]
    assert "REFRESH-MARKER" in (output / "manuals/user-guide.qmd").read_text()


def test_site_hook_stages_full_manifest_preserving_wrappers(
    publication_source: Path,
) -> None:
    wrapper = (
        publication_source / "docs/site/toolbox-reference/workflow-simulate-system.qmd"
    )
    before = wrapper.read_bytes()
    stage_guides(publication_source / "docs")
    site = publication_source / "docs/site"
    assert (site / "guide/quick-start.qmd").is_file()
    assert (site / "technical-manual/indicator-requests.qmd").is_file()
    assert (site / "technical-manual/appendices/indicator-crop-water.qmd").is_file()
    assert wrapper.read_bytes() == before
    mapping = json.loads((site / "technical-manual/publication-links.json").read_text())
    assert (
        mapping["toolbox-reference/workflow-simulate-system.qmd"][
            "../../technical-manual/indicator-requests.md#standalone-definitions"
        ]
        == "../technical-manual/indicator-requests.html#standalone-definitions"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "docs/guide/quick-start.md",
        "blueearth_cst/indicators/sets/crop_water/README.md",
        "docs/guide/assets/shape-of-a-run.html",
    ],
)
def test_missing_inputs_fail_before_refresh(
    publication_source: Path, tmp_path: Path, missing: str
) -> None:
    output = tmp_path / "build"
    stage_publications(publication_source, output)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    (publication_source / missing).unlink()
    with pytest.raises(FileNotFoundError, match="publication source"):
        stage_publications(publication_source, output)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_missing_anchor_refused(publication_source: Path, tmp_path: Path) -> None:
    source = publication_source / "docs/guide/quick-start.md"
    source.write_text(
        source.read_text().replace("#reading-1", "#missing"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Missing anchor"):
        stage_publications(publication_source, tmp_path / "build")
    assert not (tmp_path / "build").exists()


def test_output_cannot_modify_source(publication_source: Path) -> None:
    with pytest.raises(ValueError, match="executing checkout"):
        stage_publications(publication_source, REPO / "docs/site")


def test_render_disables_execution_for_both_formats(
    publication_source: Path, tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "build"
    stage_publications(publication_source, output)
    calls = []
    monkeypatch.setattr(build_publications.shutil, "which", lambda name: "quarto")
    monkeypatch.setattr(
        build_publications.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    build_publications.render_publications(output)
    assert len(calls) == 4
    assert {command[-2] for command, _ in calls} == {"html", "typst"}
    assert all(
        command[-1] == "--no-execute" and kwargs["check"] for command, kwargs in calls
    )


@pytest.mark.parametrize(
    "relative",
    [
        "guide/retired.qmd",
        "manuals/retired.pdf",
        "technical-manual/appendices/retired.html",
    ],
)
def test_stale_generated_products_refuse_without_deleting(
    publication_source: Path, tmp_path: Path, relative: str
) -> None:
    output = tmp_path / "build"
    stage_publications(publication_source, output)
    stale = output / relative
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("preserve for explicit cleanup", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale generated publication file"):
        stage_publications(publication_source, output)
    assert stale.read_text() == "preserve for explicit cleanup"
