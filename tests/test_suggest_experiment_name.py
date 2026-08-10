"""R07 B8: experiment_name is SUGGESTED once, never generated at run time.

The helper slugifies because ``basename(project_dir)`` is not guaranteed to
satisfy the grammar the workflow enforces (repo-7): ``examples/Gabon`` was live
in six shipped configs, and production project_dir values routinely carry
uppercase, hyphens or spaces. The design's original evidence ("both
gabon260725 and gabon_20260726 already satisfy the grammar") only tested names
that already conformed.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.snake_utils import (  # noqa: E402
    suggest_experiment_name,
    validate_experiment_name,
)

SNAKEDIR = Path(__file__).resolve().parents[1]
CLI = SNAKEDIR / "scripts" / "suggest_experiment_name.py"


@pytest.mark.parametrize(
    "project_dir, expected",
    [
        ("examples/Gabon", "gabon_20260728"),  # the live counterexample
        ("test_case/test_local", "test_local_20260728"),
        ("/mnt/data/My Basin-2024", "my_basin_2024_20260728"),
        (r"C:\runs\Rhine--Upper", "rhine_upper_20260728"),
        ("trailing/slash/", "slash_20260728"),
        ("__leading_junk", "leading_junk_20260728"),
    ],
)
def test_slugification(project_dir, expected):
    assert suggest_experiment_name(project_dir, "20260728") == expected


def test_suggestion_always_satisfies_the_validator():
    """The proposal is passed back through validate_experiment_name, so the
    suggester and the workflow's own gate can never disagree."""
    for pd in ("examples/Gabon", "/mnt/x/A B C", "UPPER", "9lives"):
        name = suggest_experiment_name(pd, "20260728")
        assert validate_experiment_name(name, "/tmp/proj") == name


def test_truncation_keeps_the_date_and_stays_valid():
    name = suggest_experiment_name("x/" + "a" * 200, "20260728")
    assert len(name) <= 64
    assert name.endswith("_20260728")
    assert validate_experiment_name(name, "/tmp/proj") == name


def test_basename_without_alphanumerics_raises():
    with pytest.raises(ValueError, match="no alphanumeric"):
        suggest_experiment_name("/data/---", "20260728")


def _run(cfg, *extra):
    return subprocess.run(
        [sys.executable, str(CLI), str(cfg), "--date", "20260728", *extra],
        capture_output=True,
        text=True,
    )


def _cfg(tmp_path, experiment_name=None):
    # project_dir must be under tmp_path. Since R9 P4 the command RESERVES the
    # name by creating experiments/<id>/, so a repo-relative project_dir here
    # would write real directories into the working tree on every test run --
    # it resurrected `examples/`, retired at R7, before this was caught. The
    # basename still slugifies to `gabon`, which is what these cases are about.
    project_dir = tmp_path / "Gabon"
    project_dir.mkdir(exist_ok=True)
    doc = {
        "project": {"project_dir": str(project_dir).replace("\\", "/")},
        "workflows": {"climate_experiment": {"enabled": True}},
    }
    if experiment_name is not None:
        doc["workflows"]["climate_experiment"]["experiment_name"] = experiment_name
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_cli_writes_when_absent(tmp_path):
    cfg = _cfg(tmp_path)
    res = _run(cfg)
    assert res.returncode == 0, res.stderr
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert doc["workflows"]["climate_experiment"]["experiment_name"] == "gabon_20260728"


def test_cli_refuses_to_overwrite(tmp_path):
    """The experiment name is the directory every wf3 artifact hangs off;
    silently changing it would strand a completed experiment's outputs."""
    cfg = _cfg(tmp_path, experiment_name="already_here")
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg)
    assert res.returncode != 0
    assert "already_here" in res.stderr
    assert cfg.read_text(encoding="utf-8") == before, "config must be untouched"


def test_cli_dry_run_leaves_the_config_alone(tmp_path):
    cfg = _cfg(tmp_path)
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "gabon_20260728"
    assert cfg.read_text(encoding="utf-8") == before


def test_cli_preserves_other_config_content(tmp_path):
    """Round-tripping the YAML must not drop or reorder unrelated keys."""
    cfg = _cfg(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["shared"] = {"clim_historical": "era5"}
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert _run(cfg).returncode == 0
    out = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert out["shared"] == {"clim_historical": "era5"}
    # Compare against the value actually written rather than a literal: the
    # fixture is tmp_path-based since P4 made the command reserve.
    assert out["project"]["project_dir"] == doc["project"]["project_dir"]


def test_cli_dry_run_reports_even_when_a_value_is_set(tmp_path):
    """The point of inspecting first is to SEE what would be proposed; a
    dry-run that only says "refusing" tells you nothing you did not know."""
    cfg = _cfg(tmp_path, experiment_name="already_here")
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "gabon_20260728"
    assert cfg.read_text(encoding="utf-8") == before


def test_cli_refusal_names_what_it_would_have_suggested(tmp_path):
    cfg = _cfg(tmp_path, experiment_name="already_here")
    res = _run(cfg)
    assert res.returncode != 0
    assert "already_here" in res.stderr and "gabon_20260728" in res.stderr


# --- the template must leave the key for this command to fill ------------------


def test_the_template_config_does_not_hardcode_an_experiment_name():
    """The template shipped ``experiment_name: experiment`` while this command
    refuses to overwrite any existing value — so every project copied from it
    landed in the same ``experiments/experiment/`` and the command that exists
    to name the directory could never run. The key must start out ABSENT."""
    template = yaml.safe_load(
        (SNAKEDIR / "config/workflows/snake_config.template.yml").read_text(
            encoding="utf-8"
        )
    )
    section = template["workflows"]["climate_experiment"]
    assert "experiment_name" not in section, (
        "snake_config.template.yml must not set experiment_name: a copied "
        "template would inherit the placeholder, and "
        "suggest_experiment_name.py refuses to overwrite an existing value"
    )


def test_the_test_fixtures_deliberately_KEEP_a_fixed_name():
    """The gate needs a fixed experiments/<name>/ path, so the seed configs
    keep theirs. Only the user-facing template drops the key."""
    # The two seeds no longer share a directory: the baseline seed sits beside
    # the project it writes into, dev_fast stays with the other configs.
    for rel in (
        "test_case/snake_config_model_test.yml",
        "config/workflows/snake_config_dev_fast.yml",
    ):
        doc = yaml.safe_load((SNAKEDIR / rel).read_text(encoding="utf-8"))
        assert doc["workflows"]["climate_experiment"]["experiment_name"]


# --- the write must not destroy the config it edits ----------------------------


def _annotated_cfg(tmp_path, body):
    """A config written as TEXT, so comments and layout survive to be checked."""
    project_dir = tmp_path / "Gabon"
    project_dir.mkdir(exist_ok=True)
    p = tmp_path / "cfg.yml"
    p.write_text(
        "# top-of-file banner\n"
        "project:\n"
        "  project_dir: "
        + str(project_dir).replace("\\", "/")
        + "  # where output goes\n"
        "\n"
        "workflows:\n" + body,
        encoding="utf-8",
    )
    return p


CE_BLOCK = (
    "  climate_experiment:\n"
    "    enabled: true\n"
    "    realizations_num: 2  # trailing comment\n"
    "    stress_test:\n"
    "      temp:\n"
    "        step_num: 1\n"
)


def test_cli_preserves_every_comment_in_the_config(tmp_path):
    """yaml.safe_dump discards comments. The shipped template carries ~110 of
    them and this command is the first thing a new user runs against their
    copy, so a dump would delete the annotations they were just handed."""
    cfg = _annotated_cfg(tmp_path, CE_BLOCK)
    before = cfg.read_text(encoding="utf-8")
    assert _run(cfg).returncode == 0
    after = cfg.read_text(encoding="utf-8")
    for comment in (
        "# top-of-file banner",
        "# where output goes",
        "# trailing comment",
    ):
        assert comment in after, f"{comment!r} was destroyed by the write"
    # Exactly one line added, everything else byte-identical.
    added = [ln for ln in after.splitlines() if ln not in before.splitlines()]
    assert added == ["    experiment_name: gabon_20260728"]


def test_cli_fills_a_bare_key_in_place_keeping_its_comment(tmp_path):
    """`experiment_name:` with no value parses to None, which this command
    treats as unset (its refusal test is `is not None`)."""
    cfg = _annotated_cfg(
        tmp_path,
        "  climate_experiment:\n"
        "    enabled: true\n"
        "    experiment_name:   # fill me in\n"
        "    realizations_num: 2\n",
    )
    assert _run(cfg).returncode == 0
    after = cfg.read_text(encoding="utf-8")
    assert "    experiment_name: gabon_20260728  # fill me in" in after
    assert after.count("experiment_name") == 1, "must fill in place, not duplicate"


def test_the_new_key_lands_below_the_comments_that_document_it(tmp_path):
    """The template heads the block with a comment explaining the key; the
    insertion goes after it, not between the comment and its heading."""
    cfg = _annotated_cfg(
        tmp_path,
        "  climate_experiment:\n"
        "    enabled: true\n"
        "    # experiment_name is left unset on purpose; run the command\n"
        "    realizations_num: 2\n",
    )
    assert _run(cfg).returncode == 0
    lines = cfg.read_text(encoding="utf-8").splitlines()
    assert (
        lines.index("    experiment_name: gabon_20260728")
        == lines.index("    realizations_num: 2") - 1
    )


def test_cli_leaves_unrelated_structure_intact(tmp_path):
    """The reload check is the guarantee; this pins it end to end."""
    cfg = _annotated_cfg(tmp_path, CE_BLOCK)
    before = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert _run(cfg).returncode == 0
    after = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    before["workflows"]["climate_experiment"]["experiment_name"] = "gabon_20260728"
    assert after == before


def test_an_uneditable_config_reserves_nothing(tmp_path):
    """Failing after the reservation would strand an experiments/<id>/ for a
    config the command then could not write to anyway."""
    cfg = _annotated_cfg(tmp_path, "  climate_experiment: {enabled: true}\n")
    before = cfg.read_text(encoding="utf-8")
    res = _run(cfg)
    assert res.returncode == 2
    assert "flow style" in res.stderr and "by hand" in res.stderr
    assert cfg.read_text(encoding="utf-8") == before
    assert not (tmp_path / "Gabon" / "experiments").exists(), "must reserve nothing"


def test_the_shipped_template_is_editable_by_this_command(tmp_path):
    """The template is the file this command is documented against; a layout
    it cannot edit would break the one path a new user is told to take."""
    import shutil

    src = SNAKEDIR / "config/workflows/snake_config.template.yml"
    cfg = tmp_path / "tmpl.yml"
    shutil.copy(src, cfg)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    project_dir = tmp_path / "Gabon"
    project_dir.mkdir()
    text = cfg.read_text(encoding="utf-8").replace(
        doc["project"]["project_dir"], str(project_dir).replace("\\", "/"), 1
    )
    cfg.write_text(text, encoding="utf-8")
    n_comments_before = text.count("#")

    res = _run(cfg)
    assert res.returncode == 0, res.stderr
    after = cfg.read_text(encoding="utf-8")
    assert after.count("#") == n_comments_before
    assert (
        yaml.safe_load(after)["workflows"]["climate_experiment"]["experiment_name"]
        == "gabon_20260728"
    )


def test_a_missing_climate_experiment_block_is_appended(tmp_path):
    """The yaml.safe_dump this replaced created absent blocks via setdefault;
    dropping that would break configs the command used to accept."""
    cfg = _annotated_cfg(
        tmp_path,
        "  model_creation:\n"
        "    enabled: true  # keep me\n"
        "\n"
        "# a trailing comment that belongs to no block\n",
    )
    assert _run(cfg).returncode == 0
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert doc["workflows"]["climate_experiment"] == {
        "experiment_name": "gabon_20260728"
    }
    assert doc["workflows"]["model_creation"] == {"enabled": True}
    text = cfg.read_text(encoding="utf-8")
    assert "# keep me" in text and "belongs to no block" in text
    # The appended block goes before the dangling comment, not after it.
    assert text.index("climate_experiment") < text.index("belongs to no block")
