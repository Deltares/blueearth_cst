"""`C-37` — every declared config key reaches a reader, and it fails closed.

The clean run on today's tree is the least interesting assertion here. What has
to be shown is that the check CAN fire, and — the whole reason `D-14.5` exists —
that it says so when it cannot see one of its two sides instead of reporting a
clean bill it has no basis for.

The failure mode is not hypothetical. R14's own register re-measure was run
twice because the first pass's ground-truth extraction silently returned zero
keys, and the pass then reported MORE problems while knowing LESS.
"""

from __future__ import annotations

import pytest

from dev.scripts.sweep_common import REPO_ROOT, check_floors
from dev.scripts.sweep_unread_config_keys import (
    ALLOWANCES,
    FLOORS,
    declared_keys,
    reader_keys,
    reader_surfaces,
    sweep,
)


def _project(tmp_path, template_body, reader_body="", reader="blueearth_cst/x.py"):
    """A throwaway project: one template, one reader surface.

    The reader always reads the `model:` SECTION, because a section name is a
    declared key like any other and every fixture here nests under one. Without
    it each test would report `model` alongside whatever it is actually about.
    """
    templates = tmp_path / "config/templates/example-basin"
    templates.mkdir(parents=True)
    (templates / "project_config_build_model.yml").write_text(
        template_body, encoding="utf-8"
    )
    target = tmp_path / reader
    target.parent.mkdir(parents=True, exist_ok=True)
    section = 'cfg[["model"]]\n' if target.suffix == ".R" else "cfg['model']\n"
    target.write_text((section + reader_body) if reader_body else "", encoding="utf-8")
    return tmp_path


# --- it can fire -------------------------------------------------------------


def test_a_declared_key_with_no_reader_is_reported(tmp_path):
    """The defect this exists for: a key a user can set that changes nothing,
    on a run that succeeds and says so about nothing."""
    root = _project(
        tmp_path,
        "model:\n  wflow_threads: 4\n",
        "value = model_cfg['something_else']\n",
    )
    defects, _allowed, observed = sweep(root)
    assert observed["declared"] == 2
    assert [d[2] for d in defects] == ["wflow_threads: 4"]


def test_a_commented_default_is_a_declaration_too(tmp_path):
    """ "Lines starting with `#` are optional settings, shown at their default"
    — the template header's own words, so a commented key is a promise as much
    as a live one, and the one a filled-in config will never show."""
    root = _project(tmp_path, "model:\n  #wflow_threads: 4\n", "cfg['model']\n")
    defects, _allowed, _observed = sweep(root)
    assert [d[2] for d in defects] == ["#wflow_threads: 4"]


def test_prose_in_a_template_is_not_a_declared_key(tmp_path):
    """ "# Format: observed_daily_discharge_template.csv." is a sentence. It was
    the entire output of this check's first draft."""
    root = _project(
        tmp_path,
        "model:\n  # Format: observed_daily_discharge_template.csv.\n",
        "cfg['model']\n",
    )
    defects, _allowed, observed = sweep(root)
    assert observed["declared"] == 1
    assert defects == []


@pytest.mark.parametrize(
    "reader_body",
    [
        "n = get_config(model_cfg, 'wflow_threads', 4)\n",
        "n = model_cfg['wflow_threads']\n",
        "n = model_cfg.get('wflow_threads')\n",
        "n = climate.get('wflow_threads')\n",
        "n = thresholds['wflow_threads']\n",
        "n = stress_test_cfg['temp']['wflow_threads']\n",
        "n = axis_cfg[axis_name]['wflow_threads']\n",
    ],
)
def test_the_reader_forms_this_repo_actually_uses(tmp_path, reader_body):
    """The last two are the ones a first draft got wrong: a chain records EVERY
    literal segment, and a segment subscripted by a variable is stepped over
    rather than ending the chain."""
    root = _project(tmp_path, "model:\n  wflow_threads: 4\n", reader_body)
    defects, _allowed, _observed = sweep(root)
    assert defects == [], reader_body


def test_an_R_read_counts(tmp_path):
    """R takes its config through `commandArgs` and reads it as a list, so a key
    read only there is still read."""
    root = _project(
        tmp_path,
        "model:\n  wflow_threads: 4\n",
        'n <- cfg[["wflow_threads"]]\n',
        reader="blueearth_cst/weathergen/x.R",
    )
    defects, _allowed, _observed = sweep(root)
    assert defects == []


def test_a_read_in_dev_scripts_does_not_count(tmp_path):
    """`dev/scripts/` is never part of a run — the repository's own
    invocation-model rule — so a key read only there is read by no run."""
    root = _project(
        tmp_path,
        "model:\n  wflow_threads: 4\n",
        "n = model_cfg['wflow_threads']\n",
        reader="dev/scripts/tool.py",
    )
    defects, _allowed, observed = sweep(root)
    # Both keys, including the section: the file is not a surface at all, so
    # nothing in it counts — which is the point rather than a side effect.
    assert observed["readers"] == 0
    assert [d[2] for d in defects] == ["model:", "wflow_threads: 4"]


# --- failing closed ----------------------------------------------------------


def test_an_empty_reader_side_is_BLIND_not_a_pile_of_findings(tmp_path):
    """The exact shape R14 failed in: with nothing on the reader side every
    declared key reports unread, and the run looks maximally informative while
    knowing nothing at all."""
    root = _project(tmp_path, "model:\n  wflow_threads: 4\n", "")
    _defects, _allowed, observed = sweep(root)
    assert observed["readers"] == 0
    blind = check_floors(FLOORS, observed)
    assert any(message.startswith("readers:") for message in blind)


def test_an_empty_declared_side_is_BLIND_not_a_clean_bill(tmp_path):
    """The mirror failure, and the more dangerous one: no templates found means
    no keys to check, which prints as success."""
    root = tmp_path
    (root / "blueearth_cst").mkdir(parents=True)
    (root / "blueearth_cst" / "x.py").write_text("cfg['a']\n", encoding="utf-8")
    defects, _allowed, observed = sweep(root)
    assert defects == []
    assert observed["declared"] == 0
    blind = check_floors(FLOORS, observed)
    assert any(message.startswith("declared:") for message in blind)


def test_the_floors_are_over_zero_and_under_their_measured_counts():
    for floor in FLOORS:
        assert floor.minimum > 0, floor.name
        assert floor.minimum < floor.count, floor.name
        assert floor.why.strip() and floor.measured.strip(), floor.name


def test_both_ground_truths_are_populated_on_the_live_tree():
    """Named separately from the floors: this is the assertion a reader of a
    green run is entitled to, and it should not require reading the floor
    table to believe."""
    assert len(declared_keys()) > 30
    assert len(reader_keys()) > 45
    assert len(reader_surfaces()) > 70


def test_the_live_tree_has_no_unread_declared_key():
    defects, _allowed, observed = sweep()
    assert not check_floors(FLOORS, observed)
    assert defects == [], [(str(p), n, t) for p, n, t, _ in defects]


def test_every_allowance_states_a_reason():
    for allowance in ALLOWANCES:
        assert allowance.reason.strip(), allowance.name
        assert len(allowance.reason) > 40, allowance.name


def test_the_window_bounds_are_classified_not_silently_dropped():
    """R12's frozen simulation planner reads both bounds explicitly.

    They previously needed the tuple-iteration allowance; now the matcher must
    classify their actual readers rather than silently omit the declarations.
    """
    _defects, allowed, _observed = sweep()
    assert {"start", "end"} <= set(reader_keys())
    assert not ({a[2].split(":")[0] for a in allowed} & {"start", "end"})


def test_the_templates_it_reads_are_the_ones_a_user_copies():
    """A check pointed at the wrong templates is blind in a way no floor
    catches: `config/defaults/` would give it plenty of keys, all of them
    hydromt's rather than ours. The tracked seeds are the one other source."""
    names = {path for path, _line, _text in declared_keys().values()}
    assert any(
        name.startswith("config/templates/example-basin/project_config")
        for name in names
    )
    assert all(
        name.startswith(
            (
                "config/templates/example-basin/project_config",
                "test_case/project_config_",
            )
        )
        for name in names
    )
    assert (REPO_ROOT / "config" / "templates").is_dir()


def test_the_shipped_seeds_are_declared_and_covered():
    """t2609070015: a green run now covers the seeds, not the templates only.

    `build_config` and `waterbodies_config` were once the MIRROR gap. They were
    documented all along, commented one level deeper than the matcher saw.
    """
    import yaml

    def leaves(node, out):
        if isinstance(node, dict):
            for key, value in node.items():
                out.add(key)
                leaves(value, out)
        elif isinstance(node, list):
            for value in node:
                leaves(value, out)
        return out

    seed_leaves = set()
    for path in sorted((REPO_ROOT / "test_case").glob("project_config_*.yml")):
        leaves(yaml.safe_load(path.read_text(encoding="utf-8")), seed_leaves)
    declared = set(declared_keys())
    assert seed_leaves, "no shipped seed configs found"
    assert {"build_config", "waterbodies_config", "source", "dry"} <= declared
    # Only user-chosen names outside the key grammar stay undeclared.
    assert all(" " in str(key) for key in seed_leaves - declared), (
        seed_leaves - declared
    )


def test_a_deeper_commented_default_is_a_declaration(tmp_path):
    root = _project(tmp_path, "model:\n#  build_config: x.yml\n", "")
    assert "build_config" in declared_keys(root)


@pytest.mark.parametrize(
    "reader_body",
    [
        'name = _source_name(sources_cfg, "build_config", "x")\n',
        'value = pick("build_config", "a", "b")\n',
        'value = body["build_config"]\n',
    ],
)
def test_an_accessor_or_sub_mapping_read_counts(tmp_path, reader_body):
    root = _project(tmp_path, "model:\n  build_config: x.yml\n", reader_body)
    defects, _allowed, _observed = sweep(root)
    assert not defects


def test_a_seed_key_no_template_names_is_undocumented(tmp_path):
    root = _project(tmp_path, "model:\n  kept: 1\n", 'cfg["kept"]\ncfg["extra"]\n')
    seeds = root / "test_case"
    seeds.mkdir()
    (seeds / "project_config_x.yml").write_text(
        "model:\n  kept: 1\n  extra: 2\n", encoding="utf-8"
    )
    defects, _allowed, _observed = sweep(root)
    assert [(p.name, kind) for p, _n, _t, kind in defects] == [
        ("project_config_x.yml", "UNDOCUMENTED")
    ]


def test_a_flow_style_template_mention_documents_a_seed_key(tmp_path):
    root = _project(
        tmp_path,
        "model:\n  windows:\n  - {start: 1, name: mid}\n",
        'cfg["windows"]\ncfg["start"]\nentry.get("name")\n',
    )
    seeds = root / "test_case"
    seeds.mkdir()
    (seeds / "project_config_x.yml").write_text(
        "model:\n  windows:\n  - start: 1\n    name: mid\n", encoding="utf-8"
    )
    defects, _allowed, _observed = sweep(root)
    assert not defects
