"""The Julia version pin must agree across every place that declares it.

Julia is not in the pixi env (AGENTS.md hard constraint) — it is juliaup-managed
and selected per invocation by a ``+<version>`` prefix, so nothing in the
toolchain makes the declarations agree on their own. Three files pin it
independently and a drift between them is silent until a run picks up the wrong
toolchain: the env gets instantiated at one version while the workflows run at
another.

``config/advanced_settings.yml`` (``runtime.julia_version``) is the human-facing
declaration and the only one the workflows read. ``pixi.toml`` and
``Manifest.toml`` cannot read YAML, so this module is what keeps them in step —
single-sourcing is not available here, and pretending otherwise would leave the
drift undetected.

``Project.toml``'s ``julia = "~1.11"`` compat bound is deliberately looser (a
range, and the reason for the range is documented there), so it is not compared
for equality — only checked for shape and coverage.
"""

import re
from pathlib import Path

import pytest

from blueearth_cst.shared.snake_utils import (
    DEFAULT_JULIA_THREADS,
    JULIA_VERSION,
    julia_prefix,
    validate_julia_threads,
)

REPO = Path(__file__).resolve().parents[1]


def test_the_settings_file_is_where_the_constant_comes_from():
    """Read independently of the module-level load, so a value left hardcoded
    in snake_utils would show up here rather than pass silently."""
    import yaml

    from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS_PATH

    on_disk = yaml.safe_load(ADVANCED_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert JULIA_VERSION == on_disk["runtime"]["julia_version"]


def test_the_version_is_a_quoted_three_part_string():
    """Unquoted 1.11 would be a YAML float and would silently become the
    selector `+1.11`, letting juliaup pick a patch the manifest never saw."""
    assert isinstance(JULIA_VERSION, str)
    assert len(JULIA_VERSION.split(".")) == 3


def test_pixi_install_task_pins_the_same_version():
    """pixi.toml's install-julia is what instantiates Manifest.toml."""
    text = (REPO / "pixi.toml").read_text(encoding="utf-8")
    versions = set(re.findall(r"julia \+(\d+\.\d+\.\d+)", text))
    assert versions == {JULIA_VERSION}, (
        f"pixi.toml pins {versions or 'nothing'}, snake_utils pins {JULIA_VERSION}"
    )


def test_manifest_was_resolved_against_the_same_version():
    text = (REPO / "Manifest.toml").read_text(encoding="utf-8")
    match = re.search(r'^julia_version = "([^"]+)"', text, re.M)
    assert match, "Manifest.toml carries no julia_version"
    assert match.group(1) == JULIA_VERSION


def test_project_compat_bound_admits_the_pin_and_excludes_1_12():
    """The bound is a range rather than the pin, but it must do both jobs.

    Julia's compat syntax defaults to CARET, so a bare ``1.11`` resolves to
    [1.11.0, 2.0.0) — it would admit the 1.12.x that Wflow.jl#884 is the whole
    reason for excluding, while reading as if it did not. ``~1.11`` is
    [1.11.0, 1.12.0). Asserted because the failure is silent: nothing else
    checks the bound's shape, and the exclusion is claimed in prose in two
    files (Project.toml, config/advanced_settings.yml).
    """
    text = (REPO / "Project.toml").read_text(encoding="utf-8")
    match = re.search(r'^julia = "([^"]+)"', text, re.M)
    assert match, "Project.toml carries no julia compat bound"
    bound = match.group(1)
    assert bound.startswith("~"), (
        f'julia compat is "{bound}"; a bare or caret bound admits 1.12.x'
    )
    assert JULIA_VERSION.startswith(bound.removeprefix("~") + ".")


def test_no_snakefile_hardcodes_a_julia_version():
    """The point of the constant: a version must not reappear inline in a
    shell body, where it is invisible to the checks above."""
    offenders = {
        path.name
        for path in REPO.glob("Snakefile_*")
        if re.search(r"julia \+\d+\.\d+\.\d+", path.read_text(encoding="utf-8"))
    }
    # Empty since 2026-08-13. Snakefile_climate_experiment was the one tolerated
    # offender -- the same hardcode WF1 had, deferred to the WF3 pass -- and it
    # now uses julia_prefix like every other Julia call. The allowance is
    # REMOVED rather than left permissive: it was listed so that adopting
    # julia_prefix would shrink the set, and a tolerance kept after its cause
    # is gone is how the next inline pin slips in unnoticed.
    assert offenders == set(), offenders


# --- the threads knob ------------------------------------------------------


def test_default_threads_is_the_frozen_baseline_value():
    """P3-3's recorded baselines were all measured at the (-c 3, --threads 4,
    B=1) triple; changing this default silently invalidates them."""
    assert DEFAULT_JULIA_THREADS == 4


def test_prefix_carries_version_project_and_threads():
    assert julia_prefix(8) == f"julia +{JULIA_VERSION} --project=. --threads 8"


def test_prefix_defaults_to_the_baseline_thread_count():
    assert julia_prefix() == julia_prefix(DEFAULT_JULIA_THREADS)


@pytest.mark.parametrize("bad", [0, -1, "4", 4.0, True, None])
def test_non_positive_or_non_integer_threads_are_rejected(bad):
    """Parse-time rejection: the value lands in a shell body, so a bad one would
    otherwise surface as a Julia usage error inside a rule."""
    with pytest.raises(ValueError, match="julia_threads"):
        validate_julia_threads(bad)


def test_a_valid_override_is_returned_unchanged():
    assert validate_julia_threads(12) == 12
