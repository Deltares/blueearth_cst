"""Every WF1 rule that touches the model root is ordered after its last writer.

ADR 0004. `models/hydrology/wflow/` is a hydromt model directory: successive
`setup_*` calls rewrite it in place, so it has one DECLARED producer (rule 1.03)
and several actual writers. Snakemake attributes the files to 1.03, so a reader
that declares `staticmaps.nc` is ordered after 1.03 rather than after the rule
that writes it last.

Measured on the R9 gate run: rule 1.08 `add_climate_forcing` (`hydromt update
wflow_sbm`) rewrites the WHOLE model root, 17 s after `.outputs_configured` --
the sentinel R9 P2 F5 chose as the anchor on the belief that rule 1.05 was the
last writer. Rule 1.12 read `staticmaps.nc` and finished 9 s before 1.08
rewrote it. Nothing in the DAG produced that margin, and the failure it risks is
silent: `HDF5_USE_FILE_LOCKING="FALSE"` makes a concurrent read abort below
Python with no traceback.

So the invariant is: a rule that touches the model root either WRITES it, and is
a named member of the build chain, or READS it, and declares the terminal
sentinel `.model_final`.

**Comments are stripped before anything is matched.** Otherwise this module
would be satisfied by prose -- a rule carrying `# see .model_final` would pass
while declaring nothing, and the sweep that fixed the comment would "break" the
test. That failure is not hypothetical here; it is what R9 P5 found in
`test_cli.py`'s WF2 region guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SNAKEFILE = Path(__file__).resolve().parents[1] / "Snakefile_model_creation"

#: The sentinel, and the rule that must own it: the LAST writer of the model
#: root. Both are asserted, because the sentinel is only meaningful while it is
#: attached to the last writer.
SENTINEL = ".model_final"
SENTINEL_OWNER = "add_climate_forcing"

#: Rules that WRITE the model root, in chain order. They cannot declare the
#: sentinel -- a writer that waited for the terminal marker would be a cycle --
#: so they are exempt, and this list is the exemption.
#:
#: Adding a rule here is the deliberate act the invariant turns on: it asserts
#: "this rule writes the model and is part of the build chain". Adding one that
#: merely READS the model would silently reopen the race, which is why the list
#: is short, ordered, and commented rather than inferred.
#: `setup_runtime` (1.07) was listed here until R10-1 merged it INTO 1.08. The
#: two landed on separate branches, so neither side's tests could see the
#: contradiction: this list named a rule that no longer exists, and the exemption
#: it granted now belongs to the rule that absorbed it. `add_climate_forcing` was already
#: listed, so the merge removes an entry rather than moving one.
MODEL_ROOT_WRITERS = (
    "build_wflow_model",             # 1.03 creates staticmaps.nc + the toml
    "add_reservoirs_lakes_glaciers", # 1.04 mod.write()/mod.close()
    "declare_wflow_outputs",        # 1.05 mod.write()/mod.close()
    "add_climate_forcing",                   # 1.08 hydromt update -- rewrites all of it,
                                     #      and since R10-1 also writes the
                                     #      forcing yml 1.07 used to hand it
)

#: `rule all` names targets rather than reading them.
NOT_A_READER = ("all",)


def _strip_comments(text: str) -> str:
    """Drop whole-line and trailing comments, preserving line structure."""
    out = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        out.append(stripped)
    return "\n".join(out)


def _rule_bodies() -> dict[str, str]:
    """Map rule name -> its INDENTED block, comments removed.

    The body ends at the first non-blank line at column 0, not at the next
    `rule` -- the Snakefile interleaves module-level code between rules, and
    running to the next rule swept that code into the preceding rule's body.
    Caught by this module's own first run: it reported rule 1.10
    `extract_historical_climate` as a model-root reader because the `_evaluation_pngs`
    comprehension sits after it.
    """
    text = _strip_comments(SNAKEFILE.read_text(encoding="utf-8"))
    lines = text.splitlines()
    bodies: dict[str, str] = {}
    for i, line in enumerate(lines):
        m = re.match(r"^rule\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
        if not m:
            continue
        body: list[str] = []
        for candidate in lines[i + 1:]:
            if candidate.strip() and not candidate[:1].isspace():
                break
            body.append(candidate)
        bodies[m.group(1)] = "\n".join(body)
    return bodies


BODIES = _rule_bodies()
#: Rules whose body references the model root at all.
TOUCHERS = sorted(
    name for name, body in BODIES.items()
    if "basin_dir" in body and name not in NOT_A_READER
)
READERS = [name for name in TOUCHERS if name not in MODEL_ROOT_WRITERS]


def test_the_snakefile_parsed():
    """Guard on the guard: a parse that finds no rules makes everything vacuous."""
    assert len(BODIES) > 10, f"only parsed {len(BODIES)} rules; the regex is wrong"
    assert TOUCHERS, "no rule references basin_dir; the parse or the name changed"


def test_every_declared_writer_exists():
    """The exemption list cannot name a rule that is gone."""
    for name in MODEL_ROOT_WRITERS:
        assert name in BODIES, f"MODEL_ROOT_WRITERS names {name}, which is not a rule"


def test_the_sentinel_is_owned_by_exactly_one_rule():
    """One `touch()` producer, and it is the last writer of the model root.

    If a new rule ever mutates the model after 1.08, the sentinel must move to
    it -- this test says who owns it today, it cannot say who SHOULD.
    """
    owners = [
        name for name, body in BODIES.items()
        if re.search(rf"touch\(f?\"\{{basin_dir\}}/{re.escape(SENTINEL)}\"\)", body)
    ]
    assert owners == [SENTINEL_OWNER], (
        f"{SENTINEL} should be touched by exactly {SENTINEL_OWNER}, found {owners}"
    )


@pytest.mark.parametrize("rule_name", READERS)
def test_model_root_readers_declare_the_sentinel(rule_name):
    """A rule that reads the model root and is not a writer must wait for it."""
    body = BODIES[rule_name]
    assert SENTINEL in body, (
        f"rule `{rule_name}` reads the model root but does not declare "
        f"`{SENTINEL}`, so nothing orders it after rule 1.08's rewrite of the "
        f"whole model directory. Add "
        f'`model_final = ancient(f"{{basin_dir}}/{SENTINEL}")` to its input, or '
        f"-- if it WRITES the model -- add it to MODEL_ROOT_WRITERS with a note "
        f"saying so (ADR 0004)."
    )


@pytest.mark.parametrize("rule_name", MODEL_ROOT_WRITERS)
def test_writers_do_not_wait_on_the_sentinel(rule_name):
    """A writer declaring the terminal marker as input would be a cycle."""
    body = BODIES[rule_name]
    if rule_name == SENTINEL_OWNER:
        assert "touch(" in body, f"{rule_name} should PRODUCE {SENTINEL}"
        return
    assert SENTINEL not in body, (
        f"rule `{rule_name}` is a model-root writer but references {SENTINEL}; "
        f"a writer that waits for the terminal marker is a cycle"
    )
