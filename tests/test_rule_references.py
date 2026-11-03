"""Every `checkpoints.<name>` / `rules.<name>` in a Snakefile names a real rule.

Snakemake resolves these attributes only when the input function runs, which
for a checkpoint is AFTER it completes -- so a stale name passes every dry
run and fails mid-simulation. The 2026-09-24 rule rename left three such
references behind, which is why this is checked statically.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAKEFILES = sorted(REPO.glob("*.smk")) + sorted(
    (REPO / "blueearth_cst").rglob("*.smk")
)
DEFINED = re.compile(r"^\s*(?:rule|checkpoint)\s+([A-Za-z_]\w*)\s*:", re.M)
REFERENCED = re.compile(r"\b(checkpoints|rules)\.([A-Za-z_]\w*)")


def test_every_rule_reference_resolves():
    text = "\n".join(path.read_text(encoding="utf-8") for path in SNAKEFILES)
    defined = set(DEFINED.findall(text))
    checkpoints = set(re.findall(r"^\s*checkpoint\s+(\w+)\s*:", text, re.M))
    stale = []
    for kind, name in REFERENCED.findall(text):
        pool = checkpoints if kind == "checkpoints" else defined
        if name not in pool:
            stale.append(f"{kind}.{name}")
    assert not stale, f"references to undefined rules: {sorted(set(stale))}"
