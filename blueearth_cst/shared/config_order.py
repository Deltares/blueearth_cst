"""Give a dry run the key order of the runner's execution configuration.

Real WF0-WF2 runs read the execution configuration the runner archives, which is
written with sorted keys; dry runs read the authored files. Snakemake compares
rule params by their text, so a mapping-valued param built from authored order
never matches the one recorded by a real run, and every dry run reports
"params changed". Sorting the composed mapping of an uncaptured (dry) run
removes that difference.

Imported only by the WF0-WF2 Snakefiles, so it never enters WF3/WF4 identities.
"""

from __future__ import annotations

from typing import Any


def sorted_mappings(value: Any) -> Any:
    """Return ``value`` with every nested mapping's keys in sorted order."""
    if isinstance(value, dict):
        return {key: sorted_mappings(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [sorted_mappings(item) for item in value]
    return value
