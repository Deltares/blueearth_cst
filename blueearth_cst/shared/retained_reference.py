"""Load a finished run's historical map reference for re-rendering figures."""

from __future__ import annotations

import json
from pathlib import Path

from blueearth_cst.shared.map_preparation import json_digest
from blueearth_cst.shared.map_reference import SCHEMA_VERSION

#: Where every run publishes the shared reference, relative to ``project_dir``.
REFERENCE_PATH = (
    Path("data") / "climate" / "historical" / "_engine" / "map_reference.json"
)


def load_retained_reference(project_dir: str | Path) -> dict:
    """The run's published map reference, checked for integrity only.

    The rules' ``load_reference`` also rejects current-byte drift of code,
    environment and inputs. A re-render draws retained products with today's
    plotting code, so that drift is the point, not a stale input. The schema and
    payload digest still guard against a damaged or hand-edited file.
    """
    path = Path(project_dir) / REFERENCE_PATH
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unknown historical map reference schema")
    if record.get("reference_id") != json_digest(
        {key: value for key, value in record.items() if key != "reference_id"}
    ):
        raise ValueError(f"{path}: historical map reference payload corruption")
    return record
