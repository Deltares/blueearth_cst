"""The sealed-record registry is enforced, not merely documented (R9 P5 F2).

A sealed record is kept because it is UNEDITED — it is the baseline some past
milestone's commits were checked against. The convention had exactly one
instance and no enforcement: `climate_projections.md` carried a banner,
`climate_experiment.md` was the identical kind of document and went four
milestones without one, so it read as a live WF3 contract while being stale in
paths, rule names, module locations and every Snakefile line number.

**What this module can and cannot do.** It cannot detect a document that ought
to be sealed and is not; nothing can infer "this is a record" from content, and
that judgment belongs at milestone close (AGENTS.md, Conventions). What it does
enforce is the regression that nearly shipped: P5 migrated the WF2 document
wholesale before noticing its banner and had to revert it in full. With this
module, that edit fails a test instead of landing.

Every assertion here is deliberately content-bound. A guard that merely checked
that some string appears somewhere in the file would be satisfied by prose about
the thing it guards — including prose recording its removal — and would keep
passing after the contract it pins is gone.
"""

import hashlib
import re
from datetime import date
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "dev" / "reference" / "sealed-records.yml"

#: Lines of the document head the banner must appear within. A seal banner
#: belongs at the top: the failure it prevents is a reader (or a grep-driven
#: sweep) treating the document as current, and prose 200 lines down does not
#: prevent that.
BANNER_WINDOW = 12

#: `> **SUPERSEDED — … (sealed YYYY-MM-DD).**` — structure, not a substring.
#: The captured date is cross-checked against the registry, so neither source
#: can drift alone.
BANNER_RE = re.compile(
    r"^>\s*\*\*SUPERSEDED\s*[—-].*?\(sealed\s+(\d{4}-\d{2}-\d{2})\)\.?\*\*",
    re.IGNORECASE,
)


def _normalized_sha256(path: Path) -> str:
    """Hash the file's text with newlines normalized to LF.

    NOT a hash of the bytes. `.gitattributes` marks `*.md` as `text`, so the
    repo stores LF and a Windows checkout produces CRLF; a byte hash would pass
    on one platform and fail on the other, and CI runs both.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _registry() -> list[dict]:
    spec = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return list(spec.get("sealed_records") or [])


RECORDS = _registry()
IDS = [r.get("path", f"entry-{i}") for i, r in enumerate(RECORDS)]


def test_registry_is_present_and_populated():
    """An empty registry would make every parametrized test below vacuous.

    Without this, deleting the registry's contents turns the whole module green
    — zero parameters, zero failures — which is the exact shape of guard this
    module exists to argue against.
    """
    assert REGISTRY.is_file(), f"missing registry: {REGISTRY}"
    assert RECORDS, "sealed_records is empty; the module's other tests would not run"


@pytest.mark.parametrize("record", RECORDS, ids=IDS)
def test_registry_entry_is_complete(record):
    """Every field the other tests rely on is present and well-formed."""
    for key in ("path", "sealed", "superseded_by", "current_truth", "why", "sha256"):
        assert record.get(key), f"{record.get('path')}: missing `{key}`"
    assert isinstance(record["sealed"], date), (
        f"{record['path']}: `sealed` must be an unquoted YAML date (YYYY-MM-DD)"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), (
        f"{record['path']}: `sha256` is not a sha256 hex digest"
    )


@pytest.mark.parametrize("record", RECORDS, ids=IDS)
def test_current_truth_resolves(record):
    """`current_truth` names where a reader should go instead, so it must exist.

    Completeness was checked but resolution was not, so a target could be
    renamed out from under the registry and nothing failed: `climate_experiment`
    pointed at `Snakefile_climate_experiment` for months after the workflow
    files were renamed. A pointer that leads nowhere is worse than none, because
    it reads as a live route.

    Only path-shaped values are checked; a plain-prose `current_truth` ("the
    code itself") is left alone.
    """
    target = record["current_truth"]
    if "/" not in target and not target.endswith((".md", ".smk", ".py", ".yml")):
        pytest.skip(f"{record['path']}: `current_truth` is prose, not a path")
    assert (REPO / target).exists(), (
        f"{record['path']}: `current_truth` -> {target} is not on disk"
    )


@pytest.mark.parametrize("record", RECORDS, ids=IDS)
def test_sealed_record_exists(record):
    """A registered record that has been moved or deleted is a broken seal."""
    assert (REPO / record["path"]).is_file(), (
        f"{record['path']} is registered as sealed but is not on disk"
    )


@pytest.mark.parametrize("record", RECORDS, ids=IDS)
def test_sealed_record_carries_its_banner(record):
    """The banner is in the head, and its date matches the registry.

    The date cross-check is the point: it means neither the document nor the
    registry can be edited alone. A banner whose date drifts from the registry
    is either a re-seal that was not recorded or a record that was edited.
    """
    path = REPO / record["path"]
    head = path.read_text(encoding="utf-8").splitlines()[:BANNER_WINDOW]

    matches = [m for m in (BANNER_RE.match(line) for line in head) if m]
    assert matches, (
        f"{record['path']}: no SUPERSEDED banner in the first {BANNER_WINDOW} "
        f"lines. A sealed record must say so where a reader sees it first."
    )
    banner_date = matches[0].group(1)
    assert banner_date == record["sealed"].isoformat(), (
        f"{record['path']}: banner says sealed {banner_date}, "
        f"registry says {record['sealed'].isoformat()}"
    )


@pytest.mark.parametrize("record", RECORDS, ids=IDS)
def test_sealed_record_is_unedited(record):
    """The whole point: a sealed record still hashes to what was sealed.

    If this fails, the fix is almost never to update the hash. It is to revert
    the edit — the document's value is that it was not edited. Re-hash only for
    a deliberate re-seal, and move the `sealed` date with it.
    """
    path = REPO / record["path"]
    assert _normalized_sha256(path) == record["sha256"], (
        f"{record['path']} has been EDITED since it was sealed on "
        f"{record['sealed'].isoformat()}. A sealed record is kept because it is "
        f"unedited ({record['why']}); rewriting it destroys the record it exists "
        f"to be. Revert the change, or re-seal deliberately and update both the "
        f"`sha256` and the `sealed` date in {REGISTRY.name}."
    )
