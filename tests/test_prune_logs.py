"""Report-only pruning of execution log folders (`dev/scripts/prune_logs.py`, R13)."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev/scripts"))

import prune_logs  # noqa: E402

NOW = datetime(2026, 11, 13, tzinfo=timezone.utc)
OLD = "2026-09-01_10-00-00Z__build-model"
NEW = "2026-11-12_10-00-00Z__build-model"


def _project(tmp_path):
    for name in (OLD, NEW, f"{OLD}__dry-run", "2026-09-01_wf1", "notes"):
        (tmp_path / "logs" / name).mkdir(parents=True)
    return tmp_path


def _invocation(root, identity, folder, status="succeeded"):
    path = root / "config/_engine/invocations" / f"{identity}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": status, "execution_directory": f"logs/{folder}"}),
        encoding="utf-8",
    )


def test_only_exact_old_folders_are_candidates(tmp_path):
    root = _project(tmp_path)
    found = prune_logs.candidates(root, timedelta(days=30), NOW)
    assert [path.name for path in found] == [OLD, f"{OLD}__dry-run"]


def test_folders_a_product_or_running_invocation_needs_are_kept(tmp_path):
    root = _project(tmp_path)
    _invocation(root, "a" * 32, OLD)
    _invocation(root, "b" * 32, f"{OLD}__dry-run", status="running")
    production = (
        root / "data/climate/projections/cmip6/_engine/execution/production.json"
    )
    production.parent.mkdir(parents=True)
    production.write_text(json.dumps({"generations": {"a" * 32: {}}}))
    assert prune_logs.candidates(root, timedelta(days=30), NOW) == []


def test_nothing_is_deleted_without_the_flag(tmp_path, capsys):
    root = _project(tmp_path)
    assert prune_logs.main(["--project-dir", str(root), "--older-than", "30"]) == 0
    assert (root / "logs" / OLD).is_dir()
    assert "nothing deleted" in capsys.readouterr().out
    prune_logs.main(["--project-dir", str(root), "--older-than", "30", "--delete"])
    assert not (root / "logs" / OLD).exists()
    assert (root / "logs" / NEW).is_dir()
    assert (root / "logs/notes").is_dir()
