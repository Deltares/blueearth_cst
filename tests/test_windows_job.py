"""Windows ownership must survive launcher termination safely."""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from blueearth_cst.shared.windows_job import run_owned_child


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects only")
@pytest.mark.parametrize("capture", [False, True])
def test_owned_child_exit_and_hard_parent_termination(tmp_path, capture):
    marker = tmp_path / "child.txt"
    assert (
        run_owned_child(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path(r'%s').write_text('ok')" % marker,
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                **({"CST_EXECUTION_DIRECTORY": str(tmp_path)} if capture else {}),
            },
        )
        == 0
    )
    assert marker.read_text() == "ok"

    script = tmp_path / "parent.py"
    pid_path = tmp_path / "child.pid"
    child_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from blueearth_cst.shared.windows_job import run_owned_child\n"
        f"run_owned_child([sys.executable, '-c', {child_code!r}], "
        "cwd=Path.cwd(), env=os.environ)\n",
        encoding="utf-8",
    )
    parent_env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    if capture:
        parent_env["CST_EXECUTION_DIRECTORY"] = str(tmp_path)
    parent = subprocess.Popen(
        [sys.executable, str(script)], cwd=Path.cwd(), env=parent_env
    )
    try:
        for _ in range(100):
            if pid_path.exists():
                break
            time.sleep(0.05)
        assert pid_path.exists(), "owned child never started"
        child_pid = int(pid_path.read_text())
        assert psutil.pid_exists(child_pid)
        parent.kill()
        parent.wait(timeout=5)
        for _ in range(100):
            if not psutil.pid_exists(child_pid):
                break
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid), "child survived lock owner's death"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
