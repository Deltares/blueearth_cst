"""Early console filtering must preserve diagnostics, logs and exit status."""

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blueearth_cst.shared import snakemake_console as console
from blueearth_cst.shared import windows_job


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
def test_startup_filter_preserves_raised_severity(level):
    record = logging.LogRecord("snakemake", level, "", 0, "Provided cores: 3", (), None)
    assert console._StartupFilter().filter(record) == (level != logging.INFO)


def test_profile_notice_split_writes_and_unterminated_errors():
    output = io.StringIO()
    stream = console._ProfileStream(output)
    stream.write("Using workflow specific profile profiles/default")
    stream.write(" for setting default command line arguments.\n")
    stream.write("WARNING: profile failed\nRuntimeError: incomplete")
    stream.flush()
    assert output.getvalue() == "WARNING: profile failed\nRuntimeError: incomplete"
    stream.active = False
    notice = "Using workflow specific profile x for setting default command line arguments.\n"
    stream.write(notice)
    assert output.getvalue().endswith(notice)


@pytest.mark.parametrize(
    "command", [["snakemake", "all"], [sys.executable, "-m", "snakemake", "all"]]
)
def test_owned_launcher_routes_both_snakemake_forms(monkeypatch, command):
    calls = []
    monkeypatch.setattr(
        windows_job, "run_owned_child", lambda cmd, **kwargs: calls.append(cmd) or 7
    )
    monkeypatch.setattr(
        windows_job.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 7),
    )
    assert (
        windows_job.run_project_child(command, cwd=Path.cwd(), env={}, writing=False)
        == 7
    )
    assert calls == [[sys.executable, str(Path(console.__file__)), "all"]]


@pytest.mark.process_isolation
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_real_startup_preserves_log_warnings_and_failures(tmp_path, fail, terminal):
    profile = tmp_path / "profiles/default"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("quiet: [reason]\n", encoding="utf-8")
    (tmp_path / "Snakefile").write_text(
        "from snakemake.logging import logger\n"
        "import sys\n"
        "from blueearth_cst.shared.console_style import install_console_style\n"
        "onstart:\n"
        "    install_console_style()\n"
        "    logger.info('TERMINAL=' + str(sys.stdout.isatty()))\n"
        "    logger.warning('Provided cores: 999')\n"
        "    logger.info('workflow diagnostic visible')\n"
        + ("    raise RuntimeError('startup failure visible')\n" if fail else "")
        + "rule all:\n    shell: 'echo CHILD_OUTPUT'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, console.__file__, "all", "--cores", "1"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **{key: value for key, value in os.environ.items() if key != "NO_COLOR"},
            "CST_CONSOLE_INTERACTIVE": "1" if terminal else "0",
        },
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == (1 if fail else 0), output
    assert f"TERMINAL={terminal}" in output
    assert ("\x1b[" in output) == terminal
    assert "Using workflow specific profile" not in output
    assert "Building DAG of jobs..." not in output
    assert "Provided cores: 1" not in output
    assert "Provided cores: 999" in output
    assert "workflow diagnostic visible" in output
    assert ("startup failure visible" if fail else "CHILD_OUTPUT") in output
    logs = list((tmp_path / ".snakemake/log").glob("*.log"))
    assert logs
    assert "Building DAG of jobs..." in logs[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", ["--dry-run", "--verbose"])
def test_diagnostic_modes_keep_native_streams(monkeypatch, flag):
    from snakemake import cli

    stderr = sys.stderr
    calls = []
    monkeypatch.setattr(sys, "argv", ["snakemake", flag])
    monkeypatch.setattr(cli, "main", lambda: calls.append(sys.stderr))
    console.main()
    assert calls == [stderr]


@pytest.mark.workflow_contract
@pytest.mark.skipif(sys.platform != "win32", reason="Windows fallback workaround")
@pytest.mark.parametrize("failure", ["once", "always"])
def test_parallel_completion_timeout_and_resume_preserve_outputs(tmp_path, failure):
    """Inject the recorded unconnected client while the scheduler holds its lock."""
    (tmp_path / "Snakefile").write_text(
        "import os, socket, sys\n"
        "from pathlib import Path\n"
        "from blueearth_cst.shared import windows_socketpair as bounded\n"
        "onstart:\n"
        "    bounded._SETUP_TIMEOUT = 0.02\n"
        "    original_socket = socket.socket\n"
        "    class FaultSocket(original_socket):\n"
        "        def connect(self, address):\n"
        "            frame = sys._getframe()\n"
        "            while frame is not None:\n"
        "                if frame.f_code.co_name == '_finish_jobs':\n"
        "                    failure = os.environ.get('CST_TEST_PAIR_FAILURE')\n"
        "                    marker = Path('injected')\n"
        "                    if failure == 'always' or (failure == 'once' and not marker.exists()):\n"
        "                        marker.write_text('injected', encoding='utf-8')\n"
        "                        return None\n"
        "                    break\n"
        "                frame = frame.f_back\n"
        "            return super().connect(address)\n"
        "    socket.socket = FaultSocket\n"
        "rule all:\n"
        "    input: expand('out/{i}.txt', i=range(13))\n"
        "rule produce:\n"
        "    output: 'out/{i}.txt'\n"
        "    shell: 'echo completed > {output}'\n",
        encoding="utf-8",
    )
    retained = tmp_path / "out/0.txt"
    retained.parent.mkdir()
    retained.write_bytes(b"previous successful output\n")
    retained_stat = retained.stat()
    command = [sys.executable, console.__file__, "all", "--cores", "6"]
    env = {**os.environ, "CST_TEST_PAIR_FAILURE": failure}
    first = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
    )
    output = first.stdout + first.stderr
    assert (tmp_path / "injected").exists(), output
    assert "Windows socket-pair setup timed out" in output
    assert (first.returncode != 0) == (failure == "always"), output
    if failure == "always":
        assert "after 3 attempts" in output
    resumed = subprocess.run(
        [*command, "--rerun-incomplete"],
        cwd=tmp_path,
        env={**env, "CST_TEST_PAIR_FAILURE": "off"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list((tmp_path / "out").glob("*.txt"))) == 13
    assert retained.read_bytes() == b"previous successful output\n"
    assert retained.stat().st_mtime_ns == retained_stat.st_mtime_ns
