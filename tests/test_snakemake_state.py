"""Real engine state stays in the test workspace even from a repo-root CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("attempt", [1, 2])
def test_repo_root_invocation_uses_test_owned_state(isolated_snakemake, attempt):
    workdir = isolated_snakemake
    snakefile = workdir / "Snakefile"
    snakefile.write_text(
        "from pathlib import Path\n"
        "rule all:\n"
        "    input: 'result.txt'\n"
        "rule write_result:\n"
        "    input: 'config/advanced_settings.yml', 'blueearth_cst/spatial/identity.py'\n"
        "    output: 'result.txt'\n"
        "    run:\n"
        f"        Path(output[0]).write_text('{attempt}')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "-s", str(snakefile), "-c", "1"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workdir / "result.txt").read_text() == str(attempt)
    assert (workdir / ".snakemake/iocache").is_dir()
    assert any((workdir / ".snakemake/metadata").iterdir())
