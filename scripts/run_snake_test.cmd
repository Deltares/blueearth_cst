@echo off
rem ===========================================================================
rem run_snake_test.cmd -- Windows convenience runner for the three CST workflows
rem on the tracked test config, in the required order:
rem     model_creation -> climate_projections -> climate_experiment
rem
rem Everything runs through `pixi run`, so no environment needs activating first
rem (this replaced a `call activate cst` conda step) and `dot` resolves from the
rem pixi env's graphviz rather than a system install.
rem
rem Usage:
rem     scripts\run_snake_test.cmd                 rem run all three
rem     scripts\run_snake_test.cmd --dry-run       rem validate the DAGs only
rem     scripts\run_snake_test.cmd -c 6            rem override cores
rem
rem Any arguments are forwarded verbatim to every `snakemake all` invocation, so
rem `--dry-run` is the cheap way to check this script end-to-end. A later `-c`
rem wins over the default, so `-c 6` overrides without editing the file.
rem
rem Prefer `scripts\run_workflows.py` when you want the `enabled:`-aware wrapper;
rem this script always runs all three regardless of the `workflows:` flags.
rem
rem Exits nonzero on the first failing workflow and does NOT continue to the
rem next -- a downstream workflow must not consume a failed upstream's outputs
rem (same contract as run_workflows.py). No `pause`: this must be usable
rem non-interactively.
rem ===========================================================================
setlocal

set CFG=test_case/snake_config_baseline.yml
set CORES=3
rem O-02: DAG renders belong with the artifacts of the config that produced
rem them, not at the repository root. Backslashes are required -- cmd.exe's
rem mkdir rejects forward slashes. Tracks project_dir in %CFG%.
set DAGDIR=test_case\test_local\dag

where pixi >nul 2>&1
if errorlevel 1 (
    echo [run_snake_test] ERROR: pixi is not on PATH. See docs/install.md.
    exit /b 1
)
if not exist "%DAGDIR%" mkdir "%DAGDIR%"

rem Capture the forwarded arguments ONCE here, in the top-level scope. `shift`
rem inside the subroutine does NOT rebase `%*` (a cmd.exe quirk), so passing
rem `%*` through to :workflow would hand snakemake the label and Snakefile name
rem as positional targets.
set FWD=%*

call :workflow model_creation      Snakefile_model_creation      ""
if errorlevel 1 exit /b 1
rem projections keeps --keep-going: one unavailable CMIP6 model must not abort
rem the whole ensemble.
call :workflow climate_projections Snakefile_climate_projections "--keep-going"
if errorlevel 1 exit /b 1
call :workflow climate_experiment  Snakefile_climate_experiment  ""
if errorlevel 1 exit /b 1

echo.
echo [run_snake_test] all three workflows completed.
exit /b 0

rem ---------------------------------------------------------------------------
rem :workflow <label> <snakefile> <"extra flags">
rem Caller-forwarded snakemake arguments arrive via %FWD%, not %*.
rem ---------------------------------------------------------------------------
:workflow
setlocal
set LABEL=%~1
set SNAKEFILE=%~2
set EXTRA=%~3

echo.
echo [run_snake_test] === %LABEL% ===

rem DAG render is best-effort: graphviz on win-64 needs the ffi/cairo DLL
rem aliases recreated by dev/scripts/pixi_activate.bat, so a failure here is
rem cosmetic and must never abort the run.
pixi run snakemake -s %SNAKEFILE% --configfile %CFG% --dag > "%DAGDIR%\dag_%LABEL%.dot" 2>nul
if errorlevel 1 (
    echo [run_snake_test] note: DAG export skipped.
) else (
    pixi run dot -Tpng "%DAGDIR%\dag_%LABEL%.dot" -o "%DAGDIR%\dag_%LABEL%.png" >nul 2>&1
    rem NB: no parentheses in this message -- an unescaped ')' inside a
    rem parenthesised if/else block closes the block early and cmd dies with
    rem ". was unexpected at this time."
    if errorlevel 1 echo [run_snake_test] note: DAG png skipped - graphviz unavailable.
)

rem Snakemake locks the workdir on a crashed run; clearing it here keeps the
rem script re-runnable without a manual --unlock.
pixi run snakemake --unlock -s %SNAKEFILE% --configfile %CFG% >nul 2>&1

pixi run snakemake all -c %CORES% -s %SNAKEFILE% --configfile %CFG% %EXTRA% %FWD%
if errorlevel 1 (
    echo [run_snake_test] FAILED: %LABEL%
    endlocal & exit /b 1
)
endlocal & exit /b 0
