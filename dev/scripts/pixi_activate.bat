@echo off
REM Pixi Windows activation shim. Runs on every `pixi shell` / `pixi run`.
REM
REM conda-forge's Windows graphviz/glib build imports `ffi.dll` and `cairo.dll`,
REM but libffi/cairo ship them as `ffi-8.dll` / `cairo-2.dll`. Without the alias
REM DLLs, `snakemake --dag | dot -Tpng` warns and falls back to Times fonts.
REM Recreate the missing aliases here. Idempotent (guarded by `if not exist`);
REM the cairo branch is currently a no-op because this build already ships
REM cairo.dll, but it is kept so a future rebuild can't silently regress.
REM See docs/site/setup/env-setup-notes.md for the full diagnosis.
set "BIN=%CONDA_PREFIX%\Library\bin"
if not exist "%BIN%\ffi.dll"   if exist "%BIN%\ffi-8.dll"   copy /Y "%BIN%\ffi-8.dll"   "%BIN%\ffi.dll"   >nul
if not exist "%BIN%\cairo.dll" if exist "%BIN%\cairo-2.dll" copy /Y "%BIN%\cairo-2.dll" "%BIN%\cairo.dll" >nul
