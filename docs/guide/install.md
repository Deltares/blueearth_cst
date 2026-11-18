---
title: "Install the toolbox"
subtitle: "Windows setup and data access"
---

This guide explains how to install the CST Toolbox on Windows. You will install the required software,
download the toolbox and check that its commands are available.

Input datasets are obtained separately. Running the supplied example also requires access to the
Deltares P-drive or suitable alternative data.

## Install the required software

Run the commands below in **PowerShell**, which you can open from the Windows Start menu.

### Git

Git downloads the toolbox files from GitHub.

```powershell
winget install --id Git.Git --exact --source winget
```

Close and reopen PowerShell after installation.

### Pixi

Pixi installs and manages the software packages used by the toolbox, including Python and R.
It keeps these packages together in a dedicated environment, so you do not need to install them
individually.

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

Close and reopen PowerShell so that the `pixi` command becomes available.

### Julia

The toolbox uses the Wflow hydrological model, which runs in Julia. Install Julia separately
from the packages managed by Pixi.

Install juliaup, which manages Julia versions:

```powershell
winget install Julialang.Juliaup
```

Close and reopen PowerShell, then install the Julia version required by the toolbox:

```powershell
juliaup add 1.11.7
julia +1.11.7 --version
```

The last command should report `julia version 1.11.7`. Use this version to remain consistent
with the toolbox's dependencies.

## Download the toolbox

In PowerShell, move to the folder where you want to keep the toolbox. Then run:

```powershell
git clone https://github.com/Deltares/blueearth_cst.git
cd blueearth_cst
```

The first command downloads the toolbox into a folder named `blueearth_cst`; the second opens
that folder in PowerShell. Run the remaining commands from this folder.

## Install the toolbox packages

The toolbox also uses Python and R. Install both, along with their required packages, into
the Pixi environment with a single command:

```powershell
pixi install
```

Pixi installs the package versions specified by the toolbox. Next, install the additional weather-generator
and hydrological-model packages:

```powershell
pixi run install
```

The prefix `pixi run` runs a command using the toolbox's software environment.
This installation includes the R weather generator, `weathergenr`, and the Julia packages
needed for Wflow. Downloads may take some time.

If either command stops with an error, consult the [troubleshooting guide](troubleshooting.md)
before continuing.

## Check the installation

Run the installation check:

```powershell
pixi run pytest tests/test_cli.py
```

This checks that the toolbox's workflow commands are connected correctly. Read the final summary:
if it reports `FAILED` or `ERROR`, consult the [troubleshooting guide](troubleshooting.md)
before proceeding.

This check does not run a basin simulation or assess model accuracy. You will run the example
in the next guide.

## Check access to input data

Installing the toolbox does not download the climate and model input datasets.

The supplied example uses datasets on the Deltares P-drive. Check whether the expected folder
is accessible:

```powershell
Test-Path 'P:\wflow_global\hydromt'
```

A result of `True` confirms that the folder is accessible. If the result is `False`, connect
to the drive if you have access, or configure alternative data sources before running the example.
Alternative sources must provide the required variables and source names and cover your basin
and study period. See [Creating a project](new-case.md) for configuration guidance.

Downloading CMIP6 climate-projection data also requires an internet connection. Check access
to the public CMIP6 data source:

```powershell
pixi run python -c "import gcsfs; gcsfs.GCSFileSystem(token='anon').ls('cmip6/CMIP6'); print('CMIP6 data source: accessible')"
```

If the check succeeds, it prints `CMIP6 data source: accessible`. If it reports an error,
check your internet connection and consult the [troubleshooting guide](troubleshooting.md).
This check confirms access to the data source; it does not verify the availability of every dataset.

## Next step

Follow the [quick start](quick-start.md) to run the example and inspect its outputs.
The example requires input data, computing time and disk space in addition to the software
installed here.

For help, see [Troubleshooting](troubleshooting.md). To configure another basin, see
[Creating a project](new-case.md).

## Update the toolbox

To move to a newer toolbox version, run these commands from the [toolbox folder](glossary.md#toolbox-folder),
`blueearth_cst`:

```powershell
git pull
pixi install
pixi run install
```

`git pull` downloads the new toolbox files, and `pixi install` updates the Python and R packages to the
versions the new files specify. `pixi run install` reinstalls the weather generator, `weathergenr`, and the
Julia packages only when the update changed their pinned versions; otherwise it confirms them. If you skip it
after a weather-generator update, rerun `pixi run install-rdeps` before generating climate scenarios. If the
Julia step reports a missing Julia version, install that version with `juliaup add` as in
[Install the required software](#install-the-required-software).

Then repeat the installation check in [Check the installation](#check-the-installation).
