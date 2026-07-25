BlueEarth Climate Stress Test toolbox
#####################################

.. image:: https://github.com/tanerumit/blueearth_cst/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/tanerumit/blueearth_cst/actions/workflows/ci.yml
   :alt: CI

.. note::

   **Fork status.** This is a personal fork of
   `Deltares/blueearth_cst <https://github.com/Deltares/blueearth_cst>`_.
   Three planned phases are complete:

   - **Phase 1 — Foundation** (sealed, ``v0.2.0-alpha``): replication
     baseline, pixi-based env, library upgrades (hydromt 1.x,
     Wflow.jl 1.0.x), unit-test coverage.
   - **Phase 2 — Refactor** (sealed 2026-07-23, R1–R6): modularity
     contracts, naming conventions, one milestone per workflow, then the
     structural refactor that moved ``src/`` to the ``blueearth_cst``
     package and split ``config/`` into ``workflows`` / ``catalogs`` /
     ``templates``.
   - **Phase 3 — Usability & flexibility** (sealed 2026-07-25, P3-1…P3-3):
     project/experiment structure, model-independent climate analysis,
     model-swap interchange contracts, and performance work (the wf3
     stress-test sweep is ~35 % faster, value-identical).

   Phase 4 is open; CI was its first item. ``dev/roadmap.md`` is the
   authoritative status record — this note summarises it and may lag.
   See also ``CHANGELOG.md``, which tracks releases rather than
   milestones.

The BlueEarth Climate Stress Test toolbox (``blueearth_cst``) is a
free, open-source toolkit for interactive climate risk assessment
based on bottom-up analysis principles. It enables end-users to:

- Explore the range of hydro-climatic uncertainty in a chosen
  geographic area, including natural variability and climate-change
  signals.
- Design and execute climate stress tests against user-defined
  thresholds and metrics.
- Assess the plausibility of identified vulnerabilities using climate
  model projections — i.e. estimate how sensitive a chosen metric is
  to climate change.
- Visualize results for non-specialist audiences.

The toolbox is part of the BlueEarth_ initiative and uses weathergenr_
as its weather generator and Wflow_ for hydrological modelling.

.. image:: docs/_images/CST_scheme.png

.. _BlueEarth: https://blueearth.deltares.org/
.. _weathergenr: https://github.com/Deltares/weathergenr
.. _Wflow: https://github.com/Deltares/Wflow.jl


Installation
============

``blueearth_cst`` is a Python + R + Julia toolbox. Python and R
dependencies are managed with pixi_; Julia and Wflow.jl are managed
via the standard ``Project.toml`` / ``Manifest.toml``. A single
``pixi run install`` task wires both layers together.

For a step-by-step walkthrough of a fresh install, see
``docs/install.md``.

Prerequisites
-------------

1. **pixi** (manages Python 3.12 + R 4.4 via conda-forge; the exact pins
   live in ``pixi.toml`` — R is held at 4.4 because conda-forge's 4.5
   ``r-waveslim`` build is broken on win-64).

   Windows (PowerShell):

   .. code-block:: powershell

       iwr -useb https://pixi.sh/install.ps1 | iex

   Or via winget: ``winget install prefix-dev.pixi``. Restart your
   shell after install.

2. **Julia 1.11.x** via juliaup_. conda-forge has no win-64 Julia
   build, and Wflow.jl 1.0.x deadlocks under Julia 1.12. After
   installing juliaup:

   .. code-block:: console

       juliaup add 1.11
       juliaup default 1.11

   Verify with ``julia --version`` (expect ``1.11.x``).

3. Clone the repo:

   .. code-block:: console

       git clone https://github.com/tanerumit/blueearth_cst.git
       cd blueearth_cst

Install
-------

.. code-block:: console

    pixi install         # Python + R toolchain (conda-forge)
    pixi run install     # weathergenr (R) + Wflow.jl (Julia)

The first command installs everything declared in ``pixi.toml`` into
a local ``.pixi/`` env. The second runs
``dev/scripts/install_weathergenr.R`` (installs
``tanerumit/weathergenr@v1.2.0``) and
``julia --project=. -e 'using Pkg; Pkg.instantiate()'`` (locks
Wflow.jl and ~130 transitive Julia deps from ``Manifest.toml``).

To activate the env in your shell:

.. code-block:: console

    pixi shell

Docker
------

.. warning::

   **Not supported in v0.2.0-alpha.** Docker / Linux end-to-end
   validation is **deferred** in this fork — see
   "Deferred: Linux replication" in ``dev/roadmap.md``. The
   ``Dockerfile`` builds against the pixi env but is not exercised
   in CI. Docker support will be re-introduced in a later Phase 2
   milestone.

   The instructions below describe the **v0.1.0-alpha** (upstream
   conda-based) Docker workflow and remain valid for users of that
   release. They do **not** apply to v0.2.0-alpha or later
   pixi-based releases.

A pre-built image of the v0.1.0-alpha conda-based stack remains
available at ``containers.deltares.nl/CST/cst_workflows:0.1.0``:

.. code-block:: console

    docker pull containers.deltares.nl/CST/cst_workflows:0.1.0

.. _pixi: https://pixi.sh/
.. _juliaup: https://github.com/JuliaLang/juliaup


Running
=======

The toolbox provides three Snakemake_ workflows:

- **Snakefile_model_creation** — builds a Wflow model from global
  data for the selected region and runs / analyses it for a
  historical period.
- **Snakefile_climate_projections** — derives future climate
  statistics (temperature and precipitation change) for a chosen set
  of CMIP scenarios and GCMs.
- **Snakefile_climate_experiment** — generates future weather
  realizations, applies stress-test perturbations, and runs the
  hydrological model on each realization × stress combination.

Configuration is YAML-driven. An example is at
``config/workflows/snake_config_model_test.yml``. Configs live under
``config/workflows/``, hydromt data catalogs under ``config/catalogs/``,
and hydromt/wflow/weathergen build templates under ``config/templates/``.

Each run writes its generated model and result artifacts to the
``project_dir`` set in the config. For production use, point
``project_dir`` at a location **outside the repository tree** so outputs
are kept separate from the toolbox source. (The in-repo
``examples/test_local`` directory is a dev/test convention only.)

Running from pixi shell
-----------------------

Activate the env, then invoke ``snakemake`` against the Snakefile and
config of your choice:

.. code-block:: console

    $ pixi shell
    $ cd blueearth_cst
    $ snakemake all -c 1 -s Snakefile_model_creation \
        --configfile config/workflows/snake_config_model_test.yml

See the per-workflow sections below for the recommended sequences
(DAG visualization, unlocking, full run).

Common ``snakemake`` flags:

- ``-s``: which Snakefile to run.
- ``--configfile``: path to the YAML config.
- ``-c``: number of cores (more than 1 enables parallelism).
- ``--dry-run`` (``-n``): list rule executions without running them.
- ``--unlock``: clear the working-directory lock left by a crash.
- ``--keep-going`` (``-k``): keep running independent jobs after a
  failure.

For all options see the
`Snakemake CLI documentation <https://snakemake.readthedocs.io/en/stable/executing/cli.html>`_.
More example invocations are in ``scripts/run_snake_test.cmd``.

.. _Snakemake: https://snakemake.github.io/

Running all enabled workflows with the wrapper
----------------------------------------------

Instead of invoking each Snakefile by hand, ``scripts/run_workflows.py``
reads the ``workflows.<name>.enabled`` flags in a full-orchestration
config and runs ``snakemake`` for exactly the enabled workflows, in order
(model → projections → experiment):

.. code-block:: console

    $ pixi run python scripts/run_workflows.py \
        --config config/workflows/snake_config_model_test.yml

Contract:

- Accepts **full-orchestration configs only** — a config carrying a
  ``workflows:`` section with all three subsections, each with an
  ``enabled:`` key (the ``snake_config_model_test*.yml`` /
  ``snake_config.template.yml`` class). The single-workflow
  ``snake_config_projections_*.yml`` configs carry no ``workflows:``
  section and are run directly with ``snakemake -s`` instead.
- A missing ``workflows:`` section or ``<name>.enabled`` key is a **hard
  error** naming the absent key, not a silent default.
- ``enabled:`` must parse to a real boolean: unquoted ``true`` / ``false``
  / ``yes`` / ``no`` / ``on`` / ``off`` are accepted; quoted ``"true"`` or
  integers ``1`` / ``0`` are rejected.
- The wrapper **stops on the first nonzero Snakemake exit and returns that
  code** — a failed upstream workflow is not followed by a downstream run.
- ``--cores N`` and any arguments after a ``--`` sentinel forward to every
  invocation; each workflow keeps its own flags (``--keep-going`` on
  projections only).

**Skip semantics.** ``enabled: false`` means the wrapper does not invoke
that Snakefile, so its outputs are not produced. It does **not** delete
that workflow's prior outputs and does **not** guarantee downstream
freshness: an enabled downstream workflow consumes whatever prerequisite
artifacts already exist on disk (or fails with ``MissingInputException``
if they are absent) — identical to invoking a single Snakefile directly.
You are responsible for the staleness of what a downstream workflow
consumes when you disable its prerequisite.

Running from docker image
-------------------------

.. warning::

   **v0.1.0-alpha only.** ``scripts/run_snake_docker.sh`` targets the upstream
   conda-based image. Not supported on the v0.2.0-alpha pixi-based
   fork; deferred per "Deferred: Linux replication" in
   ``dev/roadmap.md``.

A script is available to run via Docker: ``scripts/run_snake_docker.sh``.

Snakefile_model_creation
------------------------

Builds a hydrological Wflow model and runs / analyses it for a
historical period.

.. code-block:: console

    $ snakemake -s Snakefile_model_creation --configfile config/workflows/snake_config_model_test.yml --dag | dot -Tpng > dag_model.png
    $ snakemake --unlock -s Snakefile_model_creation --configfile config/workflows/snake_config_model_test.yml
    $ snakemake all -c 1 -s Snakefile_model_creation --configfile config/workflows/snake_config_model_test.yml

The first command generates a DAG visualization (requires Graphviz's
``dot``). The second clears any leftover working-directory lock from
a prior crash. The third runs the workflow.

Snakefile_climate_projections
-----------------------------

Derives future climate statistics (expected temperature and
precipitation change) for selected CMIP scenarios and GCMs.

.. code-block:: console

    $ snakemake -s Snakefile_climate_projections --configfile config/workflows/snake_config_model_test.yml --dag | dot -Tpng > dag_projections.png
    $ snakemake --unlock -s Snakefile_climate_projections --configfile config/workflows/snake_config_model_test.yml
    $ snakemake all -c 1 -s Snakefile_climate_projections --configfile config/workflows/snake_config_model_test.yml --keep-going

Snakefile_climate_experiment
----------------------------

Prepares future weather realizations and stress-test perturbations,
runs them through the hydrological model, and aggregates the
discharge statistics.

.. code-block:: console

    $ snakemake -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml --dag | dot -Tpng > dag_experiment.png
    $ snakemake --unlock -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml
    $ snakemake all -c 1 -s Snakefile_climate_experiment --configfile config/workflows/snake_config_model_test.yml


Testing
=======

The cheapest useful check is the dry-run gate, which parses all three
Snakefiles and validates their DAGs without running anything:

.. code-block:: console

    $ pixi run pytest tests/test_cli.py

The full unit suite takes a couple of minutes:

.. code-block:: console

    $ pixi run pytest tests/

Notes on what the suite does and does not cover:

- Tests that need the untracked ``examples/test_local`` fixture tree
  **skip** when it is absent, as do three end-to-end workflow tests that
  are opt-in behind ``--run-integration``. Run ``pytest -rs`` to see every
  skip reason. On a clean checkout expect ~385 passed and ~30 skipped.
- ``dev/scripts/check_baseline.py check`` and
  ``dev/scripts/semantic_tree_diff.py`` compare a produced output tree
  against a recorded baseline. They are **local-only gates** — they need
  that fixture tree, so CI cannot run them. **A green CI badge does not
  mean the baseline was checked.**
- CI (``.github/workflows/ci.yml``) runs the unit suite on
  ``ubuntu-latest`` and ``windows-latest`` for every push to ``main`` and
  every pull request.


Documentation
=============

User-facing:

- **Notebooks** — Jupyter notebooks explaining each workflow live
  under ``docs/notebooks/`` (inherited from the
  `upstream repository <https://github.com/Deltares/blueearth_cst/tree/main/docs/notebooks>`_).
- **HydroMT references** — ``docs/`` also contains HydroMT
  architecture and user-guide content.

Fork-specific (development):

- ``dev/roadmap.md`` — milestone roadmap, branching / tagging
  conventions, commit strategy.
- ``docs/install.md`` — step-by-step install walkthrough.
- ``dev/phase-1/`` — sealed foundation milestone artifacts (audits,
  plans, baseline diffs).
- ``dev/r01/`` … ``dev/r06/`` — sealed Phase 2 milestone designs and
  review records (modularity contracts, naming, the three workflows,
  structural refactor).
- ``dev/p31/``, ``dev/p32a/``, ``dev/p32b/``, ``dev/p33/`` — sealed
  Phase 3 milestone designs, review records and evidence notes.
- ``dev/followups.md`` — the open backlog, with closed items retained
  and dated.
- ``MIGRATION.md`` — the R6 rename map (old path → new path).
- ``CHANGELOG.md`` — release history (release-level; milestone detail
  lives in ``dev/roadmap.md``).


Publishing
==========

Docker
------

.. warning::

   **v0.1.0-alpha only.** The build / tag / push instructions below
   describe the upstream Deltares container registry workflow for
   the conda-based stack. Docker publishing is **still deferred** in the
   pixi-based fork — see "Deferred: Linux replication" in
   ``dev/roadmap.md``. It was *not* re-introduced during Phase 2 or 3, and
   is not currently scheduled. It remains blocked on the same thing as
   Linux replication (no Linux machine), though CI's green
   ``ubuntu-latest`` leg now shows the linux-64 half of ``pixi.lock``
   resolves and installs, which removes the largest unknown.

The entire workflow is contained in one Docker image. Build it:

.. code-block:: console

    docker build -t cst-workflow:0.0.1 .

Tag and push it under a new ``<<Tag>>``:

.. code-block:: console

    docker login -u <<deltares_email>> -p <<cli_secret>> https://containers.deltares.nl
    docker tag cst-workflow:0.0.1 containers.deltares.nl/CST/cst_workflows:<<Tag>>
    docker push containers.deltares.nl/CST/cst_workflows:<<Tag>>


License
=======

Copyright (c) 2021, Deltares.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
