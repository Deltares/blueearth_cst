BlueEarth Climate Stress Test toolbox
#####################################

The BlueEarth Climate Stress Test toolbox (blueearth_cst) is a free, open-source, and online toolbox for interactive climate risk assessment based on bottom-up analysis principles. 
The toolbox will enable end-users to: 

 - Explore the range of hydroclimatic uncertainty in a selected geographic area of choice, including natural variability and climate change signals.  

 - Design and execute a climate stress test for the response and vulnerabilities of user-defined thresholds and metrics.  

 - Make a judgment on the plausibility of vulnerabilities identified using climate model projections. As such, users should be able to estimate up to what extent the chosen metric or parameter may be sensitive to climate change. 

 - Provide a user-friendly tool with visualization elements that satisfy the needs and expectations of non-specialized audiences 

The Climate Stress Tester is part of the BlueEarth_ initiative and uses weathergenr_ as weather generator and Wflow_ for hydrological modelling.

.. image:: docs/_images/CST_scheme.png


.. _BlueEarth: https://blueearth.deltares.org/

.. _weathergenr: https://github.com/Deltares/weathergenr

.. _Wflow: https://github.com/Deltares/Wflow.jl


Installation
------------
BlueEarth CST is a python package that makes use of BlueEarth HydroMT to build the model (python), weathergenr to prepare the weather realization and stress tests (R), and Wlfow 
hydrological model (Julia). The installation steps are as follow:

 1. For both python and R installation we recommend using conda and `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

 2. Install Julia from https://julialang.org/downloads/ and Wflow https://deltares.github.io/Wflow.jl/dev/user_guide/install/#Installing-as-Julia-package

 3. Download (clone) the BlueEarth_cst ``git`` repo from `github <https://github.com/Deltares/blueearth_cst>`_, then navigate into the 
    the code folder (where the environment.yml file is located):

.. code-block:: console

    $ git clone https://github.com/Deltares/blueearth_cst.git
    $ cd blueearth_cst

 4. Make and activate a new blueearth-cst conda environment based on the environment.yml file contained in the repository. This will install all python and R dependcies to run the 
    tool:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate bluearth-cst

Running
-------
BlueEarth CST toolbox is based on several workflows developped using Snakemake_ . Three workflows are available:

 - Snakefile_model_creation: creates a Wflow model based on global data for the selected region and run and anlayse the model results for a historical period.
 - Snakefile_climate_projections: derives future climate statistics (expected temperature and precipitation change) for different RCPs and GCMs (from CMIP dataset).
 - Snakefile_climate_experiment: prepares futyre weather realizations and climate stress tests and run the realizations with the hydroloigcal model.

To prepare these workflows, you can select the different options for your model region and climate scenario using a config file. An example is available in the folder 
config/snake_config_model_test.yml.

You can then run the worflow from your choice using the snakemake command line, after activating your blueearth_cst conda environment:

.. code-block:: console

    $ conda activate bluearth-cst
    $ snakemake -s Snakefile_model_creation --configfile config/snake_config_model_test.yml  --dag | dot -Tpng > dag_all.png
    $ snakemake --unlock -s Snakefile_model_creation --configfile config/snake_config_model_test.yml
    $ snakemake all -c 1 -s Snakefile_model_creation --configfile config/snake_config_model_test.yml

The first line will activate your environment, the second creates a picture file recapitulating the different steps of the worflow, the third will if needed unlock your directory 
in order to save the future results of the workflow, and the fourth line runs the worflow (here for model creation).

With snakemake command line, you can use different options:

- -s: selection of the snakefile (workflow) to run (see list above).
- --config-file: name of the config file with the model and climate options.
- -c: number of cores to use to run the worflows (if more than 1, the workflow will be parallelized).
- --dry-run: retiurns the list of steps (rules) in the workflow that will be run, without actually running it.

There are many other options available, you can learn more in the `Snakemake CLI documentation <https://snakemake.readthedocs.io/en/stable/executing/cli.html>`_

More examples of how to run the worflows are available in the file run_snake_test.cmd .

.. _Snakemake: https://snakemake.github.io/


Documentation
-------------

Learn more about blueearth_cst in its `online documentation <http://deltares.github.io/blueearth_cst/latest/>`_


License
-------

Copyright (c) 2021, Deltares

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
