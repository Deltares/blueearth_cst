---
title: Wflow documentation — bmi
source: "https://deltares.github.io/Wflow.jl/stable/user_guide/bmi.html"
reference: "Wflow.jl team. (n.d.). *bmi*. Wflow documentation. Retrieved May 25, 2026, from sources/tools/wflow/wflow-user-guide/07-bmi.md"
topic: tools
type: documentation
promoted: 2026-05-25
tags:
  - tools
  - wflow
accessed: 2026-05-25
---

## Introduction

The [Community Surface Dynamics Modeling System](https://csdms.colorado.edu/wiki/Main_Page) (CSMDS) has developed the Basic Model Interface (BMI). BMI consists of a set of standard control and query functions that can be added by a developer to the model code and makes a model both easier to learn and easier to couple with other software elements.

For more information see also: [http://csdms.colorado.edu/wiki/BMI\_Description](http://csdms.colorado.edu/wiki/BMI_Description)

CSDMS provides specifications for the languages C, C++, Fortran and Python. Wflow, written in the [Julia programming language](https://julialang.org/), makes use of the following [Julia specification](https://github.com/Deltares/BasicModelInterface.jl), based on BMI 2.0 version.

For the BMI implementation of wflow all grids are defined as [unstructured grids](https://bmi-spec.readthedocs.io/en/latest/model_grids.html#unstructured-grids), including the special cases `scalar` and `points`. While the input (forcing and model parameters) is structured (uniform rectilinear), internally wflow works with one dimensional arrays based on the active grid cells of the 2D model domain.

## Configuration

The variables that wflow can exchange through BMI are model state and output variables. The standard names of these variables based on [CSDMS Standard Names](https://csdms.colorado.edu/wiki/CSDMS_Standard_Names) should be listed below the `API` section in the `variables` list of the TOML configuration file. Below an example of this `API` section, that lists the variables that wflow can exchange through BMI:

Variables with a third dimension, for example `layer` as part of the `SBM` `soil` model, are exposed as two-dimensional grids through the wflow BMI implementation. For these variables the index of this third dimension is required, by adding `k` to the `layer` part of the variable standard name (`k` refers to the index of the third dimension): `layer_k`. For example, the variable `soil_layer_1_water_unsaturated_zone__depth` refers to the amount of water in the unsaturated store of the first soil layer of the `SBM` `soil` model.

The model state and output variable standard names are provided in the “Model parameters and variables” section of the [Model documentation](../model_docs/parameters_intro.html).
