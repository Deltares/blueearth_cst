---
title: Preface
source: CST Development Phase Report_CLEAN_Feb14_2025 V2.docx
reference: "*Preface*. (n.d.). Retrieved May 11, 2026, from sources/tools/cst/cst-toolbox-technical-note-2025.md"
topic: tools
type: other
promoted: 2026-05-11
tags:
  - tools
  - climate-stress-testing
---

> Hydroclimatic Stress Testing Toolbox

Technical Note

July 2025

# 1. Preface

This report outlines the motivation and rationale behind the development of the Hydroclimatic Stress Testing Toolbox (CST), along with its technical features, underlying modeling and data components, and user workflow through the Graphical User Interface (GUI). CST is designed to support practitioners in climate stress testing of water investment projects for improving their long-term resilience. CST leverages automation tools and web technologies to integrate gridded datasets, modeling software, and interactive visualization tools. This integration enables rapid deployment of distributed hydrological models and the exploration of potential impacts on river flows under future climatic conditions.

Development of CST began in 2021, when the WB Global Water Practice (GWP) and Deltares initiated efforts to create an online platform for rapid hydroclimatic stress testing. The process was structured into three key activities. The first activity (A1), completed in the second quarter of 2021, focused on the conceptual design of the toolbox. This stage established its foundational structure, including a mockup GUI, user interactions, core functionalities, process components, workflows, basic software architecture, and data protocols.

The second activity (A2), completed in mid-2022, involved automating scripts and modeling tools. At this stage, a proof-of-concept version was developed without a graphical interface, demonstrating essential capabilities such as project setup and management, data extraction, rapid model building, climate stress test design, execution, and visualization.

The final activity (A3), completed in late 2023, resulted in a fully functional prototype that integrates scientific workflows, datasets, and modeling tools with a software backend and user interface. During this last phase, the prototype was tested across several geographic regions with distinct topographical and hydrological characteristics, including Mandrare (Madagascar), Birendranagar (Nepal), Tashiskari (Georgia), and multiple basins in Cambodia. These tests helped refine the toolbox’s functionality and adaptability across different environmental conditions.

Following its initial development, WB-GWP contracted Deltares in 2023 to provide technical support for climate risk assessments using the CST across ten basins. These included Kampe Omi and Kashimbilla (Nigeria), Yengice and Akusha (Azerbaijan), Chuza (Colombia), Nam Houm and Nam Xouang (Laos), and St. Paul-SP2 (Liberia). The contract was successfully completed in the fourth quarter of 2024 and provided valuable insights into common use cases for the toolbox along with the technical limitations of the underlying software. In 2024, the CST has been used in several ongoing World Bank projects, including Deltares’ contributions to the Country Climate Development Report (CCDR) for Lao PDR and the climate impact assessment of the Vakhsh Cascade in Kyrgyzstan.

CST is developed under an open-source GPL v3.0 license for ensuring free access for practitioners. The toolbox consists of three primary components: the BlueEarth CST workflows, which handle automation and data processing; the CST-API, which serves as the web backend; and the CST-frontend, which provides the user interface. These components are available via dedicated GitHub repositories. Besides these three components, CST relies on external software tools and datasets that are not available from the project repositories. Users must download and configure these additional resources to enable full functionality. Advanced users can integrate alternative modeling tools, such as different hydrological models, but this requires an in-depth understanding of the software architectures and additional pre-processing.

To make CST operational, its components must be installed and configured on a server alongside the recommended modeling tools and datasets. Running a new project in CST triggers various data processing and modeling procedures, requiring computational resources, storage, and bandwidth on the server side. These requirements vary based on the size of the project area (hydrological basins) and cannot be predicted in advance.

Currently, a preconfigured CST application is hosted on a Deltares server, available for limited use by designated World Bank staff. Advanced users can bypass the web interface and execute automation scripts directly from the command line for further customization. However, this functionality is currently restricted to internal Deltares use and is inaccessible via the internet due to technical, privacy, and security constraints.

# 1. Table of Contents

[Preface [3](#preface)](#preface)

[1 Introduction [6](#introduction)](#introduction)

[1.1 Overview [6](#overview)](#overview)

[1.2 Intended Uses of CST [8](#intended-uses-of-cst)](#intended-uses-of-cst)

[2 Technical Description [10](#technical-description)](#technical-description)

[2.1 Toolbox Architecture [10](#toolbox-architecture)](#toolbox-architecture)

[2.2 Main Components [11](#main-components)](#main-components)

[2.2.1 CST-frontend [11](#cst-frontend)](#cst-frontend)

[2.2.2 CST-API [11](#cst-api)](#cst-api)

[2.2.3 BlueEarth-CST Workflows [12](#blueearth-cst-workflows)](#blueearth-cst-workflows)

[2.3 Recommended Tools [12](#recommended-tools)](#recommended-tools)

[2.3.1 Model adapter: HydroMT [13](#model-adapter-hydromt)](#model-adapter-hydromt)

[2.3.2 Hydrological Modeling Framework: Wflow [14](#hydrological-modeling-framework-wflow)](#hydrological-modeling-framework-wflow)

[2.3.3 Weather Generator: weathergenr [15](#weather-generator-weathergenr)](#weather-generator-weathergenr)

[2.4 Recommended Datasets [16](#recommended-datasets)](#recommended-datasets)

[2.4.1 Non-meteorological Datasets [16](#non-meteorological-datasets)](#non-meteorological-datasets)

[2.4.2 Meteorological Datasets [17](#meteorological-datasets)](#meteorological-datasets)

[2.5 License governing the CST [19](#license-governing-the-cst)](#license-governing-the-cst)

[3 CST user workflow [21](#cst-user-workflow)](#cst-user-workflow)

[3.1 Step 1: Project Creation [22](#step-1-project-creation)](#step-1-project-creation)

[3.1.1 Screen 1 - Setup a Project [22](#screen-1---setup-a-project)](#screen-1---setup-a-project)

[3.1.2 Screen 2 - Define Project Area [23](#screen-2---define-project-area)](#screen-2---define-project-area)

[3.2 Step 2: Data and Model Configuration [25](#step-2-data-and-model-configuration)](#step-2-data-and-model-configuration)

[3.2.1 Screen 3 - Select Reference Climate Dataset(s) [25](#screen-3---select-reference-climate-datasets)](#screen-3---select-reference-climate-datasets)

[3.2.2 Screen 4 – Deploy and Simulate a Hydrology Model [25](#screen-4-deploy-and-simulate-a-hydrology-model)](#screen-4-deploy-and-simulate-a-hydrology-model)

[3.2.3 Screen 5 - Evaluate Model Performance [28](#screen-5---evaluate-model-performance)](#screen-5---evaluate-model-performance)

[3.3 Step 3: Hydroclimatic Stress Testing [29](#step-3-hydroclimatic-stress-testing)](#step-3-hydroclimatic-stress-testing)

[3.3.1 Screen 6 – Execute a Climate Stress Test [29](#screen-6-execute-a-climate-stress-test)](#screen-6-execute-a-climate-stress-test)

[3.3.2 Screen 7 - Explore Results [31](#screen-7---explore-results)](#screen-7---explore-results)

[4 Requirements and Limitations [35](#requirements-and-limitations)](#requirements-and-limitations)

[4.1 Expected User Profiles [35](#expected-user-profiles)](#expected-user-profiles)

[4.2 Functional requirements and limitations [36](#functional-requirements-and-limitations)](#functional-requirements-and-limitations)

[4.2.1 Project creation [36](#project-creation)](#project-creation)

[4.2.2 Data and Model Configuration [36](#data-and-model-configuration)](#data-and-model-configuration)

[4.2.3 Hydroclimatic stress testing [36](#hydroclimatic-stress-testing)](#hydroclimatic-stress-testing)

[4.3 Non-functional requirements and limitations [37](#non-functional-requirements-and-limitations)](#non-functional-requirements-and-limitations)

[4.3.1 Hosting requirements [37](#hosting-requirements)](#hosting-requirements)

[4.3.2 Installation requirements [37](#installation-requirements)](#installation-requirements)

[4.3.3 Performance and Accessibility [38](#performance-and-accessibility)](#performance-and-accessibility)

[5 References [39](#references)](#references)

[ANNEX [43](#annex)](#annex)

# 1. Introduction

## 1.1. Overview

Emerging evidence suggests that climate change is significantly disrupting the hydrological cycle, pushing it beyond previously observed patterns (IPCC, 2022). Many regions worldwide are experiencing more frequent and severe water-related hazards, including droughts, floods, storm surges, and landslides. To effectively adapt to these changing conditions, water planners must evaluate the risks posed by shifting hydrological patterns on existing infrastructure and future investments in the structural safety and long-term performance of that infrastructure.

Traditional approaches to estimating climate risks in water management often follow a "top-down" methodology, relying on scenarios generated from selected Global Circulation Models (GCMs) or Regional Climate Models (RCMs) under various emission pathways. Analyses typically begin with downscaling global projections to the regional and watershed level, and then analyzing them in water system models to estimate, for example future discharge or crop production. However, this conventional approach has significant limitations due to its heavy dependence on climate scenarios. While GCM outputs offer plausible projections of future climate conditions, they do not capture the full spectrum of uncertainties (Stainforth et al. 2007). GCMs may provide reasonable estimates of long-term trend changes at coarse spatial scales, but their reliability diminishes at the spatial scales relevant for water planning (Karger et al. 2020), and when estimating variability, seasonality, extreme events, and other factors critical to water systems. Additionally, internal climate variability—an important but often overlooked factor—is frequently neglected due to the high computational costs of accounting for it (Lehner et al. 2020). Overall, the uncertainty compounds with each step of the risk assessment process — for instance, when bias correcting coarse model outputs to relevant spatial scales and then translating hydrological outputs into localized impacts on water infrastructure (Lempert and Groves 2010; Moody and Brown 2012; Wilby and Dessai 2010). As the World Bank’s Independent Evaluation Group (IEG) highlights, “climate models have been more useful for setting context than for informing investment and policy choices” (IEG 2012).

Given these limitations, "bottom-up" methods provide an alternative approach by focusing on reducing system vulnerability to past, present, and future conditions ([Figure 1‑1](#_Ref188269253)). Bottom-up methods prioritize understanding the underlying factors that drive climate hazards and identifying strategies to enhance resilience in the face of uncertainty, unlike traditional approaches that rely heavily on climate scenarios (Maier et al. 2016). These assessments typically begin with "stress testing" the system under a wide range of plausible climate conditions to evaluate its performance and identify potential vulnerabilities (Brown 2010; Brown et al. 2012). Once key hazards are identified, the process works backward to determine how different climatic conditions could influence those risks (Fowler et al. 2024). Once risk is specified, robustness of alternative plans or infrastructure can be evaluated to support adaptation planning (Taner, Ray, and Brown 2017; M. Ü. Taner, Ray, and Brown 2019). In climate stress testing, scenarios are not directly tied to specific climate projections or forecasts but are often generated systematically using weather or climate generators. This approach allows stress tests to explore conditions beyond the range of existing climate projections and with a more adequate representation of local climatology. As a result, they offer significant flexibility, enabling assessments tailored to specific hazards in particular geographic contexts, whether evaluating individual threats or complex, compounding hazards.

Over the recent years, the climate stress testing methodology been widely applied by the World Bank to support climate-resilient decision-making across Africa, Asia, and Latin America. Some examples include basin-level investment planning in the Niger River Basin (Ghile, Moody, and Brown 2014) and the Brahmaputra River Basin (Yang et al. 2016), reservoir assessments in Peru (M. Taner et al. 2019) and Kenya (Ray & Taner, 2017), irrigation planning in Tanzania (Eluwa, 2024), and hydro-energy evaluations in Nepal (Ray et al., 2018). Additionally, climate stress testing has been incorporated as a critical component in various planning frameworks, such as the World Bank’s *Decision Tree Framework* (DTF) (Ray and Brown 2015) and UNESCO’s *Climate Risk Informed Decision Analysis* (CRIDA) (Mendoza et al., 2018) and the Hydropower Resilience Guidelines by the International Hydropower Association (IHA, 2019).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image1.png" style="width:4.41682in;height:4.31373in" />
<figcaption><p><span id="_Ref188269253" class="anchor"></span>Figure 1‑1. Conceptual steps of top-down and bottom-up climate risk assessment approaches</p></figcaption>
</figure>

While past applications have demonstrated the practical value of climate stress testing, its adoption remains limited to a small group of practitioners and has yet to become a standard practice in mainstream climate risk and adaptation planning. Several technical, operational, and financial barriers contribute to this limited uptake. First, the methodology requires the use and integration of various modeling tools, such as stochastic weather generators and water system models. In the absence of standardized, modular tools, setting up and linking these models for each project can be challenging. Second, the approach demands substantial data inputs, including meteorological time-series, land-use maps, and elevation data. While such information is available through global datasets, it often requires significant preprocessing before it can be used in modeling, adding to the complexity. Third, implementing climate stress tests requires specialized exploratory modeling algorithms and computational environments, which typically involve advanced programming and modeling skills that are not widely available outside academic and research institutions. Finally, without standardization and automation, individual applications can become resource-intensive, requiring significant time, effort, and budgets that exceed what is typically available for many projects.

In response to these challenges, the World Bank Global Water Practice (GWP) and Deltares have developed the Climate Stress Testing (CST) toolbox, a web application designed to facilitate the rapid implementation of climate stress testing for hydrological systems. The CST provides a Graphical User Interface (GUI) that integrates open-source modeling tools and freely available global data services to support hydroclimatic risk assessments at the basin scale anywhere in the world. By offering an intuitive, user-friendly interface, the toolbox enables practitioners with limited programming or analytical skills to perform comprehensive climate risk assessments directly through web screens.

The CST is designed to support long-term water-related decision-making by helping users understand the risks posed by water availability or flooding under a changing climate. It is expected to be used in rapid climate risk assessments at the World Bank, particularly in the application of protocols such as the Decision Tree Framework, the WSS Resilience Roadmap, and the Water Infrastructure Resilience Design Brief. Additionally, the toolbox will be a valuable resource for training and capacity-building workshops to demonstrate the principles of bottom-up risk assessment frameworks.

## 1.2. Intended Uses of CST

The CST was designed to simplify and streamline the bottom-up climate vulnerability assessment of hydrological systems, thereby supporting long-term policy decisions in the water sector. It serves as an accessible and user-friendly online tool, providing global capabilities to assess and quantify uncertainties and risks associated with changes in water availability, droughts, extreme weather events, and flooding.

The CST facilitates climate stress testing for any river basin worldwide by integrating freely available hydrological models, gridded datasets, and web technologies. Through CST’s graphical interface, users can delineate project areas, deploy and validate distributed hydrological models, and examine the impacts of climate change on river flow. Long-term vulnerability to climate change is assessed using key hydrological indicators, such as mean or maximum daily discharge and 7-day low flow.

Developed as a Decision-Making under Deep Uncertainty (DMDU) tool, CST differs from conventional climate assessment tools, which typically rely on predefined climate scenarios, such as those based on representative concentration pathways (RCPs). CST is flexible, allowing users to set up customized local climate scenarios by defining ranges of uncertainty in future temperature and precipitation using a weather generator. This capability enables users to conduct climate sensitivity analyses with locally consistent climate scenarios that extend beyond standard climate projections.

The CST is primarily intended for water professionals and practitioners involved in long-term climate policy decision-making, particularly those focused on river flow dynamics. It is especially useful for first-order vulnerability assessments in areas such as water availability, dam safety, navigation, flood control, and environmental flow management.

It is important to note that the CST is not intended for detailed engineering design of adaptation measures. The underlying modeling framework, which leverages gridded datasets and rapidly deployable hydrological models, is ideal for initial assessments to identify common challenges and key climate drivers. For more refined assessments, detailed hydrological modeling with local data calibration should be performed.

**What is a climate stress test?**

A climate stress test is a specialized form of sensitivity analysis used to assess how a system performs under different stressor combinations. For example, it can be applied to flood control infrastructure under varying storm surge levels.

The objective of a climate stress test is not to predict outcomes under a specific scenario, but to evaluate the system's robustness to potential future changes and identify conditions that may lead to undesirable results (e.g., infrastructure failure). By gradually varying the stressors, the test helps determine the critical thresholds where the system's performance is compromised.

A hydroclimatic stress test can conducted through the following steps:

1.  *Generate Climate Trajectories:* The test begins by producing a range of plausible future climate scenarios that explore changes in key weather variables, such as precipitation intensity, frequency, or duration. These climate trajectories are typically created using a weather generator, which preserves local variability and spatial correlations.
2.  *Simulate Hydrological Response:* The generated climate trajectories are then input into a hydrological model to simulate the impact of these conditions over time. The output is typically expressed as hydrological indicators, such as mean annual discharge, to quantify the system’s performance under changing climate conditions.
3.  *Analyze the Vulnerabilities:* The results are displayed on a climate response map, which correlates changes in system performance with shifts in climate variables, such as temperature and precipitation. This map helps identify problematic conditions where the system may become vulnerable. Additional climate information, such as summary statistics from global climate projections, can be overlaid on the response map to provide context. By examining the range of climate model outcomes, users can assess the plausibility of vulnerabilities or the level of concern.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image2.jpg" style="width:5.17986in;height:2.66143in" />
</figure>

# 1. Technical Description

## 1.1. Toolbox Architecture

The CST toolbox consists of a number of core components or building blocks that are preexisting or newly developed. These are a web frontend (graphical user interface), a web application programming interface (API), scientific workflows, modeling tools and gridded datasets ([Figure 2‑1](#_Ref188269302)).

1.  <u>Frontend:</u> The frontend is the element that the user sees and interacts with through buttons, menus, and figures. CST’s frontend consists of a set of functional pages to guide users through the climate risk assessment process. It enables users to interact with the toolbox, allowing them to input data, execute modeling procedures, and view results.
2.  <u>API:</u> Serving as the backbone of the web application, the API facilitates seamless communication between the frontend and the underlying workflows. It operates in the background and is not visible to the user.
3.  <u>Workflows:</u> Workflows are the building are a collection of scripts to schedule and execute different procedures or steps of the climate risk assessments, for example, creation of a project folder, basin delineation or model building. The workflows interact with external modeling tools and datasets.
4.  <u>Modeling tools:</u> Modeling tools are the preexisting software products or plugins that are used by the workflows when carrying activities. There are three major types of modeling tools including a model adapter for data extraction and processing, a weather generator to produce weather series under changing climatic conditions and a hydrological modeling software to convert weather series into discharge.
5.  <u>Datasets:</u> Datasets provide relevant information to climate risk assessment for hydrological modeling and climate stress testing. These include historical weather data, climate projections, elevation, soil and vegetation maps, and river widths.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image3.png" style="width:5.19245in;height:3.95041in" />
<figcaption><p><span id="_Ref188269302" class="anchor"></span>Figure 2‑1.Overview of the CST design</p></figcaption>
</figure>

## 1.2. Main Components

### 1.2.1. CST-frontend

The CST's front end is designed as a logical sequence of web screens, guiding users through every step of a climate risk assessment project. Each screen allows users to input information—via text fields, drop-down menus, or other interactive elements—execute workflows, generate previews, and explore results through dynamic visualizations. The interface consists of an about/contact page and seven key screens:

1.  *Project Catalog Page: A*llows users to create new climate risk assessment projects or browse existing ones while monitoring their progress and status.
2.  *Project Area Definition Page:* Enables the definition of the spatial extent of the study by delineating hydrographic regions of interest. Users can also specify additional point locations for risk calculations.
3.  *Baseline Climate Dataset Selection Page:* Provides options to select gridded meteorological datasets for historical temperature and precipitation variables from available datasets.
4.  *Hydrological Model Setup Page:* Facilitates the selection of relevant hydrological output variables, and deployment and execution of the model over a defined reference simulation period.
5.  *Hydrological Model Evaluation Page:* Allows users to evaluate model performance by comparing simulated discharge against observed data. This is supported by interactive charts and model goodness-of-fit criteria.
6.  *Stress Test Design and Execution Page:* Enables the configuration of boundary conditions for climate stress tests for example relative changes in mean temperature and precipitation, and execution of these tests.
7.  *Stress Testing Results Viewer Page:* Enables vulnerability assessment for each point location through climate response surfaces. Users can set vulnerability thresholds and toggle CMIP6 projections.

CST frontend is developed using a several freely available software products including [Dash framework](https://dash.plotly.com) for seamless integration with backend workflows and data handling, the [Leaflet](https://leafletjs.com) JavaScript library for interactive world maps, and [Plotly](https://plotly.com) for data visualization and plotting.

### 1.2.2. CST-API

The CST backend facilitates communication between the frontend and workflows while also managing information storage. It utilizes [FastAPI](https://fastapi.tiangolo.com/) to support Create, Read, Update, and Delete (CRUD) operations on an open-source, cross-platform database. The API's database is built on MongoDB, a scalable, document-oriented NoSQL database designed for high performance, flexibility, and user-friendliness. In each climate risk assessment project, the backend oversees project states, orchestrates containers executing workflow processes, and delivers results to the user.

The API is structured into several key modules, each serving a specific role. The routers' module defines FastAPI endpoints and injects dependencies, delegating requests to the appropriate services. Submodules like project manage endpoints for project data and runs, while container handles information on active containers. The models' module includes Pedantic models, such as Container and Project, responsible for encoding and decoding key objects used across the codebase. The adapters' module, implementing the repository pattern, abstracts interactions with external systems, including container management (container_repo), project data storage (project_repo), and region data storage (region_repo), with file storage managed via [Fsspec](https://filesystem-spec.readthedocs.io/en/latest/). Finally, the API is packaged as a [Docker image](https://www.docker.com/), making it compatible with workflow scripts and enabling endpoints to initiate workflow jobs.

For a more detailed overview of the CST-API, refer to the [online documentation:](https://github.com/Deltares/blueearth_cst_api/blob/master/docs/developer.md)

### 1.2.3. BlueEarth-CST Workflows

The analytical engine of the CST is powered by the BlueEarth CST workflows, which execute a series of tasks such as creating project folders, extracting data, and delineating river basins. These workflows are managed using [Snakemake](https://snakemake.readthedocs.io/en/stable/), a workflow management system chosen for its seamless integration with HydroMT and tools written in various programming languages, as well as its scalability across servers, clusters, grids, and cloud environments. The Snakemake workflows were developed and extended from existing BlueEarth scripts created by Deltares.

The CST includes three main snakemake workflows:

1.  *Project Creation Workflow:* handles project creation, including folder structure setup, basin delineation, climate and non-climate data extraction, model setup and simulation for the wflow hydrology model, plotting of model outputs, and computation of model goodness-of-fit statistics
2.  *Climate Projections Analysis Workflow:* extracts monthly temperature and precipitation data from CMIP6 outputs, processes and summarizes statistics for multiple models and scenarios, and generates visualizations to assess trends and variability.
3.  *Climate Stress Testing Workflow:* uses a stochastic weather generator to create a broad range of climate change scenarios. It conducts batch simulations of these scenarios using the wflow hydrology model and calculates a set of hydrological indicators from the simulation outputs.

The workflows are hosted on the [BlueEarth CST Workflows GitHub repository](https://github.com/Deltares/blueearth_cst). Each workflow

is configured to work with a set of recommended modeling tools and gridded datasets that are described in the following sections. The modeling tools include hydrological modeling software and plugins, stochastic weather generators, and model & data adapters. The datasets include meteorological data products providing gridded daily weather data and non-meteorological datasets such as global digital elevation maps, watersheds, land cover. Note that these software tools and datasets are not included within the CST repositories and must be accessed externally, either online or offline.

Advanced users can access the toolbox functionalities directly from the command prompt, bypassing the need for a GUI. This is achieved by running each workflow from a dedicated Python environment with manually configured setup files. This approach enables access to a wide range of model configuration and stress test settings, offering flexibility for tailored or in depth analyses that go beyond the capabilities of the GUI.

Further details about the workflows are provided in the Annex-A section of this report.

## 1.3. Recommended Tools

CST functionality is facilitated through a set of recommended (pre-existing) software. These are not included within the software repositories. Users need to download them and configure the CST setup files and workflow scripts to work with them based on the instructions provided. It is also possible to configure the toolbox with software and datasets beyond the list of recommended ones; however, this may require substantial scripting work.

### 1.3.1. Model adapter: HydroMT

HydroMT (Hydro Model Tools) is an open-source Python package designed to streamline the process of building and analyzing spatial geoscientific models, particularly those related to water systems (Eilander et al. 2023). HydroMT automates the entire workflow from raw data to a complete model instance, ensuring reproducibility and significantly reducing the time and effort required to set up models. Its modular design offers high flexibility, allowing users to customize and extend functionalities according to their specific needs, and it integrates seamlessly with various data sources, including remote sensing data, to enhance model accuracy and reliability ([Figure 2‑2](#_Ref188269345)).

HydroMT is focused on model setup and simulation analysis but does not include the underlying model software or datasets. Instead, it connects to models through plugins, which facilitate the pre-processing of raw data and the analysis of model outputs in a fast, reproducible manner. Examples of plugins include HydroMT-wflow (for distributed hydrology and sediment transport modeling), HydroMT-delwaq (for water quality modeling), HydroMT-sfincs (for fast 2D hydrodynamic flood modeling), and HydroMT-fiat (for flood impact modeling).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image4.png" style="width:5.40969in;height:3.3032in" />
<figcaption><p><span id="_Ref188269345" class="anchor"></span>Figure 2‑2. Schematic overview of HydroMT (Eilander et al. 2023)</p></figcaption>
</figure>

Why did we prefer HydroMT for the CST?

- Community-Driven: while maintained by Deltares, HydroMT is open-source and benefits from contributions by a diverse community of users and developers, ensuring continuous improvement and innovation.
- Comprehensive Documentation: with extensive documentation and tutorials, HydroMT is accessible to users of all experience levels, including those new to model building.
- Flexibility: its modular design allows for integration with other modeling software through plugins, enabling future expansion into other modeling domains, such as the CST.

Further description of HydroMT is provided in the Annex B and in the [HydroMT online manual](https://deltares.github.io/hydromt/stable/).

### 1.3.2. Hydrological Modeling Framework: Wflow

[Wflow](https://deltares.github.io/Wflow.jl/dev/) is a global hydrological and water system model. Developed by Deltares this tool allows users to account for precipitation, interception, snow accumulation and melt, evapotranspiration, soil water, surface water, groundwater recharge, and water demand and allocation in a fully distributed environment (Schellekens 2022). Wflow is conceived as a framework, within which multiple distributed model concepts are available, which maximizes the use of open earth observation data, making it the hydrological model of choice for data scarce environments. The vertical hydrological concepts available in wflow are HBV-96 (wflow_hbv), FLEXTopo (wflow_flextopo) and SBM (wflow_sbm). Based on gridded topography, soil, land use and climate data, wflow calculates all hydrological fluxes at any given grid cell in the model at a given time step. Wflow has been successfully applied worldwide in several peer-reviewed studies for analyzing and forecasting flood hazards (Emerton et al. 2016; Hally et al. 2015; Schaller et al. 2020), droughts and water availability (Rusli et al. 2024; Seizarwati and Syahidah 2021), climate change impacts (Bouaziz et al. 2022; Giardino et al. 2018; Leal Filho et al. 2016; Sperna Weiland et al. 2021) and land use changes (Gebremicael, Mohamed, and Van Der Zaag 2019; Hassaballah et al. 2017).

The CST uses wflow_sbm (Van Verseveld et al. 2024), the main hydrological model concept of the Wflow.jl framework. Wflow_sbm is based on Topog_SBM, which considers the soil to be a “bucket” with a saturated and unsaturated store. It uses static maps (raster-based model parameters) and meteorological forcing data to calculate all hydrological states and fluxes for each grid point and model time step. As a fully distributed, physically based hydrological model, wflow_sbm simulates all relevant hydrological processes, such as snow and glacier processes, rainfall interception, lakes, reservoirs, soil water balance and evaporation processes, vertical and horizontal subsurface flow, and overland and river flow ([Figure 2‑3](#_Ref185002306)).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image5.png" style="width:5.57323in;height:3.90879in" alt="Diagram of a tree with roots and water Description automatically generated" />
<figcaption><p><span id="_Ref185002306" class="anchor"></span>Figure 2‑3. An overview of hydrological processes in wflow_sbm</p></figcaption>
</figure>

Why wflow_sbm is preferred:

- Wflow is a free and open-source software that is used by an active community.
- Wflow can be easily coupled with other models and software applications, especially using the model adapter HydroMT
- Wflow has been applied in various catchments around the world, showing satisfactory performance (Van Verseveld et al. 2024)
- The most recent Julia implementation of Wflow provides various speed and efficiency benefits.
- Wflow_sbm, the vertical concept adopted in CST uses physics-based parameters, with clear distinction between subsurface, overland, and river flows between different cells. This makes it easy to build models using static maps (raster-based model parameters) from gridded datasets.
- Wflow has rigorous online manual and technical documentation (Schellekens 2022)

Further description of wflow and wflow_sbm is provided in the Annex A and in the [wflow technical manual](https://deltares.github.io/Wflow.jl/dev/user_guide/install/).

### 1.3.3. Weather Generator: weathergenr

Stochastic Weather generators (SWGs) are essential tools in climate stress testing to systematically and stochastically explore potential future climate conditions. CST is configured with [weathergenr](https://github.com/Deltares-research/weathergenr), an R package that is adapted from the semi-parametric multivariable, multigrid SWG developed by Steinschneider & Brown (2013). This enhanced version packages the core modeling scripts into a user-friendly R package, offering automation, improved processing speed, seamless handling of Netcdf data, and advanced plotting capabilities.

The foundational algorithm of weathergenr, derived from Steinschneider & Brown (2013), comprises three key components:

1.  *Wavelet autoregressive model:* This model accurately simulates area-averaged annual precipitation while reproducing the observed multi-year and multi-decadal variability. Such features are often neglected in many daily weather generators, which can result in poor representation of persistent wet or dry periods, such as those associated with ENSO events.
2.  *Markov chain and k-nearest-neighbor (KNN) resampling:* This component downscales annual precipitation into daily, multivariable, spatially disaggregated time series.
3.  *Quantile mapping:* This procedure adjusts the baseline daily distribution (e.g., mean and standard deviation) to align with a desired future distribution, thereby reflecting long-term shifts in climate and extending the resampled daily weather data beyond historical variability.

Recently, more advanced, weather-regime based weather generators have been developed (Rahat et al. 2022; Steinschneider et al. 2019). These tools enhance traditional methods by perturbing the dynamical and thermodynamic features of the climate, moving beyond the simple exploration of statistical changes in weather variables (Kucharski et al. 2024). While these SWGs offer distinct advantages, their application has so far been limited to specific regions and mainly restricted to academic studies.

Why weathergenr is preferred:

- Robustness across different climates: weathergenr is based on statistical time-series modeling methods that perform reliably under various climatic conditions, including extremely dry, very wet, and erratic rainfall regimes.
- Extensive validation and application: The tool has been successfully applied across numerous regions and rigorously validated through academic studies (François et al., 2021; Ray et al., 2018; Steinschneider et al., 2015; Taner et al., 2017; Whateley et al., 2015) and also used in past application of the World Bank’s Decision Tree Framework (Ray & Brown, 2015) and Climate Risk Informed Decision Analysis (Mendoza et al., 2018; Verbist et al., 2020).
- Enhanced usability and efficiency: The reformulated R package by Deltares improves the original algorithms, enabling easier integration with geospatial data and significantly enhancing computational efficiency.

## 1.4. Recommended Datasets

This section provides an overview of recommended meteorological and non-meteorological global gridded datasets configured with the CST toolbox.

Meteorological datasets serve two primary purposes: they are used as input for the hydrological model and to condition the stochastic weather generator. These datasets typically include daily time-series data of total precipitation, mean temperature, minimum and maximum temperatures, and potential evaporation at the grid level.

Non-meteorological datasets, on the other hand, are essential for setting up the wflow model for specific project locations. These include Digital Elevation Models (DEMs), as well as global datasets of lakes and reservoirs, glaciers, soil types, land use, or stream networks. Additional parameters derived from these datasets—such as soil hydraulic conductivity, rooting depth, and surface roughness—are also utilized.

Although the toolbox is currently configured to work with these specific datasets, its flexible design allows users to replace them with alternative datasets (e.g., those with higher accuracy or resolution) as long as the appropriate technical configurations are made. Also note that the CST repositories do not contain any gridded datasets directly; instead, they are accessed from a remote repository and specified through a data catalog file.

The list of recommended gridded datasets for the CST is provided in [Table 2‑1](#_Ref184915449).

### 1.4.1. Non-meteorological Datasets

[MERIT Hydro v1.0](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/) is a global high-resolution river and watershed dataset that includes river networks, drainage basins, and flow directions. Derived from global topographic data, it provides a resolution of 90 meters. In the CST, this dataset is used to delineate drainage areas and river networks for project areas.

[Rivers lin2019_v1](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL086405) is a global river network dataset derived from satellite altimetry. It features river flow directions and drainage basins at a resolution of approximately 30 arc-seconds (about 1 km). This dataset is frequently used in hydrological modeling and flood risk assessments. The CST incorporates it to enhance river geometry in project areas.

[HydroLakes](https://www.hydrosheds.org/products/hydrolakes) is a global dataset of lakes, including attributes such as area, elevation, and volume. It provides point data for lakes with a surface area greater than 1 hectare. Within the CST, HydroLakes is used to accurately represent lakes and reservoirs.

[Randolph Glacier Inventory v6.0](https://www.glims.org/RGI/randolph60.html) (RGI 6.0) is a global inventory of glacier outlines, covering both large and small glaciers. It provides vector data (polygons) with a resolution of approximately 30 meters, derived from satellite imagery. While not enabled by default in the CST, this dataset can be activated for areas with significant glacier presence.

[VITO Land Cover](https://lcviewer.vito.be/2015) provides land cover and vegetation maps derived from satellite imagery such as Sentinel-2. While resolutions vary, land cover maps typically offer a 10-meter resolution. This dataset is used in the CST to extract spatially distributed land-use and vegetation parameters for wflow modeling.

[MODIS Leaf Area Index](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD15A2H) (LAI) provides global estimates of the leaf area index, a key vegetation parameter. The dataset offers a spatial resolution of 500 meters with an 8-day temporal interval. It is used within the CST to derive additional grid-level vegetation parameters for wflow.

[SoilGrids](https://www.isric.org/explore/soilgrids) is a global dataset that provides estimates of soil properties (e.g., texture, pH, and organic carbon) at various depths, such as 0–5 cm and 5–15 cm. With a spatial resolution of 250 meters, it is derived from machine learning models trained on global soil profile data. The CST uses SoilGrids to generate soil-related parameters for wflow.

### 1.4.2. Meteorological Datasets

[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) is the fifth generation European Centre for Medium-Range Weather Forecasts (ECMWF) global reanalysis dataset that provides hourly climate data, including temperature, precipitation, wind, and other atmospheric variables. It offers a spatial resolution of approximately 31 km (0.28°) and is available from 1950 to the present. ERA5 is the default meteorological dataset in the CST due to its widespread use in hydrological modeling.

[CHIRPS](https://www.chc.ucsb.edu/data/chirps) (Climate Hazards Group InfraRed Precipitation with Station data) provides high-resolution global precipitation estimates based on a combination of satellite data and ground-based station records. With a spatial resolution of 0.05° (approximately 5 km) and daily intervals, it serves as an alternative to ERA5 in areas where ERA5 data may perform poorly.

[Coupled Model Intercomparison Project Phase 6](https://pcmdi.llnl.gov/CMIP6/) (CMIP6) is the latest phase of a collaborative international effort to provide climate projections for future global and regional climate changes. It includes simulations from over 40 climate models, offering projections for variables such as temperature, precipitation, sea-level rise, and extreme events under various greenhouse gas emission scenarios (RCPs/SSPs). Spanning future periods (e.g., 2021–2100), CMIP6 projections are used in the CST to generate summary statistics of precipitation and temperature for each model and scenario combination.

[Figure 2‑4](#_Ref188462796) presents a general overview of the BlueEarth CST workflows, along with the recommended modelling tools and datasets used in each.

<table>
<caption><p><span id="_Ref184915449" class="anchor"></span>Table 2‑1. Main features of the recommended datasets used in the CST</p></caption>
<colgroup>
<col style="width: 20%" />
<col style="width: 25%" />
<col style="width: 16%" />
<col style="width: 14%" />
<col style="width: 23%" />
</colgroup>
<thead>
<tr>
<th>Dataset</th>
<th>Description</th>
<th>Resolution</th>
<th>Approx. Size (GB)</th>
<th>Source</th>
</tr>
</thead>
<tbody>
<tr>
<td>MERIT Hydro v1.0</td>
<td>High-resolution hydrography map for river basins globally</td>
<td>1km</td>
<td>500</td>
<td>(Yamazaki et al. 2019)</td>
</tr>
<tr>
<td><p>Rivers</p>
<p>lin2019_v1</p></td>
<td>Global estimates of river widths</td>
<td>30m</td>
<td>5</td>
<td>(Lin et al. 2020)</td>
</tr>
<tr>
<td>HydroLakes</td>
<td>Database of global lake boundaries and attributes</td>
<td>varies</td>
<td>5</td>
<td>(Messager et al. 2016)</td>
</tr>
<tr>
<td>RGI v6.0</td>
<td>Randolph Glacier Inventory: dataset of global glacier outlines</td>
<td>~30m</td>
<td>5</td>
<td>(Pfeffer et al. 2014)</td>
</tr>
<tr>
<td>VITO</td>
<td>Remote sensing products including vegetation indices and biophysical variables</td>
<td>10m</td>
<td>10</td>
<td>(Buchhorn et al. 2020)</td>
</tr>
<tr>
<td>MODIS LAI</td>
<td>Leaf Area Index data from NASA's MODIS sensors</td>
<td>500m</td>
<td>10</td>
<td>(Myneni et al. n.d.)</td>
</tr>
<tr>
<td>SoilGrids</td>
<td>Global gridded soil information at fine spatial resolution</td>
<td>250m</td>
<td>5</td>
<td>(Hengl et al. 2017)</td>
</tr>
<tr>
<td>ERA5</td>
<td>Climate reanalysis dataset, providing weather and climate data</td>
<td>31 km (~0.28°)</td>
<td>1,000</td>
<td>Hersbach et al. (2020)</td>
</tr>
<tr>
<td>CHIRPS</td>
<td>Rainfall estimates from satellite and station data</td>
<td>0.05° (~5 km)</td>
<td>50</td>
<td>Funk et al. (2015)</td>
</tr>
<tr>
<td>CMIP6</td>
<td>Multimodel climate projections for past, present, and future climates</td>
<td>Varies by model</td>
<td>20,000*</td>
<td>Eyring et al. (2016)</td>
</tr>
</tbody>
</table>

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image6.png" style="width:5.70034in;height:3.93521in" />
<figcaption><p><span id="_Ref188462796" class="anchor"></span>Figure 2‑4. High-level sketch of BlueEarth CST workflows depicting the recommended tools (green) and datasets (gray) utilized in each workflow.</p></figcaption>
</figure>

## 1.5. License governing the CST

The CST toolbox components have been developed as open-source software (OSS) in accordance with the Terms of Reference (TOR) for "Consulting Services to Operationalize an Online Tool for Rapid Bottom-up Climate Risk Assessment" (World Bank 2022). Open-source software provides publicly accessible source code that anyone can inspect, modify, use, and distribute freely, subject to the terms of its license. A general explanation of OSS and license types is provided in Annex C of this report.

The developed CST components, viz., CST-frontend, CST-API, and the BlueEarth CST Workflows were released under GNU General Public License, Version 3 (GPL v3.0), in compliance with the TOR (World Bank 2022). [GPL v3.0](https://choosealicense.com/licenses/lgpl-3.0/) requires that licensed works and any modifications remain open source, with complete source code made available under the same license. Copyright and license notices must be retained, and contributors must grant express patent rights. However, a larger work using the licensed software via its provided interfaces may be distributed under different terms without requiring source code disclosure ([Table 2‑2](#_Ref189497951)).

| Permissions | Conditions | Limitations |
|----|----|----|
| Commercial use | Must provide source code when distributing | Cannot impose additional restrictions |
| Distribution | Must include GPL v3 license and copyright notice | No warranty or liability protection |
| Modification | Modified versions must also be GPL v3 licensed | Cannot add proprietary restrictions |
| Private use | No restrictions on private use | Patent grants apply if distributing |

<span id="_Ref189497951" class="anchor"></span>Table 2‑2. General features of the GNU GPL v3.0 License

Among the CST components, CST-frontend and CST-API were fully developed during this project under open-source license agreement. The BlueEarth CST workflows, were developed as modifications of preexisting scripts of Deltares, which are also open-source. Accordingly, all three components are licensed under GPL V3.0. These are all uploaded to GitHub repository, hence accessible to all, as per the ToR. [Table 2‑3](#_Ref184643315) provides the licensing details, and public repository URLs for these components.

| Component | Public URL | License |
|----|----|----|
| CST-API | <https://github.com/Deltares/blueearth_cst_api> | GPL v3.0 |
| CST-frontend | [github.com/Deltares/blueearth_cst_frontend](https://github.com/Deltares/blueearth_cst_frontend) | GPL v3.0 |
| Blueearth_cst workflows | [github.com/Deltares/blueearth_cst](https://github.com/Deltares/blueearth_cst) | GPL v3.0 |

<span id="_Ref184643315" class="anchor"></span>Table 2‑3. License and repository of the developed CST components

The developed CST components (in Table 2-3) comply with the licenses of underlying third-party software and plugins integrated into the system. These dependencies, along with their licensing details, are listed in Table 2.4.

<table>
<caption><p>Table 2‑4. External software tools and plugins used</p></caption>
<colgroup>
<col style="width: 27%" />
<col style="width: 48%" />
<col style="width: 24%" />
</colgroup>
<thead>
<tr>
<th>Third-party software</th>
<th>License</th>
<th>CST Component</th>
</tr>
</thead>
<tbody>
<tr>
<td>FastAPI</td>
<td>MIT</td>
<td rowspan="3">CST-API</td>
</tr>
<tr>
<td>MongoDB</td>
<td>Server-Side Public License (SSPL)</td>
</tr>
<tr>
<td>Docker</td>
<td>Apache 2.0</td>
</tr>
<tr>
<td>Dash</td>
<td>MIT</td>
<td rowspan="3">CST-frontend</td>
</tr>
<tr>
<td>Plotly</td>
<td>MIT</td>
</tr>
<tr>
<td>Leaflet</td>
<td>BSD 2-Clause</td>
</tr>
<tr>
<td>Snakemake</td>
<td>MIT</td>
<td>BlueEarth CST Workflows</td>
</tr>
</tbody>
</table>

Additionally, the CST system integrates preexisting software and datasets, whose licensing details are outlined in Tables 2.5 and 2.6, respectively.

| Software      | Public URL                                  | License  |
|---------------|---------------------------------------------|----------|
| Hydromt       | <https://github.com/Deltares/hydromt>       | MIT      |
| hydromt_wflow | <https://github.com/Deltares/hydromt_wflow> | MIT      |
| Wflow.jl      | <https://deltares.github.io/Wflow.jl/dev/>  | GPL v3.0 |
| weathergenr   | <https://github.com/Deltares/weathergenr>   | MIT      |

Table 2‑5. License and repository of recommended software (pre-existing)

<table>
<caption><p>Table 2‑6. License and repository of recommended datasets (third party)</p></caption>
<colgroup>
<col style="width: 27%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr>
<th>Dataset</th>
<th>License</th>
</tr>
</thead>
<tbody>
<tr>
<td>MERIT Hydro v1.0</td>
<td><p>Non-commercial use: CC-BY-NC 4.0</p>
<p>Commercial use: ODbL 1.0</p></td>
</tr>
<tr>
<td>Rivers lin2019_v1</td>
<td>CC-BY-NC 4.0</td>
</tr>
<tr>
<td>HydroLakes</td>
<td>Free for academic use</td>
</tr>
<tr>
<td>RGI v6.0</td>
<td>CC BY-SA 4.0</td>
</tr>
<tr>
<td>VITO</td>
<td>Copernicus Open Access License</td>
</tr>
<tr>
<td>MODIS LAI</td>
<td>NASA Open Data Agreement</td>
</tr>
<tr>
<td>SoilGrids</td>
<td>CC BY 4.0</td>
</tr>
<tr>
<td>ERA5</td>
<td>Copernicus License Agreement</td>
</tr>
<tr>
<td>CHIRPS</td>
<td>CC BY 4.0</td>
</tr>
<tr>
<td>CMIP6</td>
<td>Open-access (varies by dataset)</td>
</tr>
</tbody>
</table>

# 1. CST user workflow

The CST simplifies the process of conducting a bottom-up climate risk assessment for water practitioners through several, intuitive steps including defining the project area, deployment of a hydrology model and execution of a climate stress test. While the web interface automates these modeling and analysis steps, users are expected to have a basic understanding of climate risk assessment approaches and fundamental hydrology concepts.

Beyond the web interface, the CST provides advanced functionality through direct command-line access. This feature allows users to bypass the interface and achieve greater customization by modifying configuration files. Command-line access offers more control over the hydrological model, weather generator, and climate stress testing algorithms, and it also provides additional output files, such as raw simulation results. Currently, this feature is restricted to the development team but will soon be available to advanced users ("power users") with relevant expertise in programming, data structures, hydrological modeling, and climate stress testing. Further information on the different uses of the CST (i.e., standard and power user) is provided in the Requirements and Limitations (Section 4).

The standard user workflow through the CST web interface involves three primary phases across seven web screens ([Figure 3‑1](#_Ref188269600)):

1.  *Project Creation:* Users begin by creating a project within the Project Catalog (Screen 1). They then define the geographic area of interest, including basin boundaries and specific point locations, using the map interface (Screen 2).
2.  *Data and Model Configuration:* Users proceed to select a reference climate dataset that accurately reflects historical climatology (Screen 3). Next, they configure and deploy a hydrological model tailored to simulate future hydrological conditions. The model is run over the chosen reference period to produce baseline simulations (Screen 4). Users then validate the model by comparing simulated discharge outputs against observed data, ensuring accuracy before proceeding (Screen 5).
3.  *Hydroclimatic Stress Testing:* In this phase, users define the uncertainty ranges for future climatic changes and initiate the simulation experiment (Screen 6). After the experiment is complete, users analyze the results through interactive climate response surfaces (Screen 7).

<img src="../_images/cst-toolbox-technical-note-2025/image7.png" style="width:5.45996in;height:2.29716in" />

<span id="_Ref188269600" class="anchor"></span>Figure 3‑1. The typical CRA user workflow using the CST

## 1.1. Step 1: Project Creation

The typical user workflow through the CST’s graphical interface begins with creating a new project via the Project Catalog Page (Screen 1). In this initial step, users specify the geographic context of their analysis in the subsequent Define Project Area Page (Screen 2). Alternatively, users can browse existing projects (both ongoing and completed) by selecting the relevant project card in the catalog.

### 1.1.1. Screen 1 - Setup a Project

The Project Catalog ([Figure 3‑2](#_Ref185863995)) displays a comprehensive list of existing climate risk assessment projects in a card-based layout. Each project card includes key details such as the project title, geographic region, brief description, and current status (either "In Progress" or "Completed"). Additionally, users can search for specific projects by name or region using the search function.

<img src="../_images/cst-toolbox-technical-note-2025/image8.png" style="width:5.74792in;height:3.7625in" alt="A screenshot of a computer Description automatically generated" />

<span id="_Ref185863995" class="anchor"></span>Figure 3‑2. Screen 1 - Project Catalog Overview

To create a new project, users click the “New Project +” button on the upper right of the screen. This action opens a pop-up window ([Figure 3‑3](#_Ref185864194)) where users provide the following information:

- Project Name: A user-defined name for the project (spaces not allowed)

<!-- -->

- Analysis Period: The time-frame of the analysis, specified by start and end years

<!-- -->

- Short Description: A text field to provide additional context, such as a version number, analysis scope or project stakeholders.

Once the required fields are completed, users click the Create button to save the new project. The project is then added to the catalog and stored in the database, making it accessible for further configuration and analysis.

<img src="../_images/cst-toolbox-technical-note-2025/image9.png" style="width:5.74792in;height:3.59097in" alt="A screenshot of a computer Description automatically generated" />

<span id="_Ref185864194" class="anchor"></span>Figure 3‑3. Screen 1 - Project creation Pop-up Window

### 1.1.2. Screen 2 - Define Project Area

After creating a project, the next step is to define its geographic location. This involves selecting a hydrological outlet point and any additional points of interest within the same hydrographic region. These tasks are completed on the dedicated Define Project Area page.

The key steps in this process are as follows:

- Specify Basin Outlet Point: Users define the hydrological region by entering geographic coordinates (in decimal degrees) into the input field or by placing a marker on the zoomable interactive map ([Figure 3‑4](#_Ref187664872)-a).
- Select Hydrographic Region Type: Users choose between a basin or subbasin type and click the Generate Preview button to view a quick preview of the project’s drainage area.

  ([Figure 3‑4](#_Ref187664872)-b).

- Add Additional Points: Users can place additional markers on the interactive map to indicate other locations of interest for hydroclimatic risk estimation. These points must be within the same drainage area and located along the mainstem river or its tributaries ([Figure 3‑4](#_Ref187664872)-c)

Once the project area has been defined, users click the Confirm button to proceed to the next step. If the previewed area does not match the intended project area, users can reset the map using the dedicated reset button and repeat the area definition process.

<img src="../_images/cst-toolbox-technical-note-2025/image10.png" style="width:4.8535in;height:8.31683in" />

<span id="_Ref187664872" class="anchor"></span>Figure 3‑4. Screen 2 - Area Definition Process: a) User-specified outlet location of the hydrographic analysis area b) Preview of the upstream basin area of the same outlet location, c) Placement of an Additional point-location specified within the basin area.

## 1.2. Step 2: Data and Model Configuration

Once the project area is defined, the next step is to setup and configure a hydrological

simulation model that will be used for climate stress testing. This is done through three subsequent screens to specify the gridded climate dataset to represent the baseline climatology, configure, and simulate the hydrology model, and finally inspect the simulated daily discharge over a reference period.

### 1.2.1. Screen 3 - Select Reference Climate Dataset(s)

In the Climate Data screen, users select a gridded meteorological dataset to represent baseline climate conditions ([Figure 3‑5](#_Ref185875034)). The dataset selection influences both the hydrological model and the stochastic weather generator used in the climate risk assessment project.

Users can specify separate datasets for daily precipitation and temperature variables. For daily precipitation, users choose between two available datasets: ERA5 (ESMF Reanalysis v5) and CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data). For daily minimum and maximum Temperature, the only available option is ERA5.

The selected datasets serve as input for the hydrological model, which translates daily weather conditions into hydrological variables, and the weather generator, which produces new weather sequences based on historical data. It is important to note that the accuracy of these datasets varies by region, depending on factors such as the quality and density of weather stations, regional topography, data assimilation techniques, and inherent model biases.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image11.png" style="width:5.74792in;height:3.30556in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref185875034" class="anchor"></span>Figure 3‑5. Screen 3 - Baseline climate dataset selection for the active project</p></figcaption>
</figure>

### 1.2.2. Screen 4 – Deploy and Simulate a Hydrology Model

After selecting the reference climate dataset, users proceed to the next screen to configure the distributed hydrological model ([Figure 3‑6](#_Ref187671736)). For each project region, the CST deploys a customized distributed hydrological model using the wflow framework. This model extracts time-series data from the selected baseline climate dataset and integrates spatial data, such as topography, land cover, and soil properties, to simulate hydrological processes.

For each project region, the toolbox deploys a customized distributed hydrological model application. The wflow application extracts time-series data from the choice of baseline climate dataset, as well as spatially data including topography and the land cover and soil moisture datasets for modelling hydrological processes. Further description of the wflow framework and the model creation is provided in the technical description (Section 2).

Most of the model configuration and deployment process is automated in the backend through BlueEarth CST workflows. However, users can adjust the following settings:

- *Hydrological output variables:* users can select multiple options among four available outputs: river discharge (default), overland flow, actual evapotranspiration, groundwater recharge, and snow water equivalent. The river discharge is computed at point locations among the output variables, whereas all the remaining output variables are basin-averaged from the gridded results ([Table 3‑1](#_Ref187671668))
- Reference Simulation Period: Users specify the start and end years for the baseline hydrological model run. This period is used to assess the model’s performance and establish a baseline for climate stress testing.
- Upload Observed Flow Data: Users can optionally upload observed daily river discharge data to evaluate the model’s accuracy during the validation phase (Screen 5). This step is not mandatory for building and deploying the model.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image12.png" style="width:5.50425in;height:3.83376in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref187671736" class="anchor"></span>Figure 3‑6. Screen 4 - Hydrological model configuration and deployment setup</p></figcaption>
</figure>

<table style="width:100%;">
<caption><p><span id="_Ref187671668" class="anchor"></span>Table 3‑1. Hydrological output variables provided through CST</p></caption>
<colgroup>
<col style="width: 29%" />
<col style="width: 24%" />
<col style="width: 46%" />
</colgroup>
<thead>
<tr>
<th><strong>Variable</strong></th>
<th><strong>Units</strong></th>
<th><strong>Spatial and temporal features</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>River discharge</strong></td>
<td>cubic meters per second</td>
<td>Daily time-series reported at each grid, and computed at each output location</td>
</tr>
<tr>
<td><strong>Overland flow</strong></td>
<td>cubic meters per second</td>
<td rowspan="4">Daily time-series computed at each grid, and reported as basin-average values.</td>
</tr>
<tr>
<td><strong>Actual evapotranspiration</strong></td>
<td>millimeters per day</td>
</tr>
<tr>
<td><strong>Groundwater recharge</strong></td>
<td>millimeters per day</td>
</tr>
<tr>
<td><strong>Snow water equivalent</strong></td>
<td>Millimeters per day</td>
</tr>
</tbody>
</table>

Once the desired model settings are specified, users can initiate the model-building process by clicking the “Build model” button in the bottom-right corner of the screen. When the model-building process starts, a number of tasks are triggered from the “projection creation workflow”. These include extraction and processing of global datasets, adjusting and updating of wflow components and model parameters, and finally, simulating the wflow model over the baseline historical period.

The duration of the model-building and simulation process depends on the size and topographic complexity of the project region. For large basins, the process may take several hours or even days to complete. While the model is being built, users can monitor its progress by clicking the wheel icon in the upper-left corner of the screen. This opens a workflow progress overview window, which displays the real-time status of the model builder and other workflows. Alternatively, users can track the workflow progress directly from the relevant card in the Project Catalog (Screen 1).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image13.png" style="width:5.31714in;height:3.72399in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p>Figure 3‑7. Screen 4 - Workflow pop-up window in the upper-right corner of the page.</p></figcaption>
</figure>

### 1.2.3. Screen 5 - Evaluate Model Performance

Once the model deployment and reference period simulation are complete, the Model Evaluation screen is activated ([Figure 3‑8](#_Ref187674184)). This screen enables users to review the simulated daily discharge and compare it against observed data, if available. This comparison helps determine whether the deployed hydrological model produces reliable results, ensuring it is suitable for the subsequent climate risk assessment. Users can assess the model's performance and accuracy at each output point location using two methods:

- Visual Inspection: an interactive time-series plot displays the simulated daily discharge alongside the observed discharge data (if provided). Users can zoom in and out to explore the simulation results over specific time periods, including key historical events such as severe floods. This visual comparison allows for quick identification of discrepancies and patterns.
- Hydrological Performance Metrics: users can evaluate the model's accuracy using a set of goodness-of-fit metrics, which compare the simulated discharge to the observed values. The toolbox calculates several criteria that are widely used in hydrological modeling studies as shown in Table 3‑2 (Althoff & Rodrigues 2021).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image14.png" style="width:5.43377in;height:2.96012in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref187674184" class="anchor"></span>Figure 3‑8. Screen 5 – Evaluation of simulated discharge (reference period)</p></figcaption>
</figure>

| Metric | Recommended use | Value range | Ideal value |
|----|----|----|----|
| Nash-Sutcliffe Efficiency (NSE) | Overall performance | (-∞, 1\] | 1 |
| Kling-Gupta Efficiency (KGE) | Overall performance, but more balanced btw. low and peak flows | (-∞, 1\] | 1 |
| Log Nash-Sutcliffe Efficiency (NSElog) | Overall and low-flow performance | (-∞, 1\] | 1 |
| Volumetric Efficiency (VE) | Low-flow performance | \[0, 1\] | 1 |
| Percentage Bias (PBIAS) | Systematic under or overestimation | (-∞, ∞\] | 0 |

Table 3‑2. Goodness-fit criteria included in the hydrological model evaluation step

When the user is satisfied with the model performance, they click on the 'next' button to proceed to the climate stress test design/execution page.

## 1.3. Step 3: Hydroclimatic Stress Testing

The third step of the workflow involves designing and executing an automated hydroclimatic stress test experiment for the project area ([Figure 3‑9](#_Ref187691162)), followed by analyzing the results using climate response surfaces ([Figure 3‑10](#_Ref188269802)).

### 1.3.1. Screen 6 – Execute a Climate Stress Test

The climate stress test design screen allows users to implement a customized stress test based on the most up-to-date CMIP6 climate projections. The user interface highlights key configuration options while simplifying complex weather generation and data post-processing steps. The climate stress testing process consists of three primary tasks:

In this step, users analyze projected future climatic changes between the reference (baseline) period and the future analysis period. These changes are calculated by extracting gridded time-series data from the CMIP6 ensemble for each climate model and representative concentration pathway (RCP), followed by averaging over the project's spatial extent and analysis period.

The results are displayed as an interactive scatter plot (Figure 13), where users can:

- Toggle RCPs to focus on specific climate scenarios.
- Hover over a point on the scatter plot to view additional details, such as the underlying climate model and associated precipitation and temperature change levels.

This summary information from the CMIP6 ensemble provides a foundation for designing the climate stress test experiment.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image15.png" style="width:5.46612in;height:3.46247in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref187691162" class="anchor"></span>Figure 3‑9. Screen 6 - Climate stress test design screen with CMIP6 data on the right.</p></figcaption>
</figure>

The CST offers flexibility for users to design a tailored climate stress test experiment by adjusting the bounds of future climate change perturbations and the sampling size of natural variability. Currently, the CST interface allows users to specify future climate changes using three key parameters:

- Mean Temperature Change (°C)
- Mean Precipitation Change (%)
- Precipitation Variability Change (%)

For each of these perturbations, the users specify the following parameters:

- Lower Bound of Change: The minimum relative change to be explored between the future and baseline periods.
- Upper Bound of Change: The maximum relative change to be explored between the future and baseline periods.
- Transient Changes: Whether the changes occur gradually over time or immediately from the start of the simulation.
- Apply to Specific Months: Users can choose whether changes apply uniformly throughout the year or only during specific months.

To account for natural climate variability, the CST applies these changes to multiple realizations of historical weather records generated by a stochastic weather generator. Each realization represents a plausible weather sequence, creating a diverse range of scenarios and improving consideration of internal climate variability. Users can specify the number of realizations through the interface (default value: 3). It is important to note that increasing the number of realizations will require more computational resources, potentially extending processing time.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image16.png" style="width:5.57481in;height:3.44309in" alt="Screens screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref188269802" class="anchor"></span>Figure 3‑10. Screen 6 - Climate stress test design – display of climate change perturbation and natural variability sampling settings</p></figcaption>
</figure>

After defining the uncertainty ranges and climate change parameters, users can initiate the stress test by clicking the “Create Stress Test” button in the bottom-right corner of the screen. This triggers the backend climate stress test workflow, which performs the following tasks:

- Generates new daily weather sequences based on the specified perturbations and realizations.
- Simulates each generated weather sequence using the hydrological model developed in Step 2.
- Finalizes the results by calculating hydrological indicators that express changes in long-term mean conditions, low flows, and high flows.

The climate stress test process is the most time-intensive step in the workflow. Depending on the size and complexity of the project area, the process can take anywhere from several hours (for small sub-catchments) to several days (for large sub-catchments).

### 1.3.2. Screen 7 - Explore Results

The final screen of the climate stress test toolbox to assess the vulnerability and risk from climate change. This is done interactive climate surfaces that combine the results of the climate stress test and the climate information from CMIP6 projections.

The process involves a number of steps as described below:

1.  *Selection of a location:* User selects a location of analysis from the list of predefined point-locations among the point-locations specified during the project area definition (see Step 1).
2.  *Selection of Performance metric:* User specifies a performance metric to summarize the daily discharge series calculated from the climatic changes explored through the climate stress test. Each of these metrics are calculated over the entire simulation period and focus on different aspects of the changing discharge patterns regarding long-term mean conditions, low and peak flows. Currently, the CST interface only displays discharge related metrics and the metrics calculated for the other hydrological output variables, i.e., overland flow, actual evaporation, groundwater recharge, and SWE are available from the command-line for the power users. The list of available metrics from the interface are provided in [Table 3‑3](#_Ref187755512).
3.  *Specify a vulnerability threshold:* Using a slider tool, users can set a vulnerability threshold for the selected metric and location. This threshold is used to parse the results into acceptable versus non-acceptable performance conditions. The threshold value is often set based on expert judgment or based on external information available, for example, a maximum permissible flood level or a minimum flow requirement that needs to be exceeded.

Based on these three options (Location for analysis, performance metric, and the vulnerability threshold), the interactive climate response surface is updated on the right-side of the screen ([Figure 3‑11](#_Ref187755227)). The climate response surface shows how the climate response of the daily discharge metric will vary under climate change, across the changes in mean precipitation (%) on the horizontal axis and the changes in mean temperature (°C) on the vertical axis respectively. The vulnerability threshold is set to distinguish the conditions of not acceptable and acceptable outcomes through blue and red colors in the response surface.

<table style="width:94%;">
<caption><p><span id="_Ref187755512" class="anchor"></span>Table 3‑3. Hydrological performance metrics displayed through the CST interface</p></caption>
<colgroup>
<col style="width: 30%" />
<col style="width: 22%" />
<col style="width: 41%" />
</colgroup>
<thead>
<tr>
<th>Output statistics</th>
<th>Suitability</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><p>Summary statistics:</p>
<p>min, max, mean</p></td>
<td>All output variables</td>
<td>Daily discharge statistics calculated over the simulation period.</td>
</tr>
<tr>
<td><p>Empirical percentiles:</p>
<p>Q5, Q95</p></td>
<td>All output variables</td>
<td>Daily discharge values corresponding to the 5th and 95th percentiles</td>
</tr>
<tr>
<td>Q7-Day min</td>
<td>Discharge only</td>
<td>The average of minimum daily discharge over a consecutive seven-day flow</td>
</tr>
<tr>
<td>Q7-Day max</td>
<td>Discharge only</td>
<td>The average of maximum daily discharge over a consecutive seven-day flow</td>
</tr>
<tr>
<td>Wettest month mean</td>
<td>Discharge only</td>
<td>Daily mean discharge within the wettest month (over the entire period)</td>
</tr>
<tr>
<td>Driest month mean</td>
<td>Discharge only</td>
<td>Daily mean discharge within the driest month (over the entire period)</td>
</tr>
<tr>
<td>Flood: 10-year return-period</td>
<td>Discharge only</td>
<td>Peak flow with a 10-year return frequency</td>
</tr>
<tr>
<td>Drought: 10-year return-period</td>
<td>Discharge only</td>
<td>Low flow with a 10-year return frequency</td>
</tr>
<tr>
<td>Base flow index (BFI)</td>
<td>Discharge only</td>
<td>Contribution of baseflow to the total flow</td>
</tr>
</tbody>
</table>

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image17.png" style="width:5.49623in;height:3.05324in" />
<figcaption><p><span id="_Ref187755227" class="anchor"></span>Figure 3‑11. Screen 7 - Climate vulnerability assessment using response surfaces</p></figcaption>
</figure>

CST provides users the option to evaluate the climate vulnerability assessment results (from climate stress test) along with the climate information summarized from the CMIP6 projections. This is done by overlaying the CMIP data (i.e., “GCM dots”) on the of the response surface for plausibility assessment ([Figure 3‑12](#_Ref188443682)). Based on the spread of GCM projections over the response surface, users can make a judgement on whether the underlying climatic conditions are likely to occur based on available climate information.

Users can toggle the CMIP6 projections on the interface:

- RCP scenarios: SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5 (see [Table 3‑4](#_Ref190258187)).
- Climate models: Selection among twenty-one climate models (see Table 3‑5).

| Scenario | Description | Assumptions |
|----|----|----|
| SSP1-2.6 | Sustainability (Low Challenges): Emphasizes sustainable development and reduced emissions. | Strong international cooperation; Investments in renewable energy; Low population growth and high education; Efficient resource use and reduced inequality. |
| SSP2-4.5 | Middle of the Road (Intermediate Challenges): A balance between economic, social, and environmental goals. | Moderate population growth and technological advancements; Mixed energy use with both fossil fuels and renewables; Gradual progress in climate policies. |
| SSP3-7.0 | Regional Rivalry (High Challenges): Focuses on regional priorities, with limited cooperation on climate issues. | High population growth in developing regions; Weak climate policies and reliance on fossil fuels; Economic development is uneven, with reduced globalization. |
| SSP5-8.5 | Fossil-Fuelled Development (Very High Challenges): High economic growth driven by fossil fuel consumption. | High reliance on fossil fuels and energy-intensive lifestyles; Rapid population growth and urbanization; Minimal climate mitigation efforts. |

<span id="_Ref190258187" class="anchor"></span>Table 3‑4 Description of the IPCC's SSP scenarios used in the CST

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image18.png" style="width:5.74792in;height:3.26667in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref188443682" class="anchor"></span>Figure 3‑12. Screen 7 – Assessing plausibility and risk using CMIP6 projections</p></figcaption>
</figure>

| Model Name | Institution | Country | Resolution |
|----|----|----|----|
| IPSL-CM6A-LR | Institute Pierre-Simon Laplace | France | 2.5° x 1.25° |
| SAM0-UNICON | Seoul National University | South Korea | 1.25° x 1.0° |
| CESM2 | NCAR | USA | 1.25° x 0.9° |
| CESM2-WACCM | NCAR | USA | 1.25° x 0.9° |
| INM-CM4-8 | Institute of Numerical Mathematics | Russia | 2.0° x 1.5° |
| INM-CM5-0 | Institute of Numerical Mathematics | Russia | 2.0° x 1.5° |
| GFDL-ESM4 | NOAA GFDL | USA | 1.0° x 1.0° |
| NorESM2-LM | Norwegian Climate Centre | Norway | 2.5° x 1.9° |
| KACE-1-0-G | NIMS-KMA | South Korea | 1.25° x 1.0° |
| FGOALS-f3-L | Chinese Academy of Sciences | China | 1.0° x 1.0° |
| ACCESS-CM2 | CSIRO-ARCCSS | Australia | 1.875° x 1.25° |
| NorESM2-MM | Norwegian Climate Centre | Norway | 1.25° x 0.9° |
| ACCESS-ESM1-5 | CSIRO | Australia | 1.875° x 1.25° |
| CESM2-WACCM-FV2 | NCAR | USA | 2.5° x 1.9° |
| CESM2-FV2 | NCAR | USA | 2.5° x 1.9° |
| CMCC-CM2-SR5 | CMCC | Italy | 1.25° x 0.9° |
| TaiESM1 | AS-RCEC | Taiwan | 1.25° x 1.0° |
| NorCPM1 | Norwegian Climate Centre | Norway | 1.0° x 1.0° |
| IPSL-CM5A2-INCA | Institute Pierre-Simon Laplace | France | 3.75° x 1.875° |
| CMCC-CM2-HR4 | CMCC | Italy | 0.75° x 0.75° |
| CMCC-ESM2 | CMCC | Italy | 1.25° x 0.9° |
| IPSL-CM6A-LR-INCA | Institute Pierre-Simon Laplace | France | 2.5° x 1.25° |
| E3SM-1-0 | E3SM Project | USA | 1.0° x 1.0° |

Table 3‑5. CMIP6 Models included in the CST

# 1. Requirements and Limitations

This section outlines the functional and non-functional requirements, as well as the limitations, of the CST Toolbox.

## 1.1. Expected User Profiles

The CST provides a streamlined and generalized approach to climate stress testing. While the tool automates various steps using state-of-the-art data adapters, modeling tools, and visualizations, users are expected to have foundational knowledge in the following areas:

- Climate vulnerability analysis – including key concepts such as scenario analysis, climate projections, risk assessment, resilience, and robustness.
- Basic hydrology – encompassing the water cycle, basin delineation, and meteorological datasets.
- Interpretation of climate variability and extremes – understanding how climate patterns influence hydrological and infrastructure risks.

  The CST Toolbox is designed to accommodate two primary user groups: standard users and power users:

<!-- -->

- *Standard users* interact primarily through the graphical user interface (GUI). They do not require advanced technical knowledge, programming skills, or familiarity with the underlying tools. Their access is limited to predefined configuration options available within the GUI.
- *Power users* engage with the CST Toolbox via the command-line interface (CLI). They can modify advanced settings through configuration files, such as adjusting parameters for the weather generator or hydrological model. Power users also have access to raw model outputs and can execute automation scripts for further customization.

  [Table 4‑1](#_Ref190094015) summarizes the main differences between the distinct use cases.

<table style="width:100%;">
<caption><p><span id="_Ref190094015" class="anchor"></span>Table 4‑1. CST: comparison of standard and advanced user features and requirements</p></caption>
<colgroup>
<col style="width: 39%" />
<col style="width: 27%" />
<col style="width: 32%" />
</colgroup>
<thead>
<tr>
<th>Feature/requirement</th>
<th>Standard user</th>
<th>Power User</th>
</tr>
</thead>
<tbody>
<tr>
<td>Access to CST</td>
<td>Via GUI</td>
<td>Via command-line</td>
</tr>
<tr>
<td>Access to customization options</td>
<td>Limited to GUI settings</td>
<td>Extensive via YAML files</td>
</tr>
<tr>
<td>Access to results</td>
<td>GUI results page; additional data files available upon request</td>
<td>Static visuals, raw outputs, and summary files (Netcdf, CSV)</td>
</tr>
<tr>
<td>Advanced integration with other modeling tools (e.g., WEAP)</td>
<td>Not supported</td>
<td>Possible via raw output files (requires extensive setup)</td>
</tr>
<tr>
<td><p>Programming skills needed</p>
<p>(Python, R, and/or Julia)</p></td>
<td>Minimal</td>
<td>Advanced</td>
</tr>
<tr>
<td><p>Technical skills needed</p>
<p>(climate, hydrology modeling)</p></td>
<td>Minimal</td>
<td>High</td>
</tr>
</tbody>
</table>

Currently, power user functionality is restricted to internal Deltares developers due to technical, privacy, and security constraints. This functionality is inaccessible via the internet. However, these limitations may be addressed in future development phases.

The remainder of this section focuses on the requirements and limitations for standard

users interact with the CST Toolbox through the graphical interface and preconfigured setup files.

## 1.2. Functional requirements and limitations

Functional requirements and limitations refer to core features and capabilities of the automated toolbox. These affect the toolbox’s ability to deliver specific tasks or outcomes expected by users and evaluated under three phases of the user workflow that are Project creation, Data and Model Configuration, and Hydroclimatic Stress Testing (see section 3).

### 1.2.1. Project creation

The CST toolbox is preconfigured to enable users to define a project at any point-location across the world (see section 3.1). This is achieved through automated data extraction and basin delineation workflows calls from the backend. To ensure reliable performance, users should follow these guidelines:

- The spatial extent of a climate risk assessment must be within a single hydrological drainage basin or subbasin.
- Projects should contain fewer than 10 points of interest to manage data storage requirements.
- Recommended basin size ranges between 25 and 250,000 km². Smaller basins may result in resolution errors, while larger basins can be computationally challenging. Actual computational demands may vary based on topography.
- Basin delineation algorithms support diverse topographic conditions, but may struggle in flat plains or arid/desert regions.
- All point locations (including basin outlet points) should be placed on a mainstem river, tributaries, or nearby. Placing them on lakes, reservoirs, or open land may cause delineation errors.

### 1.2.2. Data and Model Configuration

The CST Toolbox is preconfigured to deploy distributed hydrological models in a given project area based on gridded data products. To ensure reliable results and computational efficiency, the following assumptions and recommendations apply:

- The CST models natural basin responses to climate conditions. Human interventions (e.g., reservoirs, water withdrawals) are not accounted for.
- The default model resolution is 0.008983° x 0.008983° (~1 km x 1 km at the equator), balancing detail and computational efficiency. Power users can modify setting via YAML files.
- Model accuracy depends on the quality of input data, including meteorological forcing, digital elevation, land cover, and soil maps. Regional variations may impact results.
- The hydrological models are not calibrated to specific basins. Users should validate results using observed data uploads and model goodness-of-fit metrics (Screen 4).

### 1.2.3. Hydroclimatic stress testing

The CST Toolbox automates climate stress testing by generating stochastic weather scenarios based on user-defined temperature and precipitation change bandwidths. These time-series are simulated through the hydrological model, and performance metrics are calculated. While the current CST implementation is robust across diverse climatic conditions, the following limitations apply:

- Climate projections are summarized via scatter plots in the GUI. Additional visuals (e.g., annual trends, cycles) are generated but accessible only via CLI.
- The stress test allows only temperature and precipitation perturbations, applied uniformly or for specific months.
- Incremental steps between minimum and maximum temperature/precipitation changes are limited to three levels to manage server-side computational resources. This setting can be modified in future phases.
- The stochastic weather generator is designed for robust performance, but may fail in extreme basins where wet/dry states are not well-defined. Adjustments are possible only via CLI.
- Hydrological indicators are limited. For example, flood return periods are capped at 10 years. Reliable calculations for longer periods (e.g., 50 or 100 years) require more extensive data and computing power.
- The climate response surface visualization in the GUI uses a preset red-blue color scheme, which cannot be changed.

Future improvements may address these limitations by enhancing user customization from the GUI (e.g., climate stress test design and stochastic weather generation), computational capacity, and hydrological indicator options.

## 1.3. Non-functional requirements and limitations

Non-functional requirements define the software’s attributes, including installation, configuration, performance, security, and scalability. These aspects influence the behavior of the CST toolbox rather than its core functionality. This section provides an overview of the hosting and performance requirements for the CST.

### 1.3.1. Hosting requirements

During the development and initial deployment phase, a preconfigured, fully functional CST application is hosted on a Deltares server. This setup allows designated World Bank staff limited access for a predefined period. Future hosting by the World Bank or third parties must meet the following requirements:

- Server Specifications: A minimum of a 2 GHz quad-core processor and 16 GB RAM. Parallel execution of multiple projects may necessitate additional processing power based on workload demands.
- Storage: At least 20 TB of storage space to accommodate gridded datasets, project inputs, configurations, and output files.
- Internet Connectivity: A stable and high-speed internet connection is required, with a recommended minimum bandwidth of 5 Mbps.

We note that each CST project initiates extensive data processing and modeling, which demands significant computational resources, storage, and bandwidth. These requirements vary depending on the project area’s size (e.g., hydrological basins) and cannot be precisely estimated in advance.

### 1.3.2. Installation requirements

At its current development stage, installing the CST toolbox requires a thorough understanding of its underlying software components and database structures. To ensure proper functionality, all essential components must be correctly installed and configured on either a personal computer or a server.

The required components include:

- *Dependencies:* Command-line deployment of Python, R, and Julia, along with their necessary packages.
- *Core components and software:* wflow software, weathergenr, CST-frontend, CST-API, and BlueEarth CST workflows.
- *Configuration*: Setting up appropriate paths for executable files and dataset locations after installation.

Due to the complexity of the installation process and the involvement of multiple components, deployment requires significant effort and specialized technical expertise. Future updates aim to simplify installation by introducing containerized deployment solutions.

### 1.3.3. Performance and Accessibility

The performance of the CST toolbox depends on server specifications and the characteristics of individual projects, such as catchment size, topography, and user-defined parameters.

Key performance considerations include:

- Initial Performance Assessment: The current CST implementation on the Deltares server provides a baseline for evaluating toolbox performance and execution times. Initial assessments indicate that a complete climate stress test analysis can take anywhere from several hours for regions smaller than 10,000 km² to multiple days for regions exceeding 100,000 km².
- Concurrent Processing: Parallel execution of multiple CST projects has not yet been rigorously tested. At this stage, it is recommended to run one project at a time to ensure optimal performance and stability.

# 1. References

Althoff, Daniel, and Lineu Neiva Rodrigues. 2021. “Goodness-of-Fit Criteria for Hydrological Models: Model Calibration and Performance Assessment.” *Journal of Hydrology* 600:126674. doi: 10.1016/j.jhydrol.2021.126674.

Bouaziz, L. J. E., E. E. Aalbers, A. H. Weerts, M. Hegnauer, H. Buiteveld, R. Lammersen, J. Stam, E. Sprokkereef, H. H. G. Savenije, and M. Hrachowitz. 2022. “Ecosystem Adaptation to Climate Change: The Sensitivity of Hydrological Predictions to Time-Dynamic Model Parameters.” *Hydrol. Earth Syst. Sci.* 26:1295–1318. doi: 10.5194/hess-26-1295-2022.

Brown, Casey. 2010. “Decision-Scaling for Robust Planning and Policy under Climate Uncertainty.” *WIREs Climate Change* 14.

Brown, Casey, Yonas Ghile, Mikaela Laverty, and Ke Li. 2012. “Decision Scaling: Linking Bottom-up Vulnerability Analysis with Climate Projections in the Water Sector.” *Water Resources Research* 48(9). doi: 10.1029/2011WR011212.

Buchhorn, Marcel, Bruno Smets, Luc Bertels, Bert De Roo, Myroslava Lesiv, Nandin-Erdene Tsendbazar, Martin Herold, and Steffen Fritz. 2020. “Copernicus Global Land Service: Land Cover 100m: Collection 3: Epoch 2015: Globe.”

Eilander, Dirk, Hélène Boisgontier, Laurène J. E. Bouaziz, Joost Buitink, Anaïs Couasnon, Brendan Dalmijn, Mark Hegnauer, Tjalling de Jong, Sibren Loos, Indra Marth, and Willem van Verseveld. 2023. “HydroMT: Automated and Reproducible Model Building Andanalysis.” *Journal of Open Source Software* 8(83):4897. doi: 10.21105/joss.04897.

Emerton, Rebecca E., Elisabeth M. Stephens, Florian Pappenberger, Thomas C. Pagano, Albrecht H. Weerts, Andy W. Wood, Peter Salamon, James D. Brown, Niclas Hjerdt, Chantal Donnelly, Calum A. Baugh, and Hannah L. Cloke. 2016. “Continental and Global Scale Flood Forecasting Systems.” *WIREs Water* 3(3):391–418. doi: 10.1002/wat2.1137.

Eyring, Veronika, Sandrine Bony, Gerald A. Meehl, Catherine A. Senior, Bjorn Stevens, Ronald J. Stouffer, and Karl E. Taylor. 2016. “Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) Experimental Design and Organization.” *Geoscientific Model Development* 9(5):1937–58. doi: 10/gbpwpt.

Fowler, Keirnan J. A., Thomas A. McMahon, Seth Westra, Avril Horne, Joseph H. A. Guillaume, Danlu Guo, Rory Nathan, Holger R. Maier, and Andrew John. 2024. “Climate Stress Testing for Water Systems: Review and Guide for Applications.” *WIREs Water* 11(6):e1747. doi: 10.1002/wat2.1747.

Funk, Chris et al. 2015. “The Climate Hazards Infrared Precipitation with Stations - A New Environmental Record for Monitoring Extremes.” *Scientific Data* 2:150066.

Gebremicael, T. G., Y. A. Mohamed, and P. Van Der Zaag. 2019. “Attributing the Hydrological Impact of Different Land Use Types and Their Long-Term Dynamics through Combining Parsimonious Hydrological Modelling, Alteration Analysis and PLSR Analysis.” *Science of The Total Environment* 660:1155–67. doi: 10.1016/j.scitotenv.2019.01.085.

Ghile, Yonas, Paul Moody, and Casey Brown. 2014. “Paleo-Reconstructed Net Basin Supply Scenarios and Their Effect on Lake Levels in the Upper Great Lakes.” *Climatic Change* 127(2):305–19. doi: 10.1007/s10584-014-1251-8.

Giardino, A., R. Schrijvershof, C. M. Nederhoff, H. De Vroeg, C. Brière, P. K. Tonnon, S. Caires, D. J. Walstra, J. Sosa, W. Van Verseveld, J. Schellekens, and C. J. Sloff. 2018. “A Quantitative Assessment of Human Interventions and Climate Change on the West African Sediment Budget.” *Ocean & Coastal Management* 156:249–65. doi: 10.1016/j.ocecoaman.2017.11.008.

Hally, A., O. Caumont, L. Garrote, E. Richard, A. Weerts, F. Delogu, E. Fiori, N. Rebora, A. Parodi, A. Mihalović, M. Ivković, L. Dekić, W. Van Verseveld, O. Nuissier, V. Ducrocq, D. D’Agostino, A. Galizia, E. Danovaro, and A. Clematis. 2015. “Hydrometeorological Multi-Model Ensemble Simulations of the 4 November 2011 Flash Flood Event in Genoa, Italy, in the Framework of the DRIHM Project.” *Natural Hazards and Earth System Sciences* 15(3):537–55. doi: 10.5194/nhess-15-537-2015.

Hassaballah, Khalid, Yasir Mohamed, Stefan Uhlenbrook, and Khalid Biro. 2017. “Analysis of Streamflow Response to Land Use and Land Cover Changes Using Satellite Data and Hydrological Modelling: Case Study of Dinder and Rahad Tributaries of the Blue Nile (Ethiopia–Sudan).” *Hydrology and Earth System Sciences* 21(10):5217–42. doi: 10.5194/hess-21-5217-2017.

Hengl, Tomislav, Jorge Mendes De Jesus, Gerard B. M. Heuvelink, Maria Ruiperez Gonzalez, Milan Kilibarda, Aleksandar Blagotić, Wei Shangguan, Marvin N. Wright, Xiaoyuan Geng, Bernhard Bauer-Marschallinger, Mario Antonio Guevara, Rodrigo Vargas, Robert A. MacMillan, Niels H. Batjes, Johan G. B. Leenaars, Eloi Ribeiro, Ichsani Wheeler, Stephan Mantel, and Bas Kempen. 2017. “SoilGrids250m: Global Gridded Soil Information Based on Machine Learning” edited by B. Bond-Lamberty. *PLOS ONE* 12(2):e0169748. doi: 10.1371/journal.pone.0169748.

Hersbach, Hans et al. 2020. “ERA5 Hourly Data on Single Levels from 1979 to Present.” *Journal of Climate* 33(23):8309–35.

Karger, Dirk Nikolaus, Dirk R. Schmatz, Gabriel Dettling, and Niklaus E. Zimmermann. 2020. “High-Resolution Monthly Precipitation and Temperature Time Series from 2006 to 2100.” *Scientific Data* 7(1):248. doi: 10.1038/s41597-020-00587-y.

Kucharski, J., S. Steinschneider, J. Herman, J. Olszewski, W. Arnold, S. Rahat, R. Maendly, and P. Ray. 2024. “Bridging the Gap between Top‐down and Bottom‐up Climate Vulnerability Assessments: Process Informed Exploratory Scenarios Identify System‐based Water Resource Vulnerabilities.” *Water Resources Research* 60(11):e2023WR036649. doi: 10.1029/2023WR036649.

Leal Filho, Walter, Haruna Musa, Gina Cavan, Paul O’Hare, and Julia Seixas, eds. 2016. *Climate Change Adaptation, Resilience and Hazards*. Cham: Springer International Publishing.

Lehner, Flavio, Clara Deser, Nicola Maher, Jochem Marotzke, Erich M. Fischer, Lukas Brunner, Reto Knutti, and Ed Hawkins. 2020. “Partitioning Climate Projection Uncertainty with Multiple Large Ensembles and CMIP5/6.” *Earth System Dynamics* 11(2):491–508. doi: 10.5194/esd-11-491-2020.

Lempert, Robert J., and David G. Groves. 2010. “Identifying and Evaluating Robust Adaptive Policy Responses to Climate Change for Water Management Agencies in the American West.” *Technological Forecasting and Social Change* 77(6):960–74. doi: 10.1016/j.techfore.2010.04.007.

Lin, Peirong, Ming Pan, George H. Allen, Renato Prata De Frasson, Zhenzhong Zeng, Dai Yamazaki, and Eric F. Wood. 2020. “Global Estimates of Reach‐Level Bankfull River Width Leveraging Big Data Geospatial Analysis.” *Geophysical Research Letters* 47(7):e2019GL086405. doi: 10.1029/2019GL086405.

Maier, H. R., J. H. A. Guillaume, H. van Delden, G. A. Riddell, M. Haasnoot, and J. H. Kwakkel. 2016. “An Uncertain Future, Deep Uncertainty, Scenarios, Robustness and Adaptation: How Do They Fit Together?” *Environmental Modelling & Software* 81:154–64. doi: 10.1016/j.envsoft.2016.03.014.

Messager, Mathieu et al. 2016. “HydroLakes: A Database of Global Lake and Reservoir Characteristics at High Spatial Resolution.” *Nature Communications* 7:13603.

Moody, Paul, and Casey Brown. 2012. “Modeling Stakeholder-Defined Climate Risk on the Upper Great Lakes: MODELING CLIMATE RISK ON THE UPPER GREAT LAKES.” *Water Resources Research* 48(10). doi: 10.1029/2012WR012497.

Myneni, RB et al. n.d. “MODIS Leaf Area Index/FPAR Data Products.”

Pfeffer, WT et al. 2014. “The Randolph Glacier Inventory: A Globally Complete Inventory of Glaciers.” *Journal of Glaciology* 60(221):537–52.

Rahat, Saiful Haque, Scott Steinschneider, John Kucharski, Wyatt Arnold, Jennifer Olzewski, Wesley Walker, Romain Maendly, Asphota Wasti, and Patrick Ray. 2022. “Characterizing Hydrologic Vulnerability under Nonstationary Climate and Antecedent Conditions Using a Process-Informed Stochastic Weather Generator.” *Journal of Water Resources Planning and Management* 148(6):04022028. doi: 10.1061/(ASCE)WR.1943-5452.0001557.

Ray, Patrick A., and Casey M. Brown. 2015. *Confronting Climate Uncertainty in Water Resources Planning and Project Design*.

Rusli, Steven Reinaldo, Victor F. Bense, Syed M. T. Mustafa, and Albrecht H. Weerts. 2024. “The Impact of Future Changes in Climate Variables and Groundwater Abstraction on Basin-Scale Groundwater Availability.” *Hydrology and Earth System Sciences* 28(22):5107–31. doi: 10.5194/hess-28-5107-2024.

Schaller, Nathalie, Jana Sillmann, Malte Müller, Reindert Haarsma, Wilco Hazeleger, Trine Jahr Hegdahl, Timo Kelder, Gijs Van Den Oord, Albrecht Weerts, and Kirien Whan. 2020. “The Role of Spatial and Temporal Model Resolution in a Flood Event Storyline Approach in Western Norway.” *Weather and Climate Extremes* 29:100259. doi: 10.1016/j.wace.2020.100259.

Schellekens, Jaap. 2022. *Wflow Documentation*.

Seizarwati, W., and M. Syahidah. 2021. “Rainfall-Runoff Simulation for Water Availability Estimation in Small Island Using Distributed Hydrological Model Wflow.” *IOP Conference Series: Earth and Environmental Science* 930(1):012050. doi: 10.1088/1755-1315/930/1/012050.

Sperna Weiland, Frederiek C., Robrecht D. Visser, Peter Greve, Berny Bisselink, Lukas Brunner, and Albrecht H. Weerts. 2021. “Estimating Regionalized Hydrological Impacts of Climate Change Over Europe by Performance-Based Weighting of CORDEX Projections.” *Frontiers in Water* 3:713537. doi: 10.3389/frwa.2021.713537.

Stainforth, D. A., M. R. Allen, E. R. Tredger, and L. A. Smith. 2007. “Confidence, Uncertainty and Decision-Support Relevance in Climate Predictions.” *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences* 365(1857):2145–61. doi: 10.1098/rsta.2007.2074.

Steinschneider, Scott, and Casey Brown. 2013. “A Semiparametric Multivariate, Multisite Weather Generator with Low-Frequency Variability for Use in Climate Risk Assessments: Weather Generator for Climate Risk.” *Water Resources Research* 49(11):7205–20. doi: 10.1002/wrcr.20528.

Steinschneider, Scott, Patrick Ray, Saiful Haque Rahat, and John Kucharski. 2019. “A Weather‐Regime‐Based Stochastic Weather Generator for Climate Vulnerability Assessments of Water Systems in the Western United States.” *Water Resources Research* 55(8):6923–45. doi: 10.1029/2018WR024446.

Taner, Mehmet Ümit, Patrick Ray, and Casey Brown. 2017. “Robustness-Based Evaluation of Hydropower Infrastructure Design under Climate Change.” *Climate Risk Management* 18:34–50. doi: 10.1016/j.crm.2017.08.002.

Taner, Mehmet Ümit, Patrick Ray, and Casey Brown. 2019. “Incorporating Multidimensional Probabilistic Information Into Robustness‐Based Water Systems Planning.” *Water Resources Research* 2018WR022909. doi: 10.1029/2018WR022909.

Taner, MU, Sergio Contreras, Johannes Hunink, and Alfredo Hijar. 2019. *El Marco Del Árbol de Decisión: Aplicación a La Cuenca de Chancay-Lambayeque, Perú*. Deltares.

Van Verseveld, Willem J., Albrecht H. Weerts, Martijn Visser, Joost Buitink, Ruben O. Imhoff, Hélène Boisgontier, Laurène Bouaziz, Dirk Eilander, Mark Hegnauer, Corine Ten Velden, and Bobby Russell. 2024. “Wflow_sbm v0.7.3, a Spatially Distributed Hydrological Model: From Global Data to Local Applications.” *Geoscientific Model Development* 17(8):3199–3234. doi: 10.5194/gmd-17-3199-2024.

Wilby, Robert L., and Suraje Dessai. 2010. “Robust Adaptation to Climate Change.” *Weather* 65(7):180–85. doi: 10.1002/wea.543.

World Bank. 2022. *Operationalize an Online Tool for Rapid Bottom-up Climate Risk Assessment. Terms of References Document.* Reference number: 1279229.

Yamazaki, Dai et al. 2019. “MERIT Hydro: A High-Resolution Global Hydrography Map Based on Latest Topography Datasets.” *Water Resources Research*.

Yang, Y. C. Ethan, Sungwook Wi, Patrick A. Ray, Casey M. Brown, and Abedalrazq F. Khalil. 2016. “The Future Nexus of the Brahmaputra River Basin: Climate, Water, Energy and Food Trajectories.” *Global Environmental Change* 37:16–30. doi: 10.1016/j.gloenvcha.2016.01.002.

# 1. ANNEX

##### 1.0.0.0.1. BlueEarth CST Workflows - Command-Line Functionality

###### 1.0.0.0.1.1. Purpose and Functionality

BlueEarth CST workflows serves the purpose of scheduling and executing all analytical procedures required by the web application/online toolbox. Power users can use the BlueEarth CST workflows from a command-line interface to bypass the graphical interface and run the underlying scripts directly. Through the command-line, users can access to advanced settings regarding underlying data, model configuration and outputs, for example gridded results.

The BlueEarth CST command-line functionality consists of a series of computer scripts and workflows that are called from the command prompt; hence, a graphical user interface (GUI) is not present. The tool is intended to run from a personal desktop computer and to execute the command line, all necessary datasets and models along with required python and R packages need to be installed locally by the user.

Currently, the command-line functionality is intended as a working environment to develop and test the essential workflows that will carry out the main tasks in the final product, as well as linking and aligning the models and datasets required in these workflows. This means that the advanced functionality is being developed mainly for the developers and those familiar with bottom-up climate risk assessment (e.g., decision-scaling) as opposed to the target audience of the CST web application, i.e., non-technical practitioners/specialists.

The command-line tool provides automation for the following functionalities:

1.  Creation and setting up of a project for climate risk assessment for a given range of geographic boundaries, i.e., defined by the basin outlet coordinates and by specifying a time period for the analysis and hydrological performance indicators(s).
2.  Extraction and processing of historical climate data from the ERA5-land gridded global dataset for the project area. Visualization of statistical trends and uncertainty from the processed data for main climate variables, i.e., precipitation and temperature.
3.  Extraction and processing of climate projections from the CMIP5 ensemble for the project area; visualization of long-term changes for the same climate variables; and finally, summarizing relevant annual climate change statistics in a spreadsheet file.
4.  Rapid deployment of a distributed hydrological model (wflow) for the project area with default (user-adjustable) parameter settings
5.  Designing a climate stress test to assess the impacts of natural climate variability and plausible climate changes on the selected performance indicators. This is done by specifying how to sample natural variability and the uncertainty bandwidths for the changes in temperature and precipitation statistics (mean and variance).
6.  Execution of a climate stress test, which will initiate the simulation of the hydrological model repeatedly for specified bandwidth of changes. Relevant performance metrics are then calculated from the stress test runs and stored within a spreadsheet.
7.  Interactive analysis and visualization of climate risks through an external web application (not part of the current assignment, developed earlier by Deltares). The web application takes the summary climate stress test results (Excel file) and summary statistics from climate model projects (Excel file) to produce interactive climate response surfaces.

The following section describes the specific workflows and underlying tasks to provide the functionality described above.

###### 1.0.0.0.1.2. Workflows and Tasks

The CST command-line functionality is organized through a series of analytical procedures called "workflows." Each workflow consists of scripts coded in R or Python programming languages and handles various technical 'jobs' or tasks, such as data extraction, processing, generation, model execution, and visualization.

User inputs to each workflow are provided through JSON-based YAML files that are easy to read and modify. The scheduling of the workflows and included tasks are carried out through snakemake files (described later).

The workflows, underlying tasks, and the required user configuration are shown in [Table 0‑1](#_Ref103064599). The three workflows included in the process are *Project Creation* (W1), *Climate Projections and Analysis* (W2), and *Climate Stress Testing* (W3). From these two workflows, two main outputs are generated:

- *stress_test_summary.csv* to summarize calculated hydrological performance indicators from all climate stress test simulations.
- *clim_proj_summary.csv* to summarize annual and/or monthly statistics from each model and scenario combination in alignment with the experimental design of the stress test.

These two files are also passed to the CST-backend to visualize climate stress test results through interactive climate response surfaces

<table>
<caption><p><span id="_Ref103064599" class="anchor"></span>Table 0‑1 Command-line workflows and associated tasks and user configuration needs</p></caption>
<colgroup>
<col style="width: 9%" />
<col style="width: 29%" />
<col style="width: 36%" />
<col style="width: 24%" />
</colgroup>
<thead>
<tr>
<th>ID</th>
<th>Name</th>
<th>Main tasks</th>
<th>Configuration</th>
</tr>
</thead>
<tbody>
<tr>
<td>W1</td>
<td><p>Project</p>
<p>Creation</p></td>
<td>New project creation through basin outlet coordinates and historical/future analysis period; extraction of gridded climate data from ERA-5/CHIRPS databases; hydrological model application building; climate and hydrological model diagnostic plots generation</td>
<td><p>User input by YAML:</p>
<p><em>config_project.yml</em></p></td>
</tr>
<tr>
<td>W2</td>
<td>Climate Projections Analysis</td>
<td>Spatial extraction of climate projections; calculation of relevant statistics for relevant climate variables</td>
<td><p>User input by YAML:</p>
<p><em>config_projections.yml</em></p></td>
</tr>
<tr>
<td>W3</td>
<td>Climate Stress Testing</td>
<td>Weather generator deployment and synthetic weather generation by sampling natural variability and monthly/annual climatic changes; generation of hydrologic responses to the synthetic weather series; stress test post-processing by calculating hydrological performance indicators</td>
<td><p>User input by YAML:</p>
<p><em>config_stress_test.yml</em></p></td>
</tr>
</tbody>
</table>

Various tasks handled within the command-line toolbox are scheduled and automated through a Python-based workflow management system called Snakemake[^1]. Snakemake uses file-based tracking of data changes to identify which parts of a workflow need to be executed again. Snakemake allows forking and confluence of data flows, waiting for all processes to be completed before continuing the execution. A snakemake-file contains rules, and each rule states the script/component to execute and the associated input and output files. The snakemake-files automate the tasks within each workflow, while making the manual calling of the workflows on the command line easier. The snakemake files also address the full automation/integration of workflows, as well as parallelization of tasks.

<img src="../_images/cst-toolbox-technical-note-2025/image20.png" style="width:5.61389in;height:2.66042in" />

Figure 0‑1 Snakemake: example workflow management (left figure) and task parallelization (right figure)

For each project, the workflows are configured through a YAML configuration file (“snakemake_config.yml”). The purpose of this YAML file is to store main project information and settings in a human-readable data-serialization language format ([Figure 0‑2](#_Ref187916567)).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image21.png" style="width:5.08629in;height:4.43245in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref187916567" class="anchor"></span>Figure 0‑2. Display of project YAML file used to configure CST workflows.</p></figcaption>
</figure>

The rest of this section provides further descriptions of the three workflows developed during this activity. Future enhancements and new features will be described through the online README pages of each workflow under the Blueearth_cst repository.

####### Workflow 1: Project Creation

A rapid climate risk assessment project through the command-line toolbox begins with a projection creation workflow to delineate the study area, define the basic features of the analysis and rapidly deploy a hydrological model application, respectively.

The first step of the workflow is to generate the necessary folder and file structures based on user input, which will then be passed to the hydrological model builder. This is done in the YAML configuration file through the following project-specific parameters:

- *project_name:* a unique name given to the project
- *project_dir:* location of the main project directory
- *starttime:* start date of the historical (baseline) period (format: yyyy-mm-dd)
- *endtime:* end date of the historical (baseline) period (format: yyyy-mm-dd)
- *model_region:* geographic coordinates (lat, lon) of the basin outlet (degrees)
- *model_resolution:* grid size for the wflow model (square kilometers)
- *data_sources:* path to a YAML that stores metadata of climate and nonclimate datasets
- output_locations: geographic locations of interest for stress test calculations and climate vulnerability analysis (default: basin outlet) (optional)
- observations_timeseries: hydrological gauge data for verifying wflow outputs (optional)

Next, the script *project_creation.py* uses the information in the project.yml file to create a folder structure under the path "project_dir/project name." The main folders created in this process are described in [Table 0‑2](#_Ref102591158).

<table>
<caption><p><span id="_Ref102591158" class="anchor"></span>Table 0‑2. General folder structure created by the projection creation workflow</p></caption>
<colgroup>
<col style="width: 22%" />
<col style="width: 77%" />
</colgroup>
<thead>
<tr>
<th>Main folder</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>/config</td>
<td><p>User configuration and model ini files such as:</p>
<p><em>project_config.yaml</em></p>
<p><em>stresstest_config.yaml</em> and</p>
<p><em>wflow_build_forcing_historical.ini</em></p></td>
</tr>
<tr>
<td>/hydrology_model</td>
<td>Wflow model parameterization and setup files</td>
</tr>
<tr>
<td>/climate_historical</td>
<td>Historical climate forcing data (netcdf) in native and downscaled resolution</td>
</tr>
<tr>
<td>/analyze_projections</td>
<td>Extracted climate projections</td>
</tr>
<tr>
<td>/stresstest_exp01</td>
<td>All Intermediate files, raw results, and visualizations are generated from a particular stress test experiment. A separate subfolder is created for each climate experiment, e.g., exp01 for the first experiment</td>
</tr>
</tbody>
</table>

After this initial input and file structure preparation phase, a hydrological model of the basin is then built using the functions provided by the hydromt_wflow plugin. If observation time series are provided at the prespecified output locations, the associated model performance statistics can be computed. Individual steps under the hydrological model builder task are shown in [Figure 0‑3](#_Ref102731832).

Note the model builder here aims to deploy a hydrological model rapidly, without any manual model calibration. The resulting model performance can vary from one region to another based on various factors based on hydrological complexity, dataset quality. To improve model performance and accuracy, advanced users can replace and update the parameter files within the model directory (/hydrology_model).

<img src="../_images/cst-toolbox-technical-note-2025/image22.png" style="width:2.02306in;height:4.35859in" alt="A diagram of a program Description automatically generated" /> <img src="../_images/cst-toolbox-technical-note-2025/image23.png" style="width:3.167in;height:4.3529in" alt="A group of graphs and diagrams Description automatically generated" />

<span id="_Ref102731832" class="anchor"></span>Figure 0‑3. Steps for hydrological model building and initialization in the project creation workflow (left figure), and computed hydrological model performance statistics (optional) (right figure)

The workflow generates a series of additional plots, stored within the project folder, to visualize basin area and geographic extent ([Figure 0‑4](#_Ref188021640)), historical climatology ([Figure 0‑5](#_Ref188021690)) based on the selected gridded datasets, and hydrological output variables over the simulated period ([Figure 0‑6](#_Ref188021671)).

<img src="../_images/cst-toolbox-technical-note-2025/image24.png" style="width:3.72528in;height:3.29455in" alt="A map of a river Description automatically generated" />

<span id="_Ref188021640" class="anchor"></span>Figure 0‑4. Basin map of the project area

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image25.png" style="width:5.71975in;height:4.07914in" />
<figcaption><p><span id="_Ref188021690" class="anchor"></span>Figure 0‑5. Climatology plots for temperature, precipitation and PET: a) annual series, b) monthly patterns, c) mean values over the historical period.</p></figcaption>
</figure>

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image26.png" style="width:5.67161in;height:3.12088in" />
<figcaption><p><span id="_Ref188021671" class="anchor"></span>Figure 0‑6. Simulated hydrological outputs a) river flow, b) actual evaporation, groundwater recharge, and overland flow.</p></figcaption>
</figure>

####### Workflow 2: Climate Projections Analysis

The climate projections analysis workflow prepares the statistical summary information from a selected climate projection experiment by calculating relative changes in annual or monthly climate statistics between a future and historical analysis period.

For the Climate Projections Analysis workflow, the following configuration options are provided in the YAML file:

- *experiment name:* name of the multimodel experiment (currently CMIP6)
- *climate_models:* one or multiple models among the following: IPSL-CM5A-LR, 'ACCESS1-0', 'ACCESS1-3', 'BNU-ESM', 'CCSM4', 'CESM1-BGC', 'CESM1-CAM5', 'CESM1-FASTCHEM', 'CESM1-WACCM', 'CMCC-CESM', 'CMCC-CM', 'CMCC-CMS', 'CNRM-CM5', 'CNRM-CM5-2', 'CSIRO-Mk3-6-0', 'CanCM4', 'CanESM2', 'EC-EARTH', 'FGOALS-g2', 'FIO-ESM', 'GFDL-CM2p1', 'GFDL-CM3', 'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-H','GISS-E2-H-CC', 'GISS-E2-R', 'GISS-E2-R-CC', 'HadCM3', 'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'IPSL-CM5B-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'MIROC4h', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MPI-ESM-P', 'MRI-CGCM3', 'MRI-ESM1', 'NorESM1-M', 'NorESM1-ME', 'bcc-csm1-1', 'bcc-csm1-1-m', 'inmcm4'.
- *climate_scenarios:* selection of representative concentration pathways (RCPs) among RCP2.6, RCP4.5, RCP6.0, and RCP8.5.
- *ensemble:* currently set to r1i1p1.
- *Variables*: precipitation and temperature (mean, minimum, and maximum)

In addition to these parameters, the user also specifies the time windows for the analysis:

- Historical_period: start and ending year of the historical analysis period (default value same with the historical climate dataset)
- Future_period: start and end year of the future analysis period, e.g., 2020-2060.

Based on the settings provided, the workflow carries out the following sequential steps ([Figure 0‑7](#_Ref102664086)):

1.  Monthly time series are extracted from the climate projections dataset.
2.  Relevant statistics are calculated for all variables from the climate models and scenarios (historical and future).
3.  Change factors are calculated as the relative change in the statistics in the future period regarding the historical period. This is repeated for all selected climate variables and climate models/RCP scenario combinations.
4.  Calculated change factors across different model runs and variables are combined into a single spreadsheet file, e.g., *analyze_projections_summary.csv*.

After the four-step process, a copy of the data_catalogue.yml is made. All model-dataset information is then removed, while the climate-dataset information is extended with climate statistics for the basin. Finally, the file is saved as climate_catalogue.yml. Statistical time series will be stored in a separate netcdf file next to the climate_catalogue.yml. The gridded climate projection data can be removed if desired after the statistics are extracted.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image27.png" style="width:5.41923in;height:4.20994in" alt="A diagram of a change Description automatically generated" />
<figcaption><p><span id="_Ref102664086" class="anchor"></span>Figure 0‑7. Main steps in the climate projections analysis workflow</p></figcaption>
</figure>

The climate projections workflow generates a number of visuals from the projected temperature and precipitation time-series, including annual trends ([Figure 0‑8](#_Ref188268857)), annual patterns/cycles ([Figure 0‑9](#_Ref188268932)), and a scatter plot of mean annual changes ([Figure 0‑10](#_Ref188268944)).

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image28.png" style="width:5.50905in;height:2.08661in" />
<figcaption><p><span id="_Ref188268857" class="anchor"></span>Figure 0‑8. Annual trends in projected temperature and precipitation from CMIP6 models.</p></figcaption>
</figure>

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image29.png" style="width:5.47445in;height:2.08661in" />
<figcaption><p><span id="_Ref188268932" class="anchor"></span>Figure 0‑9. Annual cycle/monthly pattern in projected temperature and precipitation from CMIP6 models.</p></figcaption>
</figure>

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image30.png" style="width:5.38462in;height:4.49573in" />
<figcaption><p><span id="_Ref188268944" class="anchor"></span>Figure 0‑10. Scatter plot of projected changes in mean annual temperature and precipitation from CMIP6 models.</p></figcaption>
</figure>

####### Workflow 3: Stress Test

The stress test workflow handles the design and preparation of data for climate stress testing based on user input, executing the hydrology model runs to generate stress test results, and finally post-processing of the results for the calculation of the hydrological performance indicators.

A climate stress test is configured through the following options in the project YAML file ([Figure 0‑11](#_Ref102731904))*:*

- *Experiment_name*: if not provided, a unique tag is assigned, e.g., "exp_01."
- Starttime_experiment: star year of the analysis period (default: same as the start year of the climate projection future time-window)
- Endtime_experiment: end year of the analysis period (default: same as the end year of the climate projection future time-window)
- *Realizations_number*: number of stochastic weather realizations to be generated with the weather generator for sampling natural (historical) variability. (default value: 3)

Next, an additional set of parameters define the plausible set of climate changes to be explored through the stress test. These are defined separately for temperature (daily mean, minimum, and maximum) and precipitation (daily mean).

Temperature change parameters:

- Change_step_number: number of temperature change scenarios to be generated to span the range of the uncertainty bandwidth
- Change_step_type: transient (default) or constant. The transient change applies change factors in a gradually increasing manner, starting from no change in the first analysis year. In contrast, the constant change applies the same change factors over the entire analysis period.
- Mean changes bandwidth: minimum/maximum of changes to be evaluated for the daily mean, minimum, and maximum temperature (units: degree Celsius). Change factors are reported regarding the baseline/historical value either on an annual basis (one set of minimum and maximum values) or on a monthly basis (12 sets of minimum and maximum values).

  Precipitation change parameters:

- Change_step_number: number of precipitation change scenarios to be generated to span the range of the uncertainty bandwidth
- Change_step_type: transient (default) or constant (same as in temperature).
- Mean changes bandwidth: minimum/maximum of changes to be evaluated for the daily mean precipitation (units: %). Change factors are reported regarding the baseline/historical value either on an annual or monthly basis
- Variability changes bandwidth: minimum/maximum range of changes to be evaluated for the daily variance of precipitation (units: %). Change factors are reported regarding the baseline/historical value, either on an annual or monthly basis.

Besides these essential parameters, advanced users can fine-tune the stress test through advanced weather generator settings (“weathergen_config.yml”), for example, low-frequency detection criteria or Markov-chain state thresholds.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image31.png" style="width:5.74792in;height:6.99236in" alt="A screenshot of a computer Description automatically generated" />
<figcaption><p><span id="_Ref102731904" class="anchor"></span>Figure 0‑11. Example stress test configuration settings in the project YAML file</p></figcaption>
</figure>

After the YAML file is configured, the next step is to generate the necessary folder structure and files required for the analysis. This is done with the script *experiment_creation.py* that extends the project folder with the following folder structure (assuming this is the first experiment) while it saves the settings (metadata) of the experiment in the exp01\_\_settings.yml file. The number of realization sub-folders is determined by the experiment settings. In case this is the second experiment for the same project, all files and folders will use '02' to distinguish the datasets from experiment 01 ([Figure 0‑12](#_Ref102731404)).

<table style="width:100%;">
<caption><p><span id="_Ref102731404" class="anchor"></span>Figure 0‑12. General folder structure created by stress test workflow</p></caption>
<colgroup>
<col style="width: 99%" />
</colgroup>
<thead>
<tr>
<th><p>├─── user_name</p>
<p>├─── project_title</p>
<p>├───exp01</p>
<p>exp01__settings.yml</p>
<p>exp01__metrics.yml</p>
<p>├─── exp01_realiz01</p>
<p>wgen_exp01_realiz01__origmaps.nc</p>
<p>wgen_exp01_realiz01__inmaps.nc</p>
<p>wflow_exp01_realiz01__outmaps.nc (optional)</p>
<p>wflow_exp01_realiz01__outscalars.nc</p>
<p>wflow_exp01_realiz01__metrics.csv</p>
<p>├─── exp01_realiz02</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

When conducting a stress test, each climate realization is translated into a hydrological response realization. This is a one-by-one process that can be executed simultaneously for all realizations. [Figure 0‑13](#_Ref90994371) shows the workflow, including the processing steps called in parallel and the associated input and output files. The data for each realization with the experiment is contained in one realization folder. All realization statistics are combined in one expXX_metrics.yml file.

Note that depending on the user interaction and the GUI design, this workflow could be split into two phases. Phase 1 would contain the tasks of running the weather generator and catalog the realizations. Phase 2 would be the parallel execution of the hydrological responses.

<img src="../_images/cst-toolbox-technical-note-2025/image32.png" style="width:5.07008in;height:3.84538in" alt="A screenshot of a computer Description automatically generated" /><span id="_Ref90994371" class="anchor"></span>

Figure 0‑13. Stress test workflow with simultaneous execution of different climate realizations

###### 1.0.0.0.1.3. Installation

The command-line tool can be downloaded and installed from the [Blueearth_cst repositories](https://github.com/Deltares/blueearth_cst). This section provides a brief overview of the installation. For full and most recent instructions on the installation and usage, users shall refer to the GitHub repository.

4.  *Install prerequisites:*

> Install Python and R dependencies using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
>
> Install Julia from [julialang.org](https://julialang.org/downloads/) and Wflow following its [installation guide.](https://deltares.github.io/Wflow.jl/dev/user_guide/install/#Installing-as-Julia-package)

5.  *Download the Toolbox:*

    Clone the BlueEarth_cst git repo from [GitHub](https://github.com/Deltares/blueearth_cst), then navigate into the code folder (where the environment.yml file is located):

    <img src="../_images/cst-toolbox-technical-note-2025/image33.png" style="width:5.04178in;height:0.53543in" alt="A black and white text Description automatically generated" />

6.  *Set Up Conda Environment:*

> Navigate to the folder containing the lock file and run the following commands to create
>
> and activate the environment:

<img src="../_images/cst-toolbox-technical-note-2025/image34.png" style="width:5.55542in;height:0.66046in" />

Docker:

The workflows are also available as a docker image, with all dependencies preinstalled.

##### 1.0.0.0.2. Descriptions of the Recommended CST Tools

###### 1.0.0.0.2.1. HydroMT

The HydroMT framework is part of the Deltares BlueEarth Platform. HydroMT is used to automize the model building process from (open) datasets. HydroMT is a modular Python package that allows the use of global datasets (A.2.1), read them through a data adapter (A.2.2), generate models (A.2.3) via a set of common methods (A.2.4). The tool also includes a high-level Command Line Interface (A.2.5) that allows the tool to be easily incorporated into other (online) tools.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image35.png" style="width:5.74792in;height:3.17639in" />
<figcaption><p>Figure 0‑14. Schematic of the hydromt architecture.</p></figcaption>
</figure>

####### Global datasets

The models are built from global datasets. For wflow, these datasets include static datasets (maps) to setup the stigmatization and dynamic climate datasets (time series) to actually run the model. A complete list of currently available datasets is provided in the online documentation[^2].

####### Data adapter

To read the data into standardized Python formats, a data adapter was developed. This data adapter reads datasets and will standardize units and naming of variables in such a way that all datasets can be processed in the same way inside the toolbox. The advantage is that the data can be read in native format without the need to do any preprocessing on beforehand. This makes the tool more robust in the way that when datasets are updated, these can easily be read in the same way without the need of finding out how the earlier version of the dataset was processed.

The data adapter allows reading of static and dynamic raster datasets, static and dynamic point datasets and polygons datasets. Internally, hydromt uses the current scientific Python standard data formats, such as Pandas data frames for tabular data, geo data frame vector data and array for raster datasets.

To add new datasets to the set already available, the user can specify a new dataset via the data catalog. In this catalog, the user should specify the locations and properties of the datasets. How this should be done is documented in the online documentation.

####### Methods

The structure of hydromt is setup such that high-level methods to read, write and process data can be shared over several model builders. This makes the tool robust and efficient. These methods also include GIS processing methods, to for example resample and upscale data from high- to lower resolution. Because hydromt uses state-of-the-art Python packages at the heart of the toolbox, this can be done efficiently and fast, even over large model domains.

These methods also include upscaling techniques to upscale river geometry from the highest available resolution at global scale (90x90 m<sup>2</sup>) to typical model resolution (1x1 km<sup>2</sup>). To preserve all properties of the rivers that are known on the high resolution, and to make sure the river is always projected at the same location (which is not the case with classical upscaling of the data). These techniques are described in detail in Eilander et al., 2020 (in review).

###### 1.0.0.0.2.2. Stochastic Weather Generator

Stochastic weather generators (SWGs) are mathematical algorithms that produce time series of synthetic weather data at desired spatial and temporal resolution. The parameters of the model are conditioned on existing meteorological records to ensure that the characteristics of historical weather emerge in the daily stochastic process.

In bottom-up climate risk assessment, SWGs are used frequently to perform exhaustive assessments of a system's vulnerability to climate conditions across multiple scales, including natural variability and potential climate changes. SWGs can be used to produce new realizations of a time series of weather variables that exhibit similar statistics as the historical record, thus producing an ensemble of time series that provides a sample of the historical or "natural" variability. By incrementally manipulating one or more parameters in a weather generator, one can simulate many climate scenarios that exhaustively explore potential futures that exhibit slight differences in nuanced climate characteristics, such as the intensity and frequency of daily precipitation, the serial correlation of extreme heat days, or the recurrence of long-term droughts.

The stochastic weather generator that will be developed for the climate toolbox be based on a semi-parametric approach initially developed by Steinschneider & Brown (2013). The model combines a Wavelet Autoregressive Model (WARM) to accurately simulate low-frequency variability in annual precipitation, combined with a k-nearest neighbor resampling scheme to simulate spatially distributed multivariate weather series. SWG will produce new time series of temperature and precipitation at the grid resolution required for the wflow model. These output time series help explore the effects of variations similar to observed historical conditions, as well as climate variability beyond the historical record due to changes in future temperature and precipitation. Outputs from this module are used to evaluate the effects of warming temperatures and changing precipitation on the stream flow to enable the evaluation of hydrological drought/flood performance metrics over a range of possible weather sequences.

###### 1.0.0.0.2.3. Wflow

Wflow is the Deltares open-source modelling framework for distributed rainfall-runoff modelling. It enables to maximize the use of recent-day Earth observation data and calculates all hydrological fluxes at any given location and time step, using high-resolution distributed meteorological forcing data. Examples of major processes that can be included are snow and glacier accumulation and melt, rainfall interception, soil moisture accounting, runoff generating processes and river flow. Wflow is programmed in the PCRaster-Python environment. A main benefit of wflow as compared to many other distributed hydrological models is that consists of a library of different hydrological concepts and the possibility to link it with other model components (like water quality, sediment, crop growth, reservoirs) directly or by using the basic model interface (BMI), the integration with Delft-FEWS for flood and or drought forecasting, and the coupling with OpenDA for data assimilation of point and spatial observations.

<img src="../_images/cst-toolbox-technical-note-2025/image36.png" style="width:2.675in;height:2.03125in" /><img src="../_images/cst-toolbox-technical-note-2025/image37.png" style="width:2.71667in;height:2.02083in" />

Figure B.1 An important characteristic of Wflow is the spatial distribution of hydrological variables.

####### An integrated platform linking different concepts and processes

The wflow framework consists of several modelling concepts to represent hydrological response that share the same structure but are different regarding the hydrological conceptualization. Currently, wflow includes the following modelling concepts: the HBV96 concept (wflow_hbv), the FLEX-Topo concept (wflow_topoflex), PCR-GLOBWB (wflow_pcrglobwb), the Topog_sbm concept (wflow_sbm), the AWRA—L model (0.5 degree wflow_W3RA, 0.05 degree wflow_W3), the GR4J model (wflow_gr4), the SPHY model (wflow_sphy), and the STREAM model (wflow_stream).

Next to modelling the hydrological response, wflow contains modules for the simulation of many other parts of the hydrological process such as snow melt, glaciers, kinematic wave for routing, reservoir operation and storage in lakes. Furthermore, a sediment model (wflow_sediment) is available (linked to wflow_sbm) that enables to model soil erosion and delivery to the river system. A crop growth model based on LINTUL (wflow_lintul) can be used to simulate irrigated rice dynamics (coupled via BMI to wflow_sbm).

####### Application area

Wflow has been successfully applied across the world for flood and drought forecasting,, as well as evaluating the impacts of climate change (including drought conditions) or (land) management strategies. It is also used as an instrument to provide uniform flow simulation for river basin management studies and for the assessment of water availability for the design of irrigation schemes.

The global wflow model has been identified as a major component in the provision of hydrological data services via the GLOFFIS for forecast data and Project GLOW for historical data.

The ease with which wflow can be coupled to other models has meant a rapid expansion in its application to generate headflows for other models, including for Delft-FEWS, RIBASIM, SOBEK, Delft 3D FM, among others.

####### Approach to distributed hydrological modelling

Wflow models maximize the use of available spatial data by linking parameter values to soil or land use types and by using gridded meteorological products. The distributed nature of the model implies that the model is run on each grid cell and that water flows from one grid cell to another either through the kinematic wave routine and/or through lateral groundwater flow. For each grid cell, wflow provides continuous simulated values of the hydrological state variables (channel flow, overland flow saturated zone storage etc.). The model output can be visualized as a sequence of spatial maps (gridded raster data) of hydrological variables. For any location in the catchment, time series of hydrological variables such as discharge or soil saturation can be generated.

The input data required to execute a simulation in wflow can be separated into *i)* static data concerning the description of the land surface, and *ii)* dynamic data, represented by the hydro-meteorological forcing of the model. Main input maps are generally available spatial datasets:

- Digital elevation map (DEM)
- River network map (can be generated from the DEM)
- Land use map
- Soil type map

Based on the DEM, and the river network map, the wflow model requires a local drain direction (ldd, based on the D8 algorithm) map, which forms the basis for the hydrological routing calculations.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image38.png" style="width:5.34536in;height:3.54759in" />
<figcaption><p>Figure 0‑15. Overview of wflow gridded input and output for the Swat river in Pakistan</p></figcaption>
</figure>

####### Modelling the processes of the hydrological cycle with wflow_sbm

Deltares' ongoing research on wflow focuses on the further development of the wflow_sbm concept (loosely based on Topog_sbm concept (Vertessy and Elsenbeer (1999)). Wflow_sbm is a physically based model, that makes use of a simplified representation of the Richards equation (gravity-based infiltration and vertical flow through the soil column as well as capillary rise), and kinematic wave for lateral flow through the subsurface (making it comparable to the closed source G2G (Bell et al., 2007) and TopKapi (Todini and Ciarapica, 2002) models), on top of the land surface (overland flow) and the river network in downstream direction. This way the run time performance is faster than more complex physically based hydrological models that solve the Richards equation and the shallow water equations for overland flow and through the river network. Wflow_sbm model parameters have a clear physical meaning (e.g. saturated and residual water content and saturated hydraulic conductivity) and can be derived from available large-scale digital elevation models, soil texture data and land cover information. This combination allows for comprehensive model estimation using pedotransfer functions and seamless calibrations across large domains or for the possibility of detailed land use change scenario analyses.

<figure>
<img src="../_images/cst-toolbox-technical-note-2025/image39.png" style="width:4.59368in;height:3.42424in" />
<figcaption><p>Figure 0‑16. Overview of the different processes and fluxes in the wflow_sbm model.</p></figcaption>
</figure>

Wflow_sbm incorporates the most important processes of the hydrological cycle and has the option to divide the soil column into different layers, to allow for transfer of water within the unsaturated zone. Different components of the hydrological cycle are modeled through a combination of hydrological processes nested in the model code, including:

1.  Evapotranspiration (open water and soil evaporation and transpiration) and interception losses, interception is schematized by the Gash model (Gash, 1979,) on a daily timestep and the Modified Rutter interception model on a sub-daily timestep.

<!-- -->

8.  A root water uptake reduction function (Feddes, 1978).
9.  Channel, overland, and lateral subsurface flow are modelled with the kinematic wave model.
10. Infiltration (paved and unpaved areas per grid cell (fraction))
11. Vertical flow through the unsaturated zone, and transfer to the saturated store.
12. Saturated zone (Capillary rise to the unsaturated zone, kinematic lateral subsurface flow between grid cells).

####### Reservoirs and lakes

Within the kinematic wave for channel flow, natural lakes and reservoirs can be included. Both modules use a simple mass balance approach and need as inputs inflow, precipitation and potential evapotranspiration.

The reservoir module requires the model parameters surface area, maximum release below the spillway, the minimum environmental flow requirement downstream of the reservoir, a target fraction full (of the maximum reservoir storage) and a target minimum full fraction (of the maximum reservoir storage), and the maximum reservoir storage capacity. A sigmoid curve, based on the current storage fraction of the reservoir, the minimum environmental flow requirement downstream, and the target minimum full fraction, controls the amount of the flow that the reservoir releases to fulfill the environmental flow requirement downstream. Furthermore, the reservoir releases water based on the maximum storage capacity (reservoir storage that exceeds the maximum storage capacity is always released), the target fraction full and maximum release below the spillway.

Both storage and outflow of the natural lake module are linked to the water level in the lake using a storage curve and rating curve, respectively. More detailed information about these modules can be found here: <https://wflow.readthedocs.io/en/latest/wflow_funcs.html>

####### Snow and glaciers

Several of the integrated hydrological modelling concepts in wflow provide full snow modelling functionality, notably the SBM and HBV concepts. The full snow melt equations can be found in the [technical documentation](https://wflow.readthedocs.io/en/latest/wflow_funcs.html#snow-modelling), and are adopted from the HBV-96 hydrologic model concept Because of the distributed nature of wflow, snow accumulation and snow melt are calculated per grid cell, providing a spatial overview of snow depth as output of the simulation. If precipitation occurs as snowfall, it is added to the dry snow component within the snow pack. Otherwise, it ends up in the free water reservoir, which represents the liquid water content of the snow pack. Between the two components of the snow pack, interactions take place, either through snow melt or through snow refreezing. Wflow contains an additional Glacier module, that can calculate the growth and decline of glaciers.

<img src="../_images/cst-toolbox-technical-note-2025/image40.png" style="width:5.5213in;height:2.80041in" />

Figure 0‑17, Simulated snow pack by the wflow snow module

##### 1.0.0.0.3. Open-source Software

###### 1.0.0.0.3.1. Definition

Open-source software (OSS) is software that is distributed with its source code, making it available for use, modification, and distribution with its original rights. Source code is the part of software that most computer users don’t ever see; it’s the code computer programmers manipulate to control how a program or application behaves. Programmers who have access to source code can change a program by adding to it, changing it, or fixing parts of it that aren’t working properly.

Open-source software allows users to access, modify, and distribute its source code freely, while closed-source software keeps the code proprietary and restricts modifications. Main differences between Open and Close source software are summarized in Table C-1.

| Factors | Open-Source software | Closed Source Software |
|----|----|----|
| Price | Available for nominal or zero licensing and usage charges. | Cost varies based upon the scale of the software. |
| Freedom to customize | Completely customizable, but it depends on the open-source license. Requires in-house expertise. | Change requests must be made to the company selling the software. This includes bug fixes, features, and enhancements. |
| User-friendliness | Typically, less user-friendly, but it can depend on the goals of the project and those maintaining it. | Typically, more user-friendly. As a for-profit product, adaptability and user experience are often key considerations. |
| After-sales support | Some very popular pieces of open-source software (e.g., OSS distributed by Red Hat or SUSE) have plenty of support. Otherwise, users can find help through user forums and mailing lists. | Dedicated support teams are in place. The level of service available depends on the service-level agreement (SLA). |
| Security | Source code is open for review by anyone and everyone. There is a widespread theory that more eyes on the code makes it harder for bugs to survive. However, security bugs and flaws may still exist and pose significant risk. | The company distributing the software (i.e., software owner) guarantees a certain level of support, depending on the terms of the SLA. Because the source code is closed for review, there can be security issues. If issues are found, the software distributor is responsible for fixing them. |
| Vendor lock-in | No vendor lock-in due to the associated cost. Integration into systems may create technical dependency. | In most cases, large investments are made in proprietary software. Switching to a different vendor or to an open-source solution can be costly. |
| Stability | This will depend on the current user base, the parties maintaining the software, and the number of years in the market. | Older, market-based solutions are more stable. New products have similar challenges as open-source products. If a distributor discontinues an application, the customer may be out of luck. |
| Popularity | Some open-source solutions are very popular and are even market leaders (e.g., Linux, Apache). | In some industries, proprietary software is more popular, especially if it has been in the market for many years. |
| Total cost of ownership (TCO) | TCO is lower and upfront due to minimal or no usage cost, and depends on the level of maintenance required. | TCO is much higher and depends on the size of the user base. |
| Community participation | The community participating in development, review, critique, and enhancement of the software is the essence of open source. | Closed community. |
| Interoperability with other open-source software | This will depend on the level of maintenance and goals of the group, but it is typically better than closed source software. | This will depend on the development standards. |
| Tax calculation | Difficult due to undefined monetary value. | Definite. |
| Enhancements or new features | It can be developed by the user if needed. | Request must be made to the software owner. |
| Suitability for production environment | OSS might not be technically well-designed or tested in a large-scale production environment. | Most proprietary software goes through multiple rounds of testing. However, things can still go wrong when deployed in a production environment. |
| ­Financial institution considerations | The financial industry tends to avoid open-source solutions. If used, a vetting process must take place. | Financial institutions prefer proprietary software. |
| Warranty | No warranty available. | Best for companies with security policies requiring a warranty and liability indemnity. |

Table C-1. Open vs. Close Source Software (Source: [Black Duck](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.blackduck.com%2Fglossary%2Fwhat-is-open-source-software.html&data=05%7C02%7Cumit.Taner%40deltares.nl%7C3c65d08396634ecc541308dd446f4a92%7C15f3fe0ed7124981bc7cfe949af215bb%7C0%7C0%7C638741966977875206%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=t1cJ7CP1l1TdHDorlKSjS1mQuktwG1xJMnnXmDBLelk%3D&reserved=0))

###### 1.0.0.0.3.2. License

OSS usually comes with a distribution license. This license includes the terms that define how developers can use, study, modify, and most importantly, distribute the software. OSS licenses are generally fall into two categories: permissive and copyleft. Permissive licenses, such as MIT, are highly flexible and allow others to use, modify, and even integrate the software into proprietary projects without requiring you to share your changes. On the other hand, copyleft licenses, like the General Public License (GPL) are stricter and require you to share any modifications or software derived from the original under the same license.

According to the [Black Duck Knowledgebase](https://www.blackduck.com/glossary/what-is-open-source-software.html), five of the most popular licenses are:

- MIT License
- GNU General Public License (GPL) 2.0—this is more restrictive and requires that copies of modified code are made available for public use
- Apache License 2.0

<!-- -->

- GNU General Public License (GPL) 3.0
- BSD License 2.0 (3-clause, New or Revised)—this is less restrictive

An overview of relevant OSS licenses for the CST, i.e., developed components and the underlying external (preexisting) software are provided in Table C-2.

<table>
<caption><p>Table C‑2. Comparison of OSS Licenses relevant for the CST (Source: <a href="https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.blackduck.com%2Fglossary%2Fwhat-is-open-source-software.html&amp;data=05%7C02%7Cumit.Taner%40deltares.nl%7C3c65d08396634ecc541308dd446f4a92%7C15f3fe0ed7124981bc7cfe949af215bb%7C0%7C0%7C638741966977875206%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&amp;sdata=t1cJ7CP1l1TdHDorlKSjS1mQuktwG1xJMnnXmDBLelk%3D&amp;reserved=0">Black Duck</a>)</p></caption>
<colgroup>
<col style="width: 11%" />
<col style="width: 26%" />
<col style="width: 11%" />
<col style="width: 10%" />
<col style="width: 17%" />
<col style="width: 21%" />
</colgroup>
<thead>
<tr>
<th>License</th>
<th>Description</th>
<th>Commercial Use?</th>
<th>Attribution?</th>
<th>Distribution of Modified Versions</th>
<th><p>Other</p>
<p>Notes</p></th>
</tr>
</thead>
<tbody>
<tr>
<td>CC-BY-NC 4.0</td>
<td>The Creative Commons license allows sharing and adaptation but prohibits commercial use.</td>
<td>No</td>
<td>Yes</td>
<td>Yes (Non-commercial only)</td>
<td>Primarily used for non-software works</td>
</tr>
<tr>
<td>ODbL 1.0</td>
<td>Open Database License; applies to data and databases, allowing sharing and modification under certain conditions.</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes, but must share derivatives under ODbL</td>
<td>Requires derivatives of databases to be shared alike and include attribution.</td>
</tr>
<tr>
<td>MIT</td>
<td>Permissive open-source software license allowing nearly unrestricted use, distribution, and modification.</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>Widely used for software projects; requires attribution but imposes no sharing restrictions.</td>
</tr>
<tr>
<td>GPL v3.0</td>
<td>General Public License; ensures that any derivative work remains open-source.</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes, but must also use GPL</td>
<td>Strong copyleft license; requires making source code available for derivatives.</td>
</tr>
<tr>
<td>BSD 2-Clause</td>
<td>Permissive open-source software license with minimal restrictions.</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>Simpler than GPL; does not require derivatives to remain open-source</td>
</tr>
<tr>
<td>SSPL</td>
<td>Server-Side Public License; requires open-sourcing the entire application if it uses the licensed software.</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes, but requires making broader application code available</td>
<td>Intended for server-side software to prevent misuse of open-source licenses in cloud environments</td>
</tr>
</tbody>
</table>

<img src="../_images/cst-toolbox-technical-note-2025/image41.png" style="width:8.26667in;height:11.69444in" />

[^1]:

[^2]: <https://deltares.github.io/hydromt/latest/user_guide/data.html#available-global-datasets>
