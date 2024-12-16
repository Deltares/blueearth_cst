Examples of the different workflows
-----------------------------------

To illustrate the different workflows, their inputs, outputs and how to run them, we 
provide a set of examples in the form of Jupyter notebooks. These notebooks are located 
in the `docs/notebooks` directory of the repository. 

Each notebook represent a specific snakemake workflow and are organized in the following 
categories:

1. **Historical climate.ipynb**: This notebook shows how to run the historical climate workflow in **Snakefile_climate_historical**. It extracts and analyse different climate variables from the historical climate dataset, compares to observations and search for existing trends in climate.
2. **Climate projections.ipynb**: This notebook shows how to run the climate projections workflow in **Snakefile_climate_projections**. It extracts and analyse different climate variables from the climate projections dataset (CMIP6), and derives basin averaged and gridded change in climate based on several scenarios, time horizons and climate models.
3. **Historical hydrology.ipynb**: This notebook shows how to run the hydrological workflow in **Snakefile_historical_hydrology**. It builds a hydrological Wflow model, runs it ith different climate datasets, compares to observations and search for existing trends in hydrology.
4. **Hydrological projections.ipynb**: This notebook shows how to run the hydrological projections workflow in **Snakefile_future_hydrology_delta_change**. It runs the previoulsy built hydrological model with a reference historical climate modified with monthly delta changes from the climate projections dataset.
5. **Climate Stress Test.ipynb**: This notebook shows how to run the climate stress test workflow in **Snakefile_climate_experiment**. A weather generator is used to generate synthetic and perturbed weather realizations based on the historical climate dataset. The hydrological model is then run (stress test) with these perturbed weather realizations to assess the impact of future climate on the hydrology.

To run the notebooks, you first need to install the toolbox. See the detailed explanations
in the `../../README.rst` file. Once the toolbox is installed, you can run the notebooks
from Visual Studio Code or jupyter lab or notebook (make sure the environment where the 
toolbox is installed is properly activated).

The notebooks are self-explanatory and contain detailed explanations on how to run the
different workflows. They also contain the expected outputs and how to visualize them.
The notebooks are also a good starting point to understand the different workflows and
how to adapt them to your specific needs.

*NOTE: The notebooks are provided as examples and the model used or the results presented
in the notebooks are for illustration only.* 

**Data**

The data used in the notebooks are extracts of global datasets for the Piave basin in
Italy for demonstration purpose. Most dataset are provided through HydroMT 
`artifact_data <https://deltares.github.io/hydromt/v0.10.0/user_guide/data_existing_cat.html>`_ 
and the rest in the `tests/data` directory of the repository.

The list of datasets used in the notebooks are:

.. list-table:: 
   :widths: 25 15 60
   :header-rows: 1

   * - Name
     - Type
     - Reference
   * - MERIT hydro IHU 1km
     - Hydrography
     - Eilander, D., Winsemius, H. C., Van Verseveld, W., Yamazaki, D., Weerts, A., & Ward, P. J. (2020). MERIT Hydro IHU. https://doi.org/10.5281/zenodo.7936280
   * - Global estimates of reach-level bankfull river width
     - Hydrography
     - Lin, P., M. Pan, H. E. Beck, Y. Yang, D. Yamazaki, R. Frasson, C. H. David, et al. 2019. Global Reconstruction of Naturalized River Flows at 2.94 Million Reaches. Water Resources Research (American Geophysical Union (AGU)) 55: 6499–6516. doi:10.1029/2019wr025287
   * - Copernicus Global Land Service: Land Cover 2015
     - Land Cover
     - Buchhorn, M., B. Smets, L. Bertels, B. De Roo, M. Lesiv, N.-E. Tsendbazar, M. Herold, and S. Fritz. 2020. Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2015: Globe. Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2015: Globe. Zenodo. doi:10.5281/ZENODO.3939038.
   * - MODIS/Terra+Aqua Leaf Area Index
     - Leaf Area Index
     - Myneni, R., Y. Knyazikhin, and T. Park. 2015. MCD15A3H MODIS/Terra+Aqua Leaf Area Index/FPAR 4-day L4 Global 500m SIN Grid V006. MCD15A3H MODIS/Terra+Aqua Leaf Area Index/FPAR 4-day L4 Global 500m SIN Grid V006. NASA EOSDIS Land Processes Distributed Active Archive Center. doi:10.5067/MODIS/MCD15A3H.006
   * - SoilGrids Database v2017
     - Soil properties
     - Hengl, T., J. M. de Jesus, G. B. M. Heuvelink, M. Ruiperez Gonzalez, M. Kilibarda, A. Blagotić, W. Shangguan, et al. 2017. SoilGrids250m: Global gridded soil information based on machine learning. Edited by Ben Bond-Lamberty. PLOS ONE (Public Library of Science (PLoS)) 12: e0169748. doi:10.1371/journal.pone.0169748.
   * - GRanD
     - Reservoirs
     - Lehner, B., C. Reidy Liermann, C. Revenga, C. Vörösmarty, B. Fekete, P. Crouzet, P. Döll, et al. 2011. High‐resolution mapping of the world’s reservoirs and dams for sustainable river‐flow management. Frontiers in Ecology and the Environment (Wiley) 9: 494–502. doi:10.1890/100125
   * - HydroLAKES
     - Lakes
     - Messager, M. L., B. Lehner, G. Grill, I. Nedeva, and O. Schmitt. 2016. Estimating the volume and age of water stored in global lakes using a geo-statistical approach. Nature Communications (Springer Science and Business Media LLC) 7. doi:10.1038/ncomms13603
   * - Randolph Glacier Inventory v0.6
     - Glaciers
     - Pfeffer, W. T., A. A. Arendt, A. Bliss, T. Bolch, J. G. Cogley, A. S. Gardner, J.-O. Hagen, et al. 2014. The Randolph Glacier Inventory: a globally complete inventory of glaciers. Journal of Glaciology (International Glaciological Society) 60: 537–552. doi:10.3189/2014jog13j176
   * - ERA5-Reanalysis
     - Climate
     - Hersbach, H., B. Bell, P. Berrisford, G. Biavati, A. Horányi, J. Muñoz Sabater, J. Nicolas, et al. 2018. ERA5 hourly data on pressure levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate . doi:10.24381/cds.bd0915c6
   * - CHIRPS-global
     - Climate
     - Funk, C., P. Peterson, M. Landsfeld, D. Pedreros, J. Verdin, S. Shukla, G. Husak, et al. 2015. The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Scientific Data (Springer Science and Business Media LLC) 2. doi:10.1038/sdata.2015.66
   * - CMIP6
     - Climate
     - Eyring, V., S. Bony, G. A. Meehl, C. A. Senior, B. Stevens, R. J. Stouffer, and K. E. Taylor. 2016. Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization. Geoscientific Model Development (Copernicus GmbH) 9: 1937–1958. doi:10.5194/gmd-9-1937-2016
   * - Observational station data of the ECA&D
     - Climate
     - Klein T., A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment. Int. J. of Climatol., 22, 1441-1453
   * - Global Runoff Data Centre (GRDC)
     - Hydrology
     - GRDC. 2020. https://grdc.bafg.de/
   * - MODIS/Terra Snow Cover
     - Hydrology
     - Hall, D., K. George, A. Riggs, and V. V. Salomonson. 2006. MODIS/Terra Snow Cover 5-Min L2 Swath 500m, Version 5. MODIS/Terra Snow Cover 5-Min L2 Swath 500m, Version 5. NASA National Snow and Ice Data Center Distributed Active Archive Center. doi:10.5067/ACYTYZB9BEOS