#%%
import os
import hydromt_wflow
import matplotlib.pyplot as plt
import numpy as np 
import xarray as xr

#%% Options to modify to use the script for a specific basin
cases = {
    "Nepal_Seti_500m": {
    #    "toml": "hydrology_model/run_default/wflow_sbm_era5_imdaa_clim.toml",
       "toml": "hydrology_model/run_default/wflow_sbm_era5.toml",
       },
    # "Bhutan_Damchuu": {
    #     "toml": "hydrology_model/run_default/wflow_sbm_imdaa.toml",
    #     },
    #"test_project": {
    #    "toml": "hydrology_model/run_default/wflow_sbm_era5.toml",
    #},
}
folder_p = r"p:\11210673-fao\14 Subbasins"

data_libs = [
    "deltares_data==v0.7.0", 
    r"p:\11210673-fao\fao_data.yml",
    "p:\wflow_global\hydromt_wflow\catalog.yml", #datacatalog for ksathorfrac and other water demand data
]
#data_libs = [
#    r"C:\Users\boisgont\.hydromt_data\artifact_data\v0.0.9\data_catalog.yml", 
#    r"d:\Repos\CST\blueearth_cst\tests\data\tests_data_catalog.yml",
#]

lai_fn = "modis_lai"
lulc_fn = "rlcms_2021"
lulc_mapping_fn = r"p:\11210673-fao\12 Data\ICIMOD\landuse\rlcms_mapping.csv"
soil_fn = "soilgrids"
lulc_fn_with_paddy = "landuse_with_paddy.tif"
lulc_with_paddy_mapping_fn = "landuse_with_paddy_mapping.csv"

# Method to prepare the mapping table
# any, mode, q3
lulc_sampling_method = "q3"
# Landuse classes in lulc_fn for which LAI should be zero
# vito
# lulc_zero_classes = [80, 200, 0] # decide for snow and ice 70 and urban 50
# esa_worldcover
#lulc_zero_classes = [80, 0] # decide for snow and ice 70 and urban 50
# rlcms
lulc_zero_classes = [1, 2, 4, 8, 0] # decide for snow and ice 70 and urban 50

#%% functions to modify ksatver

def get_ksmax(KsatVer, sndppt):
    """
    based on Bonetti et al. 2021

    Parameters
    ----------
    KsatVer : TYPE
        saturated hydraulic conductivity derived from PTF based on soil properties [cm/d].
    sndppt : TYPE
        percentage sand .

    Returns
    -------
    Ksmax
        KsatVer with fully developped vegetation - depends on soil texture.

    """
    ksmax = 10**(3.5 - 1.5*sndppt**0.13 + np.log10(KsatVer))
    return ksmax

def get_ks_veg(ksmax, KsatVer, LAI, alfa=4.5, beta=5):
    """
    based on Bonetti et al. 2021

    Parameters
    ----------
    ksmax : TYPE
        Saturated hydraulic conductivity with fully developed vegetation.
    KsatVer : TYPE
        Saturated hydraulic conductivity.
    LAI : TYPE
        Leaf area index.
    alfa : TYPE, optional
        Shape parameter. The default is 4.5 when using LAI.
    beta : TYPE, optional
        Shape parameter. The default is 5 when using LAI.

    Returns
    -------
    ks : TYPE
        saturated hydraulic conductivity based on soil and vegetation.

    """
    ks = ksmax - (ksmax - KsatVer) / (1 + (LAI/alfa)**beta)
    return ks

def get_ksatver_bonetti(KsatVer, sndppt, LAI, alfa=4.5, beta=5):
    """
    saturated hydraulic conductivity based on soil texture and vegetation
    based on Bonetti et al. 2021

        Parameters
    ----------
    KsatVer : TYPE
        Saturated hydraulic conductivity [cm/d].
    sndppt : TYPE
        percentage sand [%].
    LAI : TYPE
        Mean leaf area index [-].
    alfa : TYPE, optional
        Shape parameter. The default is 4.5 when using LAI.
    beta : TYPE, optional
        Shape parameter. The default is 5 when using LAI.

    Returns
    -------
    ks : TYPE
        saturated hydraulic conductivity based on soil and vegetation [cm/d].

    """
    ksmax = get_ksmax(KsatVer, sndppt)
    ks = get_ks_veg(ksmax, KsatVer, LAI, alfa, beta)
    return ks

#%% Main function

for case in cases:
    print(f"Updating model {case}")
    toml = cases[case]["toml"]
    toml_default_fn = os.path.join(folder_p, case, toml)

    # Instantiate wflow model and read/update config
    root = os.path.dirname(toml_default_fn)
    mod = hydromt_wflow.WflowModel(root, config_fn=os.path.basename(toml_default_fn), 
                                mode="r+", 
                                data_libs=data_libs)
    ds = mod.grid

    #% update model 
    #ksathorfrac
    print("Computing ksathorfrac RF map")
    mod.setup_ksathorfrac(
        ksat_fn = "ksathorfrac",
        variable = "RF_250", 
    )

    #now ksatver
    print("Computing KsatVer with Brakensiek and Cosby")
    
    #update ksatver with cosby 
    mod.setup_soilmaps(
        soil_fn=soil_fn,
        ptf_ksatver="cosby",
    )

    #rename to cosby 
    ksaver_cosby = mod.grid["KsatVer"]
    ksaver_cosby.name = "KvC"
    mod.set_grid(ksaver_cosby, "KvC")

    #to avoid confusion make KsatVer equal to Brakensiek and rename map to Brakensiek
    mod.setup_soilmaps(
        soil_fn=soil_fn,
        ptf_ksatver="brakensiek",
    )

    #first make a copy of ksatver brakensiek 
    ksaver_brakensiek = mod.grid["KsatVer"]
    ksaver_brakensiek.name = "KvB"
    mod.set_grid(ksaver_brakensiek, "KvB")

    # plt.figure(); mod.grid["KsatVer"].raster.mask_nodata().plot()
    # plt.figure(); mod.grid["KsatVer_Brakensiek"].raster.mask_nodata().plot()
    # plt.figure(); mod.grid["KsatVer_Cosby"].where(mod.grid["KsatVer_Brakensiek"]>0).raster.mask_nodata().plot()

    #% prepare LAI based on landuse map
    print("Preparing LAI based on landuse")
    # first copy and save the MODIS LAI
    lai_modis = mod.grid["LAI"]
    lai_modis.name = "LAI_modis"
    mod.set_grid(lai_modis, "LAI_modis")

    # also derive lai landuse table with "mode" and "any" sample methods
    # Prepare lai landuse tables
    for lulc_sampling_method in ["mode", "any", "q3"]:
        mod.setup_laimaps(
            lai_fn = lai_fn,
            lulc_fn = xr.open_dataarray(os.path.join(folder_p, case, "hydrology_model", "maps", lulc_fn_with_paddy)),
            lulc_sampling_method = lulc_sampling_method,
            lulc_zero_classes = lulc_zero_classes,
            buffer = 5,
        )
        #rename table
        os.rename(os.path.join(mod.root, f"lai_per_lulc.csv"), 
                os.path.join(mod.root, f"lai_per_lulc_{lulc_sampling_method}.csv")
        )

    #NB: manual step of combining if necessary LAI tables to one that will be used to create the LAI maps.
    # manual step create lai_per_lulc.csv in mod.root that will be used to create the LAI maps!

    # Derive LAI based on landuse rlcms_2021
    mod.setup_laimaps_from_lulc_mapping(
        lulc_fn = lulc_fn,
        lai_mapping_fn = os.path.join(mod.root, f"lai_per_lulc.csv")
    )

    #% modify ksatver to account for vegetation 
    print("Modifying ksatver with vegetation")
    # percentage sand is required
    #get percentage sand soilgrids 
    soilgrids = mod.data_catalog.get_rasterdataset(soil_fn, bbox = mod.geoms["basins"].geometry.total_bounds)
    snd_ppt = soilgrids["sndppt_sl1"]
    snd_ppt = snd_ppt.where(snd_ppt!=snd_ppt._FillValue, np.nan)
    snd_ppt = snd_ppt.raster.reproject_like(ds, method="average")
    snd_ppt.raster.set_nodata(np.nan)
    #interpolate
    snd_ppt = snd_ppt.raster.interpolate_na("rio_idw")
    #mask
    snd_ppt = snd_ppt.where(ds["wflow_subcatch"]>0)

    #max lai is required
    LAI_mean = ds["LAI"].max("time")
    LAI_mean.raster.set_nodata(255.)

    KSatVer_Brakensiek_Bonetti = get_ksatver_bonetti(KsatVer=mod.grid["KvB"]/10, 
                                                    sndppt=snd_ppt, 
                                                    LAI=LAI_mean, 
                                                    alfa=4.5, beta=5)

    KSatVer_Cosby_Bonetti = get_ksatver_bonetti(KsatVer=mod.grid["KvC"]/10, 
                                                    sndppt=snd_ppt, 
                                                    LAI=LAI_mean, 
                                                    alfa=4.5, beta=5)

    # plt.figure(); (KSatVer_Brakensiek_Bonetti*10).plot()
    # plt.figure(); (KSatVer_Cosby_Bonetti*10).plot()

    ksaver_brakensiek_bonetti = KSatVer_Brakensiek_Bonetti*10
    ksaver_brakensiek_bonetti.name = "KvBB"
    mod.set_grid(ksaver_brakensiek_bonetti, "KvBB")

    ksaver_cosby_bonetti = KSatVer_Cosby_Bonetti*10
    ksaver_cosby_bonetti.name = "KvCB"
    mod.set_grid(ksaver_cosby_bonetti, "KvCB")

    print("Setup irrigation (should be done after LAI and landuse)")
    mod.setup_irrigation(
       irrigated_area_fn = "lgrip30",
       irrigation_value = [2],
       cropland_class = [6],
       paddy_class = [10],
       area_threshold = 0.6,
       lai_threshold = 0.2
    )    

    print("Saving the model")
    mod.write_grid(os.path.join(folder_p, case, "hydrology_model", "staticmaps_ksat.nc"))


    #update the toml with settings for reinfiltration option
    # setting_toml = {
        # "model.surface_water_infiltration" : True,
        #add a map with threshold in meters # todo! 
        # for the area value and for the rest if local inertial value should be 1.0e-3
        # "input.lateral.land.h_thresh": "name_map",
    # }

    # for option in setting_toml:
    #     mod.set_config(option, setting_toml[option])

    mod.set_config("input.path_static", "staticmaps_ksat.nc")
    mod.write_config(os.path.basename(toml).split(".")[0] + "_ksat.toml")

