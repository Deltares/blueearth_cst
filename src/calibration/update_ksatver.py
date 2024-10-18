#%%
import os
import hydromt_wflow
import matplotlib.pyplot as plt
import numpy as np 
import xarray as xr

#%% Options to modify to use the script for a specific basin
cases = {
    "Nepal_Seti_500m_v1": {
    #    "toml": "hydrology_model/run_default/wflow_sbm_era5_imdaa_clim.toml",
       "toml": "hydrology_model/wflow_sbm.toml",
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

soil_fn = "soilgrids"

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

    KSatVer_Brakensiek_Bonetti = get_ksatver_bonetti(KsatVer=mod.grid["KsatVer"]/10, 
                                                    sndppt=snd_ppt, 
                                                    LAI=LAI_mean, 
                                                    alfa=4.5, beta=5)

    ksaver_brakensiek_bonetti = KSatVer_Brakensiek_Bonetti*10
    ksaver_brakensiek_bonetti.name = "KvB"
    mod.set_grid(ksaver_brakensiek_bonetti, "KvB")

    print("Saving the model - overwrite staticmaps")
    mod.write_grid(os.path.join(folder_p, case, "hydrology_model", "staticmaps.nc"))


    #update the toml with settings for reinfiltration option
    # setting_toml = {
        # "model.surface_water_infiltration" : True,
        #add a map with threshold in meters # todo! 
        # for the area value and for the rest if local inertial value should be 1.0e-3
        # "input.lateral.land.h_thresh": "name_map",
    # }

    # for option in setting_toml:
    #     mod.set_config(option, setting_toml[option])

    # mod.set_config("input.path_static", "staticmaps_ksat.nc")
    # mod.write_config(os.path.basename(toml).split(".")[0] + "_ksat.toml")

