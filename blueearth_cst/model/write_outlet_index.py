"""Write the outlet-to-P1 identity crosswalk with positional compatibility labels.

hydromt_wflow 1.x labels outlets with basin-derived subcatchment IDs (e.g.
130000086), but this project uses a stable positional ``wflow_{1..N}`` naming
for plots and manifest paths (see dev/reference/workflows/model_creation.md). This rule
emits an unconditional, machine-readable mapping between the two, derived
directly from outlets.geojson and populated on **every** run — unlike
performance_metrics.csv, which is empty without observations.
"""
import os

import geopandas as gpd
import pandas as pd


def build_outlet_index(outlets_path, location_registry_path):
    """Join Wflow outlets to deterministic basin/subbasin/location identities."""
    gdf = gpd.read_file(outlets_path)
    subcatchment_id = (
        gdf["fid"].astype(int)
        if "fid" in gdf.columns
        else pd.Series(gdf.index, index=gdf.index, dtype="int64")
    )
    outlets = pd.DataFrame(
        {
            "compat_station_name": [f"wflow_{i + 1}" for i in range(len(gdf))],
            "subcatchment_id": list(subcatchment_id),
            "x": gdf.geometry.x.to_numpy(),
            "y": gdf.geometry.y.to_numpy(),
        }
    )
    registry = pd.read_csv(location_registry_path)
    primary = registry.loc[
        registry["location_id"].astype(int).eq(1),
        [
            "basin_code",
            "subbasin_id",
            "subbasin_code",
            "location_code",
            "station_name",
            "wflow_id",
        ],
    ]
    if primary["subbasin_id"].duplicated().any():
        raise ValueError("location_registry has duplicate primary subbasin identities")
    result = outlets.merge(
        primary,
        left_on="subcatchment_id",
        right_on="subbasin_id",
        how="left",
        validate="one_to_one",
    )
    if result["wflow_id"].isna().any():
        missing = result.loc[result["wflow_id"].isna(), "subcatchment_id"].tolist()
        raise ValueError(f"Wflow outlets have no primary registry identity: {missing}")
    result["wflow_id"] = result["wflow_id"].astype(int)
    return result


def write_outlet_index(outlets_path, location_registry_path, out_path):
    """Build the outlet index from ``outlets_path`` and write it to ``out_path``.

    Returns the number of stations written (for an informative rule log line).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = build_outlet_index(outlets_path, location_registry_path)
    df.to_csv(out_path, index=False)
    return len(df)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            n_stations = write_outlet_index(
                sm.input.outlets_path,
                sm.input.location_registry,
                sm.output.outlet_index_path,
            )
            log_row(
                f"Wrote outlet index: {n_stations} station(s) -> "
                f"{sm.output.outlet_index_path}",
                module="outlets",
            )
    else:
        write_outlet_index(
            os.path.join(
                os.getcwd(),
                "test_case",
                "my_project",
                "hydrology_model",
                "staticgeoms",
                "outlets.geojson",
            ),
            os.path.join(
                os.getcwd(),
                "test_case",
                "my_project",
                "spatial",
                "location_registry.csv",
            ),
            os.path.join(
                os.getcwd(),
                "test_case",
                "my_project",
                "hydrology_model",
                "staticgeoms",
                "outlet_index.csv",
            ),
        )
