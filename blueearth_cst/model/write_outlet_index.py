"""Write outlet_index.csv: the positional station -> subcatchment-ID map (R3 section 4).

hydromt_wflow 1.x labels outlets with basin-derived subcatchment IDs (e.g.
130000086), but this project uses a stable positional ``wflow_{1..N}`` naming
for plots and manifest paths (see dev/workflows/model_creation.md). This rule
emits an unconditional, machine-readable mapping between the two, derived
directly from outlets.geojson and populated on **every** run — unlike
performance_metrics.csv, which is empty without observations.
"""
import os

import geopandas as gpd
import pandas as pd


def build_outlet_index(outlets_path):
    """Return a DataFrame mapping positional station_name to subcatchment id + xy.

    Columns: station_name (``wflow_1``..``wflow_N`` in file order),
    subcatchment_id (the outlets ``fid``, or the index if no ``fid`` column),
    x, y (point coordinates).
    """
    gdf = gpd.read_file(outlets_path)
    subcatchment_id = gdf["fid"] if "fid" in gdf.columns else gdf.index
    return pd.DataFrame(
        {
            "station_name": [f"wflow_{i + 1}" for i in range(len(gdf))],
            "subcatchment_id": list(subcatchment_id),
            "x": gdf.geometry.x.to_numpy(),
            "y": gdf.geometry.y.to_numpy(),
        }
    )


def write_outlet_index(outlets_path, out_path):
    """Build the outlet index from ``outlets_path`` and write it to ``out_path``.

    Returns the number of stations written (for an informative rule log line).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = build_outlet_index(outlets_path)
    df.to_csv(out_path, index=False)
    return len(df)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            n_stations = write_outlet_index(
                sm.input.outlets_path, sm.output.outlet_index_path
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
                "hydrology_model",
                "staticgeoms",
                "outlet_index.csv",
            ),
        )
