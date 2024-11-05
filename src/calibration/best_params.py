import numpy as np
import pandas as pd 
from pathlib import Path
from filelock import FileLock
import xarray as xr
import os
import ast
from itertools import combinations
import matplotlib.pyplot as plt
from plotly import graph_objects as go

def best_col(results_file, col):
    if col not in ["euclidean"]:
        sorted = results_file.sort_values(by=col, ascending=False)
        sorted[f"{col}_rank"] = sorted.index
    else:
        sorted = results_file.sort_values(by=col, ascending=True)
        sorted[f"{col}_rank"] = sorted.index
    return sorted.iloc[0], sorted
def normalize_series(series):
    """ Normalize the series such that the optimum (0) is the best value. """
    max_value = series.max()
    min_value = series.min()
    # return (series - min_value) / (max_value - min_value)
    return 1-series
def pareto_front(data):
    # Sort the data by the first metric
    sorted_data = data.sort_values(by=data.columns[0])
    pareto_indices = []

    for i in range(len(sorted_data)):
        # Check if the current point is dominated by any of the previous points
        if not any((sorted_data.iloc[i] <= sorted_data.iloc[j]).all() for j in pareto_indices):
            pareto_indices.append(i)

    return sorted_data.iloc[pareto_indices]
def pareto_pair(results_file, comb, plot_folder):
    m1, m2 = comb
    
    # Normalize the metrics
    results_file[m1] = normalize_series(results_file[m1])
    results_file[m2] = normalize_series(results_file[m2])

    # Create a DataFrame for the two metrics
    data_pair = results_file[[m1, m2]]

    # Calculate the Pareto front
    pareto_front_data = pareto_front(data_pair)
    
    # Calculate Euclidean distances for all points
    distances = results_file["euclidean"]
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Create a colormap
    cmap = plt.get_cmap('viridis_r',)  # You can choose any colormap you like

    # Get the indices of the Pareto front points
    pareto_indices = pareto_front_data.index

    # Map normalized distances to colors only for the Pareto front points
    colors = cmap(norm_distances[pareto_indices])  # Use only the distances for Pareto front points

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pareto_front_data[m1], pareto_front_data[m2], color=colors, label='Pareto Front')

    # Set symmetrical limits for x-axis based on x-values
    max_x = data_pair[m1].max()
    min_x = data_pair[m1].min()
    symmetrical_limit_x = max(abs(min_x), abs(max_x)) * 1.15
    plt.xlim(-symmetrical_limit_x, symmetrical_limit_x)

    # Set symmetrical limits for y-axis based on y-values
    max_y = data_pair[m2].max()
    min_y = data_pair[m2].min()
    symmetrical_limit_y = max(abs(min_y), abs(max_y)) * 1.15	
    plt.ylim(-symmetrical_limit_y, symmetrical_limit_y)

    # Highlight the minimum Euclidean point
    min_euclid = results_file["euclidean"].min()
    x_min_euclid = data_pair[m1].loc[results_file["euclidean"] == min_euclid]
    y_min_euclid = data_pair[m2].loc[results_file["euclidean"] == min_euclid]
    plt.scatter(x_min_euclid, y_min_euclid, color='red', label='Minimum Euclidean', s=100, marker="+")

    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar.set_label('Normalized Euclidean Distance')  # Label for the colorbar

    plt.title(f'Pareto Front for {m1} vs {m2} (Normalized)')
    plt.xlabel(m1)
    plt.ylabel(m2)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add horizontal line at y=0
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Add vertical line at x=0
    plt.legend()
    plt.grid()
    os.makedirs(plot_folder / "pareto_fronts", exist_ok=True)
    plt.savefig(Path(plot_folder, "pareto_fronts", f"{m1}_vs_{m2}.png"))
def main(calib_dir, observed, out_interactive):
    results_file = pd.read_csv(Path(calib_dir, "performance_appended.txt"), sep=",", index_col=None)
    print(results_file)
    forcing = [p for p in Path(calib_dir).parts if np.any(["era5" in p, "imdaa" in p])][0]
    metrics = [col for col in results_file.columns if not col in ["params", "gauge"]]
    ne_metrics = [m for m in metrics if m != "euclidean"]
    combs = list(combinations(ne_metrics, 2))
    plot_folder = Path(calib_dir.split("hydrology_model")[0]) / "plots" / "calibration" / forcing / "best_params"
    plot_folder.mkdir(exist_ok=True, parents=True)
    interactive_plot_folder = plot_folder / "interactive"
    interactive_plot_folder.mkdir(exist_ok=True, parents=True)
    gauge = results_file["gauge"].iloc[0]
    
    #PARETO
    # for comb in combs:
    #     pareto_pair(results_file, comb, plot_folder)
    
    observed = pd.read_csv(observed, index_col='time', parse_dates=True, sep=";")
    uncalibrated_pattern = "*ksat~1.0_f~1.0_rd~1.0_st~1.0_ml~0*.csv"
    uncalibrated_files = list(Path(calib_dir).glob(uncalibrated_pattern))
    if not uncalibrated_files:
        raise FileNotFoundError(f"No uncalibrated run file found matching pattern: {uncalibrated_pattern}")
    uncalibrated = pd.read_csv(uncalibrated_files[0], index_col=0, parse_dates=True)

    final_dict = {}

    # Create a Plotly figure
    fig = go.Figure()

    # Create a color palette for the different metrics
    colors = ['blue', 'red', 'green', 'purple']  # Add more colors if needed

    fig.add_trace(go.Scatter(x=observed.index, 
                             y=observed[f"{gauge}"], 
                             name="Observed",
                             line=dict(color='black',
                                       dash='dashdot',
                                       width=2)))
    fig.add_trace(go.Scatter(x=uncalibrated.index, 
                             y=uncalibrated[f"Q_{gauge}"], 
                             name="Uncalibrated",
                             line=dict(color='grey',
                                       dash='dashdot',
                                       width=2)))

    # Loop through metrics to plot each best parameter set
    for i, metric in enumerate(metrics):
        best_param, results_file = best_col(results_file, metric)
        print(f"{'*'*10}\nmetric: {metric}\n{results_file}\n{'*'*10}")
        
        params = best_param["params"]
        file_pattern = f"output_{params}.csv"
        file = Path(calib_dir) / file_pattern
        #testing
        # file = Path('p:\\11210673-fao\\14 Subbasins\\Bhutan_Damchhu_500m_v2\\hydrology_model\\run_calibrations\\era5_imdaa_clim_soil_cal\\output_ksat~1.0_f~0.75_rd~1.5_st~1.5_ml~0.3_cfm~3.7_gcf~30_tt~0_ttm~0_gtt~0_gsif~0.002.csv')

        # Read the data
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        
        q_series = df[f"Q_{gauge}"]
        per_run_metrics = {metric: best_param[metric] for metric in metrics}
        summary_dict = {pv.split("~")[0]:float(pv.split("~")[1]) for pv in params.split("_")}
        summary_dict.update({"gauge": gauge})
        summary_dict.update(per_run_metrics)
        final_dict[f"{metric}_best"] = summary_dict
        # Add trace for this metric's best parameter set
        fig.add_trace(
            go.Scatter(
                x=df.index,  # Assuming index represents time
                y=q_series,
                name=f'Best {metric}, {per_run_metrics}',
                line=dict(color=colors[i % len(colors)]),
                opacity=0.7
            )
        )

    # Update layout
    fig.update_layout(
        title='Discharge Time Series for Best Parameter Sets',
        xaxis_title='Time Step',
        yaxis_title='Discharge',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)

    # Save the figure
    fig.write_html(out_interactive)
    
    # Show the figure
    fig.show()

    #final_dict to csv
    final_df = pd.DataFrame(final_dict)

    final_df.to_csv(Path(calib_dir, "best_params.csv"), index=False)


if __name__ == "__main__":
    """
    Combined performance analysis
    """

    if "snakemake" in globals():
        snakemake = globals()["snakemake"]
        main(snakemake.params.calib_folder, snakemake.params.observed, snakemake.output)
    else:
        print("Not running under snakemake")
        main("p:/11210673-fao/14 Subbasins/Bhutan_Damchhu_500m_v2/hydrology_model/run_calibrations/era5_imdaa_clim_soil_cal",
            "p:/11210673-fao/12 Data/Bhutan/ObservedFlow_csv/discharge-bhutan.csv")
        
