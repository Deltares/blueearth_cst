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
    """
    Normalize the series such that:
    - Best value (originally 1) becomes 0
    - Worse values (negative) become positive numbers
    - All values are between 0 and 1
    
    Specifically for metrics like KGE/NSE where:
    - 1 is perfect
    - Values can go to negative infinity
    - We want 0 to be best after normalization
    """
    # First shift everything so perfect score (1) becomes 0
    shifted = series - 1  # Now best value (1) becomes 0, worse values are negative
    
    # Make all values positive while preserving relative distances
    positive = -shifted  # Now best value (0) becomes 0, worse values are positive
    
    # Normalize to 0-1 range
    if positive.max() != 0:  # Avoid division by zero
        normalized = positive / positive.max()
    else:
        normalized = positive
        
    return normalized
def pareto_front(data):
    """Find the pareto front points in a dataset"""
    # Sort the data by the first metric
    sorted_data = data.sort_values(by=data.columns[0])
    pareto_indices = []

    for i in range(len(sorted_data)):
        # For KGE/NSE type metrics, higher is better, so we check for domination using >=
        if not any((sorted_data.iloc[i] <= sorted_data.iloc[j]).all() for j in pareto_indices):
            pareto_indices.append(i)

    return sorted_data.iloc[pareto_indices]
def pareto_pair(results_file, comb, plot_folder, colorby_col, show=False):
    """Plot pareto front for a pair of metrics with multiple coloring schemes"""
    m1, m2 = comb
    
    # Create a DataFrame for the two metrics and soil parameters
    data_pair = results_file[[m1, m2] + colorby_col]
    
    # Calculate the Pareto front
    pareto_front_data = pareto_front(data_pair[[m1, m2]])  # Only use metrics for Pareto calculation
    
    # Add soil parameters back to pareto_front_data
    pareto_front_data = pareto_front_data.join(results_file[colorby_col])
    
    # List of parameters to color by
    color_params = colorby_col
    
    for color_param in color_params:
        # Plotting
        plt.figure(figsize=(10, 10))
        
        if color_param == 'euclidean':
            # Continuous colormap for euclidean distance
            values = results_file[color_param]
            norm_values = (values - values.min()) / (values.max() - values.min())
            cmap = plt.get_cmap('viridis_r')
            pareto_colors = cmap(norm_values[pareto_front_data.index])
            
            # Plot Pareto front points
            scatter = plt.scatter(1-pareto_front_data[m1], 1-pareto_front_data[m2], 
                                color=pareto_colors, 
                                label='Pareto Front',
                                zorder=5)
            
            # Add continuous colorbar
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
            cbar.set_label('Euclidean Distance')
            
        else:
            # Discrete colors but using a gradient
            unique_values = sorted(results_file[color_param].unique())
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(min(unique_values), max(unique_values))
            
            # Plot Pareto front points
            scatter = plt.scatter(1-pareto_front_data[m1], 1-pareto_front_data[m2],
                                c=pareto_front_data[color_param],
                                cmap=cmap,
                                norm=norm,
                                zorder=5)
            
            # Add discrete colorbar
            cbar = plt.colorbar(scatter, 
                              ticks=unique_values,
                              ax=plt.gca())
            cbar.set_label(color_param)
        
        # Highlight the minimum Euclidean point
        min_euclid_idx = results_file["euclidean"].idxmin()
        min_point = plt.scatter(1-results_file[m1][min_euclid_idx], 
                             1-results_file[m2][min_euclid_idx], 
                              color='red', s=200, marker="+", 
                              label='Min Euclidean',
                              zorder=6,
                              linewidth=2)
        
        # Add vector arrow from origin to min euclidean point
        min_x = 1-results_file[m1][min_euclid_idx]
        min_y = 1-results_file[m2][min_euclid_idx]
        plt.arrow(0, 0, min_x, min_y, 
                 color='red', alpha=0.3, 
                 length_includes_head=True,
                 head_width=0.001, head_length=0.001,
                 zorder=4)
        
        # Add reference lines at origin
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.8)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.8)
        
        # Labels and titles
        plt.title(f'Pareto Front for {m1} vs {m2}')
        plt.suptitle(f'Min Euclidean Point: {m1}={min_x:.2f}, {m2}={min_y:.2f}', 
                     y=0.95)
        plt.xlabel(f'{m1}')
        plt.ylabel(f'{m2}')
        plt.grid(True, alpha=0.3)
        
        # Legend only for Pareto front and min euclidean point
        plt.legend(['Pareto Front', 'Min Euclidean'], loc='upper left')
        
        # Save plot
        plot_subfolder = plot_folder / "pareto_fronts" / f"colored_by_{color_param}"
        os.makedirs(plot_subfolder, exist_ok=True)
        plt.savefig(plot_subfolder / f"{m1}_vs_{m2}.png", 
                    bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()
def plot_metric_vs_euclidean(results_file, metric, plot_folder, show=False):
    """Plot relationship between a metric and euclidean distance"""
    plt.figure(figsize=(10, 6))
    
    # Normalize metric if it's KGE or NSE type (where 1 is best)
    if metric != "euclidean":
        y_values = normalize_series(results_file[metric])
    else:
        y_values = results_file[metric]
    
    # Normalize distances for coloring
    distances = results_file["euclidean"]
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    # Create colormap
    cmap = plt.get_cmap('viridis_r')  # Using reversed viridis so darker = better
    
    # Create scatter plot with colormap
    scatter = plt.scatter(results_file["euclidean"], y_values, 
                         c=norm_distances, 
                         cmap=cmap, 
                         alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Euclidean Distance')
    
    # Highlight minimum euclidean point
    min_euclid_idx = results_file["euclidean"].idxmin()
    plt.scatter(results_file["euclidean"][min_euclid_idx], 
               y_values[min_euclid_idx], 
               color='red', 
               s=100, 
               marker='+', 
               label='Min Euclidean')
    
    plt.title(f'{metric} vs Euclidean Distance')
    plt.xlabel('Euclidean Distance')
    plt.ylabel(f'Normalized {metric}')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    os.makedirs(plot_folder / "metric_vs_euclidean", exist_ok=True)
    plt.savefig(plot_folder / "metric_vs_euclidean" / f"{metric}_vs_euclidean.png")
    if show:
        plt.show()
    else:
        plt.close()
def parse_params_column(results_file):
    """Convert the params column string into separate parameter columns"""
    # Create new dataframe with existing data
    parsed_df = results_file.copy()
    
    # Parse the first row to get parameter names
    sample_params = results_file['params'].iloc[0]
    param_names = [p.split('~')[0] for p in sample_params.split('_')]
    
    # Function to parse a single parameter string
    def parse_param_string(param_str):
        return {p.split('~')[0]: float(p.split('~')[1]) 
                for p in param_str.split('_')}
    
    # Create new columns for each parameter
    param_dict = parsed_df['params'].apply(parse_param_string)
    param_df = pd.DataFrame(param_dict.tolist(), index=parsed_df.index)
    
    # Combine with original dataframe and drop the original params column
    parsed_df = pd.concat([parsed_df, param_df], axis=1)
    # parsed_df.drop('params', axis=1, inplace=True)
    
    return parsed_df
def concat_cal_eval(cal_file, eval_file):
    cal = pd.read_csv(cal_file, sep=",", index_col=None)
    cal = parse_params_column(cal)
    cal['eval_type'] = 'cal'
    eval = pd.read_csv(eval_file, sep=",", index_col=None)
    eval = parse_params_column(eval)
    eval['eval_type'] = 'eval'
    return pd.concat([cal, eval], axis=0)

def main(cal_file, eval_file, calib_dir, cal_html, eval_html, combined_html):
    #concat eval and cal with the 
    eval_dict = {"cal":(cal_file, cal_html), 
                 "eval":(eval_file, eval_html), 
                 "combined":(concat_cal_eval(cal_file, eval_file), combined_html)}
    
    for eval_type, (eval_file, out_html) in eval_dict.items():
        if isinstance(eval_file, str | Path):
            results_file = pd.read_csv(eval_file, sep=",", index_col=None)
            results_file = parse_params_column(results_file)
        else:
            results_file = eval_file
        
        if eval_type == "combined":
            colorby_col = ["eval_type"]
        else:
            colorby_col = ["ksat", "f", "rd", "st", "ml"]
        
        # print(results_file)
        forcing = [p for p in Path(calib_dir).parts if np.any(["era5" in p, "imdaa" in p])][0]
        metrics = ['nse', 'kge', 'nse_log']
        ne_metrics = [m for m in metrics if m != "euclidean"]
        combs = list(combinations(ne_metrics, 2))
        plot_folder = Path(calib_dir.split("hydrology_model")[0]) / "plots" / "calibration" / forcing / eval_type / "best_params"
        plot_folder.mkdir(exist_ok=True, parents=True)
        interactive_plot_folder = plot_folder / "interactive"
        interactive_plot_folder.mkdir(exist_ok=True, parents=True)
        gauge = results_file["gauge"].iloc[0]
        
        #METRIC VS EUCLIDEAN
        for metric in ne_metrics:
            plot_metric_vs_euclidean(results_file, metric, plot_folder, show=False)

        #PARETO
        for comb in combs:
            pareto_pair(results_file, comb, plot_folder, colorby_col, show=True)
        
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
            # print(f"{'*'*10}\nmetric: {metric}\n{results_file}\n{'*'*10}")
            
            params = best_param["params"]
            file_pattern = f"output_{eval_type}_{params}.csv"
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
        fig.write_html(str(out_interactive))


        #final_dict to csv
        final_df = pd.DataFrame(final_dict)

        final_df.to_csv(Path(calib_dir, "best_params.csv"), index=False)


if __name__ == "__main__":
    """
    Combined performance analysis
    """

    if "snakemake" in globals():
        snakemake = globals()["snakemake"]
        main(snakemake.input.cal_file,
             snakemake.input.eval_file,
             snakemake.params.calib_folder, 
             snakemake.params.observed, 
             snakemake.output.cal_html,
             snakemake.output.eval_html,
             snakemake.output.combined_html)
    else:
        print("Not running under snakemake")
        main("p:/11210673-fao/14 Subbasins/Bhutan_Damchhu_500m_v2/hydrology_model/run_calibrations/era5_imdaa_clim_soil_cal",
            "p:/11210673-fao/12 Data/Bhutan/ObservedFlow_csv/discharge-bhutan.csv",
            "p:/11210673-fao/14 Subbasins/Bhutan_Damchhu_500m_v2/plots/calibration/era5_imdaa_clim_soil_cal/best_params/interactive/best_params_timeseries.html"
            )
        
