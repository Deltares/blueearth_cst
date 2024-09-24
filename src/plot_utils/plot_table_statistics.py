"""Heatmap plots for tables with statistics."""

import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_table_statistics", "plot_table_statistics_multiindex"]


def plot_table_statistics(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    x_label: str = "Scenario",
    y_label: str = "Parameters",
    cmap: str = "RdBu",
    cmap_label: str = "Relative Change [%]",
    vmin: Optional[float] = -100,
    vmax: Optional[float] = 100,
    invert_cmap_for: List[str] = [],
    bold_keyword: Optional[str] = None,
):
    """
    Plot a heatmap based on a table with a single index column.

    Other columns should represents the parameters to be plotted.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to be plotted.
    output_path : str
        Path to save the plot.
    x_label : str
        Label for the x-axis (index). Default is 'Scenario'.
    y_label : str
        Label for the y-axis (parameters). Default is 'Parameters'.
    cmap : str
        Colormap to be used. Default is 'RdBu'.
    cmap_label : str
        Label for the colormap. Default is 'Relative Change [%]'.
    vmin : int
        Minimum value for the colormap. Default is -100.
    vmax : int
        Maximum value for the colormap. Default is 100.
    invert_cmap_for : list
        List of parameters for which the colormap should be inverted. Default is [].
        If the cmap was inverted, this will be shown with a '*' added to the parameter
        name(s) in the y-axis.
    bold_keyword : str
        Keyword in the index names (scenarios) for which the values should be in bold.
    """
    # Check if colormap should be inverted for some parameters
    invert_cmap = []
    for param in invert_cmap_for:
        if param in df.columns:
            df[param] = -df[param]
            df = df.rename(columns={param: f"{param} *"})
            invert_cmap.append(f"{param} *")

    # Transpose the dataframe to have the parameters in the y-axis
    df = df.T

    # Plot the heatmap
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    im = sns.heatmap(
        df,
        ax=ax,
        annot=True,
        linewidth=0.5,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fmt=".1f",
        cbar_kws={"label": cmap_label},
    )

    # Change the x and y labels
    im.set_xlabel(x_label, fontweight="bold")
    im.set_ylabel(y_label, fontweight="bold")
    # Rotate the x labels
    plt.xticks(rotation=45, ha="center")

    # Change the annotations (inverted cmap and bold annotations)
    for i in range(len(im.texts)):
        # Check if annotation should be bold
        j = i % len(df.columns)
        if bold_keyword is not None and bold_keyword in df.columns[j][0]:
            text = im.texts[i]
            text.set_fontweight("bold")
        # Check if the cmap was inverted
        k = i // len(df.columns)
        if df.index[k] in invert_cmap:
            text = im.texts[i]
            # Change the text to positive values
            text.set_text(str(-float(text.get_text())))

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_table_statistics_multiindex(
    df,
    output_path,
    x_label="Scenario",
    y_label="Parameters",
    cmap="RdBu",
    cmap_label="Relative Change [%]",
    vmin=-100,
    vmax=100,
    index_separator=":\n",
    invert_cmap_for=[],
    bold_keyword=None,
):
    """
    Plot a heatmap based on a table with multi-index.

    Multi-index will be combined into a single index for plotting.

    See plot_table_statistics for more information on the parameters.
    """
    # Convert index legends to uppercase
    for i in range(len(df.index.levels)):
        if df.index.levels[i].dtype == "object":
            df.index = df.index.set_levels(df.index.levels[i].str.upper(), level=i)

    # Combine values in multi-index columns to create a single index
    new_index = df.index.get_level_values(0).astype(str)
    for i in range(1, len(df.index.levels)):
        new_index = [
            new_index + index_separator + df.index.get_level_values(i).astype(str)
        ]
    df.index = new_index
    df.index.name = x_label

    # Plot the heatmap
    plot_table_statistics(
        df,
        output_path,
        x_label=x_label,
        y_label=y_label,
        cmap=cmap,
        cmap_label=cmap_label,
        vmin=vmin,
        vmax=vmax,
        invert_cmap_for=invert_cmap_for,
        bold_keyword=bold_keyword,
    )
