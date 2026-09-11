# -*- coding: utf-8 -*-
"""画图工具（vendor 自源仓库 plots/，开源版去掉 seaborn 依赖）。"""

import matplotlib
matplotlib.use("Agg")  # 无头后端：headless 节点无 DISPLAY（须在 import pyplot 之前）
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import logging
import sys
from typing import List

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

def plot_forecast_metrics(
    rmse: np.ndarray,
    acc: np.ndarray,
    activity: np.ndarray,
    variables: list,
    title: str = "Forecast Metrics",
    cmap: str = "viridis",
    figsize: tuple = (20, 16),
    dpi: int = 300
) -> List[plt.Figure]:
    """Generate forecast metric line plots for RMSE, ACC, and activity in a 4x5 grid."""

    figures = []

    # Define the variables we want to plot (300hPa level)
    target_vars = [
        'z-300', 't-300', 'u-300', 'v-300', 'q-300',
        'z-500', 't-500', 'u-500', 'v-500', 'q-500',
        'z-850', 't-850', 'u-850', 'v-850', 'q-850',
        't2m', 'u10', 'v10', 'msl'
    ]

    # Get the indices of these variables in the variables list
    var_indices = [variables.index(var) for var in target_vars if var in variables]

    # Metrics to plot
    metrics = {
        'RMSE': rmse,
        'ACC': acc,
        'Activity': activity
    }

    for metric_name, metric_data in metrics.items():
        # Select only the data for our target variables
        selected_data = metric_data[:, var_indices]

        # Create figure with 4x5 grid of subplots
        fig, axes = plt.subplots(4, 5, figsize=figsize, dpi=dpi)
        fig.suptitle(f"{title} - {metric_name}", fontsize=16)

        # Flatten axes array for easy iteration
        axes = axes.ravel()

        # Create x-axis (lead times in hours)
        lead_times = np.arange(0, selected_data.shape[0] * 6, 6)

        # Plot each variable in its own subplot
        for i, (var, ax) in enumerate(zip(target_vars, axes)):
            if var in variables:  # Only plot if variable exists in data
                ax.plot(lead_times, selected_data[:, i], label=var, marker='o', markersize=4)
                ax.set_xlabel('Lead Time (hours)', fontsize=10)
                ax.set_ylabel(metric_name, fontsize=10)
                ax.set_title(var, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            else:
                ax.axis('off')  # Hide subplot if variable not in data

        # Adjust layout
        plt.tight_layout()
        figures.append(fig)

    return figures

def save_forecast_plots(
    figures: List[plt.Figure],
    output_dir: str,
    prefix: str = ["rmse", "acc", "activity"]
) -> None:
    """
    Save the forecast metric plots to files.

    Args:
        figures: List of matplotlib figures to save
        output_dir: Directory to save the plots
        prefix: Prefix for output filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, fig in enumerate(figures):
        filename = f"{prefix[i]}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=fig.dpi)
        plt.close(fig)

def plot_obsop_omb(
    tgt_tmbrs_values: np.ndarray,
    out_tmbrs_values: np.ndarray,
    mask: np.ndarray,
    variable_name: str,
    plot_dir: str,
) -> None:
    """
    Plot histogram of differences (prepbufr - ERA5) multiplied by mask with probability density.
    Args:
    obs_data: 1D numpy array containing original observations
    era5_data: 1D numpy array containing ERA5 data
    mask: 1D numpy array containing mask (0 or 1)
    variable_name: Name of the variable being compared
    Raises:
    ValueError: If input arrays have different lengths
    """
    if len(tgt_tmbrs_values) != len(out_tmbrs_values) or len(tgt_tmbrs_values) != len(mask):
        raise ValueError("All input arrays must have the same length")

    # Calculate differences multiplied by mask
    original_differences = (tgt_tmbrs_values - out_tmbrs_values) * mask

    # Filter out NaN values (where mask is 0)
    valid_mask = mask == 1
    diff_values = original_differences[valid_mask]

    if len(original_differences) == 0:
        logging.info(f"No valid data points for {variable_name}")
        return

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create histogram with probability density
    # sns.histplot(diff_values, bins=50, kde=True, color='blue', alpha=0.3, stat='density')
    plt.hist(diff_values, bins=50, density=True, alpha=0.3, color='blue', label='OMB')

    # 👇 手动添加 KDE 曲线 (使用 scipy，完全不依赖 pandas/seaborn)
    # 过滤掉 inf/nan 防止 scipy 报错
    finite_values = diff_values[np.isfinite(diff_values)]
    if len(finite_values) > 1:
        kde = stats.gaussian_kde(finite_values)
        x_range = np.linspace(np.min(finite_values), np.max(finite_values), 200)
        plt.plot(x_range, kde(x_range), color='blue', linewidth=2)

    # Add vertical line at zero (no difference)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')

    # Add labels and title
    plt.xlabel(f'OMB/Error ({variable_name})')
    plt.ylabel('Probability Density')
    plt.legend(loc='best')

    # Add statistics
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    plt.text(
        0.02, 0.95, 
        f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}\n',
        transform=plt.gca().transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/obsop_omb_{variable_name}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{plot_dir}/obsop_omb_{variable_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()