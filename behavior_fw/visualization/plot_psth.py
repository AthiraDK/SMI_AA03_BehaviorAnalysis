# behavior_fw/visualization/psth_plots.py

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from behavior_fw.utils.psth import extract_experiment_psth


from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_single_experiment_psth(
    ds: Dict,
    metrics: Optional[List[str]] = None,
    window_sec: Tuple[float, float] = (-2.0, 4.0),
    stim_duration_sec: float = 1.0,  # 1-second LED pulse
    color: str = "#F16237",
    cmap: str = "magma",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots trial-by-trial heatmap and mean +/- SEM PSTH traces for a single experiment using extract_experiment_psth."""
    if metrics is None:
        metrics = ["Vf_smooth"]

    # 1. Extract full trial PSTH matrix for all metrics using helper function
    psth_dict = extract_experiment_psth(
        ds, metrics=metrics, window_sec=window_sec
    )

    time_vec = psth_dict["time_vec"]
    uid = ds.get("uid", "Single Fly")
    n_rewards = ds.get("n_rewards", 0)

    # Setup grid: 1 heatmap row + 1 row per metric trace
    n_rows = len(metrics) + 1
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(6.5, 2.8 * n_rows),
        dpi=300,
        sharex=True,
        layout="constrained",
    )

    if n_rows == 1:
        axes = np.array([axes])

    # -------------------------------------------------------------------------
    # 1. Top Panel: Trial-by-Trial Heatmap (First Metric)
    # -------------------------------------------------------------------------
    heatmap_metric = metrics[0]
    ax_heat = axes[0]
    matrix = psth_dict.get(heatmap_metric, np.array([]))

    if matrix.size > 0 and matrix.shape[0] > 0:
        n_trials = matrix.shape[0]
        im = ax_heat.imshow(
            matrix,
            aspect="auto",
            extent=[time_vec[0], time_vec[-1], n_trials + 0.5, 0.5],
            cmap=cmap,
            origin="upper",
            zorder=1,
        )
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.02)
        cbar.set_label(heatmap_metric, fontsize=8)
        ax_heat.set_ylabel("Trial #", fontsize=10, fontweight="bold")
    else:
        ax_heat.text(0.5, 0.5, "No Trial Data Available", ha="center", va="center")

    # Stimulus Onset (0s) and Offset (stim_duration_sec) lines & shading
    # ax_heat.axvspan(0, stim_duration_sec, color="#FFFFFF", alpha=0.15, zorder=2)
    ax_heat.axvline(0, color="#FFFFFF", linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)
    ax_heat.axvline(stim_duration_sec, color="#FFFFFF", linestyle=":", linewidth=1.0, alpha=0.8, zorder=3)
    ax_heat.set_title(f"Trial Heatmap ({heatmap_metric})", fontsize=10, fontweight="bold")

    # -------------------------------------------------------------------------
    # 2. Lower Panels: Mean +/- SEM Traces
    # -------------------------------------------------------------------------
    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx + 1]

        # Highlight stimulus pulse window
        ax.axvspan(0, stim_duration_sec, color="#FF0000", alpha=0.1, zorder=1)
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
        ax.axvline(stim_duration_sec, color="#222222", linestyle=":", linewidth=1.0, alpha=0.8, zorder=2)

        m_matrix = psth_dict.get(metric, np.array([]))
        if m_matrix.size > 0 and m_matrix.shape[0] > 0:
            mean_trace = np.nanmean(m_matrix, axis=0)
            sem_trace = np.nanstd(m_matrix, axis=0) / np.sqrt(m_matrix.shape[0])

            legend_label = f"Rewards: {n_rewards if n_rewards > 0 else m_matrix.shape[0]}"
            ax.plot(
                time_vec,
                mean_trace,
                color=color,
                linewidth=1.5,
                label=legend_label,
                zorder=3,
            )
            ax.fill_between(
                time_vec,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=0.25,
                edgecolor="none",
                zorder=3,
            )

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    axes[-1].set_xlabel("Time from Reward Onset (s)", fontsize=10)
    fig.suptitle(f"Single Fly PSTH & Heatmap: {uid}", fontsize=12, fontweight="bold")

    return fig, axes



def plot_condition_psth_summary(
    datasets: List[List[Dict]],
    group_names: List[str],
    metrics: Optional[List[str]] = None,
    window_sec: Tuple[float, float] = (-2.0, 3.0),
    stim_duration_sec: float = 1.0,  # 1-second LED duration
    fps: float = 60.0,
    colors: Optional[List[str]] = None,
    title: str = "Mean velocities aligned to reward onset",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots PSTH summary subplots for multiple groups side-by-side with shared axes.

    Marks both stimulus onset (t=0) and offset (t=stim_duration_sec).

    Parameters
    ----------
    datasets : List[List[Dict]]
        List of dataset lists, e.g. [dataset_gr5a_atr, dataset_gr5a_nr, dataset_gr43a_atr].
    group_names : List[str]
        List of condition/genotype names corresponding to datasets.
    metrics : Optional[List[str]]
        List of tracking metrics to plot per row (default: ["Vf_smooth", "Vr_abs"]).
    window_sec : Tuple[float, float]
        Time window around 1st reward onset in seconds (pre, post).
    stim_duration_sec : float
        Duration of the LED stimulus pulse in seconds (default: 1.0).
    fps : float
        Frame rate in Hz (default: 60.0).
    colors : Optional[List[str]]
        List of hex color codes matching each group.
    title : str
        Global figure title.
    """
    if metrics is None:
        metrics = ["Vf_smooth", "Vr_abs"]
        

    if len(datasets) != len(group_names):
        raise ValueError("`datasets` and `group_names` must have the same length.")

    n_rows = len(metrics)
    n_cols = len(group_names)

    if colors is None:
        default_palette = ["#F16237", "#BA90EB", "#009E73", "#2B5C8F", "#E69F00"]
        colors = [default_palette[i % len(default_palette)] for i in range(n_cols)]

    start_frame = int(window_sec[0] * fps)
    end_frame = int(window_sec[1] * fps)
    time_vec = np.linspace(window_sec[0], window_sec[1], end_frame - start_frame)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        dpi=300,
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    if n_rows == 1 and n_cols == 1:
        axes_grid = np.array([[axes]])
    elif n_rows == 1:
        axes_grid = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes_grid = np.expand_dims(axes, axis=1)
    else:
        axes_grid = axes

    for col_idx, (group_ds, group_label, color) in enumerate(zip(datasets, group_names, colors)):
        total_rewards = sum(ds.get("n_rewards", 0) for ds in group_ds)

        for row_idx, metric in enumerate(metrics):
            ax = axes_grid[row_idx, col_idx]
            psth_traces = []

            for ds in group_ds:
                df_track = ds["df_track"]
                f_idx = ds.get("first_reward_idx", None)

                if f_idx is None or metric not in df_track.columns:
                    continue

                idx_start = f_idx + start_frame
                idx_end = f_idx + end_frame

                if idx_start >= 0 and idx_end <= len(df_track):
                    psth_traces.append(df_track[metric].values[idx_start:idx_end])

            # 1. Highlight the active stimulus pulse window (0s to 1s)
            ax.axvspan(0, stim_duration_sec, color="#FF0000", alpha=0.1, zorder=1)

            # 2. Add onset (t=0) and offset (t=stim_duration_sec) lines
            ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
            ax.axvline(stim_duration_sec, color="#222222", linestyle=":", linewidth=1.0, alpha=0.8, zorder=2)

            if psth_traces:
                psth_matrix = np.array(psth_traces)
                mean_trace = np.nanmean(psth_matrix, axis=0)
                sem_trace = np.nanstd(psth_matrix, axis=0) / np.sqrt(len(psth_traces))

                legend_label = f"n = {len(psth_traces)} flies\nn_rewards = {total_rewards}"

                ax.plot(
                    time_vec,
                    mean_trace,
                    color=color,
                    linewidth=1.5,
                    label=legend_label,
                    zorder=3,
                )
                ax.fill_between(
                    time_vec,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    color=color,
                    alpha=0.25,
                    edgecolor="none",
                    zorder=3,
                )

            ax.grid(True, linestyle=":", alpha=0.5)

            if row_idx == 0:
                ax.set_title(group_label, fontsize=11, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time from 1st Reward (s)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Mean {metric}", fontsize=10, fontweight="bold")

            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig, axes

