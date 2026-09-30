# behavior_fw/visualization/psth_plots.py

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from behavior_fw.utils.psth import extract_experiment_psth


def plot_single_experiment_psth(
    psth_dict: dict, metric: str = "Vf_smooth", title: str = ""
):
    """Plots trial-by-trial heatmap and mean +/- SEM PSTH for a single fly."""
    time_vec = psth_dict["time_vec"]
    matrix = psth_dict[metric]

    if matrix.shape[0] == 0:
        print(f"[{title}] No valid reward events for {metric}.")
        return None, None

    fig, (ax_heat, ax_mean) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        dpi=300,
        layout="constrained",
    )

    # 1. Trial Heatmap
    im = ax_heat.imshow(
        matrix,
        aspect="auto",
        extent=[time_vec[0], time_vec[-1], matrix.shape[0] + 0.5, 0.5],
        cmap="viridis",
        origin="upper",
    )
    ax_heat.axvline(0, color="red", linestyle="--", alpha=0.8, label="Reward Onset")
    ax_heat.set_ylabel("Trial / Encounter #", fontsize=10)
    ax_heat.set_title(
        f"Reward PSTH - {metric.replace('_smooth', '')} ({title})", fontsize=12
    )
    fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02, label="Velocity")

    # 2. Mean +/- SEM Response
    mean_trace = np.nanmean(matrix, axis=0)
    sem_trace = np.nanstd(matrix, axis=0) / np.sqrt(matrix.shape[0])

    ax_mean.plot(time_vec, mean_trace, color="#1f77b4", linewidth=2, label="Mean")
    ax_mean.fill_between(
        time_vec,
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        color="#1f77b4",
        alpha=0.3,
        label="SEM",
    )
    ax_mean.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax_mean.set_xlabel("Time from Reward Onset (s)", fontsize=10)
    ax_mean.set_ylabel("Mean Velocity", fontsize=10)
    ax_mean.grid(True, linestyle="--", alpha=0.4)

    return fig, (ax_heat, ax_mean)


def plot_condition_psth_summary(
    dataset: list,
    condition_name: str = "Gr43a_ATR",
    metrics: list = ["Vf_smooth", "Vr_smooth"],
    window_sec: tuple = (-2.0, 4.0),
):
    """Pools PSTH traces across all experiments in a condition and plots mean +/- SEM."""
    pooled_data = {m: [] for m in metrics}
    time_vec = None

    # Gather data across experiments
    for exp in dataset:
        psth_exp = extract_experiment_psth(
            exp, metrics=metrics, window_sec=window_sec
        )
        time_vec = psth_exp["time_vec"]

        for m in metrics:
            if psth_exp[m].shape[0] > 0:
                pooled_data[m].append(psth_exp[m])

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(8, 3.5 * len(metrics)),
        sharex=True,
        dpi=300,
        layout="constrained",
    )
    if len(metrics) == 1:
        axes = [axes]

    for i, m in enumerate(metrics):
        ax = axes[i]
        if not pooled_data[m]:
            continue

        # Stack all trials across all flies: (N_total_trials, N_frames)
        all_trials = np.vstack(pooled_data[m])
        n_trials = all_trials.shape[0]
        n_flies = len(pooled_data[m])

        mean_trace = np.nanmean(all_trials, axis=0)
        sem_trace = np.nanstd(all_trials, axis=0) / np.sqrt(n_trials)

        ax.plot(
            time_vec,
            mean_trace,
            color="#2ca02c" if "Vr" in m else "#1f77b4",
            linewidth=2,
            label=f"Mean (N={n_flies} flies, n={n_trials} trials)",
        )
        ax.fill_between(
            time_vec,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color="#2ca02c" if "Vr" in m else "#1f77b4",
            alpha=0.25,
        )

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Reward Onset")
        ax.set_ylabel(m.replace("_smooth", ""), fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time relative to Reward Onset (s)", fontsize=11)
    fig.suptitle(
        f"Population Reward-Triggered PSTH — {condition_name}", fontsize=14
    )

    return fig, axes



def plot_condition_psth_summary(
    datasets: List[List[Dict]],
    group_names: List[str],
    metrics: Optional[List[str]] = None,
    window_sec: Tuple[float, float] = (-2.0, 4.0),
    fps: float = 60.0,
    colors: Optional[List[str]] = None,
    title: str = "PSTH Summary",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots PSTH summary subplots for multiple groups side-by-side with shared axes.

    Parameters
    ----------
    datasets : List[List[Dict]]
        List of dataset lists, e.g. [dataset_gr5a_atr, dataset_gr5a_nr, dataset_gr43a_atr].
    group_names : List[str]
        List of condition/genotype names corresponding to datasets, e.g. ["Gr5a_ATR", "Gr5a_NR", "Gr43a_ATR"].
    metrics : Optional[List[str]]
        List of tracking metrics to plot per row (default: ["Vf_smooth", "Vr_abs"]).
    window_sec : Tuple[float, float]
        Time window around stimulus/1st reward onset in seconds (pre, post).
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

    # Calculate frame window
    start_frame = int(window_sec[0] * fps)
    end_frame = int(window_sec[1] * fps)
    time_vec = np.linspace(window_sec[0], window_sec[1], end_frame - start_frame)

    # Subplots sharing X-axis across all panels and Y-axis across each row
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        dpi=300,
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    # Ensure axes is 2D numpy array [nrows, ncols]
    if n_rows == 1 and n_cols == 1:
        axes_grid = np.array([[axes]])
    elif n_rows == 1:
        axes_grid = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes_grid = np.expand_dims(axes, axis=1)
    else:
        axes_grid = axes

    # Iterate through each group (columns) and metric (rows)
    for col_idx, (group_ds, group_label, color) in enumerate(zip(datasets, group_names, colors)):
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

            if psth_traces:
                psth_matrix = np.array(psth_traces)
                mean_trace = np.nanmean(psth_matrix, axis=0)
                sem_trace = np.nanstd(psth_matrix, axis=0) / np.sqrt(len(psth_traces))

                ax.plot(
                    time_vec,
                    mean_trace,
                    color=color,
                    linewidth=1.5,
                    label=f"n = {len(psth_traces)}",
                )
                ax.fill_between(
                    time_vec,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    color=color,
                    alpha=0.25,
                    edgecolor="none",
                )

            # Visual embellishments
            ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(True, linestyle=":", alpha=0.5)

            if row_idx == 0:
                ax.set_title(group_label, fontsize=11, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time from 1st Reward (s)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(metric, fontsize=10, fontweight="bold")

            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig, axes