# behavior_fw/visualization/psth_plots.py

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