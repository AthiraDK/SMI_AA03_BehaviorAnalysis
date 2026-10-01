from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import seaborn as sns
from behavior_fw.visualization.plot_density import _get_coords
from behavior_fw.visualization.plot_density import plot_dynamic_cartesian_densities

def plot_single_experiment_epoch_densities(
    ds: Dict,
    arena_r: float = 45.0,
    fps: float = 60.0,
    ten_min_frames: int = 36000,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """Splits a single experiment's trajectory into 3 epochs and plots 2D spatial densities side-by-side."""
    df_track = ds["df_track"]
    total_frames = len(df_track)
    f_idx = ds.get("first_reward_idx", None)

    if f_idx is None:
        f_idx = total_frames

    # Slicing the dataframe into 3 epochs matching split trajectories
    df_p1 = df_track.iloc[0:f_idx]
    df_p2 = df_track.iloc[f_idx : min(ten_min_frames, total_frames)]
    df_p3 = (
        df_track.iloc[ten_min_frames:total_frames]
        if total_frames > ten_min_frames
        else pd.DataFrame()
    )

    epochs_dict = {
        "Pre-1st Reward": df_p1,
        "Post-1st Reward (Active)": df_p2,
        "Washout (>10m)": df_p3,
    }

    # Extract reward zone metadata
    reward_zone = {
        "x": ds.get("zone_X", 0.0),
        "y": ds.get("zone_Y", 0.0),
        "r": ds.get("zone_r", 3.0),
    }

    title = f"UID: {ds.get('uid', 'Single Fly')} | Rewards: {ds.get('n_rewards', 0)}"

    return plot_dynamic_cartesian_densities(
        epochs_dict=epochs_dict,
        arena_r=arena_r,
        n_bins=n_bins,
        gamma=gamma,
        title_prefix=title,
        x_col=x_col,
        y_col=y_col,
        reward_zone=reward_zone,
    )



def plot_condition_epoch_densities(
    dataset: List[Dict],
    condition_name: str = "Condition",
    arena_r: float = 45.0,
    ten_min_frames: int = 36000,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """Pools spatial trajectories across all flies in a condition group for each epoch and plots pooled densities."""
    p1_list, p2_list, p3_list = [], [], []

    for ds in dataset:
        df_track = ds["df_track"]
        total_frames = len(df_track)
        f_idx = ds.get("first_reward_idx", None)

        if f_idx is None:
            f_idx = total_frames

        p1_list.append(df_track.iloc[0:f_idx])
        p2_list.append(df_track.iloc[f_idx : min(ten_min_frames, total_frames)])
        if total_frames > ten_min_frames:
            p3_list.append(df_track.iloc[ten_min_frames:total_frames])

    # Concatenate frames across all flies per epoch
    epochs_dict = {
        "Pre-1st Reward": pd.concat(p1_list, ignore_index=True) if p1_list else pd.DataFrame(),
        "Post-1st Reward (Active)": pd.concat(p2_list, ignore_index=True) if p2_list else pd.DataFrame(),
        "Washout (>10m)": pd.concat(p3_list, ignore_index=True) if p3_list else pd.DataFrame(),
    }

    # Common reward zone for the group
    ref_ds = dataset[0] if dataset else {}
    reward_zone = {
        "x": ref_ds.get("zone_X", 0.0),
        "y": ref_ds.get("zone_Y", 0.0),
        "r": ref_ds.get("zone_r", 3.0),
    }

    title = f"{condition_name} (N = {len(dataset)} flies)"

    return plot_dynamic_cartesian_densities(
        epochs_dict=epochs_dict,
        arena_r=arena_r,
        n_bins=n_bins,
        gamma=gamma,
        title_prefix=title,
        x_col=x_col,
        y_col=y_col,
        reward_zone=reward_zone,
    )




def plot_compare_four_conditions_densities(
    datasets: List[List[Dict]],
    group_names: List[str],
    arena_r: float = 45.0,
    ten_min_frames: int = 36000,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
    title: str = "Condition Density Comparison Across Epochs",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots a 4x3 comparison grid: 4 Condition Groups (Rows) x 3 Epochs (Columns)

    with a single shared colorbar scale across all panels.
    """
    if len(datasets) != 4 or len(group_names) != 4:
        raise ValueError("Expected exactly 4 condition datasets and 4 group names.")

    epoch_keys = ["Pre-1st Reward", "Post-1st Reward (Active)", "Washout (>10m)"]
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)

    # 1. First Pass: Compute all 2D histograms and determine global maximum density
    grid_histograms = []
    global_max_density = 0.0

    for group_ds in datasets:
        p1_list, p2_list, p3_list = [], [], []
        for ds in group_ds:
            df_track = ds["df_track"]
            total_frames = len(df_track)
            f_idx = ds.get("first_reward_idx", None) or total_frames

            p1_list.append(df_track.iloc[0:f_idx])
            p2_list.append(df_track.iloc[f_idx : min(ten_min_frames, total_frames)])
            if total_frames > ten_min_frames:
                p3_list.append(df_track.iloc[ten_min_frames:total_frames])

        epochs_data = [
            pd.concat(p1_list, ignore_index=True) if p1_list else pd.DataFrame(),
            pd.concat(p2_list, ignore_index=True) if p2_list else pd.DataFrame(),
            pd.concat(p3_list, ignore_index=True) if p3_list else pd.DataFrame(),
        ]

        row_hists = []
        for df_epoch in epochs_data:
            if len(df_epoch) > 0:
                x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
                counts, _, _ = np.histogram2d(x, y, bins=[bin_edges, bin_edges])
                frac_density = (counts / len(df_epoch)) * 100.0
                hist_t = frac_density.T  # Alignment transpose
                row_hists.append(hist_t)
                if hist_t.max() > global_max_density:
                    global_max_density = hist_t.max()
            else:
                row_hists.append(np.zeros((n_bins, n_bins)))

        grid_histograms.append(row_hists)

    if global_max_density == 0:
        global_max_density = 1.0

    # 2. Second Pass: Render Subplots with Shared PowerNorm Scaling
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14),
        dpi=300,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max_density)
    last_im = None

    for row_idx, (group_ds, g_name) in enumerate(zip(datasets, group_names)):
        ref_ds = group_ds[0] if group_ds else {}
        zx = ref_ds.get("zone_X", 0.0)
        zy = ref_ds.get("zone_Y", 0.0)
        zr = ref_ds.get("zone_r", 3.0)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            density_map = grid_histograms[row_idx][col_idx]

            last_im = ax.imshow(
                density_map,
                origin="lower",
                extent=[-arena_r, arena_r, -arena_r, arena_r],
                cmap="rocket",
                norm=norm,
                interpolation="gaussian",
            )

            # Arena & Reward Zone Overlays
            arena_circle = patches.Circle(
                (0, 0), arena_r, color="white", fill=False, linestyle="--", linewidth=1.0, alpha=0.6
            )
            zone_circle = patches.Circle(
                (zx, zy), zr, edgecolor="white", facecolor="none", linestyle="-", linewidth=1.5, zorder=10
            )
            ax.add_patch(arena_circle)
            ax.add_patch(zone_circle)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Top Column Titles
            if row_idx == 0:
                ax.set_title(epoch_keys[col_idx], fontsize=11, fontweight="bold", pad=8)

        # Row Group Labels on Left Side
        axes[row_idx, 0].set_ylabel(
            f"{g_name}\n(N={len(group_ds)})",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=15,
        )

    # Global Shared Colorbar
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("% Time in Bin", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig, axes



def compute_masked_density(
    df_list: List[pd.DataFrame],
    bin_edges: np.ndarray,
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[np.ndarray, float]:
    """Computes % Mean Time Fraction per Fly in each spatial bin, with circular masking outside arena_r."""
    n_flies = len(df_list)
    if n_flies == 0:
        n_bins = len(bin_edges) - 1
        return np.full((n_bins, n_bins), np.nan), 0.0

    total_counts = np.zeros((len(bin_edges) - 1, len(bin_edges) - 1))
    total_epoch_sec = 0.0

    for df_epoch in df_list:
        if len(df_epoch) > 0:
            x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
            counts, _, _ = np.histogram2d(x, y, bins=[bin_edges, bin_edges])
            total_counts += counts
            total_epoch_sec += len(df_epoch) / fps

    # Average epoch length per fly in seconds
    mean_epoch_sec_per_fly = total_epoch_sec / n_flies if n_flies > 0 else 1.0

    # Total seconds spent in each bin across all flies
    total_sec_in_bin = total_counts / fps

    # Mean seconds spent per fly in each bin
    mean_sec_per_fly = total_sec_in_bin / n_flies

    # Normalized % Occupancy Time per Fly (Accounts for epoch duration)
    percent_occupancy = (mean_sec_per_fly / mean_epoch_sec_per_fly) * 100.0
    hist_t = percent_occupancy.T  # Transpose for imshow alignment

    # -------------------------------------------------------------------------
    # Apply Circular Arena Masking
    # -------------------------------------------------------------------------
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    X_grid, Y_grid = np.meshgrid(bin_centers, bin_centers)
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)

    # Set grid points outside arena radius to NaN
    hist_t[R_grid > arena_r] = np.nan

    return hist_t, mean_epoch_sec_per_fly


def plot_compare_four_conditions_densities_normalized(
    datasets: List[List[Dict]],
    group_names: List[str],
    arena_r: float = 45.0,
    fps: float = 60.0,
    ten_min_frames: int = 36000,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
    cmap: str = "magma",
    title: str = "Normalized Spatial Occupancy Across Conditions & Epochs",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots a 4x3 comparison grid: 4 Condition Groups x 3 Epochs with:

    1. Equalized % Time Fraction per Fly normalization (handles epoch & fly count differences)
    2. Circular masking clipping heatmap strictly inside arena radius
    """
    if len(datasets) != 4 or len(group_names) != 4:
        raise ValueError("Expected exactly 4 condition datasets and 4 group names.")

    epoch_keys = ["Pre-1st Reward", "Post-1st Reward (Active)", "Washout (>10m)"]
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)

    grid_histograms = []
    grid_durations = []
    global_max_density = 0.0

    # 1. Compute Normalized & Masked Histograms
    for group_ds in datasets:
        p1_list, p2_list, p3_list = [], [], []
        for ds in group_ds:
            df_track = ds["df_track"]
            total_frames = len(df_track)
            f_idx = ds.get("first_reward_idx", None) or total_frames

            p1_list.append(df_track.iloc[0:f_idx])
            p2_list.append(df_track.iloc[f_idx : min(ten_min_frames, total_frames)])
            if total_frames > ten_min_frames:
                p3_list.append(df_track.iloc[ten_min_frames:total_frames])

        row_hists = []
        row_durs = []
        for df_epoch_list in [p1_list, p2_list, p3_list]:
            hist_masked, mean_dur = compute_masked_density(
                df_epoch_list, bin_edges, arena_r=arena_r, fps=fps, x_col=x_col, y_col=y_col
            )
            row_hists.append(hist_masked)
            row_durs.append(mean_dur)

            # Ignore NaNs when finding global max density
            max_val = np.nanmax(hist_masked) if not np.all(np.isnan(hist_masked)) else 0.0
            if max_val > global_max_density:
                global_max_density = max_val

        grid_histograms.append(row_hists)
        grid_durations.append(row_durs)

    if global_max_density == 0:
        global_max_density = 1.0

    # Set up colormap and handle NaN (masked out area outside arena) background
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="none")  # Transparent background outside arena circle

    # 2. Render 4x3 Grid
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14),
        dpi=300,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max_density)
    last_im = None

    for row_idx, (group_ds, g_name) in enumerate(zip(datasets, group_names)):
        ref_ds = group_ds[0] if group_ds else {}
        zx = ref_ds.get("zone_X", 0.0)
        zy = ref_ds.get("zone_Y", 0.0)
        zr = ref_ds.get("zone_r", 3.0)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            density_map = grid_histograms[row_idx][col_idx]
            dur = grid_durations[row_idx][col_idx]

            last_im = ax.imshow(
                density_map,
                origin="lower",
                extent=[-arena_r, arena_r, -arena_r, arena_r],
                cmap=current_cmap,
                norm=norm,
                interpolation="gaussian",
            )

            # Arena & Reward Zone Borders
            arena_circle = patches.Circle(
                (0, 0), arena_r, color="#444444", fill=False, linestyle="--", linewidth=1.0, alpha=0.7
            )
            zone_circle = patches.Circle(
                (zx, zy), zr, edgecolor="white", facecolor="none", linestyle="-", linewidth=1.5, zorder=10
            )
            ax.add_patch(arena_circle)
            ax.add_patch(zone_circle)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    f"{epoch_keys[col_idx]}\n(Avg: {dur:.1f}s)",
                    fontsize=10,
                    fontweight="bold",
                    pad=8,
                )

        axes[row_idx, 0].set_ylabel(
            f"{g_name}\n(N={len(group_ds)})",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=15,
        )

    # Shared Colorbar
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("% Time in Bin per Fly", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig, axes