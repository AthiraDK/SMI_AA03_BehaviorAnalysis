from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import seaborn as sns

# Import the existing 2D histogram helper functions from spatial_density.py
from behavior_fw.analysis.spatial_density import (
    compute_2d_cartesian_histogram,
    compute_2d_polar_histogram,
)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _get_coords(
    df: pd.DataFrame, x_col: str = "X", y_col: str = "Y"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts X and Y coordinates with fallback column handling."""
    x_name = x_col if x_col in df.columns else ("X" if "X" in df.columns else "x_mm")
    y_name = y_col if y_col in df.columns else ("Y" if "Y" in df.columns else "y_mm")
    
    if x_name not in df.columns or y_name not in df.columns:
        raise KeyError(f"Could not find coordinate columns ({x_col}, {y_col}) in DataFrame.")

    return df[x_name].values, df[y_name].values


def _split_into_epochs(ds: Dict) -> Dict[str, pd.DataFrame]:
    """Splits trajectory into Baseline, Active Reward (1st onset to last offset), and Washout."""
    df_track = ds["df_track"]
    df_blocks = ds.get("df_blocks", None)

    f_idx = ds.get("first_reward_idx", None)
    l_idx = ds.get("last_reward_idx", None)

    # Use stimulus blocks dataframe if available for exact timeticks
    if df_blocks is not None and len(df_blocks) > 0:
        first_onset_tick = df_blocks.iloc[0]["onset_timetick"]
        last_offset_tick = df_blocks.iloc[-1].get(
            "offset_timetick", df_blocks.iloc[-1]["onset_timetick"]
        )

        f_match = df_track[df_track["timetick"] == first_onset_tick] if "timetick" in df_track.columns else pd.DataFrame()
        l_match = df_track[df_track["timetick"] == last_offset_tick] if "timetick" in df_track.columns else pd.DataFrame()

        if not f_match.empty:
            f_idx = f_match.index[0]
        if not l_match.empty:
            l_idx = l_match.index[-1] + 1  # Include the full last reward frame offset

    total_frames = len(df_track)
    if f_idx is None:
        f_idx = total_frames
    if l_idx is None or l_idx < f_idx:
        l_idx = f_idx

    df_p1 = df_track.iloc[0:f_idx].copy()
    df_p2 = df_track.iloc[f_idx:l_idx].copy()
    df_p3 = df_track.iloc[l_idx:total_frames].copy() if l_idx < total_frames else pd.DataFrame()

    return {
        "Pre-1st Reward": df_p1,
        "Active Reward (1st to Last Offset)": df_p2,
        "Washout": df_p3,
    }


def compute_masked_cartesian_density(
    df_list: List[pd.DataFrame],
    bin_edges: np.ndarray,
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[np.ndarray, float]:
    """Computes % Mean Time Fraction per Fly in Cartesian bins with circular masking."""
    n_flies = len(df_list)
    n_bins = len(bin_edges) - 1
    if n_flies == 0:
        return np.full((n_bins, n_bins), np.nan), 0.0

    total_counts = np.zeros((n_bins, n_bins))
    total_epoch_sec = 0.0

    for df_epoch in df_list:
        if len(df_epoch) > 0:
            x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
            counts, _, _ = compute_2d_cartesian_histogram(x, y, n_bins=n_bins, arena_r=arena_r)
            total_counts += counts
            total_epoch_sec += len(df_epoch) / fps

    mean_epoch_sec_per_fly = total_epoch_sec / n_flies if n_flies > 0 else 1.0
    total_sec_in_bin = total_counts / fps
    mean_sec_per_fly = total_sec_in_bin / n_flies

    percent_occupancy = (mean_sec_per_fly / mean_epoch_sec_per_fly) * 100.0 if mean_epoch_sec_per_fly > 0 else np.zeros((n_bins, n_bins))

    # Circular Arena Masking
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    X_grid, Y_grid = np.meshgrid(bin_centers, bin_centers)
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    percent_occupancy[R_grid > arena_r] = np.nan

    return percent_occupancy, mean_epoch_sec_per_fly


def compute_polar_occupancy(
    df_list: List[pd.DataFrame],
    edges_r: np.ndarray,
    edges_theta: np.ndarray,
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[np.ndarray, float]:
    """Computes % Time Fraction per Fly in polar coordinates using compute_2d_polar_histogram."""
    n_flies = len(df_list)
    n_bins_r = len(edges_r) - 1
    n_bins_theta = len(edges_theta) - 1
    if n_flies == 0:
        return np.zeros((n_bins_r, n_bins_theta)), 0.0

    total_counts = np.zeros((n_bins_r, n_bins_theta))
    total_epoch_sec = 0.0

    for df_epoch in df_list:
        if len(df_epoch) > 0:
            x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
            r = np.hypot(x, y)
            theta = np.arctan2(y, x)

            counts, _, _ = compute_2d_polar_histogram(
                r, theta, n_bins_r=n_bins_r, n_bins_theta=n_bins_theta, arena_r=arena_r
            )
            total_counts += counts
            total_epoch_sec += len(df_epoch) / fps

    mean_epoch_sec = total_epoch_sec / n_flies if n_flies > 0 else 1.0
    total_sec_in_bin = total_counts / fps
    mean_sec_per_fly = total_sec_in_bin / n_flies

    percent_occupancy = (mean_sec_per_fly / mean_epoch_sec) * 100.0 if mean_epoch_sec > 0 else np.zeros((n_bins_r, n_bins_theta))
    return percent_occupancy, mean_epoch_sec


# =========================================================================
# 1. CARTESIAN DENSITY FUNCTIONS
# =========================================================================

def plot_single_experiment_cartesian_density(
    dataset: List[Dict],
    exp_idx: int,
    dataset_name: str = "Dataset",
    arena_r: float = 45.0,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, np.ndarray]:
    """1.a) Plots Cartesian density for a single experiment chosen by index using pcolormesh."""
    ds = dataset[exp_idx]
    epochs_dict = _split_into_epochs(ds)
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)
    X_mesh, Y_mesh = np.meshgrid(bin_edges, bin_edges)
    
    histograms = {}
    global_max = 0.0
    
    for epoch_name, df_epoch in epochs_dict.items():
        if len(df_epoch) > 0:
            hist, _ = compute_masked_cartesian_density([df_epoch], bin_edges, arena_r, x_col=x_col, y_col=y_col)
            histograms[epoch_name] = hist
            max_val = np.nanmax(hist)
            if max_val > global_max:
                global_max = max_val
        else:
            histograms[epoch_name] = np.full((n_bins, n_bins), np.nan)

    if global_max == 0:
        global_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, layout="constrained")
    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad("none")
    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

    zx, zy, zr = ds.get("zone_X", 0.0), ds.get("zone_Y", 0.0), ds.get("zone_r", 3.0)

    for i, (epoch_name, density_map) in enumerate(histograms.items()):
        ax = axes[i]
        zone = patches.Circle((0, 0), arena_r, transform=ax.transData, fill=False, ec="#EEE6E6")
        ax.add_patch(zone)

        im = ax.pcolormesh(
            X_mesh, Y_mesh, density_map, cmap=cmap, norm=norm, shading="flat", rasterized=True
        )
        im.set_clip_path(zone)
        ax.add_patch(patches.Circle((0, 0), arena_r, color="#444444", fill=False, linestyle="--"))
        ax.add_patch(patches.Circle((zx, zy), zr, edgecolor="white", facecolor="none", linewidth=0.8, linestyle=":", alpha=0.8, zorder=10))
        
        dur_sec = len(epochs_dict[epoch_name]) / 60.0
        ax.set_title(f"{epoch_name}\n({dur_sec:.1f}s)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("% Time in Bin", fontsize=10, fontweight="bold")
    
    fly_id = ds.get("uid", f"Fly_{exp_idx}")
    fig.suptitle(f"{dataset_name} | Fly ID: {fly_id}", fontsize=12, fontweight="bold")
    return fig, axes


def plot_dataset_average_cartesian_density(
    dataset: List[Dict],
    dataset_name: str = "Dataset",
    arena_r: float = 45.0,
    fps: float = 60.0,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, np.ndarray]:
    """1.b) Plots average Cartesian density across all flies using pcolormesh."""
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)
    X_mesh, Y_mesh = np.meshgrid(bin_edges, bin_edges)
    epoch_keys = ["Pre-1st Reward", "Active Reward (1st to Last Offset)", "Washout"]
    epoch_lists: Dict[str, List[pd.DataFrame]] = {k: [] for k in epoch_keys}

    for ds in dataset:
        ep = _split_into_epochs(ds)
        for k in epoch_keys:
            if len(ep[k]) > 0:
                epoch_lists[k].append(ep[k])

    grid_histograms = {}
    global_max = 0.0
    durations = {}

    for epoch_name, df_list in epoch_lists.items():
        hist, mean_dur = compute_masked_cartesian_density(df_list, bin_edges, arena_r, fps=fps, x_col=x_col, y_col=y_col)
        grid_histograms[epoch_name] = hist
        durations[epoch_name] = mean_dur
        max_val = np.nanmax(hist) if not np.all(np.isnan(hist)) else 0.0
        if max_val > global_max:
            global_max = max_val

    if global_max == 0:
        global_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, layout="constrained")
    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad("none")
    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

    ref_ds = dataset[0] if dataset else {}
    zx, zy, zr = ref_ds.get("zone_X", 0.0), ref_ds.get("zone_Y", 0.0), ref_ds.get("zone_r", 3.0)

    for i, (epoch_name, density_map) in enumerate(grid_histograms.items()):
        ax = axes[i]
        zone = patches.Circle((0, 0), arena_r, transform=ax.transData, fill=False, ec="#EEE6E6")
        ax.add_patch(zone)

        im = ax.pcolormesh(
            X_mesh, Y_mesh, density_map, cmap=cmap, norm=norm, shading="flat", rasterized=True
        )
        im.set_clip_path(zone)
        ax.add_patch(patches.Circle((0, 0), arena_r, color="#444444", fill=False, linestyle="--"))
        ax.add_patch(patches.Circle((zx, zy), zr, edgecolor="white", facecolor="none", linewidth=0.8, linestyle=":", alpha=0.8, zorder=10))
        ax.set_title(f"{epoch_name}\n(Avg: {durations[epoch_name]:.1f}s)", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("% Time in Bin per Fly", fontsize=10, fontweight="bold")
    fig.suptitle(f"{dataset_name} (Average across N = {len(dataset)} flies)", fontsize=12, fontweight="bold")
    return fig, axes


def plot_compare_conditions_cartesian(
    datasets: List[List[Dict]],
    group_names: List[str],
    normalize_across_datasets: bool = False,
    arena_r: float = 45.0,
    fps: float = 60.0,
    n_bins: int = 100,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
    title: str = "Cartesian Density Comparison Across Conditions",
) -> Tuple[plt.Figure, np.ndarray]:
    """1.c) Plots Cartesian density across multiple conditions using pcolormesh."""
    n_groups = len(datasets)
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)
    X_mesh, Y_mesh = np.meshgrid(bin_edges, bin_edges)
    epoch_keys = ["Pre-1st Reward", "Active Reward (1st to Last Offset)", "Washout"]

    grid_histograms, grid_durations = [], []
    global_max_density = 0.0
    row_max_densities = []

    for group_ds in datasets:
        p_lists = [[], [], []]
        for ds in group_ds:
            ep = _split_into_epochs(ds)
            for col_idx, k in enumerate(epoch_keys):
                if len(ep[k]) > 0:
                    p_lists[col_idx].append(ep[k])

        row_hists, row_durs = [], []
        row_max = 0.0
        for df_list in p_lists:
            hist, mean_dur = compute_masked_cartesian_density(df_list, bin_edges, arena_r, fps=fps, x_col=x_col, y_col=y_col)
            row_hists.append(hist)
            row_durs.append(mean_dur)
            max_val = np.nanmax(hist) if not np.all(np.isnan(hist)) else 0.0
            if max_val > global_max_density:
                global_max_density = max_val
            if max_val > row_max:
                row_max = max_val

        grid_histograms.append(row_hists)
        grid_durations.append(row_durs)
        row_max_densities.append(max_val if row_max == 0 else row_max)

    fig, axes = plt.subplots(nrows=n_groups, ncols=3, figsize=(12, 3.2 * n_groups), dpi=300, layout="constrained")
    if n_groups == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.get_cmap("rocket").copy()
    cmap.set_bad("none")
    global_norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max_density or 1.0)

    last_im = None
    for row_idx, (group_ds, g_name) in enumerate(zip(datasets, group_names)):
        ref_ds = group_ds[0] if group_ds else {}
        zx, zy, zr = ref_ds.get("zone_X", 0.0), ref_ds.get("zone_Y", 0.0), ref_ds.get("zone_r", 3.0)
        
        row_norm = global_norm if normalize_across_datasets else colors.PowerNorm(gamma=gamma, vmin=0, vmax=row_max_densities[row_idx])

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            density_map = grid_histograms[row_idx][col_idx]
            dur = grid_durations[row_idx][col_idx]

            zone = patches.Circle((0, 0), arena_r, transform=ax.transData, fill=False, ec="#EEE6E6")
            ax.add_patch(zone)
            im = ax.pcolormesh(
                X_mesh, Y_mesh, density_map, cmap=cmap, norm=row_norm, shading="flat", rasterized=True
            )
            im.set_clip_path(zone)
            ax.add_patch(patches.Circle((0, 0), arena_r, color="#444444", fill=False, linestyle="--"))
            ax.add_patch(patches.Circle((zx, zy), zr, edgecolor="white", facecolor="none", linewidth=0.8, linestyle=":", alpha=0.8, zorder=10))
            
            ax.set_aspect("equal")
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(f"{epoch_keys[col_idx]}\n(Avg: {dur:.1f}s)", fontsize=10, fontweight="bold")

        axes[row_idx, 0].set_ylabel(f"{g_name}\n(N={len(group_ds)})", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("% Time in Bin per Fly", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    return fig, axes


# =========================================================================
# 2. POLAR DENSITY FUNCTIONS
# =========================================================================

def plot_single_experiment_polar_density(
    dataset: List[Dict],
    exp_idx: int,
    dataset_name: str = "Dataset",
    arena_r: float = 45.0,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, np.ndarray]:
    """2.a) Single experiment polar density using pcolormesh."""
    ds = dataset[exp_idx]
    epochs_dict = _split_into_epochs(ds)
    edges_r = np.linspace(0, arena_r, n_bins_r + 1)
    edges_theta = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    Theta, R = np.meshgrid(edges_theta, edges_r)

    histograms = {}
    global_max = 0.0

    for epoch_name, df_epoch in epochs_dict.items():
        if len(df_epoch) > 0:
            hist, _ = compute_polar_occupancy([df_epoch], edges_r, edges_theta, arena_r=arena_r, x_col=x_col, y_col=y_col)
            histograms[epoch_name] = hist
            if hist.max() > global_max:
                global_max = hist.max()
        else:
            histograms[epoch_name] = np.zeros((n_bins_r, n_bins_theta))

    if global_max == 0:
        global_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, subplot_kw={"projection": "polar"}, layout="constrained")
    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

    zx, zy, zr = ds.get("zone_X", 0.0), ds.get("zone_Y", 0.0), ds.get("zone_r", 3.0)
    t_param = np.linspace(0, 2 * np.pi, 200)
    circle_x, circle_y = zx + zr * np.cos(t_param), zy + zr * np.sin(t_param)
    circle_r, circle_theta = np.hypot(circle_x, circle_y), np.arctan2(circle_y, circle_x)

    for i, (epoch_name, density_map) in enumerate(histograms.items()):
        ax = axes[i]
        mesh = ax.pcolormesh(Theta, R, density_map.T, cmap="rocket", norm=norm, shading="flat", rasterized=True)
        ax.plot(circle_theta, circle_r, color="white", linestyle=":", linewidth=1.0, zorder=10)
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"], fontsize=6)
        dur_sec = len(epochs_dict[epoch_name]) / 60.0
        ax.set_title(f"{epoch_name}\n({dur_sec:.1f}s)", fontsize=10, fontweight="bold", pad=10)

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.75, pad=0.04)
    cbar.set_label("% Time in Bin", fontsize=10, fontweight="bold")
    
    fly_id = ds.get("uid", f"Fly_{exp_idx}")
    fig.suptitle(f"{dataset_name} | Polar Density | Fly ID: {fly_id}", fontsize=12, fontweight="bold")
    return fig, axes


def plot_dataset_average_polar_density(
    dataset: List[Dict],
    dataset_name: str = "Dataset",
    arena_r: float = 45.0,
    fps: float = 60.0,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, np.ndarray]:
    """2.b) Average Polar density across all flies using pcolormesh."""
    edges_r = np.linspace(0, arena_r, n_bins_r + 1)
    edges_theta = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    Theta, R = np.meshgrid(edges_theta, edges_r)

    epoch_keys = ["Pre-1st Reward", "Active Reward (1st to Last Offset)", "Washout"]
    epoch_lists: Dict[str, List[pd.DataFrame]] = {k: [] for k in epoch_keys}

    for ds in dataset:
        ep = _split_into_epochs(ds)
        for k in epoch_keys:
            if len(ep[k]) > 0:
                epoch_lists[k].append(ep[k])

    grid_histograms, durations = {}, {}
    global_max = 0.0

    for epoch_name, df_list in epoch_lists.items():
        hist, mean_dur = compute_polar_occupancy(df_list, edges_r, edges_theta, arena_r=arena_r, fps=fps, x_col=x_col, y_col=y_col)
        grid_histograms[epoch_name] = hist
        durations[epoch_name] = mean_dur
        if hist.max() > global_max:
            global_max = hist.max()

    if global_max == 0:
        global_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, subplot_kw={"projection": "polar"}, layout="constrained")
    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max)

    ref_ds = dataset[0] if dataset else {}
    zx, zy, zr = ref_ds.get("zone_X", 0.0), ref_ds.get("zone_Y", 0.0), ref_ds.get("zone_r", 3.0)
    t_param = np.linspace(0, 2 * np.pi, 200)
    circle_x, circle_y = zx + zr * np.cos(t_param), zy + zr * np.sin(t_param)
    circle_r, circle_theta = np.hypot(circle_x, circle_y), np.arctan2(circle_y, circle_x)

    for i, (epoch_name, density_map) in enumerate(grid_histograms.items()):
        ax = axes[i]
        mesh = ax.pcolormesh(Theta, R, density_map.T, cmap="rocket", norm=norm, shading="flat", rasterized=True)
        ax.plot(circle_theta, circle_r, color="white", linestyle=":", linewidth=1.0, zorder=10)
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"], fontsize=6)
        ax.set_title(f"{epoch_name}\n(Avg: {durations[epoch_name]:.1f}s)", fontsize=10, fontweight="bold", pad=10)

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.75, pad=0.04)
    cbar.set_label("% Time in Bin per Fly", fontsize=10, fontweight="bold")
    fig.suptitle(f"{dataset_name} | Polar Density (Average across N = {len(dataset)} flies)", fontsize=12, fontweight="bold")
    return fig, axes


def plot_compare_conditions_polar(
    datasets: List[List[Dict]],
    group_names: List[str],
    normalize_across_datasets: bool = False,
    arena_r: float = 45.0,
    fps: float = 60.0,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
    title: str = "Polar Spatial Occupancy Comparison Across Conditions",
) -> Tuple[plt.Figure, np.ndarray]:
    """2.c) Multi-condition Polar density comparison matrix using pcolormesh."""
    n_groups = len(datasets)
    edges_r = np.linspace(0, arena_r, n_bins_r + 1)
    edges_theta = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    Theta, R = np.meshgrid(edges_theta, edges_r)
    epoch_keys = ["Pre-1st Reward", "Active Reward (1st to Last Offset)", "Washout"]

    grid_histograms, grid_durations = [], []
    global_max_density = 0.0
    row_max_densities = []

    for group_ds in datasets:
        p_lists = [[], [], []]
        for ds in group_ds:
            ep = _split_into_epochs(ds)
            for col_idx, k in enumerate(epoch_keys):
                if len(ep[k]) > 0:
                    p_lists[col_idx].append(ep[k])

        row_hists, row_durs = [], []
        row_max = 0.0
        for df_list in p_lists:
            hist, mean_dur = compute_polar_occupancy(df_list, edges_r, edges_theta, arena_r=arena_r, fps=fps, x_col=x_col, y_col=y_col)
            row_hists.append(hist)
            row_durs.append(mean_dur)
            max_val = hist.max()
            if max_val > global_max_density:
                global_max_density = max_val
            if max_val > row_max:
                row_max = max_val

        grid_histograms.append(row_hists)
        grid_durations.append(row_durs)
        row_max_densities.append(row_max if row_max > 0 else 1.0)

    fig, axes = plt.subplots(
        nrows=n_groups, ncols=3, figsize=(12, 3.2 * n_groups), dpi=300,
        subplot_kw={"projection": "polar"}, layout="constrained"
    )
    if n_groups == 1:
        axes = np.expand_dims(axes, axis=0)

    global_norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=global_max_density or 1.0)
    last_mesh = None

    for row_idx, (group_ds, g_name) in enumerate(zip(datasets, group_names)):
        ref_ds = group_ds[0] if group_ds else {}
        zx, zy, zr = ref_ds.get("zone_X", 0.0), ref_ds.get("zone_Y", 0.0), ref_ds.get("zone_r", 3.0)

        t_param = np.linspace(0, 2 * np.pi, 200)
        circle_x, circle_y = zx + zr * np.cos(t_param), zy + zr * np.sin(t_param)
        circle_r, circle_theta = np.hypot(circle_x, circle_y), np.arctan2(circle_y, circle_x)

        row_norm = global_norm if normalize_across_datasets else colors.PowerNorm(gamma=gamma, vmin=0, vmax=row_max_densities[row_idx])

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            density_map = grid_histograms[row_idx][col_idx]
            dur = grid_durations[row_idx][col_idx]

            last_mesh = ax.pcolormesh(Theta, R, density_map.T, cmap="rocket", norm=row_norm, shading="flat", rasterized=True)
            ax.plot(circle_theta, circle_r, color="white", linestyle=":", linewidth=0.8, alpha=0.85, zorder=10)
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"], fontsize=6, color="#666666")
            ax.grid(True, linestyle="--", alpha=0.25, color="gray")

            if row_idx == 0:
                ax.set_title(f"{epoch_keys[col_idx]}\n(Avg: {dur:.1f}s)", fontsize=10, fontweight="bold", pad=12)

        axes[row_idx, 0].set_ylabel(f"{g_name}\n(N={len(group_ds)})", fontsize=10, fontweight="bold", rotation=0, ha="right", va="center", labelpad=25)

    cbar = fig.colorbar(last_mesh, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("% Time in Bin per Fly", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    return fig, axes