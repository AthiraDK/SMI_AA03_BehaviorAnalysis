from typing import Dict, List, Optional, Tuple
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from behavior_fw.visualization.plot_density import _get_coords


def compute_polar_occupancy(
    df_list: List[pd.DataFrame],
    edges_r: np.ndarray,
    edges_theta: np.ndarray,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[np.ndarray, float]:
    """Computes % Time in Bin per Fly in polar coordinates (r, theta) for a group of dataframes."""
    n_flies = len(df_list)
    if n_flies == 0:
        return np.zeros((len(edges_r) - 1, len(edges_theta) - 1)), 0.0

    total_counts = np.zeros((len(edges_r) - 1, len(edges_theta) - 1))
    total_epoch_sec = 0.0

    for df_epoch in df_list:
        if len(df_epoch) > 0:
            x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
            
            # Convert Cartesian (x, y) -> Polar (r, theta)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

            counts, _, _ = np.histogram2d(r, theta, bins=[edges_r, edges_theta])
            total_counts += counts
            total_epoch_sec += len(df_epoch) / fps

    # Duration normalization per fly
    mean_epoch_sec = total_epoch_sec / n_flies if n_flies > 0 else 1.0
    total_sec_in_bin = total_counts / fps
    mean_sec_per_fly = total_sec_in_bin / n_flies

    # % Time Spent in Bin relative to that epoch's mean length
    percent_occupancy = (mean_sec_per_fly / mean_epoch_sec) * 100.0

    return percent_occupancy, mean_epoch_sec


def plot_compare_four_conditions_polar_densities(
    datasets: List[List[Dict]],
    group_names: List[str],
    arena_r: float = 45.0,
    fps: float = 60.0,
    ten_min_frames: int = 36000,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    gamma: float = 0.3,
    x_col: str = "X",
    y_col: str = "Y",
    cmap: str = "rocket",  # Or 'magma' if Seaborn is not imported
    title: str = "Polar Spatial Occupancy Comparison Across Conditions & Epochs",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots a 4x3 Polar grid (4 Conditions x 3 Epochs).

    Features:
    - Independent per-plot normalization (vmax calculated per epoch panel)
    - Polar (r, theta) occupancy mesh
    - Subtle thin dotted white line for the reward zone overlay
    """
    if len(datasets) != 4 or len(group_names) != 4:
        raise ValueError("Expected exactly 4 condition datasets and 4 group names.")

    # Try setting Seaborn colormap if 'rocket' is requested
    try:
        if cmap == "rocket":
            import seaborn as sns
    except ImportError:
        cmap = "magma"  # Native Matplotlib fallback

    epoch_keys = ["Pre-1st Reward", "Post-1st Reward (Active)", "Washout (>10m)"]
    edges_r = np.linspace(0, arena_r, n_bins_r + 1)
    edges_theta = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    Theta, R = np.meshgrid(edges_theta, edges_r)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14),
        dpi=300,
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )

    for row_idx, (group_ds, g_name) in enumerate(zip(datasets, group_names)):
        # Extract reward zone metadata from reference fly
        ref_ds = group_ds[0] if group_ds else {}
        zx = ref_ds.get("zone_X", 0.0)
        zy = ref_ds.get("zone_Y", 0.0)
        zr = ref_ds.get("zone_r", 3.0)

        # Convert zone circle into polar parameters for overlay
        t_param = np.linspace(0, 2 * np.pi, 200)
        circle_x = zx + zr * np.cos(t_param)
        circle_y = zy + zr * np.sin(t_param)
        circle_r = np.sqrt(circle_x**2 + circle_y**2)
        circle_theta = np.arctan2(circle_y, circle_x)

        # Separate epoch dataframes across flies in group
        p1_list, p2_list, p3_list = [], [], []
        for ds in group_ds:
            df_track = ds["df_track"]
            total_frames = len(df_track)
            f_idx = ds.get("first_reward_idx", None) or total_frames

            p1_list.append(df_track.iloc[0:f_idx])
            p2_list.append(df_track.iloc[f_idx : min(ten_min_frames, total_frames)])
            if total_frames > ten_min_frames:
                p3_list.append(df_track.iloc[ten_min_frames:total_frames])

        epoch_lists = [p1_list, p2_list, p3_list]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            density_map, mean_dur = compute_polar_occupancy(
                epoch_lists[col_idx], edges_r, edges_theta, fps=fps, x_col=x_col, y_col=y_col
            )

            # Local per-plot normalization max
            local_max = density_map.max() if density_map.max() > 0 else 1.0
            norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=local_max)

            # Render polar mesh density
            mesh = ax.pcolormesh(
                Theta,
                R,
                density_map,
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )

            # Overlay thin dotted white reward zone border
            ax.plot(
                circle_theta,
                circle_r,
                color="white",
                linestyle=":",
                linewidth=0.8,
                alpha=0.85,
                zorder=10,
            )

            # Polar grid aesthetics
            ax.set_yticklabels([])  # Hide radial distance ticks
            ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            ax.set_xticklabels(
                ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"],
                fontsize=6,
                color="#666666",
            )
            ax.grid(True, linestyle="--", alpha=0.25, color="gray")

            # Local per-panel colorbar showing max density for this specific plot
            cbar = fig.colorbar(mesh, ax=ax, shrink=0.5, pad=0.08)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("% Time in Bin", fontsize=7)

            if row_idx == 0:
                ax.set_title(
                    f"{epoch_keys[col_idx]}\n(Avg: {mean_dur:.1f}s)",
                    fontsize=10,
                    fontweight="bold",
                    pad=12,
                )

        # Row Group Labels on the Left
        axes[row_idx, 0].set_ylabel(
            f"{g_name}\n(N={len(group_ds)})",
            fontsize=10,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=25,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig, axes