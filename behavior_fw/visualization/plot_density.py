# behavior_fw/visualization/density_plots.py

from typing import Any, Dict, Optional, Tuple
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def _get_coords(
    df: pd.DataFrame, x_col: str = "X", y_col: str = "Y"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts X and Y coordinates with fallback column handling."""
    if x_col in df.columns:
        x = df[x_col].values
    elif "X" in df.columns:
        x = df["X"].values
    elif "x_mm" in df.columns:
        x = df["x_mm"].values
    else:
        raise KeyError(f"Could not find X coordinate column in DataFrame.")

    if y_col in df.columns:
        y = df[y_col].values
    elif "Y" in df.columns:
        y = df["Y"].values
    elif "y_mm" in df.columns:
        y = df["y_mm"].values
    else:
        raise KeyError(f"Could not find Y coordinate column in DataFrame.")

    return x, y


def _parse_reward_zone(
    reward_zone: Optional[Dict[str, float]]
) -> Optional[Tuple[float, float, float]]:
    """Extracts (x, y, radius) from reward_zone dictionary."""
    if not reward_zone:
        return None
    zx = reward_zone.get("x_mm", reward_zone.get("x", 0.0))
    zy = reward_zone.get("y_mm", reward_zone.get("y", 0.0))
    zr = reward_zone.get("radius_mm", reward_zone.get("radius", reward_zone.get("r", 10.0)))
    return zx, zy, zr


# -------------------------------------------------------------------------
# 1. Cartesian Density Plotter
# -------------------------------------------------------------------------
def plot_dynamic_cartesian_densities(
    epochs_dict: Dict[str, pd.DataFrame],
    arena_r: float = 45.0,
    n_bins: int = 100,
    gamma: float = 0.3,
    title_prefix: str = "",
    x_col: str = "X",
    y_col: str = "Y",
    reward_zone: Optional[Dict[str, float]] = None,
    reward_zone_mm: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """Plots 2D Cartesian spatial density (% Time in Bin) across epochs with a

    white reward zone circle.
    """
    valid_epochs = {
        k: v for k, v in epochs_dict.items() if v is not None and len(v) > 0
    }
    if not valid_epochs:
        print("No valid epoch data to plot.")
        return None, None

    # Merge reward_zone arguments
    zone_data = reward_zone or reward_zone_mm
    parsed_zone = _parse_reward_zone(zone_data)

    histograms = {}
    max_density = 0.0
    bin_edges = np.linspace(-arena_r, arena_r, n_bins + 1)

    # 1. Compute 2D Histograms
    for name, df_epoch in valid_epochs.items():
        x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)
        counts, _, _ = np.histogram2d(x, y, bins=[bin_edges, bin_edges])

        # % time in bin
        fractional_density = (counts / len(df_epoch)) * 100.0
        histograms[name] = fractional_density.T  # Transpose for imshow alignment

        if fractional_density.max() > max_density:
            max_density = fractional_density.max()

    if max_density == 0:
        max_density = 1.0

    # 2. Render Panels
    n_panels = len(valid_epochs)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.5 * n_panels, 5),
        dpi=300,
        layout="constrained",
    )
    if n_panels == 1:
        axes = np.array([axes])

    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=max_density)

    for i, (name, density_map) in enumerate(histograms.items()):
        ax = axes[i]

        im = ax.imshow(
            density_map,
            origin="lower",
            extent=[-arena_r, arena_r, -arena_r, arena_r],
            cmap="rocket",
            norm=norm,
            interpolation="gaussian",
        )

        # Arena Border
        arena_circle = patches.Circle(
            (0, 0), arena_r, color="white", fill=False, linestyle="--", linewidth=1.2, alpha=0.6
        )
        ax.add_patch(arena_circle)

        # White Border Circle for Reward Zone
        if parsed_zone:
            zx, zy, zr = parsed_zone
            zone_circle = patches.Circle(
                (zx, zy),
                zr,
                edgecolor="white",
                facecolor="none",
                linestyle="-",
                linewidth=1.8,
                zorder=10,
            )
            ax.add_patch(zone_circle)

        # Styling
        ax.set_aspect("equal")
        ax.set_xlim(-arena_r - 2, arena_r + 2)
        ax.set_ylim(-arena_r - 2, arena_r + 2)
        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)

        epoch_seconds = len(valid_epochs[name]) / 60.0
        ax.set_title(
            f"{name.replace('_', ' ')}\n({epoch_seconds:.1f} s)",
            fontsize=12,
            pad=10,
        )

    # Shared Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.03)
    cbar.set_label("% Time in Bin", fontsize=10)

    if title_prefix:
        fig.suptitle(
            f"Cartesian Spatial Density - {title_prefix}", fontsize=14, y=1.05
        )

    return fig, axes


# -------------------------------------------------------------------------
# 2. Polar Density Plotter
# -------------------------------------------------------------------------
def plot_dynamic_polar_densities(
    epochs_dict: Dict[str, pd.DataFrame],
    arena_r: float = 45.0,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    gamma: float = 0.3,
    title_prefix: str = "",
    x_col: str = "X",
    y_col: str = "Y",
    reward_zone: Optional[Dict[str, float]] = None,
    reward_zone_mm: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
    """Plots 2D Polar spatial density (% Time in Bin) across epochs with a white

    reward zone circle transformed into polar space.
    """
    valid_epochs = {
        k: v for k, v in epochs_dict.items() if v is not None and len(v) > 0
    }
    if not valid_epochs:
        print("No valid epoch data to plot.")
        return None, None

    # Merge reward_zone arguments
    zone_data = reward_zone or reward_zone_mm
    parsed_zone = _parse_reward_zone(zone_data)

    histograms = {}
    max_density = 0.0

    edges_r = np.linspace(0, arena_r, n_bins_r + 1)
    edges_theta = np.linspace(-np.pi, np.pi, n_bins_theta + 1)

    # 1. Compute Polar Density
    for name, df_epoch in valid_epochs.items():
        x, y = _get_coords(df_epoch, x_col=x_col, y_col=y_col)

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        counts, _, _ = np.histogram2d(r, theta, bins=[edges_r, edges_theta])
        fractional_density = (counts / len(df_epoch)) * 100.0

        histograms[name] = fractional_density

        if fractional_density.max() > max_density:
            max_density = fractional_density.max()

    if max_density == 0:
        max_density = 1.0

    # 2. Render Polar Panels
    n_panels = len(valid_epochs)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.5 * n_panels, 5),
        dpi=300,
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    if n_panels == 1:
        axes = np.array([axes])

    norm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=max_density)

    for i, (name, density_map) in enumerate(histograms.items()):
        ax = axes[i]

        Theta, R = np.meshgrid(edges_theta, edges_r)
        mesh = ax.pcolormesh(
            Theta,
            R,
            density_map,
            cmap="rocket",
            rasterized=True,
            norm=norm,
        )

        # Draw White Reward Zone Circle in Polar Coordinates
        if parsed_zone:
            zx, zy, zr = parsed_zone
            # Parametric Circle in Cartesian space -> polar
            t = np.linspace(0, 2 * np.pi, 200)
            circle_x = zx + zr * np.cos(t)
            circle_y = zy + zr * np.sin(t)

            circle_r = np.sqrt(circle_x**2 + circle_y**2)
            circle_theta = np.arctan2(circle_y, circle_x)

            # Plot on Polar Axes (Theta, R)
            ax.plot(
                circle_theta,
                circle_r,
                color="white",
                linestyle="-",
                linewidth=1.8,
                zorder=10,
            )

        # Aesthetics
        ax.set_yticklabels([])  # Hide radial ticks for clarity
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(
            ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"],
            fontsize=8,
            color="#555555",
        )
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        epoch_seconds = len(valid_epochs[name]) / 60.0
        ax.set_title(
            f"{name.replace('_', ' ')}\n({epoch_seconds:.1f} s)",
            fontsize=12,
            pad=15,
        )

    # Shared Colorbar
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.75, pad=0.04)
    cbar.set_label("% Time in Bin", fontsize=10)

    if title_prefix:
        fig.suptitle(
            f"Polar Spatial Occupancy - {title_prefix}", fontsize=14, y=1.05
        )

    return fig, axes