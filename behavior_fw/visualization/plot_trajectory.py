
from typing import Dict, Optional, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_single_fly_trajectory(
    ds: Dict,
    arena_r: float = 45.0,
    fps: float = 60.0,
    x_col: str = "X",
    y_col: str = "Y",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots baseline, active reward, and extinction spatial trajectories for a single fly record.

    Parameters
    ----------
    ds : Dict
        Dictionary element from your `dataset` list.
    arena_r : float
        Radius of circular arena boundary in mm.
    fps : float
        Tracking acquisition rate in Hz (default 60.0).
    """
    df_track = ds["df_track"]

    # Handle column naming variations safely
    col_x = (
        x_col
        if x_col in df_track.columns
        else ("x_mm" if "x_mm" in df_track.columns else "X")
    )
    col_y = (
        y_col
        if y_col in df_track.columns
        else ("y_mm" if "y_mm" in df_track.columns else "Y")
    )

    x = df_track[col_x].values
    y = df_track[col_y].values

    f_idx = ds.get("first_reward_idx", None)
    l_idx = ds.get("last_reward_idx", None)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300, layout="constrained")

    # 1. Full Path (Baseline / Extinction / Overall)
    ax.plot(
        x,
        y,
        color="#888888",
        alpha=0.6,
        linewidth=0.8,
        label="Path (Non-Active)",
    )

    # 2. Overlay Active Reward Path
    if f_idx is not None and l_idx is not None and f_idx < l_idx:
        ax.plot(
            x[f_idx:l_idx],
            y[f_idx:l_idx],
            color="#d62728",
            linewidth=1.2,
            alpha=0.85,
            label="Active Reward Epoch",
        )
        t0_min = f_idx / (fps * 60.0)
        title_suffix = (
            f"n_rewards = {ds.get('n_rewards', 0)} | 1st reward @ {t0_min:.2f} min"
        )
    else:
        title_suffix = f"n_rewards = {ds.get('n_rewards', 0)} (No rewards triggered)"

    # 3. Reward Zone & Arena Circles
    zone_x = ds.get("zone_X", 0.0)
    zone_y = ds.get("zone_Y", 0.0)
    zone_r = ds.get("zone_r", 3.0)

    print(zone_r, zone_x, zone_y)

    reward_zone = patches.Circle(
        (zone_x, zone_y),
        zone_r,
        ec="#d62728",
        fc="none",
        fill=False,
        linewidth=2,
        zorder=10,
        label="Reward Zone",
    )
    arena_boundary = patches.Circle(
        (0, 0),
        arena_r,
        ec="#333333",
        fc="none",
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.add_patch(reward_zone)
    ax.add_patch(arena_boundary)

    # Axis formatting
    pad = 2.0
    ax.set_xlim([-arena_r - pad, arena_r + pad])
    ax.set_ylim([-arena_r - pad, arena_r + pad])
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"{ds['uid']}\n{title_suffix}", fontsize=11, pad=10, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    return fig, ax