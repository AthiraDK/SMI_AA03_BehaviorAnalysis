
from typing import Dict, List, Optional, Tuple
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
        ec="#272424",
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
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{ds['uid']}\n{title_suffix}", fontsize=11, pad=10, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    return fig, ax


def plot_multi_experiment_split_trajectories(
    dataset: List[Dict],
    arena_r: float = 45.0,
    fps: float = 60.0,
    ten_min_frames: int = 36000,  # 10 mins at 60 fps
    x_col: str = "X",
    y_col: str = "Y",
    color_map: Optional[Dict[str, str]] = None,
    fig_title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots split trajectory subplots for multiple experiments.

    Rows represent individual experiments, while columns split trajectories into:
    1. Pre-1st reward encounter ("Baseline")
    2. Post 1st reward encounter up to 10 mins ("Stimulus")
    3. After 10 mins ("Washout")
    """
    if color_map is None:
        color_map = {
            "Baseline": "#BA90EB",
            "Stimulus": "#F16237",
            "Washout": "#009E73",
        }

    n_experiments = len(dataset)
    if n_experiments == 0:
        raise ValueError("The provided dataset list is empty.")

    fig, axes = plt.subplots(
        nrows=n_experiments,
        ncols=3,
        figsize=(13, 3.5 * n_experiments),
        dpi=300,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    # Ensure axes array is 2D if there is only 1 experiment
    if n_experiments == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Pre-1st Reward",
        "Post-1st Reward (Up to 10m)",
        "After 10m (No Reward)",
    ]

    for row_idx, ds in enumerate(dataset):
        df_track = ds["df_track"]

        # Resolve column names dynamically
        col_x = x_col if x_col in df_track.columns else ("x_mm" if "x_mm" in df_track.columns else "X")
        col_y = y_col if y_col in df_track.columns else ("y_mm" if "y_mm" in df_track.columns else "Y")

        x = df_track[col_x].values
        y = df_track[col_y].values
        total_frames = len(x)

        # Retrieve pre-calculated metadata & reward attributes
        uid = ds.get("uid", f"Fly_{row_idx}")
        n_rewards = ds.get("n_rewards", 0)
        f_idx = ds.get("first_reward_idx", None)

        # Calculate time of 1st reward encounter in minutes
        if f_idx is not None:
            t_first_min = f_idx / (fps * 60.0)
            reward_str = f"1st rew: {t_first_min:.2f} min"
        else:
            f_idx = total_frames  # Fallback if no reward occurred
            reward_str = "No rewards"

        # Define slice boundaries for the 3 time periods
        slice_p1 = slice(0, f_idx)                                       # Baseline
        slice_p2 = slice(f_idx, min(ten_min_frames, total_frames))      # Stimulus
        slice_p3 = slice(ten_min_frames, total_frames) if total_frames > ten_min_frames else None  # Washout

        slices = [slice_p1, slice_p2, slice_p3]
        epoch_keys = ["Baseline", "Stimulus", "Washout"]

        # Plot each period in its respective column
        for col_idx, (slc, epoch) in enumerate(zip(slices, epoch_keys)):
            ax = axes[row_idx, col_idx]

            if slc is not None and slc.start < slc.stop:
                ax.plot(
                    x[slc],
                    y[slc],
                    color=color_map[epoch],
                    linewidth=0.8,
                    alpha=0.85,
                )

            # Draw Reward Zone & Arena circles
            zone_x = ds.get("zone_X", 0.0)
            zone_y = ds.get("zone_Y", 0.0)
            zone_r = ds.get("zone_r", 3.0)

            reward_zone = patches.Circle(
                (zone_x, zone_y),
                zone_r,
                ec="#272424",
                fc="none",
                fill=False,
                linewidth=1.2,
                zorder=10,
            )
            arena_boundary = patches.Circle(
                (0, 0),
                arena_r,
                ec="#888888",
                fc="none",
                fill=False,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
            ax.add_patch(reward_zone)
            ax.add_patch(arena_boundary)

            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Set top row headers
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=11, fontweight="bold", pad=10)

        # Add Fly UID, total rewards, and 1st encounter time on the left side of each row
        row_label = f"UID: {uid}\nRewards: {n_rewards}\n{reward_str}"
        axes[row_idx, 0].set_ylabel(
            row_label, 
            fontsize=8, 
            fontweight="bold", 
            labelpad=12,
            rotation=0,
            ha="right",
            va="center"
        )

    # Set arena plotting limits
    pad = 2.0
    for ax in axes.ravel():
        ax.set_xlim([-arena_r - pad, arena_r + pad])
        ax.set_ylim([-arena_r - pad, arena_r + pad])

    # Add global figure title if provided
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.02)

    return fig, axes